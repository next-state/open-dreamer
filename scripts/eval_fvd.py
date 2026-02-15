"""Two-stage FVD evaluation for dynamics models.

Stage 1 (generate): Run dynamics model to produce predicted videos, save individual MP4s.
Stage 2 (evaluate): Load saved MP4s, extract I3D features, compute FVD scores.

Three FVD comparisons are reported:
  1. FVD(original, pred)       — end-to-end pipeline quality
  2. FVD(original, gt_decoded) — tokenizer reconstruction ceiling
  3. FVD(gt_decoded, pred)     — dynamics-only quality

Usage:
    uv run scripts/eval_fvd.py dynamics_ckpt=<path> mode=both num_videos=256
    uv run scripts/eval_fvd.py dynamics_ckpt=<path> mode=generate num_videos=16 batch_size=4
    uv run scripts/eval_fvd.py mode=evaluate video_dir=./logs/eval_fvd_videos
"""
import logging
from pathlib import Path

import hydra
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.data import make_iterator
from dreamer.fvd import frechet_distance, get_fvd_logits, load_i3d_pretrained
from dreamer.parallel import build_parallel
from dreamer.training import run_evaluation

logging.getLogger('absl').setLevel(logging.WARNING)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("max", lambda *args: max(args))


class DummyLogger:
    def log(self, step, **kwargs):
        pass

    def log_video(self, step, key, video, fps=5):
        pass

    def log_metrics(self, step, metrics, prefix=None):
        pass


def generate_videos(cfg):
    """Stage 1: Generate videos and save as individual MP4s."""
    dynamics_ckpt = cfg.dynamics_ckpt
    assert dynamics_ckpt, "dynamics_ckpt must be set for generation"

    num_videos = cfg.num_videos
    batch_size = cfg.batch_size
    ctx_length = cfg.ctx_length
    horizon = cfg.get("horizon", None)
    video_dir = Path(cfg.video_dir)

    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with jax.set_mesh(mesh):
        # Load checkpoint
        ckpt_path = str(dynamics_ckpt)
        if not ckpt_path.rstrip('/').endswith('checkpoints'):
            candidate = str(Path(ckpt_path) / 'checkpoints')
            if Path(candidate).exists():
                ckpt_path = candidate
        print(f"Loading checkpoint from {ckpt_path}...")
        bundle = DynamicsCheckpointBundle.from_pretrained(
            ckpt_path, mesh_rules=mesh_rules,
        )
        tokenizer = bundle.tokenizer
        dynamics = bundle.dynamics

        use_latent_data = cfg.dataset.data_type == "latent"
        T = cfg.dataset.dataloader_cfg.T

        if horizon is None:
            horizon = T - ctx_length
        assert ctx_length + horizon <= T, (
            f"ctx_length({ctx_length}) + horizon({horizon}) > T({T})")

        # Override dataloader batch size
        cfg.dataset.dataloader_cfg.B = batch_size

        dataloader = make_iterator(cfg.dataset, seed=cfg.seed, device=data_sharding)
        rng = jax.random.PRNGKey(cfg.seed)
        logger = DummyLogger()

        video_dir.mkdir(parents=True, exist_ok=True)

        collected = 0
        pbar = tqdm(total=num_videos, desc="Generating videos")

        for batch in dataloader:
            if collected >= num_videos:
                break

            rng, eval_rng = jax.random.split(rng)

            if use_latent_data:
                val_data = batch.get("latents", batch.get("videos"))
            else:
                val_data = batch["videos"]
            val_actions = batch["actions"]

            B_batch = val_data.shape[0]

            run_evaluation(
                cfg=cfg, step=0, tokenizer=tokenizer, dynamics=dynamics,
                val_data=val_data, val_actions=val_actions,
                use_latent_data=use_latent_data, vis_dir=video_dir,
                rng=eval_rng, logger=logger,
                save_videos_separately=True, sample_offset=collected,
            )

            collected += B_batch
            pbar.update(B_batch)

        pbar.close()
        print(f"Generated {collected} videos in {video_dir}/step_000000/")


def evaluate_fvd(cfg):
    """Stage 2: Load saved MP4s and compute FVD."""
    video_dir = Path(cfg.video_dir) / "step_000000"
    i3d_bs = cfg.i3d_bs
    pred_frames_only = cfg.pred_frames_only
    ctx_length = cfg.ctx_length

    assert video_dir.exists(), f"Video directory not found: {video_dir}"

    # Discover video files
    pred_files = sorted(video_dir.glob("*_pred_*.mp4"))
    gt_dec_files = sorted(video_dir.glob("*_gt_decoded_*.mp4"))
    original_files = sorted(video_dir.glob("*_original_*.mp4"))

    assert len(pred_files) > 0, f"No pred MP4s found in {video_dir}"
    assert len(gt_dec_files) == len(pred_files), (
        f"Mismatch: {len(gt_dec_files)} gt_decoded vs {len(pred_files)} pred files")

    has_originals = len(original_files) == len(pred_files)
    if not has_originals:
        print("No original files found (latent data mode); using gt_decoded as reference.")

    num_videos = len(pred_files)
    print(f"Found {num_videos} video pairs in {video_dir}")

    # Load I3D
    print("Loading I3D weights...")
    i3d_params = load_i3d_pretrained()

    def load_videos(file_list):
        """Load MP4s and return (N, T, H, W, C) uint8 array."""
        frames = []
        for f in tqdm(file_list, desc=f"Loading {file_list[0].stem.rsplit('_', 1)[0]}", leave=False):
            video = iio.imread(str(f), plugin="pyav")  # (T, H, W, C)
            frames.append(video)
        return np.stack(frames, axis=0)  # (N, T, H, W, C)

    pred_videos = load_videos(pred_files)
    gt_dec_videos = load_videos(gt_dec_files)
    ref_videos = load_videos(original_files) if has_originals else gt_dec_videos

    # Optionally trim context frames
    if pred_frames_only:
        print(f"Trimming first {ctx_length} context frames from each video")
        pred_videos = pred_videos[:, ctx_length:]
        gt_dec_videos = gt_dec_videos[:, ctx_length:]
        ref_videos = ref_videos[:, ctx_length:]

    T = pred_videos.shape[1]
    assert T >= 10, f"I3D requires >=10 frames, got {T} after trimming"
    print(f"Computing FVD on {num_videos} videos, {T} frames each")

    # Extract I3D features in batches
    print("Extracting I3D features...")
    feats_pred = get_fvd_logits(jnp.array(pred_videos), i3d_params, bs=i3d_bs)
    feats_gt_dec = get_fvd_logits(jnp.array(gt_dec_videos), i3d_params, bs=i3d_bs)
    feats_ref = get_fvd_logits(jnp.array(ref_videos), i3d_params, bs=i3d_bs)

    # Compute FVD scores
    fvd_e2e = frechet_distance(feats_ref, feats_pred)
    fvd_tokenizer = frechet_distance(feats_ref, feats_gt_dec)
    fvd_dynamics = frechet_distance(feats_gt_dec, feats_pred)

    ref_label = "original" if has_originals else "gt_decoded"
    print(f"\n{'='*60}")
    print(f"FVD Results (num_videos={num_videos}, frames={T})")
    print(f"{'='*60}")
    print(f"  FVD({ref_label}, pred)        = {fvd_e2e:>10.2f}  (end-to-end)")
    print(f"  FVD({ref_label}, gt_decoded)   = {fvd_tokenizer:>10.2f}  (tokenizer ceiling)")
    print(f"  FVD(gt_decoded, pred)          = {fvd_dynamics:>10.2f}  (dynamics only)")
    print(f"{'='*60}")


def run(cfg):
    mode = cfg.get("mode", "both")
    assert mode in ("both", "generate", "evaluate"), f"Unknown mode: {mode}"

    if mode in ("both", "generate"):
        generate_videos(cfg)
    if mode in ("both", "evaluate"):
        evaluate_fvd(cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="eval_fvd")
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
