"""Two-stage FVD evaluation for dynamics models.

Stage 1 (generate): Run dynamics model to produce predicted videos, save individual MP4s.
Stage 2 (evaluate): Load saved MP4s, extract I3D features, compute FVD scores.

Three FVD comparisons are reported:
  1. FVD(original, pred)       — end-to-end pipeline quality
  2. FVD(original, gt_decoded) — tokenizer reconstruction ceiling
  3. FVD(gt_decoded, pred)     — dynamics-only quality

Usage:
    uv run scripts/eval_fvd.py dynamics_ckpt=<path> mode=both num_videos=256
    uv run scripts/eval_fvd.py dynamics_ckpt=<path> mode=generate num_videos=16 dataset.dataloader_cfg.B=4
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

from flax import nnx
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.data import build_iterator
from dreamer.fvd import frechet_distance, get_fvd_logits, load_i3d_pretrained
from dreamer.generation import DenoiseSchedule, latent_rollout
from dreamer.parallel import build_parallel
from dreamer.sampler import sample_video, decode_chunked, decode_jit
from dreamer.actions import shift_actions
import os

logging.getLogger('absl').setLevel(logging.WARNING)

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("max", lambda *args: max(args))


def _save_video(frames, path, fps=20):
    """Save (T, H, W, C) uint8 array as MP4."""
    iio.imwrite(str(path), frames, plugin="pyav", fps=fps, codec="libx264")


def generate_videos(cfg):
    """Stage 1: Generate videos and save as individual MP4s."""
    dynamics_ckpt = cfg.dynamics_ckpt
    assert dynamics_ckpt, "dynamics_ckpt must be set for generation"

    num_videos = cfg.num_videos
    ctx_length = cfg.ctx_length
    horizon = cfg.horizon
    video_dir = Path(cfg.video_dir)

    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with jax.set_mesh(mesh):
        ckpt_path = str(dynamics_ckpt)
        if not ckpt_path.rstrip('/').endswith('checkpoints'):
            candidate = str(Path(ckpt_path) / 'checkpoints')
            if Path(candidate).exists():
                ckpt_path = candidate

        rollout_type = cfg.get("rollout_type", "ema_shortcut")
        valid_types = ("ema_shortcut", "ema_diffusion", "online_shortcut", "online_diffusion")
        assert rollout_type in valid_types, f"rollout_type must be one of {valid_types}, got {rollout_type!r}"
        use_ema = rollout_type.startswith("ema")
        use_shortcut = rollout_type.endswith("shortcut")
        dynamics_field = "dynamics_ema" if use_ema else "dynamics"
        ema_checkpoint_average_count = cfg.get("ema_checkpoint_average_count", 1)
        average_last_n = ema_checkpoint_average_count if use_ema else 1
        average_model_names = {dynamics_field} if average_last_n > 1 else None

        print(f"Loading checkpoint from {ckpt_path} (models: {dynamics_field}, tokenizer)...")
        if average_model_names:
            print(f"Averaging latest {average_last_n} EMA checkpoints one by one.")
        bundle = DynamicsCheckpointBundle.from_pretrained(
            ckpt_path, mesh_rules=mesh_rules,
            model_names={dynamics_field, "tokenizer"},
            average_model_names=average_model_names,
            average_last_n=average_last_n,
        )
        tokenizer = bundle.tokenizer
        dynamics = getattr(bundle, dynamics_field)

        use_latent_data = cfg.dataset.data_type == "latent"
        # With ctx_length=0 and raw video we never need the encoder: the rollout
        # starts from an empty context and the raw frames serve as the FVD reference.
        skip_encode = (not use_latent_data) and (ctx_length == 0)

        if use_latent_data:
            del tokenizer.encoder
            print(
                "WARNING: Using latent data — original pixel videos are not available. "
                "FVD will be computed against autoencoded (gt_decoded) videos, not originals. "
                "For scientifically rigorous evaluation, use raw video data instead."
            )
        elif skip_encode:
            # Read latent shape before freeing the encoder.
            n_latents_per_frame = tokenizer.encoder.n_latents
            del tokenizer.encoder
            print(
                "ctx_length=0: encoder not needed — generating from empty context, "
                "using raw video frames as FVD reference."
            )

        k_max = dynamics.cfg.k_max
        num_steps = 4 if use_shortcut else k_max
        schedule_config = DenoiseSchedule.init(num_steps, k_max)
        print(f"Rollout type: {rollout_type} (num_steps={num_steps}, k_max={k_max})")

        dataloader = build_iterator(
            cfg.dataset,
            seed=cfg.seed,
            device=data_sharding,
            return_actions=not use_latent_data,  # ensure video dataset returns actions
        )
        rng = jax.random.PRNGKey(cfg.seed)

        out_dir = video_dir / rollout_type
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compile rollout once before the loop so the function identity is
        # stable across batches and jax.lax.scan fuses into a single XLA op.
        if skip_encode:
            decode_chunk_size = cfg.decode_chunk_size
            def _rollout_fn(dyn, policy_actions, latents_ctx, actions_ctx, rng):
                return latent_rollout(
                    dyn, policy=policy_actions, schedule=schedule_config,
                    latents_ctx=latents_ctx, actions_ctx=actions_ctx,
                    num_steps=horizon, rng=rng, use_scan=True,
                )
            _rollout_jit = nnx.jit(_rollout_fn)

        collected = 0
        pbar = tqdm(total=num_videos, desc="Generating videos")

        for batch in dataloader:
            if collected >= num_videos:
                break

            rng, eval_rng = jax.random.split(rng)

            val_actions = batch["actions"]
            val_actions = shift_actions(val_actions, cfg.dataset.categorical_action_dim)

            if use_latent_data:
                val_data = batch.get("latents", batch.get("videos"))
                pred_frames, gt_decoded_frames, original_frames = sample_video(
                    tokenizer=tokenizer,
                    dynamics=dynamics,
                    frames=None,
                    latents=val_data,
                    actions=val_actions,
                    horizon=horizon,
                    schedule_config=schedule_config,
                    rng=eval_rng,
                    policy=None,
                    task_embedder=None,
                    use_scan=False,
                    decode_chunk_size=cfg.decode_chunk_size,
                )
            elif skip_encode:
                val_data = batch["videos"]
                B_batch = val_data.shape[0]
                D_s = dynamics.d_bottleneck

                # Empty context: generate unconditionally from GT actions only.
                empty_ctx = jnp.zeros((B_batch, 0, n_latents_per_frame, D_s), dtype=jnp.bfloat16)
                rollout = _rollout_jit(
                    dynamics,
                    val_actions[:, :horizon],
                    empty_ctx,
                    val_actions[:, :0],
                    eval_rng,
                )
                decode_fn = (
                    (lambda t, z: decode_chunked(t, z, decode_chunk_size))
                    if decode_chunk_size else decode_jit
                )
                pred_frames = jnp.clip(decode_fn(tokenizer, rollout["latents"]), 0, 255).astype(jnp.uint8)
                # Raw frames are the reference; use them as gt_decoded too since
                # there is no tokenizer reconstruction without encoding.
                original_frames = jnp.clip(val_data[:, :horizon], 0, 255).astype(jnp.uint8)
                gt_decoded_frames = original_frames
            else:
                val_data = batch["videos"]
                pred_frames, gt_decoded_frames, original_frames = sample_video(
                    tokenizer=tokenizer,
                    dynamics=dynamics,
                    frames=val_data,
                    latents=None,
                    actions=val_actions,
                    horizon=horizon,
                    schedule_config=schedule_config,
                    rng=eval_rng,
                    policy=None,
                    task_embedder=None,
                    use_scan=False,
                    decode_chunk_size=cfg.decode_chunk_size,
                )

            B_batch = val_data.shape[0]
            gt_decoded = jax.device_get(gt_decoded_frames)
            pred = jax.device_get(pred_frames)
            original = jax.device_get(original_frames) if original_frames is not None else None

            for i in range(B_batch):
                idx = collected + i
                _save_video(gt_decoded[i], out_dir / f"gt_decoded_{idx:06d}.mp4")
                _save_video(pred[i], out_dir / f"pred_{idx:06d}.mp4")
                if original is not None:
                    _save_video(original[i], out_dir / f"original_{idx:06d}.mp4")

            collected += B_batch
            pbar.update(B_batch)

        pbar.close()
        print(f"Generated {collected} videos in {out_dir}")


def evaluate_fvd(cfg):
    """Stage 2: Load saved MP4s and compute FVD."""
    rollout_type = cfg.get("rollout_type", "ema_shortcut")
    video_dir = Path(cfg.video_dir) / rollout_type
    i3d_bs = cfg.i3d_bs
    pred_frames_only = cfg.pred_frames_only
    ctx_length = cfg.ctx_length

    assert video_dir.exists(), f"Video directory not found: {video_dir}"

    pred_files = sorted(video_dir.glob("pred_*.mp4"))
    gt_dec_files = sorted(video_dir.glob("gt_decoded_*.mp4"))
    original_files = sorted(video_dir.glob("original_*.mp4"))

    assert len(pred_files) > 0, f"No pred MP4s found in {video_dir}"
    assert len(gt_dec_files) == len(pred_files), (
        f"Mismatch: {len(gt_dec_files)} gt_decoded vs {len(pred_files)} pred files")

    has_originals = len(original_files) == len(pred_files)
    if not has_originals:
        print("No original files found (latent data mode); using gt_decoded as reference.")

    num_videos = len(pred_files)
    print(f"Found {num_videos} video pairs in {video_dir}")

    print("Loading I3D weights...")
    i3d_params = load_i3d_pretrained()

    # Probe one file to get T, H, W, C without loading everything.
    _probe = iio.imread(str(pred_files[0]), plugin="pyav")
    trim = ctx_length if pred_frames_only else 0
    T_full = _probe.shape[0] - trim
    fvd_chunk_size = cfg.get("fvd_chunk_size", None)
    if fvd_chunk_size is not None:
        assert T_full % fvd_chunk_size == 0, (
            f"Video length {T_full} not divisible by fvd_chunk_size {fvd_chunk_size}")
        T = fvd_chunk_size
        num_chunks = T_full // fvd_chunk_size
    else:
        T = T_full
        num_chunks = 1
    assert T >= 10, f"I3D requires >=10 frames, got {T} after trimming"
    num_videos = num_videos * num_chunks
    if fvd_chunk_size is not None:
        print(f"Chunking {len(pred_files)} videos into {num_chunks} clips of {T} frames -> {num_videos} total clips")
    print(f"Computing FVD on {num_videos} clips, {T} frames each")

    def stream_features(file_list, desc):
        """Load and featurise videos in small batches to avoid RAM spikes."""
        # Process i3d_bs videos at a time; each may expand to num_chunks clips.
        all_feats = []
        for i in tqdm(range(0, len(file_list), i3d_bs), desc=desc):
            batch_files = file_list[i:i + i3d_bs]
            clips = []
            for f in batch_files:
                video = iio.imread(str(f), plugin="pyav")[trim:]  # (T_full, H, W, C)
                if num_chunks > 1:
                    clips.append(video.reshape(num_chunks, T, *video.shape[1:]))
                else:
                    clips.append(video[None])  # (1, T, H, W, C)
            batch = np.concatenate(clips, axis=0)  # (B*num_chunks, T, H, W, C)
            all_feats.append(np.array(get_fvd_logits(batch, i3d_params, bs=len(batch))))
        return np.concatenate(all_feats, axis=0)

    print("Extracting I3D features...")
    feats_pred   = stream_features(pred_files,     "pred")
    feats_gt_dec = stream_features(gt_dec_files,   "gt_decoded")
    feats_ref    = stream_features(original_files, "original") if has_originals else feats_gt_dec

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
