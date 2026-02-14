"""Evaluate dynamics model using FVD (Fréchet Video Distance).

Computes three FVD comparisons on the predicted horizon (excluding context frames):
  1. FVD(true frames, decoded pred latents)  — end-to-end pipeline quality
  2. FVD(true frames, decoded true latents)  — tokenizer reconstruction ceiling
  3. FVD(decoded true latents, decoded pred latents)  — dynamics-only quality

Usage:
    uv run scripts/eval_fvd.py \
        dynamics_ckpt=./logs/dynamics/checkpoints \
        num_videos=256 \
        horizon=28 \
        ctx_length=4 \
        save_videos=true \
        video_dir=./logs/eval_fvd_videos
"""
import logging
from pathlib import Path

import imageio.v3 as iio
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.data import make_iterator
from dreamer.fvd import frechet_distance, get_fvd_logits, load_i3d_pretrained
from dreamer.generation import DenoiseSchedule
from dreamer.parallel import build_parallel
from dreamer.sampler import sample_video
from dreamer.utils import apply_border

logging.getLogger('absl').setLevel(logging.WARNING)

OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("max", lambda *args: max(args))


def save_comparison_video(
    true_frames, gt_dec_frames, pred_frames, ctx_length, out_path, fps=5,
):
    """Save a side-by-side comparison video: [gt_decoded | true | predicted].

    Context frames in predicted get a red border. Each input is (B, T, H, W, C) uint8.
    Saves one MP4 per sample plus a grid MP4 with up to 4 samples stacked vertically.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    B = pred_frames.shape[0]
    num_grid = min(4, B)

    # Add red border to context frames in predicted
    pred_frames = np.array(pred_frames)  # ensure numpy
    bordered = np.array(apply_border(jnp.array(pred_frames[:, :ctx_length])))
    pred_frames[:, :ctx_length] = bordered

    # Save individual videos
    for i in range(B):
        # Side-by-side: true | gt_decoded | predicted
        frames_i = np.concatenate(
            [true_frames[i], gt_dec_frames[i], pred_frames[i]], axis=2
        )  # (T, H, 3*W, C)
        path_i = out_path / f"sample_{i:04d}.mp4"
        try:
            iio.imwrite(str(path_i), frames_i, fps=fps, plugin="pyav", codec="libx264")
        except Exception as e:
            print(f"Failed to write {path_i}: {e}")

    # Save grid video (up to 4 samples stacked vertically)
    stacked = np.stack(
        [true_frames[:num_grid], gt_dec_frames[:num_grid], pred_frames[:num_grid]]
    )  # (3, B', T, H, W, C)
    grid = rearrange(stacked, "S B T H W C -> T (B H) (S W) C", B=num_grid)
    grid_path = out_path / "grid.mp4"
    try:
        iio.imwrite(str(grid_path), grid, fps=fps, plugin="pyav", codec="libx264")
        print(f"Saved grid video to {grid_path}")
    except Exception as e:
        print(f"Failed to write grid video: {e}")


def run(cfg):
    num_videos = cfg.num_videos
    horizon = cfg.get("horizon", None)
    ctx_length = cfg.ctx_length
    i3d_bs = cfg.i3d_bs
    save_videos = cfg.save_videos
    video_dir = cfg.video_dir
    max_save = cfg.max_save
    dynamics_ckpt = cfg.dynamics_ckpt

    assert dynamics_ckpt, "dynamics_ckpt must be set (path to dynamics checkpoint directory)"

    # Parallelism
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with jax.set_mesh(mesh):
        # Load dynamics + tokenizer from checkpoint
        # Auto-append /checkpoints if the path doesn't contain it
        ckpt_path = str(dynamics_ckpt)
        if not ckpt_path.rstrip('/').endswith('checkpoints'):
            candidate = str(Path(ckpt_path) / 'checkpoints')
            if Path(candidate).exists():
                ckpt_path = candidate
        print(f"Loading checkpoint from {ckpt_path}...")
        bundle = DynamicsCheckpointBundle.from_pretrained(
            ckpt_path,
            mesh_rules=mesh_rules,
        )
        tokenizer = bundle.tokenizer
        dynamics = bundle.dynamics

        k_max = dynamics.cfg.k_max
        schedule = DenoiseSchedule.init(k_max, k_max)

        use_latent_data = cfg.dataset.data_type == "latent"
        T = cfg.dataset.T

        if horizon is None:
            horizon = T - ctx_length
        assert horizon >= 10, f"I3D requires >=10 frames, got horizon={horizon}"
        assert ctx_length + horizon <= T, (
            f"ctx_length({ctx_length}) + horizon({horizon}) > T({T})")

        # Load I3D
        print("Loading I3D weights...")
        i3d_params = load_i3d_pretrained()
        print("I3D weights loaded.")

        # Data iterator
        dataloader = make_iterator(cfg.dataset)

        rng = jax.random.PRNGKey(cfg.seed)

        # Accumulate I3D features
        feats_true = []       # true frames (original pixels)
        feats_gt_dec = []     # decoded true latents
        feats_pred_dec = []   # decoded predicted latents

        # Accumulate full-length frames for video saving
        saved_count = 0
        all_true = []
        all_gt_dec = []
        all_pred = []

        collected = 0
        pbar = tqdm(total=num_videos, desc="Collecting videos")

        for batch in dataloader:
            if collected >= num_videos:
                break

            rng, sample_rng = jax.random.split(rng)

            actions = batch["actions"]
            videos = batch.get("videos")
            latents = batch.get("latents")
            input_tensor = latents if latents is not None else videos

            B_batch = input_tensor.shape[0]

            if use_latent_data:
                pred_frames, gt_decoded_frames, _ = sample_video(
                    tokenizer, dynamics,
                    frames=None,
                    actions=actions,
                    horizon=horizon,
                    schedule_config=schedule,
                    rng=sample_rng,
                    latents=input_tensor,
                )
                # For latent data, true frames = gt decoded frames (best we have)
                true_frames = gt_decoded_frames
            else:
                pred_frames, gt_decoded_frames, original_frames = sample_video(
                    tokenizer, dynamics,
                    frames=input_tensor,
                    actions=actions,
                    horizon=horizon,
                    schedule_config=schedule,
                    rng=sample_rng,
                )
                true_frames = original_frames

            # Move to host
            true_full = jax.device_get(jnp.clip(true_frames, 0, 255).astype(jnp.uint8))
            gt_dec_full = jax.device_get(jnp.clip(gt_decoded_frames, 0, 255).astype(jnp.uint8))
            pred_full = jax.device_get(jnp.clip(pred_frames, 0, 255).astype(jnp.uint8))

            # Extract horizon portion for FVD
            true_horizon = true_full[:, ctx_length:ctx_length + horizon]
            gt_dec_horizon = gt_dec_full[:, ctx_length:ctx_length + horizon]
            pred_dec_horizon = pred_full[:, ctx_length:ctx_length + horizon]

            # Extract I3D features
            feats_true.append(get_fvd_logits(jnp.array(true_horizon), i3d_params, bs=i3d_bs))
            feats_gt_dec.append(get_fvd_logits(jnp.array(gt_dec_horizon), i3d_params, bs=i3d_bs))
            feats_pred_dec.append(get_fvd_logits(jnp.array(pred_dec_horizon), i3d_params, bs=i3d_bs))

            # Accumulate for video saving (full-length including context)
            if save_videos and saved_count < max_save:
                n_keep = min(B_batch, max_save - saved_count)
                all_true.append(true_full[:n_keep])
                all_gt_dec.append(gt_dec_full[:n_keep])
                all_pred.append(pred_full[:n_keep])
                saved_count += n_keep

            collected += B_batch
            pbar.update(B_batch)

        pbar.close()

        # Save videos
        if save_videos and all_true:
            all_true_np = np.concatenate(all_true, axis=0)
            all_gt_dec_np = np.concatenate(all_gt_dec, axis=0)
            all_pred_np = np.concatenate(all_pred, axis=0)
            print(f"Saving {all_true_np.shape[0]} videos to {video_dir}...")
            save_comparison_video(
                all_true_np, all_gt_dec_np, all_pred_np,
                ctx_length=ctx_length, out_path=video_dir,
            )

        # Concatenate all features
        feats_true = jnp.concatenate(feats_true, axis=0)[:num_videos]
        feats_gt_dec = jnp.concatenate(feats_gt_dec, axis=0)[:num_videos]
        feats_pred_dec = jnp.concatenate(feats_pred_dec, axis=0)[:num_videos]

        # Compute FVD scores
        fvd_e2e = frechet_distance(feats_true, feats_pred_dec)
        fvd_tokenizer = frechet_distance(feats_true, feats_gt_dec)
        fvd_dynamics = frechet_distance(feats_gt_dec, feats_pred_dec)

        print(f"\n{'='*60}")
        print(f"FVD Results (num_videos={feats_true.shape[0]}, horizon={horizon}, ctx={ctx_length})")
        print(f"{'='*60}")
        print(f"  FVD(true, pred_decoded)     = {fvd_e2e:>10.2f}  (end-to-end)")
        print(f"  FVD(true, gt_decoded)        = {fvd_tokenizer:>10.2f}  (tokenizer ceiling)")
        print(f"  FVD(gt_decoded, pred_decoded) = {fvd_dynamics:>10.2f}  (dynamics only)")
        print(f"{'='*60}")


@hydra.main(version_base=None, config_path="../configs", config_name="eval_fvd")
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
