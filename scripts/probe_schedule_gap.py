from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from dreamer.actions import shift_actions
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.data import make_iterator
from dreamer.generation import DenoiseSchedule, next_latent
from dreamer.parallel import build_parallel
from dreamer.sampler import decode_latents
from dreamer.utils import normalize_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare shortcut vs diffusion rollout from the same GT prefix.")
    parser.add_argument("--dynamics-ckpt", required=True)
    parser.add_argument("--array-record-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index-max", type=int, default=4)
    parser.add_argument("--num-max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--prefix-len", type=int, default=128)
    parser.add_argument("--probe-steps", type=int, default=4)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--parallel-strategy", default="data")
    parser.add_argument("--use-online", action="store_true")
    return parser.parse_args()


def build_dataset_cfg(args: argparse.Namespace):
    cfg = OmegaConf.load("configs/dataset/minecraft_vpt_latent.yaml")
    cfg.array_record_path = args.array_record_path
    cfg.index_max = args.index_max
    cfg.num_max_samples = args.num_max_samples
    cfg.dataloader_cfg.B = args.batch_size
    cfg.dataloader_cfg.short_T = args.seq_len
    cfg.dataloader_cfg.long_T = args.seq_len
    cfg.dataloader_cfg.long_ratio = 1.0
    cfg.dataloader_cfg.num_workers = args.num_workers
    cfg.dataloader_cfg.prefetch_buffer_size = 2
    cfg.dataloader_cfg.device_prefetch_buffer_size = 1
    return cfg


def rollout_from_prefix(
    *,
    dynamics,
    schedule: DenoiseSchedule,
    latents: jax.Array,
    actions,
    prefix_len: int,
    probe_steps: int,
    rng: jax.Array,
) -> jax.Array:
    B, _, n_spatial, d_latent = latents.shape
    latents_norm = normalize_latents(latents, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)
    caches = dynamics.create_static_caches(
        batch_size=B,
        n_latents=n_spatial,
        window_size=dynamics.cfg.context_length,
        dtype=dynamics.dtype,
    )

    step_idx_prefill = jnp.full((B, prefix_len), schedule.emax, dtype=jnp.int32)
    tau_idx_prefill = jnp.full((B, prefix_len), schedule.k_max, dtype=jnp.int32)
    _, (_, caches) = dynamics(
        actions[:, :prefix_len],
        step_idx_prefill,
        tau_idx_prefill,
        latents_norm[:, :prefix_len],
        caches=caches,
        deterministic=True,
    )

    pred_latents = []
    latent_shape = (B, 1, n_spatial, d_latent)
    for step_offset in range(probe_steps):
        latent_next, _, caches, rng, _ = next_latent(
            dynamics=dynamics,
            schedule=schedule,
            action=actions[:, prefix_len + step_offset],
            latent_shape=latent_shape,
            rng=jax.random.fold_in(rng, int(step_offset)),
            caches=caches,
        )
        pred_latents.append(latent_next[:, 0])

    return jnp.stack(pred_latents, axis=1)


def framewise_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2, axis=(1, 2, 3))


def save_contact(path: Path, *, gt_frames: np.ndarray, shortcut_frames: np.ndarray, diffusion_frames: np.ndarray, prefix_len: int) -> None:
    frame_indices = list(range(gt_frames.shape[0]))
    rows = [
        (gt_frames, "GT future"),
        (shortcut_frames, "Shortcut k=4"),
        (diffusion_frames, "Diffusion k=k_max"),
    ]
    fig, axes = plt.subplots(len(rows), len(frame_indices), figsize=(2.4 * len(frame_indices), 2.3 * len(rows)))
    for row_idx, (frames, row_title) in enumerate(rows):
        for col_idx, frame_idx in enumerate(frame_indices):
            axes[row_idx, col_idx].imshow(frames[frame_idx])
            axes[row_idx, col_idx].axis("off")
            axes[row_idx, col_idx].set_title(f"t={prefix_len + frame_idx}")
        axes[row_idx, 0].set_ylabel(row_title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_mse_plot(path: Path, *, shortcut_mse: np.ndarray, diffusion_mse: np.ndarray, prefix_len: int) -> None:
    timesteps = np.arange(prefix_len, prefix_len + len(shortcut_mse))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(timesteps, shortcut_mse, marker="o", linewidth=2.0, label="Shortcut k=4 vs GT")
    ax.plot(timesteps, diffusion_mse, marker="o", linewidth=2.0, label="Diffusion k=k_max vs GT")
    ax.set_xlabel("Absolute frame")
    ax.set_ylabel("Pixel MSE")
    ax.set_title("Schedule comparison from the same GT prefix")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_mp4(path: Path, *, gt_frames: np.ndarray, shortcut_frames: np.ndarray, diffusion_frames: np.ndarray) -> None:
    video = np.concatenate([gt_frames, shortcut_frames, diffusion_frames], axis=2)
    iio.imwrite(str(path), video, fps=2, plugin="pyav", codec="libx264")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    dataset_cfg = build_dataset_cfg(args)
    mesh, data_sharding, mesh_rules = build_parallel(args.parallel_strategy)

    with jax.set_mesh(mesh):
        bundle = DynamicsCheckpointBundle.from_pretrained(
            args.dynamics_ckpt,
            mesh_rules=mesh_rules,
            model_names={"dynamics", "dynamics_ema", "tokenizer"},
        )
        dynamics = bundle.dynamics if args.use_online else bundle.dynamics_ema
        tokenizer = bundle.tokenizer

        iterator = make_iterator(dataset_cfg, device=data_sharding)
        batch = next(iter(iterator))
        latents = batch["latents"]
        actions = shift_actions(batch["actions"], dataset_cfg.categorical_action_dim)

        rng = jax.random.PRNGKey(args.seed)
        rng_short, rng_long = jax.random.split(rng)
        shortcut_schedule = DenoiseSchedule.init(4, dynamics.cfg.k_max)
        diffusion_schedule = DenoiseSchedule.init(dynamics.cfg.k_max, dynamics.cfg.k_max)

        shortcut_latents = rollout_from_prefix(
            dynamics=dynamics,
            schedule=shortcut_schedule,
            latents=latents,
            actions=actions,
            prefix_len=args.prefix_len,
            probe_steps=args.probe_steps,
            rng=rng_short,
        )
        diffusion_latents = rollout_from_prefix(
            dynamics=dynamics,
            schedule=diffusion_schedule,
            latents=latents,
            actions=actions,
            prefix_len=args.prefix_len,
            probe_steps=args.probe_steps,
            rng=rng_long,
        )

        gt_future = latents[:, args.prefix_len:args.prefix_len + args.probe_steps]
        shortcut_frames = jnp.clip(decode_latents(tokenizer, shortcut_latents), 0, 255).astype(jnp.uint8)
        diffusion_frames = jnp.clip(decode_latents(tokenizer, diffusion_latents), 0, 255).astype(jnp.uint8)
        gt_frames = jnp.clip(decode_latents(tokenizer, gt_future), 0, 255).astype(jnp.uint8)
        shortcut_frames, diffusion_frames, gt_frames = jax.device_get((shortcut_frames, diffusion_frames, gt_frames))

    sample_index = int(np.clip(args.sample_index, 0, gt_frames.shape[0] - 1))
    gt_np = np.asarray(gt_frames[sample_index])
    shortcut_np = np.asarray(shortcut_frames[sample_index])
    diffusion_np = np.asarray(diffusion_frames[sample_index])
    shortcut_mse = framewise_mse(shortcut_np, gt_np)
    diffusion_mse = framewise_mse(diffusion_np, gt_np)
    schedule_gap_mse = framewise_mse(shortcut_np, diffusion_np)

    save_contact(
        output_dir / "schedule_contact.png",
        gt_frames=gt_np,
        shortcut_frames=shortcut_np,
        diffusion_frames=diffusion_np,
        prefix_len=args.prefix_len,
    )
    save_mse_plot(
        output_dir / "schedule_mse.png",
        shortcut_mse=shortcut_mse,
        diffusion_mse=diffusion_mse,
        prefix_len=args.prefix_len,
    )
    save_mp4(
        output_dir / "schedule_compare.mp4",
        gt_frames=gt_np,
        shortcut_frames=shortcut_np,
        diffusion_frames=diffusion_np,
    )

    metrics = {
        "model": "online" if args.use_online else "ema",
        "sample_index": sample_index,
        "prefix_len": int(args.prefix_len),
        "probe_steps": int(args.probe_steps),
        "context_length": None if dynamics.cfg.context_length is None else int(dynamics.cfg.context_length),
        "shortcut_vs_gt_pixel_mse_per_step": [float(x) for x in shortcut_mse],
        "diffusion_vs_gt_pixel_mse_per_step": [float(x) for x in diffusion_mse],
        "shortcut_vs_diffusion_pixel_mse_per_step": [float(x) for x in schedule_gap_mse],
        "elapsed_sec": time.time() - t0,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
