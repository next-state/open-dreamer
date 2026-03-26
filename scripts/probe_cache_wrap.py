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

from dreamer.actions import Actions, create_noop_action_like, shift_actions
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.data import make_iterator
from dreamer.generation import DenoiseSchedule, next_latent
from dreamer.parallel import build_parallel
from dreamer.sampler import decode_latents
from dreamer.utils import normalize_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe cached vs uncached rollout when the KV ring buffer is full.")
    parser.add_argument("--dynamics-ckpt", required=True)
    parser.add_argument("--array-record-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index-max", type=int, default=1)
    parser.add_argument("--num-max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix-len", type=int, default=128)
    parser.add_argument("--probe-steps", type=int, default=4)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--parallel-strategy", default="data")
    parser.add_argument("--use-online", action="store_true")
    parser.add_argument("--synthetic-static", action="store_true")
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


def repeat_noop_actions(actions: Actions, categorical_action_dim: int, seq_len: int) -> Actions:
    noop = create_noop_action_like(actions, categorical_action_dim)
    return jax.tree.map(
        lambda x: jnp.repeat(x, seq_len, axis=1) if x is not None else None,
        noop,
    )


def crop_to_window(latents_ctx_norm: jax.Array, actions_ctx: Actions, context_length: int | None) -> tuple[jax.Array, Actions]:
    if context_length is None:
        return latents_ctx_norm, actions_ctx

    keep_len = max(context_length - 1, 0)
    if latents_ctx_norm.shape[1] <= keep_len:
        return latents_ctx_norm, actions_ctx

    latents_ctx_norm = latents_ctx_norm[:, -keep_len:] if keep_len > 0 else latents_ctx_norm[:, :0]
    actions_ctx = jax.tree.map(
        lambda x: x[:, -keep_len:] if x is not None and keep_len > 0 else x[:, :0] if x is not None else None,
        actions_ctx,
    )
    return latents_ctx_norm, actions_ctx


def rollout_cached_from_prefix(
    *,
    dynamics,
    schedule: DenoiseSchedule,
    latents: jax.Array,
    actions: Actions,
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
        action_t = actions[:, prefix_len + step_offset]
        latent_next, _, caches, rng, _ = next_latent(
            dynamics=dynamics,
            schedule=schedule,
            action=action_t,
            latent_shape=latent_shape,
            rng=jax.random.fold_in(rng, int(step_offset)),
            caches=caches,
        )
        pred_latents.append(latent_next[:, 0])

    return jnp.stack(pred_latents, axis=1)


def rollout_uncached_from_prefix(
    *,
    dynamics,
    schedule: DenoiseSchedule,
    latents: jax.Array,
    actions: Actions,
    prefix_len: int,
    probe_steps: int,
    rng: jax.Array,
) -> jax.Array:
    latents_norm = normalize_latents(latents[:, :prefix_len], dynamics.cfg.latent_mean, dynamics.cfg.latent_std)
    actions_ctx = actions[:, :prefix_len]
    pred_latents = []
    latent_shape = (latents.shape[0], 1, latents.shape[2], latents.shape[3])

    for step_offset in range(probe_steps):
        latents_ctx_window, actions_ctx_window = crop_to_window(latents_norm, actions_ctx, dynamics.cfg.context_length)
        prefill_length = latents_ctx_window.shape[1]
        latent_next, _, _, _, _ = next_latent(
            dynamics=dynamics,
            schedule=schedule,
            action=actions[:, prefix_len + step_offset],
            latent_shape=latent_shape,
            rng=jax.random.fold_in(rng, int(step_offset)),
            caches=None,
            latents_ctx=latents_ctx_window,
            actions_ctx=actions_ctx_window,
            prefill_length=prefill_length,
        )
        pred_latents.append(latent_next[:, 0])
        latent_next_norm = normalize_latents(latent_next, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)
        latents_norm = jnp.concatenate([latents_norm, latent_next_norm], axis=1)
        next_action = actions[:, prefix_len + step_offset][:, None]
        actions_ctx = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), actions_ctx, next_action)

    return jnp.stack(pred_latents, axis=1)


def save_contact(path: Path, *, gt_frames: np.ndarray, cached_frames: np.ndarray, uncached_frames: np.ndarray, prefix_len: int) -> None:
    frame_indices = list(range(gt_frames.shape[0]))
    rows = [
        (gt_frames, "GT future"),
        (cached_frames, "Cached"),
        (uncached_frames, "Uncached"),
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


def save_mse_plot(path: Path, cached_mse: np.ndarray, uncached_mse: np.ndarray, prefix_len: int) -> None:
    timesteps = np.arange(prefix_len, prefix_len + len(cached_mse))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(timesteps, cached_mse, marker="o", linewidth=2.0, label="Cached vs GT")
    ax.plot(timesteps, uncached_mse, marker="o", linewidth=2.0, label="Uncached vs GT")
    ax.set_xlabel("Absolute frame")
    ax.set_ylabel("Pixel MSE")
    ax.set_title("Boundary continuation from the same GT prefix")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_mp4(path: Path, *, gt_frames: np.ndarray, cached_frames: np.ndarray, uncached_frames: np.ndarray) -> None:
    video = np.concatenate([gt_frames, cached_frames, uncached_frames], axis=2)
    iio.imwrite(str(path), video, fps=2, plugin="pyav", codec="libx264")


def framewise_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2, axis=(1, 2, 3))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = build_dataset_cfg(args)
    mesh, data_sharding, mesh_rules = build_parallel(args.parallel_strategy)

    t0 = time.time()
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

        if args.synthetic_static:
            latents = jnp.repeat(latents[:, :1], repeats=latents.shape[1], axis=1)
            actions = repeat_noop_actions(actions, dataset_cfg.categorical_action_dim, latents.shape[1])

        schedule = DenoiseSchedule.init(4, dynamics.cfg.k_max)
        rng = jax.random.PRNGKey(args.seed)
        rng_cached, rng_uncached = jax.random.split(rng)

        cached_latents = rollout_cached_from_prefix(
            dynamics=dynamics,
            schedule=schedule,
            latents=latents,
            actions=actions,
            prefix_len=args.prefix_len,
            probe_steps=args.probe_steps,
            rng=rng_cached,
        )
        uncached_latents = rollout_uncached_from_prefix(
            dynamics=dynamics,
            schedule=schedule,
            latents=latents,
            actions=actions,
            prefix_len=args.prefix_len,
            probe_steps=args.probe_steps,
            rng=rng_uncached,
        )

        gt_future = latents[:, args.prefix_len:args.prefix_len + args.probe_steps]
        cached_frames = jnp.clip(decode_latents(tokenizer, cached_latents), 0, 255).astype(jnp.uint8)
        uncached_frames = jnp.clip(decode_latents(tokenizer, uncached_latents), 0, 255).astype(jnp.uint8)
        gt_frames = jnp.clip(decode_latents(tokenizer, gt_future), 0, 255).astype(jnp.uint8)
        cached_frames, uncached_frames, gt_frames, cached_latents, uncached_latents = jax.device_get(
            (cached_frames, uncached_frames, gt_frames, cached_latents, uncached_latents)
        )

    sample_index = int(np.clip(args.sample_index, 0, gt_frames.shape[0] - 1))
    gt_np = np.asarray(gt_frames[sample_index])
    cached_np = np.asarray(cached_frames[sample_index])
    uncached_np = np.asarray(uncached_frames[sample_index])
    cached_gt_mse = framewise_mse(cached_np, gt_np)
    uncached_gt_mse = framewise_mse(uncached_np, gt_np)
    cached_uncached_mse = framewise_mse(cached_np, uncached_np)

    save_contact(
        output_dir / "boundary_contact.png",
        gt_frames=gt_np,
        cached_frames=cached_np,
        uncached_frames=uncached_np,
        prefix_len=args.prefix_len,
    )
    save_mse_plot(output_dir / "boundary_mse.png", cached_gt_mse, uncached_gt_mse, args.prefix_len)
    save_mp4(output_dir / "boundary_compare.mp4", gt_frames=gt_np, cached_frames=cached_np, uncached_frames=uncached_np)

    metrics = {
        "model": "online" if args.use_online else "ema",
        "synthetic_static": bool(args.synthetic_static),
        "sample_index": sample_index,
        "prefix_len": int(args.prefix_len),
        "probe_steps": int(args.probe_steps),
        "context_length": None if dynamics.cfg.context_length is None else int(dynamics.cfg.context_length),
        "cached_vs_uncached_latent_mse_per_step": [
            float(jnp.mean((cached_latents[:, i] - uncached_latents[:, i]) ** 2))
            for i in range(args.probe_steps)
        ],
        "cached_vs_uncached_pixel_mse_per_step": [float(x) for x in cached_uncached_mse],
        "cached_vs_gt_pixel_mse_per_step": [float(x) for x in cached_gt_mse],
        "uncached_vs_gt_pixel_mse_per_step": [float(x) for x in uncached_gt_mse],
        "elapsed_sec": time.time() - t0,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
