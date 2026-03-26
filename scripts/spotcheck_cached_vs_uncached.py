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
from dreamer.generation import DenoiseSchedule, latent_rollout, next_latent
from dreamer.parallel import build_parallel
from dreamer.sampler import decode_latents
from dreamer.utils import normalize_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cached-vs-uncached late-step spot checks.")
    parser.add_argument("--dynamics-ckpt", required=True)
    parser.add_argument("--array-record-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index-max", type=int, default=1)
    parser.add_argument("--num-max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ctx-length", type=int, default=4)
    parser.add_argument("--spot-step", type=int, default=160)
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


def crop_to_context_window(
    latents_ctx_norm: jax.Array,
    actions_ctx: Actions,
    *,
    prefill_length: int,
    context_length: int | None,
) -> tuple[jax.Array, Actions, int]:
    if context_length is None:
        return latents_ctx_norm, actions_ctx, prefill_length

    keep_len = max(context_length - 1, 0)
    if latents_ctx_norm.shape[1] <= keep_len:
        return latents_ctx_norm, actions_ctx, prefill_length

    remove_len = latents_ctx_norm.shape[1] - keep_len
    latents_ctx_norm = latents_ctx_norm[:, -keep_len:] if keep_len > 0 else latents_ctx_norm[:, :0]
    actions_ctx = jax.tree.map(
        lambda x: x[:, -keep_len:] if x is not None and keep_len > 0 else x[:, :0] if x is not None else None,
        actions_ctx,
    )
    return latents_ctx_norm, actions_ctx, max(prefill_length - remove_len, 0)


def framewise_mse(pred_frames: jax.Array, gt_frames: jax.Array) -> np.ndarray:
    pred = pred_frames.astype(jnp.float32)
    gt = gt_frames.astype(jnp.float32)
    return np.asarray(jnp.mean((pred - gt) ** 2, axis=(0, 2, 3, 4)))


def write_gt_vs_pred_mp4(path: Path, gt_frames: jax.Array, pred_frames: jax.Array, fps: int = 5) -> None:
    video = jnp.concatenate([gt_frames, pred_frames], axis=3)
    video_np = np.asarray(jax.device_get(video))
    if video_np.ndim == 5:
        video_np = video_np[0]
    iio.imwrite(str(path), video_np, fps=fps, plugin="pyav", codec="libx264")


def save_mse_plot(path: Path, mse_curve: np.ndarray, spot_step: int, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(mse_curve)), mse_curve, linewidth=2.0, color="#0f766e")
    ax.axvline(spot_step, color="#b91c1c", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pixel MSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_contact_sheet(path: Path, gt_frames: jax.Array, pred_frames: jax.Array, title: str) -> None:
    total_frames = gt_frames.shape[1]
    sample_indices = [0, 4, 32, 64, 96, 128, 160, 192, total_frames - 1]
    sample_indices = [idx for idx in sample_indices if idx < total_frames]
    gt_np = np.asarray(jax.device_get(gt_frames[0]))
    pred_np = np.asarray(jax.device_get(pred_frames[0]))

    fig, axes = plt.subplots(2, len(sample_indices), figsize=(2.2 * len(sample_indices), 4.8))
    for col, idx in enumerate(sample_indices):
        axes[0, col].imshow(gt_np[idx])
        axes[0, col].set_title(f"GT {idx}")
        axes[1, col].imshow(pred_np[idx])
        axes[1, col].set_title(f"Pred {idx}")
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_spot_figure(
    path: Path,
    *,
    gt_frame: jax.Array,
    cached_frame: jax.Array,
    uncached_frame: jax.Array,
    spot_step: int,
    title: str,
) -> None:
    gt_np = np.asarray(jax.device_get(gt_frame[0, 0]))
    cached_np = np.asarray(jax.device_get(cached_frame[0, 0]))
    uncached_np = np.asarray(jax.device_get(uncached_frame[0, 0]))
    diff_cached = np.mean(np.abs(cached_np.astype(np.float32) - gt_np.astype(np.float32)), axis=-1)
    diff_uncached = np.mean(np.abs(uncached_np.astype(np.float32) - gt_np.astype(np.float32)), axis=-1)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    panels = [
        (gt_np, f"GT {spot_step}"),
        (cached_np, "Cached"),
        (uncached_np, "Uncached"),
        (diff_cached, "Abs diff cached"),
        (diff_uncached, "Abs diff uncached"),
    ]
    for ax, (image, panel_title) in zip(axes, panels, strict=True):
        if image.ndim == 2:
            ax.imshow(image, cmap="magma")
        else:
            ax.imshow(image)
        ax.set_title(panel_title)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_case(
    *,
    case_name: str,
    tokenizer,
    dynamics,
    latents: jax.Array,
    actions: Actions,
    ctx_length: int,
    schedule: DenoiseSchedule,
    spot_step: int,
    rng: jax.Array,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_frames = latents.shape[1]
    if not (ctx_length <= spot_step < total_frames):
        raise ValueError(f"spot_step={spot_step} must be in [{ctx_length}, {total_frames - 1}]")

    horizon = total_frames - ctx_length
    rollout_start = time.time()
    rollout = latent_rollout(
        dynamics=dynamics,
        policy=actions[:, ctx_length:],
        schedule=schedule,
        latents_ctx=latents[:, :ctx_length],
        actions_ctx=actions[:, :ctx_length],
        num_steps=horizon,
        rng=rng,
        deterministic=True,
        use_kv_cache=True,
    )
    pred_latents = jax.block_until_ready(rollout["latents"])
    rollout_time = time.time() - rollout_start

    decode_start = time.time()
    gt_frames = jnp.clip(decode_latents(tokenizer, latents), 0, 255).astype(jnp.uint8)
    pred_frames = jnp.clip(decode_latents(tokenizer, pred_latents), 0, 255).astype(jnp.uint8)
    gt_frames, pred_frames = jax.device_get((gt_frames, pred_frames))
    decode_time = time.time() - decode_start

    mse_curve = framewise_mse(pred_frames, gt_frames)
    save_mse_plot(output_dir / f"{case_name}_mse.png", mse_curve, spot_step, f"{case_name}: cached rollout vs GT")
    save_contact_sheet(
        output_dir / f"{case_name}_contact.png",
        gt_frames,
        pred_frames,
        f"{case_name}: GT vs cached rollout",
    )
    write_gt_vs_pred_mp4(output_dir / f"{case_name}_gt_vs_cached.mp4", gt_frames, pred_frames)

    context_length = dynamics.cfg.context_length
    prefix_latents_norm = normalize_latents(
        pred_latents[:, :spot_step],
        dynamics.cfg.latent_mean,
        dynamics.cfg.latent_std,
    )
    prefix_actions = actions[:, :spot_step]
    prefix_latents_norm, prefix_actions, prefill_length = crop_to_context_window(
        prefix_latents_norm,
        prefix_actions,
        prefill_length=ctx_length,
        context_length=context_length,
    )

    spot_start = time.time()
    uncached_latent, _, _, _, _ = next_latent(
        dynamics=dynamics,
        schedule=schedule,
        action=actions[:, spot_step],
        latent_shape=(latents.shape[0], 1, latents.shape[2], latents.shape[3]),
        rng=jax.random.fold_in(rng, int(spot_step)),
        prefill_length=prefill_length,
        task_embedding=None,
        caches=None,
        latents_ctx=prefix_latents_norm,
        actions_ctx=prefix_actions,
    )
    uncached_latent = jax.block_until_ready(uncached_latent)
    spot_time = time.time() - spot_start

    cached_latent = pred_latents[:, spot_step:spot_step + 1]
    gt_frame = gt_frames[:, spot_step:spot_step + 1]
    cached_frame = pred_frames[:, spot_step:spot_step + 1]
    uncached_frame = jnp.clip(decode_latents(tokenizer, uncached_latent), 0, 255).astype(jnp.uint8)
    uncached_frame = jax.device_get(uncached_frame)

    save_spot_figure(
        output_dir / f"{case_name}_spot_{spot_step}.png",
        gt_frame=gt_frame,
        cached_frame=cached_frame,
        uncached_frame=uncached_frame,
        spot_step=spot_step,
        title=f"{case_name}: cached vs uncached at frame {spot_step}",
    )

    metrics = {
        "case_name": case_name,
        "spot_step": int(spot_step),
        "context_length": None if context_length is None else int(context_length),
        "ctx_length": int(ctx_length),
        "rollout_time_sec": rollout_time,
        "decode_time_sec": decode_time,
        "spotcheck_time_sec": spot_time,
        "pixel_mse_frame_32": float(mse_curve[32]) if len(mse_curve) > 32 else None,
        "pixel_mse_frame_128": float(mse_curve[128]) if len(mse_curve) > 128 else None,
        "pixel_mse_frame_spot": float(mse_curve[spot_step]),
        "pixel_mse_frame_last": float(mse_curve[-1]),
        "cached_vs_uncached_latent_mse": float(jnp.mean((cached_latent - uncached_latent) ** 2)),
        "cached_vs_uncached_pixel_mse": float(
            jnp.mean(
                (
                    cached_frame.astype(jnp.float32)
                    - jnp.asarray(uncached_frame, dtype=jnp.float32)
                ) ** 2
            )
        ),
    }

    with (output_dir / f"{case_name}_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
        schedule = DenoiseSchedule.init(4, dynamics.cfg.k_max)

        if args.synthetic_static:
            latents = jnp.repeat(latents[:, :1], repeats=latents.shape[1], axis=1)
            actions = repeat_noop_actions(actions, dataset_cfg.categorical_action_dim, latents.shape[1])
            case_name = "synthetic_static"
        else:
            case_name = "low_motion"

        metrics = run_case(
            case_name=case_name,
            tokenizer=tokenizer,
            dynamics=dynamics,
            latents=latents,
            actions=actions,
            ctx_length=args.ctx_length,
            schedule=schedule,
            spot_step=args.spot_step,
            rng=rng,
            output_dir=output_dir,
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
