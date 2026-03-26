from __future__ import annotations

import argparse
import json
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
from dreamer.generation import DenoiseSchedule, latent_rollout
from dreamer.parallel import build_parallel
from dreamer.sampler import decode_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare long-context and cropped-context rollouts on the same target horizon.")
    parser.add_argument("--dynamics-ckpt", required=True)
    parser.add_argument("--array-record-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index-max", type=int, default=1)
    parser.add_argument("--num-max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--long-context", type=int, default=128)
    parser.add_argument("--short-context-start", type=int, default=124)
    parser.add_argument("--short-context-len", type=int, default=4)
    parser.add_argument("--schedule-steps", type=int, default=4)
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


def framewise_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2, axis=(1, 2, 3))


def save_contact(path: Path, *, gt_frames: np.ndarray, long_frames: np.ndarray, short_frames: np.ndarray, abs_indices: list[int]) -> None:
    fig, axes = plt.subplots(3, len(abs_indices), figsize=(2.2 * len(abs_indices), 6.2))
    rows = [
        (gt_frames, "GT"),
        (long_frames, "128-frame prompt"),
        (short_frames, "4-frame prompt"),
    ]
    rel_indices = [idx - abs_indices[0] for idx in abs_indices]
    for row_idx, (frames, row_title) in enumerate(rows):
        for col_idx, (abs_idx, rel_idx) in enumerate(zip(abs_indices, rel_indices, strict=True)):
            axes[row_idx, col_idx].imshow(frames[rel_idx])
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(str(abs_idx))
        axes[row_idx, 0].set_ylabel(row_title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_prompt_overview(path: Path, *, full_frames: np.ndarray, long_context: int, short_start: int, short_len: int) -> None:
    long_indices = [0, 32, 64, 96, long_context - 1]
    short_indices = list(range(short_start, short_start + short_len))
    all_indices = long_indices + short_indices
    fig, axes = plt.subplots(1, len(all_indices), figsize=(2.2 * len(all_indices), 2.3))
    for col_idx, frame_idx in enumerate(all_indices):
        axes[col_idx].imshow(full_frames[frame_idx])
        axes[col_idx].axis("off")
        axes[col_idx].set_title(str(frame_idx))
    fig.suptitle("Prompt overview (long prompt frames, then cropped prompt frames)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_mse_plot(path: Path, *, gt_indices: np.ndarray, long_mse: np.ndarray, short_mse: np.ndarray, gap_mse: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gt_indices, long_mse, label="128-frame prompt vs GT", linewidth=2.0)
    ax.plot(gt_indices, short_mse, label="4-frame prompt vs GT", linewidth=2.0)
    ax.plot(gt_indices, gap_mse, label="Prompt A/B gap", linewidth=2.0, linestyle="--")
    ax.set_xlabel("Absolute frame")
    ax.set_ylabel("Pixel MSE")
    ax.set_title("Prompt-span comparison on the same target horizon")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_mp4(path: Path, *, gt_frames: np.ndarray, long_frames: np.ndarray, short_frames: np.ndarray) -> None:
    video = np.concatenate([gt_frames, long_frames, short_frames], axis=2)
    iio.imwrite(str(path), video, fps=5, plugin="pyav", codec="libx264")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.long_context + 128 > args.seq_len:
        raise ValueError("Need at least 128 future frames after the long prompt.")
    if args.short_context_start + args.short_context_len != args.long_context:
        raise ValueError("Expected the short prompt to end exactly at the long prompt boundary.")
    if args.sample_index < 0 or args.sample_index >= args.batch_size:
        raise ValueError(f"sample_index must be in [0, {args.batch_size - 1}]")

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

        horizon = 128
        rng = jax.random.PRNGKey(args.seed)
        schedule = DenoiseSchedule.init(args.schedule_steps, dynamics.cfg.k_max)

        rollout_long = latent_rollout(
            dynamics=dynamics,
            policy=actions[:, args.long_context:args.long_context + horizon],
            schedule=schedule,
            latents_ctx=latents[:, :args.long_context],
            actions_ctx=actions[:, :args.long_context],
            num_steps=horizon,
            rng=rng,
            deterministic=True,
            use_kv_cache=True,
        )
        rollout_short = latent_rollout(
            dynamics=dynamics,
            policy=actions[:, args.long_context:args.long_context + horizon],
            schedule=schedule,
            latents_ctx=latents[:, args.short_context_start:args.long_context],
            actions_ctx=actions[:, args.short_context_start:args.long_context],
            num_steps=horizon,
            rng=rng,
            deterministic=True,
            use_kv_cache=True,
        )

        pred_long_future = rollout_long["latents"][:, args.long_context:]
        pred_short_future = rollout_short["latents"][:, args.short_context_len:]
        gt_future = latents[:, args.long_context:args.long_context + horizon]

        gt_future_frames = jnp.clip(decode_latents(tokenizer, gt_future), 0, 255).astype(jnp.uint8)
        pred_long_frames = jnp.clip(decode_latents(tokenizer, pred_long_future), 0, 255).astype(jnp.uint8)
        pred_short_frames = jnp.clip(decode_latents(tokenizer, pred_short_future), 0, 255).astype(jnp.uint8)
        full_frames = jnp.clip(decode_latents(tokenizer, latents), 0, 255).astype(jnp.uint8)
        gt_future_frames, pred_long_frames, pred_short_frames, full_frames = jax.device_get(
            (gt_future_frames, pred_long_frames, pred_short_frames, full_frames)
        )

    sample_index = args.sample_index
    gt_np = np.asarray(gt_future_frames[sample_index])
    long_np = np.asarray(pred_long_frames[sample_index])
    short_np = np.asarray(pred_short_frames[sample_index])
    full_np = np.asarray(full_frames[sample_index])

    gt_indices = np.arange(args.long_context, args.long_context + horizon)
    long_mse = framewise_mse(long_np, gt_np)
    short_mse = framewise_mse(short_np, gt_np)
    gap_mse = framewise_mse(long_np, short_np)

    save_contact(
        output_dir / "future_contact.png",
        gt_frames=gt_np,
        long_frames=long_np,
        short_frames=short_np,
        abs_indices=[128, 144, 160, 192, 224, 255],
    )
    save_prompt_overview(
        output_dir / "prompt_overview.png",
        full_frames=full_np,
        long_context=args.long_context,
        short_start=args.short_context_start,
        short_len=args.short_context_len,
    )
    save_mse_plot(
        output_dir / "future_mse.png",
        gt_indices=gt_indices,
        long_mse=long_mse,
        short_mse=short_mse,
        gap_mse=gap_mse,
    )
    save_mp4(
        output_dir / "future_compare.mp4",
        gt_frames=gt_np,
        long_frames=long_np,
        short_frames=short_np,
    )

    metrics = {
        "model": "online" if args.use_online else "ema",
        "sample_index": args.sample_index,
        "schedule_steps": args.schedule_steps,
        "transformer_context_length": int(dynamics.cfg.context_length) if dynamics.cfg.context_length is not None else None,
        "long_context": args.long_context,
        "short_context_start": args.short_context_start,
        "short_context_len": args.short_context_len,
        "target_horizon": horizon,
        "long_vs_gt_frame_128": float(long_mse[0]),
        "short_vs_gt_frame_128": float(short_mse[0]),
        "gap_frame_128": float(gap_mse[0]),
        "long_vs_gt_frame_160": float(long_mse[32]),
        "short_vs_gt_frame_160": float(short_mse[32]),
        "gap_frame_160": float(gap_mse[32]),
        "long_vs_gt_frame_192": float(long_mse[64]),
        "short_vs_gt_frame_192": float(short_mse[64]),
        "gap_frame_192": float(gap_mse[64]),
        "long_vs_gt_frame_255": float(long_mse[-1]),
        "short_vs_gt_frame_255": float(short_mse[-1]),
        "gap_frame_255": float(gap_mse[-1]),
        "mean_long_vs_gt": float(np.mean(long_mse)),
        "mean_short_vs_gt": float(np.mean(short_mse)),
        "mean_prompt_gap": float(np.mean(gap_mse)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
