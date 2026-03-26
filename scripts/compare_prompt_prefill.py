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
    parser = argparse.ArgumentParser(description="Compare prompt prefill variants on shortcut rollout.")
    parser.add_argument("--dynamics-ckpt", required=True)
    parser.add_argument("--array-record-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index-max", type=int, default=4)
    parser.add_argument("--num-max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--ctx-length", type=int, default=4)
    parser.add_argument("--bottom-row-only", action="store_true")
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


def save_bottom_contact(path: Path, gt_frames: np.ndarray, clean_frames: np.ndarray, noised_frames: np.ndarray) -> None:
    sample_indices = [0, 4, 32, 64, 96, 128, 160, 192, 255]
    fig, axes = plt.subplots(3, len(sample_indices), figsize=(2.1 * len(sample_indices), 6.3))
    rows = [
        (gt_frames, "GT"),
        (clean_frames, "Clean prompt"),
        (noised_frames, "Noised prompt"),
    ]
    for row_idx, (frames, row_title) in enumerate(rows):
        for col_idx, frame_idx in enumerate(sample_indices):
            axes[row_idx, col_idx].imshow(frames[frame_idx])
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(str(frame_idx))
        axes[row_idx, 0].set_ylabel(row_title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_mse_plot(path: Path, mse_clean: np.ndarray, mse_noised: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mse_clean, label="Clean prompt", color="#b91c1c", linewidth=2.0)
    ax.plot(mse_noised, label="Noised prompt", color="#0f766e", linewidth=2.0)
    ax.axvline(160, color="#334155", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Bottom-sample pixel MSE vs GT")
    ax.set_title("Prompt prefill comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_mp4(path: Path, gt_frames: np.ndarray, clean_frames: np.ndarray, noised_frames: np.ndarray) -> None:
    video = np.concatenate([gt_frames, clean_frames, noised_frames], axis=2)
    iio.imwrite(str(path), video, fps=5, plugin="pyav", codec="libx264")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = build_dataset_cfg(args)
    mesh, data_sharding, mesh_rules = build_parallel("data")

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

        schedule = DenoiseSchedule.init(4, dynamics.cfg.k_max)
        rng = jax.random.PRNGKey(args.seed)
        rng_clean, rng_noised = jax.random.split(rng)

        rollout_clean = latent_rollout(
            dynamics=dynamics,
            policy=actions[:, args.ctx_length:],
            schedule=schedule,
            latents_ctx=latents[:, :args.ctx_length],
            actions_ctx=actions[:, :args.ctx_length],
            num_steps=args.seq_len - args.ctx_length,
            rng=rng_clean,
            deterministic=True,
            use_kv_cache=True,
            prefill_context_mode="clean",
        )
        rollout_noised = latent_rollout(
            dynamics=dynamics,
            policy=actions[:, args.ctx_length:],
            schedule=schedule,
            latents_ctx=latents[:, :args.ctx_length],
            actions_ctx=actions[:, :args.ctx_length],
            num_steps=args.seq_len - args.ctx_length,
            rng=rng_noised,
            deterministic=True,
            use_kv_cache=True,
            prefill_context_mode="ctx_noised",
        )
        latents_clean = rollout_clean["latents"]
        latents_noised = rollout_noised["latents"]

        gt_frames = jnp.clip(decode_latents(tokenizer, latents), 0, 255).astype(jnp.uint8)
        clean_frames = jnp.clip(decode_latents(tokenizer, latents_clean), 0, 255).astype(jnp.uint8)
        noised_frames = jnp.clip(decode_latents(tokenizer, latents_noised), 0, 255).astype(jnp.uint8)
        gt_frames, clean_frames, noised_frames = jax.device_get((gt_frames, clean_frames, noised_frames))

    sample_index = gt_frames.shape[0] - 1 if args.bottom_row_only or gt_frames.shape[0] > 1 else 0
    gt_bottom = np.asarray(gt_frames[sample_index])
    clean_bottom = np.asarray(clean_frames[sample_index])
    noised_bottom = np.asarray(noised_frames[sample_index])

    mse_clean = np.mean((clean_bottom.astype(np.float32) - gt_bottom.astype(np.float32)) ** 2, axis=(1, 2, 3))
    mse_noised = np.mean((noised_bottom.astype(np.float32) - gt_bottom.astype(np.float32)) ** 2, axis=(1, 2, 3))

    save_bottom_contact(output_dir / "bottom_contact.png", gt_bottom, clean_bottom, noised_bottom)
    save_mse_plot(output_dir / "bottom_mse_plot.png", mse_clean, mse_noised)
    save_mp4(output_dir / "bottom_compare.mp4", gt_bottom, clean_bottom, noised_bottom)

    metrics = {
        "model": "online" if args.use_online else "ema",
        "sample_index": int(sample_index),
        "clean_prompt_mse_160": float(mse_clean[160]),
        "clean_prompt_mse_255": float(mse_clean[-1]),
        "noised_prompt_mse_160": float(mse_noised[160]),
        "noised_prompt_mse_255": float(mse_noised[-1]),
        "delta_mse_160": float(mse_noised[160] - mse_clean[160]),
        "delta_mse_255": float(mse_noised[-1] - mse_clean[-1]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
