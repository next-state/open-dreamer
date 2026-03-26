from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe late-step one-step predictions from GT prefixes.")
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
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--schedule-steps", type=int, default=4)
    parser.add_argument("--probe-steps", default="160,192,224")
    parser.add_argument("--parallel-strategy", default="data")
    parser.add_argument("--use-online", action="store_true")
    parser.add_argument("--title", default="")
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


def frame_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2))


def one_step_from_gt_prefix(
    *,
    dynamics,
    latents: jax.Array,
    actions,
    prompt_length: int,
    probe_step: int,
    schedule: DenoiseSchedule,
    rng: jax.Array,
):
    context_start = max(0, probe_step - prompt_length)
    latents_ctx = latents[:, context_start:probe_step]
    actions_ctx = actions[:, context_start:probe_step]
    rollout = latent_rollout(
        dynamics=dynamics,
        policy=actions[:, probe_step:probe_step + 1],
        schedule=schedule,
        latents_ctx=latents_ctx,
        actions_ctx=actions_ctx,
        num_steps=1,
        rng=rng,
        deterministic=True,
        use_kv_cache=True,
    )
    return rollout["latents"][:, -1:]


def save_contact(path: Path, *, steps: list[int], gt_frames: list[np.ndarray], ar_frames: list[np.ndarray], gt_prefix_frames: list[np.ndarray]) -> None:
    fig, axes = plt.subplots(3, len(steps), figsize=(2.2 * len(steps), 6.2))
    rows = [
        (gt_frames, "GT"),
        (ar_frames, "Autoregressive"),
        (gt_prefix_frames, "GT-prefix one-step"),
    ]
    for row_idx, (frames, row_label) in enumerate(rows):
        for col_idx, (frame, step) in enumerate(zip(frames, steps, strict=True)):
            axes[row_idx, col_idx].imshow(frame)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(str(step))
        axes[row_idx, 0].set_ylabel(row_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bar_plot(path: Path, *, steps: list[int], ar_errors: list[float], gt_prefix_errors: list[float]) -> None:
    x = np.arange(len(steps))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, ar_errors, width, label="Autoregressive")
    ax.bar(x + width / 2, gt_prefix_errors, width, label="GT-prefix one-step")
    ax.set_xticks(x)
    ax.set_xticklabels([str(step) for step in steps])
    ax.set_xlabel("Probe step")
    ax.set_ylabel("Pixel MSE")
    ax.set_title("Late-step GT-prefix probes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_index < 0 or args.sample_index >= args.batch_size:
        raise ValueError(f"sample_index must be in [0, {args.batch_size - 1}]")
    probe_steps = parse_int_list(args.probe_steps)

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

        schedule = DenoiseSchedule.init(args.schedule_steps, dynamics.cfg.k_max)
        rollout = latent_rollout(
            dynamics=dynamics,
            policy=actions[:, args.prompt_length:],
            schedule=schedule,
            latents_ctx=latents[:, :args.prompt_length],
            actions_ctx=actions[:, :args.prompt_length],
            num_steps=args.seq_len - args.prompt_length,
            rng=jax.random.PRNGKey(args.seed),
            deterministic=True,
            use_kv_cache=True,
        )

        gt_frames = decode_latents(tokenizer, latents)
        ar_frames = decode_latents(tokenizer, rollout["latents"])
        gt_frames = jnp.clip(gt_frames, 0, 255).astype(jnp.uint8)
        ar_frames = jnp.clip(ar_frames, 0, 255).astype(jnp.uint8)

        sample_index = args.sample_index
        gt_frames_np = np.asarray(jax.device_get(gt_frames[sample_index]))
        ar_frames_np = np.asarray(jax.device_get(ar_frames[sample_index]))

        gt_list: list[np.ndarray] = []
        ar_list: list[np.ndarray] = []
        gt_prefix_list: list[np.ndarray] = []
        metrics: dict[str, dict[str, float | int]] = {}

        for step in probe_steps:
            one_step_latent = one_step_from_gt_prefix(
                dynamics=dynamics,
                latents=latents,
                actions=actions,
                prompt_length=args.prompt_length,
                probe_step=step,
                schedule=schedule,
                rng=jax.random.PRNGKey(args.seed + step),
            )
            one_step_frame = decode_latents(tokenizer, one_step_latent)
            one_step_frame = jnp.clip(one_step_frame, 0, 255).astype(jnp.uint8)
            one_step_frame_np = np.asarray(jax.device_get(one_step_frame[sample_index, 0]))

            gt_frame = gt_frames_np[step]
            ar_frame = ar_frames_np[step]
            gt_list.append(gt_frame)
            ar_list.append(ar_frame)
            gt_prefix_list.append(one_step_frame_np)
            metrics[str(step)] = {
                "autoregressive_mse": frame_mse(ar_frame, gt_frame),
                "gt_prefix_one_step_mse": frame_mse(one_step_frame_np, gt_frame),
            }

    save_contact(output_dir / "contact.png", steps=probe_steps, gt_frames=gt_list, ar_frames=ar_list, gt_prefix_frames=gt_prefix_list)
    save_bar_plot(
        output_dir / "mse_bars.png",
        steps=probe_steps,
        ar_errors=[float(metrics[str(step)]["autoregressive_mse"]) for step in probe_steps],
        gt_prefix_errors=[float(metrics[str(step)]["gt_prefix_one_step_mse"]) for step in probe_steps],
    )
    title = args.title or "Late-step GT-prefix probes"
    summary_lines = [
        f"# {title}",
        "",
        f"- dataset: `{args.array_record_path}`",
        f"- sample_index: `{args.sample_index}`",
        f"- prompt_length: `{args.prompt_length}`",
        f"- schedule_steps: `{args.schedule_steps}`",
        "",
        "Probe metrics:",
    ]
    for step in probe_steps:
        data = metrics[str(step)]
        summary_lines.append(
            f"- step `{step}`: autoregressive_mse={data['autoregressive_mse']}, gt_prefix_one_step_mse={data['gt_prefix_one_step_mse']}"
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
