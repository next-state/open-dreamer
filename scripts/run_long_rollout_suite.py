from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
from dreamer.utils import apply_border


RED = (255, 0, 0)
BLACK = (0, 0, 0)


@dataclass
class CaseSpec:
    label: str
    prompt_length: int
    schedule_steps: int
    context_start: int = 0


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-sequence long-rollout experiment suites.")
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
    parser.add_argument("--prompt-lengths", default="4")
    parser.add_argument("--schedule-steps-list", default="4")
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


def build_cases(prompt_lengths: list[int], schedule_steps: list[int], seq_len: int) -> list[CaseSpec]:
    if len(prompt_lengths) > 1 and len(schedule_steps) > 1:
        raise ValueError("Vary either prompt lengths or schedule steps in one run, not both.")
    if len(prompt_lengths) > 1:
        return [CaseSpec(label=f"{prompt}->{seq_len - prompt}", prompt_length=prompt, schedule_steps=schedule_steps[0]) for prompt in prompt_lengths]
    return [CaseSpec(label=f"k={steps}", prompt_length=prompt_lengths[0], schedule_steps=steps) for steps in schedule_steps]


def framewise_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2, axis=(1, 2, 3))


def decorate_prediction_frames(frames: np.ndarray, *, context_start: int, prompt_length: int) -> np.ndarray:
    decorated = jnp.asarray(frames)
    if context_start > 0:
        decorated = decorated.at[:context_start].set(apply_border(decorated[:context_start], color=BLACK))
    if prompt_length > 0:
        ctx_end = context_start + prompt_length
        decorated = decorated.at[context_start:ctx_end].set(
            apply_border(decorated[context_start:ctx_end], color=RED)
        )
    return np.asarray(decorated)


def selected_frames(total_frames: int) -> list[int]:
    picks = [0, 4, 16, 64, 128, 160, 192, 224, total_frames - 1]
    return sorted({idx for idx in picks if 0 <= idx < total_frames})


def save_contact(path: Path, *, gt_frames: np.ndarray, case_frames: list[np.ndarray], labels: list[str]) -> None:
    frame_ids = selected_frames(gt_frames.shape[0])
    fig, axes = plt.subplots(1 + len(case_frames), len(frame_ids), figsize=(2.2 * len(frame_ids), 2.0 * (1 + len(case_frames))))
    rows = [(gt_frames, "GT")] + list(zip(case_frames, labels, strict=True))
    for row_idx, (frames, row_label) in enumerate(rows):
        for col_idx, frame_idx in enumerate(frame_ids):
            axes[row_idx, col_idx].imshow(frames[frame_idx])
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(str(frame_idx))
        axes[row_idx, 0].set_ylabel(row_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_mse_plot(path: Path, *, total_frames: int, curves: list[np.ndarray], cases: list[CaseSpec]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(total_frames)
    for case, mse in zip(cases, curves, strict=True):
        display = np.full((total_frames,), np.nan, dtype=np.float32)
        display[case.prompt_length:] = mse[case.prompt_length:]
        ax.plot(xs, display, label=case.label, linewidth=2.0)
    ax.axvline(128, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Absolute frame")
    ax.set_ylabel("Pixel MSE")
    ax.set_title("Generated-frame MSE across long-rollout cases")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_dashboard(path: Path, *, contact_path: Path, mse_path: Path, title: str) -> None:
    contact = plt.imread(contact_path)
    mse = plt.imread(mse_path)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    axes[0].imshow(contact)
    axes[0].axis("off")
    axes[0].set_title(title)
    axes[1].imshow(mse)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_video(path: Path, *, gt_frames: np.ndarray, case_frames: list[np.ndarray]) -> None:
    video = np.concatenate([gt_frames] + case_frames, axis=2)
    iio.imwrite(str(path), video, fps=5, plugin="pyav", codec="libx264")


def metrics_for_case(case: CaseSpec, mse: np.ndarray) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "prompt_length": case.prompt_length,
        "generated_horizon": int(len(mse) - case.prompt_length),
        "schedule_steps": case.schedule_steps,
        "mean_generated_mse": float(np.mean(mse[case.prompt_length:])),
    }
    for frame_idx in [128, 160, 192, 224, len(mse) - 1]:
        if frame_idx < len(mse):
            metrics[f"mse_{frame_idx}"] = float(mse[frame_idx])
    return metrics


def save_summary(
    path: Path,
    *,
    title: str,
    model_name: str,
    dataset_path: str,
    sample_index: int,
    cases: list[CaseSpec],
    metrics: dict[str, dict[str, float | int]],
    video_path: Path,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- model: `{model_name}`",
        f"- dataset: `{dataset_path}`",
        f"- sample_index: `{sample_index}`",
        f"- video: `{video_path}`",
        "",
        "Border semantics:",
        "- GT column: no border",
        "- prediction columns: red border on context frames used for inference",
        "- prediction columns: black border on shown-but-unused prefix frames",
        "- generated frames: no border",
        "",
        "Case metrics:",
    ]
    for case in cases:
        case_metrics = metrics[case.label]
        metric_text = ", ".join(f"{key}={value}" for key, value in case_metrics.items())
        lines.append(f"- `{case.label}`: {metric_text}")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    prompt_lengths = parse_int_list(args.prompt_lengths)
    schedule_steps = parse_int_list(args.schedule_steps_list)
    cases = build_cases(prompt_lengths, schedule_steps, args.seq_len)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_index < 0 or args.sample_index >= args.batch_size:
        raise ValueError(f"sample_index must be in [0, {args.batch_size - 1}]")

    for case in cases:
        if case.prompt_length <= 0 or case.prompt_length >= args.seq_len:
            raise ValueError(f"Invalid prompt length {case.prompt_length} for seq_len={args.seq_len}")

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

        sample_index = args.sample_index
        gt_frames = decode_latents(tokenizer, latents)
        gt_frames = jnp.clip(gt_frames, 0, 255).astype(jnp.uint8)
        gt_frames = np.asarray(jax.device_get(gt_frames[sample_index]))

        case_frames: list[np.ndarray] = []
        mse_curves: list[np.ndarray] = []
        metrics: dict[str, dict[str, float | int]] = {}

        for case in cases:
            rng = jax.random.PRNGKey(args.seed)
            schedule = DenoiseSchedule.init(case.schedule_steps, dynamics.cfg.k_max)
            rollout = latent_rollout(
                dynamics=dynamics,
                policy=actions[:, case.prompt_length:],
                schedule=schedule,
                latents_ctx=latents[:, :case.prompt_length],
                actions_ctx=actions[:, :case.prompt_length],
                num_steps=args.seq_len - case.prompt_length,
                rng=rng,
                deterministic=True,
                use_kv_cache=True,
            )
            pred_frames = decode_latents(tokenizer, rollout["latents"])
            pred_frames = jnp.clip(pred_frames, 0, 255).astype(jnp.uint8)
            pred_frames = np.asarray(jax.device_get(pred_frames[sample_index]))

            decorated = decorate_prediction_frames(pred_frames, context_start=case.context_start, prompt_length=case.prompt_length)
            case_frames.append(decorated)

            mse = framewise_mse(pred_frames, gt_frames)
            mse_curves.append(mse)
            metrics[case.label] = metrics_for_case(case, mse)

    save_contact(output_dir / "contact.png", gt_frames=gt_frames, case_frames=case_frames, labels=[case.label for case in cases])
    save_mse_plot(output_dir / "mse.png", total_frames=args.seq_len, curves=mse_curves, cases=cases)
    title = args.title or "Long rollout suite"
    save_dashboard(output_dir / "dashboard.png", contact_path=output_dir / "contact.png", mse_path=output_dir / "mse.png", title=title)
    save_video(output_dir / "rollouts.mp4", gt_frames=gt_frames, case_frames=case_frames)
    save_summary(
        output_dir / "summary.md",
        title=title,
        model_name="online" if args.use_online else "ema",
        dataset_path=args.array_record_path,
        sample_index=args.sample_index,
        cases=cases,
        metrics=metrics,
        video_path=output_dir / "rollouts.mp4",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
