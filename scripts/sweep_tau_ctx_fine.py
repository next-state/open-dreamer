"""Fine sweep tau_ctx_target in [0.65, 0.95] with 0.01 steps."""

import logging
import time
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from einops import rearrange
from omegaconf import OmegaConf

from dreamer.data import make_iterator
from dreamer.parallel import build_parallel
from dreamer.actions import shift_actions
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.generation import DenoiseSchedule
from dreamer.sampler import sample_video
from dreamer.utils import apply_border, normalize_with_dataset_stats

logging.getLogger('absl').setLevel(logging.WARNING)

OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args), replace=True)
OmegaConf.register_new_resolver("sum", lambda *args: sum(args), replace=True)
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y, replace=True)
OmegaConf.register_new_resolver("max", lambda *args: max(args), replace=True)

jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


def evaluate_tau(dynamics_model, tokenizer, val_data, val_actions, use_latent_data, dataset_std, k_max, tau_ctx_target, rng):
    """Run evaluation with a given tau_ctx_target. Returns metrics plus frames."""
    ctx_length = 4
    T = val_data.shape[1]
    horizon = T - ctx_length

    schedule = DenoiseSchedule.init(4, k_max, tau_ctx_target=tau_ctx_target)

    rng, eval_rng = jax.random.split(rng)

    if True:  # use_latent_data
        pred_frames, gt_decoded_frames, _, _ = sample_video(
            tokenizer, dynamics_model, frames=None,
            actions=val_actions, horizon=horizon, schedule_config=schedule,
            rng=eval_rng, policy=None, task_embedder=None,
            latents=val_data,
        )
        gt_frames = gt_decoded_frames

    normalized_pred = normalize_with_dataset_stats(pred_frames[:, -horizon:], mean=0, std=dataset_std)
    normalized_gt = normalize_with_dataset_stats(gt_frames[:, -horizon:], mean=0, std=dataset_std)
    mse = float(jnp.mean((normalized_pred - normalized_gt) ** 2))
    pred_frames = pred_frames.at[:, :ctx_length].set(apply_border(pred_frames[:, :ctx_length]))
    return mse, pred_frames, gt_frames, rng


def format_tau_dirname(tau_ctx_target: float) -> str:
    tau_token = f"{tau_ctx_target:.3f}".rstrip("0").rstrip(".")
    return f"tau_{tau_token.replace('.', 'p')}"


def save_rollout_video(output_dir: Path, tau_ctx_target: float, gt_frames, online_frames, ema_frames) -> Path:
    tau_dir = output_dir / "videos" / format_tau_dirname(tau_ctx_target)
    tau_dir.mkdir(parents=True, exist_ok=True)

    num_videos = min(4, gt_frames.shape[0])
    stacked_frames = jnp.stack([gt_frames, online_frames, ema_frames])[:, :num_videos]
    video = rearrange(stacked_frames, "S B T H W C -> T (B H) (S W) C", B=num_videos)
    mp4_path = tau_dir / "rollouts_grid.mp4"
    iio.imwrite(str(mp4_path), jax.device_get(video), fps=20, plugin="pyav", codec="libx264")
    return mp4_path


def main():
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="eval_dynamics", overrides=[
            "dynamics_ckpt=/home/ubuntu/checkpoints",
            "output_dir=eval_outputs/sweep_tau_ctx_050_070",
            "dataset.array_record_path=/home/ubuntu/latents-0312-h200",
            "dataset.index_max=4",
        ])

    rng = jax.random.PRNGKey(cfg.seed)
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with jax.set_mesh(mesh):
        print("Loading checkpoint...")
        bundle = DynamicsCheckpointBundle.from_pretrained(
            cfg.dynamics_ckpt, mesh_rules=mesh_rules, model_names={"dynamics", "dynamics_ema", "tokenizer"}
        )
        k_max = bundle.dynamics_ema.cfg.k_max
        print(f"Loaded dynamics models (k_max={k_max})")

        cfg.dataset.dataloader_cfg.short_T = 128
        cfg.dataset.dataloader_cfg.long_T = 128
        print("Loading data...")
        iterator = make_iterator(cfg.dataset, device=data_sharding)
        batch = next(iter(iterator))
        actions = batch["actions"]
        actions = shift_actions(actions, cfg.dataset.categorical_action_dim)
        input_tensor = batch.get("latents")
        print(f"Data loaded: shape={input_tensor.shape}")

        dataset_std = tuple(cfg.dataset.dataset_std)[0]

        # Sweep [0.50, 0.70] in 0.01 steps = 21 values
        tau_values = np.arange(0.50, 0.71, 0.01).tolist()
        tau_values = [round(v, 2) for v in tau_values]

        print(f"\n=== Sweeping {len(tau_values)} values from {tau_values[0]} to {tau_values[-1]} ===")

        results = {}
        output_dir = Path("eval_outputs/sweep_tau_ctx_050_070")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, tau in enumerate(tau_values):
            t0 = time.time()
            # Use a fixed rng per tau for fair comparison (same noise for each tau)
            eval_rng = jax.random.PRNGKey(42)

            mse_online, online_frames, gt_frames, _ = evaluate_tau(
                bundle.dynamics, bundle.tokenizer, input_tensor, actions,
                True, dataset_std, k_max, tau, eval_rng
            )
            # Use same base seed for ema for fair comparison
            eval_rng2 = jax.random.PRNGKey(42)
            mse_ema, ema_frames, _, _ = evaluate_tau(
                bundle.dynamics_ema, bundle.tokenizer, input_tensor, actions,
                True, dataset_std, k_max, tau, eval_rng2
            )
            mp4_path = save_rollout_video(output_dir, tau, gt_frames, online_frames, ema_frames)
            results[tau] = {"online": mse_online, "ema": mse_ema, "video_path": str(mp4_path)}
            dt = time.time() - t0
            print(
                f"  [{i+1}/{len(tau_values)}] tau_ctx={tau:.2f} | "
                f"MSE online={mse_online:.6f} | MSE ema={mse_ema:.6f} | "
                f"{dt:.1f}s | video={mp4_path}"
            )

        # Save results
        results_json = {str(k): v for k, v in results.items()}
        with open(output_dir / "sweep_results.json", "w") as f:
            json.dump(results_json, f, indent=2)

        # Plot
        sorted_taus = sorted(results.keys())
        online_mses = [results[t]["online"] for t in sorted_taus]
        ema_mses = [results[t]["ema"] for t in sorted_taus]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sorted_taus, online_mses, 'o-', label='Online Shortcut', markersize=4, linewidth=1.5)
        ax.plot(sorted_taus, ema_mses, 's-', label='EMA Shortcut', markersize=4, linewidth=1.5)
        ax.set_xlabel('tau_ctx_target', fontsize=14)
        ax.set_ylabel('MSE', fontsize=14)
        ax.set_title('MSE vs tau_ctx_target (fine sweep, 0.50–0.70)', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Mark best points
        best_online = min(results, key=lambda t: results[t]["online"])
        best_ema = min(results, key=lambda t: results[t]["ema"])
        ax.axvline(x=best_online, color='blue', linestyle='--', alpha=0.4)
        ax.axvline(x=best_ema, color='orange', linestyle='--', alpha=0.4)
        ax.annotate(f'Best online={best_online:.2f}', xy=(best_online, results[best_online]["online"]),
                    xytext=(10, 10), textcoords='offset points', fontsize=10, color='blue')
        ax.annotate(f'Best EMA={best_ema:.2f}', xy=(best_ema, results[best_ema]["ema"]),
                    xytext=(10, -15), textcoords='offset points', fontsize=10, color='orange')

        plt.tight_layout()
        plot_path = output_dir / "mse_vs_tau_ctx_fine.png"
        fig.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")

        # Summary
        print(f"\n=== Summary ===")
        print(f"Best online: tau_ctx={best_online:.2f}, MSE={results[best_online]['online']:.6f}")
        print(f"Best EMA:    tau_ctx={best_ema:.2f}, MSE={results[best_ema]['ema']:.6f}")
        print(f"\n{'tau_ctx':>10} | {'MSE online':>12} | {'MSE ema':>12}")
        print("-" * 40)
        for tau in sorted_taus:
            marker = ""
            if tau == best_online:
                marker += " <-- best online"
            if tau == best_ema:
                marker += " <-- best ema"
            print(f"{tau:10.2f} | {results[tau]['online']:12.6f} | {results[tau]['ema']:12.6f}{marker}")


if __name__ == "__main__":
    main()
