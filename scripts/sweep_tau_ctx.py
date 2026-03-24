"""Sweep tau_ctx_target and plot MSE for online and EMA shortcut models."""

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
    """Run a single evaluation with a given tau_ctx_target and return metrics plus frames."""
    ctx_length = 4
    T = val_data.shape[1]
    horizon = T - ctx_length

    schedule = DenoiseSchedule.init(4, k_max, tau_ctx_target=tau_ctx_target)

    rng, eval_rng = jax.random.split(rng)

    if use_latent_data:
        pred_frames, gt_decoded_frames, _, _ = sample_video(
            tokenizer, dynamics_model, frames=None,
            actions=val_actions, horizon=horizon, schedule_config=schedule,
            rng=eval_rng, policy=None, task_embedder=None,
            latents=val_data,
        )
        gt_frames = gt_decoded_frames
    else:
        pred_frames, _, original_frames, _ = sample_video(
            tokenizer, dynamics_model, frames=val_data,
            actions=val_actions, horizon=horizon, schedule_config=schedule,
            rng=eval_rng, policy=None, task_embedder=None,
        )
        gt_frames = original_frames

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
    import hydra
    from hydra import compose, initialize

    # Load config
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="eval_dynamics", overrides=[
            "dynamics_ckpt=/home/ubuntu/checkpoints",
            "output_dir=eval_outputs/sweep_tau_ctx",
            "dataset.array_record_path=/home/ubuntu/latents-0312-h200",
            "dataset.index_max=4",
        ])

    rng = jax.random.PRNGKey(cfg.seed)
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with jax.set_mesh(mesh):
        # Load checkpoint
        print("Loading checkpoint...")
        bundle = DynamicsCheckpointBundle.from_pretrained(
            cfg.dynamics_ckpt, mesh_rules=mesh_rules, model_names={"dynamics", "dynamics_ema", "tokenizer"}
        )
        k_max = bundle.dynamics_ema.cfg.k_max
        print(f"Loaded dynamics models (k_max={k_max})")

        use_latent_data = cfg.dataset.data_type == "latent"

        # Load data
        cfg.dataset.dataloader_cfg.short_T = 128
        cfg.dataset.dataloader_cfg.long_T = 128
        print("Loading data...")
        iterator = make_iterator(cfg.dataset, device=data_sharding)
        batch = next(iter(iterator))
        actions = batch["actions"]
        actions = shift_actions(actions, cfg.dataset.categorical_action_dim)
        input_tensor = batch.get("latents") if use_latent_data else batch.get("videos")
        print(f"Data loaded: shape={input_tensor.shape}")

        dataset_std = tuple(cfg.dataset.dataset_std)[0]

        # ---- Phase 1: Coarse sweep ----
        coarse_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.88, 0.95, 0.99]
        print(f"\n=== Phase 1: Coarse sweep over {coarse_values} ===")

        results = {}  # tau -> {"online": mse, "ema": mse}
        output_dir = Path("eval_outputs/sweep_tau_ctx")
        output_dir.mkdir(parents=True, exist_ok=True)

        for tau in coarse_values:
            t0 = time.time()
            mse_online, online_frames, gt_frames, rng = evaluate_tau(
                bundle.dynamics, bundle.tokenizer, input_tensor, actions,
                use_latent_data, dataset_std, k_max, tau, rng
            )
            mse_ema, ema_frames, _, rng = evaluate_tau(
                bundle.dynamics_ema, bundle.tokenizer, input_tensor, actions,
                use_latent_data, dataset_std, k_max, tau, rng
            )
            mp4_path = save_rollout_video(output_dir, tau, gt_frames, online_frames, ema_frames)
            results[tau] = {"online": mse_online, "ema": mse_ema, "video_path": str(mp4_path)}
            dt = time.time() - t0
            print(
                f"  tau_ctx={tau:.2f} | MSE online={mse_online:.6f} | "
                f"MSE ema={mse_ema:.6f} | {dt:.1f}s | video={mp4_path}"
            )

        # Find best region
        best_tau = min(results, key=lambda t: results[t]["ema"])
        print(f"\nBest coarse tau_ctx={best_tau:.2f} (EMA MSE={results[best_tau]['ema']:.6f})")

        # ---- Phase 2: Fine sweep around best ----
        # Expand search around best value with finer resolution
        low = max(0.1, best_tau - 0.15)
        high = min(0.99, best_tau + 0.15)
        fine_values = np.linspace(low, high, 7).tolist()
        # Remove values already tested (within tolerance)
        fine_values = [v for v in fine_values if not any(abs(v - t) < 0.02 for t in results)]

        if fine_values:
            print(f"\n=== Phase 2: Fine sweep over {[f'{v:.3f}' for v in fine_values]} ===")
            for tau in fine_values:
                t0 = time.time()
                mse_online, online_frames, gt_frames, rng = evaluate_tau(
                    bundle.dynamics, bundle.tokenizer, input_tensor, actions,
                    use_latent_data, dataset_std, k_max, tau, rng
                )
                mse_ema, ema_frames, _, rng = evaluate_tau(
                    bundle.dynamics_ema, bundle.tokenizer, input_tensor, actions,
                    use_latent_data, dataset_std, k_max, tau, rng
                )
                mp4_path = save_rollout_video(output_dir, tau, gt_frames, online_frames, ema_frames)
                results[tau] = {"online": mse_online, "ema": mse_ema, "video_path": str(mp4_path)}
                dt = time.time() - t0
                print(
                    f"  tau_ctx={tau:.3f} | MSE online={mse_online:.6f} | "
                    f"MSE ema={mse_ema:.6f} | {dt:.1f}s | video={mp4_path}"
                )

        # Sort results
        sorted_taus = sorted(results.keys())
        online_mses = [results[t]["online"] for t in sorted_taus]
        ema_mses = [results[t]["ema"] for t in sorted_taus]

        # Save results
        results_json = {str(k): v for k, v in results.items()}
        with open(output_dir / "sweep_results.json", "w") as f:
            json.dump(results_json, f, indent=2)

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_taus, online_mses, 'o-', label='Online Shortcut', markersize=6)
        ax.plot(sorted_taus, ema_mses, 's-', label='EMA Shortcut', markersize=6)
        ax.set_xlabel('tau_ctx_target', fontsize=14)
        ax.set_ylabel('MSE', fontsize=14)
        ax.set_title('MSE vs tau_ctx_target', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Mark the best point
        best_tau_final = min(results, key=lambda t: results[t]["ema"])
        ax.axvline(x=best_tau_final, color='red', linestyle='--', alpha=0.5, label=f'Best tau={best_tau_final:.3f}')
        ax.legend(fontsize=12)

        plt.tight_layout()
        plot_path = output_dir / "mse_vs_tau_ctx.png"
        fig.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")

        # Print summary
        print(f"\n=== Summary ===")
        print(f"{'tau_ctx':>10} | {'MSE online':>12} | {'MSE ema':>12}")
        print("-" * 40)
        for tau in sorted_taus:
            marker = " <-- best" if tau == best_tau_final else ""
            print(f"{tau:10.3f} | {results[tau]['online']:12.6f} | {results[tau]['ema']:12.6f}{marker}")


if __name__ == "__main__":
    main()
