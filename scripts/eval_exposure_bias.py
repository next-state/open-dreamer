"""Exposure-bias rollout evaluation (§6.6 of the perturbation-matching spec).

Loads a dynamics checkpoint, rolls out `horizon` frames autoregressively from a held-out
prompt, and writes the per-frame error-vs-horizon curve (normalised latent MSE + decoded
PSNR) as CSV + a plot. Run it on the baseline (qphi.type=none) and on the learned variants
(gaussian_lowrank / flow) and overlay the CSVs to compare error growth; success is *lower*
error growth than vanilla diffusion forcing. Qphi is not loaded here — the metric depends
only on rollout quality.

Example:
    python scripts/eval_exposure_bias.py dynamics_ckpt=logs/<run>/checkpoints \\
        ctx_length=8 horizon=32 num_steps=4 output_dir=eval_outputs/<run>
"""
import logging
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from dreamer.actions import shift_actions
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.data import build_iterator
from dreamer.generation import DenoiseSchedule
from dreamer.parallel import build_parallel
from dreamer.training import rollout_error_curve

logging.getLogger('absl').setLevel(logging.WARNING)

OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("max", lambda *args: max(args))
OmegaConf.register_new_resolver("min", lambda *args: min(args))


def run(cfg):
    rng = jax.random.PRNGKey(cfg.seed)
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with jax.set_mesh(mesh):
        print(f"Loading checkpoint from: {cfg.dynamics_ckpt}")
        bundle = DynamicsCheckpointBundle.from_pretrained(
            cfg.dynamics_ckpt, mesh_rules=mesh_rules, model_names={"dynamics_ema", "tokenizer"}
        )
        dynamics = bundle.dynamics_ema
        tokenizer = bundle.tokenizer
        k_max = dynamics.cfg.k_max
        print(f"Loaded dynamics_ema (k_max={k_max}, depth={dynamics.cfg.depth})")

        use_latent_data = cfg.dataset.data_type == "latent"
        iterator = build_iterator(cfg.dataset, device=data_sharding)

        ctx_length = int(cfg.ctx_length)
        horizon = int(cfg.horizon)
        # tau_ctx_target=1.0 => clean generated context (use for perturbation-matched models);
        # keep 0.9 for the vanilla diffusion-forcing baseline.
        schedule = DenoiseSchedule.init(int(cfg.num_steps), k_max, tau_ctx_target=float(cfg.tau_ctx_target))
        print(f"rollout schedule: num_steps={cfg.num_steps}, tau_ctx_target={cfg.tau_ctx_target}")

        # Accumulate the curve over `num_batches` held-out batches.
        latent_mse = np.zeros(horizon, dtype=np.float64)
        psnr = np.zeros(horizon, dtype=np.float64)
        n = 0
        it = iter(iterator)
        for _ in range(int(cfg.num_batches)):
            batch = next(it)
            actions = shift_actions(batch["actions"], cfg.dataset.categorical_action_dim)
            data = batch.get("latents") if use_latent_data else batch.get("videos")
            if use_latent_data:
                latents = data
            else:
                latents, _ = tokenizer.encode(data, deterministic=True)
            latents = jax.lax.stop_gradient(latents).astype(dynamics.dtype)

            rng, eval_key = jax.random.split(rng)
            curve = rollout_error_curve(
                tokenizer, dynamics, latents=latents, actions=actions,
                ctx_length=ctx_length, horizon=horizon, schedule=schedule, rng=eval_key,
            )
            latent_mse += np.asarray(jax.device_get(curve['latent_mse']), dtype=np.float64)
            psnr += np.asarray(jax.device_get(curve['psnr']), dtype=np.float64)
            n += 1

        latent_mse /= n
        psnr /= n

        # Write CSV + plot.
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = cfg.get("tag", "rollout")
        csv_path = out_dir / f"exposure_bias_{tag}.csv"
        with open(csv_path, "w") as f:
            f.write("frame,latent_mse,psnr\n")
            for t in range(horizon):
                f.write(f"{t},{latent_mse[t]:.6g},{psnr[t]:.6g}\n")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            ax1.plot(range(horizon), latent_mse, marker=".")
            ax1.set_xlabel("rollout frame"); ax1.set_ylabel("normalised latent MSE")
            ax1.set_title(f"exposure-bias latent MSE [{tag}]")
            ax2.plot(range(horizon), psnr, marker=".")
            ax2.set_xlabel("rollout frame"); ax2.set_ylabel("PSNR (dB)")
            ax2.set_title(f"exposure-bias PSNR [{tag}]")
            fig.tight_layout()
            fig.savefig(out_dir / f"exposure_bias_{tag}.png", bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[eval] plot failed: {e}")

        print(f"\nExposure-bias curve ({n} batch(es), horizon={horizon}):")
        print(f"  latent MSE @1/@mid/@last = "
              f"{latent_mse[0]:.4g} / {latent_mse[horizon // 2]:.4g} / {latent_mse[-1]:.4g}")
        print(f"  PSNR(dB)   @1/@mid/@last = "
              f"{psnr[0]:.2f} / {psnr[horizon // 2]:.2f} / {psnr[-1]:.2f}")
        print(f"  saved: {csv_path.resolve()}")


@hydra.main(version_base=None, config_path="../configs", config_name="eval_exposure_bias")
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
