import logging
import types
from pathlib import Path

import hydra
import jax
from omegaconf import OmegaConf

from dreamer.configs import LoggerConfig
from dreamer.data import make_iterator
from dreamer.logging import build_logger
from dreamer.parallel import build_parallel
from dreamer.training import run_evaluation, run_evaluation_jit, run_x0_visualization
from dreamer.actions import shift_actions
from dreamer.checkpointing import DynamicsCheckpointBundle

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)

# Register OmegaConf resolvers (same as train_dynamics.py)
OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("max", lambda *args: max(args))

jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


def run(cfg):
    rng = jax.random.PRNGKey(cfg.seed)

    # Parallelism
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with jax.set_mesh(mesh):
        # Load checkpoint (includes both dynamics and tokenizer)
        print(f"Loading checkpoint from: {cfg.dynamics_ckpt}")
        bundle = DynamicsCheckpointBundle.from_pretrained(
            cfg.dynamics_ckpt, mesh_rules=mesh_rules, model_names={"dynamics", "dynamics_ema", "tokenizer"}
        )
        print(f"Loaded dynamics models (k_max={bundle.dynamics_ema.cfg.k_max}, depth={bundle.dynamics_ema.cfg.depth})")

        use_latent_data = cfg.dataset.data_type == "latent"

        # Load one batch of data
        # cfg.dataset.dataloader_cfg.short_T = 128
        # cfg.dataset.dataloader_cfg.long_T = 128
        print(f"Loading data from: {cfg.dataset.array_record_path}")
        iterator = make_iterator(cfg.dataset, device=data_sharding)
        batch = next(iter(iterator))

        actions = batch["actions"]
        actions = shift_actions(actions, cfg.dataset.categorical_action_dim)
        input_tensor = batch.get("latents") if use_latent_data else batch.get("videos")
        print(f"Batch loaded: shape={input_tensor.shape}, use_latent_data={use_latent_data}")

        # run_evaluation needs cfg.dataset.dataset_std
        # run_x0_visualization needs cfg.dynamics.k_max and cfg.dynamics.context_length
        # Both are available directly: dataset_std from cfg.dataset, dynamics fields from the loaded model
        eval_cfg = types.SimpleNamespace(
            dynamics=bundle.dynamics_ema.cfg,
            dataset=types.SimpleNamespace(dataset_std=tuple(cfg.dataset.dataset_std)),
        )

        # Output directory and logger
        output_dir = Path(cfg.output_dir)
        vis_dir = output_dir / "visualizations"

        logger_cfg = LoggerConfig(
            run_name=cfg.run_name,
            use_wandb=cfg.use_wandb,
            log_every=1,
        )
        logger = build_logger(logger_cfg, dir=str(output_dir))

        with logger:
            rng, eval_key = jax.random.split(rng)

            print("\n--- non-JIT evaluation ---")
            run_evaluation(
                eval_cfg,  # type: ignore[arg-type]
                step=cfg.step,
                tokenizer=bundle.tokenizer,
                dynamics_online=bundle.dynamics,
                dynamics_ema=bundle.dynamics_ema,
                val_data=input_tensor,
                val_actions=actions,
                use_latent_data=use_latent_data,
                vis_dir=vis_dir,
                rng=eval_key,
                logger=logger,
            )

            # rng, jit_key = jax.random.split(rng)

            # print("\n--- JIT evaluation (first call includes compile time) ---")
            # run_evaluation_jit(
            #     eval_cfg,  # type: ignore[arg-type]
            #     step=cfg.step,
            #     tokenizer=bundle.tokenizer,
            #     dynamics_online=bundle.dynamics_ema,
            #     dynamics_ema=bundle.dynamics_ema,
            #     val_data=input_tensor,
            #     val_actions=actions,
            #     use_latent_data=use_latent_data,
            #     vis_dir=vis_dir,
            #     rng=jit_key,
            #     logger=logger,
            # )

            # rng, jit_key2 = jax.random.split(rng)

            # print("\n--- JIT evaluation (second call, compiled) ---")
            # run_evaluation_jit(
            #     eval_cfg,  # type: ignore[arg-type]
            #     step=cfg.step,
            #     tokenizer=bundle.tokenizer,
            #     dynamics_online=bundle.dynamics_ema,
            #     dynamics_ema=bundle.dynamics_ema,
            #     val_data=input_tensor,
            #     val_actions=actions,
            #     use_latent_data=use_latent_data,
            #     vis_dir=vis_dir,
            #     rng=jit_key2,
            #     logger=logger,
            # )

            rng, x0_key = jax.random.split(rng)

            run_x0_visualization(
                eval_cfg,  # type: ignore[arg-type]
                step=cfg.step,
                tokenizer=bundle.tokenizer,
                dynamics=bundle.dynamics_ema,
                data=input_tensor[:1],
                actions=actions[:1],
                master_key=x0_key,
                use_latent_data=use_latent_data,
                vis_dir=vis_dir,
                logger=logger,
            )

        step_dir = vis_dir / f"step_{cfg.step:06d}"
        print(f"\nVideos saved to: {step_dir.resolve()}")


@hydra.main(version_base=None, config_path="../configs", config_name="eval_dynamics")
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
