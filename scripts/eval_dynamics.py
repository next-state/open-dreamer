import dataclasses
import logging
import types
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf

from dreamer.configs import DatasetConfig, LoggerConfig
from dreamer.data import make_iterator
from dreamer.logging import build_logger
from dreamer.parallel import build_parallel
from dreamer.training import run_evaluation, run_x0_visualization
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.utils import from_dict

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
        bundle = DynamicsCheckpointBundle.from_pretrained(cfg.dynamics_ckpt, mesh_rules=mesh_rules)
        print(f"Loaded dynamics model (k_max={bundle.dynamics.cfg.k_max}, depth={bundle.dynamics.cfg.depth})")

        # Build typed DatasetConfig from Hydra DictConfig, then override batch size
        dataset_cfg = from_dict(DatasetConfig, OmegaConf.to_container(cfg.dataset, resolve=True))
        dataset_cfg.dataloader_cfg = dataclasses.replace(dataset_cfg.dataloader_cfg, B=cfg.B)

        use_latent_data = dataset_cfg.data_type == "latent"

        # Load one batch of data
        print(f"Loading data from: {dataset_cfg.array_record_path}")
        iterator = make_iterator(dataset_cfg, device=data_sharding)
        batch = next(iter(iterator))

        actions = batch["actions"]
        input_tensor = batch.get("latents") if use_latent_data else batch.get("videos")
        T = input_tensor.shape[1]
        print(f"Batch loaded: shape={input_tensor.shape}, use_latent_data={use_latent_data}")

        # Build a minimal cfg namespace that satisfies run_evaluation and run_x0_visualization:
        #   run_evaluation uses: eval_cfg.dataset.dataset_std
        #   run_x0_visualization uses: eval_cfg.dynamics.k_max, eval_cfg.dynamics.context_length
        eval_cfg = types.SimpleNamespace(
            dynamics=bundle.dynamics.cfg,
            dataset=types.SimpleNamespace(dataset_std=dataset_cfg.dataset_std),
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

            run_evaluation(
                eval_cfg,
                step=cfg.step,
                tokenizer=bundle.tokenizer,
                dynamics=bundle.dynamics,
                val_data=input_tensor[:cfg.B],
                val_actions=actions[:cfg.B],
                use_latent_data=use_latent_data,
                vis_dir=vis_dir,
                rng=eval_key,
                logger=logger,
            )

            rng, x0_key = jax.random.split(rng)

            run_x0_visualization(
                eval_cfg,
                step=cfg.step,
                tokenizer=bundle.tokenizer,
                dynamics=bundle.dynamics,
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
