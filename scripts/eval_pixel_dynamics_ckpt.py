#!/usr/bin/env python3
"""Evaluate pixel-space dynamics checkpoint and write shortcut/diffusion videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf

from dreamer.checkpointing import DynamicsOnlyCheckpointBundle, build_checkpoint_manager
from dreamer.models import Dynamics
from dreamer.data import make_iterator
from dreamer.logging import LoggerConfig, build_logger
from dreamer.parallel import build_parallel
from dreamer.training import run_evaluation
from dreamer.utils import RunningNormalizer, build_lr_schedule, build_optimizer
from scripts.train_dynamics import _configure_pixel_neural_field_dynamics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dynamics checkpoint with current inference code.")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Checkpoint directory (e.g. logs/run/checkpoints)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation videos")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-name", type=str, default="dynamics", help="Hydra config name under configs/")
    parser.add_argument("--denoise-update-scale", type=float, default=1.0)
    parser.add_argument("--denoise-max-residual-rms", type=float, default=0.0)
    parser.add_argument("--denoise-state-clip", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.register_new_resolver("mul", lambda *vals: __import__("functools").reduce(__import__("operator").mul, vals), replace=True)
    OmegaConf.register_new_resolver("sum", lambda *vals: sum(vals), replace=True)
    OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y, replace=True)
    OmegaConf.register_new_resolver("max", lambda *vals: max(vals), replace=True)
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name=args.config_name)
    OmegaConf.set_struct(cfg, False)

    cfg.dataset.dataloader_cfg.B = args.batch_size
    cfg.dataset.dataloader_cfg.T = args.seq_len
    cfg.use_wandb = False
    cfg.dynamics.denoise_update_scale = args.denoise_update_scale
    cfg.dynamics.denoise_max_residual_rms = args.denoise_max_residual_rms
    cfg.dynamics.denoise_state_clip = args.denoise_state_clip
    _configure_pixel_neural_field_dynamics(cfg)

    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)
    with jax.set_mesh(mesh):
        rng = jax.random.PRNGKey(args.seed)
        init_key, rng = jax.random.split(rng)
        dynamics = Dynamics(cfg.dynamics, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))

        pixel_normalizer = RunningNormalizer(shape=(cfg.dataset.C,))
        dataset_mean = jnp.asarray(cfg.dataset.dataset_mean, dtype=jnp.float32)
        dataset_std = jnp.asarray(cfg.dataset.dataset_std, dtype=jnp.float32)
        pixel_normalizer.mean.value = dataset_mean
        pixel_normalizer.var.value = jnp.maximum(dataset_std * dataset_std, 1e-6)

        lr_schedule = build_lr_schedule(cfg.lr_schedule)
        optimizer = build_optimizer(cfg.optimizer, dynamics, lr_schedule, d_model=cfg.dynamics.d_model)
        bundle = DynamicsOnlyCheckpointBundle(dynamics=dynamics, dynamics_optimizer=optimizer)

        with build_checkpoint_manager(cfg.ckpt, ckpt_dir, item_names=DynamicsOnlyCheckpointBundle.get_item_names()) as manager:
            step, bundle, rng = bundle.restore(manager, rng)
            print(f"Restored dynamics at step={step - 1}")

        dataloader = make_iterator(cfg.dataset, device=data_sharding)
        batch = next(iter(dataloader))
        videos = batch["videos"][: args.batch_size]
        actions = batch["actions"][: args.batch_size]

        logger = build_logger(LoggerConfig(run_name="eval", use_wandb=False, log_every=1))
        run_evaluation(
            cfg=cfg,
            step=step - 1,
            tokenizer=None,
            dynamics=bundle.dynamics,
            val_data=videos,
            val_actions=actions,
            use_latent_data=False,
            vis_dir=output_dir,
            rng=rng,
            logger=logger,
            pixel_normalizer=pixel_normalizer,
        )


if __name__ == "__main__":
    main()
