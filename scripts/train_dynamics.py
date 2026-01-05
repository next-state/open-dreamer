from __future__ import annotations

import logging
from dataclasses import asdict
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import orbax.checkpoint as ocp
import grain

from dreamer.configs import DynamicsConfig
from dreamer.data import make_iterator
from dreamer.logging import MetricLogger
from dreamer.models import Dynamics, Tokenizer
from dreamer.parallel import create_data_model_parallel, MeshRules
from dreamer.training import run_evaluation, shortcut_forcing_step
from dreamer.utils import (
    _ensure_dir,
    build_checkpoint_manager,
    try_restore,
    get_lr_schedule,
    count_parameters_by_component,
)

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)


# ---------------------------
# Training Step
# ---------------------------


def make_encode_and_train_step(k_max: int, B_self: int, packing_factor: int):
    """
    Factory function to create a JIT-compiled encode_and_train_step.

    This avoids passing unhashable model objects as static arguments to jax.jit.
    """
    @partial(jax.jit, static_argnames=())
    def encode_and_train_step(
        tokenizer: Tokenizer,
        dynamics: Dynamics,
        optimizer: nnx.Optimizer,
        videos: jnp.ndarray,
        actions: jnp.ndarray,
        *,
        tokenizer_key: jax.Array,
        master_key: jax.Array,
        step: int
    ):
        # Phase 1: Encode videos to latents
        rngs = nnx.Rngs(mae=tokenizer_key)
        latents, _ = tokenizer.encode(
            videos,
            packing_factor=packing_factor,
            deterministic=True,
            rngs=rngs
        )

        # Phase 2: Training step
        metrics = train_step(
            dynamics, optimizer, latents, actions,
            B_self=B_self, k_max=k_max, master_key=master_key, step=step
        )

        return metrics
    
    return encode_and_train_step


@nnx.jit(static_argnames=("k_max", "B_self"))
def train_step(
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    latents: jnp.ndarray,
    actions: jnp.ndarray,
    *,
    B_self: int,
    k_max: int,
    master_key: jax.Array,
    step: int
):
    # Generate step-specific key
    step_key = jax.random.fold_in(master_key, step)

    def loss_and_aux(dynamics_model: Dynamics):
        """Loss function that takes the model and returns (loss, aux)."""
        losses, aux = shortcut_forcing_step(
            dynamics_model=dynamics_model,
            actions=actions,
            latents=latents,
            rng=step_key,
            k_max=k_max,
            B_self=B_self,
            agent_tokens=None,  # Not used in dynamics pretraining
        )
        return losses['total'], aux

    (loss_val, metrics), grads = nnx.value_and_grad(loss_and_aux, has_aux=True)(dynamics)

    # Update model with optimizer 
    optimizer.update(dynamics, grads)

    return metrics

# ---------------------------
# Main
# ---------------------------

def run(cfg: DynamicsConfig):
    """Main training loop."""
    # Setup directories
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
    vis_dir = _ensure_dir(run_dir / "viz")
    print(f"[setup] output dir: {run_dir.resolve()}")

    # Wandb
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity, 
            project=cfg.wandb_project or cfg.run_name, 
            name=cfg.run_name, 
            config=asdict(cfg), 
            dir=str(run_dir)
        )

    # Parallelism
    devices = jax.devices()
    device_count = len(devices)
    mesh, data_sharding = create_data_model_parallel(device_count, 1)

    mesh_rules = MeshRules(
        embed=None,
        mlp='model',
        attn='model',
        data='data',
        )

    with jax.set_mesh(mesh):
        key = jax.random.PRNGKey(0)
        rng, init_key = jax.random.split(key)
    
        # Load pretrained tokenizer
        tokenizer = Tokenizer.from_pretrained(cfg.tokenizer_ckpt, mesh_rules=mesh_rules)
        tokenizer_cfg = tokenizer.config

        # Initialize dynamics
        dynamics = Dynamics(cfg.dynamics, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))

        param_counts = count_parameters_by_component(dynamics)
        print(f"Parameter counts: {param_counts.get('transformer', 0)/1e6:.2f}M")

        # Optimizer
        if cfg.lr_schedule == "constant":
            lr = cfg.lr
            lr_schedule = None
        else:
            lr_schedule = get_lr_schedule(
                cfg.lr_schedule,
                cfg.init_lr,
                cfg.max_lr,
                cfg.lr_end,
                cfg.max_steps,
                cfg.warmup_steps,
                cfg.wsd_decay_steps,
            )
            lr = lr_schedule
        
        tx = optax.adamw(lr, b1=0.9, b2=0.9, weight_decay=1e-4)
        optimizer = nnx.Optimizer(dynamics, tx, wrt=nnx.Param)

        # Create JIT-compiled training step with static args bound
        encode_and_train_step = make_encode_and_train_step(
            k_max=cfg.dynamics.k_max,
            B_self=0,  # Will be set dynamically based on step
            packing_factor=cfg.dynamics.packing_factor
        )

        # Logging & checkpointing
        logger = MetricLogger(
            use_wandb=cfg.use_wandb, 
            log_every=cfg.log_every, 
            max_steps=cfg.max_steps, 
            wandb_obj=wandb
        )

        meta = {"cfg": asdict(cfg)}

        # Create data iterator
        train_dataloader = make_iterator(tokenizer_cfg.dataset)  # FIXME: in a future update, we must decouple dataset from model
        train_iterator = iter(train_dataloader)  # type: ignore

        with build_checkpoint_manager(ckpt_dir, cfg.ckpt_save_every, cfg.ckpt_max_to_keep, cfg.max_steps) as checkpoint_manager:
            start_step, dynamics, optimizer, train_iterator, rng = try_restore(
                checkpoint_manager, dynamics, optimizer, train_iterator, rng
            )

            # Training loop
            pbar = tqdm(enumerate(train_iterator, start=start_step), initial=start_step, total=cfg.max_steps)
            for step, batch in pbar:
                if step >= cfg.max_steps:
                    break
                
                # Shard batch data
                rng, tokenizer_key, master_key = jax.random.split(rng, num=3)
                videos = jax.device_put(batch["videos"], data_sharding)
                actions = jax.device_put(batch["actions"], data_sharding)

                # Compute B_self based on step (bootstrap activates after bootstrap_start)
                B_self = (videos.shape[0] // 2) * int(step >= cfg.bootstrap_start)

                # Recreate the step function with updated B_self
                encode_and_train_step = make_encode_and_train_step(
                    k_max=cfg.dynamics.k_max,
                    B_self=B_self,
                    packing_factor=cfg.dynamics.packing_factor
                )

                # Combined encoding + training step (both phases parallelized)
                aux = encode_and_train_step(
                    tokenizer, dynamics, optimizer,
                    videos, actions,
                    tokenizer_key=tokenizer_key,
                    master_key=master_key,
                    step=step
                )

                # Logging
                if logger.should_log(step):
                    metrics_cpu = jax.device_get(aux)
                    logger.log(
                        step,
                        metrics={
                            "flow_mse": metrics_cpu["flow_mse"],
                            "boot_mse": metrics_cpu["bootstrap_mse"],
                            "lr": cfg.lr if lr_schedule is None else lr_schedule(step),
                        },
                        pbar=pbar,
                    )

                # Save checkpoint
                if checkpoint_manager.should_save(step):
                    model_state = nnx.state(dynamics)
                    optimizer_state = nnx.state(optimizer)

                    checkpoint_manager_args = ocp.args.Composite(
                        model_state=ocp.args.StandardSave(model_state),  # type: ignore
                        optimizer_state=ocp.args.StandardSave(optimizer_state),  # type: ignore
                        train_dataloader_state=grain.checkpoint.CheckpointSave(train_iterator),  # type: ignore
                        rng=ocp.args.StandardSave({'key': rng}),  # type: ignore
                        meta=ocp.args.JsonSave(meta),  # type: ignore
                    )
                    checkpoint_manager.save(step, args=checkpoint_manager_args)

                # Periodic lightweight AR eval
                if cfg.write_video_every and (step % cfg.write_video_every == 0) and step > 0:
                    # Use subset of batch for visualization
                    val_videos = batch["videos"][:4]
                    val_actions = batch["actions"][:4]

                    run_evaluation(
                        cfg, tokenizer_cfg, step, tokenizer, dynamics,
                        val_videos, jnp.asarray(val_actions), vis_dir, rng
                    )

    # Finish wandb run
    if cfg.use_wandb and wandb.run is not None:
        wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="dynamics")
def main(cfg: DictConfig):
    schema = OmegaConf.structured(DynamicsConfig)
    cfg = OmegaConf.merge(schema, cfg)
    realism_cfg = OmegaConf.to_object(cfg)
    run(realism_cfg)

if __name__ == "__main__":
    main()
