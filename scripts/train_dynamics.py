from __future__ import annotations

import logging
from dataclasses import asdict
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

from dreamer.configs import DynamicsConfig
from dreamer.data import make_iterator
from dreamer.logging import MetricLogger
from dreamer.models import Dynamics, Tokenizer
from dreamer.parallel import create_data_model_parallel, MeshRules
from dreamer.training import run_evaluation, shortcut_forcing_step
from dreamer.utils import (
    _ensure_dir,
    from_dict,
    make_manager,
    get_lr_schedule,
    count_parameters_by_component,
    maybe_save,
    try_restore,
)

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)


# ---------------------------
# Training Step
# ---------------------------

# ---------------------------
# Training Step
# ---------------------------



@nnx.jit(static_argnames=("packing_factor", "k_max", "B_self"))
def encode_and_train_step(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    videos: jnp.ndarray,
    actions: jnp.ndarray,
    *,
    tokenizer_key: jax.Array,
    master_key: jax.Array,
    step: int,
    packing_factor:int,
    B_self:int,
    k_max:int,

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
        tokenizer = Tokenizer.from_pretrained(cfg.tokenizer_ckpt, mesh_rules=mesh_rules, mesh=mesh)
        tokenizer_cfg = tokenizer.config

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

        # Create state factory for abstract restoration
        dynamics_factory = lambda: Dynamics(cfg.dynamics, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        optimizer_factory = lambda: nnx.Optimizer(dynamics_factory(), tx, wrt=nnx.Param)

        # Logging & checkpointing
        logger = MetricLogger(
            use_wandb=cfg.use_wandb, 
            log_every=cfg.log_every, 
            max_steps=cfg.max_steps, 
            wandb_obj=wandb
        )

        meta = {"cfg": asdict(cfg)}
        
        with make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every) as mngr:
            dynamics, optimizer, rng, start_step = try_restore(mngr, dynamics_factory, optimizer_factory, mesh, rng)

            dataset = make_iterator(cfg.dataset)

            # cfg.max_steps + 1 to make sure we log and checkpoint at max_steps
            pbar = tqdm(enumerate(dataset, start=start_step), total=cfg.max_steps + 1)
            for step, batch in pbar:
                if step > cfg.max_steps:
                    break
                
                # Shard batch data
                rng, tokenizer_key, master_key = jax.random.split(rng, num=3)
                videos = jax.device_put(batch["videos"], data_sharding)
                actions = jax.device_put(batch["actions"], data_sharding)

                # Training step
                aux = encode_and_train_step(
                    tokenizer, dynamics, optimizer,
                    videos, actions,
                    tokenizer_key=tokenizer_key,
                    master_key=master_key,
                    step=step,
                    packing_factor=cfg.dynamics.packing_factor,
                    B_self=cfg.dataset.B//2,
                    k_max=cfg.dynamics.k_max,
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

                # Save sharded arrays directly (Orbax handles distributed write efficiently)
                maybe_save(mngr, step, dynamics, optimizer, rng, meta)

                # Periodic lightweight AR eval
                if cfg.write_video_every and (step % cfg.write_video_every == 0) and step > 0:
                    # Use subset of batch for visualization
                    val_videos = batch["videos"][:4]
                    val_actions = batch["actions"][:4]

                    run_evaluation(
                        cfg, tokenizer_cfg, step, tokenizer, dynamics,
                        val_videos, jnp.asarray(val_actions), vis_dir, rng
                    )
            
            mngr.wait_until_finished()

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
