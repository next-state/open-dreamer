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
    make_state,
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

@nnx.jit
def train_step(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    videos: jnp.ndarray,
    actions: jnp.ndarray,
    videos_self: jnp.ndarray,
    actions_self: jnp.ndarray,
    *,
    tokenizer_key: jax.Array,
    master_key: jax.Array,
):
    """Single training step: encode videos and update dynamics model."""
    # Encode videos to latents
    rngs = nnx.Rngs(mae=tokenizer_key)
    latents, _ = tokenizer.encode(
        videos,
        packing_factor=dynamics.config.packing_factor,
        deterministic=True,
        rngs=rngs
    )
    
    latents_self, _ = tokenizer.encode(
        videos_self,
        packing_factor=dynamics.config.packing_factor,
        deterministic=True,
        rngs=rngs
    )
    
    # Combine empirical and self-consistency batches
    latents_full = jnp.concatenate([latents, latents_self], axis=0)
    actions_full = jnp.concatenate([actions, actions_self], axis=0)
    B_self = videos_self.shape[0]

    # Compute loss and gradients
    def loss_and_aux(dynamics_model: Dynamics):
        losses, aux = shortcut_forcing_step(
            dynamics_model=dynamics_model,
            actions=actions_full,
            latents=latents_full,
            rng=master_key,
            k_max=dynamics.config.k_max,
            B_self=B_self,
            agent_tokens=None,
        )
        return losses['total'], aux

    (loss_val, metrics), grads = nnx.value_and_grad(loss_and_aux, has_aux=True)(dynamics)
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

        # Initialize dynamics
        dynamics_factory = lambda: Dynamics(cfg.dynamics, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        dynamics = dynamics_factory()
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
        optimizer_factory = lambda: nnx.Optimizer(dynamics_factory(), tx, wrt=nnx.Param)
        optimizer = optimizer_factory()
        # Logging & checkpointing
        logger = MetricLogger(
            use_wandb=cfg.use_wandb, 
            log_every=cfg.log_every, 
            max_steps=cfg.max_steps, 
            wandb_obj=wandb
        )

        meta = {"cfg": asdict(cfg)}


        with make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every) as mngr:
            restored = try_restore(mngr, dynamics_factory, optimizer_factory, mesh, rng)
            start_step = 0
            if restored is not None:
                latest_step, r = restored
                nnx.update(dynamics, r.model_state)
                nnx.update(optimizer, r.optimizer_state)
                rng = r.rng_state
                start_step = int(latest_step)
                cfg = from_dict(DynamicsConfig, r.meta["cfg"])
                print(f"[ckpt] Restored step {latest_step} (loaded directly to GPU)")

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

                # Split batch into empirical and bootstrap parts
                if step >= cfg.bootstrap_start:
                    split_idx = videos.shape[0] // 2
                    videos_emp = videos[:split_idx]
                    actions_emp = actions[:split_idx]
                    videos_self = videos[split_idx:]
                    actions_self = actions[split_idx:]
                else:
                    # No bootstrap: use full batch as empirical, empty bootstrap
                    videos_emp = videos
                    actions_emp = actions
                    videos_self = videos[:0]  # Empty slice with correct shape
                    actions_self = actions[:0]

                # Training step
                aux = train_step(
                    tokenizer, dynamics, optimizer,
                    videos_emp, actions_emp,
                    videos_self, actions_self,
                    tokenizer_key=tokenizer_key,
                    master_key=master_key,
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
