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

from dreamer.configs import DynamicsConfig
from dreamer.data import make_iterator
from dreamer.logging import MetricLogger
from dreamer.models import Dynamics, Tokenizer
from dreamer.parallel import create_data_model_parallel, MeshRules
from dreamer.training import run_evaluation, shortcut_forcing_step
from dreamer.utils import (
    _ensure_dir,
    make_manager,
    make_state,
    maybe_save,
    try_restore,
    get_lr_schedule,
    count_parameters_by_component,
)

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)


# ---------------------------
# Training Step
# ---------------------------


def make_short_batch_train_fn(k_max: int, packing_factor: int):
    """Factory for short batch training step (JIT compiled separately)."""
    @partial(jax.jit, static_argnames=())
    def short_batch_step(
        tokenizer: Tokenizer,
        dynamics: Dynamics,
        optimizer: nnx.Optimizer,
        videos: jnp.ndarray,
        actions: jnp.ndarray,
        *,
        tokenizer_key: jax.Array,
        master_key: jax.Array,
        step: int,
        B_self: int
    ):
        # Encode
        rngs = nnx.Rngs(mae=tokenizer_key)
        latents, _ = tokenizer.encode(
            videos, packing_factor=packing_factor,
            deterministic=True, rngs=rngs
        )

        # Train
        metrics = train_step(
            dynamics, optimizer, latents, actions,
            B_self=B_self, k_max=k_max,
            master_key=master_key, step=step
        )
        return metrics

    return short_batch_step


def make_long_batch_train_fn(k_max: int, packing_factor: int):
    """Factory for long batch training step (JIT compiled separately)."""
    @partial(jax.jit, static_argnames=())
    def long_batch_step(
        tokenizer: Tokenizer,
        dynamics: Dynamics,
        optimizer: nnx.Optimizer,
        videos: jnp.ndarray,
        actions: jnp.ndarray,
        *,
        tokenizer_key: jax.Array,
        master_key: jax.Array,
        step: int,
        B_self: int
    ):
        # Same implementation (separate JIT boundary)
        rngs = nnx.Rngs(mae=tokenizer_key)
        latents, _ = tokenizer.encode(
            videos, packing_factor=packing_factor,
            deterministic=True, rngs=rngs
        )

        metrics = train_step(
            dynamics, optimizer, latents, actions,
            B_self=B_self, k_max=k_max,
            master_key=master_key, step=step
        )
        return metrics

    return long_batch_step


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
    # Validate alternating batch length configuration
    if cfg.seq_len_long <= cfg.context_length:
        raise ValueError(
            f"seq_len_long ({cfg.seq_len_long}) must be > context_length ({cfg.context_length}) "
            "to prevent overfitting to start frames (paper Section 3.4)"
        )
    if not 0.0 <= cfg.long_batch_ratio <= 1.0:
        raise ValueError(f"long_batch_ratio must be in [0, 1], got {cfg.long_batch_ratio}")
    if cfg.seq_len_short >= cfg.seq_len_long:
        raise ValueError(f"seq_len_short ({cfg.seq_len_short}) must be < seq_len_long ({cfg.seq_len_long})")
    if cfg.finetune_start_step > cfg.max_steps:
        print(f"[warning] finetune_start_step ({cfg.finetune_start_step}) > max_steps ({cfg.max_steps})")

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

        # Initialize dynamics with context_length for sliding window attention
        dynamics = Dynamics(cfg.dynamics, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key), context_length=cfg.context_length)

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

        # Create both short and long batch iterators
        print(f"[setup] Creating short batch iterator (T={cfg.seq_len_short})")
        dataset_short = make_iterator(tokenizer_cfg.dataset, seq_len=cfg.seq_len_short)
        print(f"[setup] Creating long batch iterator (T={cfg.seq_len_long})")
        dataset_long = make_iterator(tokenizer_cfg.dataset, seq_len=cfg.seq_len_long)

        # Create both training functions (JIT compiled separately)
        print(f"[setup] Creating training functions with context_length={cfg.context_length}")
        short_batch_train_fn = make_short_batch_train_fn(
            k_max=cfg.dynamics.k_max,
            packing_factor=cfg.dynamics.packing_factor
        )
        long_batch_train_fn = make_long_batch_train_fn(
            k_max=cfg.dynamics.k_max,
            packing_factor=cfg.dynamics.packing_factor
        )

        # Logging & checkpointing
        logger = MetricLogger(
            use_wandb=cfg.use_wandb, 
            log_every=cfg.log_every, 
            max_steps=cfg.max_steps, 
            wandb_obj=wandb
        )

        opt_graphdef, opt_state = nnx.split(optimizer.opt_state)
        meta = {"cfg": asdict(cfg)}

        # Create state factory for abstract restoration
        def state_factory():
            return make_state(dynamics, opt_state, rng, step=0)

        with make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every) as mngr:
            restored = try_restore(mngr, state_factory, meta)
            start_step = 0
            if restored is not None:
                latest_step, r = restored
                nnx.update(dynamics, r.state["params"])
                nnx.update(optimizer.opt_state, r.state["opt_state"])
                rng = r.state["rng"]
                start_step = int(r.state["step"])
                # Preserve runtime flags before restoring checkpoint config
                use_wandb_override = cfg.use_wandb
                cfg.use_wandb = use_wandb_override  # Keep CLI/YAML wandb setting
                print(f"[ckpt] Restored step {latest_step} (loaded directly to GPU)")

            # Interleave short and long batch iterators
            short_iter = iter(dataset_short)
            long_iter = iter(dataset_long)

            # cfg.max_steps + 1 to make sure we log and checkpoint at max_steps
            pbar = tqdm(range(start_step, cfg.max_steps + 1), total=cfg.max_steps + 1 - start_step)
            for step in pbar:
                if step > cfg.max_steps:
                    break

                # Decide batch type based on training phase
                if step >= cfg.finetune_start_step:
                    # Finetuning phase: 100% long batches
                    use_long_batch = True
                else:
                    # Alternating phase: probabilistic sampling
                    rng, decision_key = jax.random.split(rng)
                    use_long_batch = (jax.random.uniform(decision_key) < cfg.long_batch_ratio)

                # Fetch appropriate batch and training function
                if use_long_batch:
                    batch = next(long_iter)
                    train_fn = long_batch_train_fn
                    batch_type = "long"
                else:
                    batch = next(short_iter)
                    train_fn = short_batch_train_fn
                    batch_type = "short"

                # Shard batch data
                rng, tokenizer_key, master_key = jax.random.split(rng, num=3)
                videos = jax.device_put(batch["videos"], data_sharding)
                actions = jax.device_put(batch["actions"], data_sharding)

                # Compute B_self (bootstrap examples)
                B_self = (videos.shape[0] // 2) * int(step >= cfg.bootstrap_start)

                # Training step
                aux = train_fn(
                    tokenizer, dynamics, optimizer,
                    videos, actions,
                    tokenizer_key=tokenizer_key,
                    master_key=master_key,
                    step=step,
                    B_self=B_self
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
                            "batch_type": batch_type,
                            "seq_len": cfg.seq_len_long if use_long_batch else cfg.seq_len_short,
                            "is_finetuning": int(step >= cfg.finetune_start_step),
                        },
                        pbar=pbar,
                    )

                # Save sharded arrays directly (Orbax handles distributed write efficiently)
                opt_graphdef, opt_state = nnx.split(optimizer.opt_state)
                ckpt_state = make_state(dynamics, opt_state, rng, step)
                maybe_save(mngr, step, ckpt_state, meta)

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
