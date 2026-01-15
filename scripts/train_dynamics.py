import logging
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.configs import DynamicsConfig
from dreamer.data import make_iterator
from dreamer.logging import build_logger
from dreamer.models import Dynamics, Tokenizer
from dreamer.parallel import build_parallel
from dreamer.training import run_evaluation, shortcut_forcing_step
from dreamer.utils import (
    build_checkpoint_manager,
    count_parameters_by_component,
    maybe_save,
    try_restore,
    setup_training_directories,
    build_lr_schedule,
    build_optimizer,
)
from dreamer.scaling import compute_max_steps, compute_steps_for_flops_budget

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)


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
    # Setup
    run_dir, ckpt_dir, vis_dir, meta = setup_training_directories(cfg)

    # Logging
    logger = build_logger(
        logger_cfg=cfg.logger,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(run_dir),
    )

    # Parallelism
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with (
        logger,
        jax.set_mesh(mesh),
    ):
        key = jax.random.PRNGKey(cfg.seed)
        rng, init_key = jax.random.split(key)

        # Load pretrained tokenizer
        tokenizer = Tokenizer.from_pretrained(cfg.tokenizer_ckpt, mesh_rules=mesh_rules)
        tokenizer_cfg = tokenizer.cfg

        # Initialize dynamics
        dynamics = Dynamics(cfg.dynamics, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        param_counts = count_parameters_by_component(dynamics)
        print(f"Parameter counts: {param_counts.get('transformer', 0):,}")

        # Scaling laws: compute FLOPs and max_steps from param count if enabled
        n_spatial = tokenizer_cfg.encoder.n_latents // cfg.dynamics.packing_factor
        flops_per_step = dynamics.estimate_flops(
            batch_size=cfg.dataset.B,
            seq_length=cfg.dataset.T,
            n_spatial=n_spatial,
        )

        # Tokens per step (for scaling analysis)
        data_tokens_per_step = cfg.dataset.B * cfg.dataset.T * (n_spatial + 1) # spatial + action (both from dataloader)
        total_tokens_per_step = cfg.dataset.B * cfg.dataset.T * (3 + n_spatial + cfg.dynamics.n_register)  # (action + signal + step + spatial + register)q

        if cfg.scaling_flops_budget > 0:
            # Iso-FLOPs mode: fixed compute budget, steps computed from FLOPs
            computed_steps = compute_steps_for_flops_budget(
                total_flops=cfg.scaling_flops_budget,
                flops_per_step=flops_per_step,
            )
            cfg.max_steps = computed_steps
            cfg.lr_schedule.max_steps = computed_steps
            cfg.ckpt.max_steps = computed_steps
            logger.max_steps = computed_steps
            print(f"[IsoFLOPs] {cfg.scaling_flops_budget:.2e} FLOPs / {flops_per_step:.2e} per step = {computed_steps:,} steps")
        elif cfg.scaling_tokens_per_param > 0:
            # Compute-optimal mode: fixed tokens per param ratio
            computed_steps = compute_max_steps(
                param_count=param_counts["total"],
                tokens_per_param=cfg.scaling_tokens_per_param,
                tokens_per_step=total_tokens_per_step,
            )
            cfg.max_steps = computed_steps
            cfg.lr_schedule.max_steps = computed_steps
            cfg.ckpt.max_steps = computed_steps
            logger.max_steps = computed_steps
            total_tokens = param_counts["total"] * cfg.scaling_tokens_per_param
            print(f"[Scaling] {param_counts['total']:,} params × {cfg.scaling_tokens_per_param} = {total_tokens:,.0f} tokens -> {computed_steps:,} steps")

        # Build learning rate schedule
        lr_schedule = build_lr_schedule(cfg.lr_schedule)

        # Build optimizer
        optimizer = build_optimizer(cfg.optimizer, dynamics, lr_schedule, d_model=cfg.dynamics.d_model)

        # Data iterator
        train_dataloader = make_iterator(cfg.dataset)
        train_iterator = iter(train_dataloader)  # type: ignore

        with build_checkpoint_manager(cfg.ckpt, ckpt_dir) as checkpoint_manager:
            # Resume from checkpoint
            start_step, dynamics, optimizer, train_iterator, rng = try_restore(
                checkpoint_manager, dynamics, optimizer, train_iterator, rng
            )

            # Track training time and final metrics for scaling analysis
            train_start_time = time.time()
            final_loss = 0.0

            # Training loop
            pbar = tqdm(enumerate(train_iterator, start=start_step), initial=start_step, total=cfg.max_steps)
            for step, batch in pbar:
                if step >= cfg.max_steps:
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
                    lr_value = lr_schedule(step)
                    flow_mse = float(metrics_cpu["flow_mse"])
                    final_loss = flow_mse  # Track for CSV output
                    logger.log(
                        step,
                        metrics={
                            "flow_mse": flow_mse,
                            "boot_mse": metrics_cpu["bootstrap_mse"],
                            "lr": lr_value,
                            # Cumulative metrics for W&B x-axis flexibility
                            "data_tokens_seen": data_tokens_per_step * step,
                            "total_tokens_seen": total_tokens_per_step * step,
                            "flops_spent": flops_per_step * step,
                        },
                        pbar=pbar,
                    )

                # Checkpointing
                maybe_save(checkpoint_manager, step, dynamics, optimizer, train_iterator, rng, meta)

                # Periodic lightweight AR eval
                if cfg.write_video_every and (step % cfg.write_video_every == 0) and step > 0:
                    # Use subset of batch for visualization
                    val_videos = batch["videos"][:4]
                    val_actions = batch["actions"][:4]

                    run_evaluation(
                        cfg, tokenizer_cfg, step, tokenizer, dynamics,
                        val_videos, jnp.asarray(val_actions), vis_dir, rng, logger
                    )

            # Log final scaling metrics to CSV
            train_elapsed = time.time() - train_start_time
            final_step = min(step, cfg.max_steps - 1)
            data_tokens_trained = data_tokens_per_step * final_step
            total_tokens_trained = total_tokens_per_step * final_step

            # Append one line to parent directory's results.csv (for scaling analysis)
            results_csv = run_dir.parent / "results.csv"
            csv_line = f"{cfg.run_name},{param_counts['total']},{data_tokens_per_step},{total_tokens_per_step},{flops_per_step:.6e},{cfg.scaling_flops_budget or 0},{final_step},{data_tokens_trained},{total_tokens_trained},{train_elapsed/3600:.4f},{final_loss:.6f},\n"
            with open(results_csv, "a") as f:
                f.write(csv_line)


@hydra.main(version_base=None, config_path="../configs", config_name="dynamics")
def main(cfg: DynamicsConfig):
    run(cfg)


if __name__ == "__main__":
    main()
