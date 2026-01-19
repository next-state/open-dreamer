import logging

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.configs import DynamicsConfig
from dreamer.data import make_iterator
from dreamer.logging import build_logger
from dreamer.models import Dynamics, Tokenizer
from dreamer.parallel import build_parallel
from dreamer.scaling import ScalingContext
from dreamer.training import LossRMSState, run_evaluation, shortcut_forcing_step, update_loss_rms
from dreamer.utils import (
    build_checkpoint_manager,
    count_parameters_by_component,
    maybe_save,
    try_restore,
    setup_training_directories,
    build_lr_schedule,
    build_optimizer,
)

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)


# ---------------------------
# Training Step
# ---------------------------

@nnx.jit(static_argnames=("packing_factor", "k_max", "B_self", "loss_weights"))
def encode_and_train_step(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    rms_state: LossRMSState,
    videos: jnp.ndarray,
    actions: jnp.ndarray,
    *,
    tokenizer_key: jax.Array,
    master_key: jax.Array,
    step: int,
    packing_factor: int,
    B_self: int,
    k_max: int,
    loss_weights: tuple[tuple[str, float], ...],
) -> tuple[LossRMSState, dict]:
    """
    Encode videos and run training step with RMS loss normalization.

    Uses RMS loss normalization (paper Section 3) to balance flow and bootstrap losses.
    """
    # Phase 1: Encode videos to latents
    rngs = nnx.Rngs(mae=tokenizer_key)
    latents, _ = tokenizer.encode(
        videos,
        packing_factor=packing_factor,
        deterministic=True,
        rngs=rngs
    )

    # Phase 2: Training step with RMS normalization
    rms_state, metrics = train_step(
        dynamics, optimizer, rms_state, latents, actions,
        B_self=B_self, k_max=k_max, master_key=master_key, step=step,
        loss_weights=loss_weights,
    )

    return rms_state, metrics


@nnx.jit(static_argnames=("k_max", "B_self", "loss_weights"))
def train_step(
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    rms_state: LossRMSState,
    latents: jnp.ndarray,
    actions: jnp.ndarray,
    *,
    B_self: int,
    k_max: int,
    master_key: jax.Array,
    step: int,
    loss_weights: tuple[tuple[str, float], ...],
) -> tuple[LossRMSState, dict]:
    """
    Dynamics training step with RMS loss normalization.

    Uses RMS loss normalization (paper Section 3) to balance flow and bootstrap losses
    which can have different scales depending on the noise level distribution.
    """
    # Convert loss_weights tuple to dict for easier access
    weights_dict = {name: weight for name, weight in loss_weights}

    # Generate step-specific key
    step_key = jax.random.fold_in(master_key, step)

    # Get current RMS estimates (stop gradient so they don't affect backprop)
    rms_estimates = {
        name: jax.lax.stop_gradient(est)
        for name, est in rms_state.estimates.items()
    }

    def loss_and_aux(dynamics_model: Dynamics):
        """Loss function with RMS normalization."""
        losses, aux = shortcut_forcing_step(
            dynamics_model=dynamics_model,
            actions=actions,
            latents=latents,
            rng=step_key,
            k_max=k_max,
            B_self=B_self,
            agent_tokens=None,  # Not used in dynamics pretraining
        )

        # Raw losses
        flow_loss_raw = losses['flow']
        bootstrap_loss_raw = losses['bootstrap']

        raw_losses = {
            "flow": flow_loss_raw,
            "bootstrap": bootstrap_loss_raw,
        }

        # Normalize each loss by its running RMS estimate (paper Section 3)
        normalized_losses = {}
        for name, loss in raw_losses.items():
            rms_est = rms_estimates.get(name, jnp.array(1.0))
            normalized_losses[name] = loss / (rms_est + 1e-8)

        # Combine normalized losses with fixed weights
        total_loss = jnp.array(0.0)
        for name, norm_loss in normalized_losses.items():
            weight = weights_dict.get(name, 1.0)
            total_loss = total_loss + weight * norm_loss

        aux["raw_losses"] = raw_losses
        aux["flow_loss_norm"] = normalized_losses["flow"]
        aux["bootstrap_loss_norm"] = normalized_losses["bootstrap"]

        return total_loss, aux

    (loss_val, metrics), grads = nnx.value_and_grad(loss_and_aux, has_aux=True)(dynamics)

    # Update model with optimizer
    optimizer.update(dynamics, grads)

    # Update RMS estimates with raw losses (after gradient computation)
    raw_losses = metrics.pop("raw_losses")
    new_rms_state, _ = update_loss_rms(rms_state, raw_losses, decay=0.999, warmup_steps=100)

    # Add RMS estimates to metrics for logging
    metrics["rms/flow"] = new_rms_state.estimates["flow"]
    metrics["rms/bootstrap"] = new_rms_state.estimates["bootstrap"]

    return new_rms_state, metrics

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
        print(f"Parameter counts: {param_counts['total']:,}")

        # Scaling context (handles iso-FLOPs/tokens-per-param modes + CSV output)
        n_spatial = tokenizer_cfg.encoder.n_latents // cfg.dynamics.packing_factor
        scaling = ScalingContext.create(
            cfg=cfg,
            param_count=param_counts["total"],
            flops_per_step=dynamics.estimate_flops(batch_size=cfg.dataset.B, seq_length=cfg.dataset.T, n_spatial=n_spatial),
            data_tokens_per_step=cfg.dataset.B * cfg.dataset.T * (n_spatial + 1),  # spatial + action
            total_tokens_per_step=cfg.dataset.B * cfg.dataset.T * (3 + n_spatial + cfg.dynamics.n_register),  # action + signal + step + spatial + register
            logger=logger,
            run_dir=run_dir,
        )

        # Build learning rate schedule
        lr_schedule = build_lr_schedule(cfg.lr_schedule)

        # Build optimizer
        optimizer = build_optimizer(cfg.optimizer, dynamics, lr_schedule, d_model=cfg.dynamics.d_model)

        # Initialize RMS loss normalization state (paper Section 3)
        # Balances flow and bootstrap losses which can have different scales
        rms_state = LossRMSState.init(("flow", "bootstrap"))

        # Loss weights for combining normalized losses
        # After RMS normalization, both losses have ~unit scale
        loss_weights = (
            ("flow", 1.0),
            ("bootstrap", 1.0),
        )

        # Data iterator
        train_dataloader = make_iterator(cfg.dataset)
        train_iterator = iter(train_dataloader)  # type: ignore

        with build_checkpoint_manager(
            cfg.ckpt, ckpt_dir,
            item_names=("model_state", "optimizer_state", "rms_state", "train_dataloader_state", "rngs", "meta")
        ) as checkpoint_manager:
            # Resume from checkpoint (manual handling for rms_state support)
            import orbax.checkpoint as ocp
            import grain.checkpoint
            step = checkpoint_manager.latest_step()
            if step is not None:
                model_state = nnx.state(dynamics)
                optimizer_state = nnx.state(optimizer)
                rms_state_dict = {"estimates": rms_state.estimates, "counts": rms_state.counts}

                restore_args = ocp.args.Composite(
                    model_state=ocp.args.StandardRestore(model_state),  # type: ignore
                    optimizer_state=ocp.args.StandardRestore(optimizer_state),  # type: ignore
                    rms_state=ocp.args.StandardRestore(rms_state_dict),  # type: ignore
                    train_dataloader_state=grain.checkpoint.CheckpointRestore(train_iterator),  # type: ignore
                    rngs=ocp.args.StandardRestore({"key": rng}),  # type: ignore
                )

                restored = checkpoint_manager.restore(step, args=restore_args)
                nnx.update(dynamics, restored["model_state"])
                nnx.update(optimizer, restored["optimizer_state"])
                rms_restored = restored["rms_state"]
                rms_state = LossRMSState(rms_restored["estimates"], rms_restored["counts"])
                train_iterator = restored["train_dataloader_state"]
                rng = restored["rngs"]["key"]
                start_step = step + 1
                print(f"[ckpt] Restored checkpoint from step {step}")
            else:
                start_step = 0
                print("[ckpt] No checkpoint found, starting from scratch")

            scaling.start_training()

            # Training loop
            pbar = tqdm(enumerate(train_iterator, start=start_step), initial=start_step, total=cfg.max_steps)
            for step, batch in pbar:
                if step >= cfg.max_steps:
                    break

                # Shard batch data
                rng, tokenizer_key, master_key = jax.random.split(rng, num=3)
                videos = jax.device_put(batch["videos"], data_sharding)
                actions = jax.device_put(batch["actions"], data_sharding)

                # Training step with RMS loss normalization
                rms_state, aux = encode_and_train_step(
                    tokenizer, dynamics, optimizer, rms_state,
                    videos, actions,
                    tokenizer_key=tokenizer_key,
                    master_key=master_key,
                    step=step,
                    packing_factor=cfg.dynamics.packing_factor,
                    B_self=cfg.dataset.B//2,
                    k_max=cfg.dynamics.k_max,
                    loss_weights=loss_weights,
                )

                # Logging
                if logger.should_log(step):
                    metrics_cpu = jax.device_get(aux)
                    scaling.on_step(step, metrics_cpu)
                    logger.log(
                        step,
                        metrics={
                            "flow_mse": metrics_cpu["flow_mse"],
                            "boot_mse": metrics_cpu["bootstrap_mse"],
                            "rms/flow": metrics_cpu["rms/flow"],
                            "rms/bootstrap": metrics_cpu["rms/bootstrap"],
                            "lr": lr_schedule(step),
                            **scaling.get_step_metrics(step),
                        },
                        pbar=pbar,
                    )

                # Checkpointing (with rms_state)
                if checkpoint_manager.should_save(step):
                    model_state = nnx.state(dynamics)
                    optimizer_state = nnx.state(optimizer)
                    rms_state_dict = {"estimates": rms_state.estimates, "counts": rms_state.counts}

                    save_args = ocp.args.Composite(
                        model_state=ocp.args.StandardSave(model_state),  # type: ignore
                        optimizer_state=ocp.args.StandardSave(optimizer_state),  # type: ignore
                        rms_state=ocp.args.StandardSave(rms_state_dict),  # type: ignore
                        train_dataloader_state=grain.checkpoint.CheckpointSave(train_iterator),  # type: ignore
                        rngs=ocp.args.StandardSave({'key': rng}),  # type: ignore
                        meta=ocp.args.JsonSave(meta)  # type: ignore
                    )
                    checkpoint_manager.save(step, args=save_args)

                # Periodic lightweight AR eval
                if cfg.write_video_every and (step % cfg.write_video_every == 0) and step > 0:
                    # Use subset of batch for visualization
                    val_videos = batch["videos"][:4]
                    val_actions = batch["actions"][:4]

                    run_evaluation(
                        cfg, tokenizer_cfg, step, tokenizer, dynamics,
                        val_videos, jnp.asarray(val_actions), vis_dir, rng, logger
                    )

            scaling.finalize()


@hydra.main(version_base=None, config_path="../configs", config_name="dynamics")
def main(cfg: DynamicsConfig):
    run(cfg)


if __name__ == "__main__":
    main()
