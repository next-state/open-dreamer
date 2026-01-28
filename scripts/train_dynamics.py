import logging

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.configs import DynamicsConfig
from dreamer.data import make_dual_iterators
from dreamer.logging import build_logger
from dreamer.models import Dynamics, Tokenizer
from dreamer.actions import Actions
from dreamer.parallel import build_parallel
from dreamer.scaling import ScalingContext
from dreamer.training import run_evaluation, shortcut_forcing_step
from dreamer.checkpointing import (
    DynamicsCheckpointBundle,
    TokenizerCheckpointBundle,
    build_checkpoint_manager,
)
from dreamer.utils import (
    count_parameters_by_component,
    setup_training_directories,
    build_lr_schedule,
    build_optimizer,
)

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)

# Register OmegaConf resolver for arithmetic expressions
OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("max", lambda *args: max(args))

jax.config.update("jax_compilation_cache_dir", "/scratch/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


# ---------------------------
# Training Step
# ---------------------------

@nnx.jit(static_argnames=("k_max", "B_img", "T", "context_length", "bootstrap_start", "bootstrap_fraction"))
def encode_and_train_step(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    videos: jnp.ndarray,      # Full batch (B, T, H, W, C)
    actions: Actions,         # Full batch (B, T, ...)
    *,
    tokenizer_key: jax.Array,
    master_key: jax.Array,
    step: int,
    B_img: int,               # Number of samples to treat as images
    T: int,
    k_max: int,
    context_length: int | None,  # None = use is_causal, int = sliding window with local_window_size
    bootstrap_fraction: float,
):
    rngs = nnx.Rngs(mae=tokenizer_key)

    # Encode
    latents, _ = tokenizer.encode(videos, deterministic=True, rngs=rngs)

    return latent_train_step(
        dynamics, optimizer, latents, actions,
        master_key=master_key, step=step, B_img=B_img, T=T,
        k_max=k_max, context_length=context_length,
        bootstrap_fraction=bootstrap_fraction,
    )


@nnx.jit(
        static_argnames=("k_max", "B_img", "T", "context_length", "bootstrap_start", "bootstrap_fraction"),
        donate_argnames=("latents", "actions"),
)
def latent_train_step(
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    latents: jnp.ndarray,     # Full batch (B, T, n_latents, d_bottleneck)
    actions: Actions,         # Full batch (B, T, ...)
    *,
    master_key: jax.Array,
    step: int,
    B_img: int,               # Number of samples to treat as images
    T: int,
    k_max: int,
    context_length: int | None,  # None = use is_causal, int = sliding window with local_window_size
    bootstrap_fraction: float,
):
    """Training step for pre-tokenized latent data (skips tokenizer encoding)."""
    latents = latents.astype(dynamics.dtype)

    B = latents.shape[0]
    B_self = int(B * bootstrap_fraction)
    B_emp = B - B_self

    # Identify image samples (split with same bootstrap ratio)
    idx = jnp.arange(B)
    B_img_boot = int(B_img * bootstrap_fraction)
    B_img_emp = B_img - B_img_boot
    is_img = (idx < B_img_emp) | ((idx >= B_emp) & (idx < (B_emp + B_img_boot)))

    # Build time mask for full batch
    mask_img = jnp.eye(T, dtype=jnp.bool_)                  # independent tokens
    mask_vid = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))  # causal tokens
    time_mask = jnp.where(
        is_img[:, None, None, None],
        mask_img[None, None, :, :],
        mask_vid[None, None, :, :]
    )

    # Training step
    step_key = jax.random.fold_in(master_key, step)

    def loss_fn(model: Dynamics, latents, actions, mask, context_length):
        losses, aux = shortcut_forcing_step(
            dynamics_model=model,
            actions=actions,
            latents=latents,
            rng=step_key,
            k_max=k_max,
            B_self=B_self,
            context_length=context_length, # Builds sliding window attention
            time_mask=mask,
            task_embeddings=None,  # Not used in dynamics pretraining
        )

        return losses['total'], aux

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        dynamics, 
        latents, actions, time_mask, context_length
    )

    # Update model with optimizer
    optimizer.update(dynamics, grads)

    return metrics


# ---------------------------
# Main
# ---------------------------

def run(cfg: DynamicsConfig):
    # Setup
    run_dir, ckpt_dir, vis_dir = setup_training_directories(cfg)

    # Logging
    logger = build_logger(
        logger_cfg=cfg.logger,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(run_dir),
    )

    # Parallelism
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with logger, jax.set_mesh(mesh):
        key = jax.random.PRNGKey(cfg.seed)
        rng, init_key = jax.random.split(key)

        # Check if using latent data (pre-tokenized)
        use_latent_data = cfg.dataset.data_type == "latent"

        # Load pretrained tokenizer (required for video data, optional for latent data checkpoints)
        tokenizer_bundle = TokenizerCheckpointBundle.from_pretrained(cfg.tokenizer_ckpt, mesh_rules=mesh_rules)
        tokenizer = tokenizer_bundle.tokenizer
        tokenizer_cfg = tokenizer.cfg

        # Initialize dynamics
        dynamics = Dynamics(cfg.dynamics, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        param_counts = count_parameters_by_component(dynamics)
        print(f"Parameter counts: {param_counts['total']:,}")

        # Scaling context (handles iso-FLOPs/tokens-per-param modes + CSV output)
        n_latents = tokenizer_cfg.encoder.n_latents
        n_spatial = n_latents // cfg.dynamics.packing_factor
        avg_T = int(cfg.long_batch_ratio * cfg.long_T + (1 - cfg.long_batch_ratio) * cfg.short_T)
        scaling = ScalingContext.create(
            cfg=cfg,
            param_count=param_counts["total"],
            flops_per_step=dynamics.estimate_flops(batch_size=cfg.dataset.B, seq_length=avg_T, n_latents=n_latents),
            data_tokens_per_step=cfg.dataset.B * avg_T * (n_spatial + 1),  # spatial + action
            total_tokens_per_step=cfg.dataset.B * avg_T * (3 + n_spatial + cfg.dynamics.n_register),  # action + signal + step + spatial + register
            logger=logger,
            run_dir=run_dir,
        )

        # Build learning rate schedule
        lr_schedule = build_lr_schedule(cfg.lr_schedule)

        # Build optimizer
        optimizer = build_optimizer(cfg.optimizer, dynamics, lr_schedule, d_model=cfg.dynamics.d_model)

        # Create checkpoint bundle (includes frozen tokenizer for self-contained checkpoints)
        bundle = DynamicsCheckpointBundle(
            dynamics=dynamics,
            tokenizer=tokenizer,
            dynamics_optimizer=optimizer,
        )

        # Data iterators
        short_T = cfg.short_T
        long_T = cfg.long_T
        short_dataloader, long_dataloader = make_dual_iterators(cfg.dataset, short_T=short_T, long_T=long_T)
        short_iterator = iter(short_dataloader)
        long_iterator = iter(long_dataloader)

        with build_checkpoint_manager(
            cfg.ckpt, ckpt_dir,
            item_names=DynamicsCheckpointBundle.get_item_names(
                iterator_names=("short_dataloader_state", "long_dataloader_state")
            )
        ) as checkpoint_manager:
            # Resume from checkpoint
            iterators = {"short_dataloader_state": short_iterator, "long_dataloader_state": long_iterator}
            start_step, bundle, iterators, rng = bundle.restore(
                checkpoint_manager, iterators, rng
            )
            short_iterator = iterators["short_dataloader_state"]
            long_iterator = iterators["long_dataloader_state"]

            scaling.start_training()

            # Training loop
            pbar = tqdm(range(start_step, cfg.max_steps), initial=start_step, total=cfg.max_steps)
            for step in pbar:
                if step >= cfg.max_steps:
                    break

                rng, dispatch_key, tokenizer_key, master_key = jax.random.split(rng, num=4)

                use_long = float(jax.random.uniform(dispatch_key)) < cfg.long_batch_ratio
                if use_long:
                    batch = next(long_iterator)
                    T = long_T
                    context_length = cfg.dynamics.context_length
                else:
                    batch = next(short_iterator)
                    T = short_T
                    context_length = None  # Use default causal attention

                # Shard batch data
                actions = jax.device_put(batch["actions"], data_sharding)
                videos = None if use_latent_data else jax.device_put(batch["videos"], data_sharding)
                latents = jax.device_put(batch["latents"], data_sharding) if use_latent_data else None

                # Validation step before training (as input buffers might be donated)
                if ((step % cfg.write_video_every == 0) and step > 0) or step == cfg.max_steps - 1:
                    val_videos = None if use_latent_data else videos[:4]
                    val_latents = latents[:4] if use_latent_data else None
                    val_actions = actions[:4]
                    run_evaluation(
                        cfg, step, bundle.tokenizer, bundle.dynamics,
                        val_videos=val_videos, val_latents=val_latents,
                        val_actions=val_actions, 
                        vis_dir=vis_dir, rng=rng, logger=logger
                    )

                # Training step
                B_img = int(cfg.dataset.B * cfg.image_fraction)

                if use_latent_data:
                    # Pre-tokenized latent data path
                    metrics = latent_train_step(
                        bundle.dynamics, bundle.dynamics_optimizer,
                        latents, actions,
                        master_key=master_key,
                        step=step,
                        B_img=B_img,
                        T=T,
                        k_max=cfg.dynamics.k_max,
                        context_length=context_length,
                        bootstrap_fraction=cfg.bootstrap_fraction if step >= cfg.bootstrap_start else 0.0,
                    )
                else:
                    # Video data path (requires tokenizer encoding)
                    metrics = encode_and_train_step(
                        bundle.tokenizer, bundle.dynamics, bundle.dynamics_optimizer,
                        videos, actions,
                        tokenizer_key=tokenizer_key,
                        master_key=master_key,
                        step=step,
                        B_img=B_img,
                        T=T,
                        k_max=cfg.dynamics.k_max,
                        context_length=context_length,
                        bootstrap_fraction=cfg.bootstrap_fraction if step >= cfg.bootstrap_start else 0.0,
                    )

                # Logging
                if logger.should_log(step):
                    metrics_cpu = jax.device_get(metrics)
                    scaling.on_step(step, metrics_cpu)
                    logger.log(
                        step,
                        metrics={
                            "flow_mse": metrics_cpu["flow_mse"],
                            "boot_mse": metrics_cpu["bootstrap_mse"],
                            "lr": lr_schedule(step),
                            "batch_type": 1.0 if use_long else 0.0,
                            **scaling.get_step_metrics(step),
                        },
                        pbar=pbar,
                    )

                # Checkpointing
                iterators = {"short_dataloader_state": short_iterator, "long_dataloader_state": long_iterator}
                bundle.maybe_save(checkpoint_manager, step, iterators, rng)

            scaling.finalize()


@hydra.main(version_base=None, config_path="../configs", config_name="dynamics")
def main(cfg: DynamicsConfig):
    run(cfg)


if __name__ == "__main__":
    main()
