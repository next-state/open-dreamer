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
from dreamer.actions import Actions, shift_actions
from dreamer.parallel import build_parallel
from dreamer.scaling import ScalingContext
from dreamer.training import run_evaluation, run_pixel_evaluation, shortcut_forcing_step
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
    frames_to_pixel_latents,
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

@nnx.jit(
    static_argnames=("k_max", "B_img", "T", "context_length", "bootstrap_fraction", "use_latent_data"),
    donate_argnames=("data", "actions"),
)
def train_step(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    data: jnp.ndarray,        # Full batch: videos (B, T, H, W, C) or latents (B, T, n_latents, d_bottleneck)
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
    use_latent_data: bool,    # True if data is already latents, False if data is videos
):
    if use_latent_data:
        latents = data
    else:
        rngs = nnx.Rngs(mae=tokenizer_key)
        latents, _ = tokenizer.encode(data, deterministic=True, rngs=rngs)
        latents = jax.lax.stop_gradient(latents)

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


@nnx.jit(
    static_argnames=("k_max", "B_img", "T", "context_length", "bootstrap_fraction"),
    donate_argnames=("data", "actions"),
)
def pixel_train_step(
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    data: jnp.ndarray,        # (B, T, n_latents, d_bottleneck)
    actions: Actions,         # (B, T, ...)
    *,
    master_key: jax.Array,
    step: int,
    B_img: int,
    T: int,
    k_max: int,
    context_length: int | None,
    bootstrap_fraction: float,
):
    latents = data.astype(dynamics.dtype)

    B = latents.shape[0]
    B_self = int(B * bootstrap_fraction)
    B_emp = B - B_self

    idx = jnp.arange(B)
    B_img_boot = int(B_img * bootstrap_fraction)
    B_img_emp = B_img - B_img_boot
    is_img = (idx < B_img_emp) | ((idx >= B_emp) & (idx < (B_emp + B_img_boot)))

    mask_img = jnp.eye(T, dtype=jnp.bool_)
    mask_vid = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    time_mask = jnp.where(
        is_img[:, None, None, None],
        mask_img[None, None, :, :],
        mask_vid[None, None, :, :]
    )

    step_key = jax.random.fold_in(master_key, step)

    def loss_fn(model: Dynamics, latents, actions, mask, context_length):
        losses, aux = shortcut_forcing_step(
            dynamics_model=model,
            actions=actions,
            latents=latents,
            rng=step_key,
            k_max=k_max,
            B_self=B_self,
            context_length=context_length,
            time_mask=mask,
            task_embeddings=None,
        )
        return losses['total'], aux

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        dynamics,
        latents, actions, time_mask, context_length
    )

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

        # Check data mode
        use_latent_data = cfg.dataset.data_type == "latent"
        use_pixel_data = cfg.dataset.data_type == "pixel"

        # Compute n_latents based on data mode
        if use_pixel_data:
            pixel_h, pixel_w = cfg.dataset.pixel_h, cfg.dataset.pixel_w
            n_values = pixel_h * pixel_w * cfg.dataset.C
            assert n_values % cfg.dynamics.d_bottleneck == 0, (
                f"pixel_h * pixel_w * C = {n_values} must be divisible by d_bottleneck={cfg.dynamics.d_bottleneck}"
            )
            n_latents = n_values // cfg.dynamics.d_bottleneck
            assert n_latents % cfg.dynamics.packing_factor == 0, (
                f"n_latents={n_latents} must be divisible by packing_factor={cfg.dynamics.packing_factor}"
            )
            print(f"Pixel mode: {pixel_h}x{pixel_w} -> {n_latents} pseudo-latents, {n_latents // cfg.dynamics.packing_factor} spatial tokens")
        else:
            # Load pretrained tokenizer
            tokenizer_bundle = TokenizerCheckpointBundle.from_pretrained(cfg.tokenizer_ckpt, mesh_rules=mesh_rules)
            tokenizer = tokenizer_bundle.tokenizer
            tokenizer_cfg = tokenizer.cfg
            n_latents = tokenizer_cfg.encoder.n_latents

        # Initialize dynamics
        dynamics = Dynamics(cfg.dynamics, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        param_counts = count_parameters_by_component(dynamics)
        print(f"Parameter counts: {param_counts['total']:,}")

        # Scaling context (handles iso-FLOPs/tokens-per-param modes + CSV output)
        n_spatial = n_latents // cfg.dynamics.packing_factor
        avg_T = int(cfg.long_batch_ratio * cfg.long_T + (1 - cfg.long_batch_ratio) * cfg.short_T)

        # Dynamics FLOPs: 1 pass on full batch + 2 passes on bootstrap subset
        dynamics_flops = dynamics.estimate_flops(batch_size=cfg.dataset.B, seq_length=avg_T, n_latents=n_latents)
        bootstrap_multiplier = 1 + 2 * cfg.bootstrap_fraction
        total_dynamics_flops = dynamics_flops * bootstrap_multiplier

        # Encoder FLOPs: forward-only (no gradients) when using video data
        encoder_flops = 0
        if not use_latent_data and not use_pixel_data:
            tokenizer_training_flops = tokenizer.estimate_flops(batch_size=cfg.dataset.B, seq_length=avg_T)
            encoder_flops = tokenizer_training_flops // 12  # ~1/12 of tokenizer training FLOPs (half for encoder, 1/6 for inference)

        scaling = ScalingContext.create(
            cfg=cfg,
            param_count=param_counts["total"],
            flops_per_step=total_dynamics_flops + encoder_flops,
            data_tokens_per_step=cfg.dataset.B * avg_T * (n_spatial + 1),  # spatial + action
            total_tokens_per_step=cfg.dataset.B * avg_T * (3 + n_spatial + cfg.dynamics.n_register),  # action + signal + step + spatial + register
            logger=logger,
            run_dir=run_dir,
        )

        # Build learning rate schedule
        lr_schedule = build_lr_schedule(cfg.lr_schedule)

        # Build optimizer
        optimizer = build_optimizer(cfg.optimizer, dynamics, lr_schedule, d_model=cfg.dynamics.d_model)

        # Data iterators
        short_T = cfg.short_T
        long_T = cfg.long_T
        short_dataloader, long_dataloader = make_dual_iterators(cfg.dataset, short_T=short_T, long_T=long_T)
        short_iterator = iter(short_dataloader)
        long_iterator = iter(long_dataloader)

        if use_pixel_data:
            # Pixel mode: no checkpoint bundle, no checkpointing
            scaling.start_training()

            pbar = tqdm(range(0, cfg.max_steps), initial=0, total=cfg.max_steps)
            for step in pbar:
                rng, dispatch_key, master_key = jax.random.split(rng, num=3)

                use_long = float(jax.random.uniform(dispatch_key)) < cfg.long_batch_ratio
                if use_long:
                    batch = next(long_iterator)
                    T = long_T
                    context_length = cfg.dynamics.context_length
                else:
                    batch = next(short_iterator)
                    T = short_T
                    context_length = None

                actions = jax.device_put(batch["actions"], data_sharding)
                videos = jax.device_put(batch["videos"], data_sharding)
                data = frames_to_pixel_latents(videos, pixel_h, pixel_w, cfg.dynamics.d_bottleneck)

                actions = shift_actions(actions, cfg.dataset.categorical_action_dim)

                # Validation
                if ((step % cfg.write_video_every == 0) and step > 0) or step == cfg.max_steps - 1:
                    run_pixel_evaluation(
                        cfg, step, dynamics,
                        val_data=data[:4], val_actions=actions[:4],
                        pixel_h=pixel_h, pixel_w=pixel_w,
                        vis_dir=vis_dir, rng=rng, logger=logger
                    )

                # Training step
                B_img = int(cfg.dataset.B * cfg.image_fraction)
                metrics = pixel_train_step(
                    dynamics, optimizer,
                    data, actions,
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

            scaling.finalize()

        else:
            # Normal mode: tokenizer + checkpoint bundle
            bundle = DynamicsCheckpointBundle(
                dynamics=dynamics,
                tokenizer=tokenizer,
                dynamics_optimizer=optimizer,
            )

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
                    data = jax.device_put(batch["latents"] if use_latent_data else batch["videos"], data_sharding)

                    # Action shifting: prepend "first action token" (noop) so action[t] aligns with state[t]
                    actions = shift_actions(actions, cfg.dataset.categorical_action_dim)

                    # Validation step before training (as input buffers might be donated)
                    if ((step % cfg.write_video_every == 0) and step > 0) or step == cfg.max_steps - 1:
                        val_data = data[:4]
                        val_actions = actions[:4]
                        run_evaluation(
                            cfg, step, bundle.tokenizer, bundle.dynamics,
                            val_data=val_data, val_actions=val_actions,
                            use_latent_data=use_latent_data,
                            vis_dir=vis_dir, rng=rng, logger=logger
                        )

                    # Training step
                    B_img = int(cfg.dataset.B * cfg.image_fraction)
                    metrics = train_step(
                        bundle.tokenizer, bundle.dynamics, bundle.dynamics_optimizer,
                        data, actions,
                        tokenizer_key=tokenizer_key,
                        master_key=master_key,
                        step=step,
                        B_img=B_img,
                        T=T,
                        k_max=cfg.dynamics.k_max,
                        context_length=context_length,
                        bootstrap_fraction=cfg.bootstrap_fraction if step >= cfg.bootstrap_start else 0.0,
                        use_latent_data=use_latent_data,
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
