import logging

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.configs import DynamicsConfig
from dreamer.data import make_iterator  
from dreamer.logging import build_logger
from dreamer.models import Dynamics, Tokenizer
from dreamer.actions import Actions, shift_actions
from dreamer.parallel import build_parallel
from dreamer.scaling import ScalingContext
from dreamer.training import run_evaluation, run_x0_visualization, shortcut_forcing_step
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

# jax.config.update("jax_compilation_cache_dir", "/scratch/jax_cache")
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
        latents, _ = tokenizer.encode(data, deterministic=True)
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

    # Compute gradient norm for training health diagnostics
    grad_norm = optax.global_norm(nnx.state(grads))

    # Update model with optimizer
    optimizer.update(dynamics, grads)

    return {**metrics, 'grad_norm': grad_norm}

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
        B, T = cfg.dataset.dataloader_cfg.B, cfg.dataset.dataloader_cfg.T
        avg_T = int(cfg.long_batch_ratio * cfg.long_T + (1 - cfg.long_batch_ratio) * cfg.short_T)

        # Dynamics FLOPs: 1 pass on full batch + 2 passes on bootstrap subset
        dynamics_flops = dynamics.estimate_flops(batch_size=B, seq_length=avg_T, n_latents=n_latents)
        bootstrap_multiplier = 1 + cfg.bootstrap_fraction * (2/3)  # two additional gradientless calls to dynamics model, 1/6 for inference
        total_dynamics_flops = dynamics_flops * bootstrap_multiplier

        # Encoder FLOPs: forward-only (no gradients) when using video data
        encoder_flops = 0
        if not use_latent_data:
            tokenizer_training_flops = tokenizer.estimate_flops(batch_size=B, seq_length=avg_T)
            encoder_flops = tokenizer_training_flops // 6  # ~1/6 of tokenizer training FLOPs (half for encoder, 1/3 for inference)

        scaling = ScalingContext.create(
            cfg=cfg,
            param_count=param_counts["total"],
            flops_per_step=dynamics_flops + encoder_flops,
            data_tokens_per_step=B * avg_T * (n_spatial + 1),  # spatial + action
            total_tokens_per_step=B * avg_T * (2 + n_spatial + cfg.dynamics.n_register),  # action + shortcut + spatial + register
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

        dataloader = make_iterator(cfg.dataset, device=data_sharding)
        with build_checkpoint_manager(cfg.ckpt, ckpt_dir, item_names=DynamicsCheckpointBundle.get_item_names()) as checkpoint_manager:
            # Resume from checkpoint
            start_step, bundle, rng = bundle.restore(checkpoint_manager, rng)
            scaling.start_training()


            pbar = tqdm(enumerate(dataloader, start_step), initial=start_step, total=cfg.max_steps)
            for step, batch in pbar:
                if step >= cfg.max_steps:
                    break

                rng, tokenizer_key, master_key = jax.random.split(rng, num=3)

                # Use pre-allocated batch
                actions = batch["actions"]
                videos = batch.get("videos")
                latents = batch.get("latents")
                input_tensor = latents if latents is not None else videos
                

                # Validation step before training (as input buffers might be donated)
                if ((step % cfg.write_video_every == 0) and step > 0) or step == cfg.max_steps - 1:
                    val_data = input_tensor[:4]
                    val_actions = actions[:4]
                    run_evaluation(
                        cfg, step, bundle.tokenizer, bundle.dynamics,
                        val_data=val_data, val_actions=val_actions,
                        use_latent_data=use_latent_data,
                        vis_dir=vis_dir, rng=rng, logger=logger
                    )
                    run_x0_visualization(
                        cfg, step, bundle.tokenizer, bundle.dynamics,
                        data=input_tensor[:1],
                        actions=actions[:1],
                        master_key=master_key,
                        use_latent_data=use_latent_data,
                        vis_dir=vis_dir,
                        logger=logger,
                    )

                # Training step
                metrics = train_step(
                    bundle.tokenizer, bundle.dynamics, 
                    bundle.dynamics_optimizer, input_tensor, actions,
                    master_key=master_key,
                    step=step,
                    B_img=int(B * cfg.image_fraction),
                    T=T,
                    k_max=cfg.dynamics.k_max,
                    context_length=cfg.dynamics.context_length,
                    bootstrap_fraction=cfg.bootstrap_fraction if step > cfg.bootstrap_start else 0,
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
                            "grad_norm": metrics_cpu["grad_norm"],
                            "flow_mse_low": metrics_cpu["flow_mse_low"],
                            "flow_mse_mid": metrics_cpu["flow_mse_mid"],
                            "flow_mse_high": metrics_cpu["flow_mse_high"],
                            "boot_target_norm": metrics_cpu["boot_target_norm"],
                            "boot_clip_frac": metrics_cpu["boot_clip_frac"],
                            "lr": lr_schedule(step),
                            **scaling.get_step_metrics(step),
                        },
                        pbar=pbar,
                    )

                # Checkpointing
                bundle.maybe_save(checkpoint_manager, step, rng)

            scaling.finalize()


@hydra.main(version_base=None, config_path="../configs", config_name="dynamics")
def main(cfg: DynamicsConfig):
    run(cfg)


if __name__ == "__main__":
    main()
