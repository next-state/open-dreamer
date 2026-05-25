import os


def _append_xla_flag(flag: str) -> None:
    current = os.environ.get("XLA_FLAGS", "")
    if flag not in current.split():
        os.environ["XLA_FLAGS"] = f"{current} {flag}".strip()


os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
_append_xla_flag("--xla_gpu_triton_gemm_any=True")
_append_xla_flag("--xla_gpu_enable_latency_hiding_scheduler=true")
os.environ.setdefault("JAX_ENABLE_PGLE", "true")

import logging

import hydra
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.configs import DynamicsConfig, OptimalTransportConfig
from dreamer.data import build_dual_iterator
from dreamer.logging import build_logger
from dreamer.models import Dynamics, Tokenizer
from dreamer.actions import Actions, shift_actions, NUM_BINARY_ACTIONS, NUM_CAMERA_CLASSES
from dreamer.parallel import build_parallel, MeshRules
from dreamer.scaling import ScalingContext
from dreamer.training import (
    run_evaluation,
    shortcut_forcing_step,
)
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
    build_ema_model,
    ema_update_step,
)

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

# Register OmegaConf resolver for arithmetic expressions
OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("max", lambda *args: max(args))
OmegaConf.register_new_resolver("min", lambda *args: min(args))

# jax.config.update("jax_compilation_cache_dir", "/scratch/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


# ---------------------------
# Training Step
# ---------------------------

@nnx.jit(
    static_argnames=("k_max", "B_img", "T", "n_splits", "context_length", "bootstrap_fraction", "use_latent_data", "ot_cfg"),
    donate_argnames=("data", "actions"),
)
def train_step(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    dynamics_ema: Dynamics,   # Used as target network for bootstrap half-steps
    optimizer: nnx.Optimizer,
    data: jnp.ndarray,        # Full batch: videos (B, T, H, W, C) or latents (B, T, n_latents, d_bottleneck)
    actions: Actions,         # Full batch (B, T, ...)
    *,
    master_key: jax.Array,
    step: int,
    B_img: int,               # Number of samples to treat as images
    T: int,
    n_splits: int,            # Number of block-causal chunks (1 = full causal, >1 = split into independent chunks)
    k_max: int,
    context_length: int | None,  # None = use is_causal, int = sliding window with local_window_size
    bootstrap_fraction: float,
    use_latent_data: bool,    # True if data is already latents, False if data is videos
    ot_cfg: OptimalTransportConfig,
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

    # Identify image samples (split with same bootstrap ratio).
    # Batch layout (see shortcut_forcing_step): B_self blocks of size `stride = B // B_self`,
    # block[:, 0] = bootstrap row, block[:, 1:] = empirical rows (block-major flattened).
    # First B_img_boot bootstrap rows and first B_img_emp empirical rows are image-only.
    B_img_boot = int(B_img * bootstrap_fraction)
    B_img_emp = B_img - B_img_boot
    if B_self > 0:
        stride = B // B_self
        # Mark image rows directly on the (B_self, stride) view, then flatten.
        boot_img = jnp.arange(B_self) < B_img_boot                          # (B_self,)
        emp_img = (jnp.arange(B_self * (stride - 1)) < B_img_emp).reshape(B_self, stride - 1)
        is_img_blocks = jnp.concatenate([boot_img[:, None], emp_img], axis=1)  # (B_self, stride)
        is_img = is_img_blocks.reshape(B)
    else:
        is_img = jnp.arange(B) < B_img_emp

    # Build time mask for full batch
    mask_img = jnp.eye(T, dtype=jnp.bool_)                  # independent tokens
    # Block-causal mask: n_splits independent causal chunks
    chunk_size = T // n_splits
    row_idx = jnp.arange(T)
    col_idx = jnp.arange(T)
    same_chunk = (row_idx[:, None] // chunk_size) == (col_idx[None, :] // chunk_size)
    causal = row_idx[:, None] >= col_idx[None, :]
    mask_vid = (same_chunk & causal)                         # block-causal tokens
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
            B_img_emp=B_img_emp,
            context_length=context_length, # Builds sliding window attention
            time_mask=mask,
            task_embeddings=None,  # Not used in dynamics pretraining
            bootstrap_model=dynamics_ema,  # EMA as stable target network for half-steps
            ot_cfg=ot_cfg,
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

    # Parallelism
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)
    # mesh = jax.make_mesh((cfg.dataset.dataloader_cfg.B, jax.local_device_count()//cfg.dataset.dataloader_cfg.B), ('data', 'seq'))
    # data_sharding = NamedSharding(mesh, P('data', 'seq', None, None))
    # mesh_rules = MeshRules(data='data', seq='seq', mlp='data', attn='data')

    is_main_process = jax.process_index() == 0
    is_multihost = jax.process_count() > 1

    # Logging
    logger = build_logger(logger_cfg=cfg.logger, config=OmegaConf.to_container(cfg, resolve=True), dir=str(run_dir))

    with logger, jax.set_mesh(mesh):
        key = jax.random.PRNGKey(cfg.seed)
        rng, init_key = jax.random.split(key)

        # Build a plain dataclass OT config for JIT static args.
        ot_cfg = OptimalTransportConfig(**cfg.ot)  # ty: ignore[invalid-argument-type]

        # Check if using latent data (pre-tokenized)
        use_latent_data = cfg.dataset.data_type == "latent"
        assert cfg.dataset.num_binary_actions == NUM_BINARY_ACTIONS
        assert cfg.dataset.categorical_action_dim == NUM_CAMERA_CLASSES

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
        dl_cfg = cfg.dataset.dataloader_cfg
        B = dl_cfg.B
        avg_T = dl_cfg.long_T

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

        # Build EMA model
        dynamics_ema = build_ema_model(dynamics, ema_dtype=cfg.ema_dtype)

        # Create checkpoint bundle (includes frozen tokenizer for self-contained checkpoints)
        bundle = DynamicsCheckpointBundle(
            dynamics=dynamics,
            dynamics_ema=dynamics_ema,
            tokenizer=tokenizer,
            dynamics_optimizer=optimizer,
        )

        dataloader = build_dual_iterator(cfg.dataset, device=data_sharding, dtype=cfg.dtype)
        with build_checkpoint_manager(cfg.ckpt, ckpt_dir, item_names=DynamicsCheckpointBundle.get_item_names()) as checkpoint_manager:
            # Resume from checkpoint
            start_step, bundle, rng = bundle.restore(checkpoint_manager, rng)

            scaling.start_training()

            pbar = tqdm(enumerate(dataloader, start_step), initial=start_step, total=cfg.max_steps, dynamic_ncols=True, disable=not is_main_process)
            for step, batch in pbar:
                if step >= cfg.max_steps:
                    break

                rng, master_key = jax.random.split(rng, num=2)

                n_splits = int(batch.get("n_splits", 1))

                # Use pre-allocated batch
                actions = batch["actions"]
                videos = batch.get("videos")
                latents = batch.get("latents")
                input_tensor = latents if latents is not None else videos

                actions = shift_actions(actions, cfg.dataset.categorical_action_dim)

                # Validation/visualization — all hosts must participate in JAX
                # compute (model is sharded), but only process 0 does I/O.
                do_eval = (cfg.write_video_every>0 and step>0 and (step % cfg.write_video_every == 0)) or step == cfg.max_steps - 1
                if do_eval:
                    val_data = input_tensor[:4]
                    val_actions = actions[:4]
                    run_evaluation(
                        cfg, step, bundle.tokenizer,
                        dynamics_online=bundle.dynamics,
                        dynamics_ema=bundle.dynamics_ema,
                        val_data=val_data,
                        val_actions=val_actions,
                        use_latent_data=use_latent_data,
                        vis_dir=vis_dir, rng=rng,
                        logger=logger if is_main_process else None,
                    )

                # Training step
                B, T = input_tensor.shape[:2]
                metrics = train_step(
                    bundle.tokenizer, bundle.dynamics, bundle.dynamics_ema,
                    bundle.dynamics_optimizer, input_tensor, actions,
                    master_key=master_key,
                    step=step,
                    B_img=int(B * cfg.image_fraction),
                    T=T,
                    n_splits=n_splits,
                    k_max=cfg.dynamics.k_max,
                    context_length=cfg.dynamics.context_length,
                    bootstrap_fraction=cfg.bootstrap_fraction if step > cfg.bootstrap_start else 0,
                    use_latent_data=use_latent_data,
                    ot_cfg=ot_cfg,
                )

                # EMA update
                ema_update_step(bundle.dynamics, bundle.dynamics_ema, ema_decay=cfg.ema_decay)

                # Logging — device_get on all hosts to stay in sync, only host 0 logs
                if logger.should_log(step):
                    metrics_cpu = jax.device_get(metrics)
                    if is_main_process:
                        scaling.on_step(step, metrics_cpu)
                        logger.log(
                            step,
                            metrics={
                                "flow_mse": metrics_cpu["flow_mse"],
                                "flow_mse_sequence": metrics_cpu["flow_mse_sequence"],
                                "flow_mse_image": metrics_cpu["flow_mse_image"],
                                "boot_mse": metrics_cpu["bootstrap_mse"],
                                "grad_norm": metrics_cpu["grad_norm"],
                                "flow_mse_low": metrics_cpu["flow_mse_low"],
                                "flow_mse_mid": metrics_cpu["flow_mse_mid"],
                                "flow_mse_high": metrics_cpu["flow_mse_high"],
                                "boot_target_norm": metrics_cpu["boot_target_norm"],
                                "lr": lr_schedule(step),
                                "T": T // n_splits,
                                **scaling.get_step_metrics(step),
                            },
                            pbar=pbar,
                            pbar_filter=r"^(flow_mse|boot_mse|lr)$",
                        )

                # Checkpointing
                bundle.maybe_save(checkpoint_manager, step, rng)

            if is_main_process:
                scaling.finalize()


@hydra.main(version_base=None, config_path="../configs", config_name="dynamics")
def main(cfg: DynamicsConfig):
    run(cfg)


if __name__ == "__main__":
    main()

