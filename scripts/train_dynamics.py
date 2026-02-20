"""Train pixel-space diffusion dynamics with a neural-field decoder."""

import logging
import os

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.actions import Actions
from dreamer.checkpointing import DynamicsOnlyCheckpointBundle, build_checkpoint_manager
from dreamer.configs import DynamicsConfig
from dreamer.data import make_iterator
from dreamer.logging import build_logger
from dreamer.models import Dynamics
from dreamer.parallel import build_parallel
from dreamer.scaling import ScalingContext
from dreamer.training import run_evaluation, shortcut_forcing_step
from dreamer.utils import (
    RunningNormalizer,
    build_lr_schedule,
    build_optimizer,
    count_parameters_by_component,
    setup_training_directories,
)

# Suppress absl info logs
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
logging.getLogger('absl').setLevel(logging.WARNING)

# Register OmegaConf resolver for arithmetic expressions
OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("max", lambda *args: max(args))


def _normalize_frames(
    videos: jnp.ndarray,
    pixel_normalizer: RunningNormalizer,
    dtype: jnp.dtype,
    *,
    update_stats: bool,
) -> jnp.ndarray:
    videos_01 = videos.astype(jnp.float32) / 255.0
    if update_stats:
        pixel_normalizer.update(videos_01)
    normalized = pixel_normalizer.normalize(videos_01)
    return normalized.astype(dtype)


def _num_pixel_tokens(cfg: DynamicsConfig) -> int:
    patch = cfg.dynamics.patch_size
    H_total = cfg.dataset.H + sum(cfg.dataset.padding_H)
    W_total = cfg.dataset.W + sum(cfg.dataset.padding_W)

    if (H_total % patch) != 0 or (W_total % patch) != 0:
        raise ValueError(
            f"Padded image size ({H_total}, {W_total}) must be divisible by dynamics.patch_size={patch}."
        )

    n_tokens = (H_total // patch) * (W_total // patch)
    if (n_tokens % cfg.dynamics.packing_factor) != 0:
        raise ValueError(
            f"Number of pixel tokens {n_tokens} must be divisible by packing_factor={cfg.dynamics.packing_factor}."
        )
    return n_tokens


# ---------------------------
# Training / Eval step
# ---------------------------

@nnx.jit(
    static_argnames=("k_max", "context_length", "B_self"),
    donate_argnames=("videos", "actions"),
)
def train_step(
    dynamics: Dynamics,
    pixel_normalizer: RunningNormalizer,
    optimizer: nnx.Optimizer,
    videos: jnp.ndarray,      # (B, T, H, W, C)
    actions: Actions,         # (B, T, ...)
    *,
    master_key: jax.Array,
    step: int,
    k_max: int,
    context_length: int | None,
    B_self: int,
):
    frames = _normalize_frames(videos, pixel_normalizer, dynamics.dtype, update_stats=True)
    step_key = jax.random.fold_in(master_key, step)

    def loss_fn(model: Dynamics, frames, actions, context_length):
        losses, aux = shortcut_forcing_step(
            dynamics_model=model,
            actions=actions,
            latents=frames,
            rng=step_key,
            k_max=k_max,
            context_length=context_length,
            task_embeddings=None,
            B_self=B_self,
        )
        return losses['total'], aux

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        dynamics,
        frames,
        actions,
        context_length,
    )

    optimizer.update(dynamics, grads)
    return metrics


@nnx.jit(
    static_argnames=("k_max", "context_length", "B_self"),
)
def eval_step(
    dynamics: Dynamics,
    pixel_normalizer: RunningNormalizer,
    videos: jnp.ndarray,
    actions: Actions,
    *,
    master_key: jax.Array,
    step: int,
    k_max: int,
    context_length: int | None,
    B_self: int,
):
    frames = _normalize_frames(videos, pixel_normalizer, dynamics.dtype, update_stats=False)
    step_key = jax.random.fold_in(master_key, step)

    losses, aux = shortcut_forcing_step(
        dynamics_model=dynamics,
        actions=actions,
        latents=frames,
        rng=step_key,
        k_max=k_max,
        context_length=context_length,
        task_embeddings=None,
        B_self=B_self,
    )

    return {
        "loss": losses["total"],
        "flow_mse": aux["flow_mse"],
        "bootstrap_mse": aux["bootstrap_mse"],
    }


# ---------------------------
# Main
# ---------------------------

def _configure_pixel_neural_field_dynamics(cfg: DynamicsConfig) -> None:
    if cfg.dataset.data_type != "video":
        raise ValueError(
            "Pixel-space dynamics requires dataset.data_type='video'. "
            f"Got {cfg.dataset.data_type!r}."
        )

    if getattr(cfg.dynamics, "input_space", "latent") != "pixel":
        print("[setup] Overriding dynamics.input_space='pixel' for this trainer.")
        cfg.dynamics.input_space = "pixel"

    if getattr(cfg.dynamics, "decoder_type", "linear") != "neural_field":
        print("[setup] Overriding dynamics.decoder_type='neural_field' for this trainer.")
        cfg.dynamics.decoder_type = "neural_field"

    cfg.dynamics.patch_size = cfg.dataset.patch_size
    cfg.dynamics.image_channels = cfg.dataset.C

    print(
        "[setup] pixel neural-field dynamics:"
        f" patch={cfg.dynamics.patch_size},"
        f" channels={cfg.dynamics.image_channels},"
        f" hidden_dim={cfg.dynamics.field_hidden_dim},"
        f" coord_dim={cfg.dynamics.field_coord_dim},"
        f" num_freqs={cfg.dynamics.field_num_freqs},"
        f" encoding={cfg.dynamics.field_coord_encoding},"
        f" noisy_input={cfg.dynamics.field_use_noisy_input}"
    )


def run(cfg: DynamicsConfig):
    _configure_pixel_neural_field_dynamics(cfg)

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

        # Initialize dynamics
        dynamics = Dynamics(cfg.dynamics, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        pixel_normalizer = RunningNormalizer(shape=(cfg.dataset.C,))
        dataset_mean = jnp.asarray(cfg.dataset.dataset_mean, dtype=jnp.float32)
        dataset_std = jnp.asarray(cfg.dataset.dataset_std, dtype=jnp.float32)
        if dataset_mean.shape[0] != cfg.dataset.C or dataset_std.shape[0] != cfg.dataset.C:
            raise ValueError(
                f"dataset_mean/std must have {cfg.dataset.C} channels, got "
                f"{dataset_mean.shape[0]} and {dataset_std.shape[0]}."
            )
        # Warm start with dataset-level stats so training begins near zero-mean/unit-variance.
        pixel_normalizer.mean.value = dataset_mean
        pixel_normalizer.var.value = jnp.maximum(dataset_std * dataset_std, 1e-6)

        param_counts = count_parameters_by_component(dynamics)
        print(f"Parameter counts: {param_counts['total']:,}")

        # Scaling context
        n_tokens = _num_pixel_tokens(cfg)
        n_spatial = n_tokens // cfg.dynamics.packing_factor
        B, T = cfg.dataset.dataloader_cfg.B, cfg.dataset.dataloader_cfg.T

        dynamics_flops = dynamics.estimate_flops(batch_size=B, seq_length=T, n_latents=n_tokens)

        scaling = ScalingContext.create(
            cfg=cfg,
            param_count=param_counts["total"],
            flops_per_step=dynamics_flops,
            data_tokens_per_step=B * T * (n_spatial + 1),
            total_tokens_per_step=B * T * (2 + n_spatial + cfg.dynamics.n_register),
            logger=logger,
            run_dir=run_dir,
        )

        # Build learning rate schedule
        lr_schedule = build_lr_schedule(cfg.lr_schedule)

        # Build optimizer
        optimizer = build_optimizer(cfg.optimizer, dynamics, lr_schedule, d_model=cfg.dynamics.d_model)

        # Checkpoint bundle
        bundle = DynamicsOnlyCheckpointBundle(
            dynamics=dynamics,
            dynamics_optimizer=optimizer,
        )

        dataloader = make_iterator(cfg.dataset, device=data_sharding)
        with build_checkpoint_manager(cfg.ckpt, ckpt_dir, item_names=DynamicsOnlyCheckpointBundle.get_item_names()) as checkpoint_manager:
            start_step, bundle, rng = bundle.restore(checkpoint_manager, rng)
            scaling.start_training()

            pbar = tqdm(enumerate(dataloader, start_step), initial=start_step, total=cfg.max_steps)
            for step, batch in pbar:
                if step >= cfg.max_steps:
                    break

                rng, master_key, eval_key = jax.random.split(rng, num=3)

                actions = batch["actions"]
                videos = batch.get("videos")
                if videos is None:
                    raise ValueError("Pixel-space dynamics training expects `videos` in dataloader batches.")

                should_eval = (
                    (cfg.write_video_every > 0 and (step % cfg.write_video_every == 0) and step > 0)
                    or step == cfg.max_steps - 1
                )
                B_self = int((videos.shape[0] * cfg.bootstrap_frac) if step >= cfg.bootstrap_start else 0)
                if should_eval:
                    val_data = videos[:4]
                    val_actions = actions[:4]
                    run_evaluation(
                        cfg=cfg,
                        step=step,
                        tokenizer=None,
                        dynamics=bundle.dynamics,
                        val_data=val_data,
                        val_actions=val_actions,
                        use_latent_data=False,
                        vis_dir=vis_dir,
                        rng=eval_key,
                        logger=logger,
                        pixel_normalizer=pixel_normalizer,
                    )
                    eval_metrics = eval_step(
                        bundle.dynamics,
                        pixel_normalizer,
                        val_data,
                        val_actions,
                        master_key=eval_key,
                        step=step,
                        k_max=cfg.dynamics.k_max,
                        context_length=cfg.dynamics.context_length,
                        B_self=B_self,
                    )
                    logger.log_metrics(step, jax.device_get(eval_metrics), prefix="eval/")

                metrics = train_step(
                    bundle.dynamics,
                    pixel_normalizer,
                    bundle.dynamics_optimizer,
                    videos,
                    actions,
                    master_key=master_key,
                    step=step,
                    k_max=cfg.dynamics.k_max,
                    context_length=cfg.dynamics.context_length,
                    B_self=B_self,
                )

                if logger.should_log(step):
                    metrics_cpu = jax.device_get(metrics)
                    scaling.on_step(step, metrics_cpu)
                    logger.log(
                        step,
                        metrics={
                            "flow_mse": metrics_cpu["flow_mse"],
                            "boot_mse": metrics_cpu["bootstrap_mse"],
                            "lr": lr_schedule(step),
                            **scaling.get_step_metrics(step),
                        },
                        pbar=pbar,
                        pbar_filter=r"^(flow_mse|boot_mse|lr)$",
                    )

                bundle.maybe_save(checkpoint_manager, step, rng)

            scaling.finalize()


@hydra.main(version_base=None, config_path="../configs", config_name="dynamics")
def main(cfg: DynamicsConfig):
    run(cfg)


if __name__ == "__main__":
    main()
