import logging
import os
from functools import partial

import einops
import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jaxlpips import LPIPS
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.configs import TokenizerConfig
from dreamer.training import compute_psnr
from dreamer.data import make_iterator
from dreamer.logging import build_logger
from dreamer.models import Tokenizer
from dreamer.parallel import build_parallel
from dreamer.checkpointing import (
    TokenizerCheckpointBundle,
    build_checkpoint_manager,
    get_bundle_item_names,
    try_restore_bundle,
    maybe_save_bundle,
from dreamer.utils import(
    normalize_with_dataset_stats,
    count_parameters_by_component,
    setup_training_directories,
    build_lr_schedule,
    build_optimizer,
)

# disable preallocation completely
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)

# Register OmegaConf resolver for arithmetic expressions
# Usage: ${mul:a,b,c,...} multiplies all arguments
OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args))


# ------------------------
# Losses
# ------------------------

def recon_loss_from_mae(pred, target, mae_mask):
    sq_err = (pred - target) ** 2
    masked_sq_err = jnp.asarray(jnp.where(mae_mask, sq_err, 0.0))
    total_sse = jnp.sum(masked_sq_err)
    count = jnp.maximum(mae_mask.sum(), 1.0)
    return total_sse / count

def recon_loss_full_mse(pred, target):
    sq_err = (pred - target) ** 2
    return jnp.mean(sq_err)  # scalar


lpips_loss_fn = LPIPS(pretrained_network="alexnet")

def lpips_on_mae_recon(pred, target, subsample_frac=1.0):
    # Lpips expects [-1, 1] pixel range. Normalize to [-1, 1]
    pred = (pred - 0.5) * 2
    target = (target - 0.5) * 2
    if subsample_frac < 1.0:
        B, T = pred.shape[:2]
        step = max(1, int(1.0 / subsample_frac))
        idx = jnp.arange(T)[::step]
        pred = pred[:, idx]
        target = target[:, idx]

    pred_lp = einops.rearrange(pred, "b t h w c -> (b t) h w c")
    tgt_lp  = einops.rearrange(target, "b t h w c -> (b t) h w c")
    return jnp.mean(lpips_loss_fn(pred_lp, tgt_lp))

# ------------------------
# Train step
# ------------------------

@nnx.jit(static_argnames=("lpips_weight", "lpips_frac", "dataset_mean", "dataset_std", "log_gradients", "tokenizer_loss_type"))
def train_step(model: Tokenizer, optimizer: nnx.Optimizer, videos, *, mae_key, dropout_key, step, 
               lpips_weight, lpips_frac, dataset_mean, dataset_std, log_gradients: bool, tokenizer_loss_type: str):

    def loss_fn(model: Tokenizer):
        # Create RNG inside the loss function to avoid trace level issues with grad
        rngs = nnx.Rngs(mae=mae_key, dropout=dropout_key)
        pred, (mae_mask, keep_prob) = model(videos, deterministic=False, rngs=rngs)

        # For MSE: use standardized values (matches old gradient dynamics)
        pred_norm = normalize_with_dataset_stats(pred, mean=dataset_mean, std=dataset_std)
        target_norm = normalize_with_dataset_stats(videos, mean=dataset_mean, std=dataset_std)
        if tokenizer_loss_type == "mae":
            mse = recon_loss_from_mae(pred_norm, target_norm, mae_mask)
        elif tokenizer_loss_type == "mse":
            mse = recon_loss_full_mse(pred_norm, target_norm)
        else:
            raise ValueError(f"Invalid loss type: {tokenizer_loss_type}")

        # for psnr: use [0, 1] normalized pixel range.
        psnr = compute_psnr(pred / 255.0, videos / 255.0)
        
        # For LPIPS: use [0, 1] range
        lp = jnp.array(0.0)
        if lpips_weight > 0:
            lp = lpips_on_mae_recon(pred / 255.0, videos / 255.0, lpips_frac)
        
        total = mse + lpips_weight * lp

        aux = {"loss_total": total, "loss_mse": mse, "loss_lpips": lp, "keep_prob": keep_prob, "psnr": psnr}
        return total, aux

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    if log_gradients:
        def _tree_std_mean(tree):
            std_tree = jax.tree_util.tree_map(lambda x: jnp.std(x), tree)
            leaves = jax.tree_util.tree_leaves(std_tree)
            if len(leaves) == 0:
                return jnp.array(0.0, dtype=jnp.float32)
            return jnp.mean(jnp.stack([jnp.asarray(x, dtype=jnp.float32) for x in leaves]))

        aux["grad/global_norm"] = optax.global_norm(grads)
        graphdef, grad_state = nnx.split(grads)
        aux["grad/encoder_norm"] = optax.global_norm(grad_state.get("encoder", {}))
        aux["grad/decoder_norm"] = optax.global_norm(grad_state.get("decoder", {}))
        aux["grad/encoder_std_mean"] = _tree_std_mean(grad_state.get("encoder", {}))
        aux["grad/decoder_std_mean"] = _tree_std_mean(grad_state.get("decoder", {}))

    # Update model with optimizer 
    optimizer.update(model, grads)

    return aux

# ------------------------
# Visualization
# ------------------------

@partial(jax.jit, static_argnames=())
def viz_step_jit(model: Tokenizer, videos, *, mae_key, drop_key):
    """Visualization step with NNX model."""
    rngs = nnx.Rngs(mae=mae_key, dropout=drop_key)
    recon, (mask, _) = model(videos, deterministic=False, rngs=rngs)

    masked = videos * (1.0 - mask)
    recon_masked = masked + recon * mask

    grid = jnp.concatenate([videos, masked, recon_masked, recon], axis=2)
    grid = einops.rearrange(grid[:, 0], "b h w c -> h (b w) c")
    return grid.clip(0, 255).astype(jnp.uint8)

def viz_step(model: Tokenizer, videos, rng, step, vis_dir, logger):
    rng = jax.random.fold_in(rng, step)
    mae_key, drop_key = jax.random.split(rng)

    grid = viz_step_jit(model, videos[:8,:1], mae_key=mae_key, drop_key=drop_key)
    grid = jax.device_get(grid)

    imageio.imwrite(vis_dir / f"step_{step:06d}.png", grid)

    logger.log_image(step, "reconstruction", np.array(grid), caption=f"Step {step}")

# ------------------------
# Run
# ------------------------

def run(cfg: TokenizerConfig):
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
        key = jax.random.key(cfg.seed)
        rng, init_key = jax.random.split(key)

        # Initialize tokenizer
        tokenizer = Tokenizer(cfg.tokenizer, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        param_counts = count_parameters_by_component(tokenizer)
        print(f"Parameter counts: {param_counts}")

        # Build learning rate schedule
        lr_schedule = build_lr_schedule(cfg.lr_schedule)

        # Build optimizer
        optimizer = build_optimizer(cfg.optimizer, tokenizer, lr_schedule)

        # Create checkpoint bundle
        bundle = TokenizerCheckpointBundle(
            tokenizer=tokenizer,
            tokenizer_optimizer=optimizer,
        )

        # Data iterator
        train_dataloader = make_iterator(cfg.dataset)
        train_iterator = iter(train_dataloader)  # type: ignore

        with build_checkpoint_manager(
            cfg.ckpt, ckpt_dir,
            item_names=get_bundle_item_names(bundle)
        ) as checkpoint_manager:
            # Resume from checkpoint
            start_step, bundle, train_iterator, rng = try_restore_bundle(
                checkpoint_manager, bundle, train_iterator, rng
            )

            # Training loop
            pbar = tqdm(enumerate(train_iterator, start=start_step), initial=start_step,total=cfg.max_steps)
            for step, batch in pbar:
                if step >= cfg.max_steps:
                    break

                # Create fresh RNG state for this step by folding in step number
                step_rng = jax.random.fold_in(rng, step)
                mae_key, dropout_key = jax.random.split(step_rng)

                # Shard batch data
                videos = jax.device_put(batch["videos"], data_sharding)

                aux = train_step(
                    bundle.tokenizer, bundle.tokenizer_optimizer, videos,
                    mae_key=mae_key, dropout_key=dropout_key, step=step,
                    lpips_weight=cfg.lpips_weight, lpips_frac=cfg.lpips_frac,
                    dataset_mean=tuple(cfg.dataset.dataset_mean),
                    dataset_std=tuple(cfg.dataset.dataset_std),
                    log_gradients=cfg.logger.log_gradients,
                    tokenizer_loss_type=cfg.tokenizer_loss_type
                )

                if logger.should_log(step):
                    metrics_cpu = jax.device_get(aux)
                    lr_value = lr_schedule(step)
                    mse = metrics_cpu["loss_mse"]
                    psnr = metrics_cpu["psnr"]
                    logger.log(
                        step,
                        {
                            "loss": metrics_cpu["loss_total"],
                            "mse": mse,
                            "rmse": float(jnp.sqrt(mse)),
                            "lpips": metrics_cpu["loss_lpips"],
                            "psnr": psnr,
                            "lr": lr_value,
                            **({} if not cfg.logger.log_gradients else {
                                "grad/global_norm": metrics_cpu["grad/global_norm"],
                                "grad/encoder_norm": metrics_cpu["grad/encoder_norm"],
                                "grad/decoder_norm": metrics_cpu["grad/decoder_norm"],
                                "grad/encoder_std_mean": metrics_cpu["grad/encoder_std_mean"],
                                "grad/decoder_std_mean": metrics_cpu["grad/decoder_std_mean"],
                            }),
                        },
                        pbar=pbar,
                        pbar_filter=r"^(loss|mse|lpips|psnr|lr)$",
                    )

                # Checkpointing
                maybe_save_bundle(checkpoint_manager, step, bundle, train_iterator, rng)

                if cfg.visualize_every > 0 and step % cfg.visualize_every == 0:
                    # Move a subset to host for visualization
                    viz_videos = batch["videos"][:8]
                    viz_step(bundle.tokenizer, viz_videos, step_rng, step, vis_dir, logger)


@hydra.main(version_base=None, config_path="../configs", config_name="tokenizer")
def main(cfg: TokenizerConfig):
    run(cfg)


if __name__ == "__main__":
    main()
