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
from dreamer.training import compute_psnr, LossRMSState, update_loss_rms
from dreamer.data import make_iterator
from dreamer.logging import build_logger
from dreamer.models import Tokenizer
from dreamer.parallel import build_parallel
from dreamer.scaling import ScalingContext
from dreamer.utils import (
    build_checkpoint_manager,
    try_restore,
    maybe_save,
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
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("max", lambda *args: max(args))


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

@nnx.jit(static_argnames=("lpips_frac", "dataset_mean", "dataset_std", "log_gradients", "tokenizer_loss_type", "loss_weights"))
def train_step(model: Tokenizer, optimizer: nnx.Optimizer, rms_state: LossRMSState, videos, *, mae_key, dropout_key, step,
               lpips_frac, dataset_mean, dataset_std, log_gradients: bool, tokenizer_loss_type: str,
               loss_weights: tuple[tuple[str, float], ...]):
    """
    Tokenizer training step with RMS loss normalization.

    Uses RMS loss normalization (paper Section 3) to balance MSE and LPIPS losses
    which can have very different scales.
    """
    # Convert loss_weights tuple to dict for easier access
    weights_dict = {name: weight for name, weight in loss_weights}

    # Get current RMS estimates (stop gradient so they don't affect backprop)
    rms_estimates = {
        name: jax.lax.stop_gradient(est)
        for name, est in rms_state.estimates.items()
    }

    def loss_fn(model: Tokenizer):
        rngs = nnx.Rngs(mae=mae_key, dropout=dropout_key)
        pred, (mae_mask, keep_prob) = model(videos, deterministic=False, rngs=rngs)

        pred_norm = normalize_with_dataset_stats(pred, mean=dataset_mean, std=dataset_std)
        target_norm = normalize_with_dataset_stats(videos, mean=dataset_mean, std=dataset_std)
        if tokenizer_loss_type == "mae":
            mse = recon_loss_from_mae(pred_norm, target_norm, mae_mask)
        elif tokenizer_loss_type == "mse":
            mse = recon_loss_full_mse(pred_norm, target_norm)
        else:
            raise ValueError(f"Invalid loss type: {tokenizer_loss_type}")

        psnr = compute_psnr(pred / 255.0, videos / 255.0)

        lpips = lpips_on_mae_recon(pred / 255.0, videos / 255.0, lpips_frac)

        # Raw losses for RMS tracking
        raw_losses = {"mse": mse, "lpips": lpips}

        # Normalize each loss by its running RMS estimate (paper Section 3)
        mse_norm = mse / (rms_estimates.get("mse", jnp.array(1.0)) + 1e-8)
        lpips_norm = lpips / (rms_estimates.get("lpips", jnp.array(1.0)) + 1e-8)

        # Combine normalized losses with fixed weights
        total = weights_dict.get("mse", 1.0) * mse_norm + weights_dict.get("lpips", 0.2) * lpips_norm

        aux = {
            "loss_total": total,
            "loss_mse": mse,
            "loss_lpips": lpips,
            "loss_mse_norm": mse_norm,
            "loss_lpips_norm": lpips_norm,
            "keep_prob": keep_prob,
            "psnr": psnr,
            "raw_losses": raw_losses,
        }
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

    optimizer.update(model, grads)

    # Update RMS estimates with raw losses (after gradient computation)
    raw_losses = aux.pop("raw_losses")
    new_rms_state, _ = update_loss_rms(rms_state, raw_losses, decay=0.999, warmup_steps=100)

    # Add RMS estimates to metrics for logging
    aux["rms/mse"] = new_rms_state.estimates["mse"]
    aux["rms/lpips"] = new_rms_state.estimates["lpips"]

    return new_rms_state, aux

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
        key = jax.random.key(cfg.seed)
        rng, init_key = jax.random.split(key)

        # Initialize tokenizer
        tokenizer = Tokenizer(cfg, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        param_counts = count_parameters_by_component(tokenizer)
        param_counts_formatted = {k: f"{v:,}" for k, v in param_counts.items()}
        print(f"Parameter counts: {param_counts_formatted}")

        # Scaling context (handles iso-FLOPs/tokens-per-param modes + CSV output)
        n_patches = (cfg.dataset.H // cfg.encoder.patch_size) * (cfg.dataset.W // cfg.encoder.patch_size)
        scaling = ScalingContext.create(
            cfg=cfg,
            param_count=param_counts["total"],
            flops_per_step=tokenizer.estimate_flops(batch_size=cfg.dataset.B, seq_length=cfg.dataset.T),
            data_tokens_per_step=cfg.dataset.B * cfg.dataset.T * n_patches,
            total_tokens_per_step=cfg.dataset.B * cfg.dataset.T * (n_patches + cfg.encoder.n_latents),
            logger=logger,
            run_dir=run_dir,
        )

        # Build learning rate schedule
        lr_schedule = build_lr_schedule(cfg.lr_schedule)

        # Build optimizer
        optimizer = build_optimizer(cfg.optimizer, tokenizer, lr_schedule, d_model=cfg.encoder.d_model)

        # Initialize RMS loss normalization state (paper Section 3)
        # Balances MSE and LPIPS losses which have very different scales
        rms_state = LossRMSState.init(("mse", "lpips"))

        # Loss weights for combining normalized losses (paper Eq. 3)
        # After RMS normalization, both losses have ~unit scale
        loss_weights = (
            ("mse", 1.0),
            ("lpips", cfg.lpips_weight),  # Paper uses 0.2
        )

        # Data iterator
        train_dataloader = make_iterator(cfg.dataset)
        train_iterator = iter(train_dataloader)  # type: ignore

        import orbax.checkpoint as ocp
        import grain.checkpoint
        with build_checkpoint_manager(
            cfg.ckpt, ckpt_dir,
            item_names=("model_state", "optimizer_state", "rms_state", "train_dataloader_state", "rngs", "meta")
        ) as checkpoint_manager:
            # Resume from checkpoint (manual handling for rms_state support)
            step = checkpoint_manager.latest_step()
            if step is not None:
                model_state = nnx.state(tokenizer)
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
                nnx.update(tokenizer, restored["model_state"])
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
            pbar = tqdm(enumerate(train_iterator, start=start_step), initial=start_step,total=cfg.max_steps)
            for step, batch in pbar:
                if step >= cfg.max_steps:
                    break

                # Create fresh RNG state for this step by folding in step number
                step_rng = jax.random.fold_in(rng, step)
                mae_key, dropout_key = jax.random.split(step_rng)

                # Shard batch data
                videos = jax.device_put(batch["videos"], data_sharding)

                # Training step with RMS loss normalization
                rms_state, aux = train_step(
                    tokenizer, optimizer, rms_state, videos,
                    mae_key=mae_key, dropout_key=dropout_key, step=step,
                    lpips_frac=cfg.lpips_frac,
                    dataset_mean=tuple(cfg.dataset.dataset_mean),
                    dataset_std=tuple(cfg.dataset.dataset_std),
                    log_gradients=cfg.logger.log_gradients,
                    tokenizer_loss_type=cfg.tokenizer_loss_type,
                    loss_weights=loss_weights,
                )

                if logger.should_log(step):
                    metrics_cpu = jax.device_get(aux)
                    scaling.on_step(step, metrics_cpu)
                    mse = metrics_cpu["loss_mse"]
                    logger.log(
                        step,
                        {
                            "loss": metrics_cpu["loss_total"],
                            "mse": mse,
                            "rmse": float(jnp.sqrt(mse)),
                            "lpips": metrics_cpu["loss_lpips"],
                            "psnr": metrics_cpu["psnr"],
                            "rms/mse": metrics_cpu["rms/mse"],
                            "rms/lpips": metrics_cpu["rms/lpips"],
                            "lr": lr_schedule(step),
                            **scaling.get_step_metrics(step),
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

                # Checkpointing (with rms_state)
                if checkpoint_manager.should_save(step):
                    model_state = nnx.state(tokenizer)
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

                if cfg.visualize_every > 0 and step % cfg.visualize_every == 0:
                    # Move a subset to host for visualization
                    viz_videos = batch["videos"][:8]
                    viz_step(tokenizer, viz_videos, step_rng, step, vis_dir, logger)

            scaling.finalize()


@hydra.main(version_base=None, config_path="../configs", config_name="tokenizer")
def main(cfg: TokenizerConfig):
    run(cfg)


if __name__ == "__main__":
    main()
