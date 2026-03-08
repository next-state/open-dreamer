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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dreamer.configs import TokenizerConfig
from dreamer.training import compute_psnr
from dreamer.data import make_dual_iterator
from dreamer.logging import build_logger
from dreamer.models import Tokenizer
from dreamer.parallel import build_parallel, MeshRules
from dreamer.scaling import ScalingContext

from dreamer.checkpointing import (
    TokenizerCheckpointBundle,
    build_checkpoint_manager,
)
from dreamer.utils import (
    normalize_with_dataset_stats,
    count_parameters_by_component,
    setup_training_directories,
    build_lr_schedule,
    build_optimizer,
)

# disable preallocation completely
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'
jax.config.update("jax_default_device", None)

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


def lpips_on_mae_recon(lpips_model: LPIPS, pred, target, subsample_frac=1.0):
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
    return jnp.mean(lpips_model(pred_lp, tgt_lp))

# ------------------------
# Train step
# ------------------------

@nnx.jit(static_argnames=("lpips_weight", "lpips_frac", "dataset_mean", "dataset_std", "log_gradients", "tokenizer_loss_type", "freeze_encoder"))
def train_step(model: Tokenizer, optimizer: nnx.Optimizer, lpips_model: LPIPS | None, videos, *, mae_key, dropout_key, step,
               lpips_weight, lpips_frac, dataset_mean, dataset_std, log_gradients: bool, tokenizer_loss_type: str,
               mae_p_max: jnp.ndarray | None = None, freeze_encoder: bool = False):

    def loss_fn(model: Tokenizer):
        rngs = nnx.Rngs(mae=mae_key, dropout=dropout_key)
        pred, (mae_mask, keep_prob) = model(videos, deterministic=False, rngs=rngs, mae_p_max=mae_p_max)

        pred_norm = normalize_with_dataset_stats(pred, mean=dataset_mean, std=dataset_std)
        target_norm = normalize_with_dataset_stats(videos, mean=dataset_mean, std=dataset_std)
        if tokenizer_loss_type == "mae":
            mse = recon_loss_from_mae(pred_norm, target_norm, mae_mask)
        elif tokenizer_loss_type == "mse":
            mse = recon_loss_full_mse(pred_norm, target_norm)
        else:
            raise ValueError(f"Invalid loss type: {tokenizer_loss_type}")

        psnr = compute_psnr(pred / 255.0, videos / 255.0)

        lp = jnp.array(0.0)
        if lpips_weight > 0:
            lp = lpips_on_mae_recon(lpips_model, pred / 255.0, videos / 255.0, lpips_frac)

        total = mse + lpips_weight * lp

        aux = {"loss_total": total, "loss_mse": mse, "loss_lpips": lp, "keep_prob": keep_prob, "psnr": psnr}
        return total, aux

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    if freeze_encoder:
        encoder_grad_state = nnx.state(grads.encoder)
        zeroed_state = jax.tree.map(jnp.zeros_like, encoder_grad_state)
        nnx.update(grads.encoder, zeroed_state)

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

    return aux

# ------------------------
# Visualization
# ------------------------

@partial(jax.jit, static_argnames=())
def viz_step_jit(model: Tokenizer, videos, *, mae_key, drop_key, mae_p_max=None):
    """Visualization step with NNX model."""
    rngs = nnx.Rngs(mae=mae_key, dropout=drop_key)
    recon, (mask, _) = model(videos, deterministic=False, rngs=rngs, mae_p_max=mae_p_max)

    masked = videos * (1.0 - mask)
    recon_masked = masked + recon * mask

    grid = jnp.concatenate([videos, masked, recon_masked, recon], axis=2)
    grid = einops.rearrange(grid, "b t h w c -> h (b t w) c")
    return grid.clip(0, 255).astype(jnp.uint8)

def viz_step(model: Tokenizer, videos, rng, step, vis_dir, logger, mae_p_max=None):
    rng = jax.random.fold_in(rng, step)
    mae_key, drop_key = jax.random.split(rng)

    grid = viz_step_jit(model, videos[:2,:256:256//4], mae_key=mae_key, drop_key=drop_key, mae_p_max=mae_p_max)
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
    mesh, data_sharding, mesh_rules = build_parallel("data") # simple data parallel, if you finetune on longer sequences comment this line and uncomment the three below
    # mesh = jax.make_mesh((2, jax.local_device_count()//2), ('data', 'seq'))
    # data_sharding = NamedSharding(mesh, P('data', 'seq', None, None, None))
    # mesh_rules = MeshRules(data='data', seq='seq')
    

    with logger, jax.set_mesh(mesh):
        key = jax.random.key(cfg.seed)
        rng, init_key = jax.random.split(key)

        # Initialize tokenizer
        tokenizer = Tokenizer(cfg.tokenizer, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        param_counts = count_parameters_by_component(tokenizer)
        param_counts_formatted = {k: f"{v:,}" for k, v in param_counts.items()}
        print(f"Parameter counts: {param_counts_formatted}")

        # Scaling context (handles iso-FLOPs/tokens-per-param modes + CSV output)
        n_patches = (cfg.dataset.H // cfg.tokenizer.encoder.patch_size) * (cfg.dataset.W // cfg.tokenizer.encoder.patch_size)
        dl_cfg = cfg.dataset.dataloader_cfg
        short_T = dl_cfg.short_T
        long_T = dl_cfg.long_T
        long_ratio = dl_cfg.long_ratio

        # make_dual_iterator rescales batch size as B(T)=B_long * long_T // T.
        # This keeps B*T constant across short/long steps.
        # TODO: make the ScalingContext estimate the flops instead of passing it externally
        B_long = dl_cfg.B
        B_short = B_long * long_T // short_T
        tokens_per_step_bt = B_long * long_T

        # Expected FLOPs per step under alternating short/long batches.
        flops_short = tokenizer.estimate_flops(batch_size=B_short, seq_length=short_T)
        flops_long = tokenizer.estimate_flops(batch_size=B_long, seq_length=long_T)
        expected_flops_per_step = (1.0 - long_ratio) * flops_short + long_ratio * flops_long

        scaling = ScalingContext.create(
            cfg=cfg,
            param_count=param_counts["total"],
            flops_per_step=expected_flops_per_step,
            data_tokens_per_step=tokens_per_step_bt * n_patches,
            total_tokens_per_step=tokens_per_step_bt * (n_patches + cfg.tokenizer.encoder.n_latents),
            logger=logger,
            run_dir=run_dir,
        )

        # Build learning rate schedule and optimizer
        lr_schedule = build_lr_schedule(cfg.lr_schedule)
        optimizer = build_optimizer(cfg.optimizer, tokenizer, lr_schedule, d_model=cfg.tokenizer.encoder.d_model)

        # Initialize LPIPS model once (avoids repeated HF downloads during JIT tracing)
        lpips_model = LPIPS(pretrained_network="alexnet") if cfg.lpips_weight > 0 else None

        # Create checkpoint bundle
        bundle = TokenizerCheckpointBundle(tokenizer=tokenizer, tokenizer_optimizer=optimizer)

        # Data iterator
        train_dataloader = make_dual_iterator(cfg.dataset, device=data_sharding)

        with build_checkpoint_manager(cfg.ckpt, ckpt_dir, item_names=TokenizerCheckpointBundle.get_item_names()) as checkpoint_manager:
            # Resume from checkpoint
            start_step, bundle, rng = bundle.restore(checkpoint_manager, rng)
            scaling.start_training()

            # Training loop
            pbar = tqdm(enumerate(train_dataloader, start=start_step), initial=start_step,total=cfg.max_steps, dynamic_ncols=True)
            for step, batch in pbar:
                if step >= cfg.max_steps:
                    break

                # Create fresh RNG state for this step by folding in step number
                step_rng = jax.random.fold_in(rng, step)
                mae_key, dropout_key = jax.random.split(step_rng)

                # Compute dynamic mae_p_max if finetuning
                current_mae_p_max = None
                if cfg.mae_finetune_p_max_start != cfg.mae_finetune_p_max_end:
                    frac = min(step / max(cfg.max_steps - 1, 1), 1.0)
                    current_mae_p_max = jnp.array(
                        cfg.mae_finetune_p_max_start + (cfg.mae_finetune_p_max_end - cfg.mae_finetune_p_max_start) * frac
                    )

                # Shard batch data
                aux = train_step(
                    bundle.tokenizer, bundle.tokenizer_optimizer, lpips_model, batch["videos"],
                    mae_key=mae_key, dropout_key=dropout_key, step=step,
                    lpips_weight=cfg.lpips_weight, lpips_frac=cfg.lpips_frac,
                    dataset_mean=tuple(cfg.dataset.dataset_mean),
                    dataset_std=tuple(cfg.dataset.dataset_std),
                    log_gradients=cfg.logger.log_gradients,
                    tokenizer_loss_type=cfg.tokenizer_loss_type,
                    mae_p_max=current_mae_p_max,
                    freeze_encoder=cfg.freeze_encoder,
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
                            "lr": lr_schedule(step),
                            **({} if not cfg.mae_finetune else {"mae_p_max": float(current_mae_p_max)}),
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

                # Checkpointing
                bundle.maybe_save(checkpoint_manager, step, rng)

                if cfg.visualize_every > 0 and step % cfg.visualize_every == 0:
                    # Move a subset to host for visualization
                    viz_videos = batch["videos"][:8]
                    viz_step(bundle.tokenizer, viz_videos, step_rng, step, vis_dir, logger, mae_p_max=current_mae_p_max)

            scaling.finalize()


@hydra.main(version_base=None, config_path="../configs", config_name="tokenizer")
def main(cfg: TokenizerConfig):
    run(cfg)


if __name__ == "__main__":
    main()
