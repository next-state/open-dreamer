import logging
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path

import einops
import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from jaxlpips import LPIPS
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dreamer.configs import TokenizerConfig
from dreamer.training import compute_psnr
from dreamer.data import make_iterator
from dreamer.logging import MetricLogger
from dreamer.models import Tokenizer
from dreamer.parallel import create_data_model_parallel, MeshRules
from dreamer.utils import (
    make_state,
    make_manager,
    try_restore,
    maybe_save,
    normalize_with_dataset_stats,
    from_dict,
    get_lr_schedule,
    count_parameters_by_component,
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
    masked_sq_err = jnp.where(mae_mask, sq_err, 0.0)
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

def viz_step(model: Tokenizer, videos, rng, step, run_dir, use_wandb=False):
    rng = jax.random.fold_in(rng, step)
    mae_key, drop_key = jax.random.split(rng)

    grid = viz_step_jit(model, videos[:8,:1], mae_key=mae_key, drop_key=drop_key)
    grid = jax.device_get(grid)

    out = run_dir / "viz"
    out.mkdir(exist_ok=True, parents=True)
    imageio.imwrite(out / f"step_{step:06d}.png", grid)

    if use_wandb:
        wandb.log({"reconstruction": wandb.Image(np.array(grid), caption=f"Step {step}")}, step=step)

# ------------------------
# Run
# ------------------------

def run(cfg: TokenizerConfig):
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"[setup] writing artifacts to: {run_dir.resolve()}")

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project or cfg.run_name,
            name=cfg.run_name,
            config=asdict(cfg),
            dir=str(run_dir),
        )

    # Parallelism
    devices = jax.devices()
    device_count = len(devices)
    mesh, data_sharding = create_data_model_parallel(device_count, 1)
    mesh_rules = MeshRules(embed=None, mlp='model', attn='model', data='data')

    with jax.set_mesh(mesh):
        # Create the model
        key = jax.random.key(0)
        rng, init_key = jax.random.split(key)
        tokenizer_factory = lambda: Tokenizer(cfg, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        tokenizer = tokenizer_factory()

        param_counts = count_parameters_by_component(tokenizer)
        print(f"Parameter counts: {param_counts}")

        # Learning rate schedule
        if cfg.lr_schedule == "constant":
            lr = cfg.lr
            lr_schedule = None
        else:
            lr_schedule = get_lr_schedule(cfg.lr_schedule, cfg.init_lr, cfg.max_lr, cfg.lr_end, cfg.max_steps, cfg.warmup_steps, cfg.wsd_decay_steps)
            lr = lr_schedule
        
        tx = optax.adamw(lr, b1=0.9, b2=0.9, weight_decay=1e-4)
        optimizer_factory = lambda: nnx.Optimizer(tokenizer_factory(), tx, wrt=nnx.Param)
        optimizer = optimizer_factory()

        # ---------- Checkpointing ----------
        ckpt_dir = run_dir / "checkpoints"
        meta = {"cfg": asdict(cfg)}
        
        # Create state factory for abstract restoration
        # This must create abstract shapes without capturing concrete arrays

        with make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every) as mngr:
            restored = try_restore(mngr, tokenizer_factory, optimizer_factory, mesh, rng)
            start_step = 0
            if restored is not None:
                latest_step, r = restored
                nnx.update(tokenizer, r.model_state)
                nnx.update(optimizer, r.optimizer_state)
                rng = r.rng_state
                start_step = int(latest_step)
                cfg = from_dict(TokenizerConfig, r.meta["cfg"])
                print(f"[ckpt] Restored step {latest_step} (loaded directly to GPU)")

            # ---------- Train loop ----------
            logger = MetricLogger(
                use_wandb=cfg.use_wandb,
                log_every=cfg.log_every,
                max_steps=cfg.max_steps,
                wandb_obj=wandb,
            )

            dataset = make_iterator(cfg.dataset)

            # cfg.max_steps + 1 to make sure we log and checkpoint at max_steps
            pbar = tqdm(enumerate(dataset, start = start_step), total=cfg.max_steps + 1)
            
            for step, batch in pbar:
                if step < start_step:
                    continue
                if step >= cfg.max_steps:
                    break

                # Create fresh RNG state for this step by folding in step number
                step_rng = jax.random.fold_in(rng, step)
                mae_key, dropout_key = jax.random.split(step_rng)
                
                # Shard batch data
                videos = jax.device_put(batch["videos"], data_sharding)
                
                aux = train_step(
                    tokenizer, optimizer, videos,
                    mae_key=mae_key, dropout_key=dropout_key, step=step,
                    lpips_weight=cfg.lpips_weight, lpips_frac=cfg.lpips_frac,
                    dataset_mean=tuple(cfg.dataset.dataset_mean), 
                    dataset_std=tuple(cfg.dataset.dataset_std), 
                    log_gradients=cfg.log_gradients, 
                    tokenizer_loss_type=cfg.tokenizer_loss_type
                )

                if logger.should_log(step):
                    metrics_cpu = jax.device_get(aux)
                    if lr_schedule is None:
                        lr_value = cfg.lr
                    else:
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
                            **({} if not cfg.log_gradients else {
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
                
                # Save checkpoint state
                maybe_save(mngr, step, tokenizer, optimizer, rng, meta)

                if cfg.visualize_every > 0 and step % cfg.visualize_every == 0:
                    # Move a subset to host for visualization
                    viz_videos = batch["videos"][:8]
                    viz_step(tokenizer, viz_videos, step_rng, step, run_dir, use_wandb=cfg.use_wandb)
            
            mngr.wait_until_finished()

    if cfg.use_wandb and wandb.run is not None:
        wandb.finish()

# ------------------------
# Hydra entry
# ------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="tokenizer")
def main(cfg: DictConfig):
    schema = OmegaConf.structured(TokenizerConfig)
    cfg = OmegaConf.merge(schema, cfg)
    run(OmegaConf.to_object(cfg))

if __name__ == "__main__":
    main()
