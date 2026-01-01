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
from hydra.core.hydra_config import HydraConfig
from jaxlpips import LPIPS
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dreamer.configs import TokenizerConfig
from dreamer.training import compute_psnr
from dreamer.data import make_iterator
from dreamer.logging import MetricLogger
from dreamer.models import Tokenizer
from dreamer.parallel import ParallelContext
from dreamer.utils import (
    make_state,
    make_manager,
    try_restore,
    maybe_save,
    normalize_with_dataset_stats,
    with_params,
    init_tokenizer,
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
# Forward (no module in jit)
# ------------------------

def forward_apply(apply_fn, variables, params, videos, *, mae_key, drop_key, train: bool):
    rngs = {"mae": mae_key}
    if train:
        rngs["dropout"] = drop_key

    # Merge optimized params into the state variables (constants, etc.)
    variables_with_params = with_params(variables, params)

    recon, mae_info = apply_fn(
        variables_with_params,
        videos,
        rngs=rngs,
        deterministic=not train,
    )
    return recon, mae_info

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

@partial(
    jax.jit,
    static_argnames=("apply_fn", "tx", "lpips_weight", "lpips_frac", "dataset_mean", "dataset_std", "log_gradients", "tokenizer_loss_type"),
)
def train_step(apply_fn, tx, variables, params, opt_state, videos, *, master_keys, step, lpips_weight, lpips_frac, dataset_mean, dataset_std, log_gradients: bool, tokenizer_loss_type: str):
    # FIXME: not entirely deterministic, because the key depends on the number of devices
    step_key = jax.random.fold_in(master_keys[0], step)
    mae_key, drop_key = jax.random.split(step_key, 2)

    def loss_fn(p):
        pred, (mae_mask, keep_prob) = forward_apply(apply_fn, variables, p, videos, mae_key=mae_key, drop_key=drop_key, train=True)    

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
        lp = 0
        if lpips_weight > 0:
            lp = lpips_on_mae_recon(pred / 255.0, videos / 255.0, lpips_frac)
        
        total = mse + lpips_weight * lp

        aux = {"loss_total": total, "loss_mse": mse, "loss_lpips": lp, "keep_prob": keep_prob, "psnr": psnr}
        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    if log_gradients:
        def _tree_std_mean(tree):
            std_tree = jax.tree_util.tree_map(lambda x: jnp.std(x), tree)
            leaves = jax.tree_util.tree_leaves(std_tree)
            if len(leaves) == 0:
                return jnp.array(0.0, dtype=jnp.float32)
            return jnp.mean(jnp.stack([jnp.asarray(x, dtype=jnp.float32) for x in leaves]))

        aux["grad/global_norm"] = optax.global_norm(grads)
        aux["grad/encoder_norm"] = optax.global_norm(grads["encoder"])
        aux["grad/decoder_norm"] = optax.global_norm(grads["decoder"])
        aux["grad/encoder_std_mean"] = _tree_std_mean(grads["encoder"])
        aux["grad/decoder_std_mean"] = _tree_std_mean(grads["decoder"])

    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, aux

# ------------------------
# Visualization
# ------------------------

@partial(jax.jit, static_argnames=("apply_fn",))
def viz_step_jit(apply_fn, variables, params, videos, *, mae_key, drop_key):
    recon, (mask, _) = forward_apply(apply_fn, variables, params, videos, mae_key=mae_key, drop_key=drop_key, train=True)

    masked = videos * (1.0 - mask)
    recon_masked = masked + recon * mask

    grid = jnp.concatenate([videos, masked, recon_masked, recon], axis=2)
    grid = einops.rearrange(grid[:, 0], "b h w c -> h (b w) c")
    return grid.clip(0, 255).astype(jnp.uint8)

def viz_step(apply_fn, variables, params, videos, rng, step, run_dir, use_wandb=False):
    rng = jax.random.fold_in(rng, step)
    mae_key, drop_key = jax.random.split(rng)

    grid = viz_step_jit(apply_fn, variables, params, videos[:8,:1], mae_key=mae_key, drop_key=drop_key)

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

    # Create parallel context for data parallelism
    ctx = ParallelContext.create(batch_size=cfg.dataset.B)

    rng = jax.random.PRNGKey(0)
    dataset = make_iterator(cfg.dataset)

    tokenizer = Tokenizer(cfg)
    apply_fn = tokenizer.apply
    rng, variables = init_tokenizer(rng, tokenizer, cfg)
    params = variables["params"]
    param_counts = count_parameters_by_component(params)
    print(f"Parameter counts: {param_counts}")

    if cfg.lr_schedule == "constant":
        lr = cfg.lr
        lr_schedule = None
    else:
        lr_schedule = get_lr_schedule(
            cfg.lr_schedule,
            cfg.init_lr,
            cfg.max_lr,
            cfg.lr_end,
            cfg.max_steps,
            cfg.warmup_steps,
            cfg.wsd_decay_steps,
        )
        lr = lr_schedule
    # Adamw params from Genie paper.
    tx = optax.adamw(lr, b1=0.9, b2=0.9, weight_decay=1e-4)
    opt_state = tx.init(params)

    # ---------- Checkpointing ----------
    ckpt_dir = run_dir / "checkpoints"
    mngr = make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every)

    state_example = make_state(params, opt_state, rng, step=0)
    meta = {"cfg": asdict(cfg)}

    restored = try_restore(mngr, state_example, ctx, meta)
    start_step = 0
    if restored is not None:
        # Restored state is already sharded/replicated on GPUs via ctx
        latest_step, r = restored
        params = r.state["params"]
        opt_state = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"])
        cfg = from_dict(TokenizerConfig, r.meta["cfg"])
        print(f"[ckpt] Restored step {latest_step} (loaded directly to GPU)")
    else:
        # No checkpoint - replicate initial state to GPUs
        params = ctx.replicate(params)
        opt_state = ctx.replicate(opt_state)
        print("[parallel] Replicated initial state to GPUs")
    
    # Always replicate variables (constants, not in checkpoint)
    variables = ctx.replicate(variables)

    # ---------- Train loop ----------
    logger = MetricLogger(
        use_wandb=cfg.use_wandb,
        log_every=cfg.log_every,
        max_steps=cfg.max_steps,
        wandb_obj=wandb,
    )

    pbar = tqdm(enumerate(dataset, start = start_step), total=cfg.max_steps)
    for step, batch in pbar:
        if step < start_step:
            continue
        if step >= cfg.max_steps:
            break

        rng, master_key = jax.random.split(rng)
        
        # Shard batch data
        videos = ctx.shard_batch(batch["videos"])
        
        # Generate keys matching batch size (one per sample)
        master_key = ctx.split_keys(master_key, count=videos.shape[0])
        
        params, opt_state, aux = train_step(apply_fn, tx, variables, params, opt_state, videos, master_keys=master_key, step=step, lpips_weight=cfg.lpips_weight, lpips_frac=cfg.lpips_frac, dataset_mean=tuple(cfg.dataset.dataset_mean), dataset_std=tuple(cfg.dataset.dataset_std), log_gradients=cfg.log_gradients, tokenizer_loss_type=cfg.tokenizer_loss_type)

        if logger.should_log(step):
            metrics_cpu = ctx.to_host_scalar(aux)
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
        
        # Save sharded arrays
        state = make_state(params, opt_state, rng, step)
        maybe_save(mngr, step, state, meta)

        if cfg.visualize_every > 0 and step % cfg.visualize_every == 0:
            # Move a subset to host for visualization
            viz_videos = jax.device_get(batch["videos"][:8])
            viz_params = jax.device_get(params)
            viz_variables = jax.device_get(variables)
            viz_step(apply_fn, viz_variables, viz_params, viz_videos, rng, step, run_dir, use_wandb=cfg.use_wandb)

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
