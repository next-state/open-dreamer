from functools import partial
import einops
import numpy as np
from tqdm import tqdm
from dataclasses import asdict
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import optax
import imageio
from jaxlpips import LPIPS
from pathlib import Path
import wandb
from hydra.core.hydra_config import HydraConfig

from dreamer.utils import make_state, make_manager, try_restore, maybe_save, normalize_with_dataset_stats, unnormalize_with_dataset_stats, with_params, init_tokenizer, from_dict
from dreamer.logging import MetricLogger
from dreamer.models import Tokenizer
from dreamer.data import make_iterator
from dreamer.configs import TokenizerConfig

    
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

lpips_loss_fn = LPIPS(pretrained_network="alexnet")

def lpips_on_mae_recon(pred, target, subsample_frac=1.0):
    # TODO: maybe unnormalize
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
    static_argnames=("apply_fn", "tx", "lpips_weight", "lpips_frac", "dataset_mean", "dataset_std"),
)
def train_step(apply_fn, tx, variables, params, opt_state, videos, *, master_key, step, lpips_weight, lpips_frac, dataset_mean, dataset_std):
     
    step_key = jax.random.fold_in(master_key, step)
    mae_key, drop_key = jax.random.split(step_key)

    def loss_fn(p):
        pred, (mae_mask, keep_prob) = forward_apply(apply_fn, variables, p, videos, mae_key=mae_key, drop_key=drop_key, train=True)    

        # For MSE: use standardized values (matches old gradient dynamics)
        pred_norm = normalize_with_dataset_stats(pred, mean=dataset_mean, std=dataset_std)
        target_norm = normalize_with_dataset_stats(videos, mean=dataset_mean, std=dataset_std)
        mse = recon_loss_from_mae(pred_norm, target_norm, mae_mask)
        
        # For LPIPS: use [0, 1] range
        lp = lpips_on_mae_recon(pred / 255, videos / 255, lpips_frac)
        
        total = mse + lpips_weight * lp
        mse_01 = mse * (dataset_std[0] ** 2)
        psnr = 10 * jnp.log10((1 / jnp.maximum(mse_01, 1e-10)))

        aux = {"loss_total": total, "loss_mse": mse, "loss_lpips": lp, "keep_prob": keep_prob, "psnr": psnr}
        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, aux

# ------------------------
# Visualization
# ------------------------

@partial(jax.jit, static_argnames=("apply_fn",))
def viz_step_jit(apply_fn, variables, params, videos, *, mae_key, drop_key):
    recon, (mask, _) = forward_apply(apply_fn, variables, params, videos, mae_key=mae_key, drop_key=drop_key, train=False)

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

    rng = jax.random.PRNGKey(0)
    dataset = make_iterator(cfg.dataset)

    tokenizer = Tokenizer(cfg)
    apply_fn = tokenizer.apply
    rng, variables = init_tokenizer(rng, tokenizer, cfg)
    params = variables["params"]

    tx = optax.adamw(cfg.lr)
    opt_state = tx.init(params)

    # ---------- Checkpointing ----------
    ckpt_dir = run_dir / "checkpoints"
    mngr = make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every)

    state_example = make_state(params, opt_state, rng, step=0)
    meta = {"cfg": asdict(cfg)}

    restored = try_restore(mngr, state_example, meta)
    start_step = 0
    if restored is not None:
        latest_step, r = restored
        params = r.state["params"]
        opt_state = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"])
        cfg = from_dict(TokenizerConfig, r.meta["cfg"])
        print(f"[ckpt] Restored step {latest_step}")

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

        rng, master_key = jax.random.split(rng)
        
        # Normalize videos
        videos = batch["videos"]
        params, opt_state, aux = train_step(apply_fn, tx, variables, params, opt_state, videos, master_key=master_key, step=step, lpips_weight=cfg.lpips_weight, lpips_frac=cfg.lpips_frac, dataset_mean=tuple(cfg.dataset.dataset_mean), dataset_std=tuple(cfg.dataset.dataset_std))

        if logger.should_log(step):
            mse = aux["loss_mse"]
            psnr = aux["psnr"]
            logger.log(
                step,
                {
                    "loss": aux["loss_total"],
                    "rmse": jnp.sqrt(mse),
                    "lpips": aux["loss_lpips"],
                    "psnr": psnr,
                },
                pbar=pbar,
            )

        state = make_state(params, opt_state, rng, step)
        maybe_save(mngr, step, state, meta)

        if cfg.visualize_every > 0 and step % cfg.visualize_every == 0:
            viz_step(apply_fn, variables, params, videos, rng, step, run_dir, use_wandb=cfg.use_wandb)

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
