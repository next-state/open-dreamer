from functools import partial
import einops
from tqdm import tqdm
from dataclasses import dataclass, asdict, field
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import optax
from dreamer.models import Tokenizer
from dreamer.data import make_iterator
from dreamer.configs import TokenizerConfig
import imageio
from jaxlpips import LPIPS
from pathlib import Path
import wandb
from hydra.core.hydra_config import HydraConfig
from dreamer.utils import make_state, make_manager, try_restore, maybe_save
from dreamer.logging import MetricLogger

def init_models(rng, tokenizer, videos):
    rng, params_rng, mae_rng, dropout_rng = jax.random.split(rng, 4)
    
    # Initialize the whole tokenizer
    tokenizer_vars = tokenizer.init({"params": params_rng,"mae": mae_rng,"dropout": dropout_rng}, videos, deterministic=True)
    return rng, tokenizer_vars

# --- forward (no jit; we jit the train_step) ---
def forward_apply(tokenizer, params, videos, *, mae_key, drop_key, train: bool):
    # Avoid TracerBool issues: pass a python bool here OR replace with lax.cond if needed.
    rngs = {"mae": mae_key}
    if train: rngs["dropout"] = drop_key
    
    # Wrap params for Flax
    variables = {"params": params}
    pred_btnd, mae_info = tokenizer.apply(variables, videos, rngs=rngs, deterministic=not train)
    return pred_btnd, mae_info  # mae_info = (mae_mask, keep_prob)

# --- loss ---
def recon_loss_from_mae(pred, target, mae_mask):
    sq_err = (pred - target) ** 2
    masked_sq_err = jnp.where(mae_mask, sq_err, 0.0)
    total_sse = jnp.sum(masked_sq_err)
    count = jnp.maximum(mae_mask.sum(), 1.0)
    return total_sse / count


lpips_loss_fn = LPIPS(pretrained_network="alexnet")
def lpips_on_mae_recon(
    pred, target, mae_mask,
    subsample_frac: float = 1.0
):
    # 1) Blend GT for visible patches => "recon_masked"
    # pred = jnp.where(mae_mask, pred, target) # TODO: check if this is correct

    # 3) Optional subsample frames over T to save compute
    if subsample_frac < 1.0:
        B, T = pred.shape[:2]
        step = max(1, int(1.0/subsample_frac))
        idx = jnp.arange(T)[::step]
        pred = pred[:, idx]
        target = target[:, idx]

    # 5) Flatten B,T for a single LPIPS call: (B*T,H,W,C)
    recon_lp = einops.rearrange(pred, 'b t ... -> (b t) ...')
    target_lp= einops.rearrange(target,'b t ... -> (b t) ...')

    # 6) LPIPS returns per-example loss; average it
    lp = lpips_loss_fn(recon_lp, target_lp)  # shape (BT,)
    return jnp.mean(lp)

# --- viz step ---
@partial(jax.jit, static_argnames=("tokenizer"))
def viz_step(tokenizer, params, batch, *, mae_key, drop_key):
    # batch is 0..1, scale to 0..255
    videos = batch * 255.0

    # Run full model (no dropout during viz)
    recon, (frame_mask, keep_prob) = forward_apply(
        tokenizer, params, videos,
        mae_key=mae_key, drop_key=drop_key, train=False
    )

    masked_input = videos * (1.0 - frame_mask)
    recon_masked = videos * (1.0 - frame_mask) + recon * frame_mask
    recon_full = recon

    return {
        "target": videos,
        "masked_input": masked_input,
        "recon_masked": recon_masked,
        "recon_full": recon_full,
        "mae_mask": frame_mask,
    }


# --- train step ---
@partial(jax.jit, static_argnames=("tokenizer","tx","patch","H","W","C", "lpips_weight", "lpips_frac"))
def train_step(tokenizer, tx, params, opt_state, batch, *,
               patch, H, W, C, master_key, step, lpips_weight=0.2, lpips_frac=1.0):
    """
    (master_key, params, opt_state, model_state, batch)
        │
        ▼
    [ compute grads ]
        │
        ▼
    Optax: (grads, opt_state, params) → (updates, new_opt_state)
    Flax:  params ⟶ apply updates → new_params
        │
        ▼
    return (new_params, new_opt_state, new_model_state, metrics)
    """
    # 1) Prepare data
    videos = batch['videos']/127.5 - 1   # (B,T,H,W,C)

    # 2) Make per-step RNGs (fold_in ensures different masks per step even if base key repeats)
    step_key  = jax.random.fold_in(master_key, step)
    mae_key, drop_key = jax.random.split(step_key)

    # 3) Define loss fn (closes over encoder/decoder + non-param states)
    def loss_fn(p):
        pred, mae_info = forward_apply(tokenizer, p, videos,
                                       mae_key=mae_key, drop_key=drop_key, train=True)
        mae_mask, keep_prob = mae_info
        mse = recon_loss_from_mae(pred, videos, mae_mask)

        # LPIPS on recon_masked vs target (unpatchified frames)
        lpips = lpips_on_mae_recon(pred, videos, mae_mask, subsample_frac=lpips_frac)
        total = mse + lpips_weight * lpips

        aux = {"loss_total": total, "loss_mse": mse, "loss_lpips": lpips, "keep_prob": keep_prob}

        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # 4) Update
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # 5) Return
    return new_params, opt_state, aux

def run(cfg: TokenizerConfig):
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"[setup] writing artifacts to: {run_dir.resolve()}")

    # Initialize wandb if enabled
    if cfg.use_wandb:
        wandb_project = cfg.wandb_project or cfg.run_name
        wandb.init(
            entity=cfg.wandb_entity,
            project=wandb_project,
            name=cfg.run_name,
            config=asdict(cfg),
            dir=str(run_dir),
        )

    rng = jax.random.PRNGKey(0)
    
    # data
    dataset = make_iterator(cfg.dataset)
    
    # Mock dataset for debugging
    rng, key = jax.random.split(rng)
    B, T, H, W, C = cfg.dataset.B, cfg.dataset.T, cfg.dataset.H, cfg.dataset.W, cfg.dataset.C
    dummy_videos = jax.random.randint(key, (B, T, H, W, C), 0, 255, dtype=jnp.uint8)

    tokenizer = Tokenizer(cfg)
    rng, tokenizer_vars = init_models(rng, tokenizer, dummy_videos)

    # optim
    params = tokenizer_vars["params"]
    tx = optax.adamw(cfg.lr)
    opt_state = tx.init(params)

    # ---------- ORBAX: manager + (optional) restore ----------
    # ckpt_dir = run_dir / "checkpoints"
    # mngr = make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every)

    # Build example trees for safe restore (use live shapes/dtypes).
    # state_example = make_state(params, opt_state, rng, step=0)
    # meta_example = {
    #     "H": cfg.dataset.H, "W": cfg.dataset.W, "C": cfg.dataset.C, "patch_size": cfg.patch_size
    # }

    # restored = try_restore(mngr, state_example, meta_example)
    # start_step = 0
    # if restored is not None:
    #     latest_step, r = restored
    #     # Unpack state back to your locals
    #     params = r.state["params"]
    #     opt_state = r.state["opt_state"]
    #     rng = r.state["rng"]
    #     start_step = int(r.state["step"])
    #     # Optional: you can read r.meta here if you want to sanity-check config.

    #     # No need to rebuild separate vars, we just use params.
    #     print(f"Restored checkpoint at step {latest_step} from {ckpt_dir}")

    # ---------- Train loop ----------
    try:
        logger = MetricLogger(
            use_wandb=cfg.use_wandb, 
            log_every=cfg.log_every, 
            max_steps=cfg.max_steps,
            wandb_obj=wandb
        )
        pbar = tqdm(enumerate(dataset))
        
        for step, batch in pbar:
            data_start_t = time.perf_counter()
            data_t = time.perf_counter() - data_start_t

            rng, master_key = jax.random.split(rng)
            train_start_t = time.perf_counter()
            params, opt_state, aux = train_step(
                tokenizer, tx, params, opt_state, batch,
                patch=cfg.patch_size, H=cfg.dataset.H, W=cfg.dataset.W, C=cfg.dataset.C, 
                master_key=master_key, step=step, 
                lpips_weight=cfg.lpips_weight, lpips_frac=cfg.lpips_frac,
            )
            train_t = time.perf_counter() - train_start_t
            total_t = data_t + train_t

            # Log
            if logger.should_log(step):
                mse_loss = aux['loss_mse']
                psnr = 10 * jnp.log10(1.0 / jnp.maximum(mse_loss, 1e-10))
                rmse = jnp.sqrt(mse_loss)
                
                logger.log(
                    step,
                    metrics={
                        "total": aux['loss_total'],
                        "rmse": rmse,
                        "lpips": aux['loss_lpips'],
                        "psnr": psnr,
                        "time/data": data_t,
                        "time/train": train_t,
                        "time/total": total_t,
                    },
                    pbar=pbar,
                    float_fmt=".6f"
                )

            # # Save (async)
            # state = make_state(params, opt_state, rng, step)
            # maybe_save(mngr, step, state, meta_example)

    finally:
        # Make sure any background saves finish before exit.
        # mngr.wait_until_finished()
        if cfg.use_wandb and wandb.run is not None:
            wandb.finish()
            print("[wandb] Finished logging.")

@hydra.main(version_base=None, config_path="../configs", config_name="tokenizer")
def main(cfg: DictConfig):
    schema = OmegaConf.structured(TokenizerConfig)
    cfg = OmegaConf.merge(schema, cfg)
    tokenizer_cfg = OmegaConf.to_object(cfg)
    run(tokenizer_cfg)

if __name__ == "__main__":
    main()
