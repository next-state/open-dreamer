from functools import partial
from tqdm import tqdm
from dataclasses import asdict
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import optax
from dreamer.data import make_iterator
import imageio
from jaxlpips import LPIPS
from pathlib import Path
import wandb
from hydra.core.hydra_config import HydraConfig
from dreamer.utils import temporal_patchify, temporal_unpatchify, normalize_with_dataset_stats, unnormalize_with_dataset_stats, setup_experiment_checkpointing, maybe_save_snapshot, pack_mae_params, unpack_mae_params, create_tokenizer_models, load_snapshot_weights, init_tokenizer_vars
from dreamer.logging import MetricLogger
from dreamer.configs import TokenizerTrainConfig


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# --- forward (no jit; we jit the train_step) ---
def forward_apply(encoder, decoder, enc_vars, dec_vars, patches_btnd, *, mae_key, drop_key, train: bool):
    # Avoid TracerBool issues: pass a python bool here OR replace with lax.cond if needed.
    rngs_enc = {"mae": mae_key} if not train else {"mae": mae_key, "dropout": drop_key}
    z_btLd, mae_info = encoder.apply(enc_vars, patches_btnd, rngs=rngs_enc, deterministic=not train)

    rngs_dec = {} if not train else {"dropout": drop_key}
    pred_btnd = decoder.apply(dec_vars, z_btLd, rngs=rngs_dec, deterministic=not train)
    return pred_btnd, mae_info  # mae_info = (mae_mask, keep_prob)

# --- loss ---
def recon_loss_from_mae(pred_btnd, target_btnd, mae_mask):
    pred_masked   = jnp.where(mae_mask, pred_btnd, 0.0)
    target_masked = jnp.where(mae_mask, target_btnd, 0.0)
    num = jnp.maximum(mae_mask.sum(), 1.0)
    return jnp.sum((pred_masked - target_masked) ** 2) / (num * pred_btnd.shape[-1])

# --- instantiate once (top-level / main) ---
lpips_loss_fn = None

def lpips_on_mae_recon(
    pred, target, mae_mask, *, H, W, C, patch,
    subsample_frac: float = 1.0
):
    """
    pred:    (B,T,Np,D)
    target:  (B,T,Np,D)
    mae_mask:     (B,T,Np,1)  True where patch is masked (must reconstruct)
    Returns scalar LPIPS averaged over (B,T).
    """
    # 1) Blend GT for visible patches => "recon_masked"
    recon_masked_btnd = jnp.where(mae_mask, pred, target)

    # 2) Unpatchify to (B,T,H,W,C) in [0,1]
    recon_imgs = temporal_unpatchify(recon_masked_btnd, H, W, C, patch)
    target_imgs = temporal_unpatchify(target,        H, W, C, patch)

    # 3) Optional subsample frames over T to save compute
    if subsample_frac < 1.0:
        B, T = recon_imgs.shape[:2]
        step = max(1, int(1.0/subsample_frac))
        idx = jnp.arange(T)[::step]
        recon_imgs = recon_imgs[:, idx]
        target_imgs = target_imgs[:, idx]

    # 4) Rescale to [-1,1] for LPIPS
    recon_lp = jnp.clip(recon_imgs * 2.0 - 1.0, -1.0, 1.0)
    target_lp = jnp.clip(target_imgs * 2.0 - 1.0, -1.0, 1.0)

    # 5) Flatten B,T for a single LPIPS call: (B*T,H,W,C)
    BT = recon_lp.shape[0] * recon_lp.shape[1]
    H_, W_, C_ = recon_lp.shape[2], recon_lp.shape[3], recon_lp.shape[4]
    recon_lp = recon_lp.reshape((BT, H_, W_, C_))
    target_lp = target_lp.reshape((BT, H_, W_, C_))

    # 6) LPIPS returns per-example loss; average it
    lp = lpips_loss_fn(recon_lp, target_lp)  # shape (BT,)
    return jnp.mean(lp)

# --- viz step ---
@partial(jax.jit, static_argnames=("encoder","decoder","patch"))
def viz_step(encoder, decoder, enc_vars, dec_vars, batch, *, patch, mae_key, drop_key, dataset_mean, dataset_std):
    # Same preprocessing as train
    target_btnd = temporal_patchify(batch, patch)  # (B, T, Np, D)
    target_norm = normalize_with_dataset_stats(target_btnd, mean=dataset_mean, std=dataset_std)

    # Run full model (no dropout during viz)
    pred_norm, (mae_mask_btNp1, keep_prob_bt1) = forward_apply(
        encoder, decoder, enc_vars, dec_vars, target_norm,
        mae_key=mae_key, drop_key=drop_key, train=False
    )
    pred_btnd = unnormalize_with_dataset_stats(pred_norm, mean=dataset_mean, std=dataset_std)
    pred_btnd = jnp.clip(pred_btnd, 0.0, 1.0)

    # Compose standard MAE visualization:
    # - masked_input: what the model actually sees (masked target patches)
    # - recon_masked: inpaint only masked patches (visible target patches kept as GT)
    masked_input_btnd  = jnp.where(mae_mask_btNp1, 0.0, target_btnd)
    recon_masked_btnd  = jnp.where(mae_mask_btNp1, pred_btnd, target_btnd)
    recon_full_btnd    = pred_btnd  # decoder everywhere

    return {
        "target": target_btnd,
        "masked_input": masked_input_btnd,
        "recon_masked": recon_masked_btnd,
        "recon_full": recon_full_btnd,
        "mae_mask": mae_mask_btNp1,
        "keep_prob": keep_prob_bt1,
    }


# --- train step ---
@partial(jax.jit, static_argnames=("encoder","decoder","tx","patch","H","W","C", "lpips_weight", "lpips_frac", "should_log"))
def train_step(encoder, decoder, tx, params, opt_state, enc_vars, dec_vars, batch, *,
               patch, H, W, C, master_key, step, lpips_weight=0.2, lpips_frac=1.0, should_log=False, dataset_mean, dataset_std):
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
    target_btnd = temporal_patchify(batch, patch)  # (B, T, Np, Dp)

    # 2) Make per-step RNGs (fold_in ensures different masks per step even if base key repeats)
    step_key  = jax.random.fold_in(master_key, step)
    mae_key, drop_key = jax.random.split(step_key)

    # 3) Define loss fn (closes over encoder/decoder + non-param states)
    def loss_fn(packed_params):
        # Replace params in vars
        ev, dv = unpack_mae_params(packed_params, enc_vars, dec_vars)
        target_norm = normalize_with_dataset_stats(target_btnd, mean=dataset_mean, std=dataset_std)
        pred_norm, mae_info = forward_apply(encoder, decoder, ev, dv, target_norm,
                                            mae_key=mae_key, drop_key=drop_key, train=True)
        mae_mask, keep_prob = mae_info

        mse = recon_loss_from_mae(pred_norm, target_norm, mae_mask)

        mse_pix = 0.0  # only for logging, to compute PSNR/RMSE
        pred_btnd = None
        
        if should_log or lpips_weight > 0.0:
            pred_btnd = unnormalize_with_dataset_stats(pred_norm, mean=dataset_mean, std=dataset_std)
            
        if should_log:
            mse_pix = recon_loss_from_mae(pred_btnd, target_btnd, mae_mask)

        # LPIPS on recon_masked vs target (unpatchified frames)
        if lpips_weight > 0.0:
            lpips = lpips_on_mae_recon(
                pred_btnd, target_btnd, mae_mask,
                H=H, W=W, C=C, patch=patch, subsample_frac=lpips_frac
            )
            total = mse + lpips_weight * lpips
        else:
            lpips = 0.0
            total = mse

        aux = {
            "loss_total": total,
            "loss_mse": mse,
            "loss_lpips": lpips,
            "keep_prob": keep_prob,
            "loss_mse_pix": mse_pix,
        }

        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # 4) Update
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # 5) Put params back into variables for next step
    new_enc_vars, new_dec_vars = unpack_mae_params(new_params, enc_vars, dec_vars)
    return new_params, opt_state, new_enc_vars, new_dec_vars, aux

def run(cfg: TokenizerTrainConfig):
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"[setup] writing artifacts to: {run_dir.resolve()}")

    cfg, mngr, start_step = setup_experiment_checkpointing(cfg, run_dir)

    # Populate derived attributes (n_patches etc)
    cfg.model.compute_derived(cfg.dataset)

    # Iniialize wandb if enabled
    if cfg.wandb.enabled:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.experiment.run_name,
            config=asdict(cfg),
            dir=str(run_dir),
            resume="allow",
        )

    rng = jax.random.PRNGKey(0)

    # instantiate once
    global lpips_loss_fn
    if cfg.experiment.lpips_weight > 0.0:
        lpips_loss_fn = LPIPS(pretrained_network="alexnet")

    # data
    _next_batch = make_iterator(
        cfg.dataset.B, cfg.dataset.T, cfg.dataset.H, cfg.dataset.W, cfg.dataset.C, 
        cfg.dataset.pixels_per_step, cfg.dataset.size_min, cfg.dataset.size_max, 
        cfg.dataset.hold_min, cfg.dataset.hold_max
    )
    def next_batch(rng):
        rng, (videos, actions, rewards) = _next_batch(rng)
        return rng, videos

    rng, batch_rng = jax.random.split(rng)
    rng, first_batch = next_batch(rng)  # warmup

    encoder, decoder = create_tokenizer_models(cfg.model)
    first_patches = temporal_patchify(first_batch, cfg.model.patch_size)
    input_shape = first_patches.shape
    rng, enc_vars, dec_vars = init_tokenizer_vars(
        encoder, decoder, input_shape=input_shape, rng=rng
    )

    # optim
    params = pack_mae_params(enc_vars, dec_vars)
    
    # Create optimizer
    optimizer = optax.adamw(cfg.experiment.optimizer.lr)
    opt_state = optimizer.init(params)

    # Restore weights if resuming
    if start_step > 0:
        params, opt_state, rng = load_snapshot_weights(
            mngr, start_step, params, opt_state, rng
        )
        # Update enc_vars and dec_vars with restored params
        enc_vars, dec_vars = unpack_mae_params(params, enc_vars, dec_vars)
        print(f"[restore] Resumed from checkpoint at step {start_step}")

    # Metrics
    try:
        logger = MetricLogger(
            use_wandb=cfg.wandb.enabled,
            log_every=cfg.experiment.log_every,
            max_steps=cfg.experiment.optimizer.max_steps,
            wandb_obj=wandb,
        )

        # Training Loop
        pbar = tqdm(range(start_step, cfg.experiment.optimizer.max_steps + 1), 
                    initial=start_step, 
                    total=cfg.experiment.optimizer.max_steps, 
                    desc="Training Tokenizer", 
                    dynamic_ncols=True)
        
        for step in pbar:
            # use a fixed batch for debugging
            # _, batch = next_batch(jax.random.PRNGKey(0))
            data_start_t = time.perf_counter()
            rng, batch = next_batch(rng)
            data_t = time.perf_counter() - data_start_t

            rng, master_key = jax.random.split(rng)
            train_start_t = time.perf_counter()
            should_log = logger.should_log(step)
            params, opt_state, enc_vars, dec_vars, aux = train_step(
                encoder, decoder, optimizer, params, opt_state, enc_vars, dec_vars, batch,
                patch=cfg.model.patch_size, H=cfg.dataset.H, W=cfg.dataset.W, C=cfg.dataset.C, 
                master_key=master_key, step=step, 
                lpips_weight=cfg.experiment.lpips_weight, lpips_frac=cfg.experiment.lpips_frac,
                should_log=should_log,
                dataset_mean=cfg.dataset.dataset_mean, dataset_std=cfg.dataset.dataset_std,
            )
            train_t = time.perf_counter() - train_start_t
            total_t = data_t + train_t

            # Log
            if should_log:
                mse_loss_pix = aux['loss_mse_pix']
                psnr = 10 * jnp.log10(1.0 / jnp.maximum(mse_loss_pix, 1e-10))
                rmse = jnp.sqrt(mse_loss_pix)
                
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

            # Save (async)
            maybe_save_snapshot(mngr, step, params, opt_state, rng, cfg)

            # Viz
            if cfg.experiment.visualize_every > 0 and step % cfg.experiment.visualize_every == 0:
                rng, viz_key = jax.random.split(rng)
                mae_key, drop_key, vis_batch_key = jax.random.split(viz_key, 3)
                _, viz_batch = next_batch(vis_batch_key)
                viz_batch = viz_batch[:8, :1]
                out = viz_step(encoder, decoder, enc_vars, dec_vars, viz_batch,
                               patch=cfg.model.patch_size, mae_key=mae_key, drop_key=drop_key,
                               dataset_mean=cfg.dataset.dataset_mean, dataset_std=cfg.dataset.dataset_std)
                target = jnp.concatenate(temporal_unpatchify(out["target"], cfg.dataset.H, cfg.dataset.W, cfg.dataset.C, cfg.model.patch_size).squeeze(), axis=1)
                masked_in = jnp.concatenate(temporal_unpatchify(out["masked_input"], cfg.dataset.H, cfg.dataset.W, cfg.dataset.C, cfg.model.patch_size).squeeze(), axis=1)
                rec_masked = jnp.concatenate(temporal_unpatchify(out["recon_masked"], cfg.dataset.H, cfg.dataset.W, cfg.dataset.C, cfg.model.patch_size).squeeze(), axis=1)
                rec_unmasked = jnp.concatenate(temporal_unpatchify(out["recon_full"], cfg.dataset.H, cfg.dataset.W, cfg.dataset.C, cfg.model.patch_size).squeeze(), axis=1)
                grid = jnp.concatenate([target, masked_in, rec_masked, rec_unmasked])
                grid = jnp.asarray(grid * 255.0, dtype=jnp.uint8)
                vis_path = run_dir / "viz"
                _ensure_dir(vis_path)
                imageio.imwrite(vis_path / f"step_{step:03d}.png", grid)
    finally:
        # Make sure any background saves finish before exit.
        mngr.wait_until_finished()
        if cfg.wandb.enabled and wandb.run is not None:
            wandb.finish()
            print("[wandb] Finished logging.")

@hydra.main(version_base=None, config_path="../configs", config_name="tokenizer")
def main(cfg: DictConfig):
    schema = OmegaConf.structured(TokenizerTrainConfig)
    cfg = OmegaConf.merge(schema, cfg)
    tokenizer_cfg = OmegaConf.to_object(cfg)
    
    run(tokenizer_cfg)

if __name__ == "__main__":
    main()
