from functools import partial
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax
from dreamer.models import Encoder, Decoder
from dreamer.data import make_iterator
import imageio
from jaxlpips import LPIPS
from pathlib import Path
from dreamer.utils import temporal_patchify, temporal_unpatchify, make_state, make_manager, try_restore, maybe_save, pack_mae_params, unpack_mae_params
from dreamer.logging import MetricLogger



def init_models(rng, encoder, decoder, patch_tokens, B, T, enc_n_latents, enc_d_bottleneck):
    rng, params_rng, mae_rng, dropout_rng = jax.random.split(rng, 4)

    enc_vars = encoder.init(
        {"params": params_rng, "mae": mae_rng, "dropout": dropout_rng},
        patch_tokens, deterministic=True
    )
    fake_z = jnp.zeros((B, T, enc_n_latents, enc_d_bottleneck))
    dec_vars = decoder.init(
        {"params": params_rng, "dropout": dropout_rng},
        fake_z, deterministic=True
    )
    return rng, enc_vars, dec_vars

# --- forward (no jit; we jit the train_step) ---
def forward_apply(encoder, decoder, enc_vars, dec_vars, patches_btnd, *, mae_key, drop_key, train: bool):
    # Avoid TracerBool issues: pass a python bool here OR replace with lax.cond if needed.
    rngs_enc = {"mae": mae_key} if not train else {"mae": mae_key, "dropout": drop_key}
    z_btLd, mae_info = encoder.apply(enc_vars, patches_btnd, rngs=rngs_enc, deterministic=not train)

    rngs_dec = {} if not train else {"dropout": drop_key}
    pred_btnd = decoder.apply(dec_vars, z_btLd, rngs=rngs_dec, deterministic=not train)
    return pred_btnd, mae_info  # mae_info = (mae_mask, keep_prob)

# --- loss ---
def recon_loss_from_mae(pred_btnd, patches_btnd, mae_mask):
    masked_pred   = jnp.where(mae_mask, pred_btnd, 0.0)
    masked_target = jnp.where(mae_mask, patches_btnd, 0.0)
    num = jnp.maximum(mae_mask.sum(), 1)
    return jnp.sum((masked_pred - masked_target) ** 2) / (num * pred_btnd.shape[-1])

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
def viz_step(encoder, decoder, enc_vars, dec_vars, batch, *, patch, mae_key, drop_key):
    # Same preprocessing as train
    patches_btnd = temporal_patchify(batch, patch)  # (B,T,Np,D)

    # Run full model (no dropout during viz)
    pred_btnd, (mae_mask_btNp1, keep_prob_bt1) = forward_apply(
        encoder, decoder, enc_vars, dec_vars, patches_btnd,
        mae_key=mae_key, drop_key=drop_key, train=False
    )

    # Compose standard MAE visualization:
    # - masked_input: what the model actually sees (masked patches)
    # - recon_masked: inpaint only masked patches (visible patches kept as GT)
    masked_input_btnd  = jnp.where(mae_mask_btNp1, 0.0, patches_btnd)
    recon_masked_btnd  = jnp.where(mae_mask_btNp1, pred_btnd, patches_btnd)
    recon_full_btnd    = pred_btnd  # decoder everywhere

    return {
        "target": patches_btnd,
        "masked_input": masked_input_btnd,
        "recon_masked": recon_masked_btnd,
        "recon_full": recon_full_btnd,
        "mae_mask": mae_mask_btNp1,
        "keep_prob": keep_prob_bt1,
    }


# --- train step ---
@partial(jax.jit, static_argnames=("encoder","decoder","tx","patch","H","W","C", "lpips_weight", "lpips_frac"))
def train_step(encoder, decoder, tx, params, opt_state, enc_vars, dec_vars, batch, *,
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
    patches_btnd = temporal_patchify(batch, patch)  # (B,T,Np,Dp)

    # 2) Make per-step RNGs (fold_in ensures different masks per step even if base key repeats)
    step_key  = jax.random.fold_in(master_key, step)
    mae_key, drop_key = jax.random.split(step_key)

    # 3) Define loss fn (closes over encoder/decoder + non-param states)
    def loss_fn(packed_params):
        # Replace params in vars
        ev, dv = unpack_mae_params(packed_params, enc_vars, dec_vars)
        pred, mae_info = forward_apply(encoder, decoder, ev, dv, patches_btnd,
                                       mae_key=mae_key, drop_key=drop_key, train=True)
        mae_mask, keep_prob = mae_info
        mse = recon_loss_from_mae(pred, patches_btnd, mae_mask)

        # LPIPS on recon_masked vs target (unpatchified frames)
        if lpips_weight > 0.0:
            lpips = lpips_on_mae_recon(
                pred, patches_btnd, mae_mask,
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
        }

        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # 4) Update
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # 5) Put params back into variables for next step
    new_enc_vars, new_dec_vars = unpack_mae_params(new_params, enc_vars, dec_vars)
    return new_params, opt_state, new_enc_vars, new_dec_vars, aux

if __name__ == "__main__":

    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = "test"
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)


    rng = jax.random.PRNGKey(0)
    # dataset parameters
    B, T, H, W, C = 32, 64, 32, 32, 3
    pixels_per_step = 2 # how many pixels the agent moves per step
    size_min = 6 # minimum size of the square
    size_max = 14 # maximum size of the square
    hold_min = 4 # how long the agent holds a direction for
    hold_max = 9 # how long the agent holds a direction for

    patch = 4
    num_patches = (H // patch) * (W // patch)
    D_patch = patch * patch * C

    # losses and optimization
    lpips_weight = 0.2
    if lpips_weight > 0.0:
        lpips_loss_fn = LPIPS(pretrained_network="alexnet")  # or "vgg", "squeeze"
    lpips_frac = 0.5

    # data
    _next_batch = make_iterator(B, T, H, W, C, pixels_per_step, size_min, size_max, hold_min, hold_max)
    def next_batch(rng):
        rng, (videos, actions, rewards) = _next_batch(rng)
        return rng, videos

    rng, batch_rng = jax.random.split(rng)
    rng, first_batch = next_batch(rng)  # warmup

    # models
    enc_n_latents, enc_d_bottleneck = 16, 32
    enc_kwargs = {
        "d_model": 64, "n_latents": enc_n_latents, "n_patches": num_patches, "n_heads": 4, "depth": 8, "dropout": 0.05,
        "d_bottleneck": enc_d_bottleneck, "mae_p_min": 0.0, "mae_p_max": 0.9, "time_every": 4,
    }
    dec_kwargs = {
        "d_model": 64, "n_heads": 4, "n_patches": num_patches, "n_latents": enc_n_latents, "depth": 8,
        "d_patch": D_patch, "dropout": 0.05, "time_every": 4,
    }
    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)

    first_patches = temporal_patchify(first_batch, patch)
    rng, enc_vars, dec_vars = init_models(rng, encoder, decoder, first_patches, B, T, enc_n_latents, enc_d_bottleneck)

    # optim
    params = pack_mae_params(enc_vars, dec_vars)
    tx = optax.adamw(1e-4)
    opt_state = tx.init(params)
    max_steps = 1_000_000_000

    # ---------- ORBAX: manager + (optional) restore ----------
    ckpt_dir = run_dir / "checkpoints"
    mngr = make_manager(ckpt_dir, max_to_keep=5, save_interval_steps=10_000)

    # Build example trees for safe restore (use live shapes/dtypes).
    state_example = make_state(params, opt_state, rng, step=0)
    meta_example = {"enc_kwargs": enc_kwargs, "dec_kwargs": dec_kwargs,
                    "H": H, "W": W, "C": C, "patch": patch}

    restored = try_restore(mngr, state_example, meta_example)
    start_step = 0
    if restored is not None:
        latest_step, r = restored
        # Unpack state back to your locals
        params = r.state["params"]
        opt_state = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"])
        # Optional: you can read r.meta here if you want to sanity-check config.

        # Rebuild enc_vars/dec_vars from params so downstream apply() uses the restored params.
        enc_vars, dec_vars = unpack_mae_params(params, enc_vars, dec_vars)
        print(f"Restored checkpoint at step {latest_step} from {ckpt_dir}")

    # ---------- Train loop ----------
    try:
        logger = MetricLogger(use_wandb=False, log_every=100, max_steps=max_steps) # FIXME: add wandb, add cfg
        pbar = tqdm(range(start_step, max_steps), 
                    initial=start_step, 
                    total=max_steps, 
                    desc="Training Tokenizer", 
                    dynamic_ncols=True)
        
        for step in pbar:
            # use a fixed batch for debugging
            # _, batch = next_batch(jax.random.PRNGKey(0))
            rng, batch = next_batch(rng)
            rng, master_key = jax.random.split(rng)
            params, opt_state, enc_vars, dec_vars, aux = train_step(
                encoder, decoder, tx, params, opt_state, enc_vars, dec_vars, batch,
                patch=patch, H=H, W=W, C=C, master_key=master_key, step=step, lpips_weight=lpips_weight, lpips_frac=lpips_frac,
            )

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
                        "psnr": psnr
                    },
                    pbar=pbar,
                    float_fmt=".6f"
                )

            # Save (async)
            state = make_state(params, opt_state, rng, step)
            maybe_save(mngr, step, state, meta_example)

            # Viz
            if step % 10000 == 0:
                rng, viz_key = jax.random.split(rng)
                mae_key, drop_key, vis_batch_key = jax.random.split(viz_key, 3)
                _, viz_batch = next_batch(vis_batch_key)
                viz_batch = viz_batch[:8, :1]
                out = viz_step(encoder, decoder, enc_vars, dec_vars, viz_batch,
                               patch=patch, mae_key=mae_key, drop_key=drop_key)
                target = jnp.concatenate(temporal_unpatchify(out["target"], H, W, C, patch).squeeze(), axis=1)
                masked_in = jnp.concatenate(temporal_unpatchify(out["masked_input"], H, W, C, patch).squeeze(), axis=1)
                rec_masked  = jnp.concatenate(temporal_unpatchify(out["recon_masked"], H, W, C, patch).squeeze(), axis=1)
                rec_unmasked  = jnp.concatenate(temporal_unpatchify(out["recon_full"], H, W, C, patch).squeeze(), axis=1)
                grid = jnp.concatenate([target, masked_in, rec_masked, rec_unmasked])
                grid = jnp.asarray(grid * 255.0, dtype=jnp.uint8)
                imageio.imwrite(run_dir / f"step_{step:03d}.png", grid)
    finally:
        # Make sure any background saves finish before exit.
        mngr.wait_until_finished()
