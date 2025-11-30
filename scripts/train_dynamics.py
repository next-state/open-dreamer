# train_dynamics.py
# Streaming-batch training on synthetic data with teacher-forced training and autoregressive evaluation.
# This version keeps ONLY the efficient training step and adds robust Orbax checkpointing.
# It restores the pretrained tokenizer (enc/dec) and trains the dynamics model.

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any
from functools import partial
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
import json
import time
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import imageio.v2 as imageio
import orbax.checkpoint as ocp
import wandb

from dreamer.models import Encoder, Decoder, Dynamics
from dreamer.data import make_iterator, DatasetConfig
from dreamer.utils import (
    temporal_patchify,
    pack_bottleneck_to_spatial,
    with_params,
    make_state, make_manager, try_restore, maybe_save,
    pack_mae_params,
)
from dreamer.logging import MetricLogger

from dreamer.sampler import SamplerConfig, sample_video

# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class RealismConfig:
    # IO / ckpt
    run_name: str
    tokenizer_ckpt: str
    ckpt_max_to_keep: int = 2
    ckpt_save_every: int = 10_000

    # wandb config
    use_wandb: bool = False
    wandb_entity: str | None = None  # if None, uses default entity
    wandb_project: str | None = None  # if None, uses run_name as project

    # dataset config
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # tokenizer / dynamics config
    patch: int = 4
    enc_n_latents: int = 16
    enc_d_bottleneck: int = 32
    d_model_enc: int = 64
    d_model_dyn: int = 128
    enc_depth: int = 8
    dec_depth: int = 8
    dyn_depth: int = 8
    n_heads: int = 4
    packing_factor: int = 2
    n_register: int = 4 # number of register tokens for dynamics
    n_agent: int = 1 # number of agent tokens for dynamics
    agent_space_mode: str = "wm_agent_isolated"

    # schedule
    k_max: int = 8
    bootstrap_start: int = 5_000  # warm-up steps with bootstrap masked out
    self_fraction: float = 0.25   # used once we pass bootstrap_start

    # train
    max_steps: int = 1_000_000_000
    log_every: int = 5_000
    lr: float = 3e-4

    # eval media toggle
    write_video_every: int = 10_000  # set large to reduce IO, or 0 to disable entirely

# ---------------------------
# Small helpers
# ---------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_uint8(img_f32):
    return np.asarray(np.clip(np.asarray(img_f32) * 255.0, 0, 255), dtype=np.uint8)

def _stack_wide(*imgs_hwC):
    return np.concatenate(imgs_hwC, axis=1)

def _tile_videos(trip_list_hwC: list[np.ndarray], *, ncols: int = 2, pad_color: int = 0) -> np.ndarray:
    if len(trip_list_hwC) == 0:
        raise ValueError("Empty video list")
    H, W3, C = trip_list_hwC[0].shape
    B = len(trip_list_hwC)
    nrows = math.ceil(B / ncols)
    total = nrows * ncols
    if total > B:
        blank = np.full((H, W3, C), pad_color, dtype=trip_list_hwC[0].dtype)
        trip_list_hwC = trip_list_hwC + [blank] * (total - B)
    rows = []
    idx = 0
    for _ in range(nrows):
        row_imgs = trip_list_hwC[idx:idx + ncols]
        idx += ncols
        rows.append(np.concatenate(row_imgs, axis=1))
    grid = np.concatenate(rows, axis=0)
    return grid

# ---------------------------
# Tokenizer restore (uses your Orbax layout & utils)
# ---------------------------

def load_pretrained_tokenizer(
    tokenizer_ckpt_dir: str,
    *,
    rng: jnp.ndarray,
    encoder: Encoder,
    decoder: Decoder,
    enc_vars,
    dec_vars,
    sample_patches_btnd,
):
    meta_mngr = make_manager(tokenizer_ckpt_dir, item_names=("meta",))
    latest = meta_mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No tokenizer checkpoint found in {tokenizer_ckpt_dir}")
    restored_meta = meta_mngr.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = restored_meta.meta
    enc_kwargs = meta["enc_kwargs"]
    n_lat, d_b = enc_kwargs["n_latents"], enc_kwargs["d_bottleneck"]

    rng_e1, rng_d1 = jax.random.split(rng)
    B, T = sample_patches_btnd.shape[:2]
    fake_z = jnp.zeros((B, T, n_lat, d_b), dtype=jnp.float32)
    dec_vars = decoder.init({"params": rng_d1, "dropout": rng_d1}, fake_z, deterministic=True)

    packed_example = pack_mae_params(enc_vars, dec_vars)
    tx_dummy = optax.adamw(1e-4)
    opt_state_example = tx_dummy.init(packed_example)
    state_example = make_state(packed_example, opt_state_example, rng_e1, step=0)
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)

    tok_mngr = make_manager(tokenizer_ckpt_dir, item_names=("state", "meta"))
    restored = tok_mngr.restore(
        latest,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            meta=ocp.args.JsonRestore(),
        ),
    )
    packed_params = restored.state["params"]
    enc_params = packed_params["enc"]
    dec_params = packed_params["dec"]
    new_enc_vars = with_params(enc_vars, enc_params)
    new_dec_vars = with_params(dec_vars, dec_params)
    print(f"[tokenizer] Restored encoder/decoder from {tokenizer_ckpt_dir} (step {latest})")
    return new_enc_vars, new_dec_vars, meta

# ---------------------------
# Single efficient training step (always used)
# ---------------------------

@partial(
    jax.jit,
    static_argnames=("encoder","dynamics","tx","patch","n_spatial","k_max","packing_factor","B","T","B_self"),
)
def train_step_efficient(
    encoder, dynamics, tx,
    params, opt_state,
    enc_vars, dyn_vars,
    frames, actions,
    *,
    patch: int,
    B: int, T: int, B_self: int,            # assume 0 <= B_self < B
    n_spatial: int, k_max: int, packing_factor: int,
    master_key: jnp.ndarray, step: int, bootstrap_start: int,
):
    """
    Deterministic two-branch training (one fused main forward):
      - first B_emp rows: empirical flow at d_min = 1/k_max
      - last  B_self rows: bootstrap self-consistency with d > d_min
    If step < bootstrap_start, the bootstrap contribution is masked to 0 (but we still
    execute one fused path to keep a single jit and stable shapes).
    """
    @partial(jax.jit, static_argnames=("shape_bt","k_max",))
    def _sample_tau_for_step(rng, shape_bt, k_max:int, step_idx:jnp.ndarray, *, dtype=jnp.float32):
        B_, T_ = shape_bt
        K = (1 << step_idx)
        u = jax.random.uniform(rng, (B_, T_), dtype=dtype)
        j_idx = jnp.floor(u * K.astype(dtype)).astype(jnp.int32)
        tau = j_idx.astype(dtype) / K.astype(dtype)
        tau_idx = j_idx * (k_max // K)
        return tau, tau_idx

    @partial(jax.jit, static_argnames=("shape_bt","k_max",))
    def _sample_step_excluding_dmin(rng, shape_bt, k_max:int):
        B_, T_ = shape_bt
        emax = jnp.log2(k_max).astype(jnp.int32)
        step_idx = jax.random.randint(rng, (B_, T_), 0, emax, dtype=jnp.int32)  # exclude emax
        d = 1.0 / (1 << step_idx).astype(jnp.float32)
        return d, step_idx

    # ---------- Param-free precompute ----------
    patches_btnd = temporal_patchify(frames, patch)

    # RNGs
    step_key = jax.random.fold_in(master_key, step)
    enc_key, key_sigma_full, key_step_self, key_noise_full, drop_key = jax.random.split(step_key, 5)

    # Frozen encoder → spatial tokens (clean target z1)
    z_bottleneck, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": enc_key}, deterministic=True)
    z1 = pack_bottleneck_to_spatial(z_bottleneck, n_spatial=n_spatial, k=packing_factor)  # (B,T,Sz,Dz)

    # Deterministic batch split
    B_emp  = B - B_self
    actions_full = actions
    emax = jnp.log2(k_max).astype(jnp.int32)

    # --- Step indices (encode d) ---
    step_idx_emp  = jnp.full((B_emp,  T), emax, dtype=jnp.int32)             # d = d_min
    # If B_self == 0, create a dummy 0xT array – slicing below handles it.
    d_self, step_idx_self = _sample_step_excluding_dmin(key_step_self, (B_self, T), k_max)
    step_idx_full = jnp.concatenate([step_idx_emp, step_idx_self], axis=0)   # (B,T)

    # --- Signal levels on each row's grid (one call for whole batch) ---
    sigma_full, sigma_idx_full = _sample_tau_for_step(key_sigma_full, (B, T), k_max, step_idx_full)
    sigma_emp   = sigma_full[:B_emp]
    sigma_self  = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]

    # --- Corrupt inputs: z_tilde = (1 - sigma) z0 + sigma z1 ---
    z0_full      = jax.random.normal(key_noise_full, z1.shape, dtype=z1.dtype)
    z_tilde_full = (1.0 - sigma_full)[...,None,None] * z0_full + sigma_full[...,None,None] * z1
    z_tilde_self = z_tilde_full[B_emp:]

    # --- Ramp weights ---
    w_emp  = 0.9 * sigma_emp  + 0.1
    w_self = 0.9 * sigma_self + 0.1

    # --- Half-step metadata for self rows ---
    d_half            = d_self / 2.0
    step_idx_half     = step_idx_self + 1
    sigma_plus        = sigma_self + d_half
    sigma_idx_plus    = sigma_idx_self + (k_max * d_half).astype(jnp.int32)

    def loss_and_aux(p):
        local_dyn = with_params(dyn_vars, p)
        drop_main, drop_h1, drop_h2 = jax.random.split(drop_key, 3)

        # Main forward (emp + self)
        z1_hat_full, _ = dynamics.apply(
            local_dyn, actions_full, step_idx_full, sigma_idx_full, z_tilde_full,
            rngs={"dropout": drop_main}, deterministic=False,
        )  # (B,T,Sz,Dz)

        z1_hat_emp  = z1_hat_full[:B_emp]
        z1_hat_self = z1_hat_full[B_emp:]

        # Flow loss on empirical rows (to z1)
        flow_per = jnp.mean((z1_hat_emp - z1[:B_emp])**2, axis=(2,3))        # (B_emp,T)
        loss_emp = jnp.mean(flow_per * w_emp)

        # Self-consistency (bootstrap) on self rows
        # If B_self == 0, shapes are 0-sized and reductions become NaN; guard with mask.
        do_boot = (B_self > 0) & (step >= bootstrap_start)

        def _boot_loss():
            z1_hat_half1, _ = dynamics.apply(
                local_dyn, actions_full[B_emp:], step_idx_half, sigma_idx_self, z_tilde_self,
                rngs={"dropout": drop_h1}, deterministic=False,
            )
            b_prime = (z1_hat_half1 - z_tilde_self) / (1.0 - sigma_self)[...,None,None]
            z_prime = z_tilde_self + b_prime * d_half[...,None,None]
            z1_hat_half2, _ = dynamics.apply(
                local_dyn, actions_full[B_emp:], step_idx_half, sigma_idx_plus, z_prime,
                rngs={"dropout": drop_h2}, deterministic=False,
            )
            b_doubleprime = (z1_hat_half2 - z_prime) / (1.0 - sigma_plus)[...,None,None]
            vhat_sigma = (z1_hat_self - z_tilde_self) / (1.0 - sigma_self)[...,None,None]
            vbar_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)
            boot_per = (1.0 - sigma_self)**2 * jnp.mean((vhat_sigma - vbar_target)**2, axis=(2,3))  # (B_self,T)
            loss_self = jnp.mean(boot_per * w_self)
            return loss_self, jnp.mean(boot_per)

        loss_self, boot_mse = jax.lax.cond(
            do_boot,
            _boot_loss,
            lambda: (jnp.array(0.0, dtype=z1.dtype), jnp.array(0.0, dtype=z1.dtype)),
        )

        # Combine (row-weighted by nominal B parts; denominator B keeps scale constant)
        loss = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B

        aux = {
            "flow_mse": jnp.mean(flow_per),
            "bootstrap_mse": boot_mse,
        }
        return loss, aux

    (loss_val, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, aux

# ---------------------------
# Eval regimes & plan JSON (unchanged core logic)
# ---------------------------

def _eval_regimes_for_realism(cfg, *, ctx_length: int):
    common = dict(
        k_max=cfg.k_max,
        horizon=min(32, cfg.dataset.T - ctx_length),
        ctx_length=ctx_length,
        ctx_signal_tau=1.0,   # was 0.99 — make context clean for fair PSNR
        H=cfg.dataset.H, W=cfg.dataset.W, C=cfg.dataset.C, patch=cfg.patch,
        n_spatial=cfg.enc_n_latents // cfg.packing_factor,
        packing_factor=cfg.packing_factor,
        start_mode="pure",
        rollout="autoregressive",
        # optional: see item 3 below
        # match_ctx_tau=False,
    )
    regs = []
    regs.append(("finest_pure_AR", SamplerConfig(schedule="finest", **common)))
    regs.append(("shortcut_d4_pure_AR", SamplerConfig(schedule="shortcut", d=1/4, **common)))
    return regs


def _plan_from_sampler_conf(s: SamplerConfig) -> Dict[str, Any]:
    def _is_pow2_frac(x: float) -> bool:
        if x <= 0 or x > 1: return False
        inv = round(1.0 / x)
        return abs(1.0 / inv - x) < 1e-8 and (inv & (inv - 1)) == 0

    if s.schedule == "finest":
        d = 1.0 / float(s.k_max)
    else:
        if s.d is None or not _is_pow2_frac(s.d):
            raise ValueError("shortcut schedule requires d = 1/(power of two)")
        if s.d < 1.0 / float(s.k_max):
            raise ValueError("d finer than finest")
        d = float(s.d)

    tau0 = 0.0
    S = int(round((1.0 - tau0) / d))
    e = int(round(np.log2(round(1.0 / d))))
    tau_seq = [round(tau0 + i*d, 6) for i in range(S + 1)]
    tau_seq[-1] = 1.0
    return dict(
        rollout=s.rollout,
        start_mode=s.start_mode,
        ctx_length=s.ctx_length,
        horizon=s.horizon,
        schedule=s.schedule,
        d=d,
        e=e,
        S=S,
        tau_seq=tau_seq,
        k_max=s.k_max,
        add_ctx_noise_std=getattr(s, "add_ctx_noise_std", 0.0),
    )

# ---------------------------
# Video building and saving utilities
# ---------------------------

def build_tiled_video_frames(
    gt_frames: jnp.ndarray,
    floor_frames: jnp.ndarray,
    pred_frames: jnp.ndarray,
    batch_size: int,
) -> list[np.ndarray]:
    """
    Build tiled video frames from ground truth, floor, and prediction frames.

    Each frame in the output contains a grid of triplets (GT | Floor | Pred) stacked horizontally,
    with multiple batch items tiled vertically/horizontally.

    Args:
        gt_frames: Ground truth frames (B, T, H, W, C)
        floor_frames: Floor/reference frames (B, T, H, W, C)
        pred_frames: Predicted frames (B, T, H, W, C)
        batch_size: Batch size B

    Returns:
        List of grid frames, one per time step
    """
    gt_np_all = _to_uint8(gt_frames)
    floor_np_all = _to_uint8(floor_frames)
    pred_np_all = _to_uint8(pred_frames)

    T_total = gt_np_all.shape[1]
    ncols = 1 if batch_size <= 2 else min(8, batch_size)
    grid_frames = []

    for t_idx in range(T_total):
        trip_list = [
            _stack_wide(gt_np_all[b, t_idx], floor_np_all[b, t_idx], pred_np_all[b, t_idx])
            for b in range(batch_size)
        ]
        grid_img = _tile_videos(trip_list, ncols=ncols, pad_color=0)
        grid_frames.append(grid_img)

    return grid_frames

def save_evaluation_video(
    grid_frames: list[np.ndarray],
    output_path: Path,
    tag: str,
) -> bool:
    """
    Save grid frames as an MP4 video file.

    Args:
        grid_frames: List of grid frames to write
        output_path: Path where MP4 should be saved
        tag: Tag for error messages

    Returns:
        True if successful, False otherwise
    """
    try:
        with imageio.get_writer(output_path, fps=25, codec="libx264", quality=8) as w:
            for fr in grid_frames:
                w.append_data(fr)
        return True
    except Exception as e:
        print(f"[eval:{tag}] MP4 write skipped ({e})")
        return False

def save_evaluation_plan(
    sampler_conf: SamplerConfig,
    step: int,
    mse: float,
    psnr: float,
    output_path: Path,
):
    """
    Save evaluation plan/metadata as JSON.

    Args:
        sampler_conf: Sampler configuration
        step: Training step number
        mse: Mean squared error
        psnr: Peak signal-to-noise ratio in dB
        output_path: Path where JSON should be saved
    """
    plan = _plan_from_sampler_conf(sampler_conf)
    plan["step"] = int(step)
    plan["mse"] = float(mse)
    plan["psnr_db"] = float(psnr)

    with open(output_path, "w") as f:
        json.dump(plan, f, indent=2)

# ---------------------------
# Meta for dynamics checkpoints
# ---------------------------

def make_dynamics_meta(
    *,
    enc_kwargs: dict,
    dec_kwargs: dict,
    dynamics_kwargs: dict,
    H: int, W: int, C: int,
    patch: int,
    k_max: int,
    packing_factor: int,
    n_spatial: int,
    tokenizer_ckpt_dir: str | None = None,
    cfg: Dict[str, Any] | None = None,
):
    return {
        "enc_kwargs": enc_kwargs,
        "dec_kwargs": dec_kwargs,
        "dynamics_kwargs": dynamics_kwargs,
        "H": H, "W": W, "C": C, "patch": patch,
        "k_max": k_max,
        "packing_factor": packing_factor,
        "n_spatial": n_spatial,
        "tokenizer_ckpt_dir": tokenizer_ckpt_dir,
        "cfg": cfg or {},
    }

# ---------------------------
# Training state dataclass
# ---------------------------

@dataclass
class TrainState:
    """Container for all training-related state (models, variables, optimizer, etc.)."""
    encoder: Encoder
    decoder: Decoder
    dynamics: Dynamics
    enc_vars: dict
    dec_vars: dict
    dyn_vars: dict
    params: dict
    enc_kwargs: dict
    dec_kwargs: dict
    dyn_kwargs: dict
    tx: optax.Transform
    opt_state: optax.OptState
    mae_eval_key: jnp.ndarray

# ---------------------------
# Model initialization
# ---------------------------

def initialize_models_and_tokenizer(
    cfg: RealismConfig,
    frames_init: jnp.ndarray,
    actions_init: jnp.ndarray,
) -> TrainState:
    """
    Initialize encoder, decoder, dynamics models and restore tokenizer.

    Returns:
        TrainState containing all initialized models, variables, and optimizer state.
    """
    patch = cfg.patch
    num_patches = (cfg.dataset.H // patch) * (cfg.dataset.W // patch)
    D_patch = patch * patch * cfg.dataset.C
    k_max = cfg.k_max

    enc_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        n_heads=cfg.n_heads,
        depth=cfg.enc_depth,
        dropout=0.0,
        d_bottleneck=cfg.enc_d_bottleneck,
        mae_p_min=0.0, mae_p_max=0.0,
        time_every=4, latents_only_time=True,
    )
    dec_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_heads=cfg.n_heads,
        depth=cfg.dec_depth,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        d_patch=D_patch,
        dropout=0.0,
        mlp_ratio=4.0, time_every=4, latents_only_time=True,
    )
    n_spatial = cfg.enc_n_latents // cfg.packing_factor # number of spatial tokens for dynamics
    dyn_kwargs = dict(
        d_model=cfg.d_model_dyn,
        d_bottleneck=cfg.enc_d_bottleneck,
        d_spatial=cfg.enc_d_bottleneck * cfg.packing_factor,
        n_spatial=n_spatial, n_register=cfg.n_register,
        n_heads=cfg.n_heads, depth=cfg.dyn_depth,
        space_mode=cfg.agent_space_mode, n_agent=cfg.n_agent,
        dropout=0.0, k_max=k_max, 
        time_every=4,
    )

    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)
    dynamics = Dynamics(**dyn_kwargs)

    patches_btnd = temporal_patchify(frames_init, patch)
    rng = jax.random.PRNGKey(0)
    enc_vars = encoder.init({"params": rng, "mae": rng, "dropout": rng}, patches_btnd, deterministic=True)
    fake_z = jnp.zeros((cfg.dataset.B, cfg.dataset.T, cfg.enc_n_latents, cfg.enc_d_bottleneck))
    dec_vars = decoder.init({"params": rng, "dropout": rng}, fake_z, deterministic=True)

    # Restore tokenizer
    enc_vars, dec_vars, _ = load_pretrained_tokenizer(
        cfg.tokenizer_ckpt, rng=rng,
        encoder=encoder, decoder=decoder,
        enc_vars=enc_vars, dec_vars=dec_vars,
        sample_patches_btnd=patches_btnd,
    )

    # Build initial z1 to shape dynamics init
    mae_eval_key = jax.random.PRNGKey(777)
    z_btLd, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": mae_eval_key}, deterministic=True)
    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=cfg.packing_factor)
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jnp.full((cfg.dataset.B, cfg.dataset.T), emax, dtype=jnp.int32)
    sigma_idx = jnp.full((cfg.dataset.B, cfg.dataset.T), k_max - 1, dtype=jnp.int32)
    dyn_vars = dynamics.init({"params": rng, "dropout": rng}, actions_init, step_idx, sigma_idx, z1)
    params = dyn_vars["params"]

    tx = optax.adam(cfg.lr)
    opt_state = tx.init(params)

    return TrainState(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        enc_vars=enc_vars,
        dec_vars=dec_vars,
        dyn_vars=dyn_vars,
        params=params,
        enc_kwargs=enc_kwargs,
        dec_kwargs=dec_kwargs,
        dyn_kwargs=dyn_kwargs,
        tx=tx,
        opt_state=opt_state,
        mae_eval_key=mae_eval_key,
    )

# ---------------------------
# Evaluation logic
# ---------------------------

def run_evaluation(
    cfg: RealismConfig,
    step: int,
    train_state: TrainState,
    next_batch,
    vis_dir: Path,
):
    """
    Run periodic evaluation: sample videos, compute metrics, and save visualization.

    Args:
        cfg: Configuration object
        step: Current training step
        train_state: TrainState containing all models, variables, and optimizer state
        next_batch: Data iterator function
        vis_dir: Directory for visualization outputs
    """
    val_rng = jax.random.PRNGKey(9999)
    _, (val_frames, val_actions, _) = next_batch(val_rng)
    dyn_vars_eval = with_params(train_state.dyn_vars, train_state.params)
    ctx_length = min(32, cfg.dataset.T - 1)
    regimes = _eval_regimes_for_realism(cfg, ctx_length=ctx_length)

    for tag, sampler_conf in regimes:
        sampler_conf.mae_eval_key = train_state.mae_eval_key
        sampler_conf.rng_key = jax.random.PRNGKey(4242)
        t0 = time.time()

        pred_frames, floor_frames, gt_frames = sample_video(
            encoder=train_state.encoder,
            decoder=train_state.decoder,
            dynamics=train_state.dynamics,
            enc_vars=train_state.enc_vars,
            dec_vars=train_state.dec_vars,
            dyn_vars=dyn_vars_eval,
            frames=val_frames, actions=val_actions, config=sampler_conf,
        )

        dt = time.time() - t0
        HZ = sampler_conf.horizon
        mse = float(jnp.mean((pred_frames[:, -HZ:] - gt_frames[:, -HZ:]) ** 2))
        psnr = float(10.0 * jnp.log10(1.0 / jnp.maximum(mse, 1e-12)))
        print(f"[eval:{tag}] step={step:06d} | AR_hz={HZ} | MSE={mse:.6g} | PSNR={psnr:.2f} dB | {dt:.2f}s")

        # Build tiled video frames
        grid_frames = build_tiled_video_frames(
            gt_frames=gt_frames,
            floor_frames=floor_frames,
            pred_frames=pred_frames,
            batch_size=cfg.dataset.B,
        )

        # Save video and plan
        tag_dir = _ensure_dir(vis_dir / f"step_{step:06d}")
        mp4_path = tag_dir / f"{tag}_grid.mp4"
        plan_path = tag_dir / f"{tag}_plan.json"

        save_evaluation_video(grid_frames, mp4_path, tag)
        save_evaluation_plan(sampler_conf, step, mse, psnr, plan_path)

        print(f"[eval:{tag}] wrote {mp4_path.name} and {plan_path.name} in {tag_dir}")

        # Log to wandb
        if cfg.use_wandb and wandb.run is not None:
            # Log metrics
            wandb.log({
                f"eval/{tag}/mse": mse,
                f"eval/{tag}/psnr": psnr,
                f"eval/{tag}/horizon": HZ,
                f"eval/{tag}/eval_time": dt,
            }, step=step)
            if grid_frames:
                wandb.log({
                    f"eval/{tag}/video": wandb.Video(mp4_path, format="mp4"),
                }, step=step)

# ---------------------------
# Main
# ---------------------------

def run(cfg: RealismConfig):
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
    vis_dir = _ensure_dir(run_dir / "viz")
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

    # Data iterator (streaming)
    next_batch = make_iterator(
        cfg.dataset.B, cfg.dataset.T, cfg.dataset.H, cfg.dataset.W, cfg.dataset.C,
        pixels_per_step=cfg.dataset.pixels_per_step,
        size_min=cfg.dataset.size_min, size_max=cfg.dataset.size_max,
        hold_min=cfg.dataset.hold_min, hold_max=cfg.dataset.hold_max,
        fg_min_color=0 if cfg.dataset.diversify_data else 128,
        fg_max_color=255 if cfg.dataset.diversify_data else 128,
        bg_min_color=0 if cfg.dataset.diversify_data else 255,
        bg_max_color=255 if cfg.dataset.diversify_data else 255,
    )

    # Initialize models and restore tokenizer
    init_rng = jax.random.PRNGKey(0)
    _, (frames_init, actions_init, _) = next_batch(init_rng)

    train_state = initialize_models_and_tokenizer(cfg, frames_init, actions_init)

    # Extract some values for checkpointing
    patch = cfg.patch
    k_max = cfg.k_max
    n_spatial = cfg.enc_n_latents // cfg.packing_factor

    # -------- Orbax manager & (optional) restore --------
    mngr = make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every)
    meta = make_dynamics_meta(
        enc_kwargs=train_state.enc_kwargs,
        dec_kwargs=train_state.dec_kwargs,
        dynamics_kwargs=train_state.dyn_kwargs,
        H=cfg.dataset.H, W=cfg.dataset.W, C=cfg.dataset.C, patch=patch,
        k_max=k_max, packing_factor=cfg.packing_factor, n_spatial=n_spatial,
        tokenizer_ckpt_dir=cfg.tokenizer_ckpt,
        cfg=asdict(cfg),
    )

    rng = jax.random.PRNGKey(0)
    state_example = make_state(train_state.params, train_state.opt_state, rng, step=0)
    restored = try_restore(mngr, state_example, meta)

    start_step = 0
    if restored is not None:
        latest_step, r = restored
        train_state.params = r.state["params"]
        train_state.opt_state = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"]) + 1
        train_state.dyn_vars = with_params(train_state.dyn_vars, train_state.params)
        print(f"[restore] Resumed from {ckpt_dir} at step={latest_step}")

    # -------- Training loop --------
    train_rng = jax.random.PRNGKey(2025)
    data_rng = jax.random.PRNGKey(12345)

    logger = MetricLogger(
        use_wandb=cfg.use_wandb,
        log_every=cfg.log_every,
        max_steps=cfg.max_steps,
        wandb_obj=wandb,
    )

    pbar = tqdm(range(start_step, cfg.max_steps + 1), 
                initial=start_step,
                total=cfg.max_steps,
                desc="Training Dynamics",
                dynamic_ncols=True)
    
    for step in pbar:
        # Data
        data_start_t = time.perf_counter()
        data_rng, batch_key = jax.random.split(data_rng)
        _, (frames, actions, _) = next_batch(batch_key)
        data_t = time.perf_counter() - data_start_t

        # RNG for this step
        train_rng, master_key = jax.random.split(train_rng)

        # Decide current B_self based on warm-up (static arg requires a single value; we keep B_self fixed
        # and gate its contribution inside the jit via bootstrap_start masking).
        B_self = max(0, int(round(cfg.self_fraction * cfg.dataset.B)))

        train_start_t = time.perf_counter()
        train_state.params, train_state.opt_state, aux = train_step_efficient(
            train_state.encoder, train_state.dynamics, train_state.tx,
            train_state.params, train_state.opt_state,
            train_state.enc_vars, train_state.dyn_vars,
            frames, actions,
            patch=cfg.patch, B=cfg.dataset.B, T=cfg.dataset.T, B_self=B_self,
            n_spatial=n_spatial, k_max=k_max, packing_factor=cfg.packing_factor,
            master_key=master_key, step=step, bootstrap_start=cfg.bootstrap_start,
        )
        train_t = time.perf_counter() - train_start_t
        total_t = data_t + train_t

        # Logging
        if logger.should_log(step):
            logger.log(
                step,
                metrics={
                    "flow_mse": aux["flow_mse"],
                    "boot_mse": aux["bootstrap_mse"],
                    "time/data": data_t,
                    "time/train": train_t,
                    "time/total": total_t,
                },
                pbar=pbar
            )

        # Save (async) when policy says we should
        state = make_state(train_state.params, train_state.opt_state, train_rng, step)
        maybe_save(mngr, step, state, meta)

        # Periodic lightweight AR eval
        if cfg.write_video_every and (step % cfg.write_video_every == 0):
            run_evaluation(
                cfg=cfg,
                step=step,
                train_state=train_state,
                next_batch=next_batch,
                vis_dir=vis_dir,
            )

    # Ensure all writes finished
    mngr.wait_until_finished()

    # Finish wandb run
    if cfg.use_wandb and wandb.run is not None:
        wandb.finish()
        print("[wandb] Finished logging.")


@hydra.main(version_base=None, config_path="../configs", config_name="dynamics")
def main(cfg: DictConfig):
    schema = OmegaConf.structured(RealismConfig)
    cfg = OmegaConf.merge(schema, cfg)
    realism_cfg = OmegaConf.to_object(cfg)
    
    run(realism_cfg)


if __name__ == "__main__":
    main()
