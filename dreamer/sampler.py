# sampling logic for debugging / visualization. Not JIT friendly.
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Dict, Any, Callable
from einops import reduce

import jax
import jax.numpy as jnp
import numpy as np

from dreamer.models import Encoder, Decoder, Dynamics, TaskEmbedder, PolicyHeadMTP
from dreamer.utils import (
    temporal_patchify, temporal_unpatchify,
    pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck,
)

# ---------------------------
# Config & small utilities
# ---------------------------

StartMode   = Literal["pure", "fixed", "random"]
Schedule    = Literal["finest", "shortcut"]
RolloutMode = Literal["teacher_forced", "autoregressive"]

@dataclass
class SamplerConfig:
    k_max: int
    schedule: Schedule                      # "finest" or "shortcut"
    d: Optional[float] = None               # used iff schedule == "shortcut"
    start_mode: StartMode = "pure"          # in TF: {"pure","fixed","random"}; in AR: must be "pure"
    tau0_fixed: float = 0.5                 # used iff start_mode == "fixed"
    rollout: RolloutMode = "teacher_forced" # "teacher_forced" or "autoregressive"
    horizon: int = 1
    ctx_length: int = 32
    ctx_signal_tau: Optional[float] = None  # e.g., 0.9 for slightly corrupt viz; None/1.0 = clean
    match_ctx_tau: bool = False             # NEW: corrupt context to current τ each step (uses fixed z0_ctx)

    rng_key: Optional[jax.Array] = None
    mae_eval_key: Optional[jax.Array] = None
    # decoding sizes
    H: int = 32; W: int = 32; C: int = 3; patch: int = 4
    # tokenizer shapes
    n_spatial: int = 8
    packing_factor: int = 2
    # debugging (host-side only)
    debug: bool = False
    # Called with a dict. We call it twice: once with a high-level run "plan" (kind="plan"),
    # and once per step for the scalar fields (kind="step").
    debug_hook: Optional[Callable[[dict], None]] = None

def _assert_power_of_two(k: int):
    if k < 1 or (k & (k - 1)) != 0:
        raise ValueError(f"k_max must be a positive power of two, got {k}")

def _is_power_of_two_fraction(x: float) -> bool:
    if x <= 0 or x > 1: return False
    inv = round(1.0 / x)
    return abs(1.0 / inv - x) < 1e-8 and (inv & (inv - 1)) == 0

def _step_idx_from_d(d: float, k_max: int) -> int:
    K = round(1.0 / float(d))
    if abs(1.0 / K - d) > 1e-8:
        raise ValueError(f"d={d} is not an exact 1/(power of two)")
    e = int(round(np.log2(K)))
    emax = int(round(np.log2(k_max)))
    if e > emax:
        raise ValueError(f"step bin e={e} (d={d}) is coarser than allowed emax={emax} (k_max={k_max})")
    return e

def _choose_step_size(k_max: int, schedule: Schedule, d: Optional[float]) -> float:
    _assert_power_of_two(k_max)
    if schedule == "finest":
        return 1.0 / float(k_max)
    if d is None:
        raise ValueError("schedule='shortcut' requires config.d (e.g., 1/4)")
    if not _is_power_of_two_fraction(d):
        raise ValueError(f"d must be 1/(power of two); got d={d}")
    if d < 1.0 / float(k_max):
        raise ValueError(f"d={d} is finer than d_min=1/{k_max}")
    return float(d)

def _align_to_grid(tau0: float, d: float) -> float:
    # Snap tau0 to the {0, d, 2d, ...} grid. No-op if already aligned.
    return float(np.clip(np.floor(tau0 / d) * d, 0.0, 1.0))

def _signal_idx_from_tau(tau: jnp.ndarray, k_max: int) -> jnp.ndarray:
    idx = (tau * k_max).astype(jnp.int32)
    return jnp.clip(idx, 0, k_max - 1)

def _validate_modes(cfg: SamplerConfig):
    # AR ⇒ start must be "pure"
    if cfg.rollout == "autoregressive" and cfg.start_mode != "pure":
        raise ValueError("Autoregressive rollout supports only start_mode='pure'. "
                         "Use teacher_forced for fixed/random starts.")
    # Finest vs shortcut d usage
    if cfg.schedule == "finest" and cfg.d is not None:
        raise ValueError("Provide d only for schedule='shortcut'.")
    if cfg.schedule == "shortcut" and cfg.d is None:
        raise ValueError("schedule='shortcut' requires a valid d (e.g., 1/4).")

def _tau_grid_from(k_max: int, schedule: Schedule, d_opt: Optional[float], start_tau: float) -> Tuple[jnp.ndarray, int, float, int]:
    d = _choose_step_size(k_max, schedule, d_opt)
    start_aligned = jnp.clip(jnp.floor(start_tau / d) * d, 0.0, 1.0)
    S_float = (1.0 - start_aligned) / d
    S = int(jnp.round(S_float))
    tau_seq = start_aligned + d * jnp.arange(S + 1, dtype=jnp.float32)
    tau_seq = jnp.clip(tau_seq, 0.0, 1.0)
    e = _step_idx_from_d(float(d), k_max)
    return tau_seq, S, float(d), e

# ---------- Plan builder (host-only) ----------

def _build_run_plan(cfg: SamplerConfig) -> dict:
    d_used = _choose_step_size(cfg.k_max, cfg.schedule, cfg.d)
    e = _step_idx_from_d(d_used, cfg.k_max)
    d_inv = int(round(1.0 / d_used))
    plan = {
        "kind": "plan",
        "k_max": cfg.k_max,
        "schedule": cfg.schedule,
        "d": d_used,
        "K": d_inv,
        "e": e,
        "rollout": cfg.rollout,
        "start_mode": cfg.start_mode,
        "ctx_length": cfg.ctx_length,
        "horizon": cfg.horizon,
        "match_ctx_tau": bool(cfg.match_ctx_tau),
    }
    if cfg.rollout == "autoregressive":
        plan["tau0_policy"] = "pure (0.0)"
        plan["S"] = int(1.0 / d_used)
        plan["S_range"] = (plan["S"], plan["S"])
        plan["tau0_grid"] = [0.0]
    else:
        if cfg.start_mode == "pure":
            plan["tau0_policy"] = "pure (0.0)"
            plan["S"] = int(1.0 / d_used)
            plan["S_range"] = (plan["S"], plan["S"])
            plan["tau0_grid"] = [0.0]
        elif cfg.start_mode == "fixed":
            tau0a = _align_to_grid(float(np.clip(cfg.tau0_fixed, 0.0, 1.0)), d_used)
            plan["tau0_policy"] = f"fixed(aligned={tau0a:.6f})"
            plan["S"] = int(round((1.0 - tau0a) / d_used))
            plan["S_range"] = (plan["S"], plan["S"])
            grid = [i * d_used for i in range(0, d_inv)]
            plan["tau0_grid"] = grid
        else:  # random
            plan["tau0_policy"] = f"random on grid {{0, d, ..., 1-d}}"
            plan["S_range"] = (1, int(1.0 / d_used))
            plan["tau0_grid"] = ["0", "d", "...", "1-d"] if d_inv > 16 else [i * d_used for i in range(0, d_inv)]
    return plan

def _emit_plan(plan: dict, hook: Optional[Callable[[dict], None]], enable_print: bool):
    if hook:
        hook(plan)
    if enable_print:
        keys = ["kind","k_max","schedule","d","K","e","rollout","start_mode",
                "ctx_length","horizon","match_ctx_tau","tau0_policy","S","S_range","tau0_grid"]
        msg = {k: plan[k] for k in keys if k in plan}
        print(f"[sampler] {msg}")

# ---------------------------
# Core Single-Frame Sampler
# ---------------------------

def denoise_single_latent(
    *,
    dynamics: Dynamics,
    dyn_vars: Dict[str, Any],
    actions_ctx: jnp.ndarray,     # (B, T_ctx)
    action_curr: jnp.ndarray,     # (B, 1)
    z_ctx_clean: jnp.ndarray,     # (B, T_ctx, n_spatial, D_s) clean context
    z_t_init: jnp.ndarray,        # (B, 1, n_spatial, D_s) initial latent at tau0
    k_max: int,
    d: float,
    start_mode: StartMode,
    tau0_fixed: float,
    rng_key: jax.Array,
    clean_target_next: Optional[jnp.ndarray] = None,  # (B,1,n_spatial,D_s) if TF else None
    agent_tokens: jnp.ndarray = None, # (B, T_ctx + 1, n_agent, d_model)
    match_ctx_tau: bool = False,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Denoise a single future latent using a τ-ladder.
      - Uses adaptive mixing α = (τ_{s+1} − τ_s) / (1 − τ_s)
      - If match_ctx_tau=True, corrupt context to current τ at each step using a fixed z0_ctx
      - Returns both the denoised latent and the hidden state h_t from the final step
    
    Returns:
        z_t: (B, 1, n_spatial, D_s) denoised latent
        h_t: (B, 1, n_agent, d_model) or None - hidden state from final dynamics call
    """
    B, T_ctx, n_spatial, D_s = z_ctx_clean.shape
    _assert_power_of_two(k_max)
    assert actions_ctx.shape == (B, T_ctx)
    assert action_curr.shape == (B, 1)

    # 1) choose tau0
    rng_key, r_tau, r_noise, r_ctx = jax.random.split(rng_key, 4)
    if start_mode == "pure":
        tau0 = 0.0
    elif start_mode == "fixed":
        tau0 = float(np.clip(tau0_fixed, 0.0, 1.0))
    else:
        tau0 = float(jax.random.uniform(r_tau, (), minval=0.0, maxval=1.0))
    tau0_aligned = _align_to_grid(tau0, d) if tau0 > 0.0 else 0.0

    # Base noise for context if we match τ per step
    z0_ctx = jax.random.normal(r_ctx, z_ctx_clean.shape, dtype=z_ctx_clean.dtype) if match_ctx_tau else None

    # 2) init current latent at tau0 (caller already provided it)
    z_t = z_t_init

    # 3) tau ladder & indices  (FIX: pass d as d_opt, and tau0_aligned as start_tau)
    schedule = "shortcut" if d > (1.0 / k_max) else "finest"
    tau_seq, S, d_used, e = _tau_grid_from(k_max, schedule, d, tau0_aligned)
    tau_seq_host = list(np.asarray(tau_seq))  # host loop

    # Initialize h_t_final to None (will be set on final iteration)
    h_t_final = None

    # 4) iterate S steps
    for s in range(1, len(tau_seq_host)):
        tau_prev = float(tau_seq_host[s - 1])
        tau_curr = float(tau_seq_host[s])
        # Correct per-step mixing toward clean:
        alpha = (tau_curr - tau_prev) / max(1.0 - tau_prev, 1e-8)

        # Context at current τ (if requested)
        if match_ctx_tau:
            z_ctx_tau = tau_curr * z_ctx_clean + (1.0 - tau_curr) * z0_ctx
        else:
            z_ctx_tau = z_ctx_clean

        # Build sequence and indices
        z_seq = jnp.concatenate([z_ctx_tau, z_t], axis=1)                   # (B, T_ctx+1, n_spatial, D_s)
        actions_full = jnp.concatenate([actions_ctx, action_curr], axis=1)  # (B, T_ctx+1)
        step_idx = jnp.full((B, T_ctx + 1), e, dtype=jnp.int32)
        signal_idx = jnp.full((B, T_ctx + 1), _signal_idx_from_tau(jnp.asarray(tau_curr), k_max), dtype=jnp.int32)

        # Predict clean for the current frame
        z_clean_pred_seq, h_seq = dynamics.apply(
            dyn_vars,
            actions_full,
            step_idx,
            signal_idx,
            z_seq,
            agent_tokens=agent_tokens,
            deterministic=True,
        )
        z_clean_pred = z_clean_pred_seq[:, -1:, :, :]
        
        # Store hidden state from final step (tau=1.0)
        if s == len(tau_seq_host) - 1 and h_seq is not None:
            h_t_final = h_seq[:, -1:, :, :]  # (B, 1, n_agent, d_model)

        # Update with α
        z_t = (1.0 - alpha) * z_t + alpha * z_clean_pred

    return z_t, h_t_final  # (B,1,n_spatial,D_s), (B,1,n_agent,d_model) or None

# ---------------------------
# Multi-frame rollout wrapper
# ---------------------------

def sample_video(
    *,
    encoder: Encoder,
    decoder: Decoder,
    dynamics: Dynamics,
    enc_vars: Dict[str, Any],
    dec_vars: Dict[str, Any],
    dyn_vars: Dict[str, Any],
    frames: jnp.ndarray,     # (B,T,H,W,C)
    actions: jnp.ndarray,    # (B,T)
    config: SamplerConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    B, T, H, W, C = frames.shape
    assert config.ctx_length < T, "ctx_length must be < T"
    _validate_modes(config)

    # --- One-shot plan print before any work ---
    plan = _build_run_plan(config)
    _emit_plan(plan, config.debug_hook, config.debug)

    horizon = config.horizon
    rng = config.rng_key if config.rng_key is not None else jax.random.PRNGKey(0)
    mae_key = config.mae_eval_key if config.mae_eval_key is not None else jax.random.PRNGKey(777)

    # 1) encode once (deterministic key)
    patches = temporal_patchify(frames, config.patch)
    z_btLd, _ = encoder.apply(enc_vars, patches, rngs={"mae": mae_key}, deterministic=True)
    z_all = pack_bottleneck_to_spatial(z_btLd, n_spatial=config.n_spatial, k=config.packing_factor)  # (B,T,n_spatial,D_s)

    # 2) split context vs future
    z_ctx_clean = z_all[:, :config.ctx_length, :, :]
    actions_ctx = actions[:, :config.ctx_length]
    future_actions = actions[:, config.ctx_length: config.ctx_length + horizon]
    gt_future_latents = z_all[:, config.ctx_length: config.ctx_length + horizon, :, :]

    # (Optional) single-shot context corruption for visualization "floor" only
    z_ctx_for_floor = z_ctx_clean
    if config.ctx_signal_tau is not None and config.ctx_signal_tau < 1.0:
        rng, nkey = jax.random.split(rng)
        noise = jax.random.normal(nkey, z_ctx_clean.shape, z_ctx_clean.dtype)
        tau = jnp.asarray(config.ctx_signal_tau, z_ctx_clean.dtype)
        z_ctx_for_floor = tau * z_ctx_clean + (1.0 - tau) * noise

    # 3) floor: decoder recon of (ctx + GT future)
    floor_btLd = jnp.concatenate([
        unpack_spatial_to_bottleneck(z_ctx_for_floor, n_spatial=config.n_spatial, k=config.packing_factor),
        unpack_spatial_to_bottleneck(gt_future_latents, n_spatial=config.n_spatial, k=config.packing_factor)
    ], axis=1)
    floor_patches = decoder.apply(dec_vars, floor_btLd, deterministic=True)
    floor_frames = temporal_unpatchify(floor_patches, H, W, C, config.patch)

    # 4) choose schedule/step size
    d = _choose_step_size(config.k_max, config.schedule, config.d)

    # 5) rollout
    preds: list[jnp.ndarray] = []
    n_spatial, D_s = int(z_all.shape[2]), int(z_all.shape[3])

    for t in range(horizon):
        action_curr = future_actions[:, t:t+1]
        z1_ref = gt_future_latents[:, t:t+1, :, :] if config.rollout == "teacher_forced" else None

        # Initial latent at tau0 (pure start → tau0=0)
        rng, z0key = jax.random.split(rng)
        z0 = jax.random.normal(z0key, (B, 1, n_spatial, D_s), dtype=z_all.dtype)
        z_t_init = z0

        rng, step_key = jax.random.split(rng)
        z_clean_pred, _ = denoise_single_latent(
            dynamics=dynamics,
            dyn_vars=dyn_vars,
            actions_ctx=actions_ctx,
            action_curr=action_curr,
            z_ctx_clean=z_ctx_clean,
            z_t_init=z_t_init,
            k_max=config.k_max,
            d=d,
            start_mode=config.start_mode,
            tau0_fixed=config.tau0_fixed,
            rng_key=step_key,
            clean_target_next=z1_ref,
            match_ctx_tau=config.match_ctx_tau,
        )
        preds.append(z_clean_pred)

        # advance context (AR: use our prediction; TF: use GT)
        if config.rollout == "autoregressive":
            z_ctx_clean = jnp.concatenate([z_ctx_clean, z_clean_pred], axis=1)[:, -config.ctx_length:, :, :]
            actions_ctx = jnp.concatenate([actions_ctx, action_curr], axis=1)[:, -config.ctx_length:]
        else:
            z_ctx_clean = jnp.concatenate([z_ctx_clean, z1_ref], axis=1)[:, -config.ctx_length:, :, :]
            actions_ctx = jnp.concatenate([actions_ctx, action_curr], axis=1)[:, -config.ctx_length:]

    # 6) decode predictions (prepend context for viz)
    pred_latents = jnp.concatenate(preds, axis=1)
    pred_btLd = jnp.concatenate([
        unpack_spatial_to_bottleneck(z_all[:, :config.ctx_length, :, :], n_spatial=config.n_spatial, k=config.packing_factor),
        unpack_spatial_to_bottleneck(pred_latents, n_spatial=config.n_spatial, k=config.packing_factor),
    ], axis=1)
    pred_patches = decoder.apply(dec_vars, pred_btLd, deterministic=True)
    pred_frames = temporal_unpatchify(pred_patches, H, W, C, config.patch)

    gt_frames = frames[:, :config.ctx_length + horizon]
    return pred_frames, floor_frames, gt_frames
