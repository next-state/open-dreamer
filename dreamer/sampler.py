# sampling logic for debugging / visualization. Not JIT friendly.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional, Dict, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from dreamer.models import Tokenizer, Dynamics, TaskEmbedder, PolicyHeadMTP
from dreamer.utils import (
    pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck,
    normalize_with_dataset_stats, unnormalize_with_dataset_stats,
)

# Use the JAX-friendly denoiser and schedule helper from imagination
from dreamer.imagination import DenoiseSchedule, denoise_single_latent_static, make_schedule

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
    ctx_signal_tau: float = 0.9             # e.g., 0.9 for slightly corrupt viz; None/1.0 = clean

    rng_key: Optional[jax.Array] = None
    mae_eval_key: Optional[jax.Array] = None
    # decoding sizes
    H: int = 32; W: int = 32; C: int = 3; patch: int = 4
    # tokenizer shapes
    n_spatial: int = 8
    packing_factor: int = 2
    # dataset normalization
    dataset_mean: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    dataset_std: list[float] = field(default_factory=lambda: [0.288675, 0.288675, 0.288675])
    # debugging (host-side only)
    debug: bool = False
    debug_hook: Optional[Callable[[dict], None]] = None

    def get_d(self) -> float:
        """Get the effective step size d."""
        if self.schedule == "finest":
            return 1.0 / float(self.k_max)
        if self.d is None:
            raise ValueError("schedule='shortcut' requires d")
        return float(self.d)

    def to_schedule(self, rng_key: Optional[jax.Array] = None) -> DenoiseSchedule:
        """Build a DenoiseSchedule from this config."""
        return make_schedule(
            k_max=self.k_max,
            d=self.get_d(),
            start_mode=self.start_mode,
            tau0_fixed=self.tau0_fixed,
            horizon=self.horizon,
            context_length=self.ctx_length,
            n_spatial=self.n_spatial,
            tau_ctx=self.ctx_signal_tau,
            rng_key=rng_key,
        )


def plan_from_sampler_conf(s: SamplerConfig) -> dict:
    """Convert SamplerConfig to a JSON-serializable plan dict for logging."""
    d = s.get_d()
    S = int(round(1.0 / d))
    e = int(round(np.log2(round(1.0 / d))))
    return dict(
        rollout=s.rollout,
        start_mode=s.start_mode,
        ctx_length=s.ctx_length,
        horizon=s.horizon,
        schedule=s.schedule,
        d=d,
        e=e,
        S=S,
        tau_seq=[round(i * d, 6) for i in range(S + 1)],
        k_max=s.k_max,
    )

# ---------------------------
# Multi-frame rollout wrapper
# ---------------------------

def sample_video(
    *,
    tokenizer: Tokenizer,
    tokenizer_vars: Dict[str, Any],
    dynamics: Dynamics,
    dyn_vars: Dict[str, Any],
    frames: jnp.ndarray,     # (B, T, H, W, C) in [0, 1]
    actions: jnp.ndarray,    # (B, T)
    config: SamplerConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample video predictions using Tokenizer and Dynamics.

    Args:
        tokenizer: Tokenizer module (has encode/decode methods)
        tokenizer_vars: Combined variables dict with 'params' and 'constants'
        dynamics: Dynamics model
        dyn_vars: Dynamics variables dict
        frames: Input video frames (B, T, H, W, C) normalized to [0,1]
        actions: Action sequence (B, T)
        config: SamplerConfig with rollout parameters

    Returns:
        pred_frames: (B, ctx+horizon, H, W, C) predicted frames
        floor_frames: (B, ctx+horizon, H, W, C) floor reconstruction (GT latents decoded)
        gt_frames: (B, ctx+horizon, H, W, C) ground truth frames
    """
    B, T, H, W, C = frames.shape
    assert config.ctx_length < T, "ctx_length must be < T"
    
    # Validate modes
    if config.rollout == "autoregressive" and config.start_mode != "pure":
        raise ValueError("Autoregressive rollout supports only start_mode='pure'.")

    # Debug logging
    if config.debug:
        plan = plan_from_sampler_conf(config)
        print(f"[sampler] {plan}")
        if config.debug_hook:
            config.debug_hook(plan)

    horizon = config.horizon
    rng = config.rng_key if config.rng_key is not None else jax.random.PRNGKey(0)
    mae_key = config.mae_eval_key if config.mae_eval_key is not None else jax.random.PRNGKey(777)

    # 1) encode
    frames_norm = normalize_with_dataset_stats(frames, mean=config.dataset_mean, std=config.dataset_std)
    z_btLd, _ = tokenizer.apply(
        tokenizer_vars, frames_norm, 
        method=tokenizer.encode, 
        rngs={"mae": mae_key}, 
        deterministic=True
    )
    z_all = pack_bottleneck_to_spatial(z_btLd, n_spatial=config.n_spatial, k=config.packing_factor)

    # 2) split context vs future
    z_ctx_clean = z_all[:, :config.ctx_length, :, :]
    actions_ctx = actions[:, :config.ctx_length]
    future_actions = actions[:, config.ctx_length: config.ctx_length + horizon]
    gt_future_latents = z_all[:, config.ctx_length: config.ctx_length + horizon, :, :]

    # Context corruption for visualization "floor" only
    z_ctx_for_floor = z_ctx_clean
    if config.ctx_signal_tau < 1.0:
        rng, nkey = jax.random.split(rng)
        noise = jax.random.normal(nkey, z_ctx_clean.shape, z_ctx_clean.dtype)
        tau = jnp.asarray(config.ctx_signal_tau, z_ctx_clean.dtype)
        z_ctx_for_floor = tau * z_ctx_clean + (1.0 - tau) * noise

    # 3) floor: decoder recon of (ctx + GT future)
    floor_btLd = jnp.concatenate([
        unpack_spatial_to_bottleneck(z_ctx_for_floor, n_spatial=config.n_spatial, k=config.packing_factor),
        unpack_spatial_to_bottleneck(gt_future_latents, n_spatial=config.n_spatial, k=config.packing_factor)
    ], axis=1)
    floor_frames_norm = tokenizer.apply(
        tokenizer_vars, floor_btLd, 
        method=tokenizer.decode, 
        deterministic=True
    )
    floor_frames = unnormalize_with_dataset_stats(floor_frames_norm, mean=config.dataset_mean, std=config.dataset_std)
    floor_frames = jnp.clip(floor_frames, 0.0, 1.0)

    # 4) rollout - use simplified config.to_schedule()
    schedule = config.to_schedule()
    
    preds: list[jnp.ndarray] = []
    n_spatial, D_s = int(z_all.shape[2]), int(z_all.shape[3])

    for t in range(horizon):
        action_curr = future_actions[:, t:t+1]
        z1_ref = gt_future_latents[:, t:t+1, :, :] if config.rollout == "teacher_forced" else None

        # Initial latent at tau0 (pure start → tau0=0)
        rng, z0key = jax.random.split(rng)
        z0 = jax.random.normal(z0key, (B, 1, n_spatial, D_s), dtype=z_all.dtype)

        # Call the JAX-friendly denoiser
        z_clean_pred, _h_last, _caches = denoise_single_latent_static(
            dynamics=dynamics,
            dyn_vars=dyn_vars,
            schedule=schedule,
            actions_ctx=actions_ctx,
            action_curr=action_curr,
            z_ctx_t=z_ctx_clean,
            z_noise_t=z0,
            agent_tokens=None,
            caches=None,
        )
        preds.append(z_clean_pred)

        # advance context (AR: use our prediction; TF: use GT)
        if config.rollout == "autoregressive":
            z_ctx_clean = jnp.concatenate([z_ctx_clean, z_clean_pred], axis=1)[:, -config.ctx_length:, :, :]
            actions_ctx = jnp.concatenate([actions_ctx, action_curr], axis=1)[:, -config.ctx_length:]
        else:
            z_ctx_clean = jnp.concatenate([z_ctx_clean, z1_ref], axis=1)[:, -config.ctx_length:, :, :]
            actions_ctx = jnp.concatenate([actions_ctx, action_curr], axis=1)[:, -config.ctx_length:]

    # 5) decode predictions
    if len(preds) == 0:
        raise AssertionError("No predictions were generated (preds is empty)")
    preds_tuple = tuple(jnp.asarray(p) for p in preds)
    pred_latents = jnp.concatenate(preds_tuple, axis=1)
    pred_btLd = jnp.concatenate([
        unpack_spatial_to_bottleneck(z_all[:, :config.ctx_length, :, :], n_spatial=config.n_spatial, k=config.packing_factor),
        unpack_spatial_to_bottleneck(pred_latents, n_spatial=config.n_spatial, k=config.packing_factor),
    ], axis=1)
    pred_frames_norm = tokenizer.apply(
        tokenizer_vars, pred_btLd, 
        method=tokenizer.decode, 
        deterministic=True
    )
    pred_frames = unnormalize_with_dataset_stats(pred_frames_norm, mean=config.dataset_mean, std=config.dataset_std)
    pred_frames = jnp.clip(pred_frames, 0.0, 1.0)

    gt_frames = frames[:, :config.ctx_length + horizon]
    return pred_frames, floor_frames, gt_frames