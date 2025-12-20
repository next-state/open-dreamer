# sampling logic for debugging / visualization. Not JIT friendly.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional, Dict, Any, Callable


import jax
import jax.numpy as jnp
import numpy as np

from dreamer.models import Tokenizer, Dynamics, TaskEmbedder, PolicyHeadMTP
from .generation import DenoiseSchedule, video_rollout
from dreamer.utils import (
    pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck,
    normalize_with_dataset_stats, unnormalize_with_dataset_stats,
    is_pow2_frac,
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
    frames: jax.Array,     # (B, T, H, W, C) in [0, 1]
    actions: jax.Array,    # (B, T)
    horizon: int,
    config: DenoiseSchedule,
    rng: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
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

    rng, mae_key = jax.random.split(rng)

    # 1) encode once via tokenizer.encode (handles normalization internally? No - we normalize)
    frames_norm = normalize_with_dataset_stats(frames, mean=config.dataset_mean, std=config.dataset_std)
    z_btLd, _ = tokenizer.apply(
        tokenizer_vars, frames_norm, 
        method=tokenizer.encode, 
        rngs={"mae": mae_key}, 
        deterministic=True
    )
    z_all = pack_bottleneck_to_spatial(z_btLd, n_spatial=config.n_spatial, k=config.packing_factor)  # (B,T,n_spatial,D_s)

    # 2) split context vs future
    horizon = horizon//config.packing_factor #because of the packing factor
    z_ctx_clean = z_all[:, :-horizon, :, :]
    actions_ctx = actions[:, :-horizon]
    future_actions = actions[:, -horizon:]
    gt_future_latents = z_all[:, -horizon:, :, :]

    # Single-shot context corruption for visualization "floor" only
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


    # 5) rollout
    preds: list[jnp.ndarray] = []
    n_spatial, D_s = int(z_all.shape[2]), int(z_all.shape[3])

    pred_frames = video_rollout(tokenizer, tokenzier_vars, dynamics, dyn_vars, future_actions, policy_vars = None, schedule=schedule, initial_frames=initial_frames, initial_actions=initial_actions, horizon=horizon, num_steps=4, rng=rollout_rng)
    # 6) decode predictions (prepend context for viz)
    pred_latents = jnp.concatenate(preds, axis=1)
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
