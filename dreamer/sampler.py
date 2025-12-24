# sampling logic for debugging / visualization. Not JIT friendly.
from __future__ import annotations
from typing import Tuple, Dict, Any


import jax
import jax.numpy as jnp
import numpy as np

from dreamer.models import Tokenizer, Dynamics
from .generation import DenoiseSchedule, video_rollout


# ---------------------------
# Multi-frame rollout wrapper
# ---------------------------

def sample_video(
    tokenizer: Tokenizer,
    tokenizer_vars: Dict[str, Any],
    dynamics: Dynamics,
    dyn_vars: Dict[str, Any],
    frames: jax.Array,     # (B, T, H, W, C) in [0, 1]
    actions: jax.Array,    # (B, T)
    horizon: int,
    schedule_config: DenoiseSchedule,
    rng: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Sample video predictions using Tokenizer and Dynamics.
    
    Args:
        tokenizer: Tokenizer module (has encode/decode methods)
        tokenizer_vars: Combined variables dict with 'params' and 'constants'
        dynamics: Dynamics model
        dyn_vars: Dynamics variables dict
        frames: Input video frames (B, T, H, W, C) in [0, 255]
        actions: Action sequence (B, T)
        schedule_config: DenoiseSchedule with rollout parameters
    
    Returns:
        pred_frames: (B, ctx+horizon, H, W, C) predicted frames (uint8)
        tokenized_frames: (B, ctx+horizon, H, W, C) tokenizer reconstruction (GT latents decoded) (uint8)
        frames: (B, ctx+horizon, H, W, C) ground truth frames (uint8)
    """
    B, T, H, W, C = frames.shape

    rng, mae_key = jax.random.split(rng)

    # encode frames to clean latents
    latents, _ = tokenizer.apply(
        tokenizer_vars, frames,
        packing_factor=dynamics.config.packing_factor,
        method=tokenizer.encode, 
        rngs={"mae": mae_key},
        deterministic=True
    )
    
    # Split context vs future
    frames_ctx = frames[:, :-horizon, :, :, :]
    latents_ctx_clean = latents[:, :-horizon, :, :]
    latents_future = latents[:, -horizon:, :, :]
    actions_ctx = actions[:, :-horizon]
    actions_future = actions[:, -horizon:]

    # Single-shot context corruption for visualization "floor" only
    latents_ctx = latents_ctx_clean
    # if schedule_config.tau_ctx < 1.0:  # FIXME: this is NOT taken from the evaluation config
    #     rng, nkey = jax.random.split(rng)
    #     noise = jax.random.normal(nkey, latents_ctx_clean.shape, latents_ctx_clean.dtype)
    #     tau = jnp.asarray(schedule_config.tau_ctx, latents_ctx_clean.dtype)
    #     latents_ctx = tau * latents_ctx_clean + (1.0 - tau) * noise

    # Tokenized frames for visualization
    latents_for_tokenized_frames = jnp.concatenate([latents_ctx, latents_future], axis=1)
    tokenized_frames, _ = tokenizer.apply(
        tokenizer_vars, latents_for_tokenized_frames, packing_factor=dynamics.config.packing_factor,
        method=tokenizer.decode, 
        deterministic=True
    )
    tokenized_frames = jnp.clip(tokenized_frames, 0, 255).astype(jnp.uint8)

    # Rollout
    pred_frames = video_rollout(tokenizer,
                                tokenizer_vars,
                                dynamics,
                                dyn_vars,
                                policy=actions_future,
                                policy_vars=None,
                                schedule=schedule_config,
                                frames_ctx=frames_ctx,
                                actions_ctx=actions_ctx,
                                num_steps=horizon,
                                rng=rng,
                                initial_agent_tokens=None)

    frames = jnp.clip(frames, 0, 255).astype(jnp.uint8)
    return pred_frames, tokenized_frames, frames
