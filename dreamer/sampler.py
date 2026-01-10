# sampling logic for debugging / visualization. Not JIT friendly.
from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from dreamer.models import Tokenizer, Dynamics
from .generation import DenoiseSchedule, video_rollout, video_rollout_meanflow


# ---------------------------
# Multi-frame rollout wrapper
# ---------------------------

def sample_video(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    frames: jax.Array,     # (B, T, H, W, C) in [0, 255]
    actions: jax.Array,    # (B, T)
    horizon: int,
    schedule_config: DenoiseSchedule | None,
    rng: jax.Array,
    forcing_type: str = "shortcut",
    num_meanflow_steps: int = 1,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Sample video predictions using Tokenizer and Dynamics.

    Automatically detects the forcing type and uses the appropriate sampler:
    - shortcut: Uses τ-ladder denoising with discrete conditioning
    - meanflow: Uses mean flow sampling with continuous conditioning

    Args:
        tokenizer: Tokenizer NNX model (has encode/decode methods)
        dynamics: Dynamics NNX model
        frames: Input video frames (B, T, H, W, C) in [0, 255] uint8
        actions: Action sequence (B, T)
        horizon: Number of future frames to predict
        schedule_config: DenoiseSchedule with rollout parameters (for shortcut only, can be None for meanflow)
        rng: Random key
        forcing_type: "shortcut" or "meanflow" - determines which sampler to use
        num_meanflow_steps: Number of refinement steps for meanflow (1 for direct, 4+ for multi-step)

    Returns:
        pred_frames: (B, ctx+horizon, H, W, C) predicted frames [0, 255] uint8
        tokenized_frames: (B, ctx+horizon, H, W, C) tokenizer reconstruction (GT latents decoded) [0, 255] uint8
        frames: (B, ctx+horizon, H, W, C) ground truth frames [0, 255] uint8
    """
    B, T, H, W, C = frames.shape

    rng, mae_key = jax.random.split(rng)
    rngs = nnx.Rngs(mae=mae_key)

    # Encode frames to clean latents
    latents, _ = tokenizer.encode(
        frames,
        packing_factor=dynamics.cfg.packing_factor,
        deterministic=True,
        rngs=rngs
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
    tokenized_frames, _ = tokenizer.decode(
        latents_for_tokenized_frames,
        packing_factor=dynamics.cfg.packing_factor,
        deterministic=True
    )
    tokenized_frames = jnp.clip(tokenized_frames, 0, 255).astype(jnp.uint8)

    # Rollout using appropriate sampler
    actions_future = jnp.array(actions_future)

    if forcing_type == "shortcut":
        assert schedule_config is not None, "schedule_config required for shortcut forcing"
        pred_frames = video_rollout(
            tokenizer,
            dynamics,
            policy=actions_future,
            schedule=schedule_config,
            frames_ctx=frames_ctx,
            actions_ctx=actions_ctx,
            num_steps=horizon,
            rng=rng,
            initial_agent_tokens=None
        )
    elif forcing_type == "meanflow":
        pred_frames = video_rollout_meanflow(
            tokenizer,
            dynamics,
            policy=actions_future,
            num_steps=num_meanflow_steps,
            frames_ctx=frames_ctx,
            actions_ctx=actions_ctx,
            num_rollout_steps=horizon,
            rng=rng,
            tau_ctx=0.9,
            initial_agent_tokens=None
        )
    else:
        raise ValueError(f"Unknown forcing_type: {forcing_type}. Must be 'shortcut' or 'meanflow'.")

    frames = jnp.clip(frames, 0, 255).astype(jnp.uint8)
    return pred_frames, tokenized_frames, frames
