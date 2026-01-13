# sampling logic for debugging / visualization. Not JIT friendly.
from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from dreamer.models import Tokenizer, Dynamics, PolicyHeadMTP, TaskEmbedder
from .generation import DenoiseSchedule, video_rollout


# ---------------------------
# Multi-frame rollout wrapper
# ---------------------------

def sample_video(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    frames: jax.Array,     # (B, T, H, W, C) in [0, 255]
    actions: jax.Array,    # (B, T)
    horizon: int,
    schedule_config: DenoiseSchedule,
    rng: jax.Array,
    policy: PolicyHeadMTP | None = None,
    task_embedder: TaskEmbedder | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Sample video predictions using Tokenizer and Dynamics.

    Args:
        tokenizer: Tokenizer NNX model (has encode/decode methods)
        dynamics: Dynamics NNX model
        frames: Input video frames (B, T, H, W, C) in [0, 255] uint8
        actions: Action sequence (B, T)
        horizon: Number of future frames to predict
        schedule_config: DenoiseSchedule with rollout parameters
        rng: Random key
        policy: Optional policy model. If provided, actions are sampled from the policy
                during rollout instead of using ground truth future actions.
        task_embedder: Optional task embedder. Required when policy is provided to generate
                agent tokens for the dynamics model.
    
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

    # Rollout
    # Use policy if provided, otherwise use ground truth future actions
    # When using a policy, we need agent tokens for the dynamics model to produce hidden states
    T_ctx = frames_ctx.shape[1]
    if policy is not None:
        assert task_embedder is not None, "task_embedder is required when policy is provided"
        task = jnp.zeros((B,), dtype=jnp.int32)  # Use task ID 0 for all samples
        initial_agent_tokens = task_embedder(task=task, B=B, T=T_ctx)
    else:
        initial_agent_tokens = None
    
    pred_frames = video_rollout(
        tokenizer,
        dynamics,
        policy = actions_future if policy is None else policy,
        schedule=schedule_config,
        frames_ctx=frames_ctx,
        actions_ctx=actions_ctx,
        num_steps=horizon,
        rng=rng,
        initial_agent_tokens=initial_agent_tokens,
    )

    frames = jnp.clip(frames, 0, 255).astype(jnp.uint8)
    return pred_frames, tokenized_frames, frames
