# sampling logic for debugging / visualization. Not JIT friendly.
from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from dreamer.models import Tokenizer, Dynamics, PolicyHeadMTP, TaskEmbedder
from dreamer.actions import Actions
from .generation import DenoiseSchedule, latent_rollout


# ---------------------------
# Multi-frame rollout wrapper
# ---------------------------

def sample_video(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    frames: jax.Array | None,   # (B, T, H, W, C) in [0, 255] - None if using latents
    actions: Actions,           # (B, T)
    horizon: int,
    schedule_config: DenoiseSchedule,
    rng: jax.Array,
    policy: PolicyHeadMTP | None = None,
    task_embedder: TaskEmbedder | None = None,
    latents: jax.Array | None = None,  # (B, T, n_latents, d_bottleneck) - pre-tokenized latents
) -> Tuple[jax.Array, jax.Array, jax.Array | None]:
    """
    Sample video predictions using Tokenizer and Dynamics.

    Args:
        tokenizer: Tokenizer NNX model (has encode/decode methods)
        dynamics: Dynamics NNX model
        frames: Input video frames (B, T, H, W, C) in [0, 255] uint8. None if using latents.
        actions: Action sequence (B, T)
        horizon: Number of future frames to predict
        schedule_config: DenoiseSchedule with rollout parameters
        rng: Random key
        policy: Optional policy model. If provided, actions are sampled from the policy
                during rollout instead of using ground truth future actions.
        task_embedder: Optional task embedder. Required when policy is provided to generate
                agent tokens for the dynamics model.
        latents: Optional pre-tokenized latents (B, T, n_latents, d_bottleneck). If provided,
                skips tokenizer encoding. Latents should already be unpacked.

    Returns:
        pred_frames: (B, ctx+horizon, H, W, C) predicted frames [0, 255] uint8
        gt_decoded_frames: (B, ctx+horizon, H, W, C) GT latents decoded [0, 255] uint8
        original_frames: (B, ctx+horizon, H, W, C) original frames [0, 255] uint8, or None if using latents
    """


    if latents is not None:
        # Pre-tokenized latent path
        # Latents are already unpacked
        latents = latents.astype(jnp.bfloat16)
        B, T = latents.shape[:2]
    else:
        assert frames is not None
        # Video path - encode frames
        B, T, H, W, C = frames.shape

        rng, mae_key = jax.random.split(rng)
        rngs = nnx.Rngs(mae=mae_key)

        # Encode frames to clean latents (returns unpacked)
        latents, _ = tokenizer.encode(
            frames,
            deterministic=True,
            rngs=rngs
        )

    # Split context vs future
    latents_ctx_clean, latent_future = latents[:, :-horizon, :, :], latents[:, -horizon:, :, :]
    actions_ctx, actions_future = actions[:, :-horizon], actions[:, -horizon:]

    # Single-shot context corruption for visualization "floor" only
    latents_ctx = latents_ctx_clean
    # if schedule_config.tau_ctx < 1.0:  # FIXME: this is NOT taken from the evaluation config
    #     rng, nkey = jax.random.split(rng)
    #     noise = jax.random.normal(nkey, latents_ctx_clean.shape, latents_ctx_clean.dtype)
    #     tau = jnp.asarray(schedule_config.tau_ctx, latents_ctx_clean.dtype)
    #     latents_ctx = tau * latents_ctx_clean + (1.0 - tau) * noise

    # Decode GT latents for visualization
    # CRITICAL: Decoder must be JIT-compiled to produce correct spatial layout
    gt_decoded_frames = jnp.clip(tokenizer.decode(latents, deterministic=True)[0], 0, 255).astype(jnp.uint8)

    # Rollout
    # Use policy if provided, otherwise use ground truth future actions
    # When using a policy, we need agent tokens for the dynamics model to produce hidden states
    T_ctx = latents_ctx.shape[1]
    if policy is not None:
        assert task_embedder is not None, "task_embedder is required when policy is provided"
        task = jnp.zeros((B,), dtype=jnp.int32)  # Use task ID 0 for all samples
        initial_agent_tokens = task_embedder(task=task, B=B, T=T_ctx)
    else:
        initial_agent_tokens = None

    # Use latent_rollout for both paths (frames already encoded to latents above)
    rollout_result = latent_rollout(
        dynamics,
        policy=actions_future if policy is None else policy,
        schedule=schedule_config,
        latents_ctx=latents_ctx,
        actions_ctx=actions_ctx,
        num_steps=horizon,
        rng=rng,
        initial_task_embedding=initial_agent_tokens,
        deterministic=True,
    )

    # Decode predicted latents to frames
    # CRITICAL: Decoder must be JIT-compiled to produce correct spatial layout
    pred_frames = tokenizer.decode(rollout_result['latents'], deterministic=True)[0]
    pred_frames = jnp.clip(pred_frames, 0, 255).astype(jnp.uint8)
    original_frames = jnp.clip(frames, 0, 255).astype(jnp.uint8) if frames is not None else None

    return pred_frames, gt_decoded_frames, original_frames
