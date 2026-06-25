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

# Calls under `jax.set_mesh(...)` must be jit'd: `with_partitioning` -> `with_sharding_constraint`
# is a no-op in eager mode (jax#17422), so eager activations get mis-laid-out and frames look corrupt.
@nnx.jit
def encode_jit(tokenizer: Tokenizer, frames: jax.Array) -> jax.Array:
    latents, _ = tokenizer.encode(frames, deterministic=True)
    return latents

@nnx.jit
def decode_jit(tokenizer: Tokenizer, z: jax.Array) -> jax.Array:
    frames, _ = tokenizer.decode(z, deterministic=True)
    return frames


def decode_chunked(tokenizer: Tokenizer, z: jax.Array, chunk: int = 1) -> jax.Array:
    """Decode (B, T, ...) latents in batch-chunks to cap peak decoder memory.

    Decoding a full eval batch at high resolution materializes a very large decoder
    activation (the MLP intermediate is ~B*T*S*4*d_model); for B=4, T=64 at 360x640 that
    is tens of GB in a single allocation. Splitting over the batch keeps the result
    bit-identical (samples are independent) while reducing peak memory ~B/chunk.
    """
    B = z.shape[0]
    if chunk <= 0 or B <= chunk:
        return decode_jit(tokenizer, z)
    parts = [decode_jit(tokenizer, z[i:i + chunk]) for i in range(0, B, chunk)]
    return jnp.concatenate(parts, axis=0)

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

        # Encode frames to clean latents (returns unpacked)
        latents = encode_jit(tokenizer, frames)

    # Split context vs future
    latents_ctx, latent_future = latents[:, :-horizon, :, :], latents[:, -horizon:, :, :]
    actions_ctx, actions_future = actions[:, :-horizon], actions[:, -horizon:]

    # Decode GT latents for visualization (chunked to cap peak decoder memory)
    gt_decoded_frames = decode_chunked(tokenizer, latents)
    gt_decoded_frames = jnp.clip(gt_decoded_frames, 0, 255).astype(jnp.uint8)

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

    # Decode predicted latents to frames (chunked to cap peak decoder memory)
    pred_frames = decode_chunked(tokenizer, rollout_result['latents'])
    pred_frames = jnp.clip(pred_frames, 0, 255).astype(jnp.uint8)
    original_frames = jnp.clip(frames, 0, 255).astype(jnp.uint8) if frames is not None else None

    return pred_frames, gt_decoded_frames, original_frames
