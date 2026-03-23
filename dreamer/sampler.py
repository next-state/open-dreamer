# sampling logic for debugging / visualization. Not JIT friendly.
from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx

from dreamer.models import Tokenizer, Dynamics, PolicyHeadMTP, TaskEmbedder
from dreamer.actions import Actions
from .generation import DenoiseSchedule, video_rollout, latent_rollout, next_frame
from .utils import normalize_latents


@nnx.jit
def decode_jit(tokenizer: Tokenizer, z: jax.Array) -> jax.Array:
    frames, _ = tokenizer.decode(z, deterministic=True)
    return frames


@nnx.jit
def warmup_tokenizer_cache_jit(
    tokenizer: Tokenizer,
    unnormalized_latents: jax.Array,
    tokenizer_cache,
):
    frames, tokenizer_cache = tokenizer.decode(
        unnormalized_latents,
        caches=tokenizer_cache,
        deterministic=True,
    )
    _, _, tokenizer_cache = tokenizer.encode(
        frames,
        deterministic=True,
        caches=tokenizer_cache,
    )
    return frames, tokenizer_cache


@nnx.jit
def prefill_dynamics_cache_jit(
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    actions: Actions,
    unnormalized_latents: jax.Array,
    dynamics_cache,
):
    normalized_latents = normalize_latents(
        unnormalized_latents,
        dynamics.cfg.latent_mean,
        dynamics.cfg.latent_std,
    )
    B, T = normalized_latents.shape[:2]
    step_indices = jnp.full((B, T), schedule.emax, dtype=jnp.int32)
    tau_indices = jnp.full((B, T), schedule.k_max, dtype=jnp.int32)

    _, (_, dynamics_cache) = dynamics(
        actions,
        step_indices,
        tau_indices,
        normalized_latents,
        deterministic=True,
        caches=dynamics_cache,
    )
    return dynamics_cache


@nnx.jit(static_argnames=("latent_shape",))
def next_frame_jit(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    action: Actions,
    latent_shape: Tuple,
    dynamics_cache,
    tokenizer_cache,
    rng: jax.Array,
):
    return next_frame(
        tokenizer=tokenizer,
        dynamics=dynamics,
        schedule=schedule,
        action=action,
        latent_shape=latent_shape,
        dynamics_cache=dynamics_cache,
        tokenizer_cache=tokenizer_cache,
        rng=rng,
    )


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
) -> Tuple[jax.Array, jax.Array, jax.Array | None, jax.Array]:
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
        latents, _, _ = tokenizer.encode(
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
    gt_decoded_frames = decode_jit(tokenizer, latents)
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

    if policy is None:
        window_size = T
        latent_shape = (B, 1, latents.shape[2], latents.shape[3])
        dynamics_cache = dynamics.create_static_caches(
            batch_size=B,
            n_latents=latents.shape[2],
            window_size=window_size,
            dtype=latents.dtype,
        )
        tokenizer_cache = tokenizer.create_static_caches(
            batch_size=B,
            window_size=window_size,
            dtype=latents.dtype,
        )

        cached_frames = []
        dynamics_cache = prefill_dynamics_cache_jit(
            dynamics=dynamics,
            schedule=schedule_config,
            actions=actions_ctx,
            unnormalized_latents=latents_ctx,
            dynamics_cache=dynamics_cache,
        )

        for t in range(T_ctx):
            frame_t, tokenizer_cache = warmup_tokenizer_cache_jit(
                tokenizer=tokenizer,
                unnormalized_latents=latents_ctx[:, t:t + 1],
                tokenizer_cache=tokenizer_cache,
            )
            assert frame_t is not None
            cached_frames.append(frame_t)

        for t in range(horizon):
            frame_t, _, dynamics_cache, tokenizer_cache, rng = next_frame_jit(
                tokenizer=tokenizer,
                dynamics=dynamics,
                schedule=schedule_config,
                action=actions_future[:, t],
                latent_shape=latent_shape,
                dynamics_cache=dynamics_cache,
                tokenizer_cache=tokenizer_cache,
                rng=rng,
            )
            cached_frames.append(frame_t)

        pred_frames = jnp.concatenate(cached_frames, axis=1)
        pred_frames = jnp.clip(pred_frames, 0, 255).astype(jnp.uint8)
        ode_diags = {}
    else:
        rollout_result = latent_rollout(
            dynamics,
            policy=policy,
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
        pred_frames = decode_jit(tokenizer, rollout_result['latents'])
        pred_frames = jnp.clip(pred_frames, 0, 255).astype(jnp.uint8)
        ode_diags = rollout_result.get('ode_diags', {})

    original_frames = jnp.clip(frames, 0, 255).astype(jnp.uint8) if frames is not None else None

    return pred_frames, gt_decoded_frames, original_frames, ode_diags
