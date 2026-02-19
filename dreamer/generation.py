import math
from typing import Any, Tuple

import einops
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .actions import Actions
from .models import KVCachesDict, Dynamics, PolicyHeadMTP, Tokenizer


@dataclass
class DenoiseSchedule:
    """Simple denoising schedule shared by rollout utilities."""

    num_steps: int
    k_max: int
    d: float
    step_idx: int
    tau_values: jax.Array
    tau_indices: jax.Array
    step_idx_ctx: int
    tau_idx_ctx: int
    tau_ctx: float

    @classmethod
    def init(cls, num_steps: int, k_max: int = 256, tau_ctx: float = 1.0) -> "DenoiseSchedule":
        if k_max % num_steps != 0:
            raise ValueError(f"k_max={k_max} must be divisible by num_steps={num_steps}")

        step_idx = int(math.log2(num_steps))
        if 2**step_idx != num_steps:
            raise ValueError(f"num_steps={num_steps} must be a power of two")

        d = 1 / num_steps
        tau_values = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float32)
        tau_indices = (jnp.arange(num_steps, dtype=jnp.int32) * (k_max // num_steps)).astype(jnp.int32)

        # Keep context processing fully clean for debugging.
        step_idx_ctx = int(math.log2(k_max))
        tau_idx_ctx = k_max - 1
        tau_ctx = 1.0

        return cls(num_steps, k_max, d, step_idx, tau_values, tau_indices, step_idx_ctx, tau_idx_ctx, tau_ctx)


def next_latent(
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    action: Actions,
    latent_shape: Tuple,  # (B, 1, n_spatial, D_s)
    rng: jax.Array,
    prefill_length: int | None = None,
    task_embedding: jax.Array | None = None,
    caches: KVCachesDict | None = None,
    latents_ctx: jax.Array | None = None,
    actions_ctx: Actions | None = None,
    omega: jax.Array | float = 0.0,
    exponent: jax.Array | float = 0.7,
) -> Tuple[jax.Array, jax.Array | None, KVCachesDict | None, jax.Array]:
    """Generate one latent frame with iterative denoising."""
    del prefill_length, latents_ctx, actions_ctx, omega, exponent

    if task_embedding is not None and task_embedding.shape[1] != 1:
        task_embedding = task_embedding[:, -1:]

    rng, rng_latent = jax.random.split(rng)
    latent_t = jax.random.normal(rng_latent, latent_shape)
    B, _, _, C = latent_shape
    action = action[:, None, ...]
    step_indices = jnp.full((B, 1), schedule.step_idx, dtype=jnp.int32)

    def refinement_step(latent_prev, s):
        tau_prev = schedule.tau_values[s]
        tau_curr = schedule.tau_values[s + 1]
        mix = (tau_curr - tau_prev) / jnp.maximum(1.0 - tau_prev, 1e-8)

        tau_indices = jnp.full((B, 1), schedule.tau_indices[s], dtype=jnp.int32)
        latent_clean_seq, (h_seq, _) = dynamics(
            action,
            step_indices,
            tau_indices,
            latent_prev,
            task_embeddings=task_embedding,
            deterministic=True,
            caches=caches,
        )
        latent_clean = latent_clean_seq[:, -1:, :, :C]
        h_last = h_seq[:, -1:, :, :] if isinstance(h_seq, jax.Array) else h_seq

        latent_next = (1.0 - mix) * latent_prev + mix * latent_clean
        return latent_next, h_last

    latent_t_final, h_history = jax.lax.scan(
        refinement_step,
        latent_t,
        jnp.arange(schedule.num_steps),
    )

    if caches is not None:
        step_indices_ctx = jnp.full((B, 1), schedule.step_idx_ctx, dtype=jnp.int32)
        tau_indices_ctx = jnp.full((B, 1), schedule.tau_idx_ctx, dtype=jnp.int32)
        _, (h_seq_final, caches_new) = dynamics(
            action,
            step_indices_ctx,
            tau_indices_ctx,
            latent_t_final,
            task_embeddings=task_embedding,
            deterministic=True,
            caches=caches,
        )
        h_last = h_seq_final[:, -1:, :, :] if isinstance(h_seq_final, jax.Array) else h_seq_final
    else:
        h_last = h_history[-1] if isinstance(h_history, jax.Array) else None
        caches_new = None

    return latent_t_final, h_last, caches_new, rng


def _num_pixel_tokens_from_shape(
    frame_hw: Tuple[int, int],
    patch_size: int,
) -> int:
    """Convert (H, W) frame size to patch token count."""
    H, W = frame_hw
    if (H % patch_size) != 0 or (W % patch_size) != 0:
        raise ValueError(f"Frame shape ({H}, {W}) must be divisible by patch_size={patch_size}.")
    return (H // patch_size) * (W // patch_size)


def next_observation(
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    action: Actions,
    observation_shape: Tuple,  # (B, 1, H, W, C) for pixel mode
    rng: jax.Array,
    prefill_length: int | None = None,
    task_embedding: jax.Array | None = None,
    caches: KVCachesDict | None = None,
    latents_ctx: jax.Array | None = None,
    actions_ctx: Actions | None = None,
    omega: jax.Array | float = 0.0,
    exponent: jax.Array | float = 0.7,
) -> Tuple[jax.Array, jax.Array | None, KVCachesDict | None, jax.Array]:
    """Generate one pixel observation with iterative denoising."""
    del prefill_length, latents_ctx, actions_ctx, omega, exponent

    if task_embedding is not None and task_embedding.shape[1] != 1:
        task_embedding = task_embedding[:, -1:]

    rng, rng_state = jax.random.split(rng)
    obs_t = jax.random.normal(rng_state, observation_shape)
    B = observation_shape[0]
    action = action[:, None, ...]
    step_indices = jnp.full((B, 1), schedule.step_idx, dtype=jnp.int32)

    def refinement_step(obs_prev, s):
        tau_prev = schedule.tau_values[s]
        tau_curr = schedule.tau_values[s + 1]
        mix = (tau_curr - tau_prev) / jnp.maximum(1.0 - tau_prev, 1e-8)

        tau_indices = jnp.full((B, 1), schedule.tau_indices[s], dtype=jnp.int32)
        obs_clean_seq, (h_seq, _) = dynamics(
            action,
            step_indices,
            tau_indices,
            obs_prev,
            task_embeddings=task_embedding,
            deterministic=True,
            caches=caches,
        )
        obs_clean = obs_clean_seq[:, -1:]
        h_last = h_seq[:, -1:] if isinstance(h_seq, jax.Array) else h_seq

        obs_next = (1.0 - mix) * obs_prev + mix * obs_clean
        return obs_next, h_last

    obs_t_final, h_history = jax.lax.scan(
        refinement_step,
        obs_t,
        jnp.arange(schedule.num_steps),
    )

    if caches is not None:
        step_indices_ctx = jnp.full((B, 1), schedule.step_idx_ctx, dtype=jnp.int32)
        tau_indices_ctx = jnp.full((B, 1), schedule.tau_idx_ctx, dtype=jnp.int32)
        _, (h_seq_final, caches_new) = dynamics(
            action,
            step_indices_ctx,
            tau_indices_ctx,
            obs_t_final,
            task_embeddings=task_embedding,
            deterministic=True,
            caches=caches,
        )
        h_last = h_seq_final[:, -1:] if isinstance(h_seq_final, jax.Array) else h_seq_final
    else:
        h_last = h_history[-1] if isinstance(h_history, jax.Array) else None
        caches_new = None

    return obs_t_final, h_last, caches_new, rng


def next_frame(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    action: Actions,
    latent_shape: Tuple,
    dynamics_cache: Any,
    tokenizer_cache: Any,
    rng: jax.Array,
    task_embedding: jax.Array | None = None,
    omega: jax.Array | float = 0.0,
    alpha: jax.Array | float = 0.7,
) -> Tuple[jax.Array, jax.Array | None, KVCachesDict | None, Any, jax.Array]:
    """Generate and decode one frame."""
    latent, h_last, dynamics_cache_updated, rng = next_latent(
        dynamics=dynamics,
        schedule=schedule,
        action=action,
        latent_shape=latent_shape,
        rng=rng,
        prefill_length=None,
        task_embedding=task_embedding,
        caches=dynamics_cache,
        omega=omega,
        exponent=alpha,
    )

    frame, tokenizer_cache_updated = tokenizer.decode(
        latent,
        caches=tokenizer_cache,
        deterministic=True,
    )
    frame = jnp.clip(frame, 0, 255).astype(jnp.uint8)
    return frame, h_last, dynamics_cache_updated, tokenizer_cache_updated, rng


def latent_rollout(
    dynamics: Dynamics,
    policy: PolicyHeadMTP | Actions,
    schedule: DenoiseSchedule,
    latents_ctx: jax.Array,
    actions_ctx: Actions,
    num_steps: int,
    rng: jax.Array,
    initial_task_embedding: jax.Array | None = None,
    deterministic: bool = False,
    omega: jax.Array | float = 0.0,
    alpha: jax.Array | float = 0.7,
):
    """Autoregressive rollout in latent space."""
    del omega, alpha

    B, T_ctx, n_spatial, D_s = latents_ctx.shape
    latent_shape = (B, 1, n_spatial, D_s)

    latents_ctx_orig = latents_ctx

    window_size = T_ctx + num_steps
    n_agents = policy.cfg.L if isinstance(policy, PolicyHeadMTP) else 0
    caches = dynamics.create_static_caches(
        batch_size=B,
        n_latents=n_spatial,
        window_size=window_size,
        n_agent=n_agents,
        dtype=latents_ctx.dtype,
    )

    emax = int(math.log2(schedule.k_max))
    step_idx_ctx = jnp.full((B, T_ctx), emax, dtype=jnp.int32)
    tau_idx_ctx = jnp.full((B, T_ctx), schedule.k_max - 1, dtype=jnp.int32)

    _, (h_seq, caches) = dynamics(
        actions_ctx,
        step_idx_ctx,
        tau_idx_ctx,
        latents_ctx,
        task_embeddings=initial_task_embedding,
        caches=caches,
        deterministic=True,
    )

    task_embedding = initial_task_embedding[:, -1:] if isinstance(initial_task_embedding, jax.Array) else None
    h_last = h_seq[:, -1:] if isinstance(h_seq, jax.Array) else None

    def scan_step(carry, step_idx):
        h_t, caches_t, rng = carry
        rng, rng_policy = jax.random.split(rng)

        if isinstance(policy, Actions):
            action = policy[:, step_idx]
        else:
            all_actions = policy.sample(h_t, deterministic=deterministic, rng=rng_policy)
            action = all_actions[:, 0, 0, ...]

        latent_next, h_next, caches_next, rng = next_latent(
            dynamics,
            schedule,
            action,
            latent_shape,
            rng,
            caches=caches_t,
            task_embedding=task_embedding,
        )
        return (h_next, caches_next, rng), (latent_next[:, 0], action, h_next)

    _, (rollout_latents, rollout_actions, rollout_hidden) = jax.lax.scan(
        scan_step,
        (h_last, caches, rng),
        jnp.arange(num_steps),
    )

    rollout_latents = einops.rearrange(rollout_latents, "t b s d -> b t s d")
    rollout_actions = jax.tree.map(lambda x: einops.rearrange(x, "t b ... -> b t ..."), rollout_actions)
    rollout_hidden = (
        einops.rearrange(rollout_hidden, "t b 1 n d -> b t n d")
        if isinstance(rollout_hidden, jax.Array)
        else None
    )

    out_latents = jnp.concatenate((latents_ctx_orig, rollout_latents), axis=1)
    return {
        "latents": out_latents,
        "actions": rollout_actions,
        "hidden_states": rollout_hidden,
        "context_hidden": h_seq,
    }


def pixel_rollout(
    dynamics: Dynamics,
    policy: PolicyHeadMTP | Actions,
    schedule: DenoiseSchedule,
    frames_ctx: jax.Array,   # (B, T_ctx, H, W, C), normalized space
    actions_ctx: Actions,
    num_steps: int,
    rng: jax.Array,
    initial_task_embedding: jax.Array | None = None,
    deterministic: bool = False,
    omega: jax.Array | float = 0.0,
    alpha: jax.Array | float = 0.7,
):
    """Autoregressive rollout in pixel space."""
    del omega, alpha

    B, T_ctx, H, W, C = frames_ctx.shape
    obs_shape = (B, 1, H, W, C)
    frames_ctx_orig = frames_ctx

    patch_size = getattr(dynamics.cfg, "patch_size", 8)
    n_tokens = _num_pixel_tokens_from_shape((H, W), patch_size=patch_size)

    window_size = T_ctx + num_steps
    n_agents = policy.cfg.L if isinstance(policy, PolicyHeadMTP) else 0
    caches = dynamics.create_static_caches(
        batch_size=B,
        n_latents=n_tokens,
        window_size=window_size,
        n_agent=n_agents,
        dtype=frames_ctx.dtype,
    )

    emax = int(math.log2(schedule.k_max))
    step_idx_ctx = jnp.full((B, T_ctx), emax, dtype=jnp.int32)
    tau_idx_ctx = jnp.full((B, T_ctx), schedule.k_max - 1, dtype=jnp.int32)

    _, (h_seq, caches) = dynamics(
        actions_ctx,
        step_idx_ctx,
        tau_idx_ctx,
        frames_ctx,
        task_embeddings=initial_task_embedding,
        caches=caches,
        deterministic=True,
    )

    task_embedding = initial_task_embedding[:, -1:] if isinstance(initial_task_embedding, jax.Array) else None
    h_last = h_seq[:, -1:] if isinstance(h_seq, jax.Array) else None

    def scan_step(carry, step_idx):
        h_t, caches_t, rng = carry
        rng, rng_policy = jax.random.split(rng)

        if isinstance(policy, Actions):
            action = policy[:, step_idx]
        else:
            all_actions = policy.sample(h_t, deterministic=deterministic, rng=rng_policy)
            action = all_actions[:, 0, 0, ...]

        obs_next, h_next, caches_next, rng = next_observation(
            dynamics=dynamics,
            schedule=schedule,
            action=action,
            observation_shape=obs_shape,
            rng=rng,
            caches=caches_t,
            task_embedding=task_embedding,
        )
        return (h_next, caches_next, rng), (obs_next[:, 0], action, h_next)

    _, (rollout_frames, rollout_actions, rollout_hidden) = jax.lax.scan(
        scan_step,
        (h_last, caches, rng),
        jnp.arange(num_steps),
    )

    rollout_frames = einops.rearrange(rollout_frames, "t b h w c -> b t h w c")
    rollout_actions = jax.tree.map(lambda x: einops.rearrange(x, "t b ... -> b t ..."), rollout_actions)
    rollout_hidden = (
        einops.rearrange(rollout_hidden, "t b 1 n d -> b t n d")
        if isinstance(rollout_hidden, jax.Array)
        else None
    )

    out_frames = jnp.concatenate((frames_ctx_orig, rollout_frames), axis=1)
    return {
        "frames": out_frames,
        "actions": rollout_actions,
        "hidden_states": rollout_hidden,
        "context_hidden": h_seq,
    }


def video_rollout(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    policy: PolicyHeadMTP | Actions,
    schedule: DenoiseSchedule,
    frames_ctx: jax.Array,
    actions_ctx: Actions,
    num_steps: int,
    rng: jax.Array,
    initial_task_embedding: jax.Array | None = None,
    omega: jax.Array | float = 0.0,
    alpha: jax.Array | float = 0.7,
):
    """End-to-end video rollout."""
    del omega, alpha

    from flax import nnx

    rng, mae_key = jax.random.split(rng)
    rngs = nnx.Rngs(mae=mae_key)
    latents_ctx, _ = tokenizer.encode(
        frames_ctx,
        deterministic=True,
        rngs=rngs,
    )

    # Normalize before rollout
    latents_ctx_norm = tokenizer.latent_normalizer.normalize(latents_ctx)

    rollout_result = latent_rollout(
        dynamics,
        policy,
        schedule,
        latents_ctx_norm,
        actions_ctx,
        num_steps,
        rng,
        initial_task_embedding,
        deterministic=True,
    )

    # Unnormalize before decode
    pred_latents = tokenizer.latent_normalizer.unnormalize(rollout_result["latents"])

    pred_frames, _ = tokenizer.decode(
        pred_latents,
        deterministic=True,
    )
    return jnp.clip(pred_frames, 0, 255).astype(jnp.uint8)
