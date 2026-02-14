import math
import einops
import jax
import jax.numpy as jnp
from typing import Any, Tuple
from .models import KVCachesDict, Dynamics, PolicyHeadMTP, Tokenizer
from .actions import Actions, shift_actions
from .utils import normalize_latents, unnormalize_latents
from flax.struct import dataclass
from flax import nnx


# ---------------------------
# Denoise Schedule
# ---------------------------

@dataclass
class DenoiseSchedule:
    """Schedule for shortcut forcing denoising.

    Attributes:
        n_steps: K, number of denoising steps per frame
        k_max: maximum discretization level (same as dynamics.cfg.k_max)
        tau_ctx: signal level for context corruption (1.0 = no corruption, 0.1 = paper default)
    """
    n_steps: int
    k_max: int
    tau_ctx: float = 1.0

    @classmethod
    def init(cls, n_steps: int, k_max: int, tau_ctx: float = 1.0) -> 'DenoiseSchedule':
        """Factory method: create schedule with given parameters."""
        return cls(n_steps=n_steps, k_max=k_max, tau_ctx=tau_ctx)


# ---------------------------
# Single-step τ-ladder denoiser
# ---------------------------

def next_latent(
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    action: Actions,
    latent_shape: Tuple,                   # (B, 1, n_spatial, D_s)
    rng: jax.Array,
    prefill_length: int | None = None,
    task_embedding: jax.Array | None = None,  # (B, T_ctx+1, n_agent, d_model) or (B, 1, n_agent, d_model)
    caches: KVCachesDict | None = None,
    latents_ctx: jax.Array | None = None,                     # (B, T_ctx, n_spatial, D_s)
    actions_ctx: Actions | None = None,
) -> Tuple[jax.Array, jax.Array | None, KVCachesDict | None, jax.Array]:
    """K-step shortcut forcing denoising for one new latent frame.

    Combines optional context prefill + K Euler denoising steps + final cache update.

    Args:
        dynamics: Dynamics NNX model
        schedule: DenoiseSchedule with n_steps, k_max, tau_ctx
        action: (B, 1, ...) Actions for the new frame (shifted: action that produced this frame)
        latent_shape: (B, 1, n_latents, d_bottleneck) shape for new frame
        rng: Random key
        prefill_length: optional context length (used for sliding window attention)
        task_embedding: (B, T_ctx+1, n_agent, d_model) for prefill, or (B, 1, n_agent, d_model) for non-prefill
        caches: pre-built KV cache from previous frames, or None for first call
        latents_ctx: (B, T_ctx, n_latents, d_bottleneck) context latents for prefill, or None
        actions_ctx: (B, T_ctx) context actions for prefill (will be shifted), or None

    Returns:
        (z_clean, h_t, new_caches, rng): Clean latent, hidden states, updated KV caches, updated RNG
    """
    B, T_new, n_latents, D = latent_shape
    K = schedule.n_steps
    k_max = schedule.k_max
    d_py = 1.0 / K  # Python float
    step_idx = int(math.log2(K))  # log2(K) = log2(1/d)
    emax = int(math.log2(k_max))

    # --- Normalize context latents if provided ---
    if latents_ctx is not None:
        latents_ctx_normed = normalize_latents(latents_ctx, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)
    else:
        latents_ctx_normed = None

    # --- Sample initial noise ---
    rng, noise_key = jax.random.split(rng)
    z_current = jax.random.normal(noise_key, latent_shape, dtype=jnp.bfloat16)

    # --- CONTEXT PREFILL (only when latents_ctx provided) ---
    context_caches = caches
    if latents_ctx_normed is not None:
        T_ctx = latents_ctx_normed.shape[1]

        # Apply context corruption
        tau_ctx_val = schedule.tau_ctx
        if tau_ctx_val < 1.0:
            rng, ctx_key = jax.random.split(rng)
            ctx_noise = jax.random.normal(ctx_key, latents_ctx_normed.shape, dtype=latents_ctx_normed.dtype)
            tau_c = jnp.asarray(tau_ctx_val, dtype=latents_ctx_normed.dtype)
            latents_ctx_input = (1.0 - tau_c) * ctx_noise + tau_c * latents_ctx_normed
            tau_idx_ctx = jnp.full((B, T_ctx), jnp.round(tau_c * k_max).astype(jnp.int32), dtype=jnp.int32)
        else:
            latents_ctx_input = latents_ctx_normed
            tau_idx_ctx = jnp.full((B, T_ctx), k_max, dtype=jnp.int32)

        step_idx_ctx = jnp.full((B, T_ctx), emax, dtype=jnp.int32)

        # Shift context actions for dynamics model
        actions_ctx_shifted = shift_actions(actions_ctx, dynamics.cfg.categorical_action_dim)

        # Extract context task embedding if provided
        task_emb_ctx = task_embedding[:, :T_ctx] if task_embedding is not None else None

        # Run dynamics on context to warm up KV cache
        _, (_, context_caches) = dynamics(
            actions_ctx_shifted,
            step_idx_ctx,
            tau_idx_ctx,
            latents_ctx_input,
            context_length=prefill_length,
            deterministic=True,
            caches=None,
            task_embeddings=task_emb_ctx,
        )

    # --- K DENOISING STEPS via fori_loop ---
    step_idx_arr = jnp.full((B, 1), step_idx, dtype=jnp.int32)
    task_emb_new = task_embedding[:, -1:] if task_embedding is not None else None

    # Convert to JAX scalars for use in fori_loop body
    d_jnp = jnp.array(d_py)
    k_max_jnp = jnp.array(k_max, dtype=jnp.int32)
    n_steps_jnp = jnp.array(K, dtype=jnp.int32)

    def denoise_step(k, z_curr):
        """One denoising step: run dynamics once, apply Euler step."""
        tau = k * d_jnp  # JAX float scalar
        tau_idx_val = (k * k_max_jnp // n_steps_jnp).astype(jnp.int32)
        tau_idx_arr = jnp.broadcast_to(tau_idx_val, (B, 1))

        # Forward pass (returned caches are discarded to keep context frozen)
        x1_hat, _ = dynamics(
            action,
            step_idx_arr,
            tau_idx_arr,
            z_curr,
            deterministic=True,
            caches=context_caches,
            task_embeddings=task_emb_new,
        )

        # Euler step: b = (x1_hat - z_curr) / (1 - tau), z_new = z_curr + b * d
        b = (x1_hat - z_curr) / jnp.maximum(1.0 - tau, 1e-8)
        return z_curr + b * d_jnp

    z_current = jax.lax.fori_loop(0, K, denoise_step, z_current)

    # --- CACHE UPDATE with clean frame ---
    tau_clean = jnp.full((B, 1), k_max, dtype=jnp.int32)  # tau=1.0
    step_clean = jnp.full((B, 1), emax, dtype=jnp.int32)

    _, (h_t, new_caches) = dynamics(
        action,
        step_clean,
        tau_clean,
        z_current,
        deterministic=True,
        caches=context_caches,
        task_embeddings=task_emb_new,
    )

    # --- Unnormalize and return ---
    z_out = unnormalize_latents(z_current, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)
    return z_out, h_t, new_caches, rng

def next_frame(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    action: Actions,
    latent_shape: Tuple,
    dynamics_cache: KVCachesDict | None,
    tokenizer_cache: KVCachesDict | None,
    rng: jax.Array,
    task_embedding: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array | None, KVCachesDict | None, KVCachesDict | None, jax.Array]:
    """Single frame: denoise latent then decode to video.

    Args:
        tokenizer: Tokenizer NNX model with decode method
        dynamics: Dynamics NNX model
        schedule: DenoiseSchedule
        action: (B, 1) Actions for the new frame
        latent_shape: (B, 1, n_latents, d_bottleneck)
        dynamics_cache: KV cache for dynamics model
        tokenizer_cache: KV cache for tokenizer decoder
        rng: Random key
        task_embedding: (B, 1, n_agent, d_model) task embedding for the new frame

    Returns:
        (frame, h_t, new_dynamics_cache, new_tokenizer_cache, rng)
    """
    # Denoise latent
    z_clean, h_t, new_dynamics_cache, rng = next_latent(
        dynamics,
        schedule,
        action,
        latent_shape,
        rng,
        caches=dynamics_cache,
        task_embedding=task_embedding,
    )

    # Decode to video frame
    z_clean_b16 = z_clean.astype(jnp.bfloat16)
    frame, new_tokenizer_cache = tokenizer.decode(z_clean_b16, caches=tokenizer_cache, deterministic=True)

    return frame, h_t, new_dynamics_cache, new_tokenizer_cache, rng

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
) -> dict:
    """Autoregressive latent frame generation.

    Args:
        dynamics: Dynamics NNX model
        policy: Either Actions (ground-truth future actions) or PolicyHeadMTP (learned policy)
        schedule: DenoiseSchedule
        latents_ctx: (B, T_ctx, n_latents, d_bottleneck) context latents (clean)
        actions_ctx: (B, T_ctx) context actions (raw, will be shifted for first frame)
        num_steps: number of future frames to generate
        rng: Random key
        initial_task_embedding: (B, T_ctx, n_agent, d_model) task embedding for all frames
        deterministic: whether to use deterministic sampling for policy

    Returns:
        dict with 'latents' key: (B, T_ctx+num_steps, n_latents, d_bottleneck)
    """
    B, T_ctx, n_latents, D = latents_ctx.shape

    latents_out = [latents_ctx]
    caches = None
    h_t = None
    rng_state = rng

    for t in range(num_steps):
        # --- Determine shifted action for this new frame ---
        # The "shifted" action for frame t is the action taken at frame t-1
        if isinstance(policy, Actions):
            # Ground truth actions: policy has shape (B, num_steps, ...)
            if t == 0:
                # First new frame: use last context action
                action_t = actions_ctx[:, -1:]
            else:
                # Subsequent frames: use previous future action (shifted convention)
                action_t = policy[:, t - 1 : t]
        else:
            # PolicyHeadMTP: sample from dynamics hidden states
            if h_t is None:
                # First frame: use last context action as the "shifted" action
                action_t = actions_ctx[:, -1:]
            else:
                # Sample from policy using hidden states from previous frame
                rng_state, sample_key = jax.random.split(rng_state)
                sampled_actions = policy.sample(h_t, deterministic=deterministic, rng=sample_key)
                # Take the first L=0 prediction (only need next-step action)
                action_t = jax.tree.map(lambda x: x[:, :, 0] if x is not None else None, sampled_actions)

        # --- Task embedding for this new frame ---
        if initial_task_embedding is not None:
            single_emb = initial_task_embedding[:, :1, :, :]  # (B, 1, n_agent, d_model)
            if caches is None:
                # First call: provide context + 1 new frame slot
                task_emb = jnp.concatenate(
                    [initial_task_embedding, single_emb], axis=1
                )  # (B, T_ctx+1, n_agent, d_model)
            else:
                # Subsequent calls: just 1 slot for the new frame
                task_emb = single_emb  # (B, 1, n_agent, d_model)
        else:
            task_emb = None

        # --- Denoise one new frame ---
        z_clean, h_t, caches, rng_state = next_latent(
            dynamics,
            schedule,
            action_t,
            (B, 1, n_latents, D),
            rng_state,
            prefill_length=T_ctx if caches is None else None,
            task_embedding=task_emb,
            caches=caches,
            latents_ctx=latents_ctx if caches is None else None,
            actions_ctx=actions_ctx if caches is None else None,
        )

        latents_out.append(z_clean)

    # Concatenate context + generated frames
    all_latents = jnp.concatenate(latents_out, axis=1)
    return {"latents": all_latents}

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
) -> dict:
    """Autoregressive video frame generation.

    Encodes context frames, rolls out latents autoregressively, decodes to video.

    Args:
        tokenizer: Tokenizer NNX model (has encode/decode)
        dynamics: Dynamics NNX model
        policy: Either Actions (ground-truth actions) or PolicyHeadMTP (learned policy)
        schedule: DenoiseSchedule
        frames_ctx: (B, T_ctx, H, W, C) context video frames in [0, 255]
        actions_ctx: (B, T_ctx) context actions
        num_steps: number of future frames to generate
        rng: Random key
        initial_task_embedding: (B, T_ctx, n_agent, d_model) optional task embedding

    Returns:
        dict with 'frames' (generated frames), 'latents' (generated latents)
    """
    # Encode context frames to latents
    rng, mae_key = jax.random.split(rng)
    rngs = nnx.Rngs(mae=mae_key)
    latents_ctx, _ = tokenizer.encode(frames_ctx, deterministic=True, rngs=rngs)

    # Autoregressive latent rollout
    result = latent_rollout(
        dynamics,
        policy,
        schedule,
        latents_ctx,
        actions_ctx,
        num_steps,
        rng,
        initial_task_embedding=initial_task_embedding,
        deterministic=True,
    )

    # Decode all latents to frames
    @nnx.jit
    def decode_all(z):
        frames, _ = tokenizer.decode(z, deterministic=True)
        return frames

    pred_frames = decode_all(result["latents"])

    return {"frames": pred_frames, "latents": result["latents"]}
