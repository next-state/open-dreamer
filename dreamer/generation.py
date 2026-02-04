import math
import einops
import jax
import jax.numpy as jnp
from typing import Any, Tuple
from .models import KVCachesDict, Dynamics, PolicyHeadMTP, Tokenizer
from .actions import Actions
from .utils import normalize_latents, unnormalize_latents
from flax.struct import dataclass
from flax import nnx


@dataclass
class DenoiseSchedule:
    """
    Precomputed, JAX-friendly schedule for the τ-ladder.

    Attributes:
        num_steps: number of sampling steps (k ∈ {1, 2, 4, ..., k_max}) that you take during inference. In the paper, it's 4.
        k_max: a power of two, maximum noise resolution used during diffusion training. In the paper, it's 256.
        d: Step size d=1/k ∈ {1, 1/2, 1/4, ..., 1/k_max} during inference, where k is num_steps.
        step_idx: log2(k) ∈ {0, 1, 2, ..., log2(K_max)}.
        tau_values: signal levels used during the denoising τ = [0, d, 2d, ..., 1 - d, 1].
        tau_indices: indices of the signal levels used during the denoising τ_idx = [0, k, 2k, ..., k_max].
        tau_idx_ctx: we pass slightly noised context frames, the index of that noise level (0.1 in the paper) is tau_idx_ctx.
        step_idx_ctx: the index of the noise level that starting from tau_idx_ctx brings you to 1.
        tau_ctx: noise level of context frames during autoregressive rollout.
    """

    num_steps: int
    k_max: int
    d: float
    step_idx: int
    tau_values: tuple[float, ...]
    tau_indices: tuple[int, ...]
    step_idx_ctx: int
    tau_idx_ctx: int
    tau_ctx: float

    @classmethod
    def init(cls, num_steps: int, k_max: int = 256, tau_ctx=0.9) -> "DenoiseSchedule":
        """
        Create a DenoiseSchedule object.
        Args:
            num_steps: Number of steps in the schedule.
            k_max: Maximum value of k.
        Returns:
            DenoiseSchedule object.
        """
        assert k_max % num_steps == 0, f"k_max={k_max} must be divisible by num_steps={num_steps}"
        
        d = 1 / num_steps
        step_idx = int(math.log2(num_steps))
        tau_values = tuple(i * d for i in range(num_steps + 1))
        tau_indices = tuple(i * (k_max // num_steps) for i in range(num_steps))
        
        # Compute noise level for context during autoregressive rollout
        step_idx_ctx = int(round(-math.log2(1.0 - tau_ctx)))
        tau_idx_ctx = k_max - k_max // (2 ** step_idx_ctx)
        # set tau_ctx to the exact value such that tau_ctx + step_ctx = 1
        tau_ctx = 1.0 - 2.0 ** (-step_idx_ctx)
        
        return cls(num_steps, k_max, d, step_idx, tau_values, tau_indices, step_idx_ctx, tau_idx_ctx, tau_ctx)
    
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
    task_embedding: jax.Array | None = None,  # (B, T_ctx+1, n_agent, d_model)
    caches: KVCachesDict | None = None,
    latents_ctx: jax.Array| None = None,                     # (B, T_ctx, n_spatial, D_s)
    actions_ctx: Actions | None = None,
    use_actions: bool = True,
) -> Tuple[jax.Array, jax.Array | None, KVCachesDict | None, jax.Array]:
    """
    JAX-friendly τ-ladder denoiser for a single future latent with KV caching.

    Args:
        dynamics: Dynamics NNX model
        schedule: Precomputed DenoiseSchedule
        action: action for current step
        latent_shape: Tuple representing the shape of the latent (B, 1, n_spatial, D_s)
        prefill_length: Number of ground truth latents passed during prefill.
        rng: Random number generator key
        task_embedding: Optional agent tokens (B, T_ctx+1, n_agent, d_model)
        caches: KV cache for context frames (from previous finalized frames)
        latents_ctx: Optional context latents (B, T_ctx, n_spatial, D_s) used for debugging/non-cached mode
        actions_ctx: Optional context actions (B, T_ctx) used for debugging/non-cached mode

    Returns:
        Tuple containing:
        - latent_t_final: The denoised latent (B, 1, n_spatial, D_s)
        - h_last: The final hidden state from dynamics (B, n_agent, d_model)
        - caches_new: The updated KV cache
    """
    rng, rng_latent, rng_ctx = jax.random.split(rng, 3)
    noisy_latent = jax.random.normal(rng_latent, latent_shape)
    B = latent_shape[0]

    # Null out actions if not using them
    if not use_actions:
        action = jax.tree.map(lambda _: None, action)
        if actions_ctx is not None:
            actions_ctx = jax.tree.map(lambda _: None, actions_ctx)

    latents_ctx_noised = None
    if latents_ctx is not None and caches is None:
        noise_prefill = jnp.zeros(latents_ctx[:, :prefill_length].shape)
        noise_decode = jax.random.normal(rng_ctx, latents_ctx[:, prefill_length:].shape)
        noise_ctx = jnp.concatenate([noise_prefill, noise_decode], axis=1)
        latents_ctx_noised = latents_ctx * schedule.tau_ctx + (1 - schedule.tau_ctx) * noise_ctx

    action = action[:, None, ...]  # expand squeezed-out time dimension
    
    # Convert to arrays for dynamic indexing in scan
    tau_values_arr = jnp.asarray(schedule.tau_values)
    tau_indices_arr = jnp.asarray(schedule.tau_indices)
    
    def refinement_step(latent_t, s):
        tau_prev, tau_curr = tau_values_arr[s], tau_values_arr[s+1]
        alpha = (tau_curr - tau_prev) / jnp.maximum(1.0 - tau_prev, 1e-8)

        step_idx = schedule.step_idx
        tau_idx_val = tau_indices_arr[s]

        if caches is not None:
            latent_input, actions_input = latent_t, action

            step_indices= jnp.full((B, 1), step_idx,    dtype=jnp.int32)
            tau_indices = jnp.full((B, 1), tau_idx_val, dtype=jnp.int32)

            assert task_embedding is None or task_embedding.shape[1] == noisy_latent.shape[1], f"task_embedding.shape = {task_embedding.shape}, noisy_latent.shape = {noisy_latent.shape}"
        
        else: # Used only for debugging.
            assert latents_ctx_noised is not None and actions_ctx is not None and prefill_length is not None
            latent_input  = jnp.concatenate([latents_ctx_noised, latent_t], axis=1)  # (B, T_ctx+1, n_spatial, D_s)
            actions_input = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), actions_ctx, action)  # (B, T_ctx+1)

            decode_length = latents_ctx_noised.shape[1] - prefill_length
            step_idx_prefill= jnp.full((B, prefill_length), step_idx, dtype=jnp.int32)
            step_idx_decode = jnp.full((B, decode_length),  schedule.step_idx_ctx, dtype=jnp.int32)
            step_idx_curr   = jnp.full((B, 1), step_idx, dtype=jnp.int32)
            step_indices    = jnp.concatenate([step_idx_prefill, step_idx_decode, step_idx_curr], axis=1)
            
            tau_idx_prefill= jnp.full((B, prefill_length), schedule.k_max - 1, dtype=jnp.int32)
            tau_idx_decode = jnp.full((B, decode_length), schedule.tau_idx_ctx, dtype=jnp.int32)
            tau_idx_curr   = jnp.full((B, 1), tau_idx_val, dtype=jnp.int32)
            tau_indices    = jnp.concatenate([tau_idx_prefill, tau_idx_decode, tau_idx_curr], axis=1)  # (B, T_ctx+1)

        # Dynamics call
        latent_clean_pred_seq, (h_seq, _) = dynamics(
            actions_input, step_indices, tau_indices, latent_input,
            task_embeddings=task_embedding, deterministic=True, caches=caches
        )

        latent_clean_pred = latent_clean_pred_seq[:, -1:, :, :]  # (B, 1, n_spatial, D_s)
        h_last = h_seq[:, -1:, :, :] if isinstance(h_seq, jax.Array) else h_seq  # (B, n_agent, d_model)

        # Per-step mixing toward clean latent
        latent_t_new = (1.0 - alpha) * latent_t + alpha * latent_clean_pred

        return latent_t_new, h_last

    # Run τ-ladder with JAX control flow using scan to keep carry/output structure consistent.
    latent_t_final, h_history = jax.lax.scan(
        refinement_step,
        noisy_latent,
        jnp.arange(schedule.num_steps),
    )
    
    if caches is not None:
        # update caches by doing one more refinement step with tau_ctx
        step_indices= jnp.full((B, 1), schedule.step_idx_ctx, dtype=jnp.int32)
        tau_indices = jnp.full((B, 1), schedule.tau_idx_ctx,  dtype=jnp.int32)
        
        rng, new_random_key = jax.random.split(rng)
        latent_noised_caching = latent_t_final * schedule.tau_ctx + (1 - schedule.tau_ctx) * jax.random.normal(new_random_key, shape=latent_t_final.shape, dtype=latent_t_final.dtype)

        # Dynamics call
        _, (h_seq_final, caches_new) = dynamics(
            action, step_indices, tau_indices, latent_noised_caching,
            task_embeddings=task_embedding, deterministic=True, caches=caches
        )
        h_last = h_seq_final[:, -1:, :, :] if isinstance(h_seq_final, jax.Array) else h_seq_final
    else:
        h_last = h_history[-1] if h_history is not None else None  # (B, n_agent, d_model)
        caches_new = None

    assert isinstance(h_last, jax.Array) or h_last is None

    # Unnormalize output so caller receives latents in original space
    latent_t_final = unnormalize_latents(latent_t_final, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)

    return latent_t_final, h_last, caches_new, rng

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
) -> Tuple[jax.Array, jax.Array | None, KVCachesDict | None, Any, jax.Array]:
    """
    Generate next frame using dynamics model and decode to pixels.

    Args:
        tokenizer: Tokenizer NNX model for decoding
        dynamics: Dynamics NNX model
        schedule: Denoising schedule
        action: Single action to condition on where the time dimension is squeezed (B, ...)
        latent_shape: Shape of latent (B, 1, n_spatial, D_s)
        dynamics_cache: KV cache for dynamics model from previous steps
        tokenizer_cache: KV cache for tokenizer decoder from previous steps
        rng: Random key
        task_embedding: Optional task embedding (currently unused)
        
    Returns:
        Tuple of (frame as jax.Array, h_last, updated dynamics cache, updated tokenizer cache, updated rng)
    """
    # Generate next latent using τ-ladder denoising
    latent, h_last, dynamics_cache_updated, rng = next_latent(
        dynamics=dynamics,
        schedule=schedule,
        action=action,
        latent_shape=latent_shape,
        rng=rng,
        prefill_length=None,  # No prefill for interactive generation
        task_embedding=task_embedding,
        caches=dynamics_cache,
    )
    
    # Decoder call
    frame, tokenizer_cache_updated = tokenizer.decode(
        latent,
        caches=tokenizer_cache,
        deterministic=True,
    )
    
    # Clip to valid range (keep as JAX array)
    # frame shape: (B, 1, H, W, C)
    frame = jnp.clip(frame, 0, 255).astype(jnp.uint8)
    
    return frame, h_last, dynamics_cache_updated, tokenizer_cache_updated, rng

@nnx.jit(static_argnames=('schedule', 'num_steps', 'deterministic', 'use_actions'))
def latent_rollout(
    dynamics: Dynamics,
    policy: PolicyHeadMTP | Actions | None,
    schedule: DenoiseSchedule,
    latents_ctx: jax.Array,
    actions_ctx: Actions,
    num_steps: int,
    rng: jax.Array,
    initial_task_embedding: jax.Array | None = None,
    deterministic: bool = False,
    use_actions: bool = True,
):
    """
    Autoregressive rollout in latent space.

    Args:
        dynamics: Dynamics NNX model.
        policy: Policy NNX model or sequence of actions.
        schedule: DenoiseSchedule.
        latents_ctx: (B, T_ctx, n_spatial, D_s) Context latents.
        actions_ctx: Context actions.
        num_steps: Number of steps to unroll.
        rng: Random number generator key
        initial_task_embedding: Optional (B, T_ctx, n_agent, D) agent tokens for context.
        deterministic: Whether to sample deterministic actions from the policy.
        
    Returns:
        Dict with 'latents', 'actions', 'hidden_states', 'context_hidden'
    """
    B, T_ctx, n_spatial, D_s = latents_ctx.shape
    latent_shape = (B, 1, n_spatial, D_s)

    # Null out actions if not using them
    if not use_actions:
        actions_ctx = jax.tree.map(lambda _: None, actions_ctx)

    # Normalize context latents for dynamics (keep original for output)
    latents_ctx_orig = latents_ctx
    latents_ctx = normalize_latents(latents_ctx, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)

    # Initialize caches and process context
    window_size = T_ctx + num_steps
    n_agents = policy.cfg.L if isinstance(policy, PolicyHeadMTP) else 0 if policy is None else 0
    caches = dynamics.create_static_caches(batch_size=B, n_latents=n_spatial, window_size=window_size, n_agent=n_agents, dtype=latents_ctx.dtype)

    # Run dynamics on context to prefill caches and get last hidden state
    # Use clean signal for ground truth context
    emax = int(math.log2(schedule.k_max))  # Use finest step size (emax) for prefill
    step_idx_ctx = jnp.full((B, T_ctx), emax, dtype=jnp.int32)
    tau_idx_ctx = jnp.full((B, T_ctx), schedule.k_max - 1, dtype=jnp.int32)

    # Dynamics call for prefill
    _, (h_seq, caches) = dynamics(
        actions_ctx, step_idx_ctx, tau_idx_ctx, latents_ctx,
        task_embeddings=initial_task_embedding, caches=caches, deterministic=True
    )

    # h_seq: (B, T_ctx, n_agent, D). We need the state at the last context step.
    task_embedding = initial_task_embedding[:, -1:] if isinstance(initial_task_embedding, jax.Array) else None
    h_last = h_seq[:, -1:] if isinstance(h_seq, jax.Array) else None  # (B, 1, n_agent, D)

    # 2. Scan loop for rollout
    def scan_step(carry, step_idx):
        h_t, caches_t, rng = carry

        # Sample action
        rng, rng_policy = jax.random.split(rng)
        
        if not use_actions or policy is None:
            action = Actions()  # Null action
        elif isinstance(policy, Actions):
            action = policy[:, step_idx]  # (B, ...)
        else:
            all_actions = policy.sample(h_t, deterministic=deterministic, rng=rng_policy)  # (B, T, L, ...)
            action = all_actions[:, 0, 0, ...]  # (B, ...) - use first predicted action
        
        # Predict next latent (denoising)
        latent_next, h_next, caches_next, rng = next_latent(
            dynamics, schedule, action, latent_shape, rng, caches=caches_t, task_embedding=task_embedding,
            use_actions=use_actions,
        )
        
        return (h_next, caches_next, rng), (latent_next[:, 0], action, h_next) # latent_next is (B, 1, n_spatial, D_s) 

    # Run scan
    _, (rollout_latents, rollout_actions, rollout_hidden) = jax.lax.scan(
        scan_step,
        (h_last, caches, rng),
        jnp.arange(num_steps)
    )

    # Unpack results
    rollout_latents = einops.rearrange(rollout_latents, 't b s d -> b t s d')
    rollout_actions = jax.tree.map(lambda x: einops.rearrange(x, 't b ... -> b t ...'), rollout_actions)
    # h_next has shape (B, 1, n_agent, d_model), so scan output is (t, B, 1, n_agent, d_model)
    rollout_hidden = einops.rearrange(rollout_hidden, 't b 1 n d -> b t n d') if isinstance(rollout_hidden, jax.Array) else None

    out_latents = jnp.concatenate((latents_ctx_orig, rollout_latents), axis=1)

    return {
        'latents': out_latents,
        'actions': rollout_actions,
        'hidden_states': rollout_hidden,
        'context_hidden': h_seq,
    } 

def video_rollout(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    policy: PolicyHeadMTP | Actions | None,
    schedule: DenoiseSchedule,
    frames_ctx: jax.Array,
    actions_ctx: Actions,
    num_steps: int,
    rng: jax.Array,
    initial_task_embedding: jax.Array | None = None,
    use_actions: bool = True,
):
    """
    End-to-end video generation rollout.

    Args:
        tokenizer: Tokenizer NNX model.
        dynamics: Dynamics NNX model.
        policy: Policy NNX model or sequence of actions.
        schedule: DenoiseSchedule.
        frames_ctx: (B, T_ctx, H, W, C) context frames (0-255 range).
        actions_ctx: Context actions.
        num_steps: Number of steps to unroll.
        rng: Random number generator key.
        initial_task_embedding: Optional task tokens for context.
    Returns:
        pred_frames: (B, T_ctx + num_steps, H, W, C)
    """
    from flax import nnx

    # Tokenize
    rng, mae_key = jax.random.split(rng)
    rngs = nnx.Rngs(mae=mae_key)

    latents_ctx, _ = tokenizer.encode(
        frames_ctx,
        deterministic=True,
        rngs=rngs
    )  # Encode returns (B, T, L, D)

    # Latent Rollout
    # Returns dict with 'latents', 'actions', 'hidden_states', 'context_hidden'
    rollout_result = latent_rollout(
        dynamics,
        policy,
        schedule,
        latents_ctx,
        actions_ctx,
        num_steps,
        rng,
        initial_task_embedding,
        deterministic=True,  # use deterministic policy for visualization
        use_actions=use_actions,
    )

    # Decode
    pred_frames, _ = tokenizer.decode(
        rollout_result['latents'],
        deterministic=True
    )

    return jnp.clip(pred_frames, 0, 255).astype(jnp.uint8)
