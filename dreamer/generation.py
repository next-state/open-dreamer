import math
import einops
import jax
import jax.numpy as jnp
from typing import Any, Tuple
from .models import KVCachesDict, Dynamics, PolicyHeadMTP, Tokenizer
from .actions import Actions
from .utils import normalize_latents, unnormalize_latents
from flax.struct import dataclass


@dataclass
class DenoiseSchedule:
    """
    Precomputed, JAX-friendly schedule for the τ-ladder.

    Attributes:
        num_steps: number of sampling steps (k ∈ {1, 2, 4, ..., k_max}) that you take during inference. In the paper, it's 4.
        k_max: a power of two, maximum noise resolution used during diffusion training. In the paper, it's 256.
        d: Step size d=1/k ∈ {1, 1/2, 1/4, ..., 1/k_max} during inference, where k is num_steps.
        step_idx: log2(k) ∈ {0, 1, 2, ..., log2(K_max)} for denoising.
        emax: log2(k_max), the maximum step index.
        tau_values: signal levels used during the denoising τ = [0, d, 2d, ..., 1 - d, 1].
        tau_indices: indices of the signal levels used during the denoising τ_idx = [0, k, 2k, ..., k_max].
        beta_values: precomputed Euler step mixing coefficients for each step, where beta[s] = (1 - tau[s+1]) / (1 - tau[s]).
        step_idx_ctx: step index for context frames (may differ from step_idx for finer tau_ctx control).
        tau_idx_ctx: tau index for slightly noised context frames, snapped to step_idx_ctx ladder.
        tau_ctx: noise level of context frames during autoregressive rollout.
    """

    num_steps: int
    k_max: int
    d: float
    step_idx: int
    emax: int
    tau_values: jax.Array
    tau_indices: jax.Array
    beta_values: jax.Array
    step_idx_ctx: int
    tau_idx_ctx: int
    tau_ctx: jax.Array

    @classmethod
    def init(cls, num_steps: int, k_max: int = 256, tau_ctx_target: float = 0.9, dtype=jnp.float32) -> "DenoiseSchedule":
        """
        Create a DenoiseSchedule object.
        Args:
            num_steps: Number of steps in the schedule (must be power of 2).
            k_max: Maximum value of k (noise resolution).
            tau_ctx_target: Target noise level for context frames during autoregressive rollout.
                           Will be snapped down to the nearest valid tau on an appropriate ladder.
            dtype: Dtype for precomputed arrays (should match model dtype).
        Returns:
            DenoiseSchedule object.
        """
        assert k_max % num_steps == 0, f"k_max={k_max} must be divisible by num_steps={num_steps}"

        d = 1 / num_steps
        step_idx = int(math.log2(num_steps))
        tau_values = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=dtype)
        tau_indices = jnp.arange(num_steps + 1) * (k_max // num_steps)
        beta_values = (1.0 - tau_values[1:]) / jnp.maximum(1.0 - tau_values[:-1], 1e-8)

        # Snap tau_ctx to an appropriate ladder:
        emax = int(math.log2(k_max))
        if step_idx == emax:
            # Use the same ladder for consistency with empirical training
            step_idx_ctx = step_idx
            K_ctx = num_steps
        else:
            # Use emax-1 ladder for finer control (bootstrap training uses mixed ladders excluding emax)
            step_idx_ctx = emax - 1
            K_ctx = k_max // 2  # emax - 1 ladder has K = k_max / 2
        j_ctx = int(tau_ctx_target * K_ctx)  # floor to ensure some noise
        tau_ctx = jnp.array(j_ctx / K_ctx, dtype=dtype)
        tau_idx_ctx = j_ctx * (k_max // K_ctx)

        return cls(num_steps, k_max, d, step_idx, emax, tau_values, tau_indices, beta_values, step_idx_ctx, tau_idx_ctx, tau_ctx)
    
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
    latents_ctx: jax.Array| None = None,                     # (B, T_ctx, n_spatial, D_s) roughly, padded out to max_len
    actions_ctx: Actions | None = None,
    cur_len: int | jax.Array | None = None,                  # Current length of context sequence when non-cached
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

    latents_ctx_noised = None
    if latents_ctx is not None and caches is None:
        # Apply noise to the entire context array. We use jnp.where to only noise frames >= prefill_length.
        noise_decode = jax.random.normal(rng_ctx, latents_ctx.shape)
        latents_noised = latents_ctx * schedule.tau_ctx + (1 - schedule.tau_ctx) * noise_decode
        
        time_idx = jnp.arange(latents_ctx.shape[1])[None, :, None, None]
        latents_ctx_noised = jnp.where((time_idx >= prefill_length) & (time_idx < cur_len), latents_noised, latents_ctx)

    action = action[:, None, ...]  # expand squeezed-out time dimension

    def refinement_step(latent_t, s):
        beta = schedule.beta_values[s]

        step_idx = schedule.step_idx
        tau_idx_val = schedule.tau_indices[s]

        if caches is not None:
            latent_input, actions_input = latent_t, action

            step_indices= jnp.full((B, 1), step_idx,    dtype=jnp.int32)
            tau_indices = jnp.full((B, 1), tau_idx_val, dtype=jnp.int32)

            assert task_embedding is None or task_embedding.shape[1] == noisy_latent.shape[1], f"task_embedding.shape = {task_embedding.shape}, noisy_latent.shape = {noisy_latent.shape}"
        
        else: # Used only for debugging.
            assert latents_ctx_noised is not None and actions_ctx is not None and prefill_length is not None and cur_len is not None
            # Insert latent_t into latents_ctx_noised at cur_len
            latent_input = jax.lax.dynamic_update_slice(latents_ctx_noised, latent_t, (0, cur_len, 0, 0))
            
            # Insert action into actions_ctx at cur_len
            def update_action(x_c, a_c):
                return jax.lax.dynamic_update_slice(x_c, a_c, (0, cur_len) + (0,) * (a_c.ndim-2))
            actions_input = jax.tree.map(update_action, actions_ctx, action)

            # Create step indices array
            time_idx_2d = jnp.arange(latents_ctx_noised.shape[1])[None, :]
            
            step_idx_curr_arr = jnp.where(time_idx_2d < prefill_length, schedule.emax, schedule.step_idx_ctx)
            step_indices = jnp.where(time_idx_2d == cur_len, step_idx, step_idx_curr_arr)
            
            tau_idx_curr_arr = jnp.where(time_idx_2d < prefill_length, schedule.k_max, schedule.tau_idx_ctx)
            tau_indices = jnp.where(time_idx_2d == cur_len, tau_idx_val, tau_idx_curr_arr)

        # Dynamics call
        latent_clean_pred_seq, (h_seq, _) = dynamics(
            actions_input, step_indices, tau_indices, latent_input,
            task_embeddings=task_embedding, deterministic=True, caches=caches
        )

        if caches is not None:
            latent_clean_pred = latent_clean_pred_seq[:, -1:, :, :]  # (B, 1, n_spatial, D_s)
            h_last = h_seq[:, -1:, :, :] if isinstance(h_seq, jax.Array) else h_seq  # (B, n_agent, d_model)
        else:
            latent_clean_pred = jax.lax.dynamic_slice(latent_clean_pred_seq, (0, cur_len, 0, 0), (B, 1, n_spatial, D_s))
            if isinstance(h_seq, jax.Array):
                h_last = jax.lax.dynamic_slice(h_seq, (0, cur_len, 0, 0), (B, 1, h_seq.shape[2], h_seq.shape[3]))
            elif isinstance(h_seq, type(None)):
                h_last = None
            else:
                h_last = h_seq

        # Per-step mixing toward clean latent
        latent_t_new = beta * latent_t + (1.0 - beta) * latent_clean_pred

        return latent_t_new, h_last

    # Run τ-ladder with JAX control flow using scan to keep carry/output structure consistent.
    latent_t_final, h_history = jax.lax.scan(
        refinement_step,
        noisy_latent,
        jnp.arange(schedule.num_steps),
    )
    
    if caches is not None:
        # Update caches by doing one more forward pass with tau_ctx
        step_indices = jnp.full((B, 1), schedule.step_idx_ctx, dtype=jnp.int32)
        tau_indices = jnp.full((B, 1), schedule.tau_idx_ctx, dtype=jnp.int32)
        
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
    use_kv_cache: bool = False,
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
        use_kv_cache: Whether to use KV caching in dynamics for faster rollout.
        
    Returns:
        Dict with 'latents', 'actions', 'hidden_states', 'context_hidden'
    """
    B, T_ctx, n_spatial, D_s = latents_ctx.shape
    latent_shape = (B, 1, n_spatial, D_s)

    # Normalize context latents for dynamics (keep original for output)
    latents_ctx_orig = latents_ctx
    latents_ctx = normalize_latents(latents_ctx, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)

    # Initialize caches and process context
    window_size = T_ctx + num_steps
    n_agents = policy.cfg.L if isinstance(policy, PolicyHeadMTP) else 0
    caches = dynamics.create_static_caches(batch_size=B, n_latents=n_spatial, window_size=window_size, n_agent=n_agents, dtype=latents_ctx.dtype)

    # Run dynamics on context to prefill caches and get last hidden state
    # Use clean signal for ground truth context
    step_idx_prefill = jnp.full((B, T_ctx), schedule.emax, dtype=jnp.int32)  # tau_idx=k_max was only trained with step_idx=emax (empirical rows)  # FIXME: bootstrap training uses mixed ladders excluding emax, so this might be problematic during shortcut sampling
    tau_idx_prefill = jnp.full((B, T_ctx), schedule.k_max, dtype=jnp.int32)  # tau=1.0

    if use_kv_cache:
        # Dynamics call for prefill
        _, (h_seq, caches) = dynamics(
            actions_ctx, step_idx_prefill, tau_idx_prefill, latents_ctx,
            task_embeddings=initial_task_embedding, caches=caches, deterministic=True
        )

        # h_seq: (B, T_ctx, n_agent, D). We need the state at the last context step.
        task_embedding = initial_task_embedding[:, -1:] if isinstance(initial_task_embedding, jax.Array) else None
        h_last = h_seq[:, -1:] if isinstance(h_seq, jax.Array) else None  # (B, 1, n_agent, D)
    else:
        caches, task_embedding, h_seq, h_last = None, None, None, None

    if not use_kv_cache:
        # Pad context sequences to maximum window size
        pad_len = num_steps
        latents_ctx_padded = jnp.pad(latents_ctx, ((0,0), (0, pad_len), (0,0), (0,0)))
        
        def pad_action(x):
            pad_width = [(0,0)] * x.ndim
            pad_width[1] = (0, pad_len)
            return jnp.pad(x, pad_width)
        actions_ctx_padded = jax.tree.map(pad_action, actions_ctx)
    else:
        latents_ctx_padded = latents_ctx
        actions_ctx_padded = actions_ctx

    # 2. Scan loop for rollout
    def scan_step(carry, step_idx):
        h_t, caches_t, rng, latents_ctx_carry, actions_ctx_carry = carry
        cur_len = T_ctx + step_idx

        # Sample action
        rng, rng_policy = jax.random.split(rng)
        
        if isinstance(policy, Actions):
            action = policy[:, step_idx]  # (B, ...)
        else:
            all_actions = policy.sample(h_t, deterministic=deterministic, rng=rng_policy)  # (B, T, L, ...)
            action = all_actions[:, 0, 0, ...]  # (B, ...) - use first predicted action
        
        # Predict next latent (denoising)
        latent_next, h_next, caches_next, rng = next_latent(
            dynamics, schedule, action, latent_shape, rng, caches=caches_t, task_embedding=task_embedding,
            latents_ctx=latents_ctx_carry, actions_ctx=actions_ctx_carry, prefill_length=T_ctx, cur_len=cur_len
        )
        
        if not use_kv_cache:
            # Re-normalize to append to standard normal context window representation
            latent_next_norm = normalize_latents(latent_next, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)
            latents_ctx_next = jax.lax.dynamic_update_slice(latents_ctx_carry, latent_next_norm, (0, cur_len, 0, 0))
            
            def update_action_carry(x_c, a_c):
                return jax.lax.dynamic_update_slice(x_c, a_c[:, None, ...], (0, cur_len) + (0,) * (a_c.ndim-1))
            actions_ctx_next = jax.tree.map(update_action_carry, actions_ctx_carry, action)
        else:
            latents_ctx_next = latents_ctx_carry
            actions_ctx_next = actions_ctx_carry
        
        return (h_next, caches_next, rng, latents_ctx_next, actions_ctx_next), (latent_next[:, 0], action, h_next) # latent_next is (B, 1, n_spatial, D_s) 

    # Run scan
    _, (rollout_latents, rollout_actions, rollout_hidden) = jax.lax.scan(
        scan_step,
        (h_last, caches, rng, latents_ctx_padded, actions_ctx_padded),
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
    policy: PolicyHeadMTP | Actions,
    schedule: DenoiseSchedule,
    frames_ctx: jax.Array,
    actions_ctx: Actions,
    num_steps: int,
    rng: jax.Array,
    initial_task_embedding: jax.Array | None = None,
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
    )

    # Decode
    pred_frames, _ = tokenizer.decode(
        rollout_result['latents'],
        deterministic=True
    )

    return jnp.clip(pred_frames, 0, 255).astype(jnp.uint8)
