import math
import einops
import jax
import jax.numpy as jnp
from typing import Tuple 

from .models import Dynamics, KVCache, PolicyHeadMTP, Tokenizer


from flax.struct import dataclass
from flax.typing import VariableDict


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
    tau_values: jax.Array
    tau_indices: jax.Array
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
        tau_values = jnp.linspace(0.0, 1.0, num_steps)
        tau_indices = jnp.arange(num_steps) * (k_max // num_steps)
        
        # Compute noise level for context during autoregressive rollout
        step_idx_ctx = int(jnp.round(-math.log2(1 - tau_ctx)))
        tau_idx_ctx = k_max - k_max // 2**step_idx_ctx
        tau_ctx = 1.0 - 2.0**(-step_idx_ctx) # setting tau_ctx to the exact value such that tau_ctx + step_ctx = 1
        
        return cls(num_steps, k_max, d, step_idx, tau_values, tau_indices, step_idx_ctx, tau_idx_ctx, tau_ctx)
    
# ---------------------------
# Single-step τ-ladder denoiser
# ---------------------------

def next_latent(
    dynamics: Dynamics,
    dyn_vars: VariableDict,
    schedule: DenoiseSchedule,
    action: jax.Array,                 # (B, 1)
    latent_shape: Tuple,                   # (B, 1, n_spatial, D_s)
    rng: jax.Array,
    prefill_length: int | None = None,
    agent_tokens: jax.Array | None = None,  # (B, T_ctx+1, n_agent, d_model)
    caches: KVCache | None = None,
    latents_ctx: jax.Array| None = None,                     # (B, T_ctx, n_spatial, D_s)
    actions_ctx: jax.Array | None = None,                 # (B, T_ctx)
) -> Tuple[jax.Array, jax.Array | None, KVCache | None, jax.Array]:
    """
    JAX-friendly τ-ladder denoiser for a single future latent with KV caching.

    Args:
        dynamics: Dynamics model (Flax Module)
        dyn_vars: Variables for dynamics (params + collections)
        schedule: Precomputed DenoiseSchedule
        action: (B, 1) action for current step
        latent_shape: Tuple representing the shape of the latent (B, 1, n_spatial, D_s)
        prefill_length: Number of ground truth latents passed during prefill.
        rng: Random number generator key
        agent_tokens: Optional agent tokens (B, T_ctx+1, n_agent, d_model)
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
        noise_prefill= jnp.zeros(latents_ctx[:, :prefill_length].shape)
        noise_decode = jax.random.normal(rng_ctx, latents_ctx[:, prefill_length:].shape)
        noise_ctx    = jnp.concatenate([noise_prefill, noise_decode], axis=1)
        latents_ctx_noised = latents_ctx + (1 - schedule.tau_ctx) * noise_ctx

    action = action[:, None] if action.ndim == 1 else action 
    
    def refinement_step(latent_t, s):
        tau_prev, tau_curr = schedule.tau_values[s], schedule.tau_values[s+1]
        alpha = (tau_curr - tau_prev) / jnp.maximum(1.0 - tau_prev, 1e-8)
        
        step_idx = schedule.step_idx
        tau_idx_val = schedule.tau_indices[s] 

        if caches is not None:
            latent_input, actions_input = latent_t, action

            step_indices= jnp.full((B, 1), step_idx,    dtype=jnp.int32)
            tau_indices = jnp.full((B, 1), tau_idx_val, dtype=jnp.int32)

            assert agent_tokens is None or agent_tokens.shape[1] == noisy_latent.shape[1] 
        
        else: # Used only for debugging.
            assert latents_ctx_noised is not None and actions_ctx is not None and prefill_length is not None
            latent_input  = jnp.concatenate([latents_ctx_noised, latent_t], axis=1)  # (B, T_ctx+1, n_spatial, D_s)
            actions_input = jnp.concatenate([actions_ctx, action],   axis=1)  # (B, T_ctx+1)

            decode_length = latents_ctx_noised.shape[1] - prefill_length
            step_idx_prefill= jnp.full((B, prefill_length), step_idx, dtype=jnp.int32)
            step_idx_decode = jnp.full((B, decode_length),  schedule.step_idx_ctx, dtype=jnp.int32)
            step_idx_curr   = jnp.full((B, 1), step_idx, dtype=jnp.int32)
            step_indices    = jnp.concatenate([step_idx_prefill, step_idx_decode, step_idx_curr], axis=1)
            
            tau_idx_prefill= jnp.full((B, prefill_length), schedule.k_max, dtype=jnp.int32)
            tau_idx_decode = jnp.full((B, decode_length), schedule.tau_idx_ctx, dtype=jnp.int32)
            tau_idx_curr   = jnp.full((B, 1), tau_idx_val, dtype=jnp.int32)
            tau_indices    = jnp.concatenate([tau_idx_prefill, tau_idx_decode, tau_idx_curr], axis=1) # (B, T_ctx+1)

        # Dynamics call
        latent_clean_pred_seq, (h_seq, _) = dynamics.apply(dyn_vars, actions_input, step_indices, tau_indices, latent_input, agent_tokens=agent_tokens, deterministic=True, caches=caches)

        latent_clean_pred = latent_clean_pred_seq[:, -1:, :, :]  # (B, 1, n_spatial, D_s)
        h_last = h_seq[:, -1, :, :] if isinstance(h_seq, jax.Array) else h_seq # (B, n_agent, d_model)

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
        latent_noised_caching = latent_t_final*(1-schedule.tau_ctx) + jax.random.normal(new_random_key, shape=latent_t_final.shape, dtype=latent_t_final.dtype)

        _, (h_seq_final, caches_new) = dynamics.apply(dyn_vars, action, step_indices, tau_indices, latent_noised_caching, agent_tokens=agent_tokens, deterministic=True, caches=caches)
        h_last = h_seq_final[:, -1, :, :] if isinstance(h_seq_final, jax.Array) else h_seq_final
    else:
        h_last = h_history[-1] if h_history is not None else None  # (B, n_agent, d_model)
        caches_new = None

    assert isinstance(h_last, jax.Array) or h_last is None

    return latent_t_final, h_last, caches_new, rng  


def latent_rollout(
    dynamics: Dynamics,
    dyn_vars: VariableDict,
    policy: PolicyHeadMTP | jax.Array,
    policy_vars: VariableDict | None,
    schedule: DenoiseSchedule,
    latents_ctx: jax.Array,
    actions_ctx: jax.Array,
    num_steps: int,
    rng: jax.Array,
    initial_agent_tokens: jax.Array | None = None,
):
    """
    TODO: we might want to add the Value head and the Reward head as well
    Autoregressive rollout in latent space.
    
    Args:
        dynamics: Dynamics model.
        dyn_vars: Variables for dynamics.
        policy: Policy model or array of fixed actions (B, num_steps, ...).
        policy_vars: Variables for policy.
        schedule: DenoiseSchedule.
        latents_ctx: (B, T_ctx, n_spatial, D_s) Context latents.
        actions_ctx: (B, T_ctx, ...) Context actions.
        num_steps: Number of steps to unroll.
        rng: Random number generator key.
        initial_agent_tokens: Optional (B, T_ctx, n_agent, D) agent tokens for context.
        
    Returns:
        latents: (B, num_steps, n_spatial, D_s)
        actions: (B, num_steps, ...)
    """
    B, T_ctx, n_spatial, D_s = latents_ctx.shape
    latent_shape = (B, 1, n_spatial, D_s)
    # 1. Initialize caches and process context
    # We need to compute the max window size needed: context + rollout
    window_size = T_ctx + num_steps
    caches = dynamics.create_static_caches(batch_size=B, n_spatial=n_spatial, window_size=window_size)
    
    # Run dynamics on context to prefill caches and get last hidden state
    # Use clean signal for ground truth context 
    step_idx_ctx= jnp.full((B, T_ctx), schedule.step_idx, dtype=jnp.int32)
    tau_idx_ctx = jnp.full((B, T_ctx), schedule.k_max - 1, dtype=jnp.int32)
    
    _, (h_seq, caches) = dynamics.apply(dyn_vars, actions_ctx, step_idx_ctx, tau_idx_ctx, latents_ctx, agent_tokens=initial_agent_tokens, caches=caches, deterministic=True)

    # h_seq: (B, T_ctx, n_agent, D). We need the state at the last context step.
    h_last = h_seq[:, -1] if isinstance(h_seq, jax.Array) else None # (B, n_agent, D)
    
    # 2. Scan loop for rollout
    def scan_step(carry, step_idx):
        h_t, caches_t, rng = carry
        
        # Sample action
        rng, rng_action = jax.random.split(rng)
        
        if isinstance(policy, jax.Array):
            action = policy[:, step_idx]
        else:
            assert policy_vars is not None
            logits = policy.apply(policy_vars, h_t, deterministic=False)
            assert isinstance(logits, jax.Array), "Logits should be a JAX array"
            action = jax.random.categorical(rng_action, logits) # (B, L)
        
        # Predict next latent (denoising)
        latent_next, h_next, caches_next, rng = next_latent(dynamics, dyn_vars, schedule, action, latent_shape, rng, caches=caches_t)
        
        return (h_next, caches_next, rng), latent_next[:,0] # latent_next is (B, 1, n_spatial, D_s) 

    # Run scan
    _, rollout_latents = jax.lax.scan(
        scan_step,
        (h_last, caches, rng),
        jnp.arange(num_steps)
    )
    
    # Unpack results
    rollout_latents = einops.rearrange(rollout_latents, 't b s d -> b t s d')
    out_latents = jnp.concatenate((latents_ctx, rollout_latents), axis=1)
    return out_latents 



def video_rollout(
    tokenizer: Tokenizer,
    tokenizer_vars: VariableDict,
    dynamics: Dynamics,
    dyn_vars: VariableDict,
    policy: PolicyHeadMTP | jax.Array,
    policy_vars: VariableDict | None,
    schedule: DenoiseSchedule,
    frames_ctx: jax.Array,
    actions_ctx: jax.Array,
    num_steps: int,
    rng: jax.Array,
    initial_agent_tokens: jax.Array | None = None,
):
    """
    End-to-end video generation rollout.
    Args:
        tokenizer: Tokenizer model.
        tokenizer_vars: Variables for tokenizer.
        dynamics: Dynamics model.
        dyn_vars: Variables for dynamics.
        policy: Policy model or array of fixed actions.
        policy_vars: Variables for policy.
        schedule: DenoiseSchedule.
        frames_ctx: (B, T_ctx, H, W, C) context frames (0-1 range, unnormalized).
        actions_ctx: (B, T_ctx, ...) Context actions.
        num_steps: Number of steps to unroll.
        rng: Random number generator key.
        initial_agent_tokens: Optional agent tokens.
        packing_factor: Packing factor for tokens.
        dataset_mean: Mean for normalization.
        dataset_std: Std for normalization.
    Returns:
        pred_frames: (B, T_ctx + num_steps, H, W, C)
    """
    
    # Tokenize
    rng, mae_key = jax.random.split(rng)
    latents_ctx, _ = tokenizer.apply(tokenizer_vars,
                                frames_ctx, 
                                packing_factor=dynamics.config.packing_factor, 
                                method=tokenizer.encode, 
                                rngs={"mae": mae_key}, 
                                deterministic=True) # Encode returns (B, T, L, D)
        
    # Latent Rollout
    # Returns (B, num_steps, n_spatial, D_s)
    rollout_latents = latent_rollout(dynamics,
                                     dyn_vars,
                                     policy,
                                     policy_vars,
                                     schedule,
                                     latents_ctx,
                                     actions_ctx,
                                     num_steps,
                                     rng,
                                     initial_agent_tokens)
    
    # Decode
    pred_frames, _ = tokenizer.apply(tokenizer_vars,
                                       rollout_latents,
                                       packing_factor=dynamics.config.packing_factor,
                                       method=tokenizer.decode,
                                       deterministic=True)
        
    return jnp.clip(pred_frames, 0, 255).astype(jnp.uint8)
