import math
import einops
from flax.typing import VariableDict
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple 

from .models import Dynamics, KVCache, PolicyHeadMTP


from flax.struct import dataclass


@dataclass
class DenoiseSchedule:
    """
    Precomputed, JAX-friendly schedule for the τ-ladder.

    Attributes:
        num_steps: Number of steps in the schedule, also a power of two.
        k_max: A power of two, it represents the maximum number of denoising steps possible.
        d: Step size d=1/num_steps ∈ {1, 1/2, 1/4, ..., 1/K_max}.
        step_idx: log2(1/d) ∈ {0, 1, 2, ..., log2(K_max)}.
        tau_values: Signal levels used during the denoising τ = [0, 1/d, 2/d, ..., 1 - 1/d, 1].
        tau_indices: Indices of the signal levels used during the denoising τ_idx = [0, d, 2d, ..., K_max].
    """

    num_steps: int
    k_max: int
    d: float
    step_idx: int
    tau_values: jax.Array
    tau_idx: jax.Array
    tau_idx_ctx: int
    step_idx_ctx: int = 3

    @classmethod
    def create(cls, num_steps: int, k_max: int = 256) -> "DenoiseSchedule":
        """
        Create a DenoiseSchedule object.
        Args:
            num_steps: Number of steps in the schedule.
            k_max: Maximum value of k.
        Returns:
            DenoiseSchedule object.
        """
        assert k_max % num_steps == 0, f"k_max={k_max} must be divisible by num_steps={num_steps}"
        
        d = 1/num_steps
        step_idx = int(math.log2(num_steps))
        tau_values = jnp.linspace(0.0, 1.0, num_steps + 1)
        tau_indices = jnp.arange(num_steps+1)*(k_max//num_steps)
        
        # assuming the context has τ=0.9
        step_idx_ctx = 3 # this is because the steps size d=0.1 and so the idx = log2(1/d) = 3.32
        tau_idx_ctx = k_max - 2**step_idx_ctx # this is a bit more precise than just doing τ=0.9
        
        return cls(num_steps, k_max, d, step_idx, tau_values, tau_indices, tau_idx_ctx)
    
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
    agent_tokens: jax.Array | None = None,  # (B, T_ctx+1, n_agent, d_model)
    caches: KVCache | None = None,
    latents_ctx: jax.Array| None = None,                     # (B, T_ctx, n_spatial, D_s)
    actions_ctx: jax.Array | None = None,                 # (B, T_ctx)
) -> Tuple[jax.Array, jax.Array | None, KVCache | None, jax.Array]:
    """
    JAX-friendly τ-ladder denoiser for a single future latent with KV caching.

    - Uses a precomputed schedule (τ_seq, α_seq, signal_idx_seq, step_idx).
    - Contains only JAX ops in the inner loop (no Python branching on traced values).
    - Returns the denoised latent, the final hidden state h_t from dynamics, and the updated KV cache.

    Args:
        dynamics: Dynamics model (Flax Module)
        dyn_vars: Variables for dynamics (params + collections)
        schedule: Precomputed DenoiseSchedule
        action: (B, 1) action for current step
        latent_shape: Tuple representing the shape of the latent (B, 1, n_spatial, D_s)
        rng: Random number generator key
        agent_tokens: Optional agent tokens (B, T_ctx+1, n_agent, d_model)
        caches: KV cache for context frames (from previous finalized frames)
        latents_ctx: Optional context latents (B, T_ctx, n_spatial, D_s) used for debugging/non-cached mode
        actions_ctx: Optional context actions (B, T_ctx) used for debugging/non-cached mode

    Returns:
        Tuple containing:
        - z_t_final: The denoised latent (B, 1, n_spatial, D_s)
        - h_last: The final hidden state from dynamics (B, n_agent, d_model)
        - caches_last: The updated KV cache
        - rng: Updated random number generator key
    """
    rng, rng_latent = jax.random.split(rng, 2)
    noisy_latent = jax.random.normal(rng_latent, latent_shape)
    B = latent_shape[0]
    
    def refinement_step(latent_t, s):
        tau_prev, tau_curr = schedule.tau_values[s], schedule.tau_values[s+1]
        alpha = (tau_curr - tau_prev) / jnp.maximum(1.0 - tau_prev, 1e-8)
        
        step_idx = schedule.step_idx
        tau_idx_val = schedule.tau_idx[s+1] # Condition on target/next step, matching imagination.py

        step_idx_ctx, tau_idx_ctx = schedule.step_idx_ctx, schedule.tau_idx_ctx

        if caches is not None:
            latent_input, actions_input = latent_t, action

            step_idx= jnp.full((B, 1), step_idx,    dtype=jnp.int32)
            tau_idx = jnp.full((B, 1), tau_idx_val, dtype=jnp.int32)

            assert agent_tokens is None or agent_tokens.shape[1] == noisy_latent.shape[1] 
        
        else: # Used only for debugging.
            assert latents_ctx is not None and actions_ctx is not None
            latent_input  = jnp.concatenate([latents_ctx, latent_t], axis=1)  # (B, T_ctx+1, n_spatial, D_s)
            actions_input = jnp.concatenate([actions_ctx, action],   axis=1)  # (B, T_ctx+1)

            T_ctx = latent_input.shape[1]
            step_idx_ctx  = jnp.full((B, T_ctx), step_idx_ctx, dtype=jnp.int32)
            step_idx_curr = jnp.full((B, 1),     step_idx,     dtype=jnp.int32)
            step_idx = jnp.concatenate([step_idx_ctx, step_idx_curr], axis=1)
            
            tau_idx_ctx  = jnp.full((B, T_ctx), tau_idx_ctx, dtype=jnp.int32)
            tau_idx_curr = jnp.full((B, 1),     tau_idx_val, dtype=jnp.int32)
            tau_idx = jnp.concatenate([tau_idx_ctx, tau_idx_curr], axis=1) # (B, T_ctx+1)


        # Dynamics call
        z_clean_pred_seq, (h_seq, caches_updated) = dynamics.apply(dyn_vars, actions_input, step_idx, tau_idx, latent_input, agent_tokens=agent_tokens, deterministic=True, caches=caches)

        z_clean_pred = z_clean_pred_seq[:, -1:, :, :]  # (B, 1, n_spatial, D_s)
        h_last = h_seq[:, -1, :, :] if isinstance(h_seq, jax.Array) else h_seq # (B, n_agent, d_model)

        # Per-step mixing toward clean latent
        z_t_new = (1.0 - alpha) * latent_t + alpha * z_clean_pred

        return z_t_new, (h_last, caches_updated)

    # Run τ-ladder with JAX control flow using scan to keep carry/output structure consistent.
    z_t_final, (h_history, caches_history) = jax.lax.scan(
        refinement_step,
        noisy_latent,
        jnp.arange(schedule.num_steps),
    )

    h_last = h_history[-1]  # (B, n_agent, d_model)
    assert isinstance(h_last, jax.Array) or h_last is None
    caches_last = jax.tree.map(lambda x: x[-1], caches_history)

    return z_t_final, h_last, caches_last, rng  


def latent_rollout(
    dynamics: Dynamics,
    dyn_vars: VariableDict,
    policy: PolicyHeadMTP | jax.Array,
    policy_vars: VariableDict | None,
    schedule: DenoiseSchedule,
    initial_latents: jax.Array,
    initial_actions: jax.Array,
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
        initial_latents: (B, T_ctx, n_spatial, D_s) Context latents.
        initial_actions: (B, T_ctx, ...) Context actions.
        num_steps: Number of steps to unroll.
        rng: Random number generator key.
        initial_agent_tokens: Optional (B, T_ctx, n_agent, D) agent tokens for context.
        
    Returns:
        latents: (B, num_steps, n_spatial, D_s)
        actions: (B, num_steps, ...)
    """
    B, T_ctx, n_spatial, D_s = initial_latents.shape[:2]
    
    # 1. Initialize caches and process context
    # We need to compute the max window size needed: context + rollout
    window_size = T_ctx + num_steps
    caches = dynamics.create_static_caches(batch_size=B, window_size=window_size)
    
    # Run dynamics on context to warm up caches and get last hidden state
    # Use signal=Clean (max-1) and step=0 for context
    steps_ctx = jnp.zeros((B, T_ctx), dtype=jnp.int32)
    signals_ctx = jnp.full((B, T_ctx), dynamics.k_max - 1, dtype=jnp.int32)
    
    _, (h_seq, caches) = dynamics.apply(dyn_vars, initial_actions, steps_ctx, signals_ctx, initial_latents, agent_tokens=initial_agent_tokens, caches=caches, deterministic=True)
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
            logits = policy.apply(policy_vars, h_t, deterministic=False) # Use False to enable sampling logic if any
            action = jax.random.categorical(rng_action, logits) # (B, L) # Sample discrete action
        
        # Predict next latent (denoising)
        # We need latent shape for a single frame
        latent_shape = (B, 1, n_spatial, D_s)
        
        z_next, h_next, caches_next, rng = next_latent(dynamics, dyn_vars, schedule, action, latent_shape, rng, caches_t)
        
        return (h_next, caches_next, rng), z_next[:,0] # z_next is (B, 1, n_spatial, D_s) 

    # Run scan
    _, rollout_latents = jax.lax.scan(
        scan_step,
        (h_last, caches, rng),
        jnp.arange(num_steps)
    )
    
    # Unpack results
    rollout_latents = einops.rearrange(rollout_latents, 't b s d -> b t s d')
    
    return rollout_latents 