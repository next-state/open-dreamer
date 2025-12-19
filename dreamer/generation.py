import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple 

from .models import Dynamics, KVCache


from flax.struct import dataclass

@dataclass
class DenoiseSchedule:
    """
    Precomputed, JAX-friendly schedule for the τ-ladder.

    tau_seq:        (S+1,) τ_0..τ_S (monotone, τ_0 ∈ [0,1), τ_S=1.0)
    step_idx:       (S+1,) integer step indices
    step_size:      float step size
    k_max:          scalar integer, copied from config for convenience
    num_steps:      scalar integer, copied from config for convenience
    """
    
    tau_seq: jnp.ndarray
    step_idx: jnp.ndarray
    step_size: float
    k_max: int
    num_steps: int

    @classmethod
    def create(cls, num_steps: int, k_max: int) -> "DenoiseSchedule":
        """
        Create a DenoiseSchedule object.
        Args:
            num_steps: Number of steps in the schedule.
            k_max: Maximum value of k.
        Returns:
            DenoiseSchedule object.
        """
        assert k_max % num_steps == 0, f"k_max={k_max} must be divisible by num_steps={num_steps}"
        
        tau_seq = jnp.linspace(0.0, 1.0, num_steps + 1)
        step_idx = jnp.arange(num_steps+1)*(k_max//num_steps)
        step_size = k_max / num_steps
        
        return cls(tau_seq, step_idx, step_size, k_max, num_steps)
    
# ---------------------------
# Single-step τ-ladder denoiser
# ---------------------------

def next_latent(
    *,
    dynamics: Dynamics,
    dyn_vars: Dict[str, Any],
    schedule: DenoiseSchedule,
    action: jnp.ndarray,                 # (B, 1)
    noisy_latent: jnp.ndarray,                   # (B, 1, n_spatial, D_s)
    agent_tokens: jnp.ndarray | None = None,  # (B, T_ctx+1, n_agent, d_model)
    caches: KVCache | None = None,
    latents_ctx: jnp.ndarray| None = None,                     # (B, T_ctx, n_spatial, D_s)
    actions_ctx: jnp.ndarray | None = None,                 # (B, T_ctx)
) -> Tuple[jnp.ndarray, jnp.ndarray | None, KVCache | None]:
    """
    JAX-friendly τ-ladder denoiser for a single future latent with KV caching.

    - Uses a precomputed schedule (τ_seq, α_seq, signal_idx_seq, step_idx).
    - Contains only JAX ops in the inner loop (no Python branching on traced values).
    - Returns the denoised latent, the final hidden state h_t from dynamics, and the updated KV cache.

    Args:
        dynamics: Dynamics model (Flax Module)
        dyn_vars: Variables for dynamics (params + collections)
        schedule: Precomputed DenoiseSchedule
        actions_ctx: (B, T_ctx) context actions
        action_curr: (B, 1) action for current step
        z_ctx_t: (B, T_ctx, n_spatial, D_s) context latents
        z_ctx_noise_t: (B, T_ctx, n_spatial, D_s) stabilization noise for context
        noisy_latent: (B, 1, n_spatial, D_s) initial noisy latent at τ_0
        agent_tokens: optional agent tokens (B, T_ctx+1, n_agent, d_model)
        caches: KV cache for context frames (from previous finalized frames)
    """
    B = noisy_latent.shape[0]
    
    def refinement_step(lantent_t, s):
        tau = schedule.tau_seq[s]
        alpha = 1-tau
        step_idx = schedule.step_idx[s] # TODO: check if this should be step_idx[s+1]

        if caches is not None:
            latent_input, actions_input = lantent_t, action

            step_size = jnp.full((B, 1), schedule.step_size, dtype=jnp.int32)
            step_idx  = jnp.full((B, 1), step_idx, dtype=jnp.int32)

            assert agent_tokens is None or agent_tokens.shape[1] == noisy_latent.shape[1] 
        
        else: # Used only for debugging.
            assert latents_ctx is not None and actions_ctx is not None
            latent_input  = jnp.concatenate([latents_ctx, lantent_t], axis=1)  # (B, T_ctx+1, n_spatial, D_s)
            actions_input = jnp.concatenate([actions_ctx, action],    axis=1)  # (B, T_ctx+1)

            T_ctx = latent_input.shape[1]
            step_size       = jnp.full((B, T_ctx + 1), schedule.step_size,        dtype=jnp.int32)
            
            signal_idx_ctx  = jnp.full((B, T_ctx),     int(0.9 * schedule.k_max), dtype=jnp.int32)
            signal_idx_curr = jnp.full((B, 1),         step_idx,                  dtype=jnp.int32)
            step_idx = jnp.concatenate([signal_idx_ctx, signal_idx_curr], axis=1) # (B, T_ctx+1)


        # Dynamics call
        z_clean_pred_seq, (h_seq, caches_updated) = dynamics.apply(dyn_vars, actions_input, step_size, step_idx, latent_input, agent_tokens=agent_tokens, deterministic=True, caches=caches)

        z_clean_pred = z_clean_pred_seq[:, -1:, :, :]  # (B, 1, n_spatial, D_s)
        h_last = h_seq[:, -1, :, :] if isinstance(h_seq, jax.Array) else h_seq # (B, n_agent, d_model)

        # Per-step mixing toward clean latent
        z_t_new = (1.0 - alpha) * lantent_t + alpha * z_clean_pred # TODO: on the first step alpha should be 1.0. This step does nothing

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

    return z_t_final, h_last, caches_last  # (B, 1, n_spatial, D_s), (B, n_agent, d_model), Dict[int, KVCache] | None
