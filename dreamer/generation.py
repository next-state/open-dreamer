from curses import KEY_MAX
import jax.numpy as jnp
from typing import Dict, Any, Tuple, NamedTuple

from .models import Dynamics


class DenoiseSchedule:
    """
    Precomputed, JAX-friendly schedule for the τ-ladder.

    tau_seq:        (S+1,) τ_0..τ_S (monotone, τ_0 ∈ [0,1), τ_S=1.0)
    alpha_seq:      (S,)   per-step mixing coefficients α_s
    signal_idx_seq: (S+1,) integer signal indices for each τ_s
    step_idx:       scalar integer step index e (same for all steps here)
    k_max:          scalar integer, copied from config for convenience
    """
    
    num_steps: int
    k_max: int

    def __call__(self):
        assert self.k_max % self.num_steps == 0, f"k_max={self.k_max} must be divisible by num_steps={self.num_steps}"
        
        tau_seq = jnp.linspace(0.0, 1.0, self.num_steps + 1)
        step_idx = jnp.arange(self.num_steps+1)*self.k_max//self.num_steps
        k_max = self.k_max
        
    
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
    caches: Dict[int, Any] | None = None,
    latents_ctx: jnp.ndarray| None = None,                     # (B, T_ctx, n_spatial, D_s)
    actions_ctx: jnp.ndarray | None = None,                 # (B, T_ctx)
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[int, Any] | None]:
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
    B, T_ctx, n_spatial, D_s = latents_ctx.shape
    S = schedule.alpha_seq.shape[0]

    def refinement_step(z_t, s):
        alpha = schedule.alpha_seq[s]
        signal_idx_scalar = schedule.signal_idx_seq[s + 1]

        # Input preparation
        if caches is not None:
            # Denoise only the current frame
            z_input = z_t
            actions_input = action

            step_idx_input = jnp.full((B, 1), schedule.step_idx, dtype=jnp.int32)
            signal_idx_input = jnp.full((B, 1), signal_idx_scalar, dtype=jnp.int32)

            if agent_tokens is not None:
                agent_tokens_input = agent_tokens[:, -1:, :, :]
            else:
                agent_tokens_input = None
        
        else:
            # Denoise [context, current_frame]
            z_input = jnp.concatenate([latents_ctx, z_t], axis=1)  # (B, T_ctx+1, n_spatial, D_s)
            actions_input = jnp.concatenate([actions_ctx, action], axis=1)  # (B, T_ctx+1)

            step_idx_input = jnp.full((B, T_ctx + 1), schedule.step_idx, dtype=jnp.int32) # TODO: check this
            signal_idx_ctx = jnp.full((B, T_ctx), (0.9 * schedule.k_max).astype(jnp.int32), dtype=jnp.int32)
            signal_idx_curr = jnp.full((B, 1), signal_idx_scalar, dtype=jnp.int32)
            signal_idx_input = jnp.concatenate([signal_idx_ctx, signal_idx_curr], axis=1)

            agent_tokens_input = agent_tokens

        # Dynamics call
        z_clean_pred_seq, h_seq, caches_updated = dynamics.apply(dyn_vars, actions_input, step_idx_input, signal_idx_input, z_input, agent_tokens=agent_tokens_input, deterministic=True, caches=caches)

        z_clean_pred = z_clean_pred_seq[:, -1:, :, :]  # (B, 1, n_spatial, D_s)
        h_last = h_seq[:, -1, :, :]  # (B, n_agent, d_model)

        # Per-step mixing toward clean latent
        z_t_new = (1.0 - alpha) * z_t + alpha * z_clean_pred

        return z_t_new, (h_last, caches_updated)

    # Run τ-ladder with JAX control flow using scan to keep carry/output structure consistent.
    z_t_final, (h_history, caches_history) = jax.lax.scan(
        refinement_step,
        noisy_latent,
        jnp.arange(S),
    )

    h_last = h_history[-1]  # (B, n_agent, d_model)
    caches_last = None
    if caches_history is not None:
        caches_last = jax.tree.map(lambda x: x[-1], caches_history)

    return z_t_final, h_last, caches_last  # (B, 1, n_spatial, D_s), (B, n_agent, d_model), Dict[int, KVCache] | None
