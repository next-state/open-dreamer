"""
Reusable training components for dynamics and imagination training.

This module contains:
- Sampling utilities for tau (signal level) and step size
- Loss computation functions for shortcut forcing
- Training step helpers that can be shared across training phases
"""
from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from flax.typing import VariableDict


# ---------------------------
# Sampling utilities
# ---------------------------

@partial(jax.jit, static_argnames=("shape_bt", "k_max"))
def sample_tau_for_step(
    rng: jax.Array,
    shape_bt: Tuple[int, int],
    k_max: int,
    step_idx: jnp.ndarray,
    *,
    dtype=jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample tau (signal level) values aligned to step_idx grid.
    
    This is the core sampling logic for shortcut forcing - it samples signal levels
    on the discrete grid defined by the current step size.
    
    Args:
        rng: JAX random key
        shape_bt: Tuple of (batch_size, sequence_length)
        k_max: Maximum noise resolution (e.g., 256)
        step_idx: (B, T) array of step indices encoding d = 1 / (1 << step_idx)
        dtype: Data type for computation
        
    Returns:
        tau: (B, T) Signal levels in [0, 1]
        tau_idx: (B, T) Discrete indices in [0, k_max]
        
    Example:
        If step_idx = 3, then K = 2^3 = 8, d = 1/8
        tau will be sampled uniformly from {0, 1/8, 2/8, ..., 7/8}
    """
    B_, T_ = shape_bt
    K = 1 << step_idx  # 2^step_idx
    u = jax.random.uniform(rng, (B_, T_), dtype=dtype)
    j_idx = jnp.floor(u * K.astype(dtype)).astype(jnp.int32)
    tau = j_idx.astype(dtype) / K.astype(dtype)
    tau_idx = j_idx * (k_max // K)
    return tau, tau_idx


@partial(jax.jit, static_argnames=("shape_bt", "k_max"))
def sample_step_excluding_dmin(
    rng: jax.Array,
    shape_bt: Tuple[int, int],
    k_max: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample step indices excluding the finest level (for bootstrap loss).
    
    The bootstrap loss requires coarser step sizes (d > d_min) to distill
    two half-steps into a full step.
    
    Args:
        rng: JAX random key
        shape_bt: Tuple of (batch_size, sequence_length)
        k_max: Maximum noise resolution
        
    Returns:
        d: (B, T) Step sizes (e.g., 1/128, 1/64, ..., 1/2, 1)
        step_idx: (B, T) Step indices in [0, log2(k_max) - 1]
        
    Example:
        If k_max = 256, emax = 8, samples step_idx from {0, 1, ..., 7}
        Step idx 7 gives d = 1/2, idx 0 gives d = 1/256 (but excludes d_min = 1/256)
    """
    B_, T_ = shape_bt
    emax = jnp.log2(k_max).astype(jnp.int32)
    # Sample from [0, emax) to exclude finest level at emax
    step_idx = jax.random.randint(rng, (B_, T_), 0, emax, dtype=jnp.int32)
    d = 1.0 / (1 << step_idx).astype(jnp.float32)
    return d, step_idx


# Loss weighting
def ramp_weight(sigma: jnp.ndarray, min_weight: float = 0.1, max_weight: float = 1.0) -> jnp.ndarray:
    return (max_weight - min_weight) * sigma + min_weight


# ---------------------------
# Loss computation
# ---------------------------

def compute_flow_loss(
    z_pred: jnp.ndarray,
    z_target: jnp.ndarray,
    sigma: jnp.ndarray,
    per_example: bool = False
) -> jnp.ndarray:
    """
    Flow matching loss in x-space (direct prediction of clean latents).
    
    Args:
        z_pred: (B, T, S, D) Predicted clean latents
        z_target: (B, T, S, D) Ground truth clean latents
        sigma: (B, T) Signal levels (used for weighting)
        per_example: If True, return (B, T) losses; else return scalar
        
    Returns:
        loss: Scalar or (B, T) MSE loss
    """
    mse_per_token = (z_pred - z_target) ** 2  # (B, T, S, D)
    mse_per_step = jnp.mean(mse_per_token, axis=(2, 3))  # (B, T)
    
    if per_example:
        return mse_per_step
    
    # Apply ramp weighting and reduce
    weights = ramp_weight(sigma)
    return jnp.mean(mse_per_step * weights)


def compute_bootstrap_loss(
    z_pred: jnp.ndarray,
    z_tilde: jnp.ndarray,
    b_prime: jnp.ndarray,
    b_doubleprime: jnp.ndarray,
    sigma: jnp.ndarray,
    per_example: bool = False
) -> jnp.ndarray:
    """
    Bootstrap self-consistency loss for shortcut forcing.
    
    Trains the model to predict the same endpoint whether taking one large step
    or two smaller steps. Loss is computed in v-space but scaled to x-space.
    
    Args:
        z_pred: (B, T, S, D) Predicted latent from full step
        z_tilde: (B, T, S, D) Initial noisy latent
        b_prime: (B, T, S, D) Velocity from first half-step
        b_doubleprime: (B, T, S, D) Velocity from second half-step
        sigma: (B, T) Signal levels
        per_example: If True, return (B, T) losses; else return scalar
        
    Returns:
        loss: Scalar or (B, T) bootstrap loss
    """
    # Convert full-step prediction to velocity
    v_hat = (z_pred - z_tilde) / jnp.maximum(1.0 - sigma[..., None, None], 1e-8)
    
    # Target velocity is average of two half-steps (stop gradient)
    v_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)
    
    # MSE in v-space, scaled to x-space
    v_diff = (v_hat - v_target) ** 2
    boot_per_token = (1.0 - sigma[..., None, None]) ** 2 * v_diff
    boot_per_step = jnp.mean(boot_per_token, axis=(2, 3))  # (B, T)
    
    if per_example:
        return boot_per_step
    
    # Apply ramp weighting and reduce
    weights = ramp_weight(sigma)
    return jnp.mean(boot_per_step * weights)


# ---------------------------
# Shortcut forcing step logic
# ---------------------------

def shortcut_forcing_step(
    dynamics_apply_fn: Callable,
    dynamics_vars: VariableDict,
    actions: jnp.ndarray,
    latents: jnp.ndarray,
    rng: jax.Array,
    k_max: int,
    *,
    B_self: int = 0,
    bootstrap_active: jnp.ndarray = jnp.array(True),
    agent_tokens: jnp.ndarray | None = None,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """
    Compute shortcut forcing losses (flow + bootstrap) for a batch.
    
    This is the core training logic that can be reused in both dynamics pretraining
    and imagination training phases.
    
    Args:
        dynamics_apply_fn: Model's apply function
        dynamics_vars: Model variables (params + constants)
        actions: (B, T) Action sequence
        latents: (B, T, S, D) Latent sequence (ground truth)
        rng: Random key
        k_max: Maximum noise resolution
        B_self: Number of bootstrap examples (last B_self rows of batch)
        bootstrap_active: Whether to compute bootstrap loss
        agent_tokens: Optional (B, T, n_agent, d_model) agent tokens
        
    Returns:
        losses: Dict with 'total', 'flow', 'bootstrap' keys
        aux: Dict with auxiliary metrics for logging. If agent_tokens is not None,
             aux contains 'h_states' key with (B, T, n_agent, d_model) hidden states
             from the main forward pass (computed with noisy inputs)
    """
    B, T, S, D = latents.shape
    B_emp = B - B_self
    emax = jnp.log2(k_max).astype(jnp.int32)
    
    # Split RNG
    key_sigma, key_step, key_noise, key_drop = jax.random.split(rng, 4)
    
    # --- Step indices ---
    # Empirical rows: always use finest step (d_min)
    step_idx_emp = jnp.full((B_emp, T), emax, dtype=jnp.int32)
    
    # Bootstrap rows: coarser steps (if B_self > 0)
    if B_self > 0:
        d_self, step_idx_self = sample_step_excluding_dmin(key_step, (B_self, T), k_max)
    else:
        d_self = jnp.zeros((0, T), dtype=jnp.float32)
        step_idx_self = jnp.zeros((0, T), dtype=jnp.int32)
    
    step_idx_full = jnp.concatenate([step_idx_emp, step_idx_self], axis=0)
    
    # --- Sample signal levels ---
    sigma_full, sigma_idx_full = sample_tau_for_step(key_sigma, (B, T), k_max, step_idx_full)
    sigma_emp = sigma_full[:B_emp]
    sigma_self = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]
    
    # --- Corrupt latents: z_tilde = (1 - sigma) * z0 + sigma * z1 ---
    z0 = jax.random.normal(key_noise, latents.shape, dtype=latents.dtype)
    z_tilde = (1.0 - sigma_full[..., None, None]) * z0 + sigma_full[..., None, None] * latents
    
    # --- Forward pass (full batch) ---
    drop_main, drop_h1, drop_h2 = jax.random.split(key_drop, 3)
    z_pred_full, (h_states, _) = dynamics_apply_fn(dynamics_vars, actions, step_idx_full, sigma_idx_full, z_tilde, agent_tokens=agent_tokens, rngs={"dropout": drop_main}, deterministic=False)
    
    # --- Flow loss (empirical rows) ---
    z_pred_emp = z_pred_full[:B_emp]
    loss_flow = compute_flow_loss(z_pred_emp, latents[:B_emp], sigma_emp)
    flow_mse_unweighted = jnp.mean((z_pred_emp - latents[:B_emp]) ** 2)
    
    # --- Bootstrap loss (self-consistency rows) ---
    loss_boot = jnp.array(0.0, dtype=latents.dtype)
    boot_mse_unweighted = jnp.array(0.0, dtype=latents.dtype)
    
    if B_self > 0:
        z_pred_self = z_pred_full[B_emp:]
        z_tilde_self = z_tilde[B_emp:]
        actions_self = actions[B_emp:]
        agent_tokens_self = agent_tokens[B_emp:] if agent_tokens is not None else None
    
        # Half-step metadata
        d_half = d_self / 2.0
        step_idx_half = step_idx_self + 1
        sigma_plus = sigma_self + d_half
        sigma_idx_plus = sigma_idx_self + (k_max * d_half).astype(jnp.int32)
    
        # First half-step
        z1_half1, *_ = dynamics_apply_fn(dynamics_vars, actions_self, step_idx_half, sigma_idx_self, z_tilde_self, agent_tokens=agent_tokens_self, rngs={"dropout": drop_h1}, deterministic=False)
        b_prime = (z1_half1 - z_tilde_self) / jnp.maximum(
            1.0 - sigma_self[..., None, None], 1e-8
        )
        z_prime = z_tilde_self + b_prime * d_half[..., None, None]
    
        # Second half-step
        z1_half2, *_ = dynamics_apply_fn(dynamics_vars, actions_self, step_idx_half, sigma_idx_plus, z_prime, agent_tokens=agent_tokens_self, rngs={"dropout": drop_h2}, deterministic=False)
        b_doubleprime = (z1_half2 - z_prime) / jnp.maximum(
            1.0 - sigma_plus[..., None, None], 1e-8
        )
    
        # Bootstrap loss (computed unconditionally)
        loss_boot = compute_bootstrap_loss(
            z_pred_self, z_tilde_self, b_prime, b_doubleprime, sigma_self
        )
        boot_mse_unweighted = compute_bootstrap_loss(z_pred_self, z_tilde_self, b_prime, b_doubleprime, sigma_self, per_example=True).mean()
    
        # Dynamic gating. TODO: isn't it better to simply do a weighted sum?
        bootstrap_mask = bootstrap_active.astype(latents.dtype)
        loss_boot = loss_boot * bootstrap_mask
        boot_mse_unweighted = boot_mse_unweighted * bootstrap_mask
    
    # --- Combine losses ---
    # Weight by batch composition to keep scale constant
    loss_total = ((loss_flow * (B - B_self)) + (loss_boot * B_self)) / B
    
    losses = { 'total': loss_total, 'flow': loss_flow, 'bootstrap': loss_boot}
    aux = {'flow_mse': flow_mse_unweighted, 'bootstrap_mse': boot_mse_unweighted, 'h_states': h_states}
    
    
    return losses, aux


# ---------------------------
# Symexp / Twohot helpers for reward prediction
# ---------------------------

def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric log transform for rewards."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def twohot_symlog_targets(values: jnp.ndarray, centers_log: jnp.ndarray) -> jnp.ndarray:
    """
    Convert scalar values to two-hot targets in symlog space.
    
    Args:
        values: (...,) real-valued rewards
        centers_log: (K,) bin centers in symlog space
        
    Returns:
        (..., K) two-hot targets that sum to 1
    """
    y = symlog(values)
    K = centers_log.shape[0]
    
    # Find bracketing bins
    idx_r = jnp.searchsorted(centers_log, y, side='right')
    idx_l = jnp.maximum(idx_r - 1, 0)
    idx_r = jnp.minimum(idx_r, K - 1)
    idx_l = jnp.minimum(idx_l, K - 1)
    
    # Linear interpolation weight
    c_l = jnp.take(centers_log, idx_l)
    c_r = jnp.take(centers_log, idx_r)
    denom = jnp.maximum(c_r - c_l, 1e-8)
    frac = jnp.where(idx_r == idx_l, 0.0, (y - c_l) / denom)
    
    # Two-hot encoding
    oh_l = jax.nn.one_hot(idx_l, K)
    oh_r = jax.nn.one_hot(idx_r, K)
    return oh_l * (1.0 - frac)[..., None] + oh_r * frac[..., None]


# ---------------------------
# Agent training losses (BC + Reward)
# ---------------------------

def compute_policy_loss(
    policy_head,
    policy_params: Any,
    h_states: jnp.ndarray,
    actions_btL: jnp.ndarray,
    actions_valid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute behavior cloning loss with multi-token prediction.
    
    Args:
        policy_head: Policy head model instance
        policy_params: Policy head parameters
        h_states: (B, T, n_agent, d_model) Hidden states from dynamics
        actions_btL: (B, T, L) Future action labels
        actions_valid: (B, T, L) Validity mask
        
    Returns:
        policy_loss: Scalar categorical cross-entropy loss
    """
    # Forward pass
    policy_logits = policy_head.apply(
        {"params": policy_params},
        h_states,
        deterministic=True,
    )  # (B, T, L, A)
    
    # Categorical cross-entropy for each future action
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)  # (B, T, L, A)
    action_log_probs = jnp.take_along_axis(
        log_probs,
        actions_btL[..., None],  # (B, T, L, 1)
        axis=-1
    ).squeeze(-1)  # (B, T, L)
    
    # Average over valid positions
    policy_loss = -jnp.sum(action_log_probs * actions_valid) / jnp.maximum(actions_valid.sum(), 1.0)
    
    return policy_loss


def compute_reward_loss(
    reward_head,
    reward_params: Any,
    h_states: jnp.ndarray,
    rewards_btL: jnp.ndarray,
    rewards_valid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute reward prediction loss with symexp twohot encoding.
    
    Args:
        reward_head: Reward head model instance
        reward_params: Reward head parameters
        h_states: (B, T, n_agent, d_model) Hidden states from dynamics
        rewards_btL: (B, T, L) Future reward values
        rewards_valid: (B, T, L) Validity mask
        
    Returns:
        reward_loss: Scalar categorical cross-entropy loss
    """
    # Forward pass
    reward_logits, centers_log = reward_head.apply(
        {"params": reward_params},
        h_states,
        deterministic=True,
    )  # logits: (B, T, L, K), centers: (K,)
    
    # Convert rewards to two-hot targets
    reward_targets = twohot_symlog_targets(rewards_btL, centers_log)  # (B, T, L, K)
    
    # Cross-entropy loss
    reward_log_probs = jax.nn.log_softmax(reward_logits, axis=-1)
    reward_loss_per = -jnp.sum(reward_targets * reward_log_probs, axis=-1)  # (B, T, L)
    reward_loss = jnp.sum(reward_loss_per * rewards_valid) / jnp.maximum(rewards_valid.sum(), 1.0)
    
    return reward_loss
