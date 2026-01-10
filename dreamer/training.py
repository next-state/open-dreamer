"""
Reusable training components for dynamics and imagination training.

This module contains:
- Sampling utilities for tau (signal level) and step size
- Loss computation functions for shortcut forcing
- Training step helpers that can be shared across training phases
- Evaluation and visualization utilities
"""
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import einops
import imageio.v3 as iio
import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
import optax
import time

from dreamer.generation import DenoiseSchedule
from dreamer.sampler import sample_video
from dreamer.utils import _ensure_dir, normalize_with_dataset_stats, apply_border


# ---------------------------
# Sampling utilities
# ---------------------------

@partial(jax.jit, static_argnames=("shape_bt", "k_max", "dtype"))
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


@partial(jax.jit, static_argnames=("shape_bt", "k_max", "dtype"))
def sample_step_excluding_dmin(
    rng: jax.Array,
    shape_bt: Tuple[int, int],
    k_max: int,
    *,
    dtype=jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample step indices excluding the finest level (for bootstrap loss).

    The bootstrap loss requires coarser step sizes (d > d_min) to distill
    two half-steps into a full step.

    Args:
        rng: JAX random key
        shape_bt: Tuple of (batch_size, sequence_length)
        k_max: Maximum noise resolution
        dtype: Data type for computation

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
    d = 1.0 / (1 << step_idx).astype(dtype)
    return d, step_idx


@partial(jax.jit, static_argnames=("shape_bt", "k_max", "dtype"))
def sample_r_t_for_meanflow(
    rng: jax.Array,
    shape_bt: Tuple[int, int],
    k_max: int,
    *,
    dtype=jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample (r, t) pairs for mean flow training with variable interval sizes.

    This function samples time intervals for mean flow forcing, similar to how
    shortcut models sample variable step sizes. The interval sizes are sampled
    as powers of 2 to train the model on different time horizons.

    Strategy:
        1. Sample interval size delta ∈ {1, 1/2, 1/4, ..., 1/k_max} (uniform log scale)
        2. Sample start point r ~ U[0, 1 - delta]
        3. Set t = r + delta

    This ensures r < t and samples various time horizons, allowing the model
    to learn both coarse (large delta) and fine (small delta) average velocities.

    Args:
        rng: JAX random key
        shape_bt: Tuple of (batch_size, sequence_length)
        k_max: Maximum noise resolution (defines finest interval = 1/k_max)
        dtype: Data type for computation

    Returns:
        r: (B, T) Start signal levels in [0, 1)
        t: (B, T) End signal levels in (0, 1]
        delta: (B, T) Interval sizes (t - r)

    Example:
        If k_max = 8, possible delta values are {1, 1/2, 1/4, 1/8}
        For delta = 1/4, r is sampled uniformly from [0, 3/4]
        Then t = r + 1/4, ensuring t ∈ (1/4, 1]
    """
    B_, T_ = shape_bt
    key_delta, key_r = jax.random.split(rng)

    # Sample interval size as power of 2: delta = 1 / 2^step_idx
    # step_idx in [0, log2(k_max)], giving delta in {1, 1/2, 1/4, ..., 1/k_max}
    emax = jnp.log2(k_max).astype(jnp.int32) + 1  # +1 to include delta=1
    step_idx = jax.random.randint(key_delta, (B_, T_), 0, emax, dtype=jnp.int32)
    delta = 1.0 / (1 << step_idx).astype(dtype)  # 2^(-step_idx)

    # Sample start point r uniformly in [0, 1 - delta]
    # This ensures t = r + delta stays in (0, 1]
    u = jax.random.uniform(key_r, (B_, T_), dtype=dtype)
    r = u * (1.0 - delta)

    # Compute end point
    t = r + delta

    return r, t, delta


# Loss weighting
def ramp_weight(sigma: jnp.ndarray, min_weight: float = 0.1, max_weight: float = 1.0) -> jnp.ndarray:
    return (max_weight - min_weight) * sigma + min_weight


# ---------------------------
# Loss computation
# ---------------------------
def compute_psnr(pred, target):
    """
    Assumes pred and target are in the [0, 1] pixel range.
    Computes PSNR per sample, then returns the mean PSNR. 
    """
    # Ensure float32 precision to avoid quantization artifacts
    pred = pred.astype(jnp.float32)
    target = target.astype(jnp.float32)
    pred_clipped = jnp.clip(pred, 0.0, 1.0)
    target_clipped = jnp.clip(target, 0.0, 1.0)
    # Compute MSE per (B, T) sample: reduce over spatial and channel dims
    mse_per_sample = einops.reduce(
        (pred_clipped - target_clipped) ** 2,
        "b t h w c -> b t",
        reduction="mean"
    )  
    psnr_per_sample = -10.0 * jnp.log(mse_per_sample) / jnp.log(10.0)
    return jnp.mean(psnr_per_sample)
    
def compute_flow_loss(
    z_pred: jnp.ndarray,
    z_target: jnp.ndarray,
    sigma: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Flow matching loss in x-space (direct prediction of clean latents).
    
    Args:
        z_pred: (B, T, S, D) Predicted clean latents
        z_target: (B, T, S, D) Ground truth clean latents
        sigma: (B, T) Signal levels (used for weighting)
        per_example: If True, return (B, T) losses; else return scalar
        
    Returns:
        loss: Tuple[jnp.float32, jnp.float32]
            mse_per_step: tuple of scalars. MSE loss weighted by ramp weight
            mse_per_token: tuple of scalars. MSE loss 
    """
    mse_per_token = (z_pred - z_target) ** 2  # (B, T, S, D)
    mse_per_step = jnp.mean(mse_per_token, axis=(2, 3))  # (B, T)
    
    
    # Apply ramp weighting and reduce
    weights = ramp_weight(sigma)
    return jnp.mean(mse_per_step * weights), jnp.mean(mse_per_step)


def compute_bootstrap_loss(
    z_pred: jnp.ndarray,
    z_tilde: jnp.ndarray,
    b_prime: jnp.ndarray,
    b_doubleprime: jnp.ndarray,
    sigma: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    Returns:
        loss: Tuple[jnp.float32, jnp.float32]
            mse_per_step: tuple of scalars. MSE loss weighted by ramp weight
            mse_per_token: tuple of scalars. MSE loss
    """
    # Convert full-step prediction to velocity
    v_hat = (z_pred - z_tilde) / jnp.maximum(1.0 - sigma[..., None, None], 1e-8)

    # Target velocity is average of two half-steps (stop gradient)
    v_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)

    # MSE in v-space, scaled to x-space
    v_diff = (v_hat - v_target) ** 2
    boot_per_token = (1.0 - sigma[..., None, None]) ** 2 * v_diff
    boot_per_step = jnp.mean(boot_per_token, axis=(2, 3))  # (B, T)


    # Apply ramp weighting and reduce
    weights = ramp_weight(sigma)
    return jnp.mean(boot_per_step * weights), jnp.mean(boot_per_step)


def compute_meanflow_loss(
    u_pred: jnp.ndarray,
    z_t: jnp.ndarray,
    v_target: jnp.ndarray,
    r: jnp.ndarray,
    t: jnp.ndarray,
    delta: jnp.ndarray,
    dynamics_model,
    actions: jnp.ndarray,
    agent_tokens: jnp.ndarray | None,
    rngs: nnx.Rngs,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute mean flow loss using the MeanFlow identity with JVP.

    The MeanFlow identity relates the average velocity u(z_t, r, t) over an
    interval [r, t] to the instantaneous velocity v(z_t, t) at the endpoint:

        v(z_t, t) = u(z_t, r, t) + (t - r) * (∂u/∂t + ∇_z u · v)

    We use this identity to reconstruct the instantaneous velocity from the
    predicted average velocity and compare it to the target velocity.

    Args:
        u_pred: (B, T, S, D) Predicted average velocity from model
        z_t: (B, T, S, D) Noisy latents at signal level t
        v_target: (B, T, S, D) Target velocity (x1 - x0)
        r: (B, T) Start signal levels
        t: (B, T) End signal levels
        delta: (B, T) Interval sizes (t - r)
        dynamics_model: NNX Dynamics model for recomputing u in JVP
        actions: (B, T) Action sequence
        agent_tokens: Optional (B, T, n_agent, d_model) agent tokens
        rngs: RNG state for model forward pass

    Returns:
        loss_weighted: Scalar MSE loss with ramp weighting
        loss_unweighted: Scalar MSE loss for logging

    Implementation:
        1. Define u_fn(z, r, t) that calls the dynamics model
        2. Compute ∂u/∂t using JVP with tangent (0, 0, 1)
        3. Compute ∇_z u · v_target using JVP with tangent (v_target, 0, 0)
        4. Reconstruct V_θ = u + delta * (∂u/∂t + ∇_z u · v_target)
        5. Compute MSE(V_θ, v_target) with ramp weighting
    """
    # Define function that computes u given (z, r, t)
    def u_fn(z: jnp.ndarray, r_val: jnp.ndarray, t_val: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through dynamics model to get average velocity u."""
        # Dummy values for discrete conditioning (not used in meanflow mode)
        dummy_step = jnp.zeros_like(actions, dtype=jnp.int32)
        dummy_tau = jnp.zeros_like(actions, dtype=jnp.int32)

        u_out, _ = dynamics_model(
            actions, dummy_step, dummy_tau, z,
            r_continuous=r_val,
            t_continuous=t_val,
            agent_tokens=agent_tokens,
            deterministic=True,  # MUST be True to avoid RNG issues inside JVP trace
            rngs=None  # No RNG needed when deterministic=True
        )
        return u_out

    # Compute spatial JVP: ∇_z u · v_target
    # Tangent vector: (v_target, 0, 0) for (z, r, t)
    primals = (z_t, r, t)
    tangents_spatial = (v_target, jnp.zeros_like(r), jnp.zeros_like(t))
    _, u_jvp_spatial = jax.jvp(u_fn, primals, tangents_spatial)
    # u_jvp_spatial = ∇_z u · v_target (shape: B, T, S, D)

    # Compute time derivative: ∂u/∂t
    # Tangent vector: (0, 0, 1) for (z, r, t)
    tangents_time = (jnp.zeros_like(z_t), jnp.zeros_like(r), jnp.ones_like(t))
    _, u_jvp_time = jax.jvp(u_fn, primals, tangents_time)
    # u_jvp_time = ∂u/∂t (shape: B, T, S, D)

    # Reconstruct instantaneous velocity using MeanFlow identity
    # V_θ = u + delta * (∂u/∂t + ∇_z u · v_target)
    delta_expanded = delta[..., None, None]  # (B, T, 1, 1)
    V_reconstructed = u_pred + delta_expanded * (u_jvp_time + u_jvp_spatial)

    # Compute MSE
    mse_per_token = (V_reconstructed - v_target) ** 2  # (B, T, S, D)
    mse_per_step = jnp.mean(mse_per_token, axis=(2, 3))  # (B, T)

    # Apply ramp weighting based on t (end signal level)
    # Higher signal levels (cleaner data) get more weight
    weights = ramp_weight(t)
    loss_weighted = jnp.mean(mse_per_step * weights)
    loss_unweighted = jnp.mean(mse_per_step)

    return loss_weighted, loss_unweighted


# ---------------------------
# Shortcut forcing step logic
# ---------------------------

def shortcut_forcing_step(
    dynamics_model,
    actions: jnp.ndarray,
    latents: jnp.ndarray,
    rng: jax.Array,
    k_max: int,
    *,
    B_self: int = 0,
    agent_tokens: jnp.ndarray | None = None,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """
    Compute shortcut forcing losses (flow + bootstrap) for a batch.
    
    This is the core training logic that can be reused in both dynamics pretraining
    and imagination training phases.
    
    Args:
        dynamics_model: NNX Dynamics model instance
        actions: (B, T) Action sequence
        latents: (B, T, S, D) Latent sequence (ground truth)
        rng: Random key
        k_max: Maximum noise resolution
        B_self: Number of bootstrap examples (last B_self rows of batch)
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
    key_sigma, key_step, key_noise, key_dropout1, key_dropout2, key_dropout3 = jax.random.split(rng, 6)
    
    # --- Step indices ---
    # Empirical rows: always use finest step (d_min)
    step_idx_emp = jnp.full((B_emp, T), emax, dtype=jnp.int32)
    
    # Bootstrap rows: coarser steps (if B_self > 0)
    if B_self > 0:
        d_self, step_idx_self = sample_step_excluding_dmin(key_step, (B_self, T), k_max, dtype=latents.dtype)
    else:
        d_self = jnp.zeros((0, T), dtype=latents.dtype)
        step_idx_self = jnp.zeros((0, T), dtype=jnp.int32)
    
    step_idx_full = jnp.concatenate([step_idx_emp, step_idx_self], axis=0)
    
    # --- Sample signal levels ---
    sigma_full, sigma_idx_full = sample_tau_for_step(key_sigma, (B, T), k_max, step_idx_full, dtype=latents.dtype)
    sigma_emp = sigma_full[:B_emp]
    sigma_self = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]
    
    # --- Corrupt latents: z_tilde = (1 - sigma) * z0 + sigma * z1 ---
    z0 = jax.random.normal(key_noise, latents.shape, dtype=latents.dtype)
    z_tilde = (1.0 - sigma_full[..., None, None]) * z0 + sigma_full[..., None, None] * latents
    
    # --- Forward pass (full batch) ---
    rngs1 = nnx.Rngs(dropout=key_dropout1)
    z_pred_full, (h_states, _) = dynamics_model(
        actions, step_idx_full, sigma_idx_full, z_tilde,
        agent_tokens=agent_tokens, deterministic=False, rngs=rngs1
    )
    
    # --- Flow loss (empirical rows) ---
    z_pred_emp = z_pred_full[:B_emp]
    loss_flow, flow_mse_unweighted = compute_flow_loss(z_pred_emp, latents[:B_emp], sigma_emp)
    
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
        rngs2 = nnx.Rngs(dropout=key_dropout2)
        z1_half1, *_ = dynamics_model(
            actions_self, step_idx_half, sigma_idx_self, z_tilde_self,
            agent_tokens=agent_tokens_self, deterministic=False, rngs=rngs2
        )
        b_prime = (z1_half1 - z_tilde_self) / jnp.maximum(1.0 - sigma_self[..., None, None], 1e-8)
        z_prime = z_tilde_self + b_prime * d_half[..., None, None]
    
        # Second half-step
        rngs3 = nnx.Rngs(dropout=key_dropout3)
        z1_half2, *_ = dynamics_model(
            actions_self, step_idx_half, sigma_idx_plus, z_prime,
            agent_tokens=agent_tokens_self, deterministic=False, rngs=rngs3
        )
        b_doubleprime = (z1_half2 - z_prime) / jnp.maximum(1.0 - sigma_plus[..., None, None], 1e-8)
    
        # Bootstrap loss (computed unconditionally)
        loss_boot, boot_mse_unweighted = compute_bootstrap_loss(z_pred_self, z_tilde_self, b_prime, b_doubleprime, sigma_self)
    
    # --- Combine losses ---
    # Weight by batch composition to keep scale constant
    loss_total = ((loss_flow * (B - B_self)) + (loss_boot * B_self)) / B
    
    losses = { 'total': loss_total, 'flow': loss_flow, 'bootstrap': loss_boot}
    aux = {'flow_mse': flow_mse_unweighted, 'bootstrap_mse': boot_mse_unweighted, 'h_states': h_states}
    
    
    return losses, aux


def meanflow_forcing_step(
    dynamics_model,
    actions: jnp.ndarray,
    latents: jnp.ndarray,
    rng: jax.Array,
    k_max: int,
    *,
    agent_tokens: jnp.ndarray | None = None,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """
    Compute mean flow forcing loss for a batch.

    This is an alternative to shortcut_forcing_step() that uses the MeanFlow
    identity instead of flow + bootstrap losses. It trains the model to predict
    average velocities u(z_t, r, t) over time intervals [r, t], which can then
    be used for efficient 1-step generation at inference time.

    Args:
        dynamics_model: NNX Dynamics model instance
        actions: (B, T) Action sequence
        latents: (B, T, S, D) Latent sequence (ground truth clean latents)
        rng: Random key
        k_max: Maximum noise resolution (for interval sampling)
        agent_tokens: Optional (B, T, n_agent, d_model) agent tokens

    Returns:
        losses: Dict with 'total' and 'meanflow' keys
        aux: Dict with auxiliary metrics for logging:
            - 'meanflow_mse': Unweighted MSE for logging
            - 'h_states': Hidden states from agent tokens (if provided)

    Note:
        The model is called in "continuous mode" by providing r_continuous and
        t_continuous parameters, which triggers sinusoidal embedding instead of
        discrete embedding lookup.
    """
    B, T, S, D = latents.shape

    # Split RNG for different random operations
    key_rt, key_noise, key_dropout = jax.random.split(rng, 3)

    # Sample (r, t) pairs with variable interval sizes
    # This ensures the model learns to predict average velocities over
    # different time horizons (coarse and fine)
    r, t, delta = sample_r_t_for_meanflow(key_rt, (B, T), k_max, dtype=latents.dtype)

    # Corrupt latents using linear interpolation: z_t = (1 - t) * z0 + t * z1
    # This is the standard flow matching corruption scheme
    z0 = jax.random.normal(key_noise, latents.shape, dtype=latents.dtype)
    z_t = (1.0 - t[..., None, None]) * z0 + t[..., None, None] * latents

    # Target velocity for flow matching
    v_target = latents - z0  # x1 - x0

    # Forward pass to get u prediction (average velocity)
    # The model interprets its output as u(z_t, r, t) when in meanflow mode
    rngs = nnx.Rngs(dropout=key_dropout)
    u_pred, (h_states, _) = dynamics_model(
        actions,
        jnp.zeros((B, T), dtype=jnp.int32),  # dummy step_indices (not used in meanflow mode)
        jnp.zeros((B, T), dtype=jnp.int32),  # dummy tau_indices (not used in meanflow mode)
        z_t,
        r_continuous=r,  # Start signal level (triggers continuous mode)
        t_continuous=t,  # End signal level (triggers continuous mode)
        agent_tokens=agent_tokens,
        deterministic=False,
        rngs=rngs
    )

    # Compute mean flow loss using JVP to evaluate the MeanFlow identity
    # This involves computing ∂u/∂t and ∇_z u · v_target
    loss_meanflow, meanflow_mse_unweighted = compute_meanflow_loss(
        u_pred=u_pred,
        z_t=z_t,
        v_target=v_target,
        r=r,
        t=t,
        delta=delta,
        dynamics_model=dynamics_model,
        actions=actions,
        agent_tokens=agent_tokens,
        rngs=rngs
    )

    # Return losses and auxiliary info
    losses = {'total': loss_meanflow, 'meanflow': loss_meanflow}
    aux = {'meanflow_mse': meanflow_mse_unweighted, 'h_states': h_states}

    return losses, aux


# ---------------------------
# Symexp / Twohot helpers for reward prediction
# ---------------------------

def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric log transform for rewards."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(y: jnp.ndarray) -> jnp.ndarray:
    """Inverse of symlog: symmetric exponential transform."""
    return jnp.sign(y) * (jnp.expm1(jnp.abs(y)))


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
# RL Training Losses
# ---------------------------

def compute_td_lambda_returns(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    gamma: float,
    lambda_: float,
) -> jnp.ndarray:
    """
    Compute TD(λ) returns via backward scan.
    
    At timestep t, the TD(λ) return is:
        R^λ[t] = r[t+1] + γ * ((1-λ) V[t+1] + λ R^λ[t+1])
    
    With bootstrap condition: R^λ[T] = V[T]
    
    Args:
        rewards: (B, T) rewards received at each step (r_1...r_T)
        values: (B, T+1) value predictions (V_0...V_T, includes bootstrap)
        gamma: Discount factor
        lambda_: TD(λ) mixing parameter
        
    Returns:
        td_returns: (B, T) TD(λ) targets for value training
    """
    def step(carry, inputs):
        G_next = carry
        r_t1, v_t1 = inputs
        G_t = r_t1 + gamma * ((1 - lambda_) * v_t1 + lambda_ * G_next)
        return G_t, G_t
    
    r_rev = rewards[:, ::-1]  # (B, T): r_T...r_1
    v_next = values[:, 1:]
    v_next_rev = v_next[:, ::-1]  # (B, T): V(s_T...s_1)
    _, G_rev = jax.lax.scan(
        step,
        values[:, -1],  # Bootstrap with V[T]
        (r_rev.T, v_next_rev.T),
    )
    td_returns = rearrange(G_rev[::-1], "T B -> B T")  # (B, T)
    return td_returns


def compute_value_loss(
    val_logits: jnp.ndarray,
    centers_log_val: jnp.ndarray,
    td_returns: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute value head loss using symexp twohot targets.
    
    The value head predicts a distribution over symlog-transformed values,
    using the two-hot encoding for improved learning across varying scales.
    
    Args:
        val_logits: (B, T, K) Logits from value head forward pass
        centers_log_val: (K,) Bin centers in symlog space
        td_returns: (B, T) TD(λ) return targets
        
    Returns:
        loss: Scalar categorical cross-entropy loss
        
    Notes:
        - Trains on states s_0..s_{T-1} to predict returns R_0..R_{T-1}
        - Uses symexp encoding to handle returns of varying magnitude
        - Two-hot targets provide smoother gradients than one-hot
    """
    # Convert TD returns to two-hot targets
    twohot_targets = jax.lax.stop_gradient(
        twohot_symlog_targets(td_returns, centers_log_val)
    )  # (B, T, K)
    
    # Cross-entropy loss using optax
    val_ce = optax.safe_softmax_cross_entropy(logits=val_logits, labels=twohot_targets)  # (B, T)
    val_loss = jnp.mean(val_ce)
    
    return val_loss


def compute_pmpo_loss(
    policy_logits: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    policy_prior_logits: jnp.ndarray,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Compute PMPO (Probabilistic Policy Optimization) loss.
    
    PMPO balances positive and negative advantages using sign-only information,
    making it robust to return scale variations across tasks.
    
    Loss components:
        1. Negative advantage states: maximize log-prob (encourage action)
        2. Positive advantage states: minimize log-prob (discourage action)
        3. KL divergence to behavioral prior (regularization)
    
    Args:
        policy_logits: (B, T, A) Unnormalized logits from current policy
        actions: (B, T) Action labels taken in imagination
        advantages: (B, T) Advantage estimates A(s,a) = R^λ - V(s)
        policy_prior_logits: (B, T, A) Logits from frozen BC policy (prior)
        alpha: Weight balancing positive/negative sets (default: 0.5)
        beta: Weight for KL regularization (default: 0.3)
        
    Returns:
        loss: Scalar PMPO loss
        aux: Dict containing:
            - loss_negative: Contribution from negative advantage states
            - loss_positive: Contribution from positive advantage states
            - kl_loss: KL divergence regularization term
            - n_positive: Number of states with A >= 0
            - n_negative: Number of states with A < 0
    """
    # Compute log probabilities
    logp_pi = jax.nn.log_softmax(policy_logits, axis=-1)  # (B, T, A)
    A_dim = logp_pi.shape[-1]
    
    # Log-prob of imagined actions
    actions_onehot = jax.nn.one_hot(actions.astype(jnp.int32), A_dim)
    logp_actions = jnp.sum(actions_onehot * logp_pi, axis=-1)  # (B, T)
    
    # Flatten for easier indexing
    logp_flat = rearrange(logp_actions, "B T -> (B T)")
    advantages_flat = rearrange(advantages, "B T -> (B T)")
    
    # Partition into positive/negative advantage sets
    mask_positive = advantages_flat >= 0
    mask_negative = advantages_flat < 0
    
    n_positive = jnp.sum(mask_positive)
    n_negative = jnp.sum(mask_negative)
    
    # Negative set: encourage high log-prob (maximize)
    loss_negative = jnp.where(
        n_negative > 0,
        (1 - alpha) * jnp.sum(jnp.where(mask_negative, logp_flat, 0.0)) / n_negative,
        0.0,
    )
    
    # Positive set: discourage high log-prob (minimize)
    loss_positive = jnp.where(
        n_positive > 0,
        -alpha * jnp.sum(jnp.where(mask_positive, logp_flat, 0.0)) / n_positive,
        0.0,
    )
    
    # KL(π_θ || π_BC) regularization
    logp_bc = jax.nn.log_softmax(policy_prior_logits, axis=-1)
    kl_per_state = optax.losses.kl_divergence_with_log_targets(logp_pi, logp_bc)  # (B, T)
    kl_loss = beta * jnp.mean(kl_per_state)
    
    # Total policy loss
    pi_loss = loss_negative + loss_positive + kl_loss
    
    aux = {
        'loss_negative': loss_negative,
        'loss_positive': loss_positive,
        'kl_loss': kl_loss,
        'n_positive': n_positive,
        'n_negative': n_negative,
    }
    
    return pi_loss, aux


# ---------------------------
# Agent training losses (BC + Reward)
# ---------------------------

def compute_reward_loss(
    reward_head,
    h_states: jnp.ndarray,
    rewards_btL: jnp.ndarray,
    rewards_valid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute reward prediction loss with symexp twohot encoding.
    
    Args:
        reward_head: Reward head NNX model instance
        h_states: (B, T, n_agent, d_model) Hidden states from dynamics
        rewards_btL: (B, T, L) Future reward values
        rewards_valid: (B, T, L) Validity mask
        
    Returns:
        reward_loss: Scalar categorical cross-entropy loss
    """
    # Forward pass
    reward_logits, centers_log = reward_head(h_states, deterministic=True) 
    
    assert rewards_valid.dtype == jnp.bool_, "rewards_valid must be of type bool"
    reward_targets = twohot_symlog_targets(rewards_btL, centers_log)  # (B, T, L, K)
    reward_loss_per = optax.safe_softmax_cross_entropy(logits=reward_logits, labels=reward_targets)  # (B, T, L)
    reward_loss = jnp.sum(reward_loss_per * rewards_valid) / jnp.maximum(rewards_valid.sum(), 1.0)
    
    return reward_loss


def compute_policy_loss(
    policy_head,
    h_states: jnp.ndarray,
    actions: jnp.ndarray,
    actions_valid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute behavior cloning loss with multi-token prediction.
    
    Args:
        policy_head: Policy head NNX model instance
        h_states: (B, T, n_agent, d_model) Hidden states from dynamics
        actions: (B, T, L) Future action labels
        actions_valid: (B, T, L) Validity mask
        
    Returns:
        policy_loss: Scalar categorical cross-entropy loss
    """
    # Forward pass
    policy_logits = policy_head(h_states, deterministic=True)  # (B, T, L, A)
    
    # Categorical cross-entropy for each future action
    assert actions.dtype == jnp.int32, "actions must be of type int32"
    assert actions_valid.dtype == jnp.bool_, "actions_valid must be of type bool"
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(policy_logits, actions)
    policy_loss = jnp.sum(per_token_loss * actions_valid) / jnp.maximum(actions_valid.sum(), 1.0)
    
    return policy_loss


# ---------------------------
# Evaluation and visualization
# ---------------------------

def run_evaluation(
    cfg,
    tokenizer_cfg,
    step: int,
    tokenizer,
    dynamics,
    val_videos: jnp.ndarray,
    val_actions: jnp.ndarray,
    vis_dir: Path,
    rng: jax.Array,
    logger,
):
    """
    Run periodic evaluation: sample videos, compute metrics, and save visualization.

    This function can be used in both dynamics pretraining and agent finetuning.
    Automatically detects the forcing type from the dynamics config and uses
    the appropriate sampler (shortcut or meanflow).

    Args:
        cfg: Training config
        tokenizer_cfg: Tokenizer config (for dataset stats)
        step: Current training step
        tokenizer: Tokenizer NNX model instance
        dynamics: Dynamics NNX model instance
        val_videos: (B, T, H, W, C) Validation videos
        val_actions: (B, T) Validation actions
        vis_dir: Directory to save visualizations
        rng: Random key
        logger: Logger instance for logging metrics and videos
    """
    k_max = dynamics.cfg.k_max
    forcing_type = dynamics.cfg.forcing_type

    # Define evaluation schedules based on forcing type
    if forcing_type == "shortcut":
        # Shortcut forcing: use τ-ladder with different step counts
        schedule_shortcut = DenoiseSchedule.init(4, k_max)
        schedule_diffusion = DenoiseSchedule.init(k_max, k_max)
        evaluation_schedules = {
            "shortcut": (schedule_shortcut, None),
            "diffusion": (schedule_diffusion, None)
        }
    elif forcing_type == "meanflow":
        # Meanflow forcing: use different numbers of refinement steps
        # 1-step = direct generation (main advantage of meanflow)
        # 4-step = multi-step refinement for comparison
        evaluation_schedules = {
            "meanflow_1step": (None, 1),
            "meanflow_4step": (None, 4)
        }
    else:
        raise ValueError(f"Unknown forcing_type: {forcing_type}")

    for tag, (schedule_config, num_meanflow_steps) in evaluation_schedules.items():
        t0 = time.time()
        # FIXME: only temporary for debugging
        assert val_videos.shape[1] > 5
        ctx_length = 4
        horizon = val_videos.shape[1] - ctx_length

        pred_frames, floor_frames, gt_frames = sample_video(
            tokenizer, dynamics,
            val_videos, val_actions, horizon, schedule_config, rng,
            forcing_type=forcing_type,
            num_meanflow_steps=num_meanflow_steps or 1
        )

        # Compute metrics
        dt = time.time() - t0
        dataset_std = tokenizer_cfg.dataset.dataset_std[0]
        normalized_pred = normalize_with_dataset_stats(pred_frames[:, -horizon:], mean=0, std=dataset_std)
        normalized_gt = normalize_with_dataset_stats(gt_frames[:, -horizon:], mean=0, std=dataset_std)
        mse = float(jnp.mean((normalized_pred - normalized_gt) ** 2))
        psnr = float(compute_psnr(pred_frames[:, -horizon:]/255, gt_frames[:, -horizon:]/255))
        
        print(f"[eval:{tag}] step={step:06d} | horizon={horizon} | MSE={mse:.6g} | PSNR={psnr:.2f} dB | {dt:.2f}s")

        # Build visualization
        num_videos = min(4, pred_frames.shape[0])
        
        # Add red border to context frames in prediction
        pred_frames = pred_frames.at[:, :ctx_length].set(apply_border(pred_frames[:, :ctx_length]))
        
        frames = [floor_frames, gt_frames, pred_frames]
        stacked_frames = jnp.stack(frames)[:, :num_videos]
        videos = rearrange(stacked_frames, 'S B T H W C -> T (B H) (S W) C', B=num_videos)

        # Save artifacts
        tag_dir = _ensure_dir(vis_dir / f"step_{step:06d}")
        mp4_path = tag_dir / f"{tag}_grid.mp4"

        # Save video
        try:
            videos = jax.device_get(videos)
            iio.imwrite(str(mp4_path), videos, fps=5, plugin='pyav', codec='libx264')
        except Exception as e:
            print(f"[eval:{tag}] MP4 write failed: {e}")

        # Log metrics and video
        logger.log_metrics(step, {
            f"{tag}/mse": mse,
            f"{tag}/psnr": psnr,
            f"{tag}/horizon": horizon,
            f"{tag}/eval_time": dt,
        }, prefix="eval/")

        if videos is not None:
            logger.log_video(step, f"eval/{tag}/video", mp4_path)
