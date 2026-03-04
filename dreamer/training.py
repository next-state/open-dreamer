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

from dreamer.configs import DynamicsConfig, HeadsConfig, OptimalTransportConfig
from dreamer.generation import DenoiseSchedule
from dreamer.models import Tokenizer, Dynamics, PolicyHeadMTP, TaskEmbedder
from dreamer.actions import Actions
from dreamer.sampler import sample_video
from dreamer.utils import _ensure_dir, normalize_with_dataset_stats, apply_border, normalize_latents, unnormalize_latents


# ---------------------------
# RMS Loss Normalization
# ---------------------------

class RMSLossNormalizer(nnx.Module):
    """Normalizes loss terms by their running RMS estimates."""

    def __init__(
        self,
        loss_names: list[str],
        beta: float = 0.95,
        eps: float = 1e-6,
    ):
        """
        Args:
            loss_names: List of loss term names to track
            beta: EMA decay factor (higher = slower adaptation)
            eps: Small constant for numerical stability
        """
        self.beta = beta
        self.eps = eps

        self.stats = {
            # Initialize to 1.0 so first step doesn't explode
            name: nnx.BatchStat(jnp.array(1.0)) 
            for name in loss_names
        }

    def __call__(
        self,
        losses: Dict[str, jnp.ndarray],
        update_ema: bool = True,
    ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
        """
        Normalize losses by their running RMS estimates.

        Args:
            losses: Dict mapping loss names to scalar loss values
            update_ema: Whether to update running statistics (False for eval)

        Returns:
            normalized_losses: Dict of normalized loss values
            rms_values: Dict of current RMS estimates (for logging)
        """
        normalized_losses = {}
        rms_values = {}

        for name, loss_val in losses.items():
            if name not in self.stats:
                continue

            stat = self.stats[name]
            rms = jnp.sqrt(stat.value)

            if update_ema:
                decay = 1.0 - self.beta
                loss_sq = jax.lax.stop_gradient(loss_val) ** 2
                stat.value = self.beta * stat.value + decay * loss_sq

            normalized_losses[name] = loss_val / jnp.maximum(rms, self.eps)
            rms_values[name] = rms

        return normalized_losses, rms_values


# ---------------------------
# Sampling utilities
# ---------------------------

@partial(jax.jit, static_argnames=("shape_bt", "k_max", "dtype", "include_endpoint"))
def sample_tau_for_step(
    rng: jax.Array,
    shape_bt: Tuple[int, int],
    k_max: int,
    step_idx: jnp.ndarray,
    *,
    dtype=jnp.float32,
    include_endpoint: bool = True
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
        include_endpoint: If True, include tau=1.0 (clean). Set to False for bootstrap
                         training which needs room for half-steps.

    Returns:
        tau: (B, T) Signal levels in [0, 1] (or [0, 1-d] if include_endpoint=False)
        tau_idx: (B, T) Discrete indices in [0, k_max]

    Example:
        If step_idx = 3, then K = 2^3 = 8, d = 1/8
        include_endpoint=True:  tau sampled from {0, 1/8, 2/8, ..., 7/8, 1}
        include_endpoint=False: tau sampled from {0, 1/8, 2/8, ..., 7/8}
    """
    B_, T_ = shape_bt
    K = 1 << step_idx  # 2^step_idx
    u = jax.random.uniform(rng, (B_, T_), dtype=dtype)
    if include_endpoint:
        # Sample j_idx from {0, 1, ..., K} to include tau=1.0
        j_idx = jnp.floor(u * (K + 1).astype(dtype)).astype(jnp.int32)
        j_idx = jnp.minimum(j_idx, K)  # handle edge case where u=1.0
    else:
        # Sample j_idx from {0, 1, ..., K-1} to exclude tau=1.0
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


# Loss weighting
def ramp_weight(sigma: jnp.ndarray, min_weight: float = 0.1, max_weight: float = 1.0) -> jnp.ndarray:
    return (max_weight - min_weight) * sigma + min_weight


def apply_ot_coupling(
    z0: jnp.ndarray,
    z1: jnp.ndarray,
    rng: jax.Array | None,
    *,
    ot_cfg: OptimalTransportConfig,
) -> jnp.ndarray:
    """
    Apply minibatch OT coupling between noise (z0) and data (z1).

    Coupling is computed at the per-sequence level, treating each (T, S, D)
    sequence as one sample. The output is a reweighted/assigned z0 aligned to
    the ordering of z1.
    """
    if (not ot_cfg.enabled) or (z0.shape[0] < 2):
        return z0

    B = z0.shape[0]
    x = z0.reshape(B, -1).astype(jnp.float32)
    y = z1.reshape(B, -1).astype(jnp.float32)

    a = jnp.full((B,), 1.0 / B, dtype=jnp.float32)
    b = jnp.full((B,), 1.0 / B, dtype=jnp.float32)

    # Local import to avoid hard dependency if OT is disabled.
    from ott.geometry import pointcloud
    from ott.solvers import linear

    geom = pointcloud.PointCloud(x, y, epsilon=ot_cfg.epsilon, scale_cost=ot_cfg.scale_cost)
    out = linear.solve(
        geom,
        a=a,
        b=b,
        lse_mode=ot_cfg.lse_mode,
        threshold=ot_cfg.threshold,
        max_iterations=ot_cfg.max_iter,
    )

    P = jax.lax.stop_gradient(out.matrix)

    if ot_cfg.pairing == "barycentric":
        # Map y-indexed samples to x via column-normalized transport.
        col_sum = jnp.sum(P, axis=0, keepdims=True)
        col_sum = jnp.maximum(col_sum, 1e-8)
        x_coupled = (P.T @ x) / col_sum.T
        # Renormalize per sample to match N(0, 1) statistics.
        mean = jnp.mean(x_coupled, axis=1, keepdims=True)
        var = jnp.mean((x_coupled - mean) ** 2, axis=1, keepdims=True)
        x_coupled = x_coupled / jnp.maximum(jnp.sqrt(var), 1e-8)
    elif ot_cfg.pairing == "argmax":
        # Hard assignment per y sample (not necessarily a permutation).
        idx = jnp.argmax(P, axis=0)
        x_coupled = x[idx]
    elif ot_cfg.pairing == "sample":
        if rng is None:
            raise ValueError("apply_ot_coupling: rng is required for pairing='sample'.")
        keys = jax.random.split(rng, B)
        logits = jnp.log(jnp.maximum(P.T, 1e-8))
        idx = jax.vmap(jax.random.categorical)(keys, logits)
        x_coupled = x[idx]
    else:
        raise ValueError(f"apply_ot_coupling: unknown pairing mode '{ot_cfg.pairing}'.")

    return x_coupled.reshape(z0.shape).astype(z0.dtype)


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
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    return jnp.mean(mse_per_step * weights), jnp.mean(mse_per_step), mse_per_step


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
    or two smaller steps. Computed directly in x-space by algebraically cancelling
    the (1-σ)² and 1/(1-σ) terms:
        (1-σ)² · ||(z_pred - z_tilde)/(1-σ) - v_target||²
      = ||z_pred - (z_tilde + (1-σ)·v_target)||²

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
    # Target velocity is average of two half-steps (stop gradient, clipped)
    v_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)
    v_target = jnp.clip(v_target, -4.0, 4.0)

    # X-space target avoids the numerically unstable 1/(1-σ) division in v_hat
    one_minus_sigma = (1.0 - sigma)[..., None, None]
    x_target = z_tilde + one_minus_sigma * v_target

    boot_per_token = (z_pred - x_target) ** 2
    boot_per_step = jnp.mean(boot_per_token, axis=(2, 3))  # (B, T)

    # Apply ramp weighting and reduce
    weights = ramp_weight(sigma)
    return jnp.mean(boot_per_step * weights), jnp.mean(boot_per_step)


# ---------------------------
# Shortcut forcing step logic
# ---------------------------

def shortcut_forcing_step(
    dynamics_model: Dynamics,
    actions: Actions,
    latents: jnp.ndarray,
    rng: jax.Array,
    k_max: int,
    *,
    B_self: int = 0,
    context_length: int | None = None,
    time_mask: jnp.ndarray | None = None,
    task_embeddings: jnp.ndarray | None = None,
    bootstrap_model: Dynamics | None = None,
    ot_cfg: OptimalTransportConfig = OptimalTransportConfig(),
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
        context_length: optional context length for sliding window attention. If provided,
                       creates local_window_size=(context_length - 1, 0) for causal sliding window.
        task_embeddings: Optional (B, T, n_agent, d_model) agent tokens
        bootstrap_model: Model used for the two stop-gradient half-steps that generate bootstrap
                        targets. Defaults to dynamics_model when None. Pass an EMA model here
                        for more stable bootstrap targets.
        ot_cfg: OT coupling settings.

    Returns:
        losses: Dict with 'total', 'flow', 'bootstrap' keys
        aux: Dict with auxiliary metrics for logging. If task_embeddings is not None,
             aux contains 'h_states' key with (B, T, n_agent, d_model) hidden states
             from the main forward pass (computed with noisy inputs)
    """
    B, T, S, D = latents.shape
    B_emp = B - B_self
    emax = jnp.log2(k_max).astype(jnp.int32)

    # Normalize latents before corruption (all operations happen in normalized space)
    latents = normalize_latents(latents, dynamics_model.cfg.latent_mean, dynamics_model.cfg.latent_std)

    # Split RNG
    key_sigma_emp, key_sigma_self, key_step, key_noise, key_dropout1, key_ot = jax.random.split(rng, 6)

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
    # Empirical rows: include tau=1.0 (clean frames) for diffusion forcing flow loss
    sigma_emp, sigma_idx_emp = sample_tau_for_step(
        key_sigma_emp, (B_emp, T), k_max, step_idx_emp, dtype=latents.dtype, include_endpoint=True
    )
    # Bootstrap rows: exclude tau=1.0 (need room for half-steps in bootstrap loss)
    if B_self > 0:
        sigma_self, sigma_idx_self = sample_tau_for_step(
            key_sigma_self, (B_self, T), k_max, step_idx_self, dtype=latents.dtype, include_endpoint=False
        )
    else:
        sigma_self = jnp.zeros((0, T), dtype=latents.dtype)
        sigma_idx_self = jnp.zeros((0, T), dtype=jnp.int32)

    sigma_full = jnp.concatenate([sigma_emp, sigma_self], axis=0)
    sigma_idx_full = jnp.concatenate([sigma_idx_emp, sigma_idx_self], axis=0)

    # --- Corrupt latents: z_tilde = (1 - (1-eps)*sigma) * z0 + sigma * z1 ---
    # The 1e-5 epsilon prevents the ODE from becoming singular at sigma=1
    z0 = jax.random.normal(key_noise, latents.shape, dtype=latents.dtype)
    z0 = apply_ot_coupling(
        z0,
        latents,
        key_ot,
        ot_cfg=ot_cfg,
    )
    z_tilde = (1.0 - (1.0 - 1e-5) * sigma_full[..., None, None]) * z0 + sigma_full[..., None, None] * latents
    
    # --- Forward pass (full batch) ---
    rngs1 = nnx.Rngs(dropout=key_dropout1)
    z_pred_full, (h_states, _) = dynamics_model(
        actions, step_idx_full, sigma_idx_full, z_tilde,
        context_length=context_length, time_mask=time_mask, task_embeddings=task_embeddings, deterministic=False, rngs=rngs1
    )
    
    # --- Flow loss (empirical rows) ---
    z_pred_emp = z_pred_full[:B_emp]
    loss_flow, flow_mse_unweighted, mse_per_step_emp = compute_flow_loss(z_pred_emp, latents[:B_emp], sigma_emp)

    # Per-σ-bin flow MSE: breaks down where failures occur on the noise schedule
    mask_low  = sigma_emp < 0.25
    mask_mid  = (sigma_emp >= 0.25) & (sigma_emp < 0.75)
    mask_high = sigma_emp >= 0.75
    flow_mse_low  = jnp.sum(mse_per_step_emp * mask_low)  / jnp.maximum(jnp.sum(mask_low.astype(jnp.float32)),  1.0)
    flow_mse_mid  = jnp.sum(mse_per_step_emp * mask_mid)  / jnp.maximum(jnp.sum(mask_mid.astype(jnp.float32)),  1.0)
    flow_mse_high = jnp.sum(mse_per_step_emp * mask_high) / jnp.maximum(jnp.sum(mask_high.astype(jnp.float32)), 1.0)
    
    # --- Bootstrap loss (self-consistency rows) ---
    loss_boot = jnp.array(0.0, dtype=latents.dtype)
    boot_mse_unweighted = jnp.array(0.0, dtype=latents.dtype)
    boot_target_norm = jnp.array(0.0, dtype=latents.dtype)
    
    if B_self > 0:
        z_pred_self = z_pred_full[B_emp:]
        z_tilde_self = z_tilde[B_emp:]
        actions_self = actions[B_emp:]
        time_mask_self = time_mask[B_emp:] if time_mask is not None else None  # assume aligned with batch
        task_embeddings_self = task_embeddings[B_emp:] if task_embeddings is not None else None

        # Half-step metadata
        d_half = d_self / 2.0
        step_idx_half = step_idx_self + 1
        sigma_plus = sigma_self + d_half
        sigma_idx_plus = sigma_idx_self + (k_max * d_half).astype(jnp.int32)

        _bootstrap = bootstrap_model if bootstrap_model is not None else dynamics_model

        # First half-step
        z1_half1, *_ = _bootstrap(
            actions_self, step_idx_half, sigma_idx_self, z_tilde_self,
            context_length=context_length, time_mask=time_mask_self, task_embeddings=task_embeddings_self, deterministic=True
        )
        z1_half1 = jax.lax.stop_gradient(z1_half1)
        b_prime = (z1_half1 - z_tilde_self) / jnp.maximum(1.0 - sigma_self[..., None, None], 1e-5)
        z_prime = z_tilde_self + b_prime * d_half[..., None, None]
        z_prime = jnp.clip(z_prime, -4.0, 4.0)

        # Second half-step
        z1_half2, *_ = _bootstrap(
            actions_self, step_idx_half, sigma_idx_plus, z_prime,
            context_length=context_length, time_mask=time_mask_self, task_embeddings=task_embeddings_self, deterministic=True
        )
        z1_half2 = jax.lax.stop_gradient(z1_half2)
        b_doubleprime = (z1_half2 - z_prime) / jnp.maximum(1.0 - sigma_plus[..., None, None], 1e-5)

        # Bootstrap target health diagnostics (before clipping)
        v_target_unclipped = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)
        boot_target_norm = jnp.mean(jnp.abs(v_target_unclipped))

        # Bootstrap loss (computed unconditionally)
        loss_boot, boot_mse_unweighted = compute_bootstrap_loss(z_pred_self, z_tilde_self, b_prime, b_doubleprime, sigma_self)
    
    # --- Combine losses ---
    # Weight by batch composition to keep scale constant
    loss_total = ((loss_flow * (B - B_self)) + (loss_boot * B_self)) / B
    
    losses = {'total': loss_total, 'flow': loss_flow, 'bootstrap': loss_boot}
    aux = {
        'flow_mse': flow_mse_unweighted,
        'bootstrap_mse': boot_mse_unweighted,
        'flow_mse_low': flow_mse_low,
        'flow_mse_mid': flow_mse_mid,
        'flow_mse_high': flow_mse_high,
        'boot_target_norm': boot_target_norm,
        'h_states': h_states,
    }
    
    
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
    
    # Negative set: discourage high log-prob (bad actions → push probability down)
    loss_negative = jnp.where(
        n_negative > 0,
        (1 - alpha) * jnp.sum(jnp.where(mask_negative, logp_flat, 0.0)) / n_negative,
        0.0,
    )

    # Positive set: encourage high log-prob (good actions → push probability up)
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
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute reward prediction loss with symexp twohot encoding.
    
    Args:
        reward_head: Reward head NNX model instance
        h_states: (B, T, n_agent, d_model) Hidden states from dynamics
        rewards_btL: (B, T, L) Future reward values
        rewards_valid: (B, T, L) Validity mask
        
    Returns:
        reward_loss: Scalar categorical cross-entropy loss
        metrics: Dict with breakdown metrics
    """
    # Forward pass
    reward_logits, centers_log = reward_head(h_states, deterministic=True) 
    
    assert rewards_valid.dtype == jnp.bool_, "rewards_valid must be of type bool"
    reward_targets = twohot_symlog_targets(rewards_btL, centers_log)  # (B, T, L, K)
    reward_loss_per = optax.safe_softmax_cross_entropy(logits=reward_logits, labels=reward_targets)  # (B, T, L)
    reward_loss = jnp.sum(reward_loss_per * rewards_valid) / jnp.maximum(rewards_valid.sum(), 1.0)
    # add metrics on loss over when reward is nonzero and when it is zero.
    reward_nonzero = (rewards_btL > 0) * rewards_valid
    reward_zero = (rewards_btL == 0) * rewards_valid
    reward_nonzero_count = reward_nonzero.sum()
    reward_zero_count = reward_zero.sum()
    metrics = {
        "reward_nonzero_count": reward_nonzero_count,
        "reward_zero_count": reward_zero_count,
        "reward_loss_nonzero": jnp.sum(reward_loss_per * reward_nonzero) / jnp.maximum(reward_nonzero_count, 1.0),
        "reward_loss_zero": jnp.sum(reward_loss_per * reward_zero) / jnp.maximum(reward_zero_count, 1.0),
    }
    return reward_loss, metrics



def compute_policy_loss(
    policy_head,
    h_states: jnp.ndarray,
    actions: Actions,
    actions_valid: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    Compute behavior cloning loss with multi-token prediction.

    Args:
        policy_head: Policy head NNX model instance
        h_states: (B, T, n_agent, d_model) Hidden states from dynamics
        actions: Actions object with (B, T, L, ...) future action labels
        actions_valid: (B, T, L) Validity mask

    Returns:
        losses: Dict with individual losses per modality ('binary', 'categorical', 'continuous')
    """
    assert actions_valid.dtype == jnp.bool_, "actions_valid must be of type bool"

    # Forward pass - returns dict with logits for each action type
    policy_outputs = policy_head(h_states, deterministic=True)

    losses = {}

    # Binary actions: BCE loss per key
    if "binary_logits" in policy_outputs and actions.binary is not None:
        logits = policy_outputs["binary_logits"]  # (B, T, L, num_keys)
        targets = actions.binary  # (B, T, L, num_keys)
        bce = optax.sigmoid_binary_cross_entropy(logits, targets.astype(jnp.float32))
        # Average over keys, mask over (B, T, L)
        bce_per_step = jnp.mean(bce, axis=-1)  # (B, T, L)
        losses['binary'] = jnp.sum(bce_per_step * actions_valid) / jnp.maximum(actions_valid.sum(), 1.0)

    # Categorical action: CE loss
    if "categorical_logits" in policy_outputs and actions.categorical is not None:
        logits = policy_outputs["categorical_logits"]  # (B, T, L, action_dim)
        targets = actions.categorical  # (B, T, L)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)  # (B, T, L)
        losses['categorical'] = jnp.sum(ce * actions_valid) / jnp.maximum(actions_valid.sum(), 1.0)

    # Continuous action: Gaussian NLL
    if "continuous_mean" in policy_outputs and actions.continuous is not None:
        mean = policy_outputs["continuous_mean"]  # (B, T, L, dim)
        log_var = policy_outputs["continuous_log_var"]  # (B, T, L, dim)
        targets = actions.continuous  # (B, T, L, dim)
        # NLL = 0.5 * (log_var + (x - mu)^2 / exp(log_var))
        nll = 0.5 * (log_var + (targets - mean) ** 2 * jnp.exp(-log_var))
        # Average over action dimensions, mask over (B, T, L)
        nll_per_step = jnp.mean(nll, axis=-1)  # (B, T, L)
        losses['continuous'] = jnp.sum(nll_per_step * actions_valid) / jnp.maximum(actions_valid.sum(), 1.0)

    return losses


# ---------------------------
# Evaluation and visualization
# ---------------------------


def run_evaluation(
    cfg: DynamicsConfig,
    step: int,
    tokenizer: Tokenizer,
    dynamics_online: Dynamics,
    dynamics_ema: Dynamics,
    *,
    val_data: jnp.ndarray,
    val_actions: Actions,
    use_latent_data: bool,
    vis_dir: Path,
    rng: jax.Array,
    logger,
):
    """
    Run a consolidated periodic dynamics evaluation and save one grid video.

    Grid layout:
      - Rows: rollout samples
      - Columns: [ground_truth, online_diffusion, ema_diffusion, online_shortcut, ema_shortcut]

    PSNR is computed on only the first generated frame (t = ctx_length).
    Also runs x0 and attention visualizations for both online and EMA dynamics.
    """
    if dynamics_online.cfg.k_max != dynamics_ema.cfg.k_max:
        raise ValueError(
            f"Expected matching k_max for online/ema dynamics, got "
            f"{dynamics_online.cfg.k_max} and {dynamics_ema.cfg.k_max}."
        )

    T = val_data.shape[1]
    assert T > 5, f"Sequence length {T} must be > 5"
    ctx_length = 4
    horizon = T - ctx_length
    k_max = dynamics_online.cfg.k_max

    rollout_specs = [
        ("online_diffusion", dynamics_online, DenoiseSchedule.init(k_max, k_max)),
        ("ema_diffusion", dynamics_ema, DenoiseSchedule.init(k_max, k_max)),
        ("online_shortcut", dynamics_online, DenoiseSchedule.init(4, k_max)),
        ("ema_shortcut", dynamics_ema, DenoiseSchedule.init(4, k_max)),
    ]

    dataset_std = cfg.dataset.dataset_std[0]
    psnr_windows = (1, 3, 8)
    pred_columns: Dict[str, jnp.ndarray] = {}
    rollout_metrics: Dict[str, Dict[str, float]] = {}
    ground_truth_frames: jnp.ndarray | None = None

    for tag, dynamics_model, schedule_config in rollout_specs:
        t0 = time.time()
        rng, eval_rng = jax.random.split(rng)

        if use_latent_data:
            pred_frames, gt_decoded_frames, _, _ = sample_video(
                tokenizer, dynamics_model, frames=None,
                actions=val_actions, horizon=horizon, schedule_config=schedule_config,
                rng=eval_rng, policy=None, task_embedder=None,
                latents=val_data,
            )
            gt_frames_for_metrics = gt_decoded_frames
        else:
            pred_frames, _, original_frames, _ = sample_video(
                tokenizer, dynamics_model, frames=val_data,
                actions=val_actions, horizon=horizon, schedule_config=schedule_config,
                rng=eval_rng, policy=None, task_embedder=None,
            )
            assert original_frames is not None
            gt_frames_for_metrics = original_frames

        if ground_truth_frames is None:
            ground_truth_frames = gt_frames_for_metrics

        normalized_pred = normalize_with_dataset_stats(
            pred_frames[:, -horizon:], mean=0, std=dataset_std
        )
        normalized_gt = normalize_with_dataset_stats(
            gt_frames_for_metrics[:, -horizon:], mean=0, std=dataset_std
        )
        mse = float(jnp.mean((normalized_pred - normalized_gt) ** 2))
        mse_values: Dict[int, float] = {
            n: float(
                jnp.mean(
                    (normalized_pred[:, :min(n, horizon)] - normalized_gt[:, :min(n, horizon)]) ** 2
                )
            )
            for n in psnr_windows
        }

        def _compute_window_psnr(n: int) -> float:
            n_eval = min(n, horizon)
            return float(
                compute_psnr(
                    pred_frames[:, ctx_length:ctx_length + n_eval] / 255,
                    gt_frames_for_metrics[:, ctx_length:ctx_length + n_eval] / 255,
                )
            )

        psnr_values: Dict[int, float] = {n: _compute_window_psnr(n) for n in psnr_windows}
        dt = time.time() - t0
        psnr_log = " | ".join(f"PSNR@{n}={psnr_values[n]:.2f} dB" for n in psnr_windows)

        print(
            f"[eval:{tag}] step={step:06d} | horizon={horizon} | "
            f"MSE={mse:.6g} | {psnr_log} | {dt:.2f}s"
        )

        pred_frames = pred_frames.at[:, :ctx_length].set(apply_border(pred_frames[:, :ctx_length]))
        pred_columns[tag] = pred_frames
        rollout_metrics[tag] = {
            "mse": mse,
            **{f"mse_{n}": mse_values[n] for n in psnr_windows},
            **{f"psnr_{n}": psnr_values[n] for n in psnr_windows},
            "eval_time": dt,
        }

    assert ground_truth_frames is not None
    num_videos = min(4, ground_truth_frames.shape[0])

    grid_columns = [
        ground_truth_frames,
        pred_columns["online_diffusion"],
        pred_columns["ema_diffusion"],
        pred_columns["online_shortcut"],
        pred_columns["ema_shortcut"],
    ]
    stacked_frames = jnp.stack(grid_columns)[:, :num_videos]
    videos = rearrange(stacked_frames, 'S B T H W C -> T (B H) (S W) C', B=num_videos)

    tag_dir = _ensure_dir(vis_dir / f"step_{step:06d}")
    mp4_path = tag_dir / "rollouts_grid.mp4"

    video_written = False
    try:
        videos = jax.device_get(videos)
        iio.imwrite(str(mp4_path), videos, fps=5, plugin='pyav', codec='libx264')
        video_written = True
    except Exception as e:
        print(f"[eval] consolidated MP4 write failed: {e}")

    metrics_payload: Dict[str, float] = {"horizon": float(horizon)}
    for tag, _, _ in rollout_specs:
        metrics_payload[f"{tag}/mse"] = rollout_metrics[tag]["mse"]
        for n in psnr_windows:
            metrics_payload[f"mse/{n}_step/{tag}"] = rollout_metrics[tag][f"mse_{n}"]
            metrics_payload[f"psnr/{n}_step/{tag}"] = rollout_metrics[tag][f"psnr_{n}"]
        metrics_payload[f"{tag}/eval_time"] = rollout_metrics[tag]["eval_time"]
    logger.log_metrics(step, metrics_payload, prefix="eval/")

    if video_written:
        logger.log_video(step, "eval/rollouts_grid/video", mp4_path)


# ---------------------------
# x0 visualization
# ---------------------------

@nnx.jit(static_argnames=("k_max", "T", "context_length", "use_latent_data"))
def vis_dynamics_step(
    tokenizer,
    dynamics,
    data: jnp.ndarray,
    actions,
    *,
    master_key: jax.Array,
    step: int,
    T: int,
    k_max: int,
    context_length: int | None,
    use_latent_data: bool,
):
    """Single-sequence (B=1) forward pass for x0 visualization.

    Returns z_tilde, z_pred, latents_norm (all shape (1,T,S,D)) and sigma (1,T).
    """
    if use_latent_data:
        latents = data
    else:
        latents, _ = tokenizer.encode(data, deterministic=True)
        latents = jax.lax.stop_gradient(latents)

    latents = latents.astype(dynamics.dtype)
    latents_norm = normalize_latents(latents, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)

    step_key = jax.random.fold_in(master_key, step)
    key_sigma, key_noise = jax.random.split(step_key)

    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jnp.full((1, T), emax, dtype=jnp.int32)

    sigma, sigma_idx = sample_tau_for_step(
        key_sigma, (1, T), k_max, step_idx, dtype=latents.dtype, include_endpoint=True
    )

    z0 = jax.random.normal(key_noise, latents_norm.shape, dtype=latents.dtype)
    z_tilde = (1.0 - (1.0 - 1e-5) * sigma[..., None, None]) * z0 + sigma[..., None, None] * latents_norm

    z_pred, _ = dynamics(
        actions, step_idx, sigma_idx, z_tilde,
        context_length=context_length, time_mask=None, task_embeddings=None, deterministic=True,
    )

    return z_tilde, z_pred, latents_norm, sigma


def run_x0_visualization(
    cfg,
    step: int,
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    *,
    data: jnp.ndarray,
    actions,
    master_key: jax.Array,
    use_latent_data: bool,
    vis_dir: Path,
    logger,
    name: str | None = None,
):
    """Decode noisy input, x0 prediction, and ground-truth as a contact sheet.

    Rows: [gt (green border), z_tilde / noisy input (yellow), z_pred / x0 prediction (blue)]
    Columns: one per time step, labelled with τ value.
    """
    from PIL import Image, ImageDraw
    import numpy as np

    k_max = cfg.dynamics.k_max
    context_length = cfg.dynamics.context_length
    T = int(data.shape[1])

    z_tilde, z_pred, latents_norm, sigma = vis_dynamics_step(
        tokenizer, dynamics, data, actions,
        master_key=master_key,
        step=step,
        T=T,
        k_max=k_max,
        context_length=context_length,
        use_latent_data=use_latent_data,
    )

    # Decode each set of latents (unnormalize first, then run tokenizer decoder)
    @nnx.jit
    def decode_latents(z_norm):
        z = unnormalize_latents(z_norm, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)
        frames, _ = tokenizer.decode(z, deterministic=True)
        return jnp.clip(frames, 0, 255).astype(jnp.uint8)

    gt_frames    = jax.device_get(decode_latents(latents_norm))  # (1, T, H, W, 3)
    noisy_frames = jax.device_get(decode_latents(z_tilde))       # (1, T, H, W, 3)
    pred_frames  = jax.device_get(decode_latents(z_pred))        # (1, T, H, W, 3)
    sigma_cpu    = jax.device_get(sigma)[0]                       # (T,)

    H, W = gt_frames.shape[2], gt_frames.shape[3]
    label_h = 20

    # Build (3*H + label_h) × (T*W) contact sheet
    grid = np.zeros((label_h + 3 * H, T * W, 3), dtype=np.uint8)
    for t in range(T):
        x = t * W
        grid[label_h:label_h + H,         x:x + W] = gt_frames[0, t]
        grid[label_h + H:label_h + 2 * H, x:x + W] = noisy_frames[0, t]
        grid[label_h + 2 * H:,            x:x + W] = pred_frames[0, t]

    # Stamp τ labels
    pil_img = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_img)
    for t in range(T):
        draw.text((t * W + 2, 2), f"sigma={float(sigma_cpu[t]):.2f}", fill=(255, 255, 255))

    img_array = np.array(pil_img)

    log_prefix = f"{name}/" if name else ""
    eval_prefix = f"{log_prefix}eval/"

    # Save PNG
    out_dir = _ensure_dir(vis_dir / f"step_{step:06d}" / log_prefix)
    Image.fromarray(img_array).save(str(out_dir / "x0_vis.png"))

    # Log
    logger.log_image(step, f"{eval_prefix}x0_vis", img_array, caption=f"step {step}")


def run_attention_visualization(
    cfg,
    step: int,
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    *,
    data: jnp.ndarray,
    actions,
    use_latent_data: bool,
    vis_dir: Path,
    logger,
    name: str | None = None,
):
    """Visualize per-layer attention weights to diagnose attention entropy collapse.

    Produces two plots:
      1. Entropy heatmap — (n_layers × n_heads), colour = mean entropy over query positions.
         Low values indicate attention collapse (each query attends to ≤1 position).
      2. Temporal attention matrices — one T×T heatmap per temporal layer (every time_every
         layers), averaged over spatial tokens and attention heads.

    These plots are inspired by the Self Forcing paper (arXiv 2506.08009) and can be used to
    compare healthy (small dataset) vs collapsed (large dataset) attention patterns.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from einops import rearrange

    # --- 1. Prepare a single sample of latents (B=1) ---
    sample_data = data[:1]      # (1, T, ...)
    sample_actions = actions[:1]  # (1, T, ...)

    if use_latent_data:
        latents = sample_data.astype(dynamics.dtype)
    else:
        latents, _ = tokenizer.encode(sample_data, deterministic=True)
        latents = jax.lax.stop_gradient(latents).astype(dynamics.dtype)

    latents = normalize_latents(latents, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)

    B, T = latents.shape[0], latents.shape[1]

    # --- 2. Build the token sequence (mirrors Dynamics.__call__) ---
    # Use clean-data signal level (τ=1, finest step size)
    tau_indices  = jnp.full((B, T), dynamics.cfg.k_max, dtype=jnp.int32)   # σ=1.0
    step_indices = jnp.zeros((B, T), dtype=jnp.int32)                       # step=1/k_max

    packed_enc_tokens = rearrange(latents, "b t (n p) d -> b t n (p d)", p=dynamics.packing_factor)
    spatial_tokens = dynamics.spatial_proj(packed_enc_tokens)
    action_token   = dynamics.action_encoder(actions=sample_actions, batch_time_shape=(B, T))
    register_tokens = jnp.broadcast_to(
        dynamics.register_tokens.value.astype(dynamics.dtype)[None, None],
        (B, T, dynamics.n_register, dynamics.d_model),
    )
    step_emb   = dynamics.step_embed(step_indices.astype(jnp.float32))
    signal_emb = dynamics.signal_embed(tau_indices.astype(jnp.float32) / dynamics.k_max)
    shortcut_token = jnp.concatenate([step_emb, signal_emb], axis=-1)[:, :, None, :]

    tokens = jnp.concatenate([action_token, shortcut_token, spatial_tokens, register_tokens], axis=2)

    n_latents = latents.shape[2]
    layout = dynamics.get_token_layout(n_latents=n_latents, n_agent=0)
    space_mask = layout.build_space_mask("wm_agent")

    # --- 3. Forward pass with attention weight capture (no JIT) ---
    _, _, all_weights = dynamics.transformer(
        tokens,
        space_mask=space_mask,
        time_mask=None,
        time_local_window_size=None,
        deterministic=True,
        caches=None,
        rngs=None,
        return_weights=True,
    )

    all_weights = jax.device_get(all_weights)  # list of numpy arrays, one per layer

    # --- 4. Compute per-layer per-head mean entropy ---
    n_layers = len(all_weights)
    layer_is_time = [layer.is_time_layer for layer in dynamics.transformer.layers]
    n_heads = dynamics.cfg.n_heads

    entropies = np.zeros((n_layers, n_heads), dtype=np.float32)
    time_layer_weights = {}  # layer_idx -> (N, T, T) averaged over B and S

    for i, w in enumerate(all_weights):
        if w is None:
            continue
        is_time = layer_is_time[i]
        if is_time:
            # w: (B, S, N, T, T) — causal temporal attention
            w_np = np.array(w)                   # (B, S, N, T, T)
            w_avg = w_np.mean(axis=(0, 1))       # (N, T, T)
            time_layer_weights[i] = w_avg
            # entropy along key axis, averaged over query tokens
            H = -np.where(w_avg > 0, w_avg * np.log(w_avg + 1e-9), 0.0).sum(axis=-1)  # (N, T)
            entropies[i] = H.mean(axis=-1)       # (N,)
        else:
            # w: (B, T, N, S, S) — spatial attention
            w_np = np.array(w)                   # (B, T, N, S, S)
            w_avg = w_np.mean(axis=(0, 1))       # (N, S, S)
            H = -np.where(w_avg > 0, w_avg * np.log(w_avg + 1e-9), 0.0).sum(axis=-1)  # (N, S)
            entropies[i] = H.mean(axis=-1)       # (N,)

    # --- 5. Build figure ---
    n_time_layers = len(time_layer_weights)
    n_cols = 1 + max(n_time_layers, 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, max(4, 0.35 * n_layers + 1)))

    # Left panel: entropy heatmap (n_layers × n_heads)
    ax_ent = axes[0]
    im = ax_ent.imshow(entropies, aspect="auto", cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax_ent, label="mean entropy (nats)")
    ax_ent.set_xlabel("head")
    ax_ent.set_ylabel("layer")
    ax_ent.set_title("attention entropy\n(lower = more collapsed)")
    ytick_labels = []
    for li in range(n_layers):
        is_t = layer_is_time[li]
        ytick_labels.append(f"{li} {'[T]' if is_t else '[S]'}")
    ax_ent.set_yticks(range(n_layers))
    ax_ent.set_yticklabels(ytick_labels, fontsize=7)

    # Right panels: temporal attention matrices
    for j, (li, w_avg) in enumerate(sorted(time_layer_weights.items())):
        ax = axes[1 + j]
        # Average over heads for a single T×T map
        w_mean = w_avg.mean(axis=0)  # (T, T)
        im2 = ax.imshow(w_mean, aspect="auto", cmap="plasma", interpolation="nearest",
                        vmin=0, vmax=w_mean.max())
        fig.colorbar(im2, ax=ax)
        ax.set_title(f"layer {li} [T]\ntime attn (avg heads)")
        ax.set_xlabel("key pos")
        ax.set_ylabel("query pos")

    for j in range(n_time_layers, n_cols - 1):
        axes[1 + j].axis("off")

    log_prefix = f"{name}/" if name else ""
    eval_prefix = f"{log_prefix}eval/"

    fig.suptitle(f"Attention weights [{log_prefix}step {step}]", fontsize=11)
    fig.tight_layout()

    # --- 6. Convert figure to numpy array ---
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_array = np.array(Image.open(buf))[:, :, :3]  # drop alpha if present
    plt.close(fig)

    # --- 7. Save and log ---
    out_dir = _ensure_dir(vis_dir / f"step_{step:06d}" / log_prefix)
    Image.fromarray(img_array).save(str(out_dir / "attn_vis.png"))
    logger.log_image(step, f"{eval_prefix}attn_vis", img_array, caption=f"step {step}")
