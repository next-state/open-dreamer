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

from dreamer.configs import DynamicsConfig, HeadsConfig
from dreamer.generation import DenoiseSchedule
from dreamer.models import Dynamics, PolicyHeadMTP, TaskEmbedder
from dreamer.types import Actions
from dreamer.sampler import sample_video
from dreamer.utils import _ensure_dir, normalize_with_dataset_stats, apply_border


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
    task_embeddings: jnp.ndarray | None = None,
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
        task_embeddings: Optional (B, T, n_agent, d_model) agent tokens

    Returns:
        losses: Dict with 'total', 'flow', 'bootstrap' keys
        aux: Dict with auxiliary metrics for logging. If task_embeddings is not None,
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
        task_embeddings=task_embeddings, deterministic=False, rngs=rngs1
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
        task_embeddings_self = task_embeddings[B_emp:] if task_embeddings is not None else None

        # Half-step metadata
        d_half = d_self / 2.0
        step_idx_half = step_idx_self + 1
        sigma_plus = sigma_self + d_half
        sigma_idx_plus = sigma_idx_self + (k_max * d_half).astype(jnp.int32)

        # First half-step
        rngs2 = nnx.Rngs(dropout=key_dropout2)
        z1_half1, *_ = dynamics_model(
            actions_self, step_idx_half, sigma_idx_self, z_tilde_self,
            task_embeddings=task_embeddings_self, deterministic=False, rngs=rngs2
        )
        b_prime = (z1_half1 - z_tilde_self) / jnp.maximum(1.0 - sigma_self[..., None, None], 1e-8)
        z_prime = z_tilde_self + b_prime * d_half[..., None, None]

        # Second half-step
        rngs3 = nnx.Rngs(dropout=key_dropout3)
        z1_half2, *_ = dynamics_model(
            actions_self, step_idx_half, sigma_idx_plus, z_prime,
            task_embeddings=task_embeddings_self, deterministic=False, rngs=rngs3
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
    cfg: DynamicsConfig | HeadsConfig,
    step: int,
    tokenizer,
    dynamics,
    val_videos: jnp.ndarray,
    val_actions: Actions,
    vis_dir: Path,
    rng: jax.Array,
    logger,
    policy: PolicyHeadMTP | None = None,
    task_embedder: TaskEmbedder | None = None,
):
    """
    Run periodic evaluation: sample videos, compute metrics, and save visualization.

    This function can be used in both dynamics pretraining and agent finetuning.

    Args:
        cfg: Training config
        step: Current training step
        tokenizer: Tokenizer NNX model instance
        dynamics: Dynamics NNX model instance
        val_videos: (B, T, H, W, C) Validation videos
        val_actions: (B, T) Validation actions
        vis_dir: Directory to save visualizations
        rng: Random key
        logger: Logger instance for logging metrics and videos
        policy: Optional policy model for action sampling
        task_embedder: Optional task embedder for agent tokens
    """
    k_max = dynamics.cfg.k_max
    schedule_shortcut = DenoiseSchedule.init(4, k_max)
    schedule_diffusion = DenoiseSchedule.init(k_max, k_max)

    evaluation_schedules = {"shortcut": schedule_shortcut, "diffusion": schedule_diffusion}

    for tag, schedule_config in evaluation_schedules.items():
        t0 = time.time()
        # FIXME: only temporary for debugging
        assert val_videos.shape[1] > 5
        ctx_length = 4
        horizon = val_videos.shape[1] - ctx_length

        pred_frames, floor_frames, gt_frames = sample_video(
            tokenizer, dynamics, val_videos, 
            val_actions, horizon, schedule_config, rng, policy, task_embedder
        )

        # Compute metrics
        dt = time.time() - t0
        dataset_std = cfg.dataset.dataset_std[0]
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
