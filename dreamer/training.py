"""
Reusable training components for dynamics and imagination training.

This module contains:
- Sampling utilities for sigma (signal level)
- DuMo loss computation (velocity flow-matching + flow-map self-consistency)
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

from dreamer.configs import DynamicsConfig, HeadsConfig, OptimalTransportConfig, ConsistencyConfig
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

@partial(jax.jit, static_argnames=("shape_bt", "dtype"))
def sample_sigma_uniform(
    rng: jax.Array,
    shape_bt: Tuple[int, int],
    *,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """
    Sample continuous signal levels sigma ~ Uniform(0, 1), independently per (batch, time).

    DuMo draws the diffusion time from a Beta distribution; per the design decision here we
    keep the original diffusion-forcing uniform sampling. sigma=1 corresponds to clean latents
    and sigma=0 to pure noise: z_tilde = (1 - sigma) * z0 + sigma * z1.

    Args:
        rng: JAX random key
        shape_bt: Tuple of (batch_size, sequence_length)
        dtype: Data type for computation

    Returns:
        sigma: (B, T) continuous signal levels in [0, 1)
    """
    B_, T_ = shape_bt
    return jax.random.uniform(rng, (B_, T_), dtype=dtype)


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
    out = linear.solve(geom, a=a, b=b, lse_mode=ot_cfg.lse_mode, threshold=ot_cfg.threshold, max_iterations=ot_cfg.max_iter)

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
    Flow-matching loss in velocity space (direct prediction of the velocity v = z1 - z0).

    Args:
        z_pred: (B, T, S, D) Predicted velocity v_θ
        z_target: (B, T, S, D) Conditional velocity target v = z1 - z0
        sigma: (B, T) Signal levels (used for weighting)

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


def compute_consistency_loss(
    u_pred: jnp.ndarray,
    du_dsigma: jnp.ndarray,
    v_target: jnp.ndarray,
    sigma: jnp.ndarray,
    *,
    correction_clip: float = 1.0,
    adaptive_p: float = 1.0,
    adaptive_c: float = 1e-3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    DuMo flow-map (self-consistency) loss in velocity space.

    The flow-map head predicts the average velocity u_θ(z_tilde, sigma). Along the PF-ODE
    trajectory (parameterised by the signal level sigma, with dz_tilde/dsigma = z1 - z0 and
    dsigma/dsigma = 1) the MeanFlow identity reads

        u = v + (1 - sigma) * du/dsigma

    where v = z1 - z0 is the (conditional) instantaneous velocity and du/dsigma is the total
    derivative of u_θ along the trajectory (its JVP w.r.t. (z_tilde, sigma) in the tangent
    (z1 - z0, 1)). With theta- (stop-gradient on the online params) the target is

        u_target = stop_grad( v + (1 - sigma) * du/dsigma )

    and the loss is the L2 ||u_θ - u_target||^2. Keeping the loss in velocity space anchors the
    flow-map head to data at the clean boundary: as sigma -> 1 the correction vanishes and
    u_target -> v = z1 - z0, directly supervising u_θ with the conditional velocity.

    Two MeanFlow stabilisers guard against the JVP-magnitude runaway (du/dsigma blowing up as the
    velocity field sharpens):
      * correction_clip: the (1 - sigma) * du/dsigma term is clipped (the anchor v_target is left
        untouched), bounding the worst-case target without distorting the boundary supervision.
      * adaptive weighting w = 1 / (err2 + c)^p: each (B, T) sample is normalised by its own
        (detached) error so a few large-JVP samples cannot dominate the gradient.

    Args:
        u_pred: (B, T, S, D) Flow-map head velocity prediction u_θ (gradient-carrying)
        du_dsigma: (B, T, S, D) JVP tangent du/dsigma (from jax.jvp)
        v_target: (B, T, S, D) Conditional velocity v = z1 - z0
        sigma: (B, T) Signal levels
        correction_clip: symmetric clip on the (1 - sigma) * du/dsigma correction term
        adaptive_p: MeanFlow adaptive-weight exponent (0 disables; 1.0 = standard adaptive L2)
        adaptive_c: MeanFlow adaptive-weight stabiliser constant

    Returns:
        loss: scalar adaptive-weighted consistency loss (training objective)
        unweighted_mse: scalar mean (unweighted) MSE (the metric to monitor)
        target_norm: scalar mean |(1 - sigma) * du/dsigma| diagnostic (raw, pre-clip)
    """
    # Clip the correction itself (not the whole target) so the conditional-velocity anchor
    # v_target is never distorted; diagnose with the raw (pre-clip) magnitude.
    raw_correction = (1.0 - sigma)[..., None, None] * du_dsigma       # (1 - sigma) * du/dsigma
    correction = jnp.clip(raw_correction, -correction_clip, correction_clip)

    u_target = jax.lax.stop_gradient(v_target + correction)

    cons_per_step = jnp.mean((u_pred - u_target) ** 2, axis=(2, 3))   # (B, T)

    # MeanFlow adaptive weighting: weighted mean is the training loss, unweighted is the metric.
    weight = jax.lax.stop_gradient(1.0 / (cons_per_step + adaptive_c) ** adaptive_p)  # (B, T)
    cons_loss = jnp.mean(weight * cons_per_step)
    cons_mse = jnp.mean(cons_per_step)

    target_norm = jnp.mean(jnp.abs(jax.lax.stop_gradient(raw_correction)))
    return cons_loss, cons_mse, target_norm


# ---------------------------
# DuMo forcing step logic
# ---------------------------

def dumo_forcing_step(
    dynamics_model: Dynamics,
    actions: Actions,
    latents: jnp.ndarray,
    rng: jax.Array,
    *,
    beta: float = 0.5,
    B_img_emp: int = 0,
    context_length: int | None = None,
    time_mask: jnp.ndarray | None = None,
    task_embeddings: jnp.ndarray | None = None,
    ot_cfg: OptimalTransportConfig = OptimalTransportConfig(),
    cons_cfg: ConsistencyConfig = ConsistencyConfig(),
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """
    Compute DuMo dual-momentum losses (velocity flow + flow-map consistency) for a batch.

    DuMo trains two heads from one shared backbone, both in velocity space:
      * velocity head  (flow_x_head):    standard flow-matching velocity prediction -> L_v
      * flow-map head  (flow_map_head):  MeanFlow self-consistency velocity prediction -> L_u

    Both losses are applied to every sample (no bootstrap subset). The consistency target uses
    the true Jacobian-vector product du/dsigma (via jax.jvp over the shared backbone + flow-map
    head), evaluated along the PF-ODE trajectory with tangent (z1 - z0, 1) and stop-gradient on
    the target (theta- = stop-grad of the current online params). Diffusion forcing is preserved:
    sigma is sampled per-(batch, time) and the temporal/image structure is carried by time_mask.

    Args:
        dynamics_model: NNX Dynamics model instance
        actions: (B, T) Action sequence
        latents: (B, T, S, D) Latent sequence (ground truth, clean)
        rng: Random key
        beta: Weight on the velocity loss; consistency loss gets (1 - beta).
        B_img_emp: Number of rows treated as image-only (first rows of the batch). Used only to
                   split flow MSE logging into image vs full-sequence subsets.
        context_length: optional context length for sliding window attention. If provided,
                       creates local_window_size=(context_length - 1, 0) for causal sliding window.
        time_mask: optional (B, 1, T, T) boolean mask for temporal attention.
        task_embeddings: Optional (B, T, n_agent, d_model) agent tokens.
        ot_cfg: OT coupling settings.
        cons_cfg: flow-map consistency-loss stabilisers (correction clip + adaptive weighting).

    Returns:
        losses: Dict with 'total', 'flow', 'consistency' keys.
        aux: Dict with auxiliary metrics for logging. If task_embeddings is not None, aux
             contains 'h_states' with (B, T, n_agent, d_model) hidden states from the forward pass.
    """
    B, T = latents.shape[:2]

    # Normalize latents before corruption (all operations happen in normalized space)
    latents = normalize_latents(latents, dynamics_model.cfg.latent_mean, dynamics_model.cfg.latent_std)

    # Split RNG
    key_sigma, key_noise, key_ot = jax.random.split(rng, 3)

    # --- Sample continuous signal levels sigma ~ U(0, 1) (diffusion forcing, per token-frame) ---
    sigma = sample_sigma_uniform(key_sigma, (B, T), dtype=latents.dtype)

    # --- Corrupt latents: z_tilde = (1 - sigma) * z0 + sigma * z1 ---
    z0 = jax.random.normal(key_noise, latents.shape, dtype=latents.dtype)
    z0 = apply_ot_coupling(z0, latents, key_ot, ot_cfg=ot_cfg)
    sigma_b = sigma[..., None, None]
    z_tilde = (1.0 - sigma_b) * z0 + sigma_b * latents

    # --- Trajectory tangent for the JVP ---
    # The PF-ODE trajectory in the increasing-sigma direction has velocity dz_tilde/dsigma = z1 - z0,
    # and the conditioning advances as dsigma/dsigma = 1. The total derivative du/dsigma is then the
    # JVP of the flow-map prediction at (z_tilde, sigma) in the tangent (z1 - z0, 1).
    w = latents - z0                 # conditional velocity v = z1 - z0
    sigma_tangent = jnp.ones_like(sigma)

    # --- Dual-head forward pass + JVP over the flow-map head (one shared backbone pass) ---
    # jax.jvp differentiates only w.r.t. the explicit inputs (z_tilde, sigma); the model params are
    # captured as constants for forward-mode (theta-), while the primals still carry reverse-mode
    # gradient w.r.t. params for the loss backward pass.
    def _fwd(z_in, sigma_in):
        v_u_pair, (h_t, _) = dynamics_model(
            actions, sigma_in, z_in,
            head="both", context_length=context_length, time_mask=time_mask,
            task_embeddings=task_embeddings, deterministic=True,
        )
        v_pred, u_pred = v_u_pair
        return v_pred, u_pred, h_t

    (v_pred, u_pred, h_states), (_dv, du_dsigma, _dh) = jax.jvp(
        _fwd, (z_tilde, sigma), (w, sigma_tangent)
    )

    # --- Velocity (flow-matching) loss: L_v ---
    loss_flow, flow_mse_unweighted, mse_per_step = compute_flow_loss(v_pred, w, sigma)

    # Split flow MSE by sample type (image-only rows vs temporal sequence rows).
    n_img = min(max(B_img_emp, 0), B)
    n_seq = B - n_img
    if n_img > 0:
        flow_mse_image = jnp.mean(mse_per_step[:n_img])
    else:
        flow_mse_image = jnp.array(0.0, dtype=latents.dtype)

    if n_seq > 0:
        flow_mse_sequence = jnp.mean(mse_per_step[n_img:])
    else:
        flow_mse_sequence = jnp.array(0.0, dtype=latents.dtype)

    # Per-σ-bin flow MSE: breaks down where failures occur on the noise schedule
    mask_low  = sigma < 0.25
    mask_mid  = (sigma >= 0.25) & (sigma < 0.75)
    mask_high = sigma >= 0.75
    flow_mse_low  = jnp.sum(mse_per_step * mask_low)  / jnp.maximum(jnp.sum(mask_low.astype(jnp.float32)),  1.0)
    flow_mse_mid  = jnp.sum(mse_per_step * mask_mid)  / jnp.maximum(jnp.sum(mask_mid.astype(jnp.float32)),  1.0)
    flow_mse_high = jnp.sum(mse_per_step * mask_high) / jnp.maximum(jnp.sum(mask_high.astype(jnp.float32)), 1.0)

    # --- Flow-map (self-consistency) loss: L_u ---
    loss_cons, cons_mse_unweighted, cons_target_norm = compute_consistency_loss(
        u_pred, du_dsigma, w, sigma,
        correction_clip=cons_cfg.correction_clip,
        adaptive_p=cons_cfg.adaptive_p,
        adaptive_c=cons_cfg.adaptive_c,
    )

    # --- Combine losses ---
    loss_total = beta * loss_flow + (1.0 - beta) * loss_cons

    losses = {'total': loss_total, 'flow': loss_flow, 'consistency': loss_cons}
    aux = {
        'flow_mse': flow_mse_unweighted,
        'flow_mse_sequence': flow_mse_sequence,
        'flow_mse_image': flow_mse_image,
        'consistency_mse': cons_mse_unweighted,
        'flow_mse_low': flow_mse_low,
        'flow_mse_mid': flow_mse_mid,
        'flow_mse_high': flow_mse_high,
        'consistency_target_norm': cons_target_norm,
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
      - Columns: [ground_truth, online_velocity, ema_velocity, online_flowmap, ema_flowmap]

    PSNR is computed on only the first generated frame (t = ctx_length).
    Also runs x0 and attention visualizations for both online and EMA dynamics.
    """
    T = val_data.shape[1]
    assert T > 5, f"Sequence length {T} must be > 5"
    ctx_length = 4
    horizon = T - ctx_length
    velocity_steps = dynamics_online.cfg.num_sampling_steps
    flowmap_steps = 4

    # DuMo rollouts: velocity head (multi-step Euler) and flow-map head (few-step).
    rollout_specs = [
        ("online_velocity", dynamics_online, DenoiseSchedule.init(velocity_steps), "v"),
        ("ema_velocity", dynamics_ema, DenoiseSchedule.init(velocity_steps), "v"),
        ("online_flowmap", dynamics_online, DenoiseSchedule.init(flowmap_steps), "u"),
        ("ema_flowmap", dynamics_ema, DenoiseSchedule.init(flowmap_steps), "u"),
    ]

    dataset_std = cfg.dataset.dataset_std[0]
    psnr_windows = (1, 3, 8)
    pred_columns: Dict[str, jnp.ndarray] = {}
    rollout_metrics: Dict[str, Dict[str, float]] = {}
    ground_truth_frames: jnp.ndarray | None = None

    for tag, dynamics_model, schedule_config, head in rollout_specs:
        t0 = time.time()
        rng, eval_rng = jax.random.split(rng)

        if use_latent_data:
            pred_frames, gt_decoded_frames, _ = sample_video(
                tokenizer, dynamics_model, frames=None,
                actions=val_actions, horizon=horizon, schedule_config=schedule_config,
                rng=eval_rng, policy=None, task_embedder=None,
                latents=val_data, head=head,
            )
            gt_frames_for_metrics = gt_decoded_frames
        else:
            pred_frames, _, original_frames = sample_video(
                tokenizer, dynamics_model, frames=val_data,
                actions=val_actions, horizon=horizon, schedule_config=schedule_config,
                rng=eval_rng, policy=None, task_embedder=None, head=head,
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

    # Save video and log metrics (only on main process in multi-host)
    if logger is not None:
        num_videos = min(4, ground_truth_frames.shape[0])

        grid_columns = [
            ground_truth_frames,
            pred_columns["online_velocity"],
            pred_columns["ema_velocity"],
            pred_columns["online_flowmap"],
            pred_columns["ema_flowmap"],
        ]
        stacked_frames = jnp.stack(grid_columns)[:, :num_videos]
        videos = rearrange(stacked_frames, 'S B T H W C -> T (B H) (S W) C', B=num_videos)

        tag_dir = _ensure_dir(vis_dir / f"step_{step:06d}")
        mp4_path = tag_dir / "rollouts_grid.mp4"

        video_written = False
        try:
            videos = jax.device_get(videos)
            iio.imwrite(str(mp4_path), videos, fps=20, plugin='pyav', codec='libx264')
            video_written = True
        except Exception as e:
            print(f"[eval] consolidated MP4 write failed: {e}")

        metrics_payload: Dict[str, float] = {"horizon": float(horizon)}
        for tag, _, _, _ in rollout_specs:
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

@nnx.jit(static_argnames=("T", "context_length", "use_latent_data", "head"))
def vis_dynamics_step(
    tokenizer,
    dynamics,
    data: jnp.ndarray,
    actions,
    *,
    master_key: jax.Array,
    step: int,
    T: int,
    context_length: int | None,
    use_latent_data: bool,
    head: str = "v",
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

    sigma = sample_sigma_uniform(key_sigma, (1, T), dtype=latents.dtype)

    z0 = jax.random.normal(key_noise, latents_norm.shape, dtype=latents.dtype)
    z_tilde = (1.0 - (1.0 - 1e-5) * sigma[..., None, None]) * z0 + sigma[..., None, None] * latents_norm

    v_pred, _ = dynamics(
        actions, sigma, z_tilde,
        head=head, context_length=context_length, time_mask=None, task_embeddings=None, deterministic=True,
    )

    # Heads predict velocity; convert to a clean-latent estimate for visualization:
    # x_hat = z_tilde + (1 - sigma) * v.
    z_pred = z_tilde + (1.0 - sigma)[..., None, None] * v_pred

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

    context_length = cfg.dynamics.context_length
    T = int(data.shape[1])

    z_tilde, z_pred, latents_norm, sigma = vis_dynamics_step(
        tokenizer, dynamics, data, actions,
        master_key=master_key,
        step=step,
        T=T,
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

    if logger is not None:
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
    # Use clean-data signal level (σ=1.0)
    sigma = jnp.ones((B, T), dtype=jnp.float32)

    packed_enc_tokens = rearrange(latents, "b t (n p) d -> b t n (p d)", p=dynamics.packing_factor)
    spatial_tokens = dynamics.spatial_proj(packed_enc_tokens)
    action_token   = dynamics.action_encoder(actions=sample_actions, batch_time_shape=(B, T))
    register_tokens = jnp.broadcast_to(
        dynamics.register_tokens.value.astype(dynamics.dtype)[None, None],
        (B, T, dynamics.n_register, dynamics.d_model),
    )
    signal_emb = dynamics.signal_embed(sigma)
    time_token = signal_emb[:, :, None, :]

    tokens = jnp.concatenate([action_token, time_token, spatial_tokens, register_tokens], axis=2)

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
    if logger is not None:
        out_dir = _ensure_dir(vis_dir / f"step_{step:06d}" / log_prefix)
        Image.fromarray(img_array).save(str(out_dir / "attn_vis.png"))
        logger.log_image(step, f"{eval_prefix}attn_vis", img_array, caption=f"step {step}")
