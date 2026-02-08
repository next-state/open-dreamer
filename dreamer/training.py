"""Reusable training components for dynamics and imagination training."""
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
from dreamer.actions import Actions
from dreamer.sampler import sample_video
from dreamer.utils import _ensure_dir, apply_border


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
    *,
    dtype=jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample tau on the finest training grid [0, 1) with k_max bins."""
    B_, T_ = shape_bt
    u = jax.random.uniform(rng, (B_, T_), dtype=dtype)
    tau_idx = jnp.floor(u * k_max).astype(jnp.int32)
    tau = tau_idx.astype(dtype) / k_max
    return tau, tau_idx


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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Flow matching loss in x-space (direct prediction of clean latents)."""
    mse = jnp.mean((z_pred - z_target) ** 2)
    return mse, mse


# ---------------------------
# Dynamics flow step logic
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
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """
    Compute flow-matching losses for a batch.

    The function name is kept for compatibility with training scripts.
    """
    del B_self

    B, T, _, _ = latents.shape
    emax = jnp.log2(k_max).astype(jnp.int32)

    # Sample tau and noise at full resolution.
    key_sigma, key_noise, key_dropout = jax.random.split(rng, 3)
    step_idx = jnp.full((B, T), emax, dtype=jnp.int32)
    sigma, sigma_idx = sample_tau_for_step(key_sigma, (B, T), k_max, dtype=latents.dtype)
    z0 = jax.random.normal(key_noise, latents.shape, dtype=latents.dtype)
    z_tilde = (1.0 - sigma[..., None, None]) * z0 + sigma[..., None, None] * latents

    # One dynamics pass, one loss.
    rngs = nnx.Rngs(dropout=key_dropout)
    z_pred, (h_states, _) = dynamics_model(
        actions,
        step_idx,
        sigma_idx,
        z_tilde,
        context_length=context_length,
        time_mask=time_mask,
        task_embeddings=task_embeddings,
        deterministic=False,
        rngs=rngs,
    )

    loss_flow, flow_mse = compute_flow_loss(z_pred, latents)
    loss_boot = jnp.array(0.0, dtype=latents.dtype)

    losses = {"total": loss_flow, "flow": loss_flow, "bootstrap": loss_boot}
    aux = {"flow_mse": flow_mse, "bootstrap_mse": loss_boot, "h_states": h_states}
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
    val_data: jnp.ndarray,
    val_actions: Actions,
    use_latent_data: bool,
    vis_dir: Path,
    rng: jax.Array,
    logger,
    policy: PolicyHeadMTP | None = None,
    task_embedder: TaskEmbedder | None = None,
    omega: jax.Array | float = 0.0,
    alpha: jax.Array | float = 0.7,
):
    """Run periodic evaluation: sample videos, compute metrics, and save visualization."""
    del omega, alpha

    k_max = dynamics.cfg.k_max
    schedule_diffusion = DenoiseSchedule.init(k_max, k_max)
    tag = "diffusion"

    t0 = time.time()
    T = val_data.shape[1]
    assert T > 5, f"Sequence length {T} must be > 5"
    ctx_length = 4
    horizon = T - ctx_length

    if use_latent_data:
        pred_frames, gt_decoded_frames, _ = sample_video(
            tokenizer,
            dynamics,
            frames=None,
            actions=val_actions,
            horizon=horizon,
            schedule_config=schedule_diffusion,
            rng=rng,
            policy=policy,
            task_embedder=task_embedder,
            latents=val_data,
        )
        gt_frames_for_metrics = gt_decoded_frames
    else:
        pred_frames, gt_decoded_frames, original_frames = sample_video(
            tokenizer,
            dynamics,
            frames=val_data,
            actions=val_actions,
            horizon=horizon,
            schedule_config=schedule_diffusion,
            rng=rng,
            policy=policy,
            task_embedder=task_embedder,
        )
        gt_frames_for_metrics = original_frames

    dt = time.time() - t0
    normalized_pred = tokenizer.pixel_normalizer.normalize(pred_frames[:, -horizon:] / 255.0)
    normalized_gt = tokenizer.pixel_normalizer.normalize(gt_frames_for_metrics[:, -horizon:] / 255.0)
    mse = float(jnp.mean((normalized_pred - normalized_gt) ** 2))
    psnr = float(compute_psnr(pred_frames[:, -horizon:] / 255, gt_frames_for_metrics[:, -horizon:] / 255))

    print(f"[eval:{tag}] step={step:06d} | horizon={horizon} | MSE={mse:.6g} | PSNR={psnr:.2f} dB | {dt:.2f}s")

    num_videos = min(4, pred_frames.shape[0])
    pred_frames = pred_frames.at[:, :ctx_length].set(apply_border(pred_frames[:, :ctx_length]))

    if use_latent_data:
        frames_list = [gt_decoded_frames, pred_frames]
    else:
        frames_list = [gt_decoded_frames, original_frames, pred_frames]
    stacked_frames = jnp.stack(frames_list)[:, :num_videos]
    videos = rearrange(stacked_frames, "S B T H W C -> T (B H) (S W) C", B=num_videos)

    tag_dir = _ensure_dir(vis_dir / f"step_{step:06d}")
    mp4_path = tag_dir / f"{tag}_grid.mp4"

    try:
        videos = jax.device_get(videos)
        iio.imwrite(str(mp4_path), videos, fps=5, plugin="pyav", codec="libx264")
    except Exception as e:
        print(f"[eval:{tag}] MP4 write failed: {e}")

    logger.log_metrics(
        step,
        {
            f"{tag}/mse": mse,
            f"{tag}/psnr": psnr,
            f"{tag}/horizon": horizon,
            f"{tag}/eval_time": dt,
        },
        prefix="eval/",
    )

    if videos is not None:
        logger.log_video(step, f"eval/{tag}/video", mp4_path)
