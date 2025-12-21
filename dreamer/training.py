"""
Shared training utilities for flow matching models.

Contains common patterns for:
- Flow matching with bootstrap self-consistency
- Training step construction
- Evaluation logic
"""
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ---------------------------
# Flow matching helpers
# ---------------------------

@partial(jax.jit, static_argnames=("shape_bt", "k_max"))
def sample_tau_for_step(rng, shape_bt, k_max: int, step_idx: jnp.ndarray, *, dtype=jnp.float32):
    """Sample tau values aligned to step_idx grid."""
    B_, T_ = shape_bt
    K = 1 << step_idx
    u = jax.random.uniform(rng, (B_, T_), dtype=dtype)
    j_idx = jnp.floor(u * K.astype(dtype)).astype(jnp.int32)
    tau = j_idx.astype(dtype) / K.astype(dtype)
    tau_idx = j_idx * (k_max // K)
    return tau, tau_idx


@partial(jax.jit, static_argnames=("shape_bt", "k_max"))
def sample_step_excluding_dmin(rng, shape_bt, k_max: int):
    """Sample step indices excluding the finest level (for bootstrap)."""
    B_, T_ = shape_bt
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jax.random.randint(rng, (B_, T_), 0, emax, dtype=jnp.int32)
    d = 1.0 / (1 << step_idx).astype(jnp.float32)
    return d, step_idx


def prepare_flow_matching_batch(
    latents: jnp.ndarray,
    *,
    B_self: int,
    k_max: int,
    master_key: jnp.ndarray,
    step: int,
) -> Dict[str, Any]:
    """
    Prepare all tensors needed for flow matching training.
    
    Returns a dict with:
        - B_emp, B_self: batch split sizes
        - step_idx_full, step_idx_half: discretization levels
        - sigma_full, sigma_emp, sigma_self: signal levels
        - sigma_idx_full, sigma_idx_self, sigma_idx_plus: grid indices
        - z_tilde_full, z_tilde_self: corrupted inputs
        - w_emp, w_self: ramp weights
        - d_half: half-step size for bootstrap
        - emax: max exponent
        - keys: split RNG keys
    """
    step_key = jax.random.fold_in(master_key, step)
    key_sigma_full, key_step_self, key_noise_full, key_rest = jax.random.split(step_key, 4)
    
    B, T = latents.shape[:2]
    B_emp = B - B_self
    emax = jnp.log2(k_max).astype(jnp.int32)
    
    # Step indices
    step_idx_emp = jnp.full((B_emp, T), emax, dtype=jnp.int32)
    d_self, step_idx_self = sample_step_excluding_dmin(key_step_self, (B_self, T), k_max)
    step_idx_full = jnp.concatenate([step_idx_emp, step_idx_self], axis=0)
    
    # Signal levels
    sigma_full, sigma_idx_full = sample_tau_for_step(key_sigma_full, (B, T), k_max, step_idx_full)
    sigma_emp = sigma_full[:B_emp]
    sigma_self = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]
    
    # Corrupt inputs: z_tilde = (1 - sigma) z0 + sigma z1
    z0_full = jax.random.normal(key_noise_full, latents.shape, dtype=latents.dtype)
    z_tilde_full = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[..., None, None] * latents
    z_tilde_self = z_tilde_full[B_emp:]
    
    # Ramp weights
    w_emp = 0.9 * sigma_emp + 0.1
    w_self = 0.9 * sigma_self + 0.1
    
    # Half-step metadata
    d_half = d_self / 2.0
    step_idx_half = step_idx_self + 1
    sigma_plus = sigma_self + d_half
    sigma_idx_plus = sigma_idx_self + (k_max * d_half).astype(jnp.int32)
    
    return {
        "B_emp": B_emp,
        "B_self": B_self,
        "emax": emax,
        "step_idx_full": step_idx_full,
        "step_idx_half": step_idx_half,
        "step_idx_self": step_idx_self,
        "sigma_full": sigma_full,
        "sigma_emp": sigma_emp,
        "sigma_self": sigma_self,
        "sigma_idx_full": sigma_idx_full,
        "sigma_idx_self": sigma_idx_self,
        "sigma_idx_plus": sigma_idx_plus,
        "z_tilde_full": z_tilde_full,
        "z_tilde_self": z_tilde_self,
        "w_emp": w_emp,
        "w_self": w_self,
        "d_half": d_half,
        "latents": latents,
        "key_rest": key_rest,
    }


def compute_flow_bootstrap_loss(
    *,
    dynamics_apply_fn: Callable,
    dynamics_vars: Dict[str, Any],
    actions: jnp.ndarray,
    batch_data: Dict[str, Any],
    step: int,
    bootstrap_start: int,
    drop_key: jnp.ndarray,
    agent_tokens: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Compute flow matching + bootstrap self-consistency loss.
    
    This is the core loss shared between dynamics training and BC/reward training.
    
    Args:
        dynamics_apply_fn: The dynamics.apply function
        dynamics_vars: Dict with "params" and "constants"
        actions: Action sequence (B, T)
        batch_data: Output from prepare_flow_matching_batch
        step: Current training step
        bootstrap_start: Step to start bootstrap loss
        drop_key: Dropout RNG key
        agent_tokens: Optional agent tokens (B, T, n_agent, D) for BC training
    
    Returns:
        (total_loss, aux_dict) where aux_dict contains flow_mse and bootstrap_mse
    """
    B_emp = batch_data["B_emp"]
    B_self = batch_data["B_self"]
    latents = batch_data["latents"]
    B = latents.shape[0]
    
    drop_main, drop_h1, drop_h2 = jax.random.split(drop_key, 3)
    
    # Main forward pass
    if agent_tokens is None:
        z1_hat_full, *_ = dynamics_apply_fn(
            dynamics_vars, actions, batch_data["step_idx_full"],
            batch_data["sigma_idx_full"], batch_data["z_tilde_full"],
            rngs={"dropout": drop_main}, deterministic=False
        )
    else:
        z1_hat_full, *_ = dynamics_apply_fn(
            dynamics_vars, actions, batch_data["step_idx_full"],
            batch_data["sigma_idx_full"], batch_data["z_tilde_full"],
            agent_tokens=agent_tokens,
            rngs={"dropout": drop_main}, deterministic=False
        )
    
    z1_hat_emp = z1_hat_full[:B_emp]
    z1_hat_self = z1_hat_full[B_emp:]
    
    # Flow loss on empirical rows
    flow_per = jnp.mean((z1_hat_emp - latents[:B_emp]) ** 2, axis=(2, 3))
    loss_emp = jnp.mean(flow_per * batch_data["w_emp"])
    
    # Bootstrap self-consistency on self rows
    do_boot = (B_self > 0) & (step >= bootstrap_start)
    
    def _boot_loss():
        if agent_tokens is None:
            z1_hat_half1, *_ = dynamics_apply_fn(
                dynamics_vars, actions[B_emp:], batch_data["step_idx_half"],
                batch_data["sigma_idx_self"], batch_data["z_tilde_self"],
                rngs={"dropout": drop_h1}, deterministic=False
            )
        else:
            z1_hat_half1, *_ = dynamics_apply_fn(
                dynamics_vars, actions[B_emp:], batch_data["step_idx_half"],
                batch_data["sigma_idx_self"], batch_data["z_tilde_self"],
                agent_tokens=agent_tokens[B_emp:],
                rngs={"dropout": drop_h1}, deterministic=False
            )
        
        b_prime = (z1_hat_half1 - batch_data["z_tilde_self"]) / (1.0 - batch_data["sigma_self"])[..., None, None]
        z_prime = batch_data["z_tilde_self"] + b_prime * batch_data["d_half"][..., None, None]
        
        if agent_tokens is None:
            z1_hat_half2, *_ = dynamics_apply_fn(
                dynamics_vars, actions[B_emp:], batch_data["step_idx_half"],
                batch_data["sigma_idx_plus"], z_prime,
                rngs={"dropout": drop_h2}, deterministic=False
            )
        else:
            z1_hat_half2, *_ = dynamics_apply_fn(
                dynamics_vars, actions[B_emp:], batch_data["step_idx_half"],
                batch_data["sigma_idx_plus"], z_prime,
                agent_tokens=agent_tokens[B_emp:],
                rngs={"dropout": drop_h2}, deterministic=False
            )
        
        b_doubleprime = (z1_hat_half2 - z_prime) / (1.0 - batch_data["sigma_plus"])[..., None, None]
        vhat_sigma = (z1_hat_self - batch_data["z_tilde_self"]) / (1.0 - batch_data["sigma_self"])[..., None, None]
        vbar_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)
        boot_per = (1.0 - batch_data["sigma_self"]) ** 2 * jnp.mean((vhat_sigma - vbar_target) ** 2, axis=(2, 3))
        loss_self = jnp.mean(boot_per * batch_data["w_self"])
        return loss_self, jnp.mean(boot_per)
    
    loss_self, boot_mse = jax.lax.cond(
        do_boot,
        _boot_loss,
        lambda: (jnp.array(0.0, dtype=latents.dtype), jnp.array(0.0, dtype=latents.dtype)),
    )
    
    # Combine (row-weighted)
    total_loss = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B
    
    aux = {
        "flow_mse": jnp.mean(flow_per),
        "bootstrap_mse": boot_mse,
    }
    
    return total_loss, aux


# ---------------------------
# Video/Evaluation utilities
# ---------------------------

def build_tiled_video_frames(
    gt_frames: jnp.ndarray,
    floor_frames: jnp.ndarray,
    pred_frames: jnp.ndarray,
    batch_size: int,
) -> list[np.ndarray]:
    """
    Build tiled video frames: (GT | Floor | Pred) per batch item.
    
    Shared between all training scripts for consistent visualization.
    """
    from dreamer.utils import _to_uint8, _stack_wide, _tile_videos
    
    gt_np_all = _to_uint8(gt_frames)
    floor_np_all = _to_uint8(floor_frames)
    pred_np_all = _to_uint8(pred_frames)

    T_total = gt_np_all.shape[1]
    ncols = 1 if batch_size <= 2 else min(8, batch_size)
    grid_frames = []

    for t_idx in range(T_total):
        trip_list = [
            _stack_wide(gt_np_all[b, t_idx], floor_np_all[b, t_idx], pred_np_all[b, t_idx])
            for b in range(batch_size)
        ]
        grid_img = _tile_videos(trip_list, ncols=4, pad_color=0)
        grid_frames.append(grid_img)

    return grid_frames
