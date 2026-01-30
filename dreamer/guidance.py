"""
History Guidance (HG-tf) utilities for video diffusion.

Implements history guidance methods from "History-Guided Video Diffusion" (Song et al., 2025):
- HG-v: Vanilla history guidance (CFG with history masking)
- HG-f: Fractional history guidance (partial noise on history, retains low-freq info)
- HG-t: Temporal history guidance (multiple history lengths)
- HG-tf: Combined temporal + fractional guidance (recommended)

The key insight is that during sampling, we can combine scores conditioned on
different history configurations to improve video quality:

    score_guided = score_uncond + Σ_i ω_i * (score_cond_i - score_uncond)

Where each score_cond_i conditions on different history lengths or noise levels.
"""

from __future__ import annotations
import math
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .configs import HistoryGuidanceConfig
from .models import KVCachesDict, Dynamics
from .actions import Actions
from .utils import normalize_latents


@dataclass
class GuidanceState:
    """Pre-computed cache states for different history configurations.

    During HG-tf sampling, we need multiple forward passes with different
    history configurations. This state container holds the pre-filled KV caches
    for each configuration to avoid redundant computation.

    Attributes:
        caches_full: KV caches with full history (main conditional)
        caches_long: KV caches with long history window (HG-t)
        caches_short: KV caches with short history window (HG-t)
        caches_frac: KV caches with fractionally-noised history (HG-f)
        latents_ctx_normalized: Normalized context latents for reference
        n_spatial: Number of spatial tokens per frame
        d_bottleneck: Latent dimension
    """
    caches_full: KVCachesDict
    caches_long: KVCachesDict | None
    caches_short: KVCachesDict | None
    caches_frac: KVCachesDict | None
    latents_ctx_normalized: jax.Array
    n_spatial: int
    d_bottleneck: int


def create_guidance_state(
    dynamics: Dynamics,
    config: HistoryGuidanceConfig,
    latents_ctx: jax.Array,
    actions_ctx: Actions,
    k_max: int,
    rng: jax.Array,
    n_agent: int = 0,
) -> GuidanceState:
    """
    Create all cache states needed for history guidance.

    This function pre-fills KV caches with different history configurations
    based on the guidance type. Each cache state allows conditioning on
    a different portion of history during sampling.

    Args:
        dynamics: Dynamics model instance
        config: History guidance configuration
        latents_ctx: (B, T_ctx, n_spatial, D) Context latents (already normalized)
        actions_ctx: Context actions
        k_max: Maximum noise resolution
        rng: Random key for noise generation
        n_agent: Number of agent tokens (for cache sizing)

    Returns:
        GuidanceState with all required cache states
    """
    B, T_ctx, n_spatial, D = latents_ctx.shape
    emax = int(math.log2(k_max))

    # Helper to prefill caches with given tau level
    def prefill_caches(
        latents: jax.Array,
        actions: Actions,
        tau_value: float,
        rng_noise: jax.Array,
    ) -> KVCachesDict:
        """Prefill KV caches with optionally noised latents."""
        T = latents.shape[1]

        # Add noise based on tau level (tau=1 is clean, tau=0 is full noise)
        if tau_value < 1.0:
            noise = jax.random.normal(rng_noise, latents.shape, dtype=latents.dtype)
            latents_noised = tau_value * latents + (1 - tau_value) * noise
        else:
            latents_noised = latents

        # Create fresh caches - size for context + reasonable rollout buffer
        window_size = T_ctx + 128  # Buffer for autoregressive rollout
        caches = dynamics.create_static_caches(
            batch_size=B,
            n_latents=n_spatial,
            window_size=window_size,
            n_agent=n_agent,
            dtype=latents.dtype,
        )

        # Use clean signal indices for prefill (high tau = mostly signal)
        step_idx_prefill = jnp.full((B, T), emax, dtype=jnp.int32)
        tau_idx_prefill = jnp.full((B, T), k_max - 1, dtype=jnp.int32)

        # Run dynamics to fill caches
        _, (_, caches_filled) = dynamics(
            actions, step_idx_prefill, tau_idx_prefill, latents_noised,
            caches=caches, deterministic=True
        )

        return caches_filled

    # Split random keys for different cache states
    rng_full, rng_long, rng_short, rng_frac = jax.random.split(rng, 4)

    # 1. Full history caches (clean context, standard tau_ctx=0.9)
    tau_ctx = 0.9  # Standard context noise level
    caches_full = prefill_caches(latents_ctx, actions_ctx, tau_ctx, rng_full)

    # 2. Long history caches (HG-t) - use last N frames
    caches_long = None
    if config.guidance_type in ("temporal", "tf"):
        T_long = min(config.history_long, T_ctx)
        if T_long > 0:
            latents_long = latents_ctx[:, -T_long:]
            actions_long = jax.tree.map(lambda x: x[:, -T_long:], actions_ctx)
            caches_long = prefill_caches(latents_long, actions_long, tau_ctx, rng_long)

    # 3. Short history caches (HG-t) - use last few frames
    caches_short = None
    if config.guidance_type in ("temporal", "tf"):
        T_short = min(config.history_short, T_ctx)
        if T_short > 0:
            latents_short = latents_ctx[:, -T_short:]
            actions_short = jax.tree.map(lambda x: x[:, -T_short:], actions_ctx)
            caches_short = prefill_caches(latents_short, actions_short, tau_ctx, rng_short)

    # 4. Fractionally noised history caches (HG-f)
    # Lower tau_H_frac = more noise = keeps only low-frequency info
    caches_frac = None
    if config.guidance_type in ("fractional", "tf"):
        tau_frac = config.tau_H_frac
        caches_frac = prefill_caches(latents_ctx, actions_ctx, tau_frac, rng_frac)

    return GuidanceState(
        caches_full=caches_full,
        caches_long=caches_long,
        caches_short=caches_short,
        caches_frac=caches_frac,
        latents_ctx_normalized=latents_ctx,
        n_spatial=n_spatial,
        d_bottleneck=D,
    )


def compute_guided_prediction(
    dynamics: Dynamics,
    config: HistoryGuidanceConfig,
    guidance_state: GuidanceState,
    noisy_latent: jax.Array,
    action: Actions,
    tau_idx: int,
    step_idx: int,
    task_embedding: jax.Array | None = None,
) -> jax.Array:
    """
    Compute history-guided clean latent prediction.

    Implements the HG-tf formula:
        x1_guided = x1_uncond + Σ_i ω_i * (x1_cond_i - x1_uncond)

    The unconditional score (x1_uncond) is obtained by running the forward
    pass without any history caches. Each conditional score is obtained
    using the corresponding pre-filled cache from GuidanceState.

    Args:
        dynamics: Dynamics model instance
        config: History guidance configuration
        guidance_state: Pre-computed cache states
        noisy_latent: (B, 1, n_spatial, D) Current noisy latent to denoise
        action: Current action (B, ...)
        tau_idx: Current signal level index
        step_idx: Current step index
        task_embedding: Optional task embedding (B, 1, n_agent, d_model)

    Returns:
        x1_guided: (B, 1, n_spatial, D) Guided clean latent prediction
    """
    B = noisy_latent.shape[0]

    # Prepare indices for dynamics forward pass
    step_indices = jnp.full((B, 1), step_idx, dtype=jnp.int32)
    tau_indices = jnp.full((B, 1), tau_idx, dtype=jnp.int32)

    # Expand action to have time dimension
    action_expanded = jax.tree.map(lambda x: x[:, None, ...], action)

    # Helper for dynamics forward pass
    def forward_pass(caches: KVCachesDict | None) -> jax.Array:
        """Run dynamics and extract the prediction for the last timestep."""
        x1_pred, _ = dynamics(
            action_expanded, step_indices, tau_indices, noisy_latent,
            task_embeddings=task_embedding, deterministic=True, caches=caches
        )
        return x1_pred[:, -1:, :, :]  # (B, 1, n_spatial, D)

    # 1. Unconditional score (no history - run without caches)
    # This simulates fully masked history
    x1_uncond = forward_pass(caches=None)

    # 2. Conditional score with full history
    x1_cond_full = forward_pass(caches=guidance_state.caches_full)

    # 3. Compute guided prediction based on guidance type
    if config.guidance_type == "vanilla":
        # HG-v: Simple CFG with history
        # x1_guided = x1_uncond + omega * (x1_cond - x1_uncond)
        x1_guided = x1_uncond + config.omega * (x1_cond_full - x1_uncond)

    elif config.guidance_type == "fractional":
        # HG-f: Guide using fractionally noised history
        # Uses caches_frac which has partial noise on context
        x1_cond_frac = forward_pass(caches=guidance_state.caches_frac)
        x1_guided = x1_uncond + config.omega * (x1_cond_frac - x1_uncond)

    elif config.guidance_type == "temporal":
        # HG-t: Combine scores from different history lengths
        x1_cond_long = forward_pass(caches=guidance_state.caches_long)
        x1_cond_short = forward_pass(caches=guidance_state.caches_short)

        x1_guided = (x1_uncond +
                     config.omega_long * (x1_cond_long - x1_uncond) +
                     config.omega_short * (x1_cond_short - x1_uncond))

    elif config.guidance_type == "tf":
        # HG-tf: Full combination of temporal + fractional
        x1_cond_long = forward_pass(caches=guidance_state.caches_long)
        x1_cond_short = forward_pass(caches=guidance_state.caches_short)
        x1_cond_frac = forward_pass(caches=guidance_state.caches_frac)

        x1_guided = (x1_uncond +
                     config.omega_long * (x1_cond_long - x1_uncond) +
                     config.omega_short * (x1_cond_short - x1_uncond) +
                     config.omega_frac * (x1_cond_frac - x1_uncond))
    else:
        raise ValueError(f"Unknown guidance type: {config.guidance_type}")

    return x1_guided


def update_guidance_caches(
    dynamics: Dynamics,
    guidance_state: GuidanceState,
    generated_latent: jax.Array,
    action: Actions,
    tau_idx_ctx: int,
    step_idx_ctx: int,
) -> GuidanceState:
    """
    Update all guidance caches with a newly generated latent.

    After generating a new latent, we need to update the KV caches so that
    subsequent frames can condition on it. This function updates the main
    cache (caches_full) with the new latent.

    Note: For simplicity, we only update caches_full during rollout.
    The other caches (long, short, frac) maintain their initial context.
    This is a simplification - a more sophisticated implementation could
    update all caches based on their respective strategies.

    Args:
        dynamics: Dynamics model instance
        guidance_state: Current guidance state
        generated_latent: (B, 1, n_spatial, D) Newly generated latent
        action: Action that led to this latent
        tau_idx_ctx: Tau index for caching (typically tau_idx_ctx)
        step_idx_ctx: Step index for caching

    Returns:
        Updated GuidanceState with new cache values
    """
    B = generated_latent.shape[0]

    # Prepare indices
    step_indices = jnp.full((B, 1), step_idx_ctx, dtype=jnp.int32)
    tau_indices = jnp.full((B, 1), tau_idx_ctx, dtype=jnp.int32)
    action_expanded = jax.tree.map(lambda x: x[:, None, ...], action)

    # Update main cache by running forward pass
    _, (_, caches_updated) = dynamics(
        action_expanded, step_indices, tau_indices, generated_latent,
        caches=guidance_state.caches_full, deterministic=True
    )

    # Use .replace() for flax.struct.dataclass
    return guidance_state.replace(caches_full=caches_updated)
