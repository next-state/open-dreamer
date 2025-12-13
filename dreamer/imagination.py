# imagination.py - JIT-friendly image / video sampling for the dynamics model during RL.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from einops import reduce
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import time

from dreamer.models import Dynamics, TaskEmbedder
from dreamer.utils import (
    temporal_patchify,
    temporal_unpatchify,
    pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck,
)
def _assert_power_of_two(k: int):
    if k < 1 or (k & (k - 1)) != 0:
        raise ValueError(f"k_max must be a positive power of two, got {k}")


def _step_idx_from_d(d: float, k_max: int) -> int:
    """
    Map a step size d (which must be 1/(power of two)) to an integer step index e
    compatible with the dynamics' k_max grid.
    """
    K = round(1.0 / float(d))
    if abs(1.0 / K - d) > 1e-8:
        raise ValueError(f"d={d} is not an exact 1/(power of two)")
    e = int(round(np.log2(K)))
    emax = int(round(np.log2(k_max)))
    if e > emax:
        raise ValueError(
            f"step bin e={e} (d={d}) is coarser than allowed emax={emax} (k_max={k_max})"
        )
    return e


# ---------------------------
# Config & schedule
# ---------------------------

@dataclass(frozen=True)
class ImaginationConfig:
    """
    Static configuration for imagination rollouts.
    All fields are intended to be fixed at construction time so that
    JAX can treat shapes and control-flow lengths as static.
    """

    k_max: int
    horizon: int
    context_length: int
    n_spatial: int
    d: float  # denoising step size (e.g., 1/k_max or 1/4, etc.)
    start_mode: str = "pure"  # "pure" or "fixed" (no per-call randomness)
    tau0_fixed: float = 0.0   # used iff start_mode == "fixed"
    tau_ctx: float = 0.9      # noise level on the context for autoregressive rollout stabilization
                              # 0.1 in the paper, but it's a typo, they mean 1 - 0.1 = 0.9


class DenoiseSchedule(NamedTuple):
    """
    Precomputed, JAX-friendly schedule for the τ-ladder.

    tau_seq:        (S+1,) τ_0..τ_S (monotone, τ_0 ∈ [0,1), τ_S=1.0)
    alpha_seq:      (S,)   per-step mixing coefficients α_s
    signal_idx_seq: (S+1,) integer signal indices for each τ_s
    step_idx:       scalar integer step index e (same for all steps here)
    k_max:          scalar integer, copied from config for convenience
    """

    tau_seq: jnp.ndarray
    alpha_seq: jnp.ndarray
    signal_idx_seq: jnp.ndarray
    step_idx: int
    k_max: int


def _build_static_schedule(cfg: ImaginationConfig) -> DenoiseSchedule:
    """
    Host-side helper that constructs a fixed τ-ladder schedule.
    This is run once at sampler construction time (not inside JIT).
    """
    _assert_power_of_two(cfg.k_max)

    if cfg.start_mode not in ("pure", "fixed"):
        raise ValueError(
            f"ImaginationConfig.start_mode must be 'pure' or 'fixed' for static schedules, "
            f"got {cfg.start_mode!r}"
        )

    # Validate and convert step size to step index e.
    d = float(cfg.d)
    e = _step_idx_from_d(d, cfg.k_max)

    # Determine starting τ (aligned to the step grid).
    if cfg.start_mode == "pure":
        tau0 = 0.0
    else:  # "fixed"
        # We keep this simple and snap tau0_fixed to the {0, d, 2d, ...} grid.
        tau0_raw = float(np.clip(cfg.tau0_fixed, 0.0, 1.0))
        tau0 = float(np.clip(np.floor(tau0_raw / d) * d, 0.0, 1.0))

    # Build τ sequence τ_0..τ_S and corresponding α_s.
    # S is chosen so that τ_S ≈ 1.0.
    S_float = (1.0 - tau0) / d
    S = int(round(S_float))
    if S <= 0:
        raise ValueError(f"Invalid ladder length S={S} derived from tau0={tau0}, d={d}")

    tau_seq_np = tau0 + d * np.arange(S + 1, dtype=np.float32)
    tau_seq_np = np.clip(tau_seq_np, 0.0, 1.0)

    alpha_seq_np = np.zeros(S, dtype=np.float32)
    for s in range(S):
        tau_prev = float(tau_seq_np[s])
        tau_curr = float(tau_seq_np[s + 1])
        denom = max(1.0 - tau_prev, 1e-8)
        alpha_seq_np[s] = (tau_curr - tau_prev) / denom

    # Integer signal indices for each τ (same rule as _signal_idx_from_tau, but host-side).
    signal_idx_np = (tau_seq_np * float(cfg.k_max)).astype(np.int32)
    signal_idx_np = np.clip(signal_idx_np, 0, cfg.k_max - 1)

    schedule = DenoiseSchedule(
        tau_seq=jnp.asarray(tau_seq_np, dtype=jnp.float32),
        alpha_seq=jnp.asarray(alpha_seq_np, dtype=jnp.float32),
        signal_idx_seq=jnp.asarray(signal_idx_np, dtype=jnp.int32),
        step_idx=int(e),
        k_max=int(cfg.k_max),
    )
    return schedule


# ---------------------------
# Single-step τ-ladder denoiser
# ---------------------------

def denoise_single_latent_static(
    *,
    dynamics: Dynamics,
    dyn_vars: Dict[str, Any],
    schedule: DenoiseSchedule,
    actions_ctx: jnp.ndarray,                 # (B, T_ctx)
    action_curr: jnp.ndarray,                 # (B, 1)
    z_ctx_t: jnp.ndarray,                     # (B, T_ctx, n_spatial, D_s)
    z_noise_t: jnp.ndarray,                   # (B, 1, n_spatial, D_s)
    agent_tokens: jnp.ndarray | None = None,  # (B, T_ctx+1, n_agent, d_model)
    caches: Dict[int, Any] | None = None,
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
        z_noise_t: (B, 1, n_spatial, D_s) initial noisy latent at τ_0
        agent_tokens: optional agent tokens (B, T_ctx+1, n_agent, d_model)
        caches: KV cache for context frames (from previous finalized frames)
    """
    B, T_ctx, n_spatial, D_s = z_ctx_t.shape
    S = schedule.alpha_seq.shape[0]

    def refinement_step(z_t, s):
        alpha = schedule.alpha_seq[s]
        signal_idx_scalar = schedule.signal_idx_seq[s + 1]

        # Input preparation
        if caches is not None:
            # Denoise only the current frame
            z_input = z_t
            actions_input = action_curr

            step_idx_input = jnp.full((B, 1), schedule.step_idx, dtype=jnp.int32)
            signal_idx_input = jnp.full((B, 1), signal_idx_scalar, dtype=jnp.int32)

            if agent_tokens is not None:
                agent_tokens_input = agent_tokens[:, -1:, :, :]
            else:
                agent_tokens_input = None
        
        else:
            # Denoise [context, current_frame]
            z_input = jnp.concatenate([z_ctx_t, z_t], axis=1)  # (B, T_ctx+1, n_spatial, D_s)
            actions_input = jnp.concatenate([actions_ctx, action_curr], axis=1)  # (B, T_ctx+1)

            step_idx_input = jnp.full((B, T_ctx + 1), schedule.step_idx, dtype=jnp.int32)
            signal_idx_ctx = jnp.full((B, T_ctx), (0.9 * schedule.k_max).astype(jnp.int32), dtype=jnp.int32)
            signal_idx_curr = jnp.full((B, 1), signal_idx_scalar, dtype=jnp.int32)
            signal_idx_input = jnp.concatenate([signal_idx_ctx, signal_idx_curr], axis=1)

            agent_tokens_input = agent_tokens

        # Dynamics call
        z_clean_pred_seq, h_seq, caches_updated = dynamics.apply(
            dyn_vars,
            actions_input,
            step_idx_input,
            signal_idx_input,
            z_input,
            agent_tokens=agent_tokens_input,
            deterministic=True,
            caches=caches,
        )

        z_clean_pred = z_clean_pred_seq[:, -1:, :, :]  # (B, 1, n_spatial, D_s)
        h_last = h_seq[:, -1, :, :]  # (B, n_agent, d_model)

        # Per-step mixing toward clean latent
        z_t_new = (1.0 - alpha) * z_t + alpha * z_clean_pred

        return z_t_new, (h_last, caches_updated)

    # Run τ-ladder with JAX control flow using scan to keep carry/output structure consistent.
    z_t_final, (h_history, caches_history) = jax.lax.scan(
        refinement_step,
        z_noise_t,
        jnp.arange(S),
    )

    h_last = h_history[-1]  # (B, n_agent, d_model)
    caches_last = None
    if caches_history is not None:
        caches_last = jax.tree.map(lambda x: x[-1], caches_history)

    return z_t_final, h_last, caches_last  # (B, 1, n_spatial, D_s), (B, n_agent, d_model), Dict[int, KVCache] | None


# JIT-compiled variant (optional convenience wrapper)
denoise_single_latent_static_jit = jax.jit(
    denoise_single_latent_static,
    static_argnames=("dynamics",),
)


# ---------------------------
# Imagination rollouts (latent space)
# ---------------------------

PolicyFn = Callable[
    [jnp.ndarray, jax.Array, Any],
    Tuple[jnp.ndarray, jnp.ndarray, Any],
]
"""
Generic policy callable.

Args:
    h:      (B, d_model) hidden state
    rng:    PRNGKey
    state:  arbitrary policy state PyTree

Returns:
    actions: (B,) integer actions
    logits:  (B, A) unnormalized logits for actions
    new_state: updated policy state
"""


def make_gt_action_policy_fn(
    gt_actions: jnp.ndarray,  # (B, horizon)
    action_dim: int,
) -> Tuple[PolicyFn, dict]:
    """
    Build a policy_fn/state pair that feeds ground-truth actions into rollout_latents.

    gt_actions[:, t] is used as the action at imagination step t.
    """
    init_state = {
        "gt_actions": gt_actions,           # (B, horizon)
        "t": jnp.int32(0),                  # current imagination step
    }

    def policy_fn(h: jnp.ndarray, rng: jax.Array, state: dict):
        # Ignore h and rng; just replay GT actions.
        gt_seq = state["gt_actions"]        # (B, horizon)
        t = state["t"]
        actions_t = gt_seq[:, t]            # (B,)

        # Dummy logits for interface compatibility.
        logits = jnp.zeros(
            (actions_t.shape[0], action_dim),
            dtype=h.dtype,
        )

        new_state = {
            "gt_actions": gt_seq,
            "t": t + jnp.int32(1),
        }
        return actions_t, logits, new_state

    return policy_fn, init_state


def imagine_rollouts(
    *,
    dynamics: Dynamics,
    task_embedder: TaskEmbedder,
    dyn_vars: Dict[str, Any],
    task_vars: Dict[str, Any],
    schedule: DenoiseSchedule,
    z_ctx: jnp.ndarray,        # (B, context_length, n_spatial, d_spatial)
    actions_ctx: jnp.ndarray,  # (B, context_length)
    task_ids: jnp.ndarray,         # (B,)
    horizon: int,
    policy_fn: PolicyFn,
    policy_state: Any,
    rng_key: jax.Array,
    use_kv_caching: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Core JAX-friendly imagination rollout in latent space with KV caching.

    This function:
      - Computes initial hidden state from context using dynamics.
      - Uses a generic policy_fn(h, rng, state) to produce actions & logits.
      - Uses denoise_single_latent_static to predict next latents.
      - Rolls context autoregressively.
      - Each new frame: denoise with cached past, then update cache

    Returns:
        imagined_latents: (B, horizon + 1, n_spatial, d_spatial)
            index 0 is last context state; 1..horizon are imagined future states.
        imagined_actions: (B, horizon)
        imagined_hidden:  (B, horizon + 1, d_model)
        policy_logits:    (B, horizon, A)
    """
    B, context_length, n_spatial, D_s = z_ctx.shape

    # Pre-compute agent tokens for entire rollout.
    agent_tokens_full = task_embedder.apply(
        task_vars, task_ids, B, context_length + horizon
    )  # (B, context_length + horizon, n_agent, d_model)

    # Prepare step and signal indices for initial context dynamics call (τ=1.0).
    e_jax = jnp.int32(schedule.step_idx)
    step_idx_ctx = jnp.full((B, context_length), e_jax, dtype=jnp.int32)
    signal_idx_ctx = jnp.full((B, context_length), schedule.k_max - 1, dtype=jnp.int32)

    # [PREFILL]: initial hidden state & KV cache from context
    MAX_WINDOW = 4096  # must accomodate all tokens processed by time layers
    initial_caches = (dynamics.create_static_caches(batch_size=B, window_size=MAX_WINDOW, dtype=z_ctx.dtype) 
                      if use_kv_caching else None)
    _, h_ctx_init, filled_caches = dynamics.apply(
        dyn_vars,
        actions_ctx,
        step_idx_ctx,
        signal_idx_ctx,
        z_ctx,
        agent_tokens=agent_tokens_full[:, :context_length, :, :],
        deterministic=True,
        caches=initial_caches,
    )  # (B, context_length, n_agent, d_model)

    h_pooled_init = reduce(
        h_ctx_init, "b t n_agent d_model -> b d_model", "mean"
    )  # (B, d_model)
    h = h_pooled_init  # last context state hidden

    # Starting latent is last context latent.
    z_start = z_ctx[:, -1, :, :]  # (B, n_spatial, D_s)

    # Stabilization noise for autoregressive rollout
    z_ctx_noise = jnp.zeros_like(z_ctx) # Zero for ground truth initial context

    # [DECODE]
    def scan_step(carry, t):
        z_ctx_t, actions_ctx_t, z_ctx_noise_t, h_t, policy_state_t, rng_t, caches_t = carry

        rng_t, policy_key, z_noise_t_key = jax.random.split(rng_t, 3)

        # Policy: actions + logits
        actions_t, logits_t, policy_state_next = policy_fn(h_t, policy_key, policy_state_t)
        actions_t = actions_t.astype(jnp.int32)
        action_curr = actions_t[:, None]  # (B, 1)

        # τ-ladder noise for prediction latent
        z_noise_t = jax.random.normal(
            z_noise_t_key, (B, 1, n_spatial, D_s), dtype=z_ctx_t.dtype
        )

        # Diffusion forcing stabilization noise for autoregressive rollout
        # FIXME: use config.tau_ctx
        z_ctx_noised_t = 0.9 * z_ctx_t + (1.0 - 0.9) * z_ctx_noise_t

        # Slice agent tokens for [t : t + context_length + 1]
        # (context + all imagined steps so far, plus one for the new step).
        agent_tokens_seq = jax.lax.dynamic_slice(
            agent_tokens_full,
            (0, t, 0, 0),
            (B, context_length + 1, agent_tokens_full.shape[2], agent_tokens_full.shape[3]),
        )

        z_clean_pred, h_last, caches_next = denoise_single_latent_static(
            dynamics=dynamics,
            dyn_vars=dyn_vars,
            schedule=schedule,
            actions_ctx=actions_ctx_t,
            action_curr=action_curr,
            z_ctx_t=z_ctx_noised_t,
            z_noise_t=z_noise_t,
            agent_tokens=agent_tokens_seq,
            caches=caches_t,
        )

        # h_last: (B, n_agent, d_model) → pool over agents
        h_next = reduce(h_last, "b n_agent d_model -> b d_model", "mean")

        # Autoregressive context update (teacher-free)
        z_ctx_next = jnp.concatenate(
            [z_ctx_t, z_clean_pred], axis=1
        )[:, -context_length:, :, :]
        actions_ctx_next = jnp.concatenate(
            [actions_ctx_t, action_curr], axis=1
        )[:, -context_length:]

        # Stabilization noise for autoregressive rollout
        z_ctx_noise_next = jnp.concatenate(
            [z_ctx_noise_t, z_noise_t], axis=1
        )[:, -context_length:, :, :]

        carry_next = (z_ctx_next, actions_ctx_next, z_ctx_noise_next, h_next, policy_state_next, rng_t, caches_next)

        return carry_next, (z_clean_pred[:, 0, :, :], actions_t, h_next, logits_t)

    init_carry = (z_ctx, actions_ctx, z_ctx_noise, h, policy_state, rng_key, filled_caches)
    _, outputs = jax.lax.scan(
        scan_step,
        init_carry,
        jnp.arange(horizon),
    )

    zs_pred, actions_seq, h_seq, logits_seq = outputs
    # zs_pred:    (horizon, B, n_spatial, D_s)
    # actions_seq:(horizon, B)
    # h_seq:      (horizon, B, d_model)
    # logits_seq: (horizon, B, A)

    imagined_latents = jnp.concatenate(
        [z_start[:, None, :, :], jnp.transpose(zs_pred, (1, 0, 2, 3))],
        axis=1,
    )
    imagined_actions = jnp.transpose(actions_seq, (1, 0))  # (B, horizon)
    imagined_hidden = jnp.concatenate(
        [h[None, :, :], h_seq], axis=0
    )  # (horizon+1, B, d_model)
    imagined_hidden = jnp.transpose(imagined_hidden, (1, 0, 2))  # (B, horizon+1, d_model)
    policy_logits = jnp.transpose(logits_seq, (1, 0, 2))  # (B, horizon, A)

    return imagined_latents, imagined_actions, imagined_hidden, policy_logits


imagine_rollouts_jit = jax.jit(
    imagine_rollouts,
    static_argnames=("dynamics", "task_embedder", "policy_fn", "horizon", "use_kv_caching"),
)


# ---------------------------
# OO wrapper
# ---------------------------

class ImaginationSampler:
    """
    Convenience wrapper that owns a static ImaginationConfig and schedule.

    The underlying rollout logic is still fully functional when called directly
    without JIT, but you can also use the `*_jit` helpers for performance.
    """

    def __init__(self, config: ImaginationConfig):
        self.config = config
        self.schedule = _build_static_schedule(config)

    def rollout_latents(
        self,
        *,
        dynamics: Dynamics,
        task_embedder: TaskEmbedder,
        dyn_vars: Dict[str, Any],
        task_vars: Dict[str, Any],
        z_ctx: jnp.ndarray,
        actions_ctx: jnp.ndarray,
        task_ids: jnp.ndarray,
        policy_fn: PolicyFn,
        policy_state: Any,
        rng_key: jax.Array,
        use_kv_caching: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return imagine_rollouts(
            dynamics=dynamics,
            task_embedder=task_embedder,
            dyn_vars=dyn_vars,
            task_vars=task_vars,
            schedule=self.schedule,
            z_ctx=z_ctx,
            actions_ctx=actions_ctx,
            task_ids=task_ids,
            horizon=self.config.horizon,
            policy_fn=policy_fn,
            policy_state=policy_state,
            rng_key=rng_key,
            use_kv_caching=use_kv_caching,
        )

    def rollout_latents_jit(
        self,
        *,
        dynamics: Dynamics,
        task_embedder: TaskEmbedder,
        dyn_vars: Dict[str, Any],
        task_vars: Dict[str, Any],
        z_ctx: jnp.ndarray,
        actions_ctx: jnp.ndarray,
        task_ids: jnp.ndarray,
        policy_fn: PolicyFn,
        policy_state: Any,
        rng_key: jax.Array,
        use_kv_caching: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Use the module-level jitted core, which treats dynamics/task_embedder/policy_fn
        # as static arguments.
        return imagine_rollouts_jit(
            dynamics=dynamics,
            task_embedder=task_embedder,
            dyn_vars=dyn_vars,
            task_vars=task_vars,
            schedule=self.schedule,
            z_ctx=z_ctx,
            actions_ctx=actions_ctx,
            task_ids=task_ids,
            horizon=self.config.horizon,
            policy_fn=policy_fn,
            policy_state=policy_state,
            rng_key=rng_key,
            use_kv_caching=use_kv_caching,
        )


# ---------------------------
# Self-contained tests / examples
# ---------------------------

def _compute_mse_psnr(pred: jnp.ndarray, target: jnp.ndarray) -> Tuple[float, float]:
    """
    Compute MSE and PSNR assuming inputs are in [0,1].
    """
    mse = float(jnp.mean((pred - target) ** 2))
    if mse <= 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))
    return mse, psnr


# ------- simple viz helpers (mirroring eval_bc_rew_heads) -------

def _to_uint8(img_f32: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.clip(np.asarray(img_f32) * 255.0, 0, 255),
        dtype=np.uint8,
    )


def _stack_wide(*imgs_hwC: np.ndarray) -> np.ndarray:
    """Horizontally stack same-size images."""
    return np.concatenate(imgs_hwC, axis=1)


def _tile_videos(
    trip_list_hwC: list[np.ndarray],
    *,
    ncols: int = 2,
    pad_color: int = 0,
) -> np.ndarray:
    """
    Tile a list of HxWxC images into a grid (like eval_bc_rew_heads._tile_videos).
    """
    H, W, C = trip_list_hwC[0].shape
    B = len(trip_list_hwC)
    nrows = int(np.ceil(B / ncols))
    total = nrows * ncols
    if total > B:
        blank = np.full((H, W, C), pad_color, dtype=trip_list_hwC[0].dtype)
        trip_list_hwC = trip_list_hwC + [blank] * (total - B)
    rows = []
    idx = 0
    for _ in range(nrows):
        row_imgs = trip_list_hwC[idx: idx + ncols]
        idx += ncols
        rows.append(np.concatenate(row_imgs, axis=1))
    grid = np.concatenate(rows, axis=0)
    return grid


def _save_side_by_side_strip(
    fig_path: Path,
    gt_frames_b_t_hwc: np.ndarray,
    pred_frames_b_t_hwc: np.ndarray,
    title: str,
    b_index: int = 0,
):
    """
    Save a 1D strip for a single example: GT row over Pred row.
    Assumes shapes (B, T, H, W, C).
    """
    gt = _to_uint8(gt_frames_b_t_hwc)
    pred = _to_uint8(pred_frames_b_t_hwc)
    B, T, H, W, C = gt.shape
    b = int(np.clip(b_index, 0, B - 1))

    fig, axes = plt.subplots(2, T, figsize=(T * 2.2, 4.0), constrained_layout=True)
    # Normalize axes shape so we can always index as axes[row, col]
    axes_arr = np.asarray(axes)
    if axes_arr.ndim == 1:  # e.g., T == 1 → shape (2,)
        axes_arr = axes_arr[:, None]
    axes = axes_arr
    fig.suptitle(title, fontsize=12)

    # row 0: GT
    for t in range(T):
        ax = axes[0, t]
        ax.imshow(gt[b, t])
        ax.axis("off")
        ax.set_title(f"GT t+{t}", fontsize=8)

    # row 1: Pred
    for t in range(T):
        ax = axes[1, t]
        ax.imshow(pred[b, t])
        ax.axis("off")
        ax.set_title(f"Pred t+{t}", fontsize=8)

    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def _make_real_world_models_and_batch():
    """
    Utility for tests: load real checkpoints and a single batch of data.

    Uses the same checkpoint paths and initialization logic as train_policy.py.
    """
    from dreamer.data import make_iterator
    from scripts.train_policy import RLConfig, initialize_models

    # This mirrors the __main__ config in train_policy.py but you can
    # adjust the batch/time dimensions here if desired.
    cfg = RLConfig(
        run_name="jit_sampler_test",
        bc_rew_ckpt="logs/bc_rew/checkpoints",
        use_wandb=False,
        wandb_project="tiny_dreamer_4",
    )

    next_batch = make_iterator(
        cfg.dataset.B,
        cfg.dataset.T,
        cfg.dataset.H,
        cfg.dataset.W,
        cfg.dataset.C,
        pixels_per_step=cfg.dataset.pixels_per_step,
        size_min=cfg.dataset.size_min,
        size_max=cfg.dataset.size_max,
        hold_min=cfg.dataset.hold_min,
        hold_max=cfg.dataset.hold_max,
        fg_min_color=0 if cfg.dataset.diversify_data else 128,
        fg_max_color=255 if cfg.dataset.diversify_data else 128,
        bg_min_color=0 if cfg.dataset.diversify_data else 255,
        bg_max_color=255 if cfg.dataset.diversify_data else 255,
    )

    rng = jax.random.PRNGKey(0)
    _, (frames, actions, rewards) = next_batch(rng)

    train_state = initialize_models(cfg, frames, actions)

    return cfg, train_state, frames, actions, rewards


def test_single_step_denoise_real_ckpt(use_jit: bool = False):
    """
    Single-step denoising test using real checkpoints and decoder visualization.
    """
    cfg, train_state, frames, actions, _ = _make_real_world_models_and_batch()

    patch = cfg.patch
    n_spatial = cfg.enc_n_latents // cfg.packing_factor

    # Encode full video once.
    patches = temporal_patchify(frames, patch)
    z_btLd, _ = train_state.encoder.apply(
        train_state.enc_vars,
        patches,
        rngs={"mae": train_state.mae_eval_key},
        deterministic=True,
    )
    z_all = pack_bottleneck_to_spatial(
        z_btLd, n_spatial=n_spatial, k=cfg.packing_factor
    )  # (B, T, n_spatial, d_spatial)

    # Use first (context_length) frames as context, next frame as target.
    z_ctx = z_all[:, : cfg.context_length, :, :]
    z_target = z_all[:, cfg.context_length : cfg.context_length + 1, :, :]
    actions_ctx = actions[:, : cfg.context_length]
    action_curr = actions[:, cfg.context_length : cfg.context_length + 1]

    imag_cfg = ImaginationConfig(
        k_max=cfg.k_max,
        horizon=1,
        context_length=cfg.context_length,
        n_spatial=n_spatial,
        d=cfg.imagination_d,
        start_mode="pure",
        tau0_fixed=0.0,
    )
    schedule = _build_static_schedule(imag_cfg)

    rng = jax.random.PRNGKey(123)
    z0 = jax.random.normal(
        rng, z_target.shape, dtype=z_ctx.dtype
    )  # (B, 1, n_spatial, d_spatial)

    # Warmup + timed call to compare JIT vs non-JIT.
    if use_jit:
        print("[single-step] Using JIT-compiled denoiser.")
        # Warmup (compile) and measure compilation+first-run time.
        t_compile0 = time.time()
        _ = denoise_single_latent_static_jit(
            dynamics=train_state.dynamics,
            dyn_vars=train_state.dyn_vars,
            schedule=schedule,
            actions_ctx=actions_ctx,
            action_curr=action_curr,
            z_ctx_t=z_ctx,
            z_noise_t=z0,
            agent_tokens=None,
        )
        t_compile1 = time.time()
        print(f"[single-step][jit] compile+first-run elapsed={t_compile1 - t_compile0:.4f}s")
        t0 = time.time()
        z_pred, _, _ = denoise_single_latent_static_jit(
            dynamics=train_state.dynamics,
            dyn_vars=train_state.dyn_vars,
            schedule=schedule,
            actions_ctx=actions_ctx,
            action_curr=action_curr,
            z_ctx_t=z_ctx,
            z_noise_t=z0,
            agent_tokens=None,
        )
        t1 = time.time()
        print(f"[single-step][jit] elapsed={t1 - t0:.4f}s (after warmup)")
    else:
        print("[single-step] Using non-JIT denoiser.")
        t0 = time.time()
        z_pred, _, _ = denoise_single_latent_static(
            dynamics=train_state.dynamics,
            dyn_vars=train_state.dyn_vars,
            schedule=schedule,
            actions_ctx=actions_ctx,
            action_curr=action_curr,
            z_ctx_t=z_ctx,
            z_noise_t=z0,
            agent_tokens=None,
        )
        t1 = time.time()
        print(f"[single-step][nonjit] elapsed={t1 - t0:.4f}s")

    # Decode both target and prediction.
    gt_btLd = unpack_spatial_to_bottleneck(
        z_target, n_spatial=n_spatial, k=cfg.packing_factor
    )
    pred_btLd = unpack_spatial_to_bottleneck(
        z_pred, n_spatial=n_spatial, k=cfg.packing_factor
    )

    gt_patches = train_state.decoder.apply(
        train_state.dec_vars, gt_btLd, deterministic=True
    )
    pred_patches = train_state.decoder.apply(
        train_state.dec_vars, pred_btLd, deterministic=True
    )

    gt_frames = temporal_unpatchify(
        gt_patches, cfg.dataset.H, cfg.dataset.W, cfg.dataset.C, patch
    )  # (B, 1, H, W, C)
    pred_frames = temporal_unpatchify(
        pred_patches, cfg.dataset.H, cfg.dataset.W, cfg.dataset.C, patch
    )

    mse, psnr = _compute_mse_psnr(pred_frames, gt_frames)

    print(f"[single-step] MSE={mse:.6f}, PSNR={psnr:.2f} dB")

    # ---- Visualization: simple side-by-side PNG for a few examples ----
    out_dir = Path("logs/imagination_test/jit_sampler_single")
    out_dir.mkdir(parents=True, exist_ok=True)

    # gt_frames, pred_frames: (B, 1, H, W, C) → treat as T=1 strip
    gt_np = np.asarray(gt_frames)
    pred_np = np.asarray(pred_frames)

    max_examples = min(4, gt_np.shape[0])
    for ei in range(max_examples):
        fig_path = out_dir / f"single_denoise_b{ei}.png"
        _save_side_by_side_strip(
            fig_path,
            gt_np,
            pred_np,
            title=f"Single-step denoise (example {ei})",
            b_index=ei,
        )
        print(f"[single-step] wrote {fig_path}")


def test_imagination_rollout_real_ckpt(mode: str = "policy", use_jit: bool = False, use_kv_caching: bool = False):
    """
    Full imagination rollout test using real checkpoints.

    mode:
      - "policy": use actions sampled from the policy (off-policy vs GT actions).
                  We DO NOT report PSNR/MSE vs GT here (different action distro).
      - "gt":     use ground-truth actions (teacher-forced rollout).
                  We DO report PSNR/MSE vs GT future frames.
    """
    assert mode in ("policy", "gt"), f"Unknown mode={mode!r}"
    cfg, train_state, frames, actions, _ = _make_real_world_models_and_batch()

    patch = cfg.patch
    n_spatial = cfg.enc_n_latents // cfg.packing_factor

    # Encode full video once.
    patches = temporal_patchify(frames, patch)
    z_btLd, _ = train_state.encoder.apply(
        train_state.enc_vars,
        patches,
        rngs={"mae": train_state.mae_eval_key},
        deterministic=True,
    )
    z_all = pack_bottleneck_to_spatial(
        z_btLd, n_spatial=n_spatial, k=cfg.packing_factor
    )  # (B, T, n_spatial, d_spatial)

    z_ctx = z_all[:, : cfg.context_length, :, :]
    z_future = z_all[
        :, cfg.context_length : cfg.context_length + cfg.horizon, :, :
    ]  # (B, horizon, ...)

    actions_ctx = actions[:, : cfg.context_length]

    # Common sampler configuration
    imag_cfg = ImaginationConfig(
        k_max=cfg.k_max,
        horizon=cfg.horizon,
        context_length=cfg.context_length,
        n_spatial=n_spatial,
        d=cfg.imagination_d,
        start_mode="pure",
        tau0_fixed=0.0,
    )
    sampler = ImaginationSampler(imag_cfg)

    task_ids = jnp.zeros((cfg.dataset.B,), dtype=jnp.int32)
    rng_imag = jax.random.PRNGKey(321)

    # Pick rollout function (JIT vs non-JIT) and log timing.
    if use_jit:
        print(f"[rollout] Using JIT rollout (mode={mode}).")
        rollout_fn = sampler.rollout_latents_jit
    else:
        print(f"[rollout] Using non-JIT rollout (mode={mode}).")
        rollout_fn = sampler.rollout_latents

    # -----------------------------
    # Mode 1: policy actions (off-policy vs GT actions, no PSNR/MSE vs GT)
    # -----------------------------
    if mode == "policy":
        policy_head = train_state.policy_head
        pi_vars = train_state.pi_vars

        def policy_fn(h: jnp.ndarray, rng: jax.Array, state: Any):
            h_for_policy = h[:, None, :]  # (B, 1, d_model)
            pi_logits = policy_head.apply(pi_vars, h_for_policy, deterministic=True)
            logits_t0 = pi_logits[:, 0, 0, :]  # (B, A)
            logp = jax.nn.log_softmax(logits_t0, axis=-1)
            actions_samp = jax.random.categorical(rng, logp, axis=-1)
            return actions_samp, logits_t0, state

        policy_state = None

        # Warmup (for JIT) then timed call, and record compile+first-run time.
        if use_jit:
            t_compile0 = time.time()
            _ = rollout_fn(
                dynamics=train_state.dynamics,
                task_embedder=train_state.task_embedder,
                dyn_vars=train_state.dyn_vars,
                task_vars=train_state.task_vars,
                z_ctx=z_ctx,
                actions_ctx=actions_ctx,
                task_ids=task_ids,
                policy_fn=policy_fn,
                policy_state=policy_state,
                rng_key=rng_imag,
                use_kv_caching=use_kv_caching,
            )
            t_compile1 = time.time()
            print(f"[rollout][mode=policy][jit] compile+first-run elapsed={t_compile1 - t_compile0:.4f}s")
        t0 = time.time()
        imagined_latents, imagined_actions, imagined_hidden, policy_logits = rollout_fn(
            dynamics=train_state.dynamics,
            task_embedder=train_state.task_embedder,
            dyn_vars=train_state.dyn_vars,
            task_vars=train_state.task_vars,
            z_ctx=z_ctx,
            actions_ctx=actions_ctx,
            task_ids=task_ids,
            policy_fn=policy_fn,
            policy_state=policy_state,
            rng_key=rng_imag,
            use_kv_caching=use_kv_caching,
        )
        t1 = time.time()
        print(f"[rollout][mode=policy][{'jit' if use_jit else 'nonjit'}] elapsed={t1 - t0:.4f}s")

    # -----------------------------
    # Mode 2: ground-truth actions via gt_action policy_fn (AR in latents)
    # -----------------------------
    else:  # mode == "gt"
        # Ground-truth actions for the imagination horizon
        gt_actions_seq = actions[
            :, cfg.context_length : cfg.context_length + cfg.horizon
        ]  # (B, horizon)

        gt_policy_fn, gt_policy_state = make_gt_action_policy_fn(
            gt_actions=gt_actions_seq,
            action_dim=cfg.action_dim,
        )

        if use_jit:
            t_compile0 = time.time()
            _ = rollout_fn(
                dynamics=train_state.dynamics,
                task_embedder=train_state.task_embedder,
                dyn_vars=train_state.dyn_vars,
                task_vars=train_state.task_vars,
                z_ctx=z_ctx,
                actions_ctx=actions_ctx,
                task_ids=task_ids,
                policy_fn=gt_policy_fn,
                policy_state=gt_policy_state,
                rng_key=rng_imag,
                use_kv_caching=use_kv_caching,
            )
            t_compile1 = time.time()
            print(f"[rollout][mode=gt][jit] compile+first-run elapsed={t_compile1 - t_compile0:.4f}s")
        t0 = time.time()
        imagined_latents, imagined_actions, imagined_hidden, policy_logits = rollout_fn(
            dynamics=train_state.dynamics,
            task_embedder=train_state.task_embedder,
            dyn_vars=train_state.dyn_vars,
            task_vars=train_state.task_vars,
            z_ctx=z_ctx,
            actions_ctx=actions_ctx,
            task_ids=task_ids,
            policy_fn=gt_policy_fn,
            policy_state=gt_policy_state,
            rng_key=rng_imag,
            use_kv_caching=use_kv_caching,
        )
        t1 = time.time()
        print(f"[rollout][mode=gt][{'jit' if use_jit else 'nonjit'}] elapsed={t1 - t0:.4f}s")

    # Decode imagined future latents (indices 1..horizon) and ground truth.
    imagined_future_latents = imagined_latents[:, 1:, :, :]  # (B, horizon, ...)
    imagined_btLd = unpack_spatial_to_bottleneck(
        imagined_future_latents, n_spatial=n_spatial, k=cfg.packing_factor
    )
    z_future_btLd = unpack_spatial_to_bottleneck(
        z_future, n_spatial=n_spatial, k=cfg.packing_factor
    )

    imagined_patches = train_state.decoder.apply(
        train_state.dec_vars, imagined_btLd, deterministic=True
    )
    gt_patches = train_state.decoder.apply(
        train_state.dec_vars, z_future_btLd, deterministic=True
    )

    imagined_frames = temporal_unpatchify(
        imagined_patches, cfg.dataset.H, cfg.dataset.W, cfg.dataset.C, patch
    )  # (B, horizon, H, W, C)
    gt_frames = temporal_unpatchify(
        gt_patches, cfg.dataset.H, cfg.dataset.W, cfg.dataset.C, patch
    )

    if mode == "policy":
        print("[rollout] mode='policy' (policy-sampled actions; skipping PSNR/MSE vs GT).")
    else:
        mse, psnr = _compute_mse_psnr(imagined_frames, gt_frames)
        print(f"[rollout][mode=gt] MSE={mse:.6f}, PSNR={psnr:.2f} dB")

    print(
        f"[rollout] imagined_latents={imagined_latents.shape}, "
        f"imagined_actions={imagined_actions.shape}, "
        f"imagined_hidden={imagined_hidden.shape}, "
        f"policy_logits={policy_logits.shape}"
    )

    # Visualization: MP4 grid (GT | Imagined) and a few strips
    suffix = "policy" if mode == "policy" else "gt_actions"
    out_dir = Path("logs/imagination_test") / f"jit_sampler_rollout_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_np = np.asarray(gt_frames)          # (B, horizon, H, W, C)
    pred_np = np.asarray(imagined_frames)  # (B, horizon, H, W, C)
    B, H_hor, Hh, Ww, Cc = gt_np.shape

    grid_frames = []
    for t in range(H_hor):
        pair_list = [
            _stack_wide(_to_uint8(gt_np[b, t]), _to_uint8(pred_np[b, t]))
            for b in range(B)
        ]
        grid_frames.append(_tile_videos(pair_list, ncols=min(2, B)))

    out_mp4 = out_dir / f"rollout_{suffix}_gt_vs_imagined.mp4"
    with imageio.get_writer(out_mp4, fps=8, codec="libx264", quality=8) as w:
        for fr in grid_frames:
            w.append_data(fr)
    print(f"[rollout] wrote {out_mp4}")

    max_examples = min(4, B)
    for ei in range(max_examples):
        fig_path = out_dir / f"rollout_{suffix}_strip_b{ei}.png"
        _save_side_by_side_strip(
            fig_path,
            gt_np,
            pred_np,
            title=f"Rollout ({suffix}) GT vs Imagined (example {ei})",
            b_index=ei,
        )
        print(f"[rollout] wrote {fig_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JIT-friendly imagination sampler tests.")
    parser.add_argument(
        "--test",
        type=str,
        default="none",
        choices=["none", "single", "rollout"],
        help="Which self-contained test to run.",
    )
    parser.add_argument(
        "--rollout_mode",
        type=str,
        default="policy",
        choices=["policy", "gt"],
        help="Rollout mode for --test=rollout: 'policy' (sampled actions) or 'gt' (teacher-forced).",
    )
    parser.add_argument(
        "--jit",
        action="store_true",
        help="If set, use JIT-compiled versions of the sampler/denoiser in tests.",
    )
    parser.add_argument(
        "--use_kv_caching",
        action="store_true",
        help="If set, use KV caching during imagination rollouts.",
    )
    args = parser.parse_args()

    if args.test == "single":
        test_single_step_denoise_real_ckpt(use_jit=args.jit)
    elif args.test == "rollout":
        test_imagination_rollout_real_ckpt(mode=args.rollout_mode, use_jit=args.jit, use_kv_caching=args.use_kv_caching)
    else:
        print(
            "No test selected. Use --test=single or --test=rollout to run "
            "self-contained checks of the JIT-friendly sampler."
        )
