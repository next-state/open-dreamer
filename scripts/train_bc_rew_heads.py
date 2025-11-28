# train_bc_rew_heads.py
# given the pretrained world model, train the agent tokens with bc and rew prediction.
# while still applying the diffusion / shortcut loss.
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any
from functools import partial
import json
import time
import math
from einops import reduce

import jax
import jax.numpy as jnp
import numpy as np
import optax
import imageio.v2 as imageio
import orbax.checkpoint as ocp
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# UPDATED: bring in heads + task embedder
from dreamer.models import Encoder, Decoder, Dynamics, TaskEmbedder, PolicyHeadMTP, RewardHeadMTP
from dreamer.data import make_iterator
from dreamer.utils import (
    temporal_patchify,
    pack_bottleneck_to_spatial,
    with_params,
    make_state, make_manager, try_restore, maybe_save,
    pack_mae_params,
)

from dreamer.sampler import SamplerConfig, sample_video

# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class RealismConfig:
    # IO / ckpt
    run_name: str
    tokenizer_ckpt: str
    pretrained_dyn_ckpt: str
    log_dir: str = "./logs"
    ckpt_max_to_keep: int = 2
    ckpt_save_every: int = 10_000


    # wandb config
    use_wandb: bool = False
    wandb_entity: str | None = None  # if None, uses default entity
    wandb_project: str | None = None  # if None, uses run_name as project

    # data
    B: int = 64
    T: int = 64
    H: int = 32
    W: int = 32
    C: int = 3
    pixels_per_step: int = 2
    size_min: int = 6
    size_max: int = 14
    hold_min: int = 4
    hold_max: int = 9
    diversify_data: bool = True
    action_dim: int = 4 # number of categorical actions

    # tokenizer / dynamics config
    patch: int = 4
    enc_n_latents: int = 16
    enc_d_bottleneck: int = 32
    d_model_enc: int = 64
    d_model_dyn: int = 128
    enc_depth: int = 8
    dec_depth: int = 8
    dyn_depth: int = 8
    n_heads: int = 4
    packing_factor: int = 2
    n_register: int = 4 # number of register tokens for dynamics
    n_agent: int = 1 # number of agent tokens for dynamics

    # UPDATED: default to wm_agent (fine-tuning with agent readouts)
    agent_space_mode: str = "wm_agent"

    # schedule
    k_max: int = 8
    bootstrap_start: int = 5_000  # warm-up steps with bootstrap masked out
    self_fraction: float = 0.25   # used once we pass bootstrap_start

    # train
    max_steps: int = 1_000_000_000
    log_every: int = 5_000
    lr: float = 3e-4

    # eval media toggle
    write_video_every: int = 10_000  # set large to reduce IO, or 0 to disable entirely

    # NEW: multi-token prediction (MTP) settings
    L: int = 2                      # predict next L actions/rewards
    num_reward_bins: int = 101      # twohot bins for symexp rewards
    reward_log_low: float = -3.0    # log-space lower bound for reward bins (tune per dataset)
    reward_log_high: float = 3.0   # log-space upper bound for reward bins (tune per dataset)
    n_tasks: int = 128              # task-ID space for TaskEmbedder
    use_task_ids: bool = True       # True: discrete task IDs; False: vector embed
    
    # Loss weighting (to balance scales across different loss components)
    loss_weight_shortcut: float = 1.0    # weight for flow/bootstrap loss (MSE units)
    loss_weight_policy: float = 1.0      # weight for policy CE loss (nats)
    loss_weight_reward: float = 1.0      # weight for reward CE loss (nats)

# ---------------------------
# Small helpers
# ---------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_uint8(img_f32):
    return np.asarray(np.clip(np.asarray(img_f32) * 255.0, 0, 255), dtype=np.uint8)

def _stack_wide(*imgs_hwC):
    return np.concatenate(imgs_hwC, axis=1)

def _tile_videos(trip_list_hwC: list[np.ndarray], *, ncols: int = 2, pad_color: int = 0) -> np.ndarray:
    if len(trip_list_hwC) == 0:
        raise ValueError("Empty video list")
    H, W3, C = trip_list_hwC[0].shape
    B = len(trip_list_hwC)
    nrows = math.ceil(B / ncols)
    total = nrows * ncols
    if total > B:
        blank = np.full((H, W3, C), pad_color, dtype=trip_list_hwC[0].dtype)
        trip_list_hwC = trip_list_hwC + [blank] * (total - B)
    rows = []
    idx = 0
    for _ in range(nrows):
        row_imgs = trip_list_hwC[idx:idx + ncols]
        idx += ncols
        rows.append(np.concatenate(row_imgs, axis=1))
    grid = np.concatenate(rows, axis=0)
    return grid

# ---------------------------
# Tokenizer restore (uses your Orbax layout & utils)
# ---------------------------

def load_pretrained_tokenizer(
    tokenizer_ckpt_dir: str,
    *,
    rng: jnp.ndarray,
    encoder: Encoder,
    decoder: Decoder,
    enc_vars,
    dec_vars,
    sample_patches_btnd,
):
    meta_mngr = make_manager(tokenizer_ckpt_dir, item_names=("meta",))
    latest = meta_mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No tokenizer checkpoint found in {tokenizer_ckpt_dir}")
    restored_meta = meta_mngr.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = restored_meta.meta
    enc_kwargs = meta["enc_kwargs"]
    n_lat, d_b = enc_kwargs["n_latents"], enc_kwargs["d_bottleneck"]

    rng_e1, rng_d1 = jax.random.split(rng)
    B, T = sample_patches_btnd.shape[:2]
    fake_z = jnp.zeros((B, T, n_lat, d_b), dtype=jnp.float32)
    dec_vars = decoder.init({"params": rng_d1, "dropout": rng_d1}, fake_z, deterministic=True)

    packed_example = pack_mae_params(enc_vars, dec_vars)
    tx_dummy = optax.adamw(1e-4)
    opt_state_example = tx_dummy.init(packed_example)
    state_example = make_state(packed_example, opt_state_example, rng_e1, step=0)
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)

    tok_mngr = make_manager(tokenizer_ckpt_dir, item_names=("state", "meta"))
    restored = tok_mngr.restore(
        latest,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            meta=ocp.args.JsonRestore(),
        ),
    )
    packed_params = restored.state["params"]
    enc_params = packed_params["enc"]
    dec_params = packed_params["dec"]
    new_enc_vars = with_params(enc_vars, enc_params)
    new_dec_vars = with_params(dec_vars, dec_params)
    print(f"[tokenizer] Restored encoder/decoder from {tokenizer_ckpt_dir} (step {latest})")
    return new_enc_vars, new_dec_vars, meta

# ---------------------------
# NEW: symexp/twohot helpers for reward targets
# ---------------------------

def _symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

def _twohot_symlog_targets(values, centers_log):
    """
    values: (...,) real rewards
    centers_log: (K,) bin centers in symlog space (log-scale symmetric grid)
    returns: (..., K) two-hot targets that sum to 1
    """
    y = _symlog(values)
    K = centers_log.shape[0]

    idx_r = jnp.searchsorted(centers_log, y, side='right')
    idx_l = jnp.maximum(idx_r - 1, 0)

    idx_r = jnp.minimum(idx_r, K - 1)
    idx_l = jnp.minimum(idx_l, K - 1)

    c_l = jnp.take(centers_log, idx_l)
    c_r = jnp.take(centers_log, idx_r)
    denom = jnp.maximum(c_r - c_l, 1e-8)
    frac = jnp.where(idx_r == idx_l, 0.0, (y - c_l) / denom)

    oh_l = jax.nn.one_hot(idx_l, K)
    oh_r = jax.nn.one_hot(idx_r, K)
    return oh_l * (1.0 - frac)[..., None] + oh_r * frac[..., None]

def _gather_future_actions(labels_bt, L):
    """
    labels_bt: (B, T) int class ids
    returns: labels_btL (B, T, L) and mask_btL (B, T, L) where mask=0 for out-of-range (t+n>=T)
    
    At timestep t, predicts actions[t+1], actions[t+2], ..., actions[t+L]
    (following Dreamer convention: action a_i happens before state s_i)
    """
    B, T = labels_bt.shape
    labels_pad = jnp.pad(labels_bt, ((0,0),(0,L)), constant_values=-1)
    
    # Vectorized version: use advanced indexing instead of list comprehension
    # Create indices for all L offsets at once
    # Offsets start at 1 because we predict the NEXT L actions (t+1, t+2, ..., t+L)
    offsets = jnp.arange(1, L+1)  # (L,) = [1, 2, ..., L]
    indices = jnp.arange(T)[:, None] + offsets[None, :]  # (T, L)
    labels_btL = labels_pad[:, indices]  # (B, T, L) - JAX broadcasts batch dim
    valid_btL = (labels_btL >= 0)
    return labels_btL, valid_btL

def _gather_future_rewards(values_bt, L):
    """
    values_bt: (B, T) float values (e.g., rewards)
    returns: values_btL (B, T, L) and mask_btL (B, T, L) where mask=0 for invalid
    
    At timestep t, predicts values[t], values[t+1], ..., values[t+L-1]
    Following Dreamer convention: r0 is dummy (invalid), so we predict r_t from h_t for t >= 1.
    The first offset (n=0) predicts r_t, which depends on a_t that h_t can see.
    """
    B, T = values_bt.shape
    values_pad = jnp.pad(values_bt, ((0,0),(0,L-1)), constant_values=0.0)
    
    # Vectorized version: offsets start at 0 to predict CURRENT and next L-1 values
    offsets = jnp.arange(0, L)  # (L,) = [0, 1, ..., L-1]
    indices = jnp.arange(T)[:, None] + offsets[None, :]  # (T, L)
    values_btL = values_pad[:, indices]  # (B, T, L)
    
    # Validity: 
    #   - index must be >= 1 (skip r0 which is dummy)
    #   - index must be < T (stay in bounds)
    #   - timestep t must be >= 1 (don't predict from h_0, which has invalid reward)
    # For timestep t, we access index t+offset, so valid when: t >= 1 AND 1 <= t+offset < T
    valid_btL = (indices >= 1) & (indices < T) & (jnp.arange(T)[:, None] >= 1)  # (T, L)
    valid_btL = jnp.broadcast_to(valid_btL[None, :, :], (B, T, L))  # (B, T, L)
    
    return values_btL, valid_btL
# ---------------------------
# Single efficient training step (always used)
# ---------------------------

@partial(
    jax.jit,
    static_argnames=("encoder","dynamics","task_embedder","policy_head","reward_head",
                     "tx","patch","n_spatial","k_max","packing_factor","B","T","B_self","L",
                     "loss_weight_shortcut","loss_weight_policy","loss_weight_reward"),
)
def train_step_efficient(
    encoder, dynamics, task_embedder, policy_head, reward_head, tx,
    params, opt_state,
    enc_vars, dyn_vars, task_vars, pi_vars, rew_vars,
    frames, actions, rewards,
    *,
    patch: int,
    B: int, T: int, B_self: int,            # assume 0 <= B_self < B
    n_spatial: int, k_max: int, packing_factor: int,
    L: int,
    master_key: jnp.ndarray, step: int, bootstrap_start: int,
    loss_weight_shortcut: float,
    loss_weight_policy: float,
    loss_weight_reward: float,
):
    """
    Deterministic two-branch training (one fused main forward):
      - first B_emp rows: empirical flow at d_min = 1/k_max
      - last  B_self rows: bootstrap self-consistency with d > d_min
    Adds multi-token policy/reward losses from agent readouts h_t.
    """
    @partial(jax.jit, static_argnames=("shape_bt","k_max",))
    def _sample_tau_for_step(rng, shape_bt, k_max:int, step_idx:jnp.ndarray, *, dtype=jnp.float32):
        B_, T_ = shape_bt
        K = (1 << step_idx)
        u = jax.random.uniform(rng, (B_, T_), dtype=dtype)
        j_idx = jnp.floor(u * K.astype(dtype)).astype(jnp.int32)
        tau = j_idx.astype(dtype) / K.astype(dtype)
        tau_idx = j_idx * (k_max // K)
        return tau, tau_idx

    @partial(jax.jit, static_argnames=("shape_bt","k_max",))
    def _sample_step_excluding_dmin(rng, shape_bt, k_max:int):
        B_, T_ = shape_bt
        emax = jnp.log2(k_max).astype(jnp.int32)
        step_idx = jax.random.randint(rng, (B_, T_), 0, emax, dtype=jnp.int32)  # exclude emax
        d = 1.0 / (1 << step_idx).astype(jnp.float32)
        return d, step_idx

    # ---------- Param-free precompute ----------
    patches_btnd = temporal_patchify(frames, patch)

    # RNGs
    step_key = jax.random.fold_in(master_key, step)
    (enc_key, key_sigma_full, key_step_self, key_noise_full,
     drop_key, drop_pi, drop_rw, task_key) = jax.random.split(step_key, 8)

    # Frozen encoder → spatial tokens (clean target z1)
    z_bottleneck, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": enc_key}, deterministic=True)
    z1 = pack_bottleneck_to_spatial(z_bottleneck, n_spatial=n_spatial, k=packing_factor)  # (B,T,Sz,Dz)

    # Deterministic batch split
    B_emp  = B - B_self
    actions_full = actions
    emax = jnp.log2(k_max).astype(jnp.int32)

    # --- Step indices (encode d) ---
    step_idx_emp  = jnp.full((B_emp,  T), emax, dtype=jnp.int32)             # d = d_min
    d_self, step_idx_self = _sample_step_excluding_dmin(key_step_self, (B_self, T), k_max)
    step_idx_full = jnp.concatenate([step_idx_emp, step_idx_self], axis=0)   # (B,T)

    # --- Signal levels on each row's grid (one call for whole batch) ---
    sigma_full, sigma_idx_full = _sample_tau_for_step(key_sigma_full, (B, T), k_max, step_idx_full)
    sigma_emp   = sigma_full[:B_emp]
    sigma_self  = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]

    # --- Corrupt inputs: z_tilde = (1 - sigma) z0 + sigma z1 ---
    z0_full      = jax.random.normal(key_noise_full, z1.shape, dtype=z1.dtype)
    z_tilde_full = (1.0 - sigma_full)[...,None,None] * z0_full + sigma_full[...,None,None] * z1
    z_tilde_self = z_tilde_full[B_emp:]

    # --- Ramp weights ---
    w_emp  = 0.9 * sigma_emp  + 0.1
    w_self = 0.9 * sigma_self + 0.1

    # --- Half-step metadata for self rows ---
    d_half            = d_self / 2.0
    step_idx_half     = step_idx_self + 1
    sigma_plus        = sigma_self + d_half
    sigma_idx_plus    = sigma_idx_self + (k_max * d_half).astype(jnp.int32)

    # --- Agent tokens from (dummy) task IDs (B,) -> (B,T,n_agent,D) ---
    dummy_task_ids = jnp.zeros((B,), dtype=jnp.int32)

    def loss_and_aux(p):
        # Bind params into modules (preserving other collections)
        local_dyn  = with_params(dyn_vars,  p["dyn"])
        local_task = with_params(task_vars, p["task"])
        local_pi   = with_params(pi_vars,   p["pi"])
        local_rw   = with_params(rew_vars,  p["rew"])

        # Agent tokens (recomputed under current task params)
        agent_tokens = task_embedder.apply(local_task, dummy_task_ids, B, T)

        # Main forward (emp + self) → z1_hat and agent readouts h
        z1_hat_full, h_btnd = dynamics.apply(
            local_dyn, actions_full, step_idx_full, sigma_idx_full, z_tilde_full,
            agent_tokens=agent_tokens, rngs={"dropout": drop_key}, deterministic=False,
        )  # z1_hat_full: (B,T,Sz,Dz), h: (B,T,n_agent,D)

        # Pool agents (mean) → (B,T,D)
        h_pooled_btd = reduce(h_btnd, "b t n_agent d -> b t d", "mean")
        # ---------- Flow loss on empirical rows (to z1) ----------
        z1_hat_emp  = z1_hat_full[:B_emp]
        z1_hat_self = z1_hat_full[B_emp:]

        flow_per = jnp.mean((z1_hat_emp - z1[:B_emp])**2, axis=(2,3))        # (B_emp,T)
        loss_emp = jnp.mean(flow_per * w_emp)

        # ---------- Self-consistency (bootstrap) on self rows ----------
        do_boot = (B_self > 0) & (step >= bootstrap_start)

        def _boot_loss():
            z1_hat_half1, _ = dynamics.apply(
                local_dyn, actions_full[B_emp:], step_idx_half, sigma_idx_self, z_tilde_self,
                agent_tokens=agent_tokens[B_emp:], rngs={"dropout": drop_key}, deterministic=False,
            )
            b_prime = (z1_hat_half1 - z_tilde_self) / (1.0 - sigma_self)[...,None,None]
            z_prime = z_tilde_self + b_prime * d_half[...,None,None]
            z1_hat_half2, _ = dynamics.apply(
                local_dyn, actions_full[B_emp:], step_idx_half, sigma_idx_plus, z_prime,
                agent_tokens=agent_tokens[B_emp:], rngs={"dropout": drop_key}, deterministic=False,
            )
            b_doubleprime = (z1_hat_half2 - z_prime) / (1.0 - sigma_plus)[...,None,None]
            vhat_sigma = (z1_hat_self - z_tilde_self) / (1.0 - sigma_self)[...,None,None]
            vbar_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)
            boot_per = (1.0 - sigma_self)**2 * jnp.mean((vhat_sigma - vbar_target)**2, axis=(2,3))  # (B_self,T)
            loss_self = jnp.mean(boot_per * w_self)
            return loss_self, jnp.mean(boot_per)

        loss_self, boot_mse = jax.lax.cond(
            do_boot,
            _boot_loss,
            lambda: (jnp.array(0.0, dtype=z1.dtype), jnp.array(0.0, dtype=z1.dtype)),
        )

        # ---------- MTP: Policy (categorical CE over next L actions) ----------
        # logits: (B,T,L,A)
        pi_logits = policy_head.apply(local_pi, h_pooled_btd, rngs={"dropout": drop_pi}, deterministic=False)

        labels_btL, valid_btL = _gather_future_actions(actions_full, L)  # (B,T,L), (B,T,L)
        logp = jax.nn.log_softmax(pi_logits, axis=-1)                # (B,T,L,A)
        A = logp.shape[-1]
        safe_labels = jnp.where(valid_btL, labels_btL, 0)
        tgt = jax.nn.one_hot(safe_labels, A) * valid_btL[..., None]  # (B,T,L,A)
        nll = -jnp.sum(tgt * logp, axis=-1)                          # (B,T,L)
        denom = jnp.maximum(valid_btL.sum(), 1)
        pi_ce = jnp.sum(nll) / denom

        # ---------- MTP: Reward (symexp twohot CE over current and future L rewards) ----------
        # rewards: (B, T) where rewards[:, 0] = r0 (dummy, invalid)
        #          and rewards[:, 1] = r1, rewards[:, 2] = r2, ... are valid
        # At timestep t >= 1, we predict rewards[t], rewards[t+1], ..., rewards[t+L-1]
        # The first offset (n=0) predicts r_t from h_t, which contains a_t
        # We skip t=0 entirely since h_0 has no valid reward to predict
        rew_btL, valid_rew_btL = _gather_future_rewards(rewards, L)  # (B,T,L), (B,T,L)

        # Get bin centers (constants collection) and logits in one forward pass
        rew_logits, centers_log = reward_head.apply(local_rw, h_pooled_btd, rngs={"dropout": drop_rw}, deterministic=False)
        twohot = _twohot_symlog_targets(rew_btL, centers_log)               # (B,T,L,K)
        logq = jax.nn.log_softmax(rew_logits, axis=-1)                      # (B,T,L,K)
        ce_rew = -jnp.sum(twohot * logq, axis=-1)                           # (B,T,L)
        rw_ce = jnp.sum(ce_rew * valid_rew_btL) / jnp.maximum(valid_rew_btL.sum(), 1)
        # ---------- Combine ----------
        shortcut_loss = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B
        loss = (loss_weight_shortcut * shortcut_loss + 
                loss_weight_policy * pi_ce + 
                loss_weight_reward * rw_ce)

        aux = {
            "flow_mse": jnp.mean(flow_per),
            "bootstrap_mse": boot_mse,
            "pi_ce": pi_ce,
            "rw_ce": rw_ce,
            "w_shortcut": loss_weight_shortcut * shortcut_loss,
            "w_pi_ce": loss_weight_policy * pi_ce,
            "w_rw_ce": loss_weight_reward * rw_ce,
        }
        return loss, aux

    (loss_val, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, aux

# ---------------------------
# Eval regimes & plan JSON (unchanged core logic)
# ---------------------------

def _eval_regimes_for_realism(cfg, *, ctx_length: int):
    common = dict(
        k_max=cfg.k_max,
        horizon=min(32, cfg.T - ctx_length),
        ctx_length=ctx_length,
        ctx_signal_tau=1.0,   # was 0.99 — make context clean for fair PSNR
        H=cfg.H, W=cfg.W, C=cfg.C, patch=cfg.patch,
        n_spatial=cfg.enc_n_latents // cfg.packing_factor,
        packing_factor=cfg.packing_factor,
        start_mode="pure",
        rollout="autoregressive",
    )
    regs = []
    regs.append(("finest_pure_AR", SamplerConfig(schedule="finest", **common)))
    regs.append(("shortcut_d4_pure_AR", SamplerConfig(schedule="shortcut", d=1/4, **common)))
    return regs


def _plan_from_sampler_conf(s: SamplerConfig) -> Dict[str, Any]:
    def _is_pow2_frac(x: float) -> bool:
        if x <= 0 or x > 1: return False
        inv = round(1.0 / x)
        return abs(1.0 / inv - x) < 1e-8 and (inv & (inv - 1)) == 0

    if s.schedule == "finest":
        d = 1.0 / float(s.k_max)
    else:
        if s.d is None or not _is_pow2_frac(s.d):
            raise ValueError("shortcut schedule requires d = 1/(power of two)")
        if s.d < 1.0 / float(s.k_max):
            raise ValueError("d finer than finest")
        d = float(s.d)

    tau0 = 0.0
    S = int(round((1.0 - tau0) / d))
    e = int(round(np.log2(round(1.0 / d))))
    tau_seq = [round(tau0 + i*d, 6) for i in range(S + 1)]
    tau_seq[-1] = 1.0
    return dict(
        rollout=s.rollout,
        start_mode=s.start_mode,
        ctx_length=s.ctx_length,
        horizon=s.horizon,
        schedule=s.schedule,
        d=d,
        e=e,
        S=S,
        tau_seq=tau_seq,
        k_max=s.k_max,
        add_ctx_noise_std=getattr(s, "add_ctx_noise_std", 0.0),
    )

# ---------------------------
# Video building and saving utilities
# ---------------------------

def build_tiled_video_frames(
    gt_frames: jnp.ndarray,
    floor_frames: jnp.ndarray,
    pred_frames: jnp.ndarray,
    batch_size: int,
) -> list[np.ndarray]:
    """
    Build tiled video frames from ground truth, floor, and prediction frames.

    Each frame in the output contains a grid of triplets (GT | Floor | Pred) stacked horizontally,
    with multiple batch items tiled vertically/horizontally.

    Args:
        gt_frames: Ground truth frames (B, T, H, W, C)
        floor_frames: Floor/reference frames (B, T, H, W, C)
        pred_frames: Predicted frames (B, T, H, W, C)
        batch_size: Batch size B

    Returns:
        List of grid frames, one per time step
    """
    gt_np_all = _to_uint8(gt_frames)
    floor_np_all = _to_uint8(floor_frames)
    pred_np_all = _to_uint8(pred_frames)

    T_total = gt_np_all.shape[1]
    ncols = 1 if batch_size <= 2 else min(2, batch_size)
    grid_frames = []

    for t_idx in range(T_total):
        trip_list = [
            _stack_wide(gt_np_all[b, t_idx], floor_np_all[b, t_idx], pred_np_all[b, t_idx])
            for b in range(batch_size)
        ]
        grid_img = _tile_videos(trip_list, ncols=ncols, pad_color=0)
        grid_frames.append(grid_img)

    return grid_frames

def save_evaluation_video(
    grid_frames: list[np.ndarray],
    output_path: Path,
    tag: str,
) -> bool:
    """
    Save grid frames as an MP4 video file.

    Args:
        grid_frames: List of grid frames to write
        output_path: Path where MP4 should be saved
        tag: Tag for error messages

    Returns:
        True if successful, False otherwise
    """
    try:
        with imageio.get_writer(output_path, fps=25, codec="libx264", quality=8) as w:
            for fr in grid_frames:
                w.append_data(fr)
        return True
    except Exception as e:
        print(f"[eval:{tag}] MP4 write skipped ({e})")
        return False

def save_evaluation_plan(
    sampler_conf: SamplerConfig,
    step: int,
    mse: float,
    psnr: float,
    output_path: Path,
):
    """
    Save evaluation plan/metadata as JSON.

    Args:
        sampler_conf: Sampler configuration
        step: Training step number
        mse: Mean squared error
        psnr: Peak signal-to-noise ratio in dB
        output_path: Path where JSON should be saved
    """
    plan = _plan_from_sampler_conf(sampler_conf)
    plan["step"] = int(step)
    plan["mse"] = float(mse)
    plan["psnr_db"] = float(psnr)

    with open(output_path, "w") as f:
        json.dump(plan, f, indent=2)

# ---------------------------
# Meta for dynamics checkpoints
# ---------------------------

def make_dynamics_meta(
    *,
    enc_kwargs: dict,
    dec_kwargs: dict,
    dynamics_kwargs: dict,
    H: int, W: int, C: int,
    patch: int,
    k_max: int,
    packing_factor: int,
    n_spatial: int,
    tokenizer_ckpt_dir: str | None = None,
    cfg: Dict[str, Any] | None = None,
):
    return {
        "enc_kwargs": enc_kwargs,
        "dec_kwargs": dec_kwargs,
        "dynamics_kwargs": dynamics_kwargs,
        "H": H, "W": W, "C": C, "patch": patch,
        "k_max": k_max,
        "packing_factor": packing_factor,
        "n_spatial": n_spatial,
        "tokenizer_ckpt_dir": tokenizer_ckpt_dir,
        "cfg": cfg or {},
    }

# ---------------------------
# Training state dataclass
# ---------------------------

@dataclass
class TrainState:
    """Container for all training-related state (models, variables, optimizer, etc.)."""
    encoder: Encoder
    decoder: Decoder
    dynamics: Dynamics
    # NEW modules
    task_embedder: TaskEmbedder
    policy_head: PolicyHeadMTP
    reward_head: RewardHeadMTP

    # vars/collections
    enc_vars: dict
    dec_vars: dict
    dyn_vars: dict
    task_vars: dict
    pi_vars: dict
    rew_vars: dict

    # params packed for a single optimizer (subtrees: dyn/task/pi/rew)
    params: dict
    enc_kwargs: dict
    dec_kwargs: dict
    dyn_kwargs: dict
    tx: optax.Transform
    opt_state: optax.OptState
    mae_eval_key: jnp.ndarray

# ---------------------------
# Model initialization
# ---------------------------
def load_pretrained_dynamics_params(ckpt_dir: str, dyn_vars: dict) -> dict:
    tx_dummy = optax.adam(1e-3)
    opt_state_example = tx_dummy.init(dyn_vars["params"])
    rng = jax.random.PRNGKey(0)
    state_example = {"params": dyn_vars["params"], "opt_state": opt_state_example, "rng": rng, "step": jnp.int32(0)}
    mngr = make_manager(ckpt_dir, item_names=("state","meta"))
    restored = try_restore(mngr, state_example, meta_example={})
    if restored is None:
        raise FileNotFoundError(f"No dynamics checkpoint found in {ckpt_dir}")
    _, r = restored
    p = r.state["params"]
    return p["dyn"] if isinstance(p, dict) and "dyn" in p else p

def initialize_models_and_tokenizer(
    cfg: RealismConfig,
    frames_init: jnp.ndarray,
    actions_init: jnp.ndarray,
) -> TrainState:
    """
    Initialize encoder, decoder, dynamics models and restore tokenizer.
    Also initialize TaskEmbedder, PolicyHeadMTP, RewardHeadMTP and pack params.
    """
    patch = cfg.patch
    num_patches = (cfg.H // patch) * (cfg.W // patch)
    D_patch = patch * patch * cfg.C
    k_max = cfg.k_max

    enc_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        n_heads=cfg.n_heads,
        depth=cfg.enc_depth,
        dropout=0.0,
        d_bottleneck=cfg.enc_d_bottleneck,
        mae_p_min=0.0, mae_p_max=0.0,
        time_every=4, latents_only_time=True,
    )
    dec_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_heads=cfg.n_heads,
        depth=cfg.dec_depth,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        d_patch=D_patch,
        dropout=0.0,
        mlp_ratio=4.0, time_every=4, latents_only_time=True,
    )
    n_spatial = cfg.enc_n_latents // cfg.packing_factor # number of spatial tokens for dynamics
    dyn_kwargs = dict(
        d_model=cfg.d_model_dyn,
        d_bottleneck=cfg.enc_d_bottleneck,
        d_spatial=cfg.enc_d_bottleneck * cfg.packing_factor,
        n_spatial=n_spatial, n_register=cfg.n_register,
        n_heads=cfg.n_heads, depth=cfg.dyn_depth,
        space_mode=cfg.agent_space_mode, n_agent=cfg.n_agent,
        dropout=0.0, k_max=k_max, 
        time_every=4,
    )

    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)
    dynamics = Dynamics(**dyn_kwargs)

    patches_btnd = temporal_patchify(frames_init, patch)
    rng = jax.random.PRNGKey(0)
    enc_vars = encoder.init({"params": rng, "mae": rng, "dropout": rng}, patches_btnd, deterministic=True)
    fake_z = jnp.zeros((cfg.B, cfg.T, cfg.enc_n_latents, cfg.enc_d_bottleneck))
    dec_vars = decoder.init({"params": rng, "dropout": rng}, fake_z, deterministic=True)

    # Restore tokenizer
    enc_vars, dec_vars, _ = load_pretrained_tokenizer(
        cfg.tokenizer_ckpt, rng=rng,
        encoder=encoder, decoder=decoder,
        enc_vars=enc_vars, dec_vars=dec_vars,
        sample_patches_btnd=patches_btnd,
    )

    # Build initial z1 to shape dynamics init
    mae_eval_key = jax.random.PRNGKey(777)
    z_btLd, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": mae_eval_key}, deterministic=True)
    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=cfg.packing_factor)
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jnp.full((cfg.B, cfg.T), emax, dtype=jnp.int32)
    sigma_idx = jnp.full((cfg.B, cfg.T), k_max - 1, dtype=jnp.int32)
    dyn_vars = dynamics.init({"params": rng, "dropout": rng}, actions_init, step_idx, sigma_idx, z1)
    print(f"[dynamics] Loading pretrained params from {cfg.pretrained_dyn_ckpt}")
    dyn_pre = load_pretrained_dynamics_params(cfg.pretrained_dyn_ckpt, dyn_vars)
    dyn_vars = with_params(dyn_vars, dyn_pre)

    # NEW: heads & task embedder
    task_embedder = TaskEmbedder(d_model=cfg.d_model_dyn, n_agent=cfg.n_agent,
                                 use_ids=cfg.use_task_ids, n_tasks=cfg.n_tasks)
    policy_head  = PolicyHeadMTP(d_model=cfg.d_model_dyn, action_dim=cfg.action_dim, L=cfg.L)
    reward_head  = RewardHeadMTP(d_model=cfg.d_model_dyn, L=cfg.L, num_bins=cfg.num_reward_bins,
                                 log_low=cfg.reward_log_low, log_high=cfg.reward_log_high)

    rng_task, rng_pi, rng_rw = jax.random.split(jax.random.PRNGKey(1), 3)
    dummy_task_ids = jnp.zeros((cfg.B,), dtype=jnp.int32)
    task_vars = task_embedder.init({"params": rng_task}, dummy_task_ids, cfg.B, cfg.T)

    fake_h = jnp.zeros((cfg.B, cfg.T, cfg.d_model_dyn), dtype=jnp.float32)
    pi_vars  = policy_head.init({"params": rng_pi, "dropout": rng_pi}, fake_h, deterministic=True)
    rew_vars = reward_head.init({"params": rng_rw, "dropout": rng_rw}, fake_h, deterministic=True)

    params = {
        "dyn": dyn_vars["params"],
        "task": task_vars["params"],
        "pi": pi_vars["params"],
        "rew": rew_vars["params"],
    }

    tx = optax.adam(cfg.lr)
    opt_state = tx.init(params)

    return TrainState(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        task_embedder=task_embedder,
        policy_head=policy_head,
        reward_head=reward_head,
        enc_vars=enc_vars,
        dec_vars=dec_vars,
        dyn_vars=dyn_vars,
        task_vars=task_vars,
        pi_vars=pi_vars,
        rew_vars=rew_vars,
        params=params,
        enc_kwargs=enc_kwargs,
        dec_kwargs=dec_kwargs,
        dyn_kwargs=dyn_kwargs,
        tx=tx,
        opt_state=opt_state,
        mae_eval_key=mae_eval_key,
    )

# ---------------------------
# Evaluation logic
# ---------------------------

def run_evaluation(
    cfg: RealismConfig,
    step: int,
    train_state: TrainState,
    next_batch,
    vis_dir: Path,
):
    """
    Run periodic evaluation: sample videos, compute metrics, and save visualization.
    """
    val_rng = jax.random.PRNGKey(9999)
    _, (val_frames, val_actions, val_rewards) = next_batch(val_rng)
    # UPDATED: bind only the dynamics params
    dyn_vars_eval = with_params(train_state.dyn_vars, train_state.params["dyn"])
    ctx_length = min(32, cfg.T - 1)
    regimes = _eval_regimes_for_realism(cfg, ctx_length=ctx_length)

    for tag, sampler_conf in regimes:
        sampler_conf.mae_eval_key = train_state.mae_eval_key
        sampler_conf.rng_key = jax.random.PRNGKey(4242)
        t0 = time.time()

        pred_frames, floor_frames, gt_frames = sample_video(
            encoder=train_state.encoder,
            decoder=train_state.decoder,
            dynamics=train_state.dynamics,
            enc_vars=train_state.enc_vars,
            dec_vars=train_state.dec_vars,
            dyn_vars=dyn_vars_eval,
            frames=val_frames, actions=val_actions, config=sampler_conf,
        )

        dt = time.time() - t0
        HZ = sampler_conf.horizon
        mse = float(jnp.mean((pred_frames[:, -HZ:] - gt_frames[:, -HZ:]) ** 2))
        psnr = float(10.0 * jnp.log10(1.0 / jnp.maximum(mse, 1e-12)))
        print(f"[eval:{tag}] step={step:06d} | AR_hz={HZ} | MSE={mse:.6g} | PSNR={psnr:.2f} dB | {dt:.2f}s")

        grid_frames = build_tiled_video_frames(
            gt_frames=gt_frames,
            floor_frames=floor_frames,
            pred_frames=pred_frames,
            batch_size=cfg.B,
        )

        tag_dir = _ensure_dir(vis_dir / f"step_{step:06d}")
        mp4_path = tag_dir / f"{tag}_grid.mp4"
        save_evaluation_video(grid_frames, mp4_path, tag)
        print(f"[eval:{tag}] wrote {mp4_path.name} in {tag_dir}")

        if cfg.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                f"eval/{tag}/mse": mse,
                f"eval/{tag}/psnr": psnr,
                f"eval/{tag}/horizon": HZ,
                f"eval/{tag}/eval_time": dt,
            }, step=step)
            if grid_frames:
                wandb.log({
                    f"eval/{tag}/video": wandb.Video(mp4_path, format="mp4"),
                }, step=step)

# ---------------------------
# Main
# ---------------------------

def run(cfg: RealismConfig):
    # Initialize wandb if enabled
    if cfg.use_wandb:
        if not WANDB_AVAILABLE:
            print("[warning] wandb requested but not installed. Install with: pip install wandb")
            print("[warning] Continuing without wandb logging.")
        else:
            wandb_project = cfg.wandb_project or cfg.run_name
            wandb.init(
                entity=cfg.wandb_entity,
                project=wandb_project,
                name=cfg.run_name,
                config=asdict(cfg),
                dir=str(Path(cfg.log_dir).resolve()),
            )
            print(f"[wandb] Initialized run: {wandb.run.name if wandb.run else 'N/A'}")

    # Output dirs
    root = _ensure_dir(Path(cfg.log_dir))
    run_dir = _ensure_dir(root / cfg.run_name)
    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
    vis_dir = _ensure_dir(run_dir / "viz")
    print(f"[setup] writing artifacts to: {run_dir.resolve()}")

    # Data iterator (streaming)
    next_batch = make_iterator(
        cfg.B, cfg.T, cfg.H, cfg.W, cfg.C,
        pixels_per_step=cfg.pixels_per_step,
        size_min=cfg.size_min, size_max=cfg.size_max,
        hold_min=cfg.hold_min, hold_max=cfg.hold_max,
        fg_min_color=0 if cfg.diversify_data else 128,
        fg_max_color=255 if cfg.diversify_data else 128,
        bg_min_color=0 if cfg.diversify_data else 255,
        bg_max_color=255 if cfg.diversify_data else 255,
    )

    # Initialize models and restore tokenizer
    init_rng = jax.random.PRNGKey(0)
    _, (frames_init, actions_init, _) = next_batch(init_rng)

    train_state = initialize_models_and_tokenizer(cfg, frames_init, actions_init)

    # Extract some values for checkpointing
    patch = cfg.patch
    k_max = cfg.k_max
    n_spatial = cfg.enc_n_latents // cfg.packing_factor

    # -------- Orbax manager & (optional) restore --------
    mngr = make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every)
    meta = make_dynamics_meta(
        enc_kwargs=train_state.enc_kwargs,
        dec_kwargs=train_state.dec_kwargs,
        dynamics_kwargs=train_state.dyn_kwargs,
        H=cfg.H, W=cfg.W, C=cfg.C, patch=patch,
        k_max=k_max, packing_factor=cfg.packing_factor, n_spatial=n_spatial,
        tokenizer_ckpt_dir=cfg.tokenizer_ckpt,
        cfg=asdict(cfg),
    )

    rng = jax.random.PRNGKey(0)
    state_example = make_state(train_state.params, train_state.opt_state, rng, step=0)
    restored = try_restore(mngr, state_example, meta)

    start_step = 0
    if restored is not None:
        latest_step, r = restored
        train_state.params = r.state["params"]
        train_state.opt_state = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"]) + 1
        # UPDATED: bind subtrees into module vars
        train_state.dyn_vars  = with_params(train_state.dyn_vars,  train_state.params["dyn"])
        train_state.task_vars = with_params(train_state.task_vars, train_state.params["task"])
        train_state.pi_vars   = with_params(train_state.pi_vars,   train_state.params["pi"])
        train_state.rew_vars  = with_params(train_state.rew_vars,  train_state.params["rew"])
        print(f"[restore] Resumed from {ckpt_dir} at step={latest_step}")

    # -------- Training loop --------
    train_rng = jax.random.PRNGKey(2025)
    data_rng = jax.random.PRNGKey(12345)

    start_wall = time.time()
    for step in range(start_step, cfg.max_steps + 1):
        # Data
        data_rng, batch_key = jax.random.split(data_rng)
        _, (frames, actions, rewards) = next_batch(batch_key)

        # RNG for this step
        train_rng, master_key = jax.random.split(train_rng)

        # Decide current B_self based on warm-up (static arg requires a single value; we keep B_self fixed
        # and gate its contribution inside the jit via bootstrap_start masking).
        B_self = max(0, int(round(cfg.self_fraction * cfg.B)))

        train_step_start_time = time.time()
        train_state.params, train_state.opt_state, aux = train_step_efficient(
            train_state.encoder, train_state.dynamics, train_state.task_embedder,
            train_state.policy_head, train_state.reward_head, train_state.tx,
            train_state.params, train_state.opt_state,
            train_state.enc_vars, train_state.dyn_vars, train_state.task_vars,
            train_state.pi_vars, train_state.rew_vars,
            frames, actions, rewards,
            patch=cfg.patch, B=cfg.B, T=cfg.T, B_self=B_self,
            n_spatial=n_spatial, k_max=k_max, packing_factor=cfg.packing_factor,
            L=cfg.L,
            master_key=master_key, step=step, bootstrap_start=cfg.bootstrap_start,
            loss_weight_shortcut=cfg.loss_weight_shortcut,
            loss_weight_policy=cfg.loss_weight_policy,
            loss_weight_reward=cfg.loss_weight_reward,
        )

        # Logging
        if (step % cfg.log_every == 0) or (step == cfg.max_steps):
            flow_mse = float(aux['flow_mse'])
            boot_mse = float(aux['bootstrap_mse'])
            pi_ce    = float(aux['pi_ce'])   
            rw_ce   = float(aux['rw_ce'])
            w_shortcut = float(aux['w_shortcut'])
            w_pi_ce = float(aux['w_pi_ce'])
            w_rw_ce = float(aux['w_rw_ce'])
            step_time = time.time() - train_step_start_time
            total_time = time.time() - start_wall

            pieces = [
                f"[train] step={step:06d}",
                f"flow_mse={flow_mse:.6g}",
                f"boot_mse={boot_mse:.6g}",
                f"w_pi_ce={w_pi_ce:.4f}",
                f"w_rw_ce={w_rw_ce:.4f}",
                f"t={step_time:.4f}s",
                f"total_t={total_time:.1f}s",
            ]
            print(" | ".join(pieces))

            # Log to wandb
            if cfg.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "train/flow_mse": flow_mse,
                    "train/bootstrap_mse": boot_mse,
                    "train/pi_ce": pi_ce,
                    "train/rw_ce": rw_ce,
                    "train/w_shortcut": w_shortcut,
                    "train/w_pi_ce": w_pi_ce,
                    "train/w_rw_ce": w_rw_ce,
                    "train/step_time": step_time,
                    "train/total_time": total_time,
                    "step": step,
                }, step=step)

        # Save (async) when policy says we should
        state = make_state(train_state.params, train_state.opt_state, train_rng, step)
        maybe_save(mngr, step, state, meta)

        # Periodic lightweight AR eval
        if cfg.write_video_every and (step % cfg.write_video_every == 0):
            run_evaluation(
                cfg=cfg,
                step=step,
                train_state=train_state,
                next_batch=next_batch,
                vis_dir=vis_dir,
            )

    # Ensure all writes finished
    mngr.wait_until_finished()

    # Save final config
    (run_dir / "config.txt").write_text("\n".join([f"{k}={v}" for k, v in asdict(cfg).items()]))

    # Finish wandb run
    if cfg.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        print("[wandb] Finished logging.")


if __name__ == "__main__":
    cfg = RealismConfig(
        run_name="train_bc_rew_flippedrew_test",
        tokenizer_ckpt="/vast/projects/dineshj/lab/hued/tiny_dreamer_4/logs/pretrained_mae/checkpoints",
        pretrained_dyn_ckpt="/vast/projects/dineshj/lab/hued/tiny_dreamer_4/logs/train_ndynamics_newattn/checkpoints",
        use_wandb=False,
        wandb_entity="edhu",
        wandb_project="tiny_dreamer_4",
        log_dir="/vast/projects/dineshj/lab/hued/tiny_dreamer_4/logs",
        max_steps=1_000_000_000,
        log_every=5_000,
        lr=1e-4,
        write_video_every=100_000,
        ckpt_save_every=100_000,
        ckpt_max_to_keep=2,
        loss_weight_shortcut=1.0,
        loss_weight_policy=0.3,
        loss_weight_reward=0.3,
        action_dim=4,
    )
    print("Running realism config:\n  " + "\n  ".join([f"{k}={v}" for k,v in asdict(cfg).items()]))
    run(cfg)
