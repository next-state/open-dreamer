# eval_bc_rew_heads.py
# Evaluate dynamics + policy/reward heads in teacher-forced and autoregressive modes.
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple
import math
import csv

import jax
import jax.numpy as jnp
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp

# Project imports
from dreamer.models import Encoder, Decoder, Dynamics, TaskEmbedder, PolicyHeadMTP, RewardHeadMTP
from dreamer.data import make_iterator
from dreamer.utils import (
    temporal_patchify, pack_bottleneck_to_spatial, 
    with_params, make_state, make_manager, pack_mae_params,
)

from dreamer.sampler import SamplerConfig, sample_video


# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class EvalConfig:
    # Paths
    run_ckpt_dir: str                         # training run checkpoints dir to restore params from
    tokenizer_ckpt: str                       # tokenizer ckpt (for enc/dec)
    out_dir: str = "./eval_out"

    # Data
    B: int = 8
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
    action_dim: int = 4

    # Tokenizer / dynamics
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
    n_register: int = 4
    n_agent: int = 1
    agent_space_mode: str = "wm_agent"
    k_max: int = 8

    # Heads
    L: int = 2
    num_reward_bins: int = 101
    reward_log_low: float = -3.0    # log-space lower bound for reward bins (tune per dataset)
    reward_log_high: float = 3.0   # log-space upper bound for reward bins (tune per dataset)
    n_tasks: int = 128
    use_task_ids: bool = True

    # Sampler/eval
    ctx_length: int = 32
    horizon: int = 16
    schedule: str = "finest"  # "finest" or "shortcut"
    d: float | None = None    # e.g., 0.25 for shortcut
    ctx_signal_tau: float = 1.0
    match_ctx_tau: bool = False

    # Visualization
    max_examples_to_plot: int = 4  # number of sequences to render as strips

    # Safety: ensure heads never see future actions when predicting next actions
    paranoid_no_leak: bool = True

# ---------------------------
# Utilities (reward bins, gatherers, plotting)
# ---------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_uint8(img_f32):
    return np.asarray(np.clip(np.asarray(img_f32) * 255.0, 0, 255), dtype=np.uint8)

def _stack_wide(*imgs_hwC):
    return np.concatenate(imgs_hwC, axis=1)

def _tile_videos(trip_list_hwC: list[np.ndarray], *, ncols: int = 2, pad_color: int = 0) -> np.ndarray:
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

def _symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

def _symexp(y):
    # inverse of symlog
    return jnp.sign(y) * (jnp.expm1(jnp.abs(y)))

def _twohot_symlog_targets(values, centers_log):
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
    B, T = labels_bt.shape
    labels_pad = jnp.pad(labels_bt, ((0,0),(0,L)), constant_values=-1)
    offsets = jnp.arange(1, L+1)
    idx = jnp.arange(T)[:, None] + offsets[None, :]
    labels_btL = labels_pad[:, idx]
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

def _save_reward_lineplot(fig_path: Path,
                          gt_rewards_h: np.ndarray | None,
                          pred_rewards_h: np.ndarray,
                          title: str):
    """
    gt_rewards_h: (H,) or None
    pred_rewards_h: (H,)
    """
    H = pred_rewards_h.shape[0]
    xs = np.arange(1, H+1)
    plt.figure(figsize=(max(6, H*0.6), 3.0))
    if gt_rewards_h is not None:
        plt.plot(xs, gt_rewards_h, label="GT reward", linewidth=2)
    plt.plot(xs, pred_rewards_h, label="Pred reward", linewidth=2)
    plt.xlabel("timestep (+offset)")
    plt.ylabel("reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()

def _save_strip(fig_path: Path, frames_b_t_hwc: np.ndarray, ctx_len: int, hor: int,
                gt_actions_bt: np.ndarray | None,
                pred_actions_bt: np.ndarray | None,
                gt_rewards_bt: np.ndarray | None,
                pred_rewards_bt: np.ndarray | None,
                title: str,
                b_index: int = 0):   # <-- NEW
    """
    Save a single sequence strip with text under each future frame.

    - frames_b_t_hwc: (B, T_total, H, W, C)
    - gt_* arrays should be (B, T_total) if provided
    - pred_* arrays may be (B, T_total) OR (B, horizon). We auto-detect.
    We always annotate the horizon window [ctx_len : ctx_len + hor).
    """
    frames = _to_uint8(frames_b_t_hwc)
    B, Ttot = frames.shape[:2]
    b = int(np.clip(b_index, 0, B-1))   # <-- use requested example
    cols = hor
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, cols, figsize=(cols*2.2, 4.0), constrained_layout=True)
    fig.suptitle(title, fontsize=12)

    # helpers to index either full-length (Ttot) or horizon-length arrays
    def _pick(arr_bt: np.ndarray | None, t_offset: int) -> float | int | None:
        if arr_bt is None:
            return None
        # allow both (B, Ttot) and (B, hor)
        if arr_bt.shape[1] == Ttot:         # full timeline
            return arr_bt[b, ctx_len + t_offset]
        elif arr_bt.shape[1] == hor:        # horizon-only
            return arr_bt[b, t_offset]
        else:
            # unexpected shape: be defensive
            return None

    # row 0: images of future frames
    for i in range(hor):
        ax = axes[0, i]
        ax.imshow(frames[b, ctx_len + i])
        ax.axis('off')
        ax.set_title(f"t+{i+1}", fontsize=9)

    # row 1: annotations
    for i in range(hor):
        parts = []
        ga = _pick(gt_actions_bt, i)
        pa = _pick(pred_actions_bt, i)
        gr = _pick(gt_rewards_bt, i)
        pr = _pick(pred_rewards_bt, i)

        if ga is not None and pa is not None:
            parts.append(f"act GT={int(ga)} / Pred={int(pa)}")
        elif pa is not None:
            parts.append(f"act Pred={int(pa)}")

        if gr is not None and pr is not None:
            parts.append(f"rew GT={float(gr):.3f} / Pred={float(pr):.3f}")
        elif pr is not None:
            parts.append(f"rew Pred={float(pr):.3f}")

        ax = axes[1, i]
        ax.axis('off')
        ax.text(0.5, 0.5, "\n".join(parts) if parts else "", ha='center', va='center', fontsize=8)

    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

# ---------------------------
# Restorers
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
    tx_dummy = optax = __import__("optax").adamw(1e-4)
    opt_state_example = optax.init(packed_example)
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

def restore_run_params(run_ckpt_dir: str, params_example: dict, opt_state_example, rng):
    mngr = make_manager(run_ckpt_dir, item_names=("state","meta"))
    abstract_state = make_state(params_example, opt_state_example, rng, step=0)
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, abstract_state)
    restore_args = ocp.args.Composite(
        state=ocp.args.StandardRestore(abstract_state),
        meta=ocp.args.JsonRestore(),
    )
    latest = mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No training checkpoint found in {run_ckpt_dir}")
    restored = mngr.restore(latest, args=restore_args)
    print(f"[restore] Loaded training checkpoint {latest} from {run_ckpt_dir}")
    return latest, restored

# ---------------------------
# Model init (mirror of training shapes)
# ---------------------------

def init_models_and_restore(cfg: EvalConfig):
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
    n_spatial = cfg.enc_n_latents // cfg.packing_factor
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
    task_embedder = TaskEmbedder(d_model=cfg.d_model_dyn, n_agent=cfg.n_agent,
                                 use_ids=cfg.use_task_ids, n_tasks=cfg.n_tasks)
    policy_head  = PolicyHeadMTP(d_model=cfg.d_model_dyn, action_dim=cfg.action_dim, L=cfg.L)
    reward_head  = RewardHeadMTP(d_model=cfg.d_model_dyn, L=cfg.L, num_bins=cfg.num_reward_bins,
                                 log_low=cfg.reward_log_low, log_high=cfg.reward_log_high)

    # shape init
    rng = jax.random.PRNGKey(0)
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
    _, (frames_init, actions_init, _) = next_batch(rng)

    patches_btnd = temporal_patchify(frames_init, cfg.patch)
    enc_vars = encoder.init({"params": rng, "mae": rng, "dropout": rng}, patches_btnd, deterministic=True)
    fake_z = jnp.zeros((cfg.B, cfg.T, cfg.enc_n_latents, cfg.enc_d_bottleneck))
    dec_vars = decoder.init({"params": rng, "dropout": rng}, fake_z, deterministic=True)

    # restore tokenizer
    enc_vars, dec_vars, _ = load_pretrained_tokenizer(
        cfg.tokenizer_ckpt, rng=rng, encoder=encoder, decoder=decoder,
        enc_vars=enc_vars, dec_vars=dec_vars, sample_patches_btnd=patches_btnd,
    )

    # init dynamics vars from shapes
    mae_eval_key = jax.random.PRNGKey(777)
    z_btLd, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": mae_eval_key}, deterministic=True)
    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=cfg.packing_factor)
    emax = jnp.log2(cfg.k_max).astype(jnp.int32)
    step_idx = jnp.full((cfg.B, cfg.T), emax, dtype=jnp.int32)
    sigma_idx = jnp.full((cfg.B, cfg.T), cfg.k_max - 1, dtype=jnp.int32)
    dyn_vars = dynamics.init({"params": rng, "dropout": rng}, actions_init, step_idx, sigma_idx, z1)

    # init heads/task
    rng_task, rng_pi, rng_rw = jax.random.split(jax.random.PRNGKey(1), 3)
    dummy_task_ids = jnp.zeros((cfg.B,), dtype=jnp.int32)
    task_vars = task_embedder.init({"params": rng_task}, dummy_task_ids, cfg.B, cfg.T)
    fake_h = jnp.zeros((cfg.B, cfg.T, cfg.d_model_dyn), dtype=jnp.float32)
    pi_vars  = policy_head.init({"params": rng_pi, "dropout": rng_pi}, fake_h, deterministic=True)
    rew_vars = reward_head.init({"params": rng_rw, "dropout": rng_rw}, fake_h, deterministic=True)

    # restore training checkpoint (params for dyn/task/pi/rew)
    params_example = {
        "dyn": dyn_vars["params"],
        "task": task_vars["params"],
        "pi": pi_vars["params"],
        "rew": rew_vars["params"],
    }
    # dummy opt state just to satisfy shape
    import optax
    tx = optax.adam(1e-4)
    opt_state_example = tx.init(params_example)
    latest, restored = restore_run_params(cfg.run_ckpt_dir, params_example, opt_state_example, rng)
    params = restored.state["params"]

    # bind into vars used for .apply
    dyn_vars  = with_params(dyn_vars,  params["dyn"])
    task_vars = with_params(task_vars, params["task"])
    pi_vars   = with_params(pi_vars,   params["pi"])
    rew_vars  = with_params(rew_vars,  params["rew"])

    return dict(
        encoder=encoder, decoder=decoder, dynamics=dynamics,
        task_embedder=task_embedder, policy_head=policy_head, reward_head=reward_head,
        enc_vars=enc_vars, dec_vars=dec_vars, dyn_vars=dyn_vars,
        task_vars=task_vars, pi_vars=pi_vars, rew_vars=rew_vars,
        mae_eval_key=mae_eval_key, latest_step=int(restored.state["step"]),
        next_batch=next_batch, n_spatial=n_spatial
    )

# ---------------------------
# Teacher-Forced head eval
# ---------------------------

def eval_teacher_forced(cfg: EvalConfig, env, out_dir: Path):
    enc, dec, dyn = env["encoder"], env["decoder"], env["dynamics"]
    task, pi_head, rw_head = env["task_embedder"], env["policy_head"], env["reward_head"]
    enc_vars, dec_vars, dyn_vars = env["enc_vars"], env["dec_vars"], env["dyn_vars"]
    task_vars, pi_vars, rew_vars = env["task_vars"], env["pi_vars"], env["rew_vars"]
    n_spatial = env["n_spatial"]

    # batch
    rng = jax.random.PRNGKey(123)
    _, (frames, actions, rewards) = env["next_batch"](rng)

    # encode to clean latents
    patches = temporal_patchify(frames, cfg.patch)
    z_btLd, _ = enc.apply(enc_vars, patches, rngs={"mae": env["mae_eval_key"]}, deterministic=True)
    z_all = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=cfg.packing_factor)

    # run dynamics once per timestep with "pure/clean" inputs: sigma=1 → z_tilde=z1, sigma_idx=k_max-1
    emax = jnp.log2(cfg.k_max).astype(jnp.int32)
    step_idx = jnp.full((cfg.B, cfg.T), emax, dtype=jnp.int32)
    sigma_idx = jnp.full((cfg.B, cfg.T), cfg.k_max - 1, dtype=jnp.int32)
    z_tilde = z_all  # tau=1 ⇒ z_tilde=z1

    # agent tokens
    dummy_task_ids = jnp.zeros((cfg.B,), dtype=jnp.int32)
    agents = task.apply(task_vars, dummy_task_ids, cfg.B, cfg.T)

    z1_hat_full, h = dyn.apply(
        dyn_vars, actions, step_idx, sigma_idx, z_tilde,
        agent_tokens=agents, deterministic=True
    )
        # run dynamics once per timestep with "pure/clean" inputs: sigma=1 → z_tilde=z1, sigma_idx=k_max-1
    emax = jnp.log2(cfg.k_max).astype(jnp.int32)
    step_idx_full = jnp.full((cfg.B, cfg.T), emax, dtype=jnp.int32)
    sigma_idx_full = jnp.full((cfg.B, cfg.T), cfg.k_max - 1, dtype=jnp.int32)
    z_tilde_full = z_all  # tau=1 ⇒ z_tilde=z1

    # agent tokens
    dummy_task_ids = jnp.zeros((cfg.B,), dtype=jnp.int32)
    agents_full = task.apply(task_vars, dummy_task_ids, cfg.B, cfg.T)

    # ---- Fast bulk pass (may leak if the dynamics weren't strictly causal) ----
    z1_hat_full_bulk, h_bulk = dyn.apply(
        dyn_vars, actions, step_idx_full, sigma_idx_full, z_tilde_full,
        agent_tokens=agents_full, deterministic=True
    )
    h_pooled_bulk = jnp.mean(h_bulk, axis=2) if h_bulk is not None else jnp.zeros((cfg.B, cfg.T, cfg.d_model_dyn), z_all.dtype)

    # ---- Paranoid no-leak pass: compute h_t with prefix [0..t] only ----
    if cfg.paranoid_no_leak:
        h_collect = []
        for t in range(cfg.T):
            Tt = t + 1
            z_slice = z_tilde_full[:, :Tt, :, :]
            a_slice = actions[:, :Tt]
            step_idx = step_idx_full[:, :Tt]
            sigma_idx = sigma_idx_full[:, :Tt]
            agents = agents_full[:, :Tt, :, :] if agents_full.ndim == 4 else agents_full[:, :Tt]

            _, h_t = dyn.apply(
                dyn_vars, a_slice, step_idx, sigma_idx, z_slice,
                agent_tokens=agents, deterministic=True
            )
            h_last = jnp.mean(h_t, axis=2)[:, -1, :]  # (B, D)
            h_collect.append(h_last)

        h_pooled_prefix = jnp.stack(h_collect, axis=1)  # (B, T, D)
        h_for_heads = h_pooled_prefix
    else:
        h_for_heads = h_pooled_bulk

    # ---------------- Heads on h_for_heads ----------------
    # Policy head
    pi_logits = pi_head.apply(pi_vars, h_for_heads, deterministic=True)  # (B,T,L,A)
    labels_btL, valid_btL = _gather_future_actions(actions, cfg.L)
    logp = jax.nn.log_softmax(pi_logits, axis=-1)
    A = logp.shape[-1]
    safe_labels = jnp.where(valid_btL, labels_btL, 0)
    tgt = jax.nn.one_hot(safe_labels, A) * valid_btL[..., None]
    nll = -jnp.sum(tgt * logp, axis=-1)                          # (B,T,L)
    denom = jnp.maximum(valid_btL.sum(), 1)
    pi_ce = jnp.sum(nll) / denom
    pred_top1 = jnp.argmax(logp, axis=-1)                        # (B,T,L)
    acc = jnp.sum((pred_top1 == labels_btL) * valid_btL) / jnp.maximum(valid_btL.sum(), 1)

    # Reward head
    rw_logits, centers_log = rw_head.apply(rew_vars, h_for_heads, deterministic=True)  # (B,T,L,K), (K,)
    rew_btL, valid_rew_btL = _gather_future_rewards(rewards, cfg.L)
    twohot = _twohot_symlog_targets(rew_btL, centers_log)
    logq = jax.nn.log_softmax(rw_logits, axis=-1)
    ce_rew = -jnp.sum(twohot * logq, axis=-1)                           # (B,T,L)
    rw_ce = jnp.sum(ce_rew * valid_rew_btL) / jnp.maximum(valid_rew_btL.sum(), 1)

    # Decode predicted rewards (expectation in symlog space)
    probs = jnp.exp(logq)  # (B,T,L,K)
    exp_symlog = jnp.sum(probs * centers_log[None, None, None, :], axis=-1)  # (B,T,L)
    exp_reward = _symexp(exp_symlog)  # decoded to real space

    # Optional: compare with bulk (leak-prone) numbers to flag issues
    if cfg.paranoid_no_leak:
        pi_logits_bulk = pi_head.apply(pi_vars, h_pooled_bulk, deterministic=True)
        logp_bulk = jax.nn.log_softmax(pi_logits_bulk, axis=-1)
        nll_bulk = -jnp.sum(tgt * logp_bulk, axis=-1)
        pi_ce_bulk = jnp.sum(nll_bulk) / denom
        pred_top1_bulk = jnp.argmax(logp_bulk, axis=-1)
        acc_bulk = jnp.sum((pred_top1_bulk == labels_btL) * valid_btL) / jnp.maximum(valid_btL.sum(), 1)

        diff_acc = float(jnp.abs(acc - acc_bulk))
        diff_ce = float(jnp.abs(pi_ce - pi_ce_bulk))
        if diff_acc > 1e-4 or diff_ce > 1e-4:
            print(f"[eval:TF][paranoid] prefix-only vs bulk differ: Δacc={diff_acc:.6g}, Δpi_ce={diff_ce:.6g} "
                  f"(using prefix-only results)")

    # roll visuals with your sampler in TF mode (for the same batch)
    sampler_conf = SamplerConfig(
        k_max=cfg.k_max, schedule=("finest" if cfg.schedule=="finest" else "shortcut"),
        d=(cfg.d if cfg.schedule=="shortcut" else None),
        start_mode="pure", rollout="teacher_forced",
        horizon=cfg.horizon, ctx_length=cfg.ctx_length,
        ctx_signal_tau=cfg.ctx_signal_tau, match_ctx_tau=cfg.match_ctx_tau,
        rng_key=jax.random.PRNGKey(4242), mae_eval_key=env["mae_eval_key"],
        H=cfg.H, W=cfg.W, C=cfg.C, patch=cfg.patch,
        n_spatial=n_spatial, packing_factor=cfg.packing_factor,
    )
    pred_frames, floor_frames, gt_frames = sample_video(
        encoder=enc, decoder=dec, dynamics=dyn,
        enc_vars=enc_vars, dec_vars=dec_vars, dyn_vars=dyn_vars,
        frames=frames, actions=actions, config=sampler_conf,
    )

    # Save a grid video (GT|Floor|Pred)
    Ttot = cfg.ctx_length + cfg.horizon
    grid_frames = []
    for t in range(Ttot):
        trip_list = [_stack_wide(_to_uint8(gt_frames[b, t]),
                                 _to_uint8(floor_frames[b, t]),
                                 _to_uint8(pred_frames[b, t]))
                     for b in range(cfg.B)]
        grid_frames.append(_tile_videos(trip_list, ncols=min(2, cfg.B)))
    out_mp4 = out_dir / "tf_frames_grid.mp4"
    with imageio.get_writer(out_mp4, fps=25, codec="libx264", quality=8) as w:
        for fr in grid_frames:
            w.append_data(fr)
    print(f"[eval:TF] wrote {out_mp4}")

    # Save a few example strips
    # For strip plots, project L-step predictions to per-frame at offset=0 (current timestep)
    o1 = 0  # use offset 0 to predict current reward r[t]
    o1 = min(o1, cfg.L - 1)

    # Full-length (B, T)
    pred_actions_t = pred_top1[:, :, o1]
    pred_rewards_t = exp_reward[:, :, o1]
    # Window to the eval horizon [ctx : ctx+h)
    pred_actions_h = pred_actions_t[:, cfg.ctx_length : cfg.ctx_length + cfg.horizon]  # (B, h)
    pred_rewards_h = pred_rewards_t[:, cfg.ctx_length : cfg.ctx_length + cfg.horizon]  # (B, h)


    # --- Align GT to CURRENT timestep for fair comparison/visuals ---
    # predictions at each column i correspond to a_{ctx+i}, r_{ctx+i} (current timestep)
    gt_actions_h = actions[:, cfg.ctx_length : cfg.ctx_length + cfg.horizon]  # (B, horizon)
    gt_rewards_h = rewards[:, cfg.ctx_length : cfg.ctx_length + cfg.horizon]  # (B, horizon)

    # Strips: use horizon-length GT aligned to current timestep
    for ei in range(min(cfg.max_examples_to_plot, cfg.B)):
        fig_path = out_dir / f"tf_strip_b{ei}.png"
        _save_strip(
            fig_path, np.asarray(gt_frames), cfg.ctx_length, cfg.horizon,
            gt_actions_bt=np.asarray(gt_actions_h),   # (B, horizon)
            pred_actions_bt=np.asarray(pred_actions_h),    # (B, horizon)
            gt_rewards_bt=np.asarray(gt_rewards_h),   # (B, horizon)
            pred_rewards_bt=np.asarray(pred_rewards_h),    # (B, horizon)
            title=f"Teacher-Forced: example {ei}",
            b_index=ei,
        )
        print(f"[eval:TF] wrote {fig_path}")

    # Reward line plots (GT vs Pred) over the horizon, aligned to current timestep
    for ei in range(min(cfg.max_examples_to_plot, cfg.B)):
        gt_line = np.asarray(gt_rewards_h[ei])
        pred_line = np.asarray(pred_rewards_h[ei])
        plot_path = out_dir / f"tf_reward_line_b{ei}.png"
        _save_reward_lineplot(plot_path, gt_line, pred_line, title=f"TF reward curve (b={ei})")


    # Metrics CSV
    metrics_csv = out_dir / "tf_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pi_ce", "pi_acc_top1", "rw_ce"])
        w.writerow([float(pi_ce), float(acc), float(rw_ce)])
    print(f"[eval:TF] pi_ce={float(pi_ce):.4f}  acc@1={float(acc):.4f}  rw_ce={float(rw_ce):.4f}")
    return dict(pi_ce=float(pi_ce), acc_top1=float(acc), rw_ce=float(rw_ce))

# ---------------------------
# Autoregressive head eval (log predicted rewards; compare actions)
# ---------------------------

def eval_autoregressive(cfg: EvalConfig, env, out_dir: Path):
    enc, dec, dyn = env["encoder"], env["decoder"], env["dynamics"]
    task, pi_head, rw_head = env["task_embedder"], env["policy_head"], env["reward_head"]
    enc_vars, dec_vars, dyn_vars = env["enc_vars"], env["dec_vars"], env["dyn_vars"]
    task_vars, pi_vars, rew_vars = env["task_vars"], env["pi_vars"], env["rew_vars"]
    n_spatial = env["n_spatial"]

    rng = jax.random.PRNGKey(999)
    _, (frames, actions, rewards) = env["next_batch"](rng)

    # Rollout frames autoregressively (using GT actions)
    sampler_conf = SamplerConfig(
        k_max=cfg.k_max, schedule=("finest" if cfg.schedule=="finest" else "shortcut"),
        d=(cfg.d if cfg.schedule=="shortcut" else None),
        start_mode="pure", rollout="autoregressive",
        horizon=cfg.horizon, ctx_length=cfg.ctx_length,
        ctx_signal_tau=cfg.ctx_signal_tau, match_ctx_tau=cfg.match_ctx_tau,
        rng_key=jax.random.PRNGKey(101010), mae_eval_key=env["mae_eval_key"],
        H=cfg.H, W=cfg.W, C=cfg.C, patch=cfg.patch,
        n_spatial=n_spatial, packing_factor=cfg.packing_factor,
    )
    pred_frames, floor_frames, gt_frames = sample_video(
        encoder=enc, decoder=dec, dynamics=dyn,
        enc_vars=enc_vars, dec_vars=dec_vars, dyn_vars=dyn_vars,
        frames=frames, actions=actions, config=sampler_conf,
    )

    # We want heads' predictions per generated step. We'll compute h on the final τ for each future step.
    # Re-encode context, and for each step t, run a single-step clean call with z_clean_pred (from sampler)
    patches = temporal_patchify(frames, cfg.patch)
    z_btLd, _ = enc.apply(enc_vars, patches, rngs={"mae": env["mae_eval_key"]}, deterministic=True)
    z_all = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=cfg.packing_factor)
    z_ctx_clean = z_all[:, :cfg.ctx_length, :, :]

    # Extract the predicted latents we used to decode (ctx + preds already in pred_frames, but we recompute h)
    # For head eval, we only need h at each predicted step.
    emax = jnp.log2(cfg.k_max).astype(jnp.int32)
    if cfg.schedule == "finest":
        d_used = 1.0 / float(cfg.k_max)
        e = int(round(np.log2(1.0 / d_used)))
    else:
        assert cfg.d is not None, "shortcut schedule requires d"
        d_used = float(cfg.d)
        e = int(round(np.log2(1.0 / d_used)))

    # Build actions windows and compute h per future step
    B = cfg.B
    Lh = cfg.horizon
    pred_act_top1 = []
    pred_rew_real = []

    # For h inference, we approximate σ as the final τ for that step; use z_tilde=z_clean_pred (pure/clean).
    # So sigma_idx = k_max - 1; step_idx = e (consistent with training "finest"/shortcut bin).
    step_idx_ = jnp.full((B, cfg.ctx_length + 1), e, dtype=jnp.int32)
    sigma_idx_ = jnp.full((B, cfg.ctx_length + 1), cfg.k_max - 1, dtype=jnp.int32)

    # Build agent tokens once for max length, slice per step
    dummy_task_ids = jnp.zeros((B,), dtype=jnp.int32)
    agents_full = task.apply(task_vars, dummy_task_ids, B, cfg.ctx_length + 1)

    # We'll also need the predicted latent at each step; we can re-encode pred_frames to get z for viz-aligned heads.
    # (This is slightly heavier but simple and faithful to what was decoded.)
    pred_patches = temporal_patchify(pred_frames, cfg.patch)
    pred_btLd, _ = enc.apply(enc_vars, pred_patches, rngs={"mae": env["mae_eval_key"]}, deterministic=True)
    pred_latents_full = pack_bottleneck_to_spatial(pred_btLd, n_spatial=n_spatial, k=cfg.packing_factor)
    pred_future_latents = pred_latents_full[:, cfg.ctx_length:, :, :]  # (B, horizon, S, D)

    for t in range(Lh):
        action_curr = actions[:, cfg.ctx_length + t: cfg.ctx_length + t + 1]  # (B,1)
        z_clean_pred = pred_future_latents[:, t:t+1, :, :]                    # (B,1,S,D)

        z_seq = jnp.concatenate([z_ctx_clean, z_clean_pred], axis=1)                  # (B, ctx+1, S, D)
        actions_seq = jnp.concatenate([actions[:, :cfg.ctx_length], action_curr], 1)  # (B, ctx+1)

        z1_hat_seq, h_seq = dyn.apply(
            dyn_vars, actions_seq, step_idx_, sigma_idx_, z_seq,
            agent_tokens=agents_full[:, :cfg.ctx_length+1], deterministic=True
        )
        h_pooled = jnp.mean(h_seq, axis=2)[:, -1]  # (B, D)

        # heads
        pi_logits = pi_head.apply(pi_vars, h_pooled[:, None, :], deterministic=True)[:, 0, 0, :]  # (B,A)
        logp = jax.nn.log_softmax(pi_logits, axis=-1)
        top1 = jnp.argmax(logp, axis=-1)  # (B,)
        pred_act_top1.append(top1)

        rw_logits, centers_log = rw_head.apply(rew_vars, h_pooled[:, None, :], deterministic=True)  # (B,1,1,K)
        probs = jax.nn.softmax(rw_logits[:, 0, 0, :], axis=-1)  # (B,K)
        exp_symlog = jnp.sum(probs * centers_log[None, :], axis=-1)
        exp_reward = _symexp(exp_symlog)                        # (B,)
        pred_rew_real.append(exp_reward)

        # advance context for next step (AR)
        z_ctx_clean = jnp.concatenate([z_ctx_clean, z_clean_pred], axis=1)[:, -cfg.ctx_length:, :, :]

    pred_act_top1 = jnp.stack(pred_act_top1, axis=1)        # (B, horizon)
    pred_rew_real = jnp.stack(pred_rew_real, axis=1)        # (B, horizon)

    # Save grid video
    grid_frames = []
    Ttot = cfg.ctx_length + cfg.horizon
    for t in range(Ttot):
        trip_list = [_stack_wide(_to_uint8(gt_frames[b, t]),
                                 _to_uint8(floor_frames[b, t]),
                                 _to_uint8(pred_frames[b, t]))
                     for b in range(cfg.B)]
        grid_frames.append(_tile_videos(trip_list, ncols=min(2, cfg.B)))
    out_mp4 = out_dir / "ar_frames_grid.mp4"
    with imageio.get_writer(out_mp4, fps=25, codec="libx264", quality=8) as w:
        for fr in grid_frames:
            w.append_data(fr)
    print(f"[eval:AR] wrote {out_mp4}")

    # Strip plots (predicted rewards + actions vs GT actions)
    Ttot = cfg.ctx_length + cfg.horizon   
    # Accuracy against NEXT actions: a_{t+1}
    gt_next_actions_p1 = actions[:, cfg.ctx_length+1 : cfg.ctx_length + cfg.horizon + 1]  # (B, horizon)
    # Our per-step preds (B, horizon) correspond to next; the very last pred has no GT next-next label
    pred_for_acc = pred_act_top1[:, :-1]                 # (B, horizon-1)
    gt_for_acc   = gt_next_actions_p1[:, :-1]            # (B, horizon-1)
    acc = jnp.mean((pred_for_acc == gt_for_acc).astype(jnp.float32))

    # Save strips using +1-shifted GT (so column i shows GT/Pred for a_{ctx+i+1})
    for ei in range(min(cfg.max_examples_to_plot, cfg.B)):
        fig_path = out_dir / f"ar_strip_b{ei}.png"
        _save_strip(
            fig_path, np.asarray(gt_frames), cfg.ctx_length, cfg.horizon,
            gt_actions_bt=np.asarray(gt_next_actions_p1),  # (B, horizon) ← shifted
            pred_actions_bt=np.asarray(pred_act_top1),     # (B, horizon)
            gt_rewards_bt=None,                            # keep None per your preference
            pred_rewards_bt=np.asarray(pred_rew_real),     # (B, horizon)
            title=f"Autoregressive: example {ei}",
            b_index=ei,
        )
        print(f"[eval:AR] wrote {fig_path}")

    # Reward line plots (we DO have GT; show it for clarity)
    # pred_rew_real[t] predicts r[ctx_length + t] (current reward at timestep ctx_length + t)
    gt_line_all = np.asarray(rewards[:, cfg.ctx_length : cfg.ctx_length + cfg.horizon])
    for ei in range(min(cfg.max_examples_to_plot, cfg.B)):
        gt_line = gt_line_all[ei]
        pred_line = np.asarray(pred_rew_real[ei])
        plot_path = out_dir / f"ar_reward_line_b{ei}.png"
        _save_reward_lineplot(plot_path, gt_line, pred_line, title=f"AR reward curve (b={ei})")

    # Metrics CSV (next-action, horizon-1 due to boundary)
    metrics_csv = out_dir / "ar_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["act_acc_top1_horizon_minus1"])
        w.writerow([float(acc)])
    print(f"[eval:AR] act_acc_top1 over horizon-1={cfg.horizon-1}: {float(acc):.4f}")
    return dict(act_acc_top1=float(acc))

# ---------------------------
# Main
# ---------------------------

def main(cfg: EvalConfig):
    out_dir = _ensure_dir(Path(cfg.out_dir))
    print("Eval config:\n  " + "\n  ".join([f"{k}={v}" for k,v in asdict(cfg).items()]))

    env = init_models_and_restore(cfg)

    # Create separate subdirs
    tf_dir = _ensure_dir(out_dir / "teacher_forced")
    ar_dir = _ensure_dir(out_dir / "autoregressive")

    tf_metrics = eval_teacher_forced(cfg, env, tf_dir)
    ar_metrics = eval_autoregressive(cfg, env, ar_dir)

    # summary
    (out_dir / "SUMMARY.txt").write_text(
        "\n".join([
            "Teacher-Forced:",
            f"  pi_ce = {tf_metrics['pi_ce']:.4f}",
            f"  acc@1 = {tf_metrics['acc_top1']:.4f}",
            f"  rw_ce = {tf_metrics['rw_ce']:.4f}",
            "",
            "Autoregressive:",
            f"  act_acc@1(horizon) = {ar_metrics['act_acc_top1']:.4f}",
        ])
    )
    print(f"[eval] Wrote summary to {out_dir/'SUMMARY.txt'}")

if __name__ == "__main__":
    # EXAMPLE PATHS — adjust to your environment
    cfg = EvalConfig(
        run_ckpt_dir="/vast/projects/dineshj/lab/hued/tiny_dreamer_4/logs/train_bc_rew_flippedrew/checkpoints",
        tokenizer_ckpt="/vast/projects/dineshj/lab/hued/tiny_dreamer_4/logs/pretrained_mae/checkpoints",
        out_dir="/vast/projects/dineshj/lab/hued/tiny_dreamer_4/logs/eval_bc_rew_heads_shortcut_flippedrew",
        B=8, T=64, H=32, W=32, C=3,
        action_dim=4,
        ctx_length=32, horizon=16,
        schedule="shortcut", d=1/4,           # or schedule="shortcut", d=0.25
        ctx_signal_tau=1.0, match_ctx_tau=False,
        max_examples_to_plot=4,
        paranoid_no_leak=False,
    )
    main(cfg)
