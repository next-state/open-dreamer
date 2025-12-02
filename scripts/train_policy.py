"""
JIT-friendly policy/value training using imagination rollouts.

High-level outline (from the docstring plan):

- Trajectory Generation
    - Roll out the policy in latent space starting from a ground-truth context.
    - Unroll π(a|s) from s0 for `horizon` steps, creating latent states s1…sT.
    - Collect policy actions a1…aT and hidden states h0…hT.

- Reward / Value annotation
    - Use the reward head on h1…hT to get r1…rT.
    - Use the value head on h0…hT to get V0…VT.
    - Compute TD-λ returns G0…G{T-1} using V1…VT and r1…rT (with bootstrap VT).

- Value / Policy updates
    - Train V_head on (s0…s{T-1}) to regress G0…G{T-1}.
    - Train policy head on (s0…s{T-1}, a1…aT, G0…G{T-1}, V0…V{T-1}) using PMPO.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any
from functools import partial
from tqdm import tqdm
import math
import time

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax import struct
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import wandb

from dreamer.models import (
    Encoder,
    Decoder,
    Dynamics,
    TaskEmbedder,
    PolicyHeadMTP,
    RewardHeadMTP,
    ValueHead,
)
from dreamer.data import make_iterator, make_env_reset_fn, make_env_step_fn, DatasetConfig
from dreamer.utils import (
    temporal_patchify,
    pack_bottleneck_to_spatial,
    with_params,
    make_state,
    make_manager,
    try_restore,
    maybe_save,
    pack_mae_params,
)
from dreamer.imagination import (
    ImaginationConfig,
    DenoiseSchedule,
    _build_static_schedule,
    imagine_rollouts_core,
)
from dreamer.logging import MetricLogger


# ---------------------------
# Config
# ---------------------------


@dataclass(frozen=True)
class RLConfig:
    # IO / ckpt
    run_name: str
    bc_rew_ckpt: str  # checkpoint from train_bc_rew_heads.py
    ckpt_max_to_keep: int = 2
    ckpt_save_every: int = 10_000

    # wandb config
    use_wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str | None = None

    # dataset config
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    action_dim: int = 4

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
    n_kv_heads: int = 2
    qk_norm_type: str | None = None
    packing_factor: int = 2
    n_register: int = 4
    n_agent: int = 1
    agent_space_mode: str = "wm_agent"

    # schedule
    k_max: int = 8

    # train
    max_steps: int = 1_000_000_000
    log_every: int = 5_000
    lr: float = 3e-4

    # eval media toggle
    write_video_every: int = 10_000
    visualize_every: int = 25_000

    # RL-specific
    L: int = 2
    num_reward_bins: int = 101
    reward_log_low: float = -3.0
    reward_log_high: float = 3.0
    num_value_bins: int = 101
    n_tasks: int = 128
    use_task_ids: bool = True

    # RL hyperparameters
    gamma: float = 0.997
    lambda_: float = 0.95
    horizon: int = 32
    context_length: int = 16
    imagination_d: float = 1.0 / 4
    alpha: float = 0.5
    beta: float = 0.3

    # Evaluation
    eval_every: int = 50_000
    eval_episodes: int = 4
    eval_horizon: int = 32
    eval_batch_size: int = 4
    max_eval_examples_to_plot: int = 4


# ---------------------------
# Small helpers
# ---------------------------


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_uint8(img_f32: np.ndarray) -> np.ndarray:
    """Convert float32 images in [0,1] to uint8 [0,255]."""
    return np.asarray(np.clip(np.asarray(img_f32) * 255.0, 0, 255), dtype=np.uint8)


def _tile_frames_grid(
    frames_b_hwc: np.ndarray,
    *,
    ncols: int = 2,
    pad_color: int = 0,
) -> np.ndarray:
    """
    Tile a batch of frames (B, H, W, C) into a grid image (H*nrows, W*ncols, C).
    """
    B, H, W, C = frames_b_hwc.shape
    nrows = math.ceil(B / ncols)
    total = nrows * ncols

    frames_list = [frames_b_hwc[b] for b in range(B)]
    if total > B:
        blank = np.full((H, W, C), pad_color, dtype=frames_b_hwc.dtype)
        frames_list += [blank] * (total - B)

    rows = []
    idx = 0
    for _ in range(nrows):
        row_imgs = frames_list[idx : idx + ncols]
        idx += ncols
        rows.append(np.concatenate(row_imgs, axis=1))
    grid = np.concatenate(rows, axis=0)
    return grid


def _save_real_env_grid_video(
    out_path: Path,
    frames_b_t_hwc: np.ndarray,
    *,
    fps: int = 25,
) -> None:
    """
    Save a grid MP4 summarizing all episodes in a real-env eval rollout.

    frames_b_t_hwc: (B, T, H, W, C) float32 in [0,1] or uint8.
    """
    frames_np = _to_uint8(frames_b_t_hwc)
    B, T = frames_np.shape[:2]
    ncols = min(2, B)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as w:
        for t in range(T):
            grid_frame = _tile_frames_grid(frames_np[:, t], ncols=ncols)
            w.append_data(grid_frame)


def _save_real_env_strip(
    fig_path: Path,
    frames_b_t_hwc: np.ndarray,
    actions_bt: np.ndarray,
    rewards_bt: np.ndarray,
    *,
    title: str,
    b_index: int = 0,
    max_steps: int | None = None,
) -> None:
    """
    Save a strip visualization for a single real-env episode.

    - frames_b_t_hwc: (B, T, H, W, C)
    - actions_bt:     (B, T)
    - rewards_bt:     (B, T) with rewards_bt[:, 0] typically NaN (dummy)
    """
    frames = _to_uint8(frames_b_t_hwc)
    actions = np.asarray(actions_bt)
    rewards = np.asarray(rewards_bt)

    B, T = frames.shape[:2]
    if T <= 1:
        return  # nothing meaningful to plot

    b = int(np.clip(b_index, 0, B - 1))

    # Skip t=0 (dummy reward); each column corresponds to entering frame t.
    t_indices = np.arange(1, T)
    if max_steps is not None:
        t_indices = t_indices[:max_steps]
    hor = int(len(t_indices))
    if hor == 0:
        return

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        hor,
        figsize=(hor * 2.2, 4.0),
        constrained_layout=True,
    )
    if hor == 1:
        # When hor==1, matplotlib returns 1D axes; normalize to 2D for simplicity.
        axes = np.array([[axes[0]], [axes[1]]])

    fig.suptitle(title, fontsize=12)

    # Row 0: images
    for i, t in enumerate(t_indices):
        ax = axes[0, i]
        ax.imshow(frames[b, t])
        ax.axis("off")
        ax.set_title(f"t={t}", fontsize=9)

    # Row 1: annotations (action leading into this frame and reward on entering it)
    for i, t in enumerate(t_indices):
        a_t = int(actions[b, t])
        r_t = float(rewards[b, t])
        txt = f"act={a_t}\nrew={r_t:.3f}"

        ax = axes[1, i]
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            txt,
            ha="center",
            va="center",
            fontsize=8,
        )

    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


@partial(
    jax.jit,
    static_argnames=("context_length", "H", "W", "C"),
)
def sample_contexts(
    videos: jnp.ndarray,  # (B, T, H, W, C)
    actions: jnp.ndarray,  # (B, T)
    rewards: jnp.ndarray,  # (B, T)
    rng: jnp.ndarray,
    context_length: int,
    H: int,
    W: int,
    C: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample valid contexts from videos.

    For each video in the batch, samples a random start index from
    [0, T - context_length + 1) and extracts a subsequence of length
    `context_length`.
    """
    del H, W, C  # static, used for shape only
    B, T = videos.shape[:2]

    max_start = T - context_length + 1

    start_indices = jax.random.randint(
        rng, (B,), minval=0, maxval=max_start, dtype=jnp.int32
    )

    def extract_context_frames(video_seq, start_idx):
        return jax.lax.dynamic_slice(
            video_seq,
            start_indices=(start_idx, 0, 0, 0),
            slice_sizes=(context_length, videos.shape[2], videos.shape[3], videos.shape[4]),
        )

    def extract_context_1d(seq, start_idx):
        return jax.lax.dynamic_slice(
            seq,
            start_indices=(start_idx,),
            slice_sizes=(context_length,),
        )

    context_frames = jax.vmap(extract_context_frames, in_axes=(0, 0))(videos, start_indices)
    context_actions = jax.vmap(extract_context_1d, in_axes=(0, 0))(actions, start_indices)
    context_rewards = jax.vmap(extract_context_1d, in_axes=(0, 0))(rewards, start_indices)

    return context_frames, context_actions, context_rewards


# ---------------------------
# Checkpoint loading helpers
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
    """
    Load pretrained encoder/decoder from tokenizer checkpoint.
    """
    meta_mngr = make_manager(tokenizer_ckpt_dir, item_names=("meta",))
    latest = meta_mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No tokenizer checkpoint found in {tokenizer_ckpt_dir}")
    restored_meta = meta_mngr.restore(
        latest,
        args=ocp.args.Composite(meta=ocp.args.JsonRestore()),
    )
    meta = restored_meta.meta
    enc_kwargs = meta["enc_kwargs"]
    n_lat, d_b = enc_kwargs["n_latents"], enc_kwargs["d_bottleneck"]

    rng_e1, rng_d1 = jax.random.split(rng)
    B, T = sample_patches_btnd.shape[:2]
    fake_z = jnp.zeros((B, T, n_lat, d_b), dtype=jnp.float32)
    dec_vars = decoder.init(
        {"params": rng_d1, "dropout": rng_d1},
        fake_z,
        deterministic=True,
    )

    packed_example = pack_mae_params(enc_vars, dec_vars)
    tx_dummy = optax.adamw(1e-4)
    opt_state_example = tx_dummy.init(packed_example)
    state_example = make_state(packed_example, opt_state_example, rng_e1, step=0)
    abstract_state = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, state_example
    )

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


def load_bc_rew_checkpoint(
    bc_rew_ckpt_dir: str,
    *,
    rng: jnp.ndarray,
    dynamics: Dynamics,
    task_embedder: TaskEmbedder,
    policy_head_bc: PolicyHeadMTP,
    reward_head: RewardHeadMTP,
    dyn_vars,
    task_vars,
    pi_bc_vars,
    rew_vars,
    sample_actions: jnp.ndarray,
    sample_z1: jnp.ndarray,
):
    """
    Load pretrained dynamics, task_embedder, BC policy head, and reward head.
    """
    params_example = {
        "dyn": dyn_vars["params"],
        "task": task_vars["params"],
        "pi": pi_bc_vars["params"],
        "rew": rew_vars["params"],
    }
    tx_dummy = optax.adam(1e-3)
    opt_state_example = tx_dummy.init(params_example)
    state_example = make_state(params_example, opt_state_example, rng, step=0)

    mngr = make_manager(bc_rew_ckpt_dir, item_names=("state", "meta"))
    restored = try_restore(mngr, state_example, meta_example={})
    if restored is None:
        raise FileNotFoundError(f"No BC/rew checkpoint found in {bc_rew_ckpt_dir}")

    latest_step, r = restored
    loaded_params = r.state["params"]

    dyn_params = loaded_params["dyn"]
    task_params = loaded_params["task"]
    pi_bc_params = loaded_params["pi"]
    rew_params = loaded_params["rew"]

    dyn_vars_loaded = with_params(dyn_vars, dyn_params)
    task_vars_loaded = with_params(task_vars, task_params)
    pi_bc_vars_loaded = with_params(pi_bc_vars, pi_bc_params)
    rew_vars_loaded = with_params(rew_vars, rew_params)

    meta = r.meta if hasattr(r, "meta") and r.meta is not None else {}

    print(
        f"[bc_rew] Restored dynamics/task/policy_bc/reward "
        f"from {bc_rew_ckpt_dir} (step {latest_step})"
    )
    return dyn_vars_loaded, task_vars_loaded, pi_bc_vars_loaded, rew_vars_loaded, meta


# ---------------------------
# Training state dataclass
# ---------------------------


@dataclass
class TrainState:
    """Container for all training-related state (models, variables, optimizer, etc.)."""

    # Frozen models (loaded from checkpoints, not trained)
    encoder: Encoder
    decoder: Decoder
    dynamics: Dynamics
    task_embedder: TaskEmbedder
    policy_head_bc: PolicyHeadMTP
    reward_head: RewardHeadMTP

    # Trainable models
    policy_head: PolicyHeadMTP
    value_head: ValueHead

    # vars/collections (frozen)
    enc_vars: dict
    dec_vars: dict
    dyn_vars: dict
    task_vars: dict
    pi_bc_vars: dict
    rew_vars: dict

    # vars/collections (trainable)
    pi_vars: dict
    val_vars: dict

    # params for optimizer (pi/val only)
    params: dict
    enc_kwargs: dict
    dec_kwargs: dict
    dyn_kwargs: dict
    tx: optax.Transform
    opt_state: optax.OptState
    mae_eval_key: jnp.ndarray


@struct.dataclass
class EpisodeResult:
    """JAX-friendly container for a batch of evaluation rollouts."""

    frames: jnp.ndarray   # (B, horizon+1, H, W, C)
    actions: jnp.ndarray  # (B, horizon+1)
    rewards: jnp.ndarray  # (B, horizon+1)
    returns: jnp.ndarray  # (B,)


# ---------------------------
# Model initialization
# ---------------------------


def initialize_models(
    cfg: RLConfig,
    frames_init: jnp.ndarray,
    actions_init: jnp.ndarray,
) -> TrainState:
    """
    Initialize all models and load pretrained checkpoints.
    """
    patch = cfg.patch
    num_patches = (cfg.dataset.H // patch) * (cfg.dataset.W // patch)
    D_patch = patch * patch * cfg.dataset.C
    k_max = cfg.k_max

    enc_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        depth=cfg.enc_depth,
        dropout_rate=0.0,
        qk_norm_type=cfg.qk_norm_type,
        d_bottleneck=cfg.enc_d_bottleneck,
        mae_p_min=0.0,
        mae_p_max=0.0,
        time_every=4,
    )
    dec_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        depth=cfg.dec_depth,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        d_patch=D_patch,
        dropout_rate=0.0,
        qk_norm_type=cfg.qk_norm_type,
        mlp_ratio=4.0,
        time_every=4,
    )
    n_spatial = cfg.enc_n_latents // cfg.packing_factor
    dyn_kwargs = dict(
        d_model=cfg.d_model_dyn,
        d_bottleneck=cfg.enc_d_bottleneck,
        d_spatial=cfg.enc_d_bottleneck * cfg.packing_factor,
        n_spatial=n_spatial,
        n_register=cfg.n_register,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        depth=cfg.dyn_depth,
        dropout_rate=0.0,
        qk_norm_type=cfg.qk_norm_type,
        k_max=k_max,
        time_every=4,
        n_agent=cfg.n_agent,
    )

    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)
    dynamics = Dynamics(**dyn_kwargs)
    task_embedder = TaskEmbedder(
        d_model=cfg.d_model_dyn,
        use_ids=cfg.use_task_ids,
        n_tasks=cfg.n_tasks,
    )
    policy_head_bc = PolicyHeadMTP(
        d_model=cfg.d_model_dyn,
        action_dim=cfg.action_dim,
        L=cfg.L,
    )
    reward_head = RewardHeadMTP(
        d_model=cfg.d_model_dyn,
        L=cfg.L,
        num_bins=cfg.num_reward_bins,
        log_low=cfg.reward_log_low,
        log_high=cfg.reward_log_high,
    )

    policy_head = PolicyHeadMTP(
        d_model=cfg.d_model_dyn,
        action_dim=cfg.action_dim,
        L=cfg.L,
    )
    value_head = ValueHead(
        d_model=cfg.d_model_dyn,
        num_bins=cfg.num_value_bins,
    )

    rng = jax.random.PRNGKey(0)
    patches_btnd = temporal_patchify(frames_init, patch)
    enc_vars = encoder.init(
        {"params": rng, "mae": rng, "dropout": rng},
        patches_btnd,
        deterministic=True,
    )
    fake_z = jnp.zeros(
        (cfg.dataset.B, cfg.dataset.T, cfg.enc_n_latents, cfg.enc_d_bottleneck),
        dtype=jnp.float32,
    )
    dec_vars = decoder.init(
        {"params": rng, "dropout": rng},
        fake_z,
        deterministic=True,
    )

    # Read BC/rew meta to find tokenizer checkpoint directory.
    bc_rew_mngr = make_manager(cfg.bc_rew_ckpt, item_names=("meta",))
    bc_rew_latest = bc_rew_mngr.latest_step()
    if bc_rew_latest is None:
        raise FileNotFoundError(f"No BC/rew checkpoint found in {cfg.bc_rew_ckpt}")
    bc_rew_meta_restored = bc_rew_mngr.restore(
        bc_rew_latest,
        args=ocp.args.Composite(meta=ocp.args.JsonRestore()),
    )
    bc_rew_meta = bc_rew_meta_restored.meta
    tokenizer_ckpt = bc_rew_meta.get("tokenizer_ckpt_dir") or bc_rew_meta.get(
        "cfg", {}
    ).get("tokenizer_ckpt")
    if tokenizer_ckpt is None:
        raise ValueError(
            f"Could not find tokenizer_ckpt in BC/rew checkpoint meta: {bc_rew_meta}"
        )

    enc_vars, dec_vars, _ = load_pretrained_tokenizer(
        tokenizer_ckpt,
        rng=rng,
        encoder=encoder,
        decoder=decoder,
        enc_vars=enc_vars,
        dec_vars=dec_vars,
        sample_patches_btnd=patches_btnd,
    )

    mae_eval_key = jax.random.PRNGKey(777)
    z_btLd, _ = encoder.apply(
        enc_vars,
        patches_btnd,
        rngs={"mae": mae_eval_key},
        deterministic=True,
    )
    z1 = pack_bottleneck_to_spatial(
        z_btLd,
        n_spatial=n_spatial,
        k=cfg.packing_factor,
    )
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jnp.full((cfg.dataset.B, cfg.dataset.T), emax, dtype=jnp.int32)
    sigma_idx = jnp.full((cfg.dataset.B, cfg.dataset.T), k_max - 1, dtype=jnp.int32)
    dyn_vars = dynamics.init(
        {"params": rng, "dropout": rng},
        actions_init,
        step_idx,
        sigma_idx,
        z1,
    )

    rng_task, rng_pi_bc, rng_rw = jax.random.split(jax.random.PRNGKey(1), 3)
    dummy_task_ids = jnp.zeros((cfg.dataset.B,), dtype=jnp.int32)
    task_vars = task_embedder.init(
        {"params": rng_task},
        dummy_task_ids,
        cfg.dataset.B,
        cfg.dataset.T,
    )

    fake_h = jnp.zeros(
        (cfg.dataset.B, cfg.dataset.T, cfg.d_model_dyn),
        dtype=jnp.float32,
    )
    pi_bc_vars = policy_head_bc.init(
        {"params": rng_pi_bc, "dropout": rng_pi_bc},
        fake_h,
        deterministic=True,
    )
    rew_vars = reward_head.init(
        {"params": rng_rw, "dropout": rng_rw},
        fake_h,
        deterministic=True,
    )

    dyn_vars, task_vars, pi_bc_vars, rew_vars, _ = load_bc_rew_checkpoint(
        cfg.bc_rew_ckpt,
        rng=rng,
        dynamics=dynamics,
        task_embedder=task_embedder,
        policy_head_bc=policy_head_bc,
        reward_head=reward_head,
        dyn_vars=dyn_vars,
        task_vars=task_vars,
        pi_bc_vars=pi_bc_vars,
        rew_vars=rew_vars,
        sample_actions=actions_init,
        sample_z1=z1,
    )

    rng_pi, rng_val = jax.random.split(jax.random.PRNGKey(2), 2)
    pi_vars = policy_head.init(
        {"params": rng_pi, "dropout": rng_pi},
        fake_h,
        deterministic=True,
    )
    val_vars = value_head.init(
        {"params": rng_val, "dropout": rng_val},
        fake_h,
        deterministic=True,
    )

    params = {
        "pi": pi_vars["params"],
        "val": val_vars["params"],
    }

    tx = optax.adam(cfg.lr)
    opt_state = tx.init(params)

    return TrainState(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        task_embedder=task_embedder,
        policy_head_bc=policy_head_bc,
        reward_head=reward_head,
        policy_head=policy_head,
        value_head=value_head,
        enc_vars=enc_vars,
        dec_vars=dec_vars,
        dyn_vars=dyn_vars,
        task_vars=task_vars,
        pi_bc_vars=pi_bc_vars,
        rew_vars=rew_vars,
        pi_vars=pi_vars,
        val_vars=val_vars,
        params=params,
        enc_kwargs=enc_kwargs,
        dec_kwargs=dec_kwargs,
        dyn_kwargs=dyn_kwargs,
        tx=tx,
        opt_state=opt_state,
        mae_eval_key=mae_eval_key,
    )


# ---------------------------
# Meta for RL checkpoints
# ---------------------------


def make_rl_meta(
    *,
    enc_kwargs: dict,
    dec_kwargs: dict,
    dynamics_kwargs: dict,
    H: int,
    W: int,
    C: int,
    patch: int,
    k_max: int,
    packing_factor: int,
    n_spatial: int,
    bc_rew_ckpt_dir: str | None = None,
    cfg: Dict[str, Any] | None = None,
):
    return {
        "enc_kwargs": enc_kwargs,
        "dec_kwargs": dec_kwargs,
        "dynamics_kwargs": dynamics_kwargs,
        "H": H,
        "W": W,
        "C": C,
        "patch": patch,
        "k_max": k_max,
        "packing_factor": packing_factor,
        "n_spatial": n_spatial,
        "bc_rew_ckpt_dir": bc_rew_ckpt_dir,
        "cfg": cfg or {},
    }


# ---------------------------
# Encoding / dynamics helpers
# ---------------------------


def encode_frames_to_spatial(
    frames: jnp.ndarray,
    *,
    encoder: Encoder,
    enc_vars: dict,
    mae_eval_key: jnp.ndarray,
    patch: int,
    n_spatial: int,
    packing_factor: int,
) -> jnp.ndarray:
    """
    Encode input frames into packed spatial tokens suitable for the dynamics model.

    Args:
        frames: (B, T_ctx, H, W, C) float32 in [0,1]

    Returns:
        z_ctx: (B, T_ctx, n_spatial, d_spatial)
    """
    patches_btnd = temporal_patchify(frames, patch)
    z_btLd, _ = encoder.apply(
        enc_vars,
        patches_btnd,
        rngs={"mae": mae_eval_key},
        deterministic=True,
    )
    z_ctx = pack_bottleneck_to_spatial(
        z_btLd,
        n_spatial=n_spatial,
        k=packing_factor,
    )
    return z_ctx


def compute_hidden_from_context(
    z_ctx: jnp.ndarray,
    actions_ctx: jnp.ndarray,
    *,
    dynamics: Dynamics,
    task_embedder: TaskEmbedder,
    dyn_vars: dict,
    task_vars: dict,
    task_ids: jnp.ndarray,
    step_idx_scalar: int,
    k_max: int,
) -> jnp.ndarray:
    """
    Map a context window in latent space to the current pooled hidden state h_t.

    Args:
        z_ctx:       (B, T_ctx, n_spatial, d_spatial)
        actions_ctx: (B, T_ctx) int32, with a0 at index 0.

    Returns:
        h_t: (B, d_model) pooled agent hidden state at the last context timestep.
    """
    B, T_ctx = z_ctx.shape[:2]

    step_idx_ctx = jnp.full(
        (B, T_ctx),
        jnp.int32(step_idx_scalar),
        dtype=jnp.int32,
    )
    signal_idx_ctx = jnp.full(
        (B, T_ctx),
        jnp.int32(k_max - 1),
        dtype=jnp.int32,
    )

    agent_tokens = task_embedder.apply(
        task_vars,
        task_ids,
        B,
        T_ctx,
    )  # (B, T_ctx, n_agent, d_model)

    _, h_ctx = dynamics.apply(
        dyn_vars,
        actions_ctx,
        step_idx_ctx,
        signal_idx_ctx,
        z_ctx,
        agent_tokens=agent_tokens,
        deterministic=True,
    )  # (B, T_ctx, n_agent, d_model)

    h_last = h_ctx[:, -1, :, :]  # (B, n_agent, d_model)
    h_t = jnp.mean(h_last, axis=1)  # (B, d_model)
    return h_t


def sample_action_from_policy(
    h_t: jnp.ndarray,
    *,
    policy_head: PolicyHeadMTP,
    pi_vars: dict,
    rng: jax.Array | None = None,
    greedy: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample an action from the policy head given the current hidden state.

    Args:
        h_t: (B, d_model)

    Returns:
        actions: (B,) int32
        logits:  (B, A) unnormalized action logits
    """
    del rng  # currently unused (deterministic eval by default)

    h_for_policy = h_t[:, None, :]  # (B, 1, d_model)
    pi_logits = policy_head.apply(
        pi_vars,
        h_for_policy,
        deterministic=True,
    )  # (B, 1, L, A)
    logits_t0 = pi_logits[:, 0, 0, :]  # (B, A)

    if greedy:
        actions = jnp.argmax(logits_t0, axis=-1).astype(jnp.int32)
    else:
        logp = jax.nn.log_softmax(logits_t0, axis=-1)
        rng_sample = jax.random.PRNGKey(0) if rng is None else rng
        actions = jax.random.categorical(rng_sample, logp, axis=-1).astype(jnp.int32)

    return actions, logits_t0


# ---------------------------
# Real-environment evaluation rollout
# ---------------------------


@partial(
    jax.jit,
    static_argnames=(
        "encoder",
        "dynamics",
        "task_embedder",
        "policy_head",
        "env_reset_fn",
        "env_step_fn",
        "horizon",
        "patch",
        "n_spatial",
        "packing_factor",
        "max_context",
    ),
)
def eval_rollout_real_env(
    *,
    encoder: Encoder,
    dynamics: Dynamics,
    task_embedder: TaskEmbedder,
    policy_head: PolicyHeadMTP,
    enc_vars: dict,
    dyn_vars: dict,
    task_vars: dict,
    pi_vars: dict,
    mae_eval_key: jnp.ndarray,
    env_reset_fn,
    env_step_fn,
    task_ids: jnp.ndarray,
    horizon: int,
    patch: int,
    n_spatial: int,
    packing_factor: int,
    max_context: int,
    schedule_step_idx: int,
    k_max: int,
    rng_key: jax.Array,
) -> EpisodeResult:
    """
    Roll the learned policy in the *real* environment for `horizon` steps.

    Uses a growing/sliding context in latent space:
      - Encode observations with the MAE encoder.
      - Feed the last `max_context` (z_ctx, actions_ctx) into dynamics+task_embedder
        to get the current hidden state h_t.
      - Query the policy head to get a_{t+1}, step the env, and repeat.
    """
    # Split RNG for env reset vs rollout (policy sampling).
    rng_env, rng_roll = jax.random.split(rng_key)

    # Reset environment and get initial observation / null action.
    env_state, s0, a0, r0 = env_reset_fn(rng_env)
    B, H, W, C = s0.shape

    # Encode initial observation into spatial tokens (single frame).
    z0 = encode_frames_to_spatial(
        s0[:, None, ...],
        encoder=encoder,
        enc_vars=enc_vars,
        mae_eval_key=mae_eval_key,
        patch=patch,
        n_spatial=n_spatial,
        packing_factor=packing_factor,
    )  # (B, 1, n_spatial, d_spatial)

    # Initialize sliding context by tiling the initial frame and action.
    z_ctx_init = jnp.tile(z0, (1, max_context, 1, 1))  # (B, max_context, n_spatial, d_spatial)
    actions_ctx_init = jnp.tile(a0[:, None], (1, max_context))  # (B, max_context)

    def scan_body(carry, t):
        env_state_t, z_ctx_t, actions_ctx_t, rng_t = carry

        # Current hidden state from context window.
        h_t = compute_hidden_from_context(
            z_ctx_t,
            actions_ctx_t,
            dynamics=dynamics,
            task_embedder=task_embedder,
            dyn_vars=dyn_vars,
            task_vars=task_vars,
            task_ids=task_ids,
            step_idx_scalar=schedule_step_idx,
            k_max=k_max,
        )

        # Policy action (greedy by default).
        rng_t, policy_rng = jax.random.split(rng_t)
        actions_t, _ = sample_action_from_policy(
            h_t,
            policy_head=policy_head,
            pi_vars=pi_vars,
            rng=policy_rng,
            greedy=True,
        )

        # Environment step.
        env_state_next, obs_next, rewards_next, dones_next = env_step_fn(
            env_state_t,
            actions_t,
        )
        del dones_next  # currently unused

        # Encode next observation and update sliding context.
        z_next = encode_frames_to_spatial(
            obs_next[:, None, ...],
            encoder=encoder,
            enc_vars=enc_vars,
            mae_eval_key=mae_eval_key,
            patch=patch,
            n_spatial=n_spatial,
            packing_factor=packing_factor,
        )  # (B, 1, n_spatial, d_spatial)

        z_ctx_next = jnp.concatenate([z_ctx_t, z_next], axis=1)[
            :, -max_context:, :, :
        ]
        actions_ctx_next = jnp.concatenate(
            [actions_ctx_t, actions_t[:, None]],
            axis=1,
        )[:, -max_context:]

        carry_next = (env_state_next, z_ctx_next, actions_ctx_next, rng_t)
        outputs_t = (obs_next, actions_t, rewards_next)
        return carry_next, outputs_t

    init_carry = (env_state, z_ctx_init, actions_ctx_init, rng_roll)
    _, outputs = jax.lax.scan(
        scan_body,
        init_carry,
        jnp.arange(horizon),
    )

    obs_seq, actions_seq, rewards_seq = outputs
    # obs_seq:     (horizon, B, H, W, C)
    # actions_seq: (horizon, B)
    # rewards_seq: (horizon, B)

    obs_seq = jnp.transpose(obs_seq, (1, 0, 2, 3, 4))        # (B, horizon, H, W, C)
    actions_seq = jnp.transpose(actions_seq, (1, 0))         # (B, horizon)
    rewards_seq = jnp.transpose(rewards_seq, (1, 0))         # (B, horizon)

    # Prepend initial step (s0, a0, r0) to get full sequences of length horizon+1.
    frames_full = jnp.concatenate([s0[:, None, ...], obs_seq], axis=1)        # (B, H+1, H, W, C)
    actions_full = jnp.concatenate([a0[:, None], actions_seq], axis=1)        # (B, horizon+1)
    rewards_full = jnp.concatenate([r0[:, None], rewards_seq], axis=1)        # (B, horizon+1)

    # Episode returns (ignore r0 which is NaN).
    returns = jnp.nansum(rewards_full[:, 1:], axis=-1)  # (B,)

    return EpisodeResult(
        frames=frames_full,
        actions=actions_full,
        rewards=rewards_full,
        returns=returns,
    )


# ---------------------------
# Symlog helpers for TD-λ
# ---------------------------


def _symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def _symexp(y):
    return jnp.sign(y) * (jnp.expm1(jnp.abs(y)))


def _twohot_symlog_targets(values, centers_log):
    """
    values: (...,) real values (e.g., TD(lambda) returns)
    centers_log: (K,) bin centers in symlog space (log-scale symmetric grid)
    returns: (..., K) two-hot targets that sum to 1
    """
    y = _symlog(values)
    K = centers_log.shape[0]

    idx_r = jnp.searchsorted(centers_log, y, side="right")
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


# ---------------------------
# Real-environment evaluation wrapper
# ---------------------------


def evaluate_policy_real_env(
    train_state: TrainState,
    cfg: RLConfig,
    env_reset_fn,
    env_step_fn,
    *,
    schedule_step_idx: int,
    k_max: int,
    rng_key: jax.Array,
) -> tuple[Dict[str, float], Dict[str, Any], jax.Array]:
    """
    High-level helper that runs `eval_rollout_real_env` for multiple episodes,
    aggregates metrics, and returns episode data for visualization.
    """
    n_spatial = cfg.enc_n_latents // cfg.packing_factor
    horizon = cfg.eval_horizon
    batch_size = cfg.eval_batch_size

    num_batches = int(np.ceil(cfg.eval_episodes / batch_size))

    all_returns = []
    first_result: EpisodeResult | None = None
    eval_rng = rng_key

    for b in range(num_batches):
        eval_rng, subkey = jax.random.split(eval_rng)
        task_ids = jnp.zeros((batch_size,), dtype=jnp.int32)

        result = eval_rollout_real_env(
            encoder=train_state.encoder,
            dynamics=train_state.dynamics,
            task_embedder=train_state.task_embedder,
            policy_head=train_state.policy_head,
            enc_vars=train_state.enc_vars,
            dyn_vars=train_state.dyn_vars,
            task_vars=train_state.task_vars,
            pi_vars=train_state.pi_vars,
            mae_eval_key=train_state.mae_eval_key,
            env_reset_fn=env_reset_fn,
            env_step_fn=env_step_fn,
            task_ids=task_ids,
            horizon=horizon,
            patch=cfg.patch,
            n_spatial=n_spatial,
            packing_factor=cfg.packing_factor,
            max_context=cfg.context_length,
            schedule_step_idx=schedule_step_idx,
            k_max=k_max,
            rng_key=subkey,
        )

        all_returns.append(result.returns)
        if first_result is None:
            first_result = result

    all_returns_arr = jnp.concatenate(all_returns, axis=0)[: cfg.eval_episodes]

    metrics: Dict[str, float] = {
        "eval/return_mean": float(jnp.mean(all_returns_arr)),
        "eval/return_std": float(jnp.std(all_returns_arr)),
        "eval/return_min": float(jnp.min(all_returns_arr)),
        "eval/return_max": float(jnp.max(all_returns_arr)),
    }

    media: Dict[str, Any] = {}
    if first_result is not None:
        # Keep a small batch of frames/actions/rewards for later visualization.
        media = {
            "frames": first_result.frames,
            "actions": first_result.actions,
            "rewards": first_result.rewards,
        }

    return metrics, media, eval_rng


# ---------------------------
# JITted training step (imagination + TD-λ + PMPO)
# ---------------------------


@partial(
    jax.jit,
    static_argnames=(
        "encoder",
        "dynamics",
        "task_embedder",
        "reward_head",
        "value_head",
        "policy_head",
        "policy_head_bc",
        "tx",
        "horizon",
        "gamma",
        "lambda_",
        "alpha",
        "beta",
        "context_length",
        "patch",
        "n_spatial",
        "packing_factor",
    ),
)
def train_step(
    *,
    encoder: Encoder,
    dynamics: Dynamics,
    task_embedder: TaskEmbedder,
    reward_head: RewardHeadMTP,
    value_head: ValueHead,
    policy_head: PolicyHeadMTP,
    policy_head_bc: PolicyHeadMTP,
    tx: optax.Transform,
    params: dict,
    opt_state: optax.OptState,
    enc_vars: dict,
    dyn_vars: dict,
    task_vars: dict,
    rew_vars: dict,
    val_vars: dict,
    pi_vars: dict,
    pi_bc_vars: dict,
    mae_eval_key: jnp.ndarray,
    schedule: DenoiseSchedule,
    videos: jnp.ndarray,  # (B, T, H, W, C)
    actions_full: jnp.ndarray,  # (B, T)
    rewards_full: jnp.ndarray,  # (B, T)
    task_ids: jnp.ndarray,  # (B,)
    horizon: int,
    gamma: float,
    lambda_: float,
    alpha: float,
    beta: float,
    context_length: int,
    patch: int,
    n_spatial: int,
    packing_factor: int,
    rng_key: jax.Array,
):
    """
    Single training step:
      - sample context windows from videos
      - encode to latents
      - run JIT-friendly imagination rollouts in latent space
      - compute TD-λ value targets and PMPO policy loss
      - update policy/value head params.
    """

    # RNG splits for context sampling and losses (including imagination).
    rng_key, ctx_key, loss_key = jax.random.split(rng_key, 3)

    # ----- Context sampling -----
    context_frames, context_actions, _ = sample_contexts(
        videos,
        actions_full,
        rewards_full,
        ctx_key,
        context_length,
        videos.shape[2],
        videos.shape[3],
        videos.shape[4],
    )

    # ----- Encode context frames to latents -----
    context_patches = temporal_patchify(context_frames, patch)
    z_btLd, _ = encoder.apply(
        enc_vars,
        context_patches,
        rngs={"mae": mae_eval_key},
        deterministic=True,
    )
    z_context = pack_bottleneck_to_spatial(
        z_btLd,
        n_spatial=n_spatial,
        k=packing_factor,
    )  # (B, context_length, n_spatial, d_spatial)

    def loss_fn(p):
        # Unpack params
        val_params_local = p["val"]
        pi_params_local = p["pi"]

        # Bind params into model variables
        val_vars_local = {**val_vars, "params": val_params_local}
        pi_vars_local = {**pi_vars, "params": pi_params_local}

        # Split rng for dropout and imagination
        rng_rew, rng_val, rng_pi, rng_imag = jax.random.split(loss_key, 4)

        # ----- Imagination rollout in latent space (depends on policy params) -----
        def policy_fn(h: jnp.ndarray, rng: jax.Array, state: Any):
            # h: (B, d_model) – treat as constant feature for the policy
            h_sg = jax.lax.stop_gradient(h)
            rng_act, rng_drop = jax.random.split(rng)
            h_for_policy = h_sg[:, None, :]  # (B, 1, d_model)

            pi_logits = policy_head.apply(
                pi_vars_local,
                h_for_policy,
                rngs={"dropout": rng_drop},
                deterministic=False,
            )  # (B, 1, L, A)
            logits_t0 = pi_logits[:, 0, 0, :]  # (B, A)

            logp = jax.nn.log_softmax(logits_t0, axis=-1)
            actions = jax.random.categorical(rng_act, logp, axis=-1)
            return actions.astype(jnp.int32), logits_t0, state

        imagined_latents, imagined_actions, imagined_hidden_states, policy_logits = (
            imagine_rollouts_core(
                dynamics=dynamics,
                task_embedder=task_embedder,
                dyn_vars=dyn_vars,
                task_vars=task_vars,
                schedule=schedule,
                z_context=z_context,
                context_actions=context_actions,
                task_ids=task_ids,
                horizon=horizon,
                policy_fn=policy_fn,
                policy_state=None,
                rng_key=rng_imag,
            )
        )
        del imagined_latents  # not needed for losses

        # Treat imagined hidden states as frozen features for heads.
        imagined_hidden_states_sg = jax.lax.stop_gradient(imagined_hidden_states)

        # ----- Reward prediction (skip initial timestep, which has no reward) -----
        rw_logits, centers_log_rw = reward_head.apply(
            rew_vars,
            imagined_hidden_states_sg[:, 1:, :],  # (B, horizon, d_model)
            rngs={"dropout": rng_rew},
            deterministic=False,
        )  # rw_logits: (B, horizon, K), centers_log_rw: (K,)

        probs_rw = jax.nn.softmax(rw_logits[:, :, 0, :], axis=-1)  # (B, horizon, K)
        exp_symlog_rw = jnp.sum(
            probs_rw * centers_log_rw[None, None, :],
            axis=-1,
        )  # (B, horizon)
        rewards = _symexp(exp_symlog_rw)  # (B, horizon), corresponds to r1..rT

        # ----- Value head -----
        val_logits, centers_log_val = value_head.apply(
            val_vars_local,
            imagined_hidden_states_sg,
            rngs={"dropout": rng_val},
            deterministic=False,
        )  # (B, horizon + 1, K)

        probs_val = jax.nn.softmax(val_logits, axis=-1)  # (B, horizon+1, K)
        exp_symlog_val = jnp.sum(
            probs_val * centers_log_val[None, None, :],
            axis=-1,
        )
        values = _symexp(exp_symlog_val)  # (B, horizon + 1)

        # ----- TD-λ returns -----
        # R^λ[t] = r[t+1] + γ * ((1-λ) v[t+1] + λ R^λ[t+1]), with R^λ[T] = v[T]
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
            values[:, -1],
            (r_rev.T, v_next_rev.T),
        )
        td_returns = rearrange(G_rev[::-1], "T B -> B T")  # (B, horizon)
        twohot_targets = jax.lax.stop_gradient(
            _twohot_symlog_targets(td_returns, centers_log_val)
        )  # (B, horizon, K)

        # Cross-entropy loss over value distribution
        logq_val = jax.nn.log_softmax(
            val_logits[:, :-1],
            axis=-1,
        )  # (B, horizon, K)
        val_ce = -jnp.sum(twohot_targets * logq_val, axis=-1)  # (B, horizon)
        val_loss = jnp.mean(val_ce)

        # ----- Policy head loss (PMPO) -----
        # Advantages A_t = R^λ_t - v_t for t=0..T-1
        advantages = jax.lax.stop_gradient(
            td_returns - values[:, :-1]
        )  # (B, horizon)
        # Use logits from imagination rollout (already depend on pi_vars_local)
        pi_logits_t0 = policy_logits  # (B, horizon, A)
        logp_pi = jax.nn.log_softmax(pi_logits_t0, axis=-1)  # (B, horizon, A)

        # Log-prob of imagined actions
        A_dim = logp_pi.shape[-1]
        actions_onehot = jax.nn.one_hot(
            imagined_actions.astype(jnp.int32),
            A_dim,
        )  # (B, horizon, A)
        logp_actions = jnp.sum(actions_onehot * logp_pi, axis=-1)  # (B, horizon)

        logp_flat = rearrange(logp_actions, "B T -> (B T)")
        advantages_flat = rearrange(advantages, "B T -> (B T)")

        mask_positive = advantages_flat >= 0
        mask_negative = advantages_flat < 0

        n_positive = jnp.sum(mask_positive)
        n_negative = jnp.sum(mask_negative)

        # Negative set: encourage high log-prob
        loss_negative = jnp.where(
            n_negative > 0,
            (1 - alpha)
            * jnp.sum(jnp.where(mask_negative, logp_flat, 0.0))
            / n_negative,
            0.0,
        )

        # Positive set: discourage high log-prob
        loss_positive = jnp.where(
            n_positive > 0,
            -alpha
            * jnp.sum(jnp.where(mask_positive, logp_flat, 0.0))
            / n_positive,
            0.0,
        )

        # KL(π_θ || π_BC) regularization
        pi_bc_logits = policy_head_bc.apply(
            pi_bc_vars,
            imagined_hidden_states[:, :horizon, :],
            deterministic=True,
        )  # (B, horizon, L, A)
        pi_bc_logits_t0 = pi_bc_logits[:, :, 0, :]  # (B, horizon, A)
        logp_bc = jax.nn.log_softmax(pi_bc_logits_t0, axis=-1)

        p_pi = jax.nn.softmax(pi_logits_t0, axis=-1)
        kl_per_state = jnp.sum(
            p_pi * (logp_pi - logp_bc),
            axis=-1,
        )  # (B, horizon)
        kl_loss = beta * jnp.mean(kl_per_state)

        pi_loss = loss_negative + loss_positive + kl_loss
        total_loss = val_loss + pi_loss

        aux = {
            "val_loss": val_loss,
            "pi_loss": pi_loss,
            "pi_loss_negative": loss_negative,
            "pi_loss_positive": loss_positive,
            "pi_kl_loss": kl_loss,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "mean_advantage": jnp.mean(advantages),
            "mean_td_return": jnp.mean(td_returns),
        }

        return total_loss, aux

    (loss, aux), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(params)
    del loss

    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update vars with new params so that next step can reuse them.
    new_val_vars = {**val_vars, "params": new_params["val"]}
    new_pi_vars = {**pi_vars, "params": new_params["pi"]}

    return new_params, new_opt_state, new_val_vars, new_pi_vars, aux, rng_key


# ---------------------------
# Main training loop
# ---------------------------


def run(cfg: RLConfig):
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
    vis_dir = _ensure_dir(run_dir / "viz")
    print(f"[setup] writing artifacts to: {run_dir.resolve()}")

    # Initialize wandb if enabled
    if cfg.use_wandb:
        wandb_project = cfg.wandb_project or cfg.run_name
        wandb.init(
            entity=cfg.wandb_entity,
            project=wandb_project,
            name=cfg.run_name,
            config=asdict(cfg),
            dir=str(run_dir),
        )

    # Data iterator
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

    # Initialize models and load checkpoints
    init_rng = jax.random.PRNGKey(0)
    _, (frames_init, actions_init, rewards_init) = next_batch(init_rng)
    del rewards_init

    train_state = initialize_models(cfg, frames_init, actions_init)

    patch = cfg.patch
    k_max = cfg.k_max
    n_spatial = cfg.enc_n_latents // cfg.packing_factor

    # Imagination schedule (static)
    imag_cfg = ImaginationConfig(
        k_max=k_max,
        horizon=cfg.horizon,
        context_length=cfg.context_length,
        n_spatial=n_spatial,
        d=cfg.imagination_d,
        start_mode="pure",
        tau0_fixed=0.0,
        match_ctx_tau=False,
    )
    schedule = _build_static_schedule(imag_cfg)

    # Real-environment evaluation env fns.
    env_reset_fn = make_env_reset_fn(
        batch_size=cfg.eval_batch_size,
        height=cfg.dataset.H,
        width=cfg.dataset.W,
        channels=cfg.dataset.C,
        pixels_per_step=cfg.dataset.pixels_per_step,
        size_min=cfg.dataset.size_min,
        size_max=cfg.dataset.size_max,
        fg_min_color=0 if cfg.dataset.diversify_data else 128,
        fg_max_color=255 if cfg.dataset.diversify_data else 128,
        bg_min_color=0 if cfg.dataset.diversify_data else 255,
        bg_max_color=255 if cfg.dataset.diversify_data else 255,
    )
    env_step_fn = make_env_step_fn(
        height=cfg.dataset.H,
        width=cfg.dataset.W,
        channels=cfg.dataset.C,
    )

    # Checkpoint manager and optional restore
    mngr = make_manager(
        ckpt_dir,
        max_to_keep=cfg.ckpt_max_to_keep,
        save_interval_steps=cfg.ckpt_save_every,
    )
    meta = make_rl_meta(
        enc_kwargs=train_state.enc_kwargs,
        dec_kwargs=train_state.dec_kwargs,
        dynamics_kwargs=train_state.dyn_kwargs,
        H=cfg.dataset.H,
        W=cfg.dataset.W,
        C=cfg.dataset.C,
        patch=patch,
        k_max=k_max,
        packing_factor=cfg.packing_factor,
        n_spatial=n_spatial,
        bc_rew_ckpt_dir=cfg.bc_rew_ckpt,
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
        train_state.pi_vars = {
            **train_state.pi_vars,
            "params": train_state.params["pi"],
        }
        train_state.val_vars = {
            **train_state.val_vars,
            "params": train_state.params["val"],
        }
        print(f"[restore] Resumed from {ckpt_dir} at step={latest_step}")

    # Training loop
    train_rng = jax.random.PRNGKey(2025)
    data_rng = jax.random.PRNGKey(12345)
    eval_rng = jax.random.PRNGKey(98765)

    logger = MetricLogger(
        use_wandb=cfg.use_wandb,
        log_every=cfg.log_every,
        max_steps=cfg.max_steps,
        wandb_obj=wandb,
    )

    pbar = tqdm(range(start_step, cfg.max_steps + 1), 
                initial=start_step, 
                total=cfg.max_steps,
                desc="Training Policy",
                dynamic_ncols=True)

    for step in pbar:
        # Sample batch
        data_start_t = time.perf_counter()
        data_rng, batch_key = jax.random.split(data_rng)
        _, (videos, actions_full, rewards_full) = next_batch(batch_key)
        data_t = time.perf_counter() - data_start_t

        # Task IDs (currently dummy zeros)
        task_ids = jnp.zeros((cfg.dataset.B,), dtype=jnp.int32)

        # JITted train step
        train_rng, step_key = jax.random.split(train_rng)
        train_start_t = time.perf_counter()
        (
            new_params,
            new_opt_state,
            new_val_vars,
            new_pi_vars,
            aux,
            train_rng,
        ) = train_step(
            encoder=train_state.encoder,
            dynamics=train_state.dynamics,
            task_embedder=train_state.task_embedder,
            reward_head=train_state.reward_head,
            value_head=train_state.value_head,
            policy_head=train_state.policy_head,
            policy_head_bc=train_state.policy_head_bc,
            tx=train_state.tx,
            params=train_state.params,
            opt_state=train_state.opt_state,
            enc_vars=train_state.enc_vars,
            dyn_vars=train_state.dyn_vars,
            task_vars=train_state.task_vars,
            rew_vars=train_state.rew_vars,
            val_vars=train_state.val_vars,
            pi_vars=train_state.pi_vars,
            pi_bc_vars=train_state.pi_bc_vars,
            mae_eval_key=train_state.mae_eval_key,
            schedule=schedule,
            videos=videos,
            actions_full=actions_full,
            rewards_full=rewards_full,
            task_ids=task_ids,
            horizon=cfg.horizon,
            gamma=cfg.gamma,
            lambda_=cfg.lambda_,
            alpha=cfg.alpha,
            beta=cfg.beta,
            context_length=cfg.context_length,
            patch=patch,
            n_spatial=n_spatial,
            packing_factor=cfg.packing_factor,
            rng_key=step_key,
        )
        train_t = time.perf_counter() - train_start_t
        total_t = data_t + train_t
        train_state.params = new_params
        train_state.opt_state = new_opt_state
        train_state.val_vars = new_val_vars
        train_state.pi_vars = new_pi_vars

        # Periodically evaluate the policy in the real environment.
        if cfg.eval_every > 0 and (step % cfg.eval_every == 0):
            metrics_eval, media_eval, eval_rng = evaluate_policy_real_env(
                train_state,
                cfg,
                env_reset_fn,
                env_step_fn,
                schedule_step_idx=schedule.step_idx,
                k_max=k_max,
                rng_key=eval_rng,
            )

            print(
                f"[eval] step={step:06d} | "
                f"return_mean={metrics_eval['eval/return_mean']:.4f} | "
                f"return_std={metrics_eval['eval/return_std']:.4f} | "
                f"return_min={metrics_eval['eval/return_min']:.4f} | "
                f"return_max={metrics_eval['eval/return_max']:.4f}"
            )

            # Optional visualization: write MP4 + strip plots at a lower frequency.
            video_path: Path | None = None
            strip0_path: Path | None = None
            if (
                cfg.write_video_every > 0
                and step % cfg.write_video_every == 0
                and media_eval
            ):
                frames = media_eval.get("frames")
                actions = media_eval.get("actions")
                rewards = media_eval.get("rewards")

                if frames is not None and actions is not None and rewards is not None:
                    frames_np = np.asarray(frames)
                    actions_np = np.asarray(actions)
                    rewards_np = np.asarray(rewards)

                    # Defensive shape checks.
                    if frames_np.ndim == 5 and actions_np.ndim == 2 and rewards_np.ndim == 2:
                        B, T = frames_np.shape[:2]

                        # Grid video over all eval episodes.
                        video_path = vis_dir / f"real_env_eval_step{step:06d}.mp4"
                        _save_real_env_grid_video(video_path, frames_np)

                        # Strip plots for a few episodes.
                        num_examples = min(cfg.max_eval_examples_to_plot, B)
                        for b_idx in range(num_examples):
                            fig_path = (
                                vis_dir
                                / f"real_env_eval_strip_step{step:06d}_b{b_idx}.png"
                            )
                            _save_real_env_strip(
                                fig_path,
                                frames_np,
                                actions_np,
                                rewards_np,
                                title=f"Real Env Eval (step={step}, b={b_idx})",
                                b_index=b_idx,
                                max_steps=cfg.eval_horizon,
                            )
                            if b_idx == 0:
                                strip0_path = fig_path

                        print(
                            f"[viz:eval] Saved real-env eval video/strips to {vis_dir}"
                        )

            if cfg.use_wandb and wandb.run is not None:
                log_payload: Dict[str, Any] = {
                    "eval/return_mean": metrics_eval["eval/return_mean"],
                    "eval/return_std": metrics_eval["eval/return_std"],
                    "eval/return_min": metrics_eval["eval/return_min"],
                    "eval/return_max": metrics_eval["eval/return_max"],
                }

                # Attach media if just written this step.
                if video_path is not None:
                    log_payload["eval/video_real_env"] = wandb.Video(
                        str(video_path),
                        fps=25,
                        format="mp4",
                    )
                if strip0_path is not None:
                    log_payload["eval/strip_real_env_b0"] = wandb.Image(
                        str(strip0_path)
                    )

                wandb.log(log_payload, step=step)

        if logger.should_log(step):
            logger.log(
                step,
                metrics={
                    "val_loss": aux["val_loss"],
                    "pi_loss": aux["pi_loss"],
                    "pi_neg": aux["pi_loss_negative"],
                    "pi_pos": aux["pi_loss_positive"],
                    "pi_kl": aux["pi_kl_loss"],
                    "mean_adv": aux["mean_advantage"],
                    "mean_td": aux["mean_td_return"],
                    "n_pos": aux["n_positive"],
                    "n_neg": aux["n_negative"],
                    "time/data": data_t,
                    "time/train": train_t,
                    "time/total": total_t,
                },
                pbar=pbar,
            )

        # Save checkpoint
        state = make_state(train_state.params, train_state.opt_state, train_rng, step)
        maybe_save(mngr, step, state, meta)

    mngr.wait_until_finished()

    if cfg.use_wandb and wandb.run is not None:
        wandb.finish()
        print("[wandb] Finished logging.")


@hydra.main(version_base=None, config_path="../configs", config_name="policy")
def main(cfg: DictConfig):
    schema = OmegaConf.structured(RLConfig)
    cfg = OmegaConf.merge(schema, cfg)
    rl_cfg = OmegaConf.to_object(cfg)
    
    run(rl_cfg)


if __name__ == "__main__":
    main()