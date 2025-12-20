# train_dynamics.py
"""
Dynamics model training with teacher-forced flow matching and bootstrap self-consistency.

Architecture:
  - Loads pretrained tokenizer (frozen)
  - Trains dynamics model on latent space
  - Periodic autoregressive evaluation with video visualization
"""
from __future__ import annotations

import logging

from dreamer.generation import DenoiseSchedule
from dreamer.sampler import sample_video
# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)

import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict

import hydra
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from einops import rearrange
from flax.core import FrozenDict
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dreamer.configs import DynamicsConfig, TokenizerConfig
from dreamer.data import make_iterator
from dreamer.logging import MetricLogger
from dreamer.models import Dynamics, Tokenizer
# from dreamer.sampler import SamplerConfig, sample_video, plan_from_sampler_conf
from dreamer.utils import (
    _ensure_dir,
    from_dict,
    init_dynamics,
    make_manager,
    make_state,
    maybe_save,
    try_restore,
)


# ---------------------------
# Training step helpers (don't nest to improve JIT caching speed)
# ---------------------------

@partial(jax.jit, static_argnames=("shape_bt", "k_max"))
def _sample_tau_for_step(rng, shape_bt, k_max: int, step_idx: jnp.ndarray, *, dtype=jnp.float32):
    """Sample tau values aligned to step_idx grid."""
    B_, T_ = shape_bt
    K = 1 << step_idx
    u = jax.random.uniform(rng, (B_, T_), dtype=dtype)
    j_idx = jnp.floor(u * K.astype(dtype)).astype(jnp.int32)
    tau = j_idx.astype(dtype) / K.astype(dtype)
    tau_idx = j_idx * (k_max // K)
    return tau, tau_idx


@partial(jax.jit, static_argnames=("shape_bt", "k_max"))
def _sample_step_excluding_dmin(rng, shape_bt, k_max: int):
    """Sample step indices excluding the finest level (for bootstrap)."""
    B_, T_ = shape_bt
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jax.random.randint(rng, (B_, T_), 0, emax, dtype=jnp.int32)
    d = 1.0 / (1 << step_idx).astype(jnp.float32)
    return d, step_idx

# ---------------------------
# Training step
# ---------------------------

@partial(jax.jit, static_argnames=("dynamics", "tx", "k_max", "B_self"))
def train_step(
    dynamics, tx, params, opt_state, constants, latents, actions,
    *, B_self: int, k_max: int, master_key: jnp.ndarray, step: int, bootstrap_start: int
):
    """
    Two-branch training step with fused forward pass.
    
    Branches:
      - Empirical flow (first B_emp rows): standard flow matching at d_min = 1/k_max
      - Bootstrap (last B_self rows): self-consistency loss with coarser d > d_min
    
    Bootstrap contribution is masked to 0 when step < bootstrap_start.
    """
    # RNGs
    step_key = jax.random.fold_in(master_key, step)
    key_sigma_full, key_step_self, key_noise_full, drop_key = jax.random.split(
        step_key, 4
    )

    # Deterministic batch split
    B, T, S, D = latents.shape
    B_emp = B - B_self
    actions_full = actions
    emax = jnp.log2(k_max).astype(jnp.int32)

    # --- Step indices (encode d) ---
    step_idx_emp = jnp.full((B_emp, T), emax, dtype=jnp.int32)  # d = d_min
    # If B_self == 0, create a dummy 0xT array – slicing below handles it.
    d_self, step_idx_self = _sample_step_excluding_dmin(
        key_step_self, (B_self, T), k_max
    )
    step_idx_full = jnp.concatenate([step_idx_emp, step_idx_self], axis=0)  # (B,T)

    # --- Signal levels on each row's grid (one call for whole batch) ---
    sigma_full, sigma_idx_full = _sample_tau_for_step(
        key_sigma_full, (B, T), k_max, step_idx_full
    )
    sigma_emp = sigma_full[:B_emp]
    sigma_self = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]

    # --- Corrupt inputs: z_tilde = (1 - sigma) z0 + sigma z1 ---
    z0_full = jax.random.normal(key_noise_full, latents.shape, dtype=latents.dtype)
    z_tilde_full = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[
        ..., None, None
    ] * latents
    z_tilde_self = z_tilde_full[B_emp:]

    # --- Ramp weights ---
    w_emp = 0.9 * sigma_emp + 0.1
    w_self = 0.9 * sigma_self + 0.1

    # --- Half-step metadata for self rows ---
    d_half = d_self / 2.0
    step_idx_half = step_idx_self + 1
    sigma_plus = sigma_self + d_half
    sigma_idx_plus = sigma_idx_self + (k_max * d_half).astype(jnp.int32)

    def loss_and_aux(p):
        local_dyn = {"params": p, "constants": constants}
        drop_main, drop_h1, drop_h2 = jax.random.split(drop_key, 3)

        # Main forward (emp + self)
        z1_hat_full, *_ = dynamics.apply(local_dyn, actions_full, step_idx_full, 
            sigma_idx_full, z_tilde_full, rngs={"dropout": drop_main}, deterministic=False)  # (B,T,Sz,Dz)

        z1_hat_emp = z1_hat_full[:B_emp]
        z1_hat_self = z1_hat_full[B_emp:]

        # Flow loss on empirical rows (to z1)
        flow_per = jnp.mean(
            (z1_hat_emp - latents[:B_emp]) ** 2, axis=(2, 3)
        )  # (B_emp,T)
        loss_emp = jnp.mean(flow_per * w_emp)

        # Self-consistency (bootstrap) on self rows
        # If B_self == 0, shapes are 0-sized and reductions become NaN; guard with mask.
        do_boot = (B_self > 0) & (step >= bootstrap_start)

        def _boot_loss():
            z1_hat_half1, *_ = dynamics.apply(local_dyn, actions_full[B_emp:], step_idx_half, 
                sigma_idx_self, z_tilde_self, rngs={"dropout": drop_h1}, deterministic=False)
            b_prime = (z1_hat_half1 - z_tilde_self) / (1.0 - sigma_self)[..., None, None]
            z_prime = z_tilde_self + b_prime * d_half[..., None, None]
            z1_hat_half2, *_ = dynamics.apply(local_dyn, actions_full[B_emp:], step_idx_half,
                sigma_idx_plus, z_prime, rngs={"dropout": drop_h2}, deterministic=False)
            b_doubleprime = (z1_hat_half2 - z_prime) / (1.0 - sigma_plus)[..., None, None] # (B_self, T, n_spatial, D_s)
            vhat_sigma = (z1_hat_self - z_tilde_self) / (1.0 - sigma_self)[..., None, None] # (B_self, T, n_spatial, D_s)
            vbar_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)
            boot_per = (1.0 - sigma_self) ** 2 * jnp.mean((vhat_sigma - vbar_target) ** 2, axis=(2, 3))  # (B_self,T)
            loss_self = jnp.mean(boot_per * w_self)
            return loss_self, jnp.mean(boot_per)

        loss_self, boot_mse = jax.lax.cond(
            do_boot,
            _boot_loss,
            lambda: (
                jnp.array(0.0, dtype=latents.dtype),
                jnp.array(0.0, dtype=latents.dtype),
            ),
        )

        # Combine (row-weighted by nominal B parts; denominator B keeps scale constant)
        loss = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B

        aux = {
            "flow_mse": jnp.mean(flow_per),
            "bootstrap_mse": boot_mse,
        }
        return loss, aux

    (loss_val, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, aux

# ---------------------------
# Evaluation helpers
# ---------------------------

def run_evaluation(
    *,
    cfg: DynamicsConfig,
    tokenizer_cfg: TokenizerConfig,
    step: int,
    tokenizer: Tokenizer,
    tokenizer_vars: Dict[str, Any],
    dynamics: Dynamics,
    dynamics_params: Dict[str, Any],
    dynamics_constants: Dict[str, Any],
    val_videos: jnp.ndarray,
    val_actions: jnp.ndarray,
    vis_dir: Path,
    rng: jax.Array,
):
    """
    Run periodic evaluation: sample videos, compute metrics, and save visualization.
    
    Uses unified Tokenizer with encode/decode methods.
    """

    schedule_shortcut = DenoiseSchedule.init(4, 256)
    schedule_diffusion = DenoiseSchedule.init(256, 256)

    evaluation_schedules = {"shortcut": schedule_shortcut, 
                            "diffusion": schedule_diffusion}

    dyn_vars = {"params": dynamics_params, "constants": dynamics_constants}

    for tag, schedule_config in evaluation_schedules.items():
        t0 = time.time()

        # FIXME: only temporary for debugging
        assert val_videos.shape[1] > 5
        ctx_length = 4
        horizon = val_videos.shape[1] - ctx_length

        pred_frames, floor_frames, gt_frames = sample_video(
            tokenizer=tokenizer,
            tokenizer_vars=tokenizer_vars,
            dynamics=dynamics,
            dyn_vars=dyn_vars,
            frames=val_videos,
            actions=val_actions,
            horizon=horizon,
            schedule_config=schedule_config,
            rng=rng,
        )

        # Compute metrics
        dt = time.time() - t0
        # FIXME: this PSNR is wrong, see commit 73f7922a63cb3f88ca7b6435a5b3b14566281c8e
        mse = float(jnp.mean((pred_frames[:, -horizon:] - gt_frames[:, -horizon:]) ** 2))
        psnr = float(10.0 * jnp.log10(1.0 / jnp.maximum(mse, 1e-12)))
        print(f"[eval:{tag}] step={step:06d} | horizon={horizon} | MSE={mse:.6g} | PSNR={psnr:.2f} dB | {dt:.2f}s")

        # Build visualization
        num_videos = min(4, pred_frames.shape[0])
        frames = [pred_frames, floor_frames, gt_frames]
        stacked_frames = jnp.stack(frames)[:, :num_videos]
        videos = rearrange(stacked_frames, 'S B T H W C -> T (B H) (S W) C', B=num_videos)

        # Save artifacts
        tag_dir = _ensure_dir(vis_dir / f"step_{step:06d}")
        mp4_path = tag_dir / f"{tag}_grid.mp4"

        # Save video
        try:
            iio.imwrite(str(mp4_path), videos, fps=5, plugin='pyav', codec='libx264')
        except Exception as e:
            print(f"[eval:{tag}] MP4 write failed: {e}")

        # Log to wandb
        if cfg.use_wandb and wandb.run is not None:
            wandb.log({
                f"eval/{tag}/mse": mse,
                f"eval/{tag}/psnr": psnr,
                f"eval/{tag}/horizon": horizon,
                f"eval/{tag}/eval_time": dt,
            }, step=step)
            if videos:
                wandb.log({
                    f"eval/{tag}/video": wandb.Video(mp4_path, format="mp4"),
                }, step=step)

# ---------------------------
# Main
# ---------------------------

def run(cfg: DynamicsConfig):
    """Main training loop."""
    # Setup directories
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
    vis_dir = _ensure_dir(run_dir / "viz")
    print(f"[setup] output dir: {run_dir.resolve()}")

    # Wandb
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project or cfg.run_name,
            name=cfg.run_name,
            config=asdict(cfg),
            dir=str(run_dir),
        )

    # Load frozen tokenizer
    rng = jax.random.PRNGKey(0)
    tokenizer, tokenizer_vars, tokenizer_cfg = Tokenizer.from_pretrained(cfg.tokenizer_ckpt)

    # Initialize dynamics
    dynamics = Dynamics(cfg.dynamics)
    rng, dynamics_variables = init_dynamics(rng, dynamics, tokenizer_cfg)
    dynamics_params = dynamics_variables["params"]
    dynamics_constants = dynamics_variables.get("constants", FrozenDict())

    # Optimizer
    tx = optax.adamw(cfg.lr)
    opt_state = tx.init(dynamics_params)

    # Logging & checkpointing
    logger = MetricLogger(
        use_wandb=cfg.use_wandb,
        log_every=cfg.log_every,
        max_steps=cfg.max_steps,
        wandb_obj=wandb,
    )
    mngr = make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every)

    state_example = make_state(dynamics_params, opt_state, rng, step=0)
    meta = {"cfg": asdict(cfg)}

    restored = try_restore(mngr, state_example, meta)
    start_step = 0
    if restored is not None:
        latest_step, r = restored
        dynamics_params = r.state["params"]
        opt_state = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"])
        # Preserve runtime flags before restoring checkpoint config
        use_wandb_override = cfg.use_wandb
        # cfg = from_dict(DynamicsConfig, r.meta["cfg"])
        cfg.use_wandb = use_wandb_override  # Keep CLI/YAML wandb setting
        print(f"[ckpt] Restored step {latest_step}")

    dataset = make_iterator(tokenizer_cfg.dataset)
    pbar = tqdm(enumerate(dataset, start=start_step), total=cfg.max_steps)
    for step, batch in pbar:
        # Data
        rng, tokenizer_key, master_key = jax.random.split(rng, num=3)

        # Normalize videos
        videos = batch["videos"]
        actions = batch["actions"]
        # shift the actions by one and put the "first action token" = 15 at the beginning 
        actions = jnp.concatenate((jnp.full_like(actions[:,0:1], fill_value = 15), actions[:,:-1]), axis=1)
        latents, _ = tokenizer.apply(tokenizer_vars, videos, packing_factor=cfg.dynamics.packing_factor, rngs={"mae": tokenizer_key}, method=tokenizer.encode)

        dynamics_params, opt_state, aux = train_step(dynamics, tx, 
            dynamics_params, opt_state, dynamics_constants, latents, actions, 
            B_self=videos.shape[0] // 2, k_max=cfg.dynamics.k_max, master_key=master_key,
            step=step, bootstrap_start=cfg.bootstrap_start)

        # Logging
        if logger.should_log(step):
            logger.log(
                step,
                metrics={
                    "flow_mse": aux["flow_mse"],
                    "boot_mse": aux["bootstrap_mse"],
                },
                pbar=pbar,
            )

        # Save (async) when policy says we should
        state = make_state(dynamics_params, opt_state, rng, step)
        maybe_save(mngr, step, state, meta)

        # Periodic lightweight AR eval
        if cfg.write_video_every and (step % cfg.write_video_every == 0) and step > 0:
            # Use current batch as validation data (simplest approach)
            val_videos = batch["videos"].astype(jnp.float32) / 255.0
            run_evaluation(
                cfg=cfg,
                tokenizer_cfg=tokenizer_cfg,
                step=step,
                tokenizer=tokenizer,
                tokenizer_vars=tokenizer_vars,
                dynamics=dynamics,
                dynamics_params=dynamics_params,
                dynamics_constants=dynamics_constants,
                val_videos=val_videos,
                val_actions=actions,
                vis_dir=vis_dir,
                rng=rng,
            )

    # Finish wandb run
    if cfg.use_wandb and wandb.run is not None:
        wandb.finish()

@hydra.main(version_base=None, config_path="../configs", config_name="dynamics")
def main(cfg: DictConfig):
    schema = OmegaConf.structured(DynamicsConfig)
    cfg = OmegaConf.merge(schema, cfg)
    realism_cfg = OmegaConf.to_object(cfg)
    run(realism_cfg)

if __name__ == "__main__":
    main()
