# train_dynamics.py
# Streaming-batch training on synthetic data with teacher-forced training and autoregressive evaluation.
# This version keeps ONLY the efficient training step and adds robust Orbax checkpointing.
# It restores the pretrained tokenizer (enc/dec) and trains the dynamics model.

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from functools import partial
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import wandb

from dreamer.models import Dynamics, Tokenizer
from dreamer.data import make_iterator
from dreamer.configs import DynamicsConfig
from dreamer.utils import (
    normalize_with_dataset_stats,
    with_params,
    make_state, make_manager, try_restore, maybe_save,
    pack_mae_params,
    init_tokenizer, init_dynamics,
    _ensure_dir, _to_uint8, _stack_wide, _tile_videos,
    load_pretrained_tokenizer,
)
from dreamer.logging import MetricLogger

from dreamer.sampler import SamplerConfig, sample_video

# ---------------------------
# Config
# ---------------------------


def init_models(rng, tokenizer, dynamics, cfg):
    rng, tokenizer_variables = init_tokenizer(rng, tokenizer, cfg)
    rng, dynamics_variables = init_dynamics(rng, dynamics, cfg)
    return rng, tokenizer_variables, dynamics_variables

# ---------------------------
# Single efficient training step (always used)
# ---------------------------

@partial(
    jax.jit,
    static_argnames=("dynamics","tx","k_max","B_self"),
)
def train_step_efficient(
    dynamics, tx,
    params, opt_state,
    dyn_vars,
    latents, actions,
    *,
    B_self: int, k_max: int,
    master_key: jnp.ndarray, step: int, bootstrap_start: int,
):
    """
    Deterministic two-branch training (one fused main forward):
      - first B_emp rows: empirical flow at d_min = 1/k_max
      - last  B_self rows: bootstrap self-consistency with d > d_min
    If step < bootstrap_start, the bootstrap contribution is masked to 0 (but we still
    execute one fused path to keep a single jit and stable shapes).
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

    # RNGs
    step_key = jax.random.fold_in(master_key, step)
    key_sigma_full, key_step_self, key_noise_full, drop_key = jax.random.split(step_key, 4)

    # Deterministic batch split
    B, T, S, D = latents.shape
    B_emp  = B - B_self
    actions_full = actions
    emax = jnp.log2(k_max).astype(jnp.int32)

    # --- Step indices (encode d) ---
    step_idx_emp  = jnp.full((B_emp,  T), emax, dtype=jnp.int32)             # d = d_min
    # If B_self == 0, create a dummy 0xT array – slicing below handles it.
    d_self, step_idx_self = _sample_step_excluding_dmin(key_step_self, (B_self, T), k_max)
    step_idx_full = jnp.concatenate([step_idx_emp, step_idx_self], axis=0)   # (B,T)

    # --- Signal levels on each row's grid (one call for whole batch) ---
    sigma_full, sigma_idx_full = _sample_tau_for_step(key_sigma_full, (B, T), k_max, step_idx_full)
    sigma_emp   = sigma_full[:B_emp]
    sigma_self  = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]

    # --- Corrupt inputs: z_tilde = (1 - sigma) z0 + sigma z1 ---
    z0_full      = jax.random.normal(key_noise_full, latents.shape, dtype=latents.dtype)
    z_tilde_full = (1.0 - sigma_full)[...,None,None] * z0_full + sigma_full[...,None,None] * latents
    z_tilde_self = z_tilde_full[B_emp:]

    # --- Ramp weights ---
    w_emp  = 0.9 * sigma_emp  + 0.1
    w_self = 0.9 * sigma_self + 0.1

    # --- Half-step metadata for self rows ---
    d_half            = d_self / 2.0
    step_idx_half     = step_idx_self + 1
    sigma_plus        = sigma_self + d_half
    sigma_idx_plus    = sigma_idx_self + (k_max * d_half).astype(jnp.int32)

    def loss_and_aux(p):
        local_dyn = with_params(dyn_vars, p) # What is this?
        drop_main, drop_h1, drop_h2 = jax.random.split(drop_key, 3)

        # Main forward (emp + self)
        z1_hat_full, *_ = dynamics.apply(
            local_dyn, actions_full, step_idx_full, sigma_idx_full, z_tilde_full,
            rngs={"dropout": drop_main}, deterministic=False,
        )  # (B,T,Sz,Dz)

        z1_hat_emp  = z1_hat_full[:B_emp]
        z1_hat_self = z1_hat_full[B_emp:]

        # Flow loss on empirical rows (to z1)
        flow_per = jnp.mean((z1_hat_emp - latents[:B_emp])**2, axis=(2,3))        # (B_emp,T)
        loss_emp = jnp.mean(flow_per * w_emp)

        # Self-consistency (bootstrap) on self rows
        # If B_self == 0, shapes are 0-sized and reductions become NaN; guard with mask.
        do_boot = (B_self > 0) & (step >= bootstrap_start)

        def _boot_loss():
            z1_hat_half1, *_ = dynamics.apply(
                local_dyn, actions_full[B_emp:], step_idx_half, sigma_idx_self, z_tilde_self,
                rngs={"dropout": drop_h1}, deterministic=False,
            )
            b_prime = (z1_hat_half1 - z_tilde_self) / (1.0 - sigma_self)[...,None,None]
            z_prime = z_tilde_self + b_prime * d_half[...,None,None]
            z1_hat_half2, *_ = dynamics.apply(
                local_dyn, actions_full[B_emp:], step_idx_half, sigma_idx_plus, z_prime,
                rngs={"dropout": drop_h2}, deterministic=False,
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
            lambda: (jnp.array(0.0, dtype=latents.dtype), jnp.array(0.0, dtype=latents.dtype)),
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
# Main
# ---------------------------

def run(cfg: DynamicsConfig):
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

    rng = jax.random.PRNGKey(0)
    dataset = make_iterator(cfg.dataset)

    tokenizer, tokenizer_params, tokenizer_cfg = Tokenizer.from_pretrained(cfg.tokenizer_ckpt)
    dynamics = Dynamics(cfg.dynamics)
    rng, dynamics_variables = init_dynamics(rng, dynamics, tokenizer_cfg)
    dynamics_params = dynamics_variables["params"]
    
    tx = optax.adamw(cfg.lr)
    opt_state = tx.init(dynamics_params)

    # -------- Training loop --------
    logger = MetricLogger(
        use_wandb=cfg.use_wandb,
        log_every=cfg.log_every,
        max_steps=cfg.max_steps,
        wandb_obj=wandb,
    )

    pbar = tqdm(enumerate(dataset))
    for step, batch in pbar:
        # Data
        rng, tokenizer_key, master_key = jax.random.split(rng, num=3)

        # Normalize videos
        videos = batch["videos"].astype(jnp.float32) / 255.0
        actions = batch["actions"]
        videos = normalize_with_dataset_stats(
            videos, 
            mean=cfg.dataset.dataset_mean, 
            std=cfg.dataset.dataset_std
        )

        latents = tokenizer.encoder.apply({"params":tokenizer_params}, videos, rngs=tokenizer_key)

        dynamics_params, opt_state, aux = train_step_efficient(
            dynamics, tx, dynamics_params, opt_state, dynamics_vaes, latents, actions,
            B_self=videos.shape[0]//2, k_max=cfg.dynamics.k_max, master_key=master_key, step=step, bootstrap_start=cfg.dynamics.bootstrap_start,
        )

        # Logging
        if logger.should_log(step):
            logger.log(
                step,
                metrics={
                    "flow_mse": aux["flow_mse"],
                    "boot_mse": aux["bootstrap_mse"],
                },
                pbar=pbar
            )

        # Save (async) when policy says we should
        # state = make_state(train_state.params, train_state.opt_state, train_rng, step)
        # maybe_save(mngr, step, state, meta)

        # Periodic lightweight AR eval
        # if cfg.write_video_every and (step % cfg.write_video_every == 0):
        #     run_evaluation(cfg=cfg, step=step, train_state=train_state, next_batch=next_batch, vis_dir=vis_dir)

    # Ensure all writes finished
    # mngr.wait_until_finished()

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

    
    
# ------- OLD CODE

# # ---------------------------
# # Eval regimes & plan JSON (unchanged core logic)
# # ---------------------------

# def _eval_regimes_for_realism(cfg, *, ctx_length: int):
#     common = dict(
#         k_max=cfg.k_max,
#         horizon=min(32, cfg.dataset.T - ctx_length),
#         ctx_length=ctx_length,
#         ctx_signal_tau=1.0,   # was 0.99 — make context clean for fair PSNR
#         H=cfg.dataset.H, W=cfg.dataset.W, C=cfg.dataset.C, patch=cfg.patch,
#         n_spatial=cfg.enc_n_latents // cfg.packing_factor,
#         packing_factor=cfg.packing_factor,
#         dataset_mean=cfg.dataset.dataset_mean,
#         dataset_std=cfg.dataset.dataset_std,
#         start_mode="pure",
#         rollout="autoregressive",
#     )
#     regs = []
#     regs.append(("finest_pure_AR", SamplerConfig(schedule="finest", **common)))
#     regs.append(("shortcut_d4_pure_AR", SamplerConfig(schedule="shortcut", d=1/4, **common)))
#     return regs


# def _plan_from_sampler_conf(s: SamplerConfig) -> Dict[str, Any]:
#     def _is_pow2_frac(x: float) -> bool:
#         if x <= 0 or x > 1: return False
#         inv = round(1.0 / x)
#         return abs(1.0 / inv - x) < 1e-8 and (inv & (inv - 1)) == 0

#     if s.schedule == "finest":
#         d = 1.0 / float(s.k_max)
#     else:
#         if s.d is None or not _is_pow2_frac(s.d):
#             raise ValueError("shortcut schedule requires d = 1/(power of two)")
#         if s.d < 1.0 / float(s.k_max):
#             raise ValueError("d finer than finest")
#         d = float(s.d)

#     tau0 = 0.0
#     S = int(round((1.0 - tau0) / d))
#     e = int(round(np.log2(round(1.0 / d))))
#     tau_seq = [round(tau0 + i*d, 6) for i in range(S + 1)]
#     tau_seq[-1] = 1.0
#     return dict(
#         rollout=s.rollout,
#         start_mode=s.start_mode,
#         ctx_length=s.ctx_length,
#         horizon=s.horizon,
#         schedule=s.schedule,
#         d=d,
#         e=e,
#         S=S,
#         tau_seq=tau_seq,
#         k_max=s.k_max,
#         add_ctx_noise_std=getattr(s, "add_ctx_noise_std", 0.0),
#     )

# # ---------------------------
# # Video building and saving utilities
# # ---------------------------

# def build_tiled_video_frames(
#     gt_frames: jnp.ndarray,
#     floor_frames: jnp.ndarray,
#     pred_frames: jnp.ndarray,
#     batch_size: int,
# ) -> list[np.ndarray]:
#     """
#     Build tiled video frames from ground truth, floor, and prediction frames.

#     Each frame in the output contains a grid of triplets (GT | Floor | Pred) stacked horizontally,
#     with multiple batch items tiled vertically/horizontally.

#     Args:
#         gt_frames: Ground truth frames (B, T, H, W, C)
#         floor_frames: Floor/reference frames (B, T, H, W, C)
#         pred_frames: Predicted frames (B, T, H, W, C)
#         batch_size: Batch size B

#     Returns:
#         List of grid frames, one per time step
#     """
#     gt_np_all = _to_uint8(gt_frames)
#     floor_np_all = _to_uint8(floor_frames)
#     pred_np_all = _to_uint8(pred_frames)

#     T_total = gt_np_all.shape[1]
#     ncols = 1 if batch_size <= 2 else min(8, batch_size)
#     grid_frames = []

#     for t_idx in range(T_total):
#         trip_list = [
#             _stack_wide(gt_np_all[b, t_idx], floor_np_all[b, t_idx], pred_np_all[b, t_idx])
#             for b in range(batch_size)
#         ]
#         grid_img = _tile_videos(trip_list, ncols=ncols, pad_color=0)
#         grid_frames.append(grid_img)

#     return grid_frames

# def save_evaluation_video(
#     grid_frames: list[np.ndarray],
#     output_path: Path,
#     tag: str,
# ) -> bool:
#     """
#     Save grid frames as an MP4 video file.

#     Args:
#         grid_frames: List of grid frames to write
#         output_path: Path where MP4 should be saved
#         tag: Tag for error messages

#     Returns:
#         True if successful, False otherwise
#     """
#     print(f"[eval:{tag}] MP4 write started...")
#     try:
#         with imageio.get_writer(output_path, fps=25, codec="libx264", quality=8) as w:
#             for fr in grid_frames:
#                 w.append_data(fr)
#         print(f"[eval:{tag}] MP4 write completed successfully.")
#         return True
#     except Exception as e:
#         print(f"[eval:{tag}] MP4 write skipped ({e})")
#         return False

# def save_evaluation_plan(
#     sampler_conf: SamplerConfig,
#     step: int,
#     mse: float,
#     psnr: float,
#     output_path: Path,
# ):
#     """
#     Save evaluation plan/metadata as JSON.

#     Args:
#         sampler_conf: Sampler configuration
#         step: Training step number
#         mse: Mean squared error
#         psnr: Peak signal-to-noise ratio in dB
#         output_path: Path where JSON should be saved
#     """
#     plan = _plan_from_sampler_conf(sampler_conf)
#     plan["step"] = int(step)
#     plan["mse"] = float(mse)
#     plan["psnr_db"] = float(psnr)

#     with open(output_path, "w") as f:
#         json.dump(plan, f, indent=2)

# # ---------------------------
# # Meta for dynamics checkpoints
# # ---------------------------

# def make_dynamics_meta(
#     *,
#     enc_kwargs: dict,
#     dec_kwargs: dict,
#     dynamics_kwargs: dict,
#     H: int, W: int, C: int,
#     patch: int,
#     k_max: int,
#     packing_factor: int,
#     n_spatial: int,
#     tokenizer_ckpt_dir: str | None = None,
#     cfg: Dict[str, Any] | None = None,
# ):
#     return {
#         "enc_kwargs": enc_kwargs,
#         "dec_kwargs": dec_kwargs,
#         "dynamics_kwargs": dynamics_kwargs,
#         "H": H, "W": W, "C": C, "patch": patch,
#         "k_max": k_max,
#         "packing_factor": packing_factor,
#         "n_spatial": n_spatial,
#         "tokenizer_ckpt_dir": tokenizer_ckpt_dir,
#         "cfg": cfg or {},
#     }

# # ---------------------------
# # Model initialization
# # ---------------------------

# def initialize_models_and_tokenizer(
#     cfg: DynamicsConfig,
#     frames_init: jnp.ndarray,
#     actions_init: jnp.ndarray,
# ) -> TrainState:
#     """
#     Initialize encoder, decoder, dynamics models and restore tokenizer.

#     Returns:
#         TrainState containing all initialized models, variables, and optimizer state.
#     """
#     patch = cfg.patch
#     num_patches = (cfg.dataset.H // patch) * (cfg.dataset.W // patch)
#     D_patch = patch * patch * cfg.dataset.C
#     k_max = cfg.k_max

#     enc_kwargs = dict(
#         d_model=cfg.d_model_enc,
#         n_latents=cfg.enc_n_latents,
#         n_patches=num_patches,
#         n_heads=cfg.n_heads,
#         n_kv_heads=cfg.n_kv_heads,
#         depth=cfg.enc_depth,
#         dropout_rate=0.0,
#         qk_norm_type=cfg.qk_norm_type,
#         rope_theta=cfg.rope_theta,
#         d_bottleneck=cfg.enc_d_bottleneck,
#         mae_p_min=0.0, mae_p_max=0.0,
#         mlp_ratio=4.0,
#         time_every=4,
#     )
#     dec_kwargs = dict(
#         d_model=cfg.d_model_enc,
#         n_heads=cfg.n_heads,
#         n_kv_heads=cfg.n_kv_heads,
#         depth=cfg.dec_depth,
#         n_latents=cfg.enc_n_latents,
#         n_patches=num_patches,
#         d_patch=D_patch,
#         dropout_rate=0.0,
#         qk_norm_type=cfg.qk_norm_type,
#         rope_theta=cfg.rope_theta,
#         mlp_ratio=4.0, time_every=4,
#     )
#     n_spatial = cfg.enc_n_latents // cfg.packing_factor # number of spatial tokens for dynamics
#     dyn_kwargs = dict(
#         d_model=cfg.d_model_dyn,
#         d_bottleneck=cfg.enc_d_bottleneck,
#         d_spatial=cfg.enc_d_bottleneck * cfg.packing_factor,
#         n_spatial=n_spatial, n_register=cfg.n_register,
#         n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, depth=cfg.dyn_depth,
#         n_agent=cfg.n_agent,
#         dropout_rate=0.0,
#         qk_norm_type=cfg.qk_norm_type,
#         rope_theta=cfg.rope_theta,
#         k_max=k_max, 
#         mlp_ratio=4.0,
#         time_every=4,
#     )

#     encoder = Encoder(**enc_kwargs)
#     decoder = Decoder(**dec_kwargs)
#     dynamics = Dynamics(**dyn_kwargs)

#     patches_btnd = temporal_patchify(frames_init, patch)
#     rng = jax.random.PRNGKey(0)
#     enc_vars = encoder.init({"params": rng, "mae": rng, "dropout": rng}, patches_btnd, deterministic=True)
#     fake_z = jnp.zeros((cfg.dataset.B, cfg.dataset.T, cfg.enc_n_latents, cfg.enc_d_bottleneck))
#     dec_vars = decoder.init({"params": rng, "dropout": rng}, fake_z, deterministic=True)

#     # Restore tokenizer
#     enc_vars, dec_vars, _ = load_pretrained_tokenizer(
#         cfg.tokenizer_ckpt, rng=rng,
#         encoder=encoder, decoder=decoder,
#         enc_vars=enc_vars, dec_vars=dec_vars,
#         sample_patches_btnd=patches_btnd,
#     )

#     # Build initial z1 to shape dynamics init
#     mae_eval_key = jax.random.PRNGKey(777)
#     patches_norm = normalize_with_dataset_stats(patches_btnd, mean=cfg.dataset.dataset_mean, std=cfg.dataset.dataset_std)
#     z_btLd, _ = encoder.apply(enc_vars, patches_norm, rngs={"mae": mae_eval_key}, deterministic=True)
#     z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=cfg.packing_factor)
#     emax = jnp.log2(k_max).astype(jnp.int32)
#     step_idx = jnp.full((cfg.dataset.B, cfg.dataset.T), emax, dtype=jnp.int32)
#     sigma_idx = jnp.full((cfg.dataset.B, cfg.dataset.T), k_max - 1, dtype=jnp.int32)
#     dyn_vars = dynamics.init({"params": rng, "dropout": rng}, actions_init, step_idx, sigma_idx, z1)
#     params = dyn_vars["params"]

#     tx = optax.adam(cfg.lr)
#     opt_state = tx.init(params)

#     return TrainState(
#         encoder=encoder,
#         decoder=decoder,
#         dynamics=dynamics,
#         enc_vars=enc_vars,
#         dec_vars=dec_vars,
#         dyn_vars=dyn_vars,
#         params=params,
#         enc_kwargs=enc_kwargs,
#         dec_kwargs=dec_kwargs,
#         dyn_kwargs=dyn_kwargs,
#         tx=tx,
#         opt_state=opt_state,
#         mae_eval_key=mae_eval_key,
#     )

# # ---------------------------
# # Evaluation logic
# # ---------------------------

# def run_evaluation(
#     cfg: DynamicsConfig,
#     step: int,
#     train_state: TrainState,
#     next_batch,
#     vis_dir: Path,
# ):
#     """
#     Run periodic evaluation: sample videos, compute metrics, and save visualization.

#     Args:
#         cfg: Configuration object
#         step: Current training step
#         train_state: TrainState containing all models, variables, and optimizer state
#         next_batch: Data iterator function
#         vis_dir: Directory for visualization outputs
#     """
#     val_rng = jax.random.PRNGKey(9999)
#     _, (val_frames, val_actions, _) = next_batch(val_rng)
#     dyn_vars_eval = with_params(train_state.dyn_vars, train_state.params)
#     ctx_length = min(32, cfg.dataset.T - 1)
#     regimes = _eval_regimes_for_realism(cfg, ctx_length=ctx_length)

#     for tag, sampler_conf in regimes:
#         sampler_conf.mae_eval_key = train_state.mae_eval_key
#         sampler_conf.rng_key = jax.random.PRNGKey(4242)
#         t0 = time.time()

#         pred_frames, floor_frames, gt_frames = sample_video(
#             encoder=train_state.encoder,
#             decoder=train_state.decoder,
#             dynamics=train_state.dynamics,
#             enc_vars=train_state.enc_vars,
#             dec_vars=train_state.dec_vars,
#             dyn_vars=dyn_vars_eval,
#             frames=val_frames, actions=val_actions, config=sampler_conf,
#         )

#         dt = time.time() - t0
#         HZ = sampler_conf.horizon
#         mse = float(jnp.mean((pred_frames[:, -HZ:] - gt_frames[:, -HZ:]) ** 2))
#         psnr = float(10.0 * jnp.log10(1.0 / jnp.maximum(mse, 1e-12)))
#         print(f"[eval:{tag}] step={step:06d} | AR_hz={HZ} | MSE={mse:.6g} | PSNR={psnr:.2f} dB | {dt:.2f}s")

#         # Build tiled video frames
#         grid_frames = build_tiled_video_frames(
#             gt_frames=gt_frames,
#             floor_frames=floor_frames,
#             pred_frames=pred_frames,
#             batch_size=cfg.dataset.B,
#         )

#         # Save video and plan
#         tag_dir = _ensure_dir(vis_dir / f"step_{step:06d}")
#         mp4_path = tag_dir / f"{tag}_grid.mp4"
#         plan_path = tag_dir / f"{tag}_plan.json"

#         save_evaluation_video(grid_frames, mp4_path, tag)
#         save_evaluation_plan(sampler_conf, step, mse, psnr, plan_path)

#         print(f"[eval:{tag}] wrote {mp4_path.name} and {plan_path.name} in {tag_dir}")

#         # Log to wandb
#         if cfg.use_wandb and wandb.run is not None:
#             # Log metrics
#             wandb.log({
#                 f"eval/{tag}/mse": mse,
#                 f"eval/{tag}/psnr": psnr,
#                 f"eval/{tag}/horizon": HZ,
#                 f"eval/{tag}/eval_time": dt,
#             }, step=step)
#             if grid_frames:
#                 wandb.log({
#                     f"eval/{tag}/video": wandb.Video(mp4_path, format="mp4"),
#                 }, step=step)

