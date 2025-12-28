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
from dataclasses import asdict
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from flax.core import FrozenDict
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dreamer.configs import DynamicsConfig, TokenizerConfig
from dreamer.data import make_iterator
from dreamer.logging import MetricLogger
from dreamer.models import Dynamics, Tokenizer
from dreamer.training import run_evaluation, shortcut_forcing_step  # NEW: Reusable training components
from dreamer.utils import (
    _ensure_dir,
    init_dynamics,
    make_manager,
    make_state,
    maybe_save,
    try_restore,
    get_lr_schedule,
    count_parameters_by_component,
)
# jax.config.update("jax_debug_nans", True)
# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)

# ---------------------------
# Training step (now using reusable components from dreamer.training)
# ---------------------------

@partial(jax.jit, static_argnames=("dynamics", "tx", "k_max", "B_self"))
def train_step(
    dynamics, tx, params, opt_state, constants, latents, actions,
    *, B_self: int, k_max: int, master_key: jnp.ndarray, step: int):
    """
    Training step using shortcut forcing (flow matching + bootstrap self-consistency).
    
    Now uses reusable components from dreamer.training for maintainability and code reuse.
    
    Branches:
      - Empirical flow (first B_emp rows): standard flow matching at d_min = 1/k_max
      - Bootstrap (last B_self rows): self-consistency loss with coarser d > d_min
    
    Bootstrap contribution is masked to 0 when step < bootstrap_start.
    """
    step_key = jax.random.fold_in(master_key, step)
    
    def loss_and_aux(p):
        vars_dict = {"params": p, "constants": constants}
        losses, aux = shortcut_forcing_step(
            dynamics_apply_fn=dynamics.apply,
            dynamics_vars=vars_dict,
            actions=actions,
            latents=latents,
            rng=step_key,
            k_max=k_max,
            B_self=B_self,
            agent_tokens=None,  # Not used in dynamics pretraining
        )
        return losses['total'], aux
    
    (loss_val, metrics), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, metrics

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
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project or cfg.run_name, name=cfg.run_name, config=asdict(cfg), dir=str(run_dir))

    # Load frozen tokenizer
    rng = jax.random.PRNGKey(0)
    tokenizer, tokenizer_vars, tokenizer_cfg = Tokenizer.from_pretrained(cfg.tokenizer_ckpt)

    # Initialize dynamics
    dynamics = Dynamics(cfg.dynamics)
    rng, dynamics_variables = init_dynamics(rng, dynamics, tokenizer_cfg)
    dynamics_params = dynamics_variables["params"]
    dynamics_constants = dynamics_variables.get("constants", FrozenDict())
    param_counts = count_parameters_by_component(dynamics_params)
    print(f"Parameter counts: {param_counts}")

    # Optimizer
    if cfg.lr_schedule == "constant":
        lr = cfg.lr
        lr_schedule = None
    else:
        lr_schedule = get_lr_schedule(
            cfg.lr_schedule,
            cfg.init_lr,
            cfg.max_lr,
            cfg.lr_end,
            cfg.max_steps,
            cfg.warmup_steps,
            cfg.wsd_decay_steps,
        )
        lr = lr_schedule
    tx = optax.adamw(lr, b1=0.9, b2=0.9, weight_decay=1e-4)
    opt_state = tx.init(dynamics_params)

    # Logging & checkpointing
    logger = MetricLogger( use_wandb=cfg.use_wandb, log_every=cfg.log_every, max_steps=cfg.max_steps, wandb_obj=wandb)
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
        if step > cfg.max_steps:
            break
        # Data
        rng, tokenizer_key, master_key = jax.random.split(rng, num=3)

        # Normalize videos
        videos = batch["videos"]
        actions = batch["actions"]
        latents, _ = tokenizer.apply(tokenizer_vars, videos, packing_factor=cfg.dynamics.packing_factor, rngs={"mae": tokenizer_key}, method=tokenizer.encode)

        dynamics_params, opt_state, aux = train_step(dynamics, tx, 
            dynamics_params, opt_state, dynamics_constants, latents, actions, 
            B_self=(videos.shape[0] // 2)*(step >= cfg.bootstrap_start), # This will make the function compile twice. TODO: see if it's worth fixing this
            k_max=cfg.dynamics.k_max, master_key=master_key, step=step)

        # Logging
        if logger.should_log(step):
            logger.log(
                step,
                metrics={
                    "flow_mse": aux["flow_mse"],
                    "boot_mse": aux["bootstrap_mse"],
                    "lr": cfg.lr if lr_schedule is None else lr_schedule(step),
                },
                pbar=pbar,
            )

        # Save (async) when policy says we should
        state = make_state(dynamics_params, opt_state, rng, step)
        maybe_save(mngr, step, state, meta)

        # Periodic lightweight AR eval
        if cfg.write_video_every and (step % cfg.write_video_every == 0) and step > 0:
            # Use current batch as validation data (simplest approach)
            val_videos = batch["videos"]
            run_evaluation(cfg, tokenizer_cfg, step, tokenizer, tokenizer_vars, dynamics, dynamics_params, dynamics_constants, val_videos, jnp.asarray(actions), vis_dir, rng)

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
