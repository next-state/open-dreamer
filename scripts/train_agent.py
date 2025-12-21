"""
Phase 3: Agent finetuning (Behavior Cloning + Reward Model)

This script trains task-conditioned policy and reward heads on top of a pretrained
world model, while optionally continuing dynamics training to preserve capability.

Architecture:
  - Loads pretrained tokenizer (frozen)
  - Loads pretrained dynamics model
  - Adds agent tokens with task embeddings
  - Trains policy head (behavior cloning with MTP)
  - Trains reward head (reward prediction with symexp twohot)
  - Optionally continues dynamics training to prevent catastrophic forgetting
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.core import FrozenDict
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dreamer.configs import BCRewConfig
from dreamer.data import make_iterator
from dreamer.logging import MetricLogger
from dreamer.models import Dynamics, PolicyHeadMTP, RewardHeadMTP, TaskEmbedder, Tokenizer
from dreamer.training import (
    compute_policy_loss,
    compute_reward_loss,
    shortcut_forcing_step,
    symlog,
    twohot_symlog_targets,
)
from dreamer.utils import (
    _ensure_dir,
    from_dict,
    make_manager,
    make_state,
    maybe_save,
    try_restore,
)

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)

# ---------------------------
# Multi-token prediction (MTP) helpers
# ---------------------------

def gather_future_actions(actions_bt: jnp.ndarray, L: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Gather future actions for multi-token prediction.
    
    At timestep t, predicts actions[t+1], actions[t+2], ..., actions[t+L]
    (Following Dreamer convention: action a_i happens before state s_i)
    
    Args:
        actions_bt: (B, T) action labels
        L: number of future steps to predict
        
    Returns:
        actions_btL: (B, T, L) future actions
        valid_btL: (B, T, L) mask (0 for out-of-range)
    """
    B, T = actions_bt.shape
    actions_pad = jnp.pad(actions_bt, ((0, 0), (0, L)), constant_values=-1)
    
    offsets = jnp.arange(1, L + 1)  # [1, 2, ..., L]
    indices = jnp.arange(T)[:, None] + offsets[None, :]  # (T, L)
    actions_btL = actions_pad[:, indices]  # (B, T, L)
    valid_btL = (actions_btL >= 0)
    
    return actions_btL, valid_btL


def gather_future_rewards(rewards_bt: jnp.ndarray, L: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Gather future rewards for multi-token prediction.
    
    At timestep t, predicts rewards[t], rewards[t+1], ..., rewards[t+L-1]
    (Following Dreamer convention: r_t is the reward from h_t)
    
    Args:
        rewards_bt: (B, T) reward values
        L: number of future steps to predict
        
    Returns:
        rewards_btL: (B, T, L) future rewards
        valid_btL: (B, T, L) mask (0 for invalid)
    """
    B, T = rewards_bt.shape
    rewards_pad = jnp.pad(rewards_bt, ((0, 0), (0, L - 1)), constant_values=0.0)
    
    offsets = jnp.arange(0, L)  # [0, 1, ..., L-1]
    indices = jnp.arange(T)[:, None] + offsets[None, :]  # (T, L)
    rewards_btL = rewards_pad[:, indices]  # (B, T, L)
    
    # Valid when: t >= 1 AND 1 <= t+offset < T
    # (skip r0 which is dummy, and stay in bounds)
    valid_btL = (indices >= 1) & (indices < T) & (jnp.arange(T)[:, None] >= 1)
    valid_btL = jnp.broadcast_to(valid_btL[None, :, :], (B, T, L))
    
    return rewards_btL, valid_btL


# ---------------------------
# Training step
# ---------------------------

@partial(jax.jit, static_argnames=("tokenizer", "dynamics", "task_embedder", "policy_head", "reward_head", "tx_dict", "k_max", "L_mtp"))
def train_step(
    tokenizer,
    dynamics,
    task_embedder,
    policy_head,
    reward_head,
    tx_dict,
    # State
    tokenizer_vars,
    dynamics_params,
    dynamics_constants,
    policy_params,
    reward_params,
    opt_states,
    # Data
    batch,
    rng,
    step: int,
    # Config
    k_max: int,
    L_mtp: int,
    bootstrap_start: int,
    dynamics_loss_weight: float,
):
    """
    Agent finetuning step with BC + reward prediction + optional dynamics loss.
    
    Args:
        Models: tokenizer, dynamics, task_embedder, policy_head, reward_head
        tx_dict: Dict of optimizers for each component
        State: parameters and optimizer states
        batch: Data batch with videos, actions, tasks, rewards
        rng: Random key
        step: Training step number
        Config: hyperparameters
        
    Returns:
        new_params, new_opt_states, metrics
    """
    B = batch["videos"].shape[0]
    
    # Split RNG
    rng, enc_key, dyn_key = jax.random.split(rng, 3)
    
    # 1. Encode videos to latents (frozen tokenizer)
    latents, _ = jax.lax.stop_gradient(
        tokenizer.apply(
            tokenizer_vars,
            batch["videos"],
            packing_factor=dynamics.config.packing_factor,
            method=tokenizer.encode,
            rngs={"mae": enc_key},
            deterministic=True,
        )
    )
    
    # 2. Create task-conditioned agent tokens
    _, T_video, _, _ = latents.shape
    agent_tokens_bt = task_embedder.apply(
        {"params": {}},  # Task embedder has no learnable params (just lookup)
        task=batch["tasks"],
        B=B,
        T=T_video,
    )
    
    # 3. Gather future actions and rewards for MTP
    actions_btL, actions_valid = gather_future_actions(batch["actions"], L_mtp)
    rewards_btL, rewards_valid = gather_future_rewards(batch["rewards"], L_mtp)
    
    # 4. Define combined loss
    def loss_fn(pol_p, rew_p, dyn_p):
        # Dynamics loss (also returns hidden states for BC/reward training)
        dyn_vars = {"params": dyn_p, "constants": dynamics_constants}
        
        dyn_losses, dyn_aux = shortcut_forcing_step(
            dynamics_apply_fn=dynamics.apply,
            dynamics_vars=dyn_vars,
            actions=batch["actions"],
            latents=latents,
            rng=dyn_key,
            k_max=k_max,
            B_self=B // 2,
            bootstrap_active=(step >= bootstrap_start),
            agent_tokens=agent_tokens_bt,
        )
        dynamics_loss = dyn_losses['total']
        h_states = dyn_aux['h_states']  # (B, T, n_agent, d_model)
        
        # Policy loss (BC with MTP)
        policy_loss = compute_policy_loss(
            policy_head, pol_p, h_states, actions_btL, actions_valid
        )
        
        # Reward loss (symexp twohot with MTP)
        reward_loss = compute_reward_loss(
            reward_head, rew_p, h_states, rewards_btL, rewards_valid
        )
        
        # Combine losses
        total_loss = policy_loss + reward_loss + dynamics_loss_weight * dynamics_loss
        
        aux = {
            "policy_loss": policy_loss,
            "reward_loss": reward_loss,
            "dynamics_loss": dynamics_loss,
            **dyn_aux,
        }
        
        return total_loss, aux
    
    # 5. Compute gradients
    # Only update policy and reward params (dynamics can be frozen or finetuned)
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)
    (loss_val, metrics), (pol_grads, rew_grads, dyn_grads) = grad_fn(
        policy_params, reward_params, dynamics_params
    )
    
    # 6. Apply updates
    pol_updates, new_pol_opt = tx_dict["policy"].update(pol_grads, opt_states["policy"], policy_params)
    new_policy_params = optax.apply_updates(policy_params, pol_updates)
    
    rew_updates, new_rew_opt = tx_dict["reward"].update(rew_grads, opt_states["reward"], reward_params)
    new_reward_params = optax.apply_updates(reward_params, rew_updates)
    
    # Dynamics update (optional)
    if continue_dynamics_loss and dynamics_loss_weight > 0:
        dyn_updates, new_dyn_opt = tx_dict["dynamics"].update(dyn_grads, opt_states["dynamics"], dynamics_params)
        new_dynamics_params = optax.apply_updates(dynamics_params, dyn_updates)
    else:
        new_dynamics_params = dynamics_params
        new_dyn_opt = opt_states["dynamics"]
    
    new_params = {
        "dynamics": new_dynamics_params,
        "policy": new_policy_params,
        "reward": new_reward_params,
    }
    
    new_opt_states = {
        "dynamics": new_dyn_opt,
        "policy": new_pol_opt,
        "reward": new_rew_opt,
    }
    
    return new_params, new_opt_states, metrics


# ---------------------------
# Main
# ---------------------------

def run(cfg: BCRewConfig):
    """Main training loop for agent finetuning."""
    # Setup directories
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
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
    
    # Load pretrained tokenizer and dynamics
    rng = jax.random.PRNGKey(0)
    print(f"[setup] Loading pretrained tokenizer from {cfg.tokenizer_ckpt}")
    tokenizer, tokenizer_vars, tokenizer_cfg = Tokenizer.from_pretrained(cfg.tokenizer_ckpt)
    
    print(f"[setup] Loading pretrained dynamics from {cfg.dynamics_ckpt}")
    dynamics, dynamics_vars = Dynamics.from_pretrained(cfg.dynamics_ckpt)
    dynamics_params = dynamics_vars["params"]
    dynamics_constants = dynamics_vars.get("constants", FrozenDict())
    
    # Initialize task embedder, policy, and reward heads
    print("[setup] Initializing agent components")
    task_embedder = TaskEmbedder(
        n_tasks=10,  # TODO: Get from config
        n_agent=dynamics.config.n_agent,
        d_model=dynamics.config.d_model,
        use_ids=True,
    )
    
    policy_head = PolicyHeadMTP(
        d_model=dynamics.config.d_model,
        d_hidden=dynamics.config.d_model,
        n_actions=cfg.action_dim,
        L=8,  # MTP length
        kind="categorical",
    )
    
    reward_head = RewardHeadMTP(
        d_model=dynamics.config.d_model,
        d_hidden=dynamics.config.d_model,
        L=8,  # MTP length
        K=255,  # Number of bins
        min_val=-20.0,
        max_val=20.0,
    )
    
    # Initialize parameters
    rng, pol_key, rew_key = jax.random.split(rng, 3)
    
    # Dummy inputs for initialization
    dummy_h = jnp.zeros((1, 4, dynamics.config.n_agent, dynamics.config.d_model))
    
    policy_params = policy_head.init(pol_key, dummy_h, deterministic=True)["params"]
    reward_vars = reward_head.init(rew_key, dummy_h, deterministic=True)
    reward_params = reward_vars["params"]
    
    # Optimizers
    tx_dict = {
        "policy": optax.adamw(cfg.lr_policy),
        "reward": optax.adamw(cfg.lr_reward),
        "dynamics": optax.adamw(cfg.lr_dynamics) if cfg.continue_dynamics_loss else optax.set_to_zero(),
    }
    
    opt_states = {
        "policy": tx_dict["policy"].init(policy_params),
        "reward": tx_dict["reward"].init(reward_params),
        "dynamics": tx_dict["dynamics"].init(dynamics_params),
    }
    
    # Logging & checkpointing
    logger = MetricLogger(
        use_wandb=cfg.use_wandb,
        log_every=cfg.log_every,
        max_steps=cfg.max_steps,
        wandb_obj=wandb,
    )
    mngr = make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every)
    
    # Try to restore checkpoint
    state_example = make_state(
        {"policy": policy_params, "reward": reward_params, "dynamics": dynamics_params},
        {"policy": opt_states["policy"], "reward": opt_states["reward"], "dynamics": opt_states["dynamics"]},
        rng,
        step=0
    )
    meta = {"cfg": asdict(cfg)}
    
    restored = try_restore(mngr, state_example, meta)
    start_step = 0
    if restored is not None:
        latest_step, r = restored
        policy_params = r.state["params"]["policy"]
        reward_params = r.state["params"]["reward"]
        dynamics_params = r.state["params"]["dynamics"]
        opt_states = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"])
        print(f"[ckpt] Restored step {latest_step}")
    
    # Dataset
    dataset = make_iterator(cfg.dataset)
    
    # Training loop
    pbar = tqdm(enumerate(dataset, start=start_step), total=cfg.max_steps)
    for step, batch in pbar:
        if step >= cfg.max_steps:
            break
        
        rng, step_key = jax.random.split(rng)
        
        # Training step
        new_params, opt_states, metrics = train_step(
            tokenizer=tokenizer,
            dynamics=dynamics,
            task_embedder=task_embedder,
            policy_head=policy_head,
            reward_head=reward_head,
            tx_dict=tx_dict,
            tokenizer_vars=tokenizer_vars,
            dynamics_params=new_params.get("dynamics", dynamics_params) if step > start_step else dynamics_params,
            dynamics_constants=dynamics_constants,
            policy_params=new_params.get("policy", policy_params) if step > start_step else policy_params,
            reward_params=new_params.get("reward", reward_params) if step > start_step else reward_params,
            opt_states=opt_states,
            batch=batch,
            rng=step_key,
            step=step,
            k_max=dynamics.config.k_max,
            L_mtp=8,
            bootstrap_start=cfg.bootstrap_start,
            continue_dynamics_loss=cfg.continue_dynamics_loss,
            dynamics_loss_weight=cfg.dynamics_loss_weight,
        )
        
        # Update params
        dynamics_params = new_params["dynamics"]
        policy_params = new_params["policy"]
        reward_params = new_params["reward"]
        
        # Logging
        if logger.should_log(step):
            logger.log(step, metrics=metrics, pbar=pbar)
        
        # Save checkpoint
        state = make_state(new_params, opt_states, rng, step)
        maybe_save(mngr, step, state, meta)
    
    # Finish wandb run
    if cfg.use_wandb and wandb.run is not None:
        wandb.finish()
    
    print("[done] Agent finetuning complete!")


@hydra.main(version_base=None, config_path="../configs", config_name="bc_rew")
def main(cfg: DictConfig):
    schema = OmegaConf.structured(BCRewConfig)
    cfg = OmegaConf.merge(schema, cfg)
    agent_cfg = OmegaConf.to_object(cfg)
    run(agent_cfg)


if __name__ == "__main__":
    main()
