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
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from flax.typing import VariableDict
import hydra
import jax
import jax.numpy as jnp
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
    run_evaluation,
    run_agent_visualization,
    shortcut_forcing_step,
)
from dreamer.utils import (
    _ensure_dir,
    make_manager,
    make_state,
    maybe_save,
    try_restore,
    count_parameters_by_component,
    get_lr_schedule
)

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)

# disable preallocation completely
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# ---------------------------
# Hashable optimizer container
# ---------------------------

@dataclass(frozen=True)
class OptimizerContainer:
    """Hashable container for optimizers to pass as static argument to JIT."""
    task_embedder: optax.GradientTransformationExtraArgs
    policy: optax.GradientTransformationExtraArgs
    reward: optax.GradientTransformationExtraArgs
    dynamics: optax.GradientTransformationExtraArgs
    

# ---------------------------
# Multi-token prediction (MTP) helpers
# ---------------------------

def gather_future_actions(actions_bt: jnp.ndarray, L: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Gather future actions for multi-token prediction.
    
    At timestep t, predicts actions[t+1], actions[t+2], ..., actions[t+L]
    (Following Dreamer convention: action a_i happens before state s_i)
    
    Note: Paper equation uses n=0..L, but with Dreamer's convention where a_t is the 
    action TO TAKE from state s_t, we predict L future actions starting from a_{t+1}.
    
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
    f"""
    Gather future rewards for multi-token prediction.
    
    At timestep t, predicts rewards[t+1], ..., rewards[t+L]
    (Following Dreamer convention: r[t+1] is the reward from executing a[t+1] from h_t)
    
    Args:
        rewards_bt: (B, T) reward values
        L: number of future steps to predict
        
    Returns:
        rewards_btL: (B, T, L) future rewards
        valid_btL: (B, T, L) mask (0 for invalid)
    """
    B, T = rewards_bt.shape
    rewards_pad = jnp.pad(rewards_bt, ((0, 0), (0, L)), constant_values=jnp.nan)
    
    offsets = jnp.arange(1, L + 1)  # [1, 2, ..., L]
    indices = jnp.arange(T)[:, None] + offsets[None, :]  # (T, L)
    rewards_btL = rewards_pad[:, indices]  # (B, T, L)
    
    # Valid when: t >= 0 AND 0 <= t+offset < T
    valid_btL = (indices >= 0) & (indices < T)
    valid_btL = jnp.broadcast_to(valid_btL[None, :, :], (B, T, L))
    
    return rewards_btL, valid_btL


# ---------------------------
# Training step
# ---------------------------

@partial(jax.jit, static_argnames=("tokenizer", "dynamics", "task_embedder", "policy_head", "reward_head", "optimizers", "k_max", "L_mtp", "B_self", "packing_factor"))
def train_step(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    task_embedder: TaskEmbedder,
    policy_head: PolicyHeadMTP,
    reward_head: RewardHeadMTP,
    optimizers: OptimizerContainer,
    # State
    dynamics_params: VariableDict,
    dynamics_constants: VariableDict,
    task_embedder_params: VariableDict,
    policy_params: VariableDict,
    reward_params: VariableDict,
    reward_constants: VariableDict,
    tokenizer_vars: VariableDict,
    opt_states,
    # Data
    videos: jax.Array,
    batch,
    rng: jax.Array,
    # Config
    k_max: int,
    L_mtp: int,
    B_self: int,
    packing_factor: int,
    loss_weight_shortcut: float,
    loss_weight_policy: float,
    loss_weight_reward: float,
):
    """
    Agent finetuning step with BC + reward prediction + optional dynamics loss.
    
    Args:
        tokenizer: Tokenizer model (frozen, passed as static arg)
        Models: dynamics, task_embedder, policy_head, reward_head
        tx_dict: Dict of optimizers for each component
        State: parameters and optimizer states
        videos: Video data to encode
        batch: Data batch with actions, tasks, rewards
        rng: Random key
        Config: hyperparameters
        
    Returns:
        new_params, new_opt_states, metrics
    """
    # Split RNG for tokenizer and dynamics
    rng, tokenizer_key, dyn_key = jax.random.split(rng, 3)
    
    # Encode videos to latents (frozen tokenizer - now inside train_step for JIT)
    latents, _ = tokenizer.apply(
        tokenizer_vars,
        videos,
        packing_factor=packing_factor,
        method=tokenizer.encode,
        rngs={"mae": tokenizer_key},
    )
    
    B, T_video, _, _ = latents.shape
    
    # 1. Create task-conditioned agent tokens
    # For now, pass task ID 0 for all samples in batch
    task = jnp.zeros((B,), dtype=jnp.int32)
    agent_tokens_bt = task_embedder.apply({"params": task_embedder_params}, task=task, B=B, T=T_video)
    
    # 3. Gather future actions and rewards for MTP
    actions_btL, actions_valid = gather_future_actions(batch["actions"], L_mtp)
    rewards_btL, rewards_valid = gather_future_rewards(batch["rewards"], L_mtp)
    
    # 4. Define combined loss
    def loss_fn(pol_p, rew_p, dyn_p):
        # Dynamics loss (also returns hidden states for BC/reward training)
        dyn_vars = {"params": dyn_p, "constants": dynamics_constants}
        
        dyn_losses, dyn_aux = shortcut_forcing_step(dynamics.apply, dyn_vars, batch["actions"], latents, dyn_key, k_max, B_self=B_self, agent_tokens=agent_tokens_bt)
        dynamics_loss, h_states = dyn_losses['total'], dyn_aux['h_states']
        
        # TODO: See if we should compute the losses and the gradients sequentially (see figure 2 of https://arxiv.org/pdf/2404.19737 and comment in pull request #16)
        policy_loss = compute_policy_loss(policy_head, pol_p, h_states, actions_btL, actions_valid)
        reward_loss, reward_metrics = compute_reward_loss(reward_head, {"params": rew_p, "constants": reward_constants}, h_states, rewards_btL, rewards_valid)
        
        # Combine losses with weights
        w_policy_loss = loss_weight_policy * policy_loss
        w_reward_loss = loss_weight_reward * reward_loss
        w_dynamics_loss = loss_weight_shortcut * dynamics_loss
        total_loss = w_policy_loss + w_reward_loss + w_dynamics_loss
        
        # Filter out non-scalar metrics (h_states is used above but shouldn't be logged)
        aux = {
            # Unweighted losses
            "policy_loss": policy_loss,
            "reward_loss": reward_loss,
            "dynamics_loss": dynamics_loss,
            # Weighted losses
            "w_policy_loss": w_policy_loss,
            "w_reward_loss": w_reward_loss,
            "w_dynamics_loss": w_dynamics_loss,
            # Other metrics
            "flow_mse": dyn_aux["flow_mse"],
            "bootstrap_mse": dyn_aux["bootstrap_mse"],
            **reward_metrics
        }
        
        return total_loss, aux
    
    # 5. Compute gradients
    # Only update policy and reward params (dynamics can be frozen or finetuned)
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)
    (loss_val, metrics), (pol_grads, rew_grads, dyn_grads) = grad_fn(
        policy_params, reward_params, dynamics_params
    )
    
    # 6. Apply updates
    pol_updates, new_pol_opt = optimizers.policy.update(pol_grads, opt_states["policy"], policy_params)
    new_policy_params = optax.apply_updates(policy_params, pol_updates)
    
    rew_updates, new_rew_opt = optimizers.reward.update(rew_grads, opt_states["reward"], reward_params)
    new_reward_params = optax.apply_updates(reward_params, rew_updates)
    
    # Dynamics update (always compute, but optimizer is set_to_zero if not continuing)
    dyn_updates, new_dyn_opt = optimizers.dynamics.update(dyn_grads, opt_states["dynamics"], dynamics_params)
    new_dynamics_params = optax.apply_updates(dynamics_params, dyn_updates)
    
    new_params = {"task_embedder": task_embedder_params, "dynamics": new_dynamics_params, "policy": new_policy_params, "reward": new_reward_params}
    new_opt_states = {"task_embedder": opt_states["task_embedder"], "dynamics": new_dyn_opt, "policy": new_pol_opt, "reward": new_rew_opt}
    
    return new_params, new_opt_states, metrics


# ---------------------------
# Main
# ---------------------------

def run(cfg: BCRewConfig):
    """Main training loop for agent finetuning."""
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
    
    # Load pretrained tokenizer and dynamics
    rng = jax.random.PRNGKey(0)
    print(f"[setup] Loading pretrained dynamics from {cfg.dynamics_ckpt}")
    print(f"[setup] Loading pretrained tokenizer from {cfg.tokenizer_ckpt}")
    dynamics, dynamics_vars, dynamics_cfg, tokenizer, tokenizer_vars, tokenizer_cfg = Dynamics.from_pretrained(cfg.dynamics_ckpt)
    dynamics_params = dynamics_vars["params"]
    dynamics_constants = dynamics_vars.get("constants", FrozenDict())
    
    # Initialize task embedder, policy, and reward heads
    print("[setup] Initializing agent components")
    task_embedder = TaskEmbedder(d_model=dynamics.config.d_model, n_agent=cfg.n_agent, use_ids=cfg.use_task_ids, n_tasks=cfg.n_tasks)
    policy_head = PolicyHeadMTP(d_model=dynamics.config.d_model, action_dim=dynamics.config.action_dim, L=cfg.L)
    reward_head = RewardHeadMTP(d_model=dynamics.config.d_model, L=cfg.L, num_bins=cfg.num_reward_bins, log_low=cfg.reward_log_low, log_high=cfg.reward_log_high)
    
    # Initialize parameters
    rng, task_key, pol_key, rew_key = jax.random.split(rng, 4)
    
    # Dummy inputs for initialization
    dummy_h = jnp.zeros((1, 4, cfg.L, dynamics.config.d_model))  # (B, T, D) for heads (agent dim already pooled)
    dummy_task = jnp.zeros((1,), dtype=jnp.int32) if cfg.use_task_ids else jnp.zeros((1, cfg.n_tasks))
    task_embedder_params = task_embedder.init(task_key, task=dummy_task, B=1, T=4)["params"]
    policy_params = policy_head.init(pol_key, dummy_h, deterministic=True)["params"]
    reward_vars = reward_head.init(rew_key, dummy_h, deterministic=True)
    reward_params = reward_vars["params"]
    reward_constants = reward_vars.get("constants", FrozenDict())
    
    # Optimizers
    adamw = partial(optax.adamw, b1=0.9, b2=0.9, weight_decay=1e-4)
    optimizers = OptimizerContainer(
        task_embedder=adamw(cfg.lr_policy),  # Use same LR as policy
        policy=adamw(cfg.lr_policy),
        reward=adamw(cfg.lr_reward),
        dynamics=adamw(cfg.lr_dynamics)
    )
    opt_states = {
        "task_embedder": optimizers.task_embedder.init(task_embedder_params),
        "policy": optimizers.policy.init(policy_params),
        "reward": optimizers.reward.init(reward_params),
        "dynamics": optimizers.dynamics.init(dynamics_params),
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
        {"task_embedder": task_embedder_params, "policy": policy_params, "reward": reward_params, "dynamics": dynamics_params},
        {"task_embedder": opt_states["task_embedder"], "policy": opt_states["policy"], "reward": opt_states["reward"], "dynamics": opt_states["dynamics"]},
        rng,
        step=0
    )
    meta = {"cfg": asdict(cfg)}
    
    restored = try_restore(mngr, state_example, meta)
    start_step = 0
    if restored is not None:
        latest_step, r = restored
        params = r.state["params"]
        opt_states = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"])
        print(f"[ckpt] Restored step {latest_step}")
    else:
        # Initialize params dict as single source of truth
        params = {
            "task_embedder": task_embedder_params,
            "policy": policy_params,
            "reward": reward_params,
            "dynamics": dynamics_params,
        }
    param_counts = count_parameters_by_component(params)
    print(f"Parameter counts: {param_counts}")

    # Dataset
    dataset = make_iterator(tokenizer_cfg.dataset)

    # Training loop
    pbar = tqdm(enumerate(dataset, start=start_step), total=cfg.max_steps)
    for step, batch in pbar:
        if step >= cfg.max_steps:
            break
        
        rng, step_key = jax.random.split(rng, 2)
        
        # Get videos and compute batch size
        videos = batch["videos"]
        actions = batch["actions"]
        B_self = int(videos.shape[0] * cfg.self_fraction) * (step >= cfg.bootstrap_start)
        # val_videos = batch["videos"]
        # val_rewards = batch["rewards"]

        # if (batch['rewards'] >= 10).any():
        #     print("debugging step", step)
        #     import ipdb; ipdb.set_trace()
        # else:
        #     print("skipping step", step)
        #     continue
        # reward_vars = {"params": params["reward"], "constants": reward_constants}
        # run_agent_visualization(
        #     cfg=cfg,
        #     tokenizer_cfg=tokenizer_cfg,
        #     step=step,
        #     tokenizer=tokenizer,
        #     tokenizer_vars=tokenizer_vars,
        #     dynamics=dynamics,
        #     dynamics_params=params["dynamics"],
        #     dynamics_constants=dynamics_constants,
        #     task_embedder=task_embedder,
        #     task_embedder_params=params["task_embedder"],
        #     reward_head=reward_head,
        #     reward_vars=reward_vars,
        #     policy_head=policy_head,
        #     policy_params=params["policy"],
        #     val_videos=val_videos,
        #     val_actions=jnp.asarray(actions),
        #     val_rewards=val_rewards,
        #     vis_dir=vis_dir,
        #     rng=rng,
        # )    
        params, opt_states, metrics = train_step(
            # Models
            tokenizer=tokenizer,
            dynamics=dynamics,
            task_embedder=task_embedder,
            policy_head=policy_head,
            reward_head=reward_head,
            optimizers=optimizers,
            # State
            dynamics_params=params["dynamics"],
            dynamics_constants=dynamics_constants,
            task_embedder_params=params["task_embedder"],
            policy_params=params["policy"],
            reward_params=params["reward"],
            reward_constants=reward_constants,
            tokenizer_vars=tokenizer_vars,
            opt_states=opt_states,
            # Data
            videos=videos,
            batch=batch,
            rng=step_key,
            # Config
            k_max=dynamics.config.k_max,
            L_mtp=cfg.L,
            B_self=B_self,
            packing_factor=dynamics_cfg.packing_factor,
            loss_weight_shortcut=cfg.loss_weight_shortcut,
            loss_weight_policy=cfg.loss_weight_policy,
            loss_weight_reward=cfg.loss_weight_reward,
        )
        
        # Logging
        if logger.should_log(step):
            logger.log(step, metrics=metrics, pbar=pbar)
        
        # Save checkpoint
        state = make_state(params, opt_states, rng, step)
        maybe_save(mngr, step, state, meta)
        
        # Periodic lightweight AR eval
        if cfg.write_video_every and (step % cfg.write_video_every == 0) and step > 0:
            # Use current batch as validation data (simplest approach)
            val_videos = batch["videos"]
            val_rewards = batch["rewards"]
            reward_vars = {"params": params["reward"], "constants": reward_constants}
            run_agent_visualization(
                cfg=cfg,
                tokenizer_cfg=tokenizer_cfg,
                step=step,
                tokenizer=tokenizer,
                tokenizer_vars=tokenizer_vars,
                dynamics=dynamics,
                dynamics_params=params["dynamics"],
                dynamics_constants=dynamics_constants,
                task_embedder=task_embedder,
                task_embedder_params=params["task_embedder"],
                reward_head=reward_head,
                reward_vars=reward_vars,
                policy_head=policy_head,
                policy_params=params["policy"],
                val_videos=val_videos,
                val_actions=jnp.asarray(actions),
                val_rewards=val_rewards,
                vis_dir=vis_dir,
                rng=rng,
            )    
    # Finish wandb run
    if cfg.use_wandb and wandb.run is not None:
        wandb.finish()
    
    print("[done] Agent finetuning complete!")


@hydra.main(version_base=None, config_path="../configs", config_name="heads")
def main(cfg: DictConfig):
    schema = OmegaConf.structured(BCRewConfig)
    cfg = OmegaConf.merge(schema, cfg)
    agent_cfg = OmegaConf.to_object(cfg)
    run(agent_cfg)


if __name__ == "__main__":
    main()
