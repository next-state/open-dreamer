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
from typing import Dict

from flax.typing import VariableDict
import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from flax.core import FrozenDict
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dreamer.configs import BCRewConfig
from dreamer.data import make_iterator
from dreamer.logging import MetricLogger
from dreamer.models import Dynamics, PolicyHeadMTP, RewardHeadMTP, TaskEmbedder, Tokenizer
from dreamer.parallel import ParallelContext
from dreamer.training import (
    compute_policy_loss,
    compute_reward_loss,
    run_evaluation,
    shortcut_forcing_step,
)
from dreamer.utils import (
    _ensure_dir,
    make_manager,
    make_state,
    maybe_save,
    try_restore,
    to_jnp_dtype,
)

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)

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

@partial(jax.jit, static_argnames=("dynamics", "task_embedder", "policy_head", "reward_head", "optimizers", "k_max", "L_mtp", "B_self"))
def train_step(
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
    opt_states,
    # Data
    latents: jax.Array,
    batch,
    rng: jax.Array,
    step: int,
    # Config
    k_max: int,
    L_mtp: int,
    B_self: int,
    dynamics_loss_weight: float,
):
    """
    Agent finetuning step with BC + reward prediction + optional dynamics loss.
    
    Args:
        Models: dynamics, task_embedder, policy_head, reward_head
        tx_dict: Dict of optimizers for each component
        State: parameters and optimizer states
        latents: Precomputed latents from tokenizer
        batch: Data batch with actions, tasks, rewards
        rng: Random key
        step: Training step number
        Config: hyperparameters
        
    Returns:
        new_params, new_opt_states, metrics
    """
    B, T_video, _, _ = latents.shape
    
    # Split RNG
    rng, dyn_key = jax.random.split(rng, 2)
    
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
        # TODO: put gather_future_rewards and gather_future_actions inside of the compute_policy_loss function and compute_future_actions functions
        policy_loss = compute_policy_loss(policy_head, pol_p, h_states, actions_btL, actions_valid)
        reward_loss = compute_reward_loss(reward_head, {"params": rew_p, "constants": reward_constants}, h_states, rewards_btL, rewards_valid)
        
        # Combine losses
        total_loss = policy_loss + reward_loss + dynamics_loss_weight * dynamics_loss
        
        # Filter out non-scalar metrics (h_states is used above but shouldn't be logged)
        aux = {"policy_loss": policy_loss, "reward_loss": reward_loss, "dynamics_loss": dynamics_loss, 
               "flow_mse": dyn_aux["flow_mse"], "bootstrap_mse": dyn_aux["bootstrap_mse"]}
        
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
    
    # Create parallel context for data parallelism
    ctx = ParallelContext.create(batch_size=cfg.dataset.B)
    
    # Load pretrained tokenizer and dynamics
    rng = jax.random.PRNGKey(0)
    print(f"[setup] Loading pretrained dynamics and tokenizer from {cfg.dynamics_ckpt}")
    dynamics, tokenizer = Dynamics.from_pretrained(cfg.dynamics_ckpt, ctx)
    dynamics_cfg = dynamics.config
    tokenizer_cfg = tokenizer.config
    # nnx.split with multiple filters returns (graphdef, state1, state2, ...)
    _, *dynamics_states = nnx.split(dynamics, nnx.Param, nnx.BatchStat, ...)
    dynamics_state = nnx.State.merge(*dynamics_states)
    dynamics_params = dynamics_state
    dynamics_vars = {"params": dynamics_params}
    dynamics_constants = FrozenDict()
    
    # For tokenizer, create similar compatibility wrappers
    _, *tokenizer_states = nnx.split(tokenizer, nnx.Param, nnx.BatchStat, ...)
    tokenizer_state = nnx.State.merge(*tokenizer_states)
    tokenizer_vars = {"params": tokenizer_state}
    
    # Initialize task embedder, policy, and reward heads
    print("[setup] Initializing agent components")
    task_embedder = TaskEmbedder(d_model=dynamics.config.d_model, n_agent=cfg.n_agent, use_ids=cfg.use_task_ids, n_tasks=cfg.n_tasks, dtype=cfg.dtype, param_dtype=cfg.param_dtype)
    policy_head = PolicyHeadMTP(d_model=dynamics.config.d_model, action_dim=dynamics.config.action_dim, L=cfg.L, dtype=cfg.dtype, param_dtype=cfg.param_dtype)
    reward_head = RewardHeadMTP(d_model=dynamics.config.d_model, L=cfg.L, num_bins=cfg.num_reward_bins, log_low=cfg.reward_log_low, log_high=cfg.reward_log_high, dtype=cfg.dtype, param_dtype=cfg.param_dtype)
    
    # Initialize parameters
    rng, task_key, pol_key, rew_key = jax.random.split(rng, 4)
    
    # Dummy inputs for initialization
    dummy_h = jnp.zeros((1, 4, cfg.L, dynamics.config.d_model), dtype=to_jnp_dtype(cfg.dtype))  # (B, T, D) for heads (agent dim already pooled)
    dummy_task = jnp.zeros((1,), dtype=jnp.int32) if cfg.use_task_ids else jnp.zeros((1, cfg.n_tasks))
    task_embedder_params = task_embedder.init(task_key, task=dummy_task, B=1, T=4)["params"]
    policy_params = policy_head.init(pol_key, dummy_h, deterministic=True)["params"]
    reward_vars = reward_head.init(rew_key, dummy_h, deterministic=True)
    reward_params = reward_vars["params"]
    reward_constants = reward_vars.get("constants", FrozenDict())
    
    # Optimizers
    optimizers = OptimizerContainer(
        task_embedder=optax.adamw(cfg.lr_policy),  # Use same LR as policy
        policy=optax.adamw(cfg.lr_policy),
        reward=optax.adamw(cfg.lr_reward),
        dynamics=optax.adamw(cfg.lr_dynamics)
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
    
    restored = try_restore(mngr, state_example, ctx, meta)
    start_step = 0
    if restored is not None:
        # Restored state is already sharded/replicated on GPUs via ctx
        latest_step, r = restored
        task_embedder_params = r.state["params"]["task_embedder"]
        policy_params = r.state["params"]["policy"]
        reward_params = r.state["params"]["reward"]
        dynamics_params = r.state["params"]["dynamics"]
        opt_states = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"])
        print(f"[ckpt] Restored step {latest_step} (loaded directly to GPU)")
    else:
        # No checkpoint - replicate initial state to GPUs
        task_embedder_params = ctx.replicate(task_embedder_params)
        policy_params = ctx.replicate(policy_params)
        reward_params = ctx.replicate(reward_params)
        opt_states = ctx.replicate(opt_states)
        print("[parallel] Replicated initial state to GPUs")
    
    # Replicate reward constants
    reward_constants = ctx.replicate(reward_constants)
    
    # Dataset
    dataset = make_iterator(cfg.dataset)
    
    # Training loop
    pbar = tqdm(enumerate(dataset, start=start_step), total=cfg.max_steps)
    for step, batch in pbar:
        if step >= cfg.max_steps:
            break
        
        rng, tokenizer_key, step_key = jax.random.split(rng, 3)
        
        # Shard batch data
        videos = ctx.shard_data(batch["videos"])
        actions = ctx.shard_data(batch["actions"])
        rewards = ctx.shard_data(batch["rewards"])
        
        # Generate keys matching batch size (one per sample)
        tokenizer_key = ctx.split_keys(tokenizer_key, count=videos.shape[0])
        step_key = ctx.split_keys(step_key, count=videos.shape[0])
        
        # Encode videos to latents (frozen tokenizer - outside train_step)
        B, T, H, W, C = videos.shape
        # shift the actions by one and put the "first action token" = 15 at the beginning 
        actions = jnp.concatenate((jnp.full_like(actions[:,0:1], fill_value = 15), actions[:,:-1]), axis=1) # TODO: pass this to the train step!
        latents, _ = tokenizer.apply(tokenizer_vars, videos, packing_factor=dynamics_cfg.packing_factor, rngs={"mae": tokenizer_key}, method=tokenizer.encode)
        
        # Create batch dict with sharded data
        batch_sharded = {"actions": actions, "rewards": rewards}
        
        # Training step
        new_params, opt_states, metrics = train_step(
            dynamics=dynamics,
            task_embedder=task_embedder,
            policy_head=policy_head,
            reward_head=reward_head,
            optimizers=optimizers,
            dynamics_params=new_params.get("dynamics", dynamics_params) if step > start_step else dynamics_params,
            dynamics_constants=dynamics_constants,
            task_embedder_params=new_params.get("task_embedder", task_embedder_params) if step > start_step else task_embedder_params,
            policy_params=new_params.get("policy", policy_params) if step > start_step else policy_params,
            reward_params=new_params.get("reward", reward_params) if step > start_step else reward_params,
            reward_constants=reward_constants,
            opt_states=opt_states,
            latents=latents,
            batch=batch_sharded,
            rng=step_key,
            step=step,
            k_max=dynamics.config.k_max,
            L_mtp=cfg.L,
            B_self=(B // 2)*(step >= cfg.bootstrap_start), # This will make the function compile twice. TODO: see if it's worth fixing this
            dynamics_loss_weight=cfg.dynamics_loss_weight,
        )
        
        # Update params
        dynamics_params = new_params["dynamics"]
        policy_params = new_params["policy"]
        reward_params = new_params["reward"]
        
        # Logging
        if logger.should_log(step):
            metrics_cpu = jax.device_get(metrics)
            logger.log(step, metrics=metrics_cpu, pbar=pbar)
        
        # Save sharded arrays directly
        state = make_state(new_params, opt_states, rng, step)
        maybe_save(mngr, step, state, meta)
        
        # Periodic lightweight AR eval
        if cfg.write_video_every and (step % cfg.write_video_every == 0) and step > 0:
            # Use current batch as validation data (simplest approach) - move to host
            val_videos = jax.device_get(batch["videos"][:4])
            val_actions = jax.device_get(actions[:4])
            run_evaluation(
                cfg, tokenizer_cfg, step, 
                tokenizer,
                dynamics,
                val_videos, val_actions, vis_dir, rng
            )
    
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
