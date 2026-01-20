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

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.configs import HeadsConfig
from dreamer.data import make_iterator
from dreamer.logging import build_logger
from dreamer.models import Dynamics, PolicyHeadMTP, RewardHeadMTP, TaskEmbedder, Tokenizer
from dreamer.parallel import build_parallel
from dreamer.training import (
    compute_policy_loss,
    compute_reward_loss,
    run_evaluation,
    shortcut_forcing_step,
)
from dreamer.checkpointing import (
    DynamicsCheckpointBundle,
    HeadsCheckpointBundle,
    build_checkpoint_manager,
)
from dreamer.utils import (
    build_lr_schedule,
    build_optimizer,
    setup_training_directories,
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

@nnx.jit(static_argnames=("packing_factor", "k_max", "L_mtp", "B_self"))
def encode_and_train_step(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    task_embedder: TaskEmbedder,
    policy_head: PolicyHeadMTP,
    reward_head: RewardHeadMTP,
    dynamics_optimizer: nnx.Optimizer,
    task_embedder_optimizer: nnx.Optimizer,
    policy_optimizer: nnx.Optimizer,
    reward_optimizer: nnx.Optimizer,
    videos: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    *,
    tokenizer_key: jax.Array,
    master_key: jax.Array,
    step: int,
    packing_factor: int,
    k_max: int,
    L_mtp: int,
    B_self: int,
    dynamics_loss_weight: float,
) -> dict:
    """
    Encode videos and run training step.

    Combines frozen tokenizer encoding with agent finetuning in a single JIT.
    """
    # Phase 1: Encode (tokenizer frozen)
    rngs = nnx.Rngs(mae=tokenizer_key)
    latents, _ = tokenizer.encode(videos, packing_factor=packing_factor,
                                  deterministic=True, rngs=rngs)

    # Phase 2: Train
    metrics = train_step(
        dynamics, task_embedder, policy_head, reward_head,
        dynamics_optimizer, task_embedder_optimizer,
        policy_optimizer, reward_optimizer,
        latents, actions, rewards,
        master_key=master_key, step=step,
        k_max=k_max, L_mtp=L_mtp, B_self=B_self,
        dynamics_loss_weight=dynamics_loss_weight
    )
    return metrics


@nnx.jit(static_argnames=("k_max", "L_mtp", "B_self"))
def train_step(
    dynamics: Dynamics,
    task_embedder: TaskEmbedder,
    policy_head: PolicyHeadMTP,
    reward_head: RewardHeadMTP,
    dynamics_optimizer: nnx.Optimizer,
    task_embedder_optimizer: nnx.Optimizer,
    policy_optimizer: nnx.Optimizer,
    reward_optimizer: nnx.Optimizer,
    latents: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    *,
    master_key: jax.Array,
    step: int,
    k_max: int,
    L_mtp: int,
    B_self: int,
    dynamics_loss_weight: float,
) -> dict:
    """
    Agent finetuning step with BC + reward prediction + optional dynamics loss.

    Models are updated in place by their respective optimizers.

    Returns:
        metrics: Dict of scalar metrics for logging
    """
    # Generate step-specific key
    step_key = jax.random.fold_in(master_key, step)
    B, T_video, _, _ = latents.shape

    # Create task-conditioned agent tokens
    # For now, pass task ID 0 for all samples in batch
    task = jnp.zeros((B,), dtype=jnp.int32)
    agent_tokens_bt = task_embedder(task=task, B=B, T=T_video)

    # Gather future actions and rewards for MTP
    actions_btL, actions_valid = gather_future_actions(actions, L_mtp)
    rewards_btL, rewards_valid = gather_future_rewards(rewards, L_mtp)

    # Define combined loss
    # Takes all models as a tuple to enable differentiation w.r.t. all of them
    def loss_fn(models):
        dyn, task, pol, rew = models
        # Dynamics loss (also returns hidden states for BC/reward training)
        dyn_losses, dyn_aux = shortcut_forcing_step(
            dyn, actions, latents, step_key, k_max,
            B_self=B_self, task_embeddings=agent_tokens_bt
        )
        dynamics_loss, h_states = dyn_losses['total'], dyn_aux['h_states']

        # TODO: See if we should compute the losses and the gradients sequentially (see figure 2 of https://arxiv.org/pdf/2404.19737 and comment in pull request #16)
        # TODO: put gather_future_rewards and gather_future_actions inside of the compute_policy_loss function and compute_future_actions functions
        policy_loss = compute_policy_loss(pol, h_states, actions_btL, actions_valid)
        reward_loss, reward_metrics = compute_reward_loss(rew, h_states, rewards_btL, rewards_valid)

        # Combine losses
        total_loss = policy_loss + reward_loss + dynamics_loss_weight * dynamics_loss

        aux = {"policy_loss": policy_loss, "reward_loss": reward_loss, "dynamics_loss": dynamics_loss, "flow_mse": dyn_aux["flow_mse"], "bootstrap_mse": dyn_aux["bootstrap_mse"], **reward_metrics}

        return total_loss, aux

    # Compute gradients (pass all models as a tuple)
    (loss_val, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        (dynamics, task_embedder, policy_head, reward_head)
    )

    # Update all models with their respective optimizers
    dynamics_optimizer.update(dynamics, grads[0])
    task_embedder_optimizer.update(task_embedder, grads[1])
    policy_optimizer.update(policy_head, grads[2])
    reward_optimizer.update(reward_head, grads[3])

    return metrics


# ---------------------------
# Main
# ---------------------------

def run(cfg: HeadsConfig):
    """Main training loop for agent finetuning."""
    # Setup
    run_dir, ckpt_dir, vis_dir = setup_training_directories(cfg)
    
    # Logging
    logger = build_logger(
        logger_cfg=cfg.logger,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(run_dir),
    )

    # Parallelism
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with logger, jax.set_mesh(mesh):
        
        # Load pretrained tokenizer and dynamics
        key = jax.random.key(cfg.seed)
        rng, init_key = jax.random.split(key)

        print(f"[setup] Loading pretrained dynamics and tokenizer from {cfg.dynamics_ckpt}")
        dynamics_bundle = DynamicsCheckpointBundle.from_pretrained(cfg.dynamics_ckpt, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        dynamics = dynamics_bundle.dynamics
        tokenizer = dynamics_bundle.tokenizer
        dynamics_cfg = dynamics.cfg
        tokenizer_cfg = tokenizer.cfg

        # Initialize task embedder, policy, and reward heads
        print("[setup] Initializing agent components")
        rng, task_key, pol_key, rew_key = jax.random.split(rng, 4)

        task_embedder = TaskEmbedder(cfg.task_embedder, mesh_rules=mesh_rules, rngs=nnx.Rngs(task_key))
        policy_head = PolicyHeadMTP(cfg.policy_head, mesh_rules=mesh_rules, rngs=nnx.Rngs(pol_key))
        reward_head = RewardHeadMTP(cfg.reward_head, mesh_rules=mesh_rules, rngs=nnx.Rngs(rew_key))

        # Build learning rate schedules
        d_model = dynamics.cfg.d_model
        lr_schedule_policy = build_lr_schedule(cfg.lr_schedule_policy, d_model=d_model)
        lr_schedule_reward = build_lr_schedule(cfg.lr_schedule_reward, d_model=d_model)
        lr_schedule_dynamics = build_lr_schedule(cfg.lr_schedule_dynamics, d_model=d_model)

        # Build optimizers
        task_embedder_optimizer = build_optimizer(cfg.optimizer, task_embedder, lr_schedule_policy)
        policy_optimizer = build_optimizer(cfg.optimizer, policy_head, lr_schedule_policy)
        reward_optimizer = build_optimizer(cfg.optimizer, reward_head, lr_schedule_reward)
        dynamics_optimizer = build_optimizer(cfg.optimizer, dynamics, lr_schedule_dynamics)

        # Data iterator
        train_dataloader = make_iterator(cfg.dataset)
        train_iterator = iter(train_dataloader)  # type: ignore

        # Create checkpoint bundle (includes frozen tokenizer for self-contained checkpoints)
        bundle = HeadsCheckpointBundle(
            tokenizer=tokenizer,
            dynamics=dynamics,
            task_embedder=task_embedder,
            policy_head=policy_head,
            reward_head=reward_head,
            dynamics_optimizer=dynamics_optimizer,
            task_embedder_optimizer=task_embedder_optimizer,
            policy_optimizer=policy_optimizer,
            reward_optimizer=reward_optimizer,
        )

        # Checkpointing
        with build_checkpoint_manager(
            cfg.ckpt, ckpt_dir,
            item_names=HeadsCheckpointBundle.get_item_names()
        ) as checkpoint_manager:

            # Restore from checkpoint
            start_step, bundle, train_iterator, rng = bundle.restore(
                checkpoint_manager, train_iterator, rng
            )

            # Training loop
            pbar = tqdm(enumerate(train_iterator, start=start_step), initial=start_step, total=cfg.max_steps)
            for step, batch in pbar:
                if step >= cfg.max_steps:
                    break

                # Split RNG
                rng, tokenizer_key, master_key = jax.random.split(rng, num=3)

                # Shard batch data
                videos = jax.device_put(batch["videos"], data_sharding)
                actions = jax.device_put(batch["actions"], data_sharding)
                rewards = jax.device_put(batch["rewards"], data_sharding)

                # Action shifting: prepend "first action token" = 15
                B, T = actions.shape
                actions = jnp.concatenate((jnp.full_like(actions[:, 0:1], fill_value=15), actions[:, :-1]), axis=1)

                # Training step (encodes videos and trains)
                metrics = encode_and_train_step(
                    bundle.tokenizer, bundle.dynamics, bundle.task_embedder, bundle.policy_head, bundle.reward_head,
                    bundle.dynamics_optimizer, bundle.task_embedder_optimizer,
                    bundle.policy_optimizer, bundle.reward_optimizer,
                    videos, actions, rewards,
                    tokenizer_key=tokenizer_key,
                    master_key=master_key,
                    step=step,
                    packing_factor=dynamics_cfg.packing_factor,
                    k_max=bundle.dynamics.cfg.k_max,
                    L_mtp=cfg.policy_head.L,
                    B_self=(B // 2) * (step >= cfg.bootstrap_start),  # This will make the function compile twice. TODO: see if it's worth fixing this
                    dynamics_loss_weight=cfg.dynamics_loss_weight,
                )

                # Logging
                if logger.should_log(step):
                    metrics_cpu = jax.device_get(metrics)
                    logger.log(step, metrics=metrics_cpu, pbar=pbar)

                # Checkpointing
                bundle.maybe_save(checkpoint_manager, step, train_iterator, rng)

                # Periodic lightweight AR eval
                if cfg.write_video_every and (step % cfg.write_video_every == 0) and step > 0:
                    # Use current batch as validation data
                    val_videos = batch["videos"][:4]
                    val_actions = actions[:4]
                    run_evaluation(
                        cfg, step,
                        bundle.tokenizer, bundle.dynamics,
                        val_videos, val_actions, vis_dir, rng, logger, bundle.policy_head,
                        bundle.task_embedder
                    )

    print("[done] Agent finetuning complete!")


@hydra.main(version_base=None, config_path="../configs", config_name="heads")
def main(cfg: HeadsConfig):
    run(cfg)


if __name__ == "__main__":
    main()
