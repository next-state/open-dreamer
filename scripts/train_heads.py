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
from dreamer.actions import Actions, shift_actions
from dreamer.parallel import build_parallel
from dreamer.training import (
    compute_policy_loss,
    compute_reward_loss,
    run_evaluation,
    shortcut_forcing_step,
    RMSLossNormalizer,
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

def gather_future_actions(actions: Actions, BTL: tuple[int, int, int]) -> tuple[Actions, jnp.ndarray]:
    """
    Gather future actions for multi-token prediction.

    At timestep t, predicts actions[t+1], actions[t+2], ..., actions[t+L]
    (Following Dreamer convention: action a_i happens before state s_i)

    Note: Paper equation uses n=0..L, but with Dreamer's convention where a_t is the
    action TO TAKE from state s_t, we predict L future actions starting from a_{t+1}.

    Args:
        actions: Actions object with (B, T, ...) shaped arrays
        BTL: tuple containing (B, T, L)

    Returns:
        actions_btL: Actions object with (B, T, L, ...) future actions
        valid_btL: (B, T, L) mask (False for out-of-range)
    """
    B, T, L = BTL

    starts = jnp.arange(T)                               # [0, 1, ..., T-1]
    offsets = jnp.arange(1, L + 1)                       # [1, 2, ..., L]
    gather_indices = starts[:, None] + offsets[None, :]  # (T, L)
    valid_mask_tl = gather_indices < T
    valid_btL = jnp.broadcast_to(valid_mask_tl[None, ...], (B, T, L))

    def pad_and_gather(x):
        pad_width = [(0, 0), (0, L)] + [(0, 0)] * (x.ndim - 2)
        x_pad = jnp.pad(x, pad_width, mode='constant', constant_values=0)
        
        # We slice axis 1 using (T, L) indices -> resulting in (B, T, L, ...)
        return x_pad[:, gather_indices]
    
    actions_btL = jax.tree.map(pad_and_gather, actions)
    return actions_btL, valid_btL


def gather_future_rewards(rewards_bt: jnp.ndarray, BTL: tuple[int, int, int]) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    B, T, L = BTL
    
    starts = jnp.arange(T)                               # [0, 1, ..., T-1]
    offsets = jnp.arange(0, L)                           # [0, 1, ..., L-1]
    gather_indices = starts[:, None] + offsets[None, :]  # (T, L)
    valid_mask_tl = (gather_indices < T) & (gather_indices > 0)
    valid_btL = jnp.broadcast_to(valid_mask_tl[None, ...], (B, T, L))

    pad_width = ((0, 0), (0, L))
    rewards_pad = jnp.pad(rewards_bt, pad_width, constant_values=0.0)
    
    rewards_btL = rewards_pad[:, gather_indices]
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
    loss_normalizer: RMSLossNormalizer,
    videos: jax.Array,
    actions: Actions,
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
        loss_normalizer,
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
    loss_normalizer: RMSLossNormalizer,
    latents: jax.Array,
    actions: Actions,
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
    actions_btL, actions_valid = gather_future_actions(actions, (B, T_video, L_mtp))
    rewards_btL, rewards_valid = gather_future_rewards(rewards, (B, T_video, L_mtp))

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
        policy_losses = compute_policy_loss(pol, h_states, actions_btL, actions_valid)
        reward_loss, reward_metrics = compute_reward_loss(rew, h_states, rewards_btL, rewards_valid)


        raw_losses = {
            "reward": reward_loss,
            "dynamics": dynamics_loss,
            **{f"policy_{k}": v for k, v in policy_losses.items()},
        }
        normalized, rms_info = loss_normalizer(raw_losses)

        # Combine normalized losses
        policy_loss_normalized = sum(v for k, v in normalized.items() if k.startswith("policy_"))
        total_loss = policy_loss_normalized + normalized["reward"] + dynamics_loss_weight * normalized["dynamics"]

        aux = {
            "reward_loss": reward_loss,
            "dynamics_loss": dynamics_loss,
            "flow_mse": dyn_aux["flow_mse"],
            "bootstrap_mse": dyn_aux["bootstrap_mse"],
            **{f"policy_loss_{k}": v for k, v in policy_losses.items()},
            **{f"rms_{k}": v for k, v in rms_info.items()},
            **reward_metrics,
        }

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
        lr_schedule_policy = build_lr_schedule(cfg.lr_schedule_policy)
        lr_schedule_reward = build_lr_schedule(cfg.lr_schedule_reward)
        lr_schedule_dynamics = build_lr_schedule(cfg.lr_schedule_dynamics)

        # Build optimizers
        task_embedder_optimizer = build_optimizer(cfg.optimizer, task_embedder, lr_schedule_policy, d_model=d_model)
        policy_optimizer = build_optimizer(cfg.optimizer, policy_head, lr_schedule_policy, d_model=d_model)
        reward_optimizer = build_optimizer(cfg.optimizer, reward_head, lr_schedule_reward, d_model=d_model)
        dynamics_optimizer = build_optimizer(cfg.optimizer, dynamics, lr_schedule_dynamics, d_model=d_model)

        # Data iterator
        train_dataloader = make_iterator(cfg.dataset)
        train_iterator = iter(train_dataloader)

        # Create RMS loss normalizer for balancing policy/reward/dynamics losses
        loss_normalizer = RMSLossNormalizer(loss_names=[
            'policy_binary', 'policy_categorical', 'policy_continuous',
            'reward', 'dynamics'
        ])

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
            loss_normalizer=loss_normalizer,
        )

        # Checkpointing
        with build_checkpoint_manager(
            cfg.ckpt, ckpt_dir,
            item_names=HeadsCheckpointBundle.get_item_names(
                iterator_names=("train_dataloader_state",)
            )
        ) as checkpoint_manager:

            # Restore from checkpoint
            iterators = {"train_dataloader_state": train_iterator}
            start_step, bundle, iterators, rng = bundle.restore(
                checkpoint_manager, iterators, rng
            )
            train_iterator = iterators["train_dataloader_state"]

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

                # Action shifting: prepend "first action token" (noop) so action[t] aligns with state[t]
                actions = shift_actions(actions, cfg.dataset.categorical_action_dim)

                # Training step (encodes videos and trains)
                B = cfg.dataset.B
                metrics = encode_and_train_step(
                    bundle.tokenizer, bundle.dynamics, bundle.task_embedder, bundle.policy_head, bundle.reward_head,
                    bundle.dynamics_optimizer, bundle.task_embedder_optimizer,
                    bundle.policy_optimizer, bundle.reward_optimizer,
                    bundle.loss_normalizer,
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
                iterators = {"train_dataloader_state": train_iterator}
                bundle.maybe_save(checkpoint_manager, step, iterators, rng)

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
