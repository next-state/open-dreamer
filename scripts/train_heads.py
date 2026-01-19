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

import grain.checkpoint
import hydra
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.configs import HeadsConfig
from dreamer.data import make_iterator
from dreamer.logging import build_logger
from dreamer.models import Dynamics, PolicyHeadMTP, RewardHeadMTP, TaskEmbedder, Tokenizer
from dreamer.parallel import build_parallel
from dreamer.training import (
    LossRMSState,
    compute_policy_loss,
    compute_reward_loss,
    run_evaluation,
    shortcut_forcing_step,
    update_loss_rms,
)
from dreamer.utils import (
    build_checkpoint_manager,
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

@nnx.jit(static_argnames=("packing_factor", "k_max", "L_mtp", "B_self", "loss_weights"))
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
    rms_state: LossRMSState,
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
    loss_weights: tuple[tuple[str, float], ...],
) -> tuple[LossRMSState, dict]:
    """
    Encode videos and run training step.

    Combines frozen tokenizer encoding with agent finetuning in a single JIT.
    Uses RMS loss normalization for stable multi-task training (paper Section 3).
    """
    # Phase 1: Encode (tokenizer frozen)
    rngs = nnx.Rngs(mae=tokenizer_key)
    latents, _ = tokenizer.encode(videos, packing_factor=packing_factor,
                                  deterministic=True, rngs=rngs)

    # Phase 2: Train with RMS normalization
    rms_state, metrics = train_step(
        dynamics, task_embedder, policy_head, reward_head,
        dynamics_optimizer, task_embedder_optimizer,
        policy_optimizer, reward_optimizer,
        rms_state,
        latents, actions, rewards,
        master_key=master_key, step=step,
        k_max=k_max, L_mtp=L_mtp, B_self=B_self,
        loss_weights=loss_weights,
    )
    return rms_state, metrics


@nnx.jit(static_argnames=("k_max", "L_mtp", "B_self", "loss_weights"))
def train_step(
    dynamics: Dynamics,
    task_embedder: TaskEmbedder,
    policy_head: PolicyHeadMTP,
    reward_head: RewardHeadMTP,
    dynamics_optimizer: nnx.Optimizer,
    task_embedder_optimizer: nnx.Optimizer,
    policy_optimizer: nnx.Optimizer,
    reward_optimizer: nnx.Optimizer,
    rms_state: LossRMSState,
    latents: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    *,
    master_key: jax.Array,
    step: int,
    k_max: int,
    L_mtp: int,
    B_self: int,
    loss_weights: tuple[tuple[str, float], ...],
) -> tuple[LossRMSState, dict]:
    """
    Agent finetuning step with BC + reward prediction + optional dynamics loss.

    Uses RMS loss normalization (paper Section 3): Each loss is normalized by its
    running RMS estimate before combining with fixed weights. This allows training
    with multiple modalities/heads that have different loss scales.

    Models are updated in place by their respective optimizers.

    Args:
        loss_weights: Tuple of (name, weight) pairs for combining normalized losses.
            Example: (("policy", 1.0), ("reward", 1.0), ("dynamics", 0.1))

    Returns:
        rms_state: Updated LossRMSState with new running estimates
        metrics: Dict of scalar metrics for logging
    """
    # Convert loss_weights tuple to dict for easier access
    weights_dict = {name: weight for name, weight in loss_weights}

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

    # Get current RMS estimates (stop gradient so they don't affect backprop)
    rms_estimates = {
        name: jax.lax.stop_gradient(est)
        for name, est in rms_state.estimates.items()
    }

    # Define combined loss with RMS normalization
    # Takes all models as a tuple to enable differentiation w.r.t. all of them
    def loss_fn(models):
        dyn, task_emb, pol, rew = models
        # Dynamics loss (also returns hidden states for BC/reward training)
        dyn_losses, dyn_aux = shortcut_forcing_step(
            dyn, actions, latents, step_key, k_max,
            B_self=B_self, agent_tokens=agent_tokens_bt
        )
        dynamics_loss_raw, h_states = dyn_losses['total'], dyn_aux['h_states']

        # Policy and reward losses
        policy_loss_raw = compute_policy_loss(pol, h_states, actions_btL, actions_valid)
        reward_loss_raw = compute_reward_loss(rew, h_states, rewards_btL, rewards_valid)

        # Collect raw losses
        raw_losses = {
            "policy": policy_loss_raw,
            "reward": reward_loss_raw,
            "dynamics": dynamics_loss_raw,
        }

        # Normalize each loss by its running RMS estimate (paper Section 3)
        # The RMS estimate is stop_gradient'd so gradients flow through loss only
        normalized_losses = {}
        for name, loss in raw_losses.items():
            rms_est = rms_estimates.get(name, jnp.array(1.0))
            normalized_losses[name] = loss / (rms_est + 1e-8)

        # Combine normalized losses with fixed weights
        total_loss = jnp.array(0.0)
        for name, norm_loss in normalized_losses.items():
            weight = weights_dict.get(name, 1.0)
            total_loss = total_loss + weight * norm_loss

        aux = {
            "policy_loss": policy_loss_raw,
            "reward_loss": reward_loss_raw,
            "dynamics_loss": dynamics_loss_raw,
            "policy_loss_norm": normalized_losses["policy"],
            "reward_loss_norm": normalized_losses["reward"],
            "dynamics_loss_norm": normalized_losses["dynamics"],
            "flow_mse": dyn_aux["flow_mse"],
            "bootstrap_mse": dyn_aux["bootstrap_mse"],
            "raw_losses": raw_losses,  # For RMS update
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

    # Update RMS estimates with raw losses (after gradient computation)
    raw_losses = metrics.pop("raw_losses")
    new_rms_state, _ = update_loss_rms(rms_state, raw_losses, decay=0.999, warmup_steps=100)

    # Add RMS estimates to metrics for logging
    metrics["rms/policy"] = new_rms_state.estimates["policy"]
    metrics["rms/reward"] = new_rms_state.estimates["reward"]
    metrics["rms/dynamics"] = new_rms_state.estimates["dynamics"]

    return new_rms_state, metrics


# ---------------------------
# Main
# ---------------------------

def run(cfg: HeadsConfig):
    """Main training loop for agent finetuning."""
    # Setup
    run_dir, ckpt_dir, vis_dir, meta = setup_training_directories(cfg)
    
    # Logging
    logger = build_logger(
        logger_cfg=cfg.logger,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(run_dir),
    )

    # Parallelism
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with (
        logger,
        jax.set_mesh(mesh),
    ):
        
        # Load pretrained tokenizer and dynamics
        key = jax.random.key(cfg.seed)
        rng, init_key = jax.random.split(key)

        print(f"[setup] Loading pretrained dynamics and tokenizer from {cfg.dynamics_ckpt}")
        dynamics, tokenizer = Dynamics.from_pretrained(cfg.dynamics_ckpt, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        dynamics_cfg = dynamics.cfg
        tokenizer_cfg = tokenizer.cfg

        # Initialize task embedder, policy, and reward heads
        print("[setup] Initializing agent components")
        rng, task_key, pol_key, rew_key = jax.random.split(rng, 4)

        task_embedder = TaskEmbedder(
            d_model=dynamics.cfg.d_model, n_agent=cfg.n_agent,
            use_ids=cfg.use_task_ids, n_tasks=cfg.n_tasks,
            dtype=cfg.dtype, param_dtype=cfg.param_dtype,
            mesh_rules=mesh_rules, rngs=nnx.Rngs(task_key)
        )
        policy_head = PolicyHeadMTP(
            d_model=dynamics.cfg.d_model, action_dim=dynamics.cfg.action_dim,
            L=cfg.L, dtype=cfg.dtype, param_dtype=cfg.param_dtype,
            mesh_rules=mesh_rules, rngs=nnx.Rngs(pol_key)
        )
        reward_head = RewardHeadMTP(
            d_model=dynamics.cfg.d_model, L=cfg.L, num_bins=cfg.num_reward_bins,
            log_low=cfg.reward_log_low, log_high=cfg.reward_log_high,
            dtype=cfg.dtype, param_dtype=cfg.param_dtype,
            mesh_rules=mesh_rules, rngs=nnx.Rngs(rew_key)
        )

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

        # Initialize RMS loss normalization state (paper Section 3)
        # This normalizes losses by running RMS estimates before combining
        rms_state = LossRMSState.init(("policy", "reward", "dynamics"))

        # Loss weights for combining normalized losses
        # After RMS normalization, all losses have ~unit scale, so weights are intuitive
        loss_weights = (
            ("policy", cfg.loss_weight_policy),
            ("reward", cfg.loss_weight_reward),
            ("dynamics", cfg.dynamics_loss_weight),
        )

        # Data iterator
        train_dataloader = make_iterator(cfg.dataset)
        train_iterator = iter(train_dataloader)  # type: ignore

        # Checkpointing
        with build_checkpoint_manager(
            cfg.ckpt, ckpt_dir,
            item_names=(
                "dynamics_state", "dynamics_optimizer_state",
                "task_embedder_state", "task_embedder_optimizer_state",
                "policy_state", "policy_optimizer_state",
                "reward_state", "reward_optimizer_state",
                "rms_state",  # RMS loss normalization state
                "train_dataloader_state", "rngs", "meta"
            )
        ) as checkpoint_manager:
            # Restore from checkpoint
            step = checkpoint_manager.latest_step()
            if step is not None:
                # Extract states from all models/optimizers
                dynamics_state = nnx.state(dynamics)
                task_embedder_state = nnx.state(task_embedder)
                policy_state = nnx.state(policy_head)
                reward_state = nnx.state(reward_head)

                dynamics_opt_state = nnx.state(dynamics_optimizer)
                task_embedder_opt_state = nnx.state(task_embedder_optimizer)
                policy_opt_state = nnx.state(policy_optimizer)
                reward_opt_state = nnx.state(reward_optimizer)

                # RMS state as dict for checkpointing
                rms_state_dict = {"estimates": rms_state.estimates, "counts": rms_state.counts}

                # Create restore args composite
                restore_args = ocp.args.Composite(
                    dynamics_state=ocp.args.StandardRestore(dynamics_state),  # type: ignore
                    task_embedder_state=ocp.args.StandardRestore(task_embedder_state),  # type: ignore
                    policy_state=ocp.args.StandardRestore(policy_state),  # type: ignore
                    reward_state=ocp.args.StandardRestore(reward_state),  # type: ignore
                    dynamics_optimizer_state=ocp.args.StandardRestore(dynamics_opt_state),  # type: ignore
                    task_embedder_optimizer_state=ocp.args.StandardRestore(task_embedder_opt_state),  # type: ignore
                    policy_optimizer_state=ocp.args.StandardRestore(policy_opt_state),  # type: ignore
                    reward_optimizer_state=ocp.args.StandardRestore(reward_opt_state),  # type: ignore
                    rms_state=ocp.args.StandardRestore(rms_state_dict),  # type: ignore
                    train_dataloader_state=grain.checkpoint.CheckpointRestore(train_iterator),  # type: ignore
                    rngs=ocp.args.StandardRestore({"key": rng}),  # type: ignore
                )

                # Restore and update all models
                restored = checkpoint_manager.restore(step, args=restore_args)
                nnx.update(dynamics, restored["dynamics_state"])
                nnx.update(task_embedder, restored["task_embedder_state"])
                nnx.update(policy_head, restored["policy_state"])
                nnx.update(reward_head, restored["reward_state"])
                nnx.update(dynamics_optimizer, restored["dynamics_optimizer_state"])
                nnx.update(task_embedder_optimizer, restored["task_embedder_optimizer_state"])
                nnx.update(policy_optimizer, restored["policy_optimizer_state"])
                nnx.update(reward_optimizer, restored["reward_optimizer_state"])
                rms_restored = restored["rms_state"]
                rms_state = LossRMSState(rms_restored["estimates"], rms_restored["counts"])
                train_iterator = restored["train_dataloader_state"]
                rng = restored["rngs"]["key"]
                start_step = step + 1
                print(f"[ckpt] Restored step {step}")
            else:
                start_step = 0
                print("[ckpt] No checkpoint found, starting from scratch")

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

                # Training step (encodes videos and trains) with RMS loss normalization
                rms_state, metrics = encode_and_train_step(
                    tokenizer, dynamics, task_embedder, policy_head, reward_head,
                    dynamics_optimizer, task_embedder_optimizer,
                    policy_optimizer, reward_optimizer,
                    rms_state,
                    videos, actions, rewards,
                    tokenizer_key=tokenizer_key,
                    master_key=master_key,
                    step=step,
                    packing_factor=dynamics_cfg.packing_factor,
                    k_max=dynamics.cfg.k_max,
                    L_mtp=cfg.L,
                    B_self=(B // 2) * (step >= cfg.bootstrap_start),  # This will make the function compile twice. TODO: see if it's worth fixing this
                    loss_weights=loss_weights,
                )

                # Logging
                if logger.should_log(step):
                    metrics_cpu = jax.device_get(metrics)
                    logger.log(step, metrics=metrics_cpu, pbar=pbar)

                # Checkpointing (inline save)
                if checkpoint_manager.should_save(step):
                    # Extract states
                    dynamics_state = nnx.state(dynamics)
                    task_embedder_state = nnx.state(task_embedder)
                    policy_state = nnx.state(policy_head)
                    reward_state = nnx.state(reward_head)
                    dynamics_opt_state = nnx.state(dynamics_optimizer)
                    task_embedder_opt_state = nnx.state(task_embedder_optimizer)
                    policy_opt_state = nnx.state(policy_optimizer)
                    reward_opt_state = nnx.state(reward_optimizer)

                    # RMS state as dict for checkpointing
                    rms_state_dict = {"estimates": rms_state.estimates, "counts": rms_state.counts}

                    # Create save args composite
                    save_args = ocp.args.Composite(
                        dynamics_state=ocp.args.StandardSave(dynamics_state),  # type: ignore
                        task_embedder_state=ocp.args.StandardSave(task_embedder_state),  # type: ignore
                        policy_state=ocp.args.StandardSave(policy_state),  # type: ignore
                        reward_state=ocp.args.StandardSave(reward_state),  # type: ignore
                        dynamics_optimizer_state=ocp.args.StandardSave(dynamics_opt_state),  # type: ignore
                        task_embedder_optimizer_state=ocp.args.StandardSave(task_embedder_opt_state),  # type: ignore
                        policy_optimizer_state=ocp.args.StandardSave(policy_opt_state),  # type: ignore
                        reward_optimizer_state=ocp.args.StandardSave(reward_opt_state),  # type: ignore
                        rms_state=ocp.args.StandardSave(rms_state_dict),  # type: ignore
                        train_dataloader_state=grain.checkpoint.CheckpointSave(train_iterator),  # type: ignore
                        rngs=ocp.args.StandardSave({'key': rng}),  # type: ignore
                        meta=ocp.args.JsonSave(meta)  # type: ignore
                    )
                    checkpoint_manager.save(step, args=save_args)

                # Periodic lightweight AR eval
                if cfg.write_video_every and (step % cfg.write_video_every == 0) and step > 0:
                    # Use current batch as validation data
                    val_videos = batch["videos"][:4]
                    val_actions = actions[:4]
                    run_evaluation(
                        cfg, tokenizer_cfg, step,
                        tokenizer, dynamics,
                        val_videos, val_actions, vis_dir, rng, logger
                    )

    print("[done] Agent finetuning complete!")


@hydra.main(version_base=None, config_path="../configs", config_name="heads")
def main(cfg: HeadsConfig):
    run(cfg)


if __name__ == "__main__":
    main()
