"""
Refactored RL policy training using Flax NNX state management.

This version leverages Flax NNX's object-oriented API with mutable state handling,
eliminating the need for manual state threading and .apply() calls.

Key NNX features used (all references from https://flax.readthedocs.io/en/latest/):
- nnx.Module: Eager initialization with explicit state (nnx_basics.html)
- Direct __call__: "NNX modules are called directly like regular Python objects" (nnx_basics.html)
- nnx.Optimizer: Encapsulates optimizer state with reference semantics (optimizer.html)
- nnx.jit: Transform with automatic state handling (jax_and_nnx_transforms.html)
- nnx.value_and_grad: Gradient computation with state updates (nnx_basics.html)

Improvements over Linen-style implementation:
- ~50% less code due to automatic state management
- No .apply() calls - direct module invocation
- No manual opt_state threading
- Stateful operations update in-place
- No need for state container class - just use modules directly
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple
from functools import partial
from tqdm import tqdm
import time
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax import nnx

from dreamer.models import (
    Dynamics,
    Tokenizer,
    TaskEmbedder,
    PolicyHeadMTP,
    RewardHeadMTP,
    ValueHead,
)
from dreamer.training import (
    compute_td_lambda_returns,
    compute_value_loss,
    compute_pmpo_loss,
    symlog,
    symexp,
)
from dreamer.generation import latent_rollout, DenoiseSchedule
from dreamer.data import build_iterator
from dreamer.configs import RLConfig, DynamicsConfig
from dreamer.utils import make_manager
from dreamer.logging import MetricLogger

logging.getLogger('absl').setLevel(logging.WARNING)

class TrainableModels(nnx.Module):
    def __init__(self, policy: PolicyHeadMTP, value: ValueHead):
        self.policy = policy
        self.value = value

# ---------------------------
# Helper Functions
# ---------------------------

def load_pretrained_heads(cfg: RLConfig, dynamics_cfg) -> Tuple[TaskEmbedder, PolicyHeadMTP, RewardHeadMTP]:
    """
    Load pretrained heads (task_embedder, policy_bc, reward_head) from checkpoint.
    
    Args:
        cfg: RL training configuration
        dynamics_cfg: Dynamics model config (for d_model)
        
    Returns:
        Tuple of (task_embedder, policy_bc, reward_head) as NNX modules
    """
    mngr = make_manager(cfg.heads_ckpt, item_names=("meta", "task_embedder", "policy_head", "reward_head"))
    latest = mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {cfg.heads_ckpt}")

    # Initialize NNX modules
    task_embedder = TaskEmbedder(
        d_model=dynamics_cfg.d_model,
        use_ids=cfg.use_task_ids,
        n_tasks=cfg.n_tasks,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
        rngs=nnx.Rngs(0),
    )
    policy_bc = PolicyHeadMTP(
        d_model=dynamics_cfg.d_model,
        action_dim=cfg.action_dim,
        L=cfg.L,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
        rngs=nnx.Rngs(1),
    )
    reward_head = RewardHeadMTP(
        d_model=dynamics_cfg.d_model,
        L=cfg.L,
        num_bins=cfg.num_reward_bins,
        log_low=cfg.reward_log_low,
        log_high=cfg.reward_log_high,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
        rngs=nnx.Rngs(2),
    )
    
    # Get state structure for checkpoint loading
    task_state = nnx.state(task_embedder)
    pi_bc_state = nnx.state(policy_bc)
    rew_state = nnx.state(reward_head)
    
    # Load pretrained weights from checkpoint
    restored = mngr.restore(
        latest,
        args=ocp.args.Composite(
            meta=ocp.args.JsonRestore(),
            task_embedder=ocp.args.StandardRestore(task_state),
            policy_head=ocp.args.StandardRestore(pi_bc_state),
            reward_head=ocp.args.StandardRestore(rew_state),
        ),
    )
    
    # Update NNX modules with loaded parameters
    nnx.update(task_embedder, restored.task_embedder)
    nnx.update(policy_bc, restored.policy_head)
    nnx.update(reward_head, restored.reward_head)
    
    print(f"[Pretrained Heads] Loaded from checkpoint at step {latest}")
    
    return task_embedder, policy_bc, reward_head


def initialize_rl_training(cfg: RLConfig, ctx):
    """
    Initialize all models for RL training.
    
    NNX initialization pattern (from nnx_basics.html):
    "All the parameters of a Module are usually created eagerly in __init__"
    
    This function:
    1. Loads pretrained NNX modules from checkpoints
    2. Creates new NNX modules for trainable heads
    3. Creates optimizer for trainable models
    
    Args:
        cfg: RL training configuration
        ctx: Parallel context for distributed loading
        
    Returns:
        Tuple of all components needed for training
    """
    # ---------------------------
    # 1. Load pretrained models (returns NNX modules)
    # ---------------------------
    
    print(f"Loading pretrained models from {cfg.heads_ckpt}...")
    
    # Per models.py: Dynamics.from_pretrained returns (dynamics, tokenizer)
    # Both are NNX modules with parameters already loaded
    dynamics, tokenizer = Dynamics.from_pretrained(cfg.heads_ckpt, ctx)
    dynamics_cfg = dynamics.cfg
    
    # Load pretrained heads (task_embedder, policy_bc, reward_head)
    task_embedder, policy_bc, reward_head = load_pretrained_heads(cfg, dynamics_cfg)
    
    print(f"[Loaded] Dynamics, Tokenizer, TaskEmbedder, PolicyBC, RewardHead")
    
    # ---------------------------
    # 2. Initialize trainable heads (NNX modules)
    # ---------------------------
    
    # Clone the pretrained policy_bc to initialize the trainable policy
    # This gives us a better starting point than random initialization
    graphdef, state = nnx.split(policy_bc)
    policy_head = nnx.merge(graphdef, state)  # Clone by splitting and merging
    
    value_head = ValueHead(
        d_model=dynamics_cfg.d_model,
        L=cfg.n_agent,
        num_bins=cfg.num_value_bins,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
        mesh_rules=mesh_rules,
        rngs=nnx.Rngs(101),
    )
    
    trainable = TrainableModels(policy_head, value_head)
    print(f"[Initialized] Trainable PolicyHead and ValueHead")
    
    # ---------------------------
    # 3. Create NNX optimizer
    # ---------------------------
    
    # Create optimizer with reference to trainable models
    # Per optimizer.html: "The optimizer will maintain mutable references to the parameters"
    tx = optax.adam(cfg.lr)
    optimizer = nnx.Optimizer(trainable, tx, wrt=nnx.Param)
    
    print(f"[Created] Optimizer (lr={cfg.lr})")
    
    
    schedule = DenoiseSchedule.init(num_steps = 4, k_max=dynamics_cfg.k_max)
    
    rngs = nnx.Rngs(cfg.seed)
    step = nnx.Variable(0)  # Mutable step counter
    
    return (tokenizer, dynamics, task_embedder, policy_bc, reward_head, optimizer, schedule, cfg, rngs, step)


# ---------------------------
# Training Step
# ---------------------------

@nnx.jit
def train_step(
    # Frozen models
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    task_embedder: TaskEmbedder,
    policy_bc: PolicyHeadMTP,
    reward_head: RewardHeadMTP,
    # Trainable models (via optimizer)
    optimizer: nnx.Optimizer,
    # Infrastructure
    schedule: DenoiseSchedule,
    cfg: RLConfig,
    # Mutable state
    rngs: nnx.Rngs,
    step: nnx.Variable,
    # Batch data
    batch: dict,
) -> dict:
    """
    Single training step using NNX automatic state management.
    
    Per guides/jax_and_nnx_transforms.html:
    "@nnx.jit approach: Flax NNX transforms enable stateful-looking code.
    You pass nnx.Module instances directly and mutate them in place.
    The decorator handles state management internally."
    
    Per nnx_basics.html:
    "NNX modules are called directly like regular Python objects"
    
    This function:
    1. Calls NNX modules directly (no .apply())
    2. Mutates optimizer (which updates params)
    3. Mutates step counter
    4. Returns only metrics (state updates happen in-place)
    
    Args:
        All NNX modules and state are passed directly (mutated in-place)
        batch: Training batch with 'videos', 'actions', 'task_ids'
        
    Returns:
        metrics: Dict with losses and diagnostics
    """
    # Extract batch data
    videos = batch['videos']  # (B, T, H, W, C)
    actions = batch['actions']  # (B, T)
    task_ids = batch.get('task_ids', jnp.zeros((videos.shape[0],), dtype=jnp.int32))
    
    B, T = videos.shape[:2]
    T_ctx = cfg.context_length
    horizon = cfg.horizon
    
    # Get packing factor from dynamics config
    packing_factor = dynamics.cfg.packing_factor
    
    # Encode context frames
    z_ctx, _ = tokenizer.encode(
        videos[:, :T_ctx],
        deterministic=True,
        packing_factor=packing_factor,
    )  # (B, T_ctx, n_spatial, D_s)
    
    # Create agent tokens for context
    # Direct call to NNX module
    agent_tokens_ctx = task_embedder(
        task_ids,
        B, T_ctx,
    )  # (B, T_ctx, n_agent, d_model)
    
    # Define loss function
    def loss_fn(trainable: TrainableModels):
        """
        Compute loss for trainable models.
        
        Note: This function receives the trainable models container.
        The optimizer will compute gradients with respect to
        trainable.policy and trainable.value parameters.
        """
        # Extract trainable models
        policy_head = trainable.policy
        value_head = trainable.value
        
        # Get RNG for imagination
        rng_imag = rngs()
        
        rollout_result = latent_rollout(
            dynamics=dynamics,  
            policy=policy_head,  
            schedule=schedule,
            latents_ctx=z_ctx,
            actions_ctx=actions[:, :T_ctx],
            num_steps=horizon,
            rng=rng_imag,
            initial_task_embedding=agent_tokens_ctx,
            deterministic=False,
        )
        
        # Extract results
        imagined_actions = rollout_result['actions']  # (B, H)
        rollout_hidden = rollout_result['hidden_states']  # (B, H, n_agent, d_model)
        context_hidden = rollout_result['context_hidden']  # (B, T_ctx, n_agent, d_model)
        
        # Concatenate for full sequence
        hidden_states = jnp.concatenate([context_hidden, rollout_hidden], axis=1)
        # (B, T_ctx + H, n_agent, d_model)
        
        # Stop gradients for value/reward targets
        h_sg = jax.lax.stop_gradient(hidden_states)
        
        # ---------------------------
        # 2. Compute rewards 
        # ---------------------------
        # Predict rewards from rollout hidden states only
        # Rewards r_{t+1} are predicted from state s_t
        rew_logits, rew_centers = reward_head(rollout_hidden, deterministic=True)
        rew_logits = rew_logits[:,:,0]  # with [:,:,0] we only take the first head
        # Convert to scalar rewards
        probs_rew = jax.nn.softmax(rew_logits, axis=-1)
        rewards = jnp.sum(probs_rew * symexp(rew_centers), axis=-1)
        rewards = jax.lax.stop_gradient(rewards)
        # rewards is now (B, H) corresponding to r_{T_ctx+1}...r_{T_ctx+H}
        
        # ---------------------------
        # 3. Compute values
        # ---------------------------
        val_logits, val_centers = value_head(h_sg, deterministic=False, rngs=rngs)
        val_logits = val_logits[:,:,0]  # with [:,:,0] we only take the first head
        # Convert to scalar values
        probs_val = jax.nn.softmax(val_logits, axis=-1)
        values = jnp.sum(probs_val * symexp(val_centers), axis=-1)
        values = values[:, -horizon-1:]  # (B, H+1)
        
        # 4. Compute TD-lambda returns
        td_returns = compute_td_lambda_returns(
            rewards=rewards,
            values=values,
            gamma=cfg.gamma,
            lambda_=cfg.lambda_,
        )  # (B, H)
        
        # ---------------------------
        # 5. Compute value loss
        # ---------------------------
        # Use value head outputs from above (no re-computation needed)
        val_loss = compute_value_loss(
            val_logits=val_logits[:, -horizon-1:-1],  # (B, H, K)
            centers_log_val=value_head.symexp_centers_log,
            td_returns=td_returns,
        )
        
        # ---------------------------
        # 6. Compute policy logits 
        # ---------------------------
        pi_bc_logits = policy_bc(h_sg[:, -horizon:], deterministic=True)  # (B, H, A)
        pi_logits =  policy_head(h_sg[:, -horizon:], deterministic=False, rngs=rngs)  # (B, H, A)
        
        # 7. Compute advantages
        advantages = td_returns - values[:, :-1]  # (B, H)
        
        # 8. Compute PMPO loss
        pi_loss, pmpo_aux = compute_pmpo_loss(
            policy_logits=pi_logits,
            actions=imagined_actions,
            advantages=advantages,
            policy_prior_logits=pi_bc_logits,
            alpha=cfg.alpha,
            beta=cfg.beta,
        )
        
        # 9. Total loss
        total_loss = pi_loss + val_loss
        
        # Metrics
        aux = {
            'loss/total': total_loss,
            'loss/policy': pi_loss,
            'loss/value': val_loss,
            'loss/pmpo_negative': pmpo_aux['loss_negative'],
            'loss/pmpo_positive': pmpo_aux['loss_positive'],
            'loss/pmpo_kl': pmpo_aux['kl_loss'],
            'stats/n_positive': pmpo_aux['n_positive'],
            'stats/n_negative': pmpo_aux['n_negative'],
            'stats/mean_reward': jnp.mean(rewards),
            'stats/mean_value': jnp.mean(values),
            'stats/mean_advantage': jnp.mean(advantages),
            'stats/mean_td_return': jnp.mean(td_returns),
        }
        
        return total_loss, aux
    
    # ---------------------------
    # Compute gradients and update
    # ---------------------------
    
    # Compute loss and gradients
    # Per nnx_basics.html: "nnx.value_and_grad for gradient computation"
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(trainable)
    
    optimizer.update(grads)
    
    # Update step counter
    # NNX Variables can be mutated directly
    step.value += 1
    
    # Add gradient norms to metrics
    grad_pi_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads.policy)))
    grad_val_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads.value)))
    metrics['stats/grad_pi_norm'] = grad_pi_norm
    metrics['stats/grad_val_norm'] = grad_val_norm
    
    return metrics


# ---------------------------
# Main Training Loop
# ---------------------------

def run(cfg: RLConfig, ctx):
    """Main training loop with NNX state management."""
    
    print(f"\n{'='*80}")
    print(f"RL Policy Training (NNX Version)")
    print(f"{'='*80}\n")
    print(f"Config: {OmegaConf.to_yaml(cfg)}\n")
    
    # Load dataset
    print("Loading dataset...")
    data_iter = build_iterator(
        data_path=cfg.data_path,
        batch_size=cfg.dataset.B,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    
    # Initialize all components
    print("Initializing models...")
    tokenizer, dynamics, task_embedder, policy_bc, reward_head, optimizer, schedule, cfg, rngs, step = initialize_rl_training(cfg, ctx)
    
    print(f"Initialized at step {step.value}")
    
    # Setup logging
    logger = MetricLogger(cfg.log_dir)
    
    # Training loop
    print(f"\nStarting training for {cfg.num_steps} steps...\n")
    pbar = tqdm(total=cfg.num_steps, initial=step.value)
    
    for batch in data_iter:
        if step.value >= cfg.num_steps:
            break
        
        # Convert batch to JAX arrays
        batch_jax = {
            'videos': jnp.array(batch['videos']),
            'actions': jnp.array(batch['actions']),
            'task_ids': jnp.array(batch.get('task_ids', np.zeros((cfg.dataset.B,), dtype=np.int32))),
        }
        
        # Training step (models are mutated in-place)
        metrics = train_step(tokenizer, dynamics, task_embedder, policy_bc, reward_head, optimizer, schedule, cfg, rngs, step, batch_jax)
        
        # Logging
        if step.value % cfg.log_every == 0:
            metrics_cpu = {k: float(v) for k, v in metrics.items()}
            logger.log(metrics_cpu, step=step.value)
            pbar.set_postfix(loss=f"{metrics_cpu['loss/total']:.4f}")
        
        # Checkpointing
        if step.value % cfg.save_every == 0:
            # TODO: Implement NNX checkpointing
            # Per graph.html:
            # "Use nnx.split() to decompose into GraphDef and State for saving"
            pass
        
        pbar.update(1)
    
    pbar.close()
    print(f"\nTraining complete! Final step: {step.value}")
    
    # Return only what's needed for checkpointing/inspection
    # The trainable models are in optimizer.model
    return optimizer, step


@hydra.main(version_base=None, config_path="../configs", config_name="rl_policy")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    
    # Create parallel context (needed for from_pretrained)
    # For single GPU, this is a simple wrapper
    class SimpleContext:
        pass
    
    ctx = SimpleContext()
    optimizer, step = run(cfg, ctx)
    print(f"Training finished at step {step.value}")


if __name__ == "__main__":
    main()
