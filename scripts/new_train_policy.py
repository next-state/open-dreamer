"""
Refactored JIT-friendly policy/value training using imagination rollouts.

This version uses structured state containers and reusable functions from
dreamer.training and dreamer.generation to eliminate code duplication and
improve maintainability.

Key improvements over original train_policy.py:
- Uses state containers (3 arguments instead of 20+)
- Leverages latent_rollout from dreamer.generation
- Uses RL loss functions from dreamer.training
- ~300 lines vs 1800+ in original
"""

from __future__ import annotations

import logging
logging.getLogger('absl').setLevel(logging.WARNING)

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple
from functools import partial
from tqdm import tqdm
import time

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from einops import rearrange

from dreamer.models import (
    Dynamics,
    Tokenizer,
    TaskEmbedder,
    PolicyHeadMTP,
    RewardHeadMTP,
    ValueHead,
)
from dreamer.state import (
    FrozenModels,
    FrozenVars,
    TrainableParams,
    TrainableState,
    RLTrainingSystem,
)
from dreamer.training import (
    compute_td_lambda_returns,
    compute_value_loss,
    compute_pmpo_loss,
    symlog,
    symexp,
)
from dreamer.generation import latent_rollout, DenoiseSchedule
from dreamer.data import make_iterator
from dreamer.configs import RLConfig
from dreamer.utils import (
    temporal_patchify,
    pack_bottleneck_to_spatial,
    normalize_with_dataset_stats,
    make_manager,
    to_jnp_dtype,
)
from dreamer.logging import MetricLogger


# ---------------------------
# Initialization
# ---------------------------

# Checkpoint loading is now handled by Dynamics.from_pretrained() and Tokenizer.from_pretrained()
# These one-liners replace ~100 lines of manual loading code


def initialize_rl_training(
    cfg: RLConfig,
    frames_init: jnp.ndarray,
    actions_init: jnp.ndarray,
) -> Tuple[RLTrainingSystem, TrainableState]:
    """
    Initialize all models and load pretrained checkpoints.
    
    Args:
        cfg: RL training configuration
        frames_init: Sample frames for initialization (B, T, H, W, C)
        actions_init: Sample actions for initialization (B, T)
        
    Returns:
        system: Static training system (frozen models, config)
        state: Mutable training state (params, opt_state, rng, step)
    """
    # ---------------------------
    # 1. Load pretrained models using one-liner (like reactor.py)
    # ---------------------------
    
    print(f"Loading pretrained models from {cfg.bc_rew_ckpt}...")
    dynamics, dyn_vars, dyn_cfg, tokenizer, tok_vars, tok_cfg = Dynamics.from_pretrained(cfg.bc_rew_ckpt)
    
    # Extract components from checkpoint meta
    # The dynamics checkpoint includes task_embedder, policy_bc, and reward_head
    mngr = make_manager(cfg.bc_rew_ckpt, item_names=("meta", "task_embedder", "policy_head", "reward_head"))
    latest = mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {cfg.bc_rew_ckpt}")
    
    # Initialize task embedder, policy_bc, reward_head from checkpoint
    rng = jax.random.PRNGKey(0)
    task_embedder = TaskEmbedder(
        d_model=dyn_cfg.dynamics.d_model,
        use_ids=cfg.use_task_ids,
        n_tasks=cfg.n_tasks,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
    )
    policy_bc = PolicyHeadMTP(
        d_model=dyn_cfg.dynamics.d_model,
        action_dim=cfg.action_dim,
        L=cfg.L,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
    )
    reward_head = RewardHeadMTP(
        d_model=dyn_cfg.dynamics.d_model,
        L=cfg.L,
        num_bins=cfg.num_reward_bins,
        log_low=cfg.reward_log_low,
        log_high=cfg.reward_log_high,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
    )
    
    # Initialize vars to get structure
    dummy_task_ids = jnp.zeros((cfg.dataset.B,), dtype=jnp.int32)
    task_vars = task_embedder.init({"params": rng}, dummy_task_ids, cfg.dataset.B, cfg.dataset.T)
    
    fake_h = jnp.zeros(
        (cfg.dataset.B, cfg.dataset.T, cfg.n_agent, dyn_cfg.dynamics.d_model),
        dtype=to_jnp_dtype(cfg.dtype),
    )
    pi_bc_vars = policy_bc.init({"params": rng, "dropout": rng}, fake_h, deterministic=True)
    rew_vars = reward_head.init({"params": rng, "dropout": rng}, fake_h, deterministic=True)
    
    # Load pretrained weights
    restored = mngr.restore(
        latest,
        args=ocp.args.Composite(
            meta=ocp.args.JsonRestore(),
            task_embedder=ocp.args.StandardRestore(task_vars),
            policy_head=ocp.args.StandardRestore(pi_bc_vars),
            reward_head=ocp.args.StandardRestore(rew_vars),
        ),
    )
    task_vars = restored.task_embedder
    pi_bc_vars = restored.policy_head
    rew_vars = restored.reward_head
    
    # ---------------------------
    # 2. Initialize trainable heads (policy and value)
    # ---------------------------
    
    policy_head = PolicyHeadMTP(
        d_model=dyn_cfg.dynamics.d_model,
        action_dim=cfg.action_dim,
        L=cfg.L,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
    )
    
    value_head = ValueHead(
        d_model=dyn_cfg.dynamics.d_model,
        num_bins=cfg.num_value_bins,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
    )
    
    rng_pi, rng_val = jax.random.split(jax.random.PRNGKey(2), 2)
    pi_vars = policy_head.init({"params": rng_pi, "dropout": rng_pi}, fake_h, deterministic=True)
    val_vars = value_head.init({"params": rng_val, "dropout": rng_val}, fake_h, deterministic=True)
    
    # ---------------------------
    # 3. Create state containers
    # ---------------------------
    
    frozen_models = FrozenModels(
        encoder=tokenizer.encoder,  # Extract encoder from tokenizer
        decoder=tokenizer.decoder,  # Extract decoder from tokenizer
        dynamics=dynamics,
        task_embedder=task_embedder,
        policy_bc=policy_bc,
        reward_head=reward_head,
        tokenizer=tokenizer,  # Store full tokenizer for convenience
    )
    
    # Fixed MAE key for consistent encoding
    mae_eval_key = jax.random.PRNGKey(777)
    
    frozen_vars = FrozenVars(
        enc=tok_vars,  # Tokenizer vars contain both encoder and decoder
        dec=tok_vars,  # Same vars used for both
        dyn=dyn_vars,
        task=task_vars,
        pi_bc=pi_bc_vars,
        rew=rew_vars,
        mae_key=mae_eval_key,
    )
    
    params = TrainableParams(
        pi=pi_vars["params"],
        val=val_vars["params"],
    )
    
    tx = optax.adam(cfg.lr)
    opt_state = tx.init(params)
    
    # Create denoise schedule from loaded config
    k_max = dyn_cfg.dynamics.k_max
    emax = jnp.log2(k_max).astype(jnp.int32)
    schedule = DenoiseSchedule(k_max=k_max, step_idx=emax)
    
    system = RLTrainingSystem(
        frozen_models=frozen_models,
        frozen_vars=frozen_vars,
        policy_head=policy_head,
        value_head=value_head,
        tx=tx,
        schedule=schedule,
        cfg=cfg,
    )
    
    state = TrainableState(
        params=params,
        opt_state=opt_state,
        rng=jax.random.PRNGKey(cfg.seed),
        step=0,
    )
    
    return system, state


# ---------------------------
# Training Step
# ---------------------------

@partial(jax.jit, static_argnames=("system",))
def train_step(
    system: RLTrainingSystem,
    state: TrainableState,
    batch: dict,
) -> Tuple[TrainableState, dict]:
    """
    Single training step for policy and value head.
    
    Args:
        system: Static training system (frozen models, config)
        state: Mutable training state (params, opt_state, rng, step)
        batch: Training batch with 'videos', 'actions', 'task_ids'
        
    Returns:
        new_state: Updated training state
        metrics: Dict with losses and diagnostics
    """
    cfg = system.cfg
    
    # Split RNG
    rng, rng_enc, rng_imag, rng_val = jax.random.split(state.rng, 4)
    
    # Extract batch data
    videos = batch['videos']  # (B, T, H, W, C)
    actions = batch['actions']  # (B, T)
    task_ids = batch.get('task_ids', jnp.zeros((videos.shape[0],), dtype=jnp.int32))
    
    B, T = videos.shape[:2]
    T_ctx = cfg.context_length
    horizon = cfg.horizon
    
    # Get packing factor from dynamics config
    packing_factor = system.frozen_models.dynamics.config.packing_factor
    
    # Encode context frames to latents using tokenizer
    z_ctx, _ = system.frozen_models.tokenizer.apply(
        system.frozen_vars.enc,
        videos[:, :T_ctx],
        method=system.frozen_models.tokenizer.encode,
        packing_factor=packing_factor,
        rngs={"mae": system.frozen_vars.mae_key},
        deterministic=True,
    )  # (B, T_ctx, n_spatial, D_s)
    
    # Create agent tokens for context
    agent_tokens_ctx = system.frozen_models.task_embedder.apply(
        system.frozen_vars.task,
        task_ids,
        B, T_ctx,
    )  # (B, T_ctx, n_agent, d_model)
    
    # Define loss function
    def loss_fn(params):
        # 1. Imagination rollout using latent_rollout from generation.py
        rollout_result = latent_rollout(
            dynamics=system.frozen_models.dynamics,
            dyn_vars=system.frozen_vars.dyn,
            policy=system.policy_head,
            policy_vars={'params': params.pi},
            schedule=system.schedule,
            latents_ctx=z_ctx,
            actions_ctx=actions[:, :T_ctx],
            num_steps=horizon,
            rng=rng_imag,
            initial_agent_tokens=agent_tokens_ctx,
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
        
        # 2. Compute rewards from hidden states
        rew_logits, centers_log_rew = system.frozen_models.reward_head.apply(
            system.frozen_vars.rew,
            h_sg[:, :-1],  # (B, T_ctx + H - 1, n_agent, d_model)
            deterministic=True,
        )
        # Convert to scalar rewards
        probs_rew = jax.nn.softmax(rew_logits, axis=-1)
        rewards = jnp.sum(probs_rew * symexp(centers_log_rew), axis=-1)
        # (B, T_ctx + H - 1) -> take last H steps
        rewards = rewards[:, -horizon:]  # (B, H)
        
        # 3. Compute values from hidden states
        val_logits, centers_log_val = system.value_head.apply(
            {'params': params.val},
            h_sg,  # (B, T_ctx + H, n_agent, d_model)
            deterministic=False,
            rngs={'dropout': rng_val},
        )
        # Convert to scalar values
        probs_val = jax.nn.softmax(val_logits, axis=-1)
        values = jnp.sum(probs_val * symexp(centers_log_val), axis=-1)
        # (B, T_ctx + H) -> need last H+1 for bootstrapping
        values = values[:, -horizon-1:]  # (B, H+1)
        
        # 4. Compute TD-lambda returns
        td_returns = compute_td_lambda_returns(
            rewards=rewards,
            values=values,
            gamma=cfg.gamma,
            lambda_=cfg.lambda_,
        )  # (B, H)
        
        # 5. Compute value loss
        val_loss = compute_value_loss(
            value_head=system.value_head,
            val_vars={'params': params.val},
            hidden_states=h_sg[:, -horizon-1:],  # (B, H+1, n_agent, d_model)
            td_returns=td_returns,
            rng=rng_val,
        )
        
        # 6. Compute policy logits (from BC prior and current policy)
        pi_bc_logits = system.frozen_models.policy_bc.apply(
            system.frozen_vars.pi_bc,
            h_sg[:, -horizon:],  # (B, H, n_agent, d_model)
            deterministic=True,
        )  # (B, H, A)
        
        pi_logits = system.policy_head.apply(
            {'params': params.pi},
            hidden_states[:, -horizon:],  # (B, H, n_agent, d_model)
            deterministic=False,
        )  # (B, H, A)
        
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
    
    # Compute gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Apply updates
    updates, new_opt_state = system.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    # Create new state
    new_state = TrainableState(
        params=new_params,
        opt_state=new_opt_state,
        rng=rng,
        step=state.step + 1,
    )
    
    # Add gradient norms to metrics
    grad_pi_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads.pi)))
    grad_val_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads.val)))
    metrics['stats/grad_pi_norm'] = grad_pi_norm
    metrics['stats/grad_val_norm'] = grad_val_norm
    
    return new_state, metrics


# ---------------------------
# Main Training Loop
# ---------------------------

def run(cfg: RLConfig):
    """Main training loop."""
    
    print(f"\n{'='*80}")
    print(f"RL Policy Training (Refactored Version)")
    print(f"{'='*80}\n")
    print(f"Config: {OmegaConf.to_yaml(cfg)}\n")
    
    # Load dataset
    print("Loading dataset...")
    data_iter = make_iterator(
        data_path=cfg.data_path,
        batch_size=cfg.dataset.B,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    
    # Get sample batch for initialization
    sample_batch = next(data_iter)
    frames_init = jnp.array(sample_batch['videos'])
    actions_init = jnp.array(sample_batch['actions'])
    
    # Initialize
    print("Initializing models...")
    system, state = initialize_rl_training(cfg, frames_init, actions_init)
    print(f"Initialized at step {state.step}")
    
    # Setup logging
    logger = MetricLogger(cfg.log_dir)
    
    # Training loop
    print(f"\nStarting training for {cfg.num_steps} steps...\n")
    pbar = tqdm(total=cfg.num_steps, initial=state.step)
    
    for batch in data_iter:
        if state.step >= cfg.num_steps:
            break
        
        # Convert batch to JAX arrays
        batch_jax = {
            'videos': jnp.array(batch['videos']),
            'actions': jnp.array(batch['actions']),
            'task_ids': jnp.array(batch.get('task_ids', np.zeros((cfg.dataset.B,), dtype=np.int32))),
        }
        
        # Training step
        state, metrics = train_step(system, state, batch_jax)
        
        # Logging
        if state.step % cfg.log_every == 0:
            metrics_cpu = {k: float(v) for k, v in metrics.items()}
            logger.log(metrics_cpu, step=state.step)
            pbar.set_postfix(loss=f"{metrics_cpu['loss/total']:.4f}")
        
        # Checkpointing
        if state.step % cfg.save_every == 0:
            # TODO: Implement checkpointing
            pass
        
        pbar.update(1)
    
    pbar.close()
    print(f"\nTraining complete! Final step: {state.step}")
    
    return state


@hydra.main(version_base=None, config_path="../configs", config_name="rl_policy")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    final_state = run(cfg)
    print(f"Training finished at step {final_state.step}")


if __name__ == "__main__":
    main()
