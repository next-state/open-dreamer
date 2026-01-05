# Master Plan: `train_policy.py` Refactoring

**Date:** 2025-12-30  
**Status:** Phase 1 Complete ✅ | Phase 2 In Progress  
**Version:** Consolidated from all previous plans

---

## 📊 Progress Overview

| Phase | Task | Status |
|-------|------|--------|
| **Phase 1** | Extract RL loss functions to `dreamer/training.py` | ✅ **COMPLETE** |
| **Phase 2** | Modify `generation.py` to return hidden states | ✅ **COMPLETE** |
| **Phase 3** | Create state container dataclasses | ⏳ **TODO** |
| **Phase 4** | Refactor `train_policy.py` structure | ⏳ **TODO** |
| **Phase 5** | Add utility functions | ⏳ **TODO** |
| **Phase 6** | Testing and validation | ⏳ **TODO** |

---

## 🎯 Project Goals

1. **Eliminate ALL code duplication** - Move reusable RL components to `dreamer.training`
2. **Structured state management** - Reduce function arguments from 20+ to 3-5 using dataclass containers
3. **Leverage existing infrastructure** - Use `latent_rollout` from `dreamer.generation` (no new imagination module)
4. **Match clean structure** of `train_dynamics.py` and `train_heads.py`
5. **Single-GPU simplicity** for this refactor

---

## ✅ Phase 1: Extract RL Loss Functions (COMPLETED)

### What Was Done

Added 4 new functions to `/home/ubuntu/dreamer4-jax-private/dreamer/training.py`:

#### 1. `symexp` (inverse of symlog)
```python
def symexp(y: jnp.ndarray) -> jnp.ndarray:
    """Inverse of symlog: symmetric exponential transform."""
    return jnp.sign(y) * (jnp.expm1(jnp.abs(y)))
```
**Location:** After `symlog` function (~line 320)

#### 2. `compute_td_lambda_returns`
```python
def compute_td_lambda_returns(
    rewards: jnp.ndarray,      # (B, T)
    values: jnp.ndarray,       # (B, T+1)
    gamma: float,
    lambda_: float,
) -> jnp.ndarray:
    """
    Compute TD(λ) returns via backward scan.
    
    At timestep t: R^λ[t] = r[t+1] + γ * ((1-λ) V[t+1] + λ R^λ[t+1])
    Bootstrap: R^λ[T] = V[T]
    """
```
**Location:** After symexp/twohot helpers (~line 355)  
**Source:** Extracted from `train_policy.py` lines 1360-1380

#### 3. `compute_value_loss`
```python
def compute_value_loss(
    value_head,
    val_vars: VariableDict,
    hidden_states: jnp.ndarray,    # (B, T+1, d_model)
    td_returns: jnp.ndarray,       # (B, T)
    rng: jax.Array,
) -> jnp.ndarray:
    """
    Compute value head loss using symexp twohot targets.
    Uses two-hot encoding for improved learning across varying scales.
    """
```
**Location:** After `compute_td_lambda_returns`  
**Source:** Extracted from `train_policy.py` lines 1388-1401

#### 4. `compute_pmpo_loss`
```python
def compute_pmpo_loss(
    policy_logits: jnp.ndarray,        # (B, T, A)
    actions: jnp.ndarray,              # (B, T)
    advantages: jnp.ndarray,           # (B, T)
    policy_prior_logits: jnp.ndarray,  # (B, T, A)
    alpha: float = 0.5,
    beta: float = 0.3,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Compute PMPO (Probabilistic Policy Optimization) loss.
    
    Balances positive/negative advantages using sign-only information,
    making it robust to return scale variations across tasks.
    
    Returns:
        loss: Scalar PMPO loss
        aux: Dict with loss_negative, loss_positive, kl_loss, n_positive, n_negative
    """
```
**Location:** After `compute_value_loss`  
**Source:** Extracted from `train_policy.py` lines 1404-1450

### Result
✅ All RL-specific loss functions now in centralized, reusable location  
✅ No code duplication between scripts  
✅ Type-annotated and documented  
✅ Ready for use in refactored `train_policy.py`

---

## ✅ Phase 2: Modify `generation.py` (COMPLETED)

### What Was Done

Modified `latent_rollout` in `/home/ubuntu/dreamer4-jax-private/dreamer/generation.py` to **always return a dict** with all computed values.

### Key Changes

**Before:**
```python
def latent_rollout(...) -> jnp.ndarray:
    # ... scan loop ...
    return out_latents  # Just (B, T_ctx + num_steps, n_spatial, D_s)
```

**After:**
```python
def latent_rollout(...) -> dict:
    # ... scan loop returns (latent, action, hidden_state) ...
    return {
        'latents': out_latents,           # (B, T_ctx + num_steps, n_spatial, D_s)
        'actions': rollout_actions,       # (B, num_steps, ...)
        'hidden_states': rollout_hidden,  # (B, num_steps, n_agent, D)
        'context_hidden': h_seq,          # (B, T_ctx, n_agent, D)
    }
```

### Implementation Details

1. **Modified scan_step return** (line 315):
   ```python
   # OLD: return (h_next, caches_next, rng), latent_next[:,0]
   # NEW: return (h_next, caches_next, rng), (latent_next[:,0], action, h_next)
   ```

2. **Updated scan unpacking** (lines 318-326):
   ```python
   _, (rollout_latents, rollout_actions, rollout_hidden) = jax.lax.scan(...)
   
   # Rearrange all outputs
   rollout_latents = einops.rearrange(rollout_latents, 't b s d -> b t s d')
   rollout_actions = einops.rearrange(rollout_actions, 't b ... -> b t ...')
   rollout_hidden = einops.rearrange(rollout_hidden, 't b n d -> b t n d')
   ```

3. **Updated `video_rollout`** to use dict return (lines 343-408):
   ```python
   rollout_result = latent_rollout(...)
   pred_frames, _ = tokenizer.apply(tokenizer_vars, rollout_result['latents'], ...)
   ```

### Benefits
✅ **No extra dynamics forward pass** - returns values already computed  
✅ **Clean interface** - caller ignores unneeded values  
✅ **Ready for RL** - provides hidden states and actions for loss computation  
✅ **Philosophy:** Always return everything computed, throw away what you don't need

---

## ⏳ Phase 3: Create State Container Dataclasses (TODO)

### Goal
Reduce function arguments from 20+ to 3-5 using hierarchical state containers.

### Create New File: `dreamer/state.py`

```python
"""State management containers for RL training."""

from dataclasses import dataclass
from flax import struct
import optax
import jax.numpy as jnp
from typing import Any

from dreamer.models import (
    Encoder, Decoder, Dynamics, TaskEmbedder,
    PolicyHeadMTP, RewardHeadMTP, ValueHead
)

# ---------------------------
# Frozen State (Static, never updated)
# ---------------------------

@dataclass(frozen=True)
class FrozenModels:
    """All pretrained (frozen) model instances."""
    encoder: Encoder
    decoder: Decoder
    dynamics: Dynamics
    task_embedder: TaskEmbedder
    policy_bc: PolicyHeadMTP  # Behavioral prior
    reward_head: RewardHeadMTP


@dataclass(frozen=True)
class FrozenVars:
    """All frozen model variables."""
    enc: dict
    dec: dict
    dyn: dict
    task: dict
    pi_bc: dict
    rew: dict
    mae_key: jax.Array  # For encoder MAE dropout


# ---------------------------
# Trainable State (Mutable, updated each step)
# ---------------------------

@struct.dataclass
class TrainableParams:
    """Trainable parameters (JAX pytree)."""
    pi: dict  # Policy head params
    val: dict  # Value head params


@struct.dataclass
class TrainableState:
    """Mutable training state (JAX pytree for scan/jit)."""
    params: TrainableParams
    opt_state: optax.OptState
    rng: jax.Array
    step: int


# ---------------------------
# Complete Training System
# ---------------------------

@dataclass(frozen=True)
class RLTrainingSystem:
    """Complete RL training system with all components."""
    # Frozen components (static args to JIT)
    frozen_models: FrozenModels
    frozen_vars: FrozenVars
    
    # Trainable models (static args to JIT)
    policy_head: PolicyHeadMTP
    value_head: ValueHead
    
    # Training infrastructure (static)
    tx: optax.GradientTransformation
    schedule: Any  # DenoiseSchedule
    cfg: Any  # RLConfig
```

### Why This Structure?

**Before (current `train_policy.py`):**
```python
def train_step(
    encoder, decoder, dynamics, task_embedder,
    policy_head_bc, reward_head, policy_head, value_head,
    enc_vars, dec_vars, dyn_vars, task_vars,
    pi_bc_vars, rew_vars, pi_vars, val_vars,
    params, opt_state, tx, batch, schedule, rng, cfg,
    ...
):  # 20+ arguments! 😱
```

**After (with containers):**
```python
@partial(jax.jit, static_argnames=("system",))
def train_step(
    system: RLTrainingSystem,
    state: TrainableState,
    batch: dict,
) -> Tuple[TrainableState, dict]:
    # Just 3 arguments! ✨
```

---

## ⏳ Phase 4: Refactor `train_policy.py` Structure (TODO)

### New Structure Overview

```python
# scripts/train_policy.py (refactored)

import jax
import jax.numpy as jnp
import optax
from functools import partial
from pathlib import Path

from dreamer.state import FrozenModels, FrozenVars, TrainableParams, TrainableState, RLTrainingSystem
from dreamer.training import (
    compute_td_lambda_returns,
    compute_value_loss,
    compute_pmpo_loss,
    symlog,
)
from dreamer.generation import latent_rollout, DenoiseSchedule
from dreamer.utils import load_pretrained_components

# ---------------------------
# Initialization (~80 lines)
# ---------------------------

def initialize_rl_training(cfg: RLConfig) -> Tuple[RLTrainingSystem, TrainableState]:
    """
    Load pretrained models and initialize trainable components.
    
    Returns:
        system: Static training system (frozen models, config)
        state: Mutable training state (params, opt_state, rng)
    """
    # 1. Load pretrained components
    frozen_models, frozen_vars = load_pretrained_components(
        dynamics_ckpt=cfg.dynamics_ckpt,
        tokenizer_ckpt=cfg.tokenizer_ckpt,
        task_embedder_ckpt=cfg.task_embedder_ckpt,
        policy_bc_ckpt=cfg.policy_bc_ckpt,
        reward_ckpt=cfg.reward_ckpt,
    )
    
    # 2. Initialize trainable models
    rng = jax.random.PRNGKey(cfg.seed)
    rng, pi_key, val_key = jax.random.split(rng, 3)
    
    policy_head = PolicyHeadMTP(num_actions=cfg.num_actions, ...)
    value_head = ValueHead(...)
    
    # 3. Initialize parameters
    dummy_hidden = jnp.zeros((1, cfg.n_agent, cfg.d_model))
    pi_vars = policy_head.init(pi_key, dummy_hidden, deterministic=True)
    val_vars = value_head.init(val_key, dummy_hidden, deterministic=True)
    
    params = TrainableParams(pi=pi_vars['params'], val=val_vars['params'])
    
    # 4. Create optimizer
    tx = optax.adam(learning_rate=cfg.lr)
    opt_state = tx.init(params)
    
    # 5. Create schedule
    schedule = DenoiseSchedule(cfg.k_max, cfg.step_idx)
    
    # 6. Package everything
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
        rng=rng,
        step=0,
    )
    
    return system, state


# ---------------------------
# Training Step (~100 lines)
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
    task_ids = batch['task_ids']  # (B,)
    
    B, T = videos.shape[:2]
    T_ctx = cfg.context_length
    
    # Encode context frames
    z_ctx, _ = system.frozen_models.encoder.apply(
        system.frozen_vars.enc,
        videos[:, :T_ctx],
        packing_factor=system.frozen_models.dynamics.config.packing_factor,
        method=system.frozen_models.encoder.encode,
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
        # 1. Imagination rollout using EXISTING latent_rollout
        rollout_result = latent_rollout(
            dynamics=system.frozen_models.dynamics,
            dyn_vars=system.frozen_vars.dyn,
            policy=system.policy_head,
            policy_vars={'params': params.pi},
            schedule=system.schedule,
            latents_ctx=z_ctx,
            actions_ctx=actions[:, :T_ctx],
            num_steps=cfg.horizon,
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
        rewards = rewards[:, -cfg.horizon:]  # (B, H)
        
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
        values = values[:, -cfg.horizon-1:]  # (B, H+1)
        
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
            hidden_states=h_sg[:, -cfg.horizon-1:],  # (B, H+1, n_agent, d_model)
            td_returns=td_returns,
            rng=rng_val,
        )
        
        # 6. Compute policy logits (from BC prior and current policy)
        pi_bc_logits = system.frozen_models.policy_bc.apply(
            system.frozen_vars.pi_bc,
            h_sg[:, -cfg.horizon:],  # (B, H, n_agent, d_model)
            deterministic=True,
        )  # (B, H, A)
        
        pi_logits = system.policy_head.apply(
            {'params': params.pi},
            hidden_states[:, -cfg.horizon:],  # (B, H, n_agent, d_model)
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
    
    return new_state, metrics


# ---------------------------
# Main Loop (~50 lines)
# ---------------------------

def run(cfg: RLConfig):
    """Main training loop."""
    
    # Initialize
    system, state = initialize_rl_training(cfg)
    
    # Load dataset
    dataset = load_dataset(cfg.data_path, cfg.batch_size)
    
    # Training loop
    for epoch in range(cfg.num_epochs):
        for batch in dataset:
            state, metrics = train_step(system, state, batch)
            
            # Logging
            if state.step % cfg.log_every == 0:
                print(f"Step {state.step}: {metrics}")
            
            # Checkpointing
            if state.step % cfg.save_every == 0:
                save_checkpoint(state, cfg.checkpoint_dir / f"step_{state.step}")
    
    return state


def main():
    cfg = RLConfig(...)
    final_state = run(cfg)
    print(f"Training complete! Final step: {final_state.step}")


if __name__ == "__main__":
    main()
```

### Key Improvements

**Before:**
- 1800+ lines
- 280-line `train_step` with 20+ arguments
- Duplicated loss computations
- Mixed concerns

**After:**
- ~300 lines total
- 100-line `train_step` with 3 arguments
- Reuses functions from `dreamer.training`
- Clean separation: init / train / run

---

## ⏳ Phase 5: Add Utility Functions (TODO)

### Create `dreamer/utils.py` enhancements

```python
"""Utility functions for loading pretrained components."""

from pathlib import Path
from typing import Tuple
import jax

from dreamer.state import FrozenModels, FrozenVars
from dreamer.models import (
    Encoder, Decoder, Dynamics, TaskEmbedder,
    PolicyHeadMTP, RewardHeadMTP
)


def load_pretrained_components(
    dynamics_ckpt: Path,
    tokenizer_ckpt: Path,
    task_embedder_ckpt: Path,
    policy_bc_ckpt: Path,
    reward_ckpt: Path,
) -> Tuple[FrozenModels, FrozenVars]:
    """
    Load all pretrained (frozen) components for RL training.
    
    Args:
        dynamics_ckpt: Path to dynamics checkpoint
        tokenizer_ckpt: Path to tokenizer checkpoint (contains encoder + decoder)
        task_embedder_ckpt: Path to task embedder checkpoint
        policy_bc_ckpt: Path to behavioral cloning policy checkpoint
        reward_ckpt: Path to reward head checkpoint
        
    Returns:
        frozen_models: Container with all frozen model instances
        frozen_vars: Container with all frozen model variables
    """
    # Load models and variables (implementation depends on checkpoint format)
    encoder, enc_vars = load_encoder(tokenizer_ckpt)
    decoder, dec_vars = load_decoder(tokenizer_ckpt)
    dynamics, dyn_vars = load_dynamics(dynamics_ckpt)
    task_embedder, task_vars = load_task_embedder(task_embedder_ckpt)
    policy_bc, pi_bc_vars = load_policy(policy_bc_ckpt)
    reward_head, rew_vars = load_reward_head(reward_ckpt)
    
    # Create MAE key for encoder dropout
    mae_key = jax.random.PRNGKey(0)  # Fixed for frozen encoder
    
    frozen_models = FrozenModels(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        task_embedder=task_embedder,
        policy_bc=policy_bc,
        reward_head=reward_head,
    )
    
    frozen_vars = FrozenVars(
        enc=enc_vars,
        dec=dec_vars,
        dyn=dyn_vars,
        task=task_vars,
        pi_bc=pi_bc_vars,
        rew=rew_vars,
        mae_key=mae_key,
    )
    
    return frozen_models, frozen_vars
```

---

## ⏳ Phase 6: Testing and Validation (TODO)

### Validation Checklist

- [ ] Verify refactored code produces identical loss values as original
- [ ] Check gradient magnitudes match
- [ ] Confirm checkpoint saving/loading works
- [ ] Test single-GPU training runs
- [ ] Validate metrics logging
- [ ] Run for 1000 steps and compare to original implementation

### Testing Strategy

1. **Numerical equivalence test:**
   ```python
   # Run both versions on same batch with same RNG
   old_metrics = old_train_step(...)
   new_metrics = new_train_step(...)
   assert jnp.allclose(old_metrics['loss/total'], new_metrics['loss/total'])
   ```

2. **Integration test:**
   - Run full training for 1000 steps
   - Compare final model parameters
   - Check learning curves match

3. **Ablation test:**
   - Test each phase independently
   - Ensure no regressions

---

## 📁 Files Modified/Created

### Created
- ✅ `/home/ubuntu/dreamer4-jax-private/dreamer/training.py` - Added 4 RL loss functions
- ⏳ `/home/ubuntu/dreamer4-jax-private/dreamer/state.py` - State containers (TODO)
- ⏳ `/home/ubuntu/dreamer4-jax-private/dreamer/utils.py` - Enhanced with loading utilities (TODO)

### Modified
- ✅ `/home/ubuntu/dreamer4-jax-private/dreamer/generation.py` - `latent_rollout` now returns dict
- ⏳ `/home/ubuntu/dreamer4-jax-private/scripts/train_policy.py` - Full refactor (TODO)

---

## 🎓 Key Design Decisions

### 1. Why No New Imagination Module?
**Decision:** Use existing `latent_rollout` from `dreamer.generation`  
**Rationale:** 
- Already implements KV-cached autoregressive rollout
- Battle-tested in video generation
- Minimal modification needed (just return hidden states)
- Avoids code duplication

### 2. Why Dataclass Containers?
**Decision:** Hierarchical state containers instead of 20+ function arguments  
**Rationale:**
- Dramatically improves readability
- Clear separation of frozen vs trainable state
- JIT-friendly with `@dataclass(frozen=True)` and `@struct.dataclass`
- Easier to extend (add new model = add one field)

### 3. Why Always Return Dict from `latent_rollout`?
**Decision:** No flag, just always return everything  
**Rationale:**
- Simpler interface (no boolean flags)
- No performance penalty (caller ignores unneeded values)
- Philosophy: Return what's computed, throw away what you don't need

### 4. Why Extract Loss Functions?
**Decision:** Move TD-λ, PMPO, value loss to `dreamer.training`  
**Rationale:**
- Eliminates duplication
- Follows pattern of `train_dynamics.py` and `train_heads.py`
- Reusable across different training scripts
- Easier to test in isolation

---

## 📊 Expected Metrics

### Code Size Reduction
- **Before:** 1800+ lines in `train_policy.py`
- **After:** ~300 lines in `train_policy.py` + ~200 lines in shared modules
- **Reduction:** ~70% reduction in script size

### Argument Count Reduction
- **Before:** `train_step` with 20+ arguments
- **After:** `train_step` with 3 arguments
- **Reduction:** 85% reduction in argument count

### Duplication Elimination
- **Before:** TD-λ, PMPO, value loss duplicated across scripts
- **After:** All in `dreamer.training`, imported where needed
- **Reduction:** 100% duplication elimination

---

## 🚀 Next Steps

1. **Create `dreamer/state.py`** with dataclass containers
2. **Add `load_pretrained_components` to `dreamer/utils.py`**
3. **Refactor `train_policy.py`** using new structure
4. **Run validation tests** to ensure numerical equivalence
5. **Update documentation** and add inline comments

---

## 📚 References

- Original implementations: `scripts/train_policy.py` (lines 1360-1450)
- Pattern examples: `scripts/train_dynamics.py`, `scripts/train_heads.py`
- Generation code: `dreamer/generation.py` (lines 246-341)
- Training utilities: `dreamer/training.py`

---

## ✅ Summary

### What's Been Done
1. ✅ **Phase 1 Complete:** All RL loss functions extracted to `dreamer/training.py`
2. ✅ **Phase 2 Complete:** `latent_rollout` modified to return dict with hidden states

### What's Left
3. ⏳ Create state container dataclasses in `dreamer/state.py`
4. ⏳ Add loading utilities to `dreamer/utils.py`
5. ⏳ Refactor `train_policy.py` structure
6. ⏳ Run validation tests

### Philosophy
- **No code duplication** - Extract to shared modules
- **Leverage existing infrastructure** - Use `latent_rollout`, not new imagination module
- **Structured state** - Dataclass containers over 20+ arguments
- **Clean separation** - Frozen vs trainable, models vs vars vs config
- **Always return computed values** - Caller ignores what they don't need

**Result:** Cleaner, more maintainable, and easier to extend! 🎉
