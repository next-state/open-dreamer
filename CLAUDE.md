# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dreamer 4 is an unofficial implementation of the Dreamer 4 world model and RL agent in pure JAX. The system trains an action-conditioned video diffusion model, then trains a policy with RL entirely within the learned world model (imagination).
The paper is availabe at `docs/main.txt`

**Key architecture components:**
- **Causal Tokenizer**: Masked autoencoder (MAE) that compresses video into latent tokens
- **Interactive Dynamics Model**: Learns temporal dynamics in latent space using space-time axial attention with τ-ladder diffusion (shortcut forcing)
- **Agent Tokens**: Task-conditioned representations for policy/reward prediction
- **Policy/Reward Heads**: BC and reward prediction heads using multi-token prediction (MTP)

## Build & Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
uv sync

# Activate virtual environment
source .venv/bin/activate

# Install package in editable mode
uv pip install -e .
```

### Training Pipeline

Dreamer 4 uses a 4-phase training pipeline. Each phase depends on checkpoints from the previous phase:

```bash
# Phase 1: Train causal tokenizer (MAE on videos)
uv run scripts/train_tokenizer.py

# Phase 2: Train interactive dynamics model (requires tokenizer checkpoint)
uv run scripts/train_dynamics.py tokenizer_ckpt=./logs/tokenizer/checkpoints

# Phase 3: Train BC/reward heads (requires tokenizer + dynamics checkpoints)
uv run scripts/train_heads.py tokenizer_ckpt=./logs/tokenizer/checkpoints dynamics_ckpt=./logs/train_dynamics/checkpoints

# Phase 4: Train policy in imagination (requires BC/reward checkpoint)
# Two versions available:
uv run scripts/train_policy.py bc_rew_ckpt=./logs/bc_rew/checkpoints
# OR the refactored NNX version:
uv run scripts/new_train_policy.py bc_rew_ckpt=./logs/bc_rew/checkpoints
```

### Configuration Management

- Config files live in `configs/` and use Hydra for hierarchical configuration
- Override config values via command-line: `python script.py key=value`
- Enable wandb logging: `use_wandb=True`
- Default configs: `configs/common.yaml` (shared settings like dtype, wandb project)
- Per-phase configs: `configs/tokenizer.yaml`, `configs/dynamics.yaml`, `configs/heads.yaml`, `configs/policy.yaml`

### Checkpointing

- Checkpoints save to `logs/{run_name}/checkpoints/` by default
- Controlled by `ckpt_max_to_keep` and `ckpt_save_every` in configs
- Uses Orbax for checkpoint management

## Code Architecture

### Core Library (`dreamer/`)

The main implementation is organized into focused modules:

**`models.py`**: Core neural network architectures
- `KVCache`: Ring buffer KV cache for efficient autoregressive generation
- `Tokenizer`: Causal MAE with encoder/decoder (patchify → latent → unpatchify)
- `Dynamics`: Interactive dynamics model with space-time axial attention
- `TaskEmbedder`: Task-conditioned embeddings for agent tokens
- `PolicyHeadMTP`: Multi-token prediction policy head
- `RewardHeadMTP`: Symexp twohot reward prediction head
- All models use Flax NNX (new object-oriented API, not Linen)

**`generation.py`**: Imagination/rollout logic
- `DenoiseSchedule`: Precomputed τ-ladder schedule for diffusion sampling
- `next_latent()`: Single-step denoising for autoregressive latent rollout
- `next_frame()`: Combines `next_latent()` + tokenizer decoding
- `imagine_sequence()`: Full autoregressive rollout in latent space
- All functions are JIT-compiled and use KV caching for speed

**`training.py`**: Reusable training utilities
- τ (tau) sampling for shortcut forcing: `sample_tau_for_step()`, `sample_step_excluding_dmin()`
- Loss computation: `shortcut_forcing_step()`, `compute_policy_loss()`, `compute_reward_loss()`
- Evaluation: `run_evaluation()` with video logging

**`data.py`**: Dataset and environment implementations
- `make_iterator()`: Creates data loader for training
- Bouncing square synthetic dataset (default)
- CoinRun support via `procgen-mirror` and ArrayRecord format
- All data normalized to mean=0.5, std=sqrt(1/12)

**`configs.py`**: Dataclass-based configuration objects
- `DatasetConfig`, `TokenizerConfig`, `DynamicsConfig`, `BCRewConfig`, `PolicyConfig`
- All configs are frozen dataclasses for immutability

**`parallel.py`**: Data parallelism utilities
- `ParallelContext`: Manages JAX mesh, sharding, device placement
- Batch dimension sharded across devices via `data_sharding`
- Model parameters replicated via `replicated_sharding`

**`utils.py`**: Helper functions
- Checkpoint management: `make_state()`, `try_restore()`, `maybe_save()`, `make_manager()`
- Normalization: `normalize_with_dataset_stats()`, `unnormalize_with_dataset_stats()`
- Model initialization: `init_tokenizer()`, etc.

**`sampler.py`**: Non-JIT sampling for debugging/visualization
- Used when you need readable, step-by-step denoising (slower than `generation.py`)

**`logging.py`**: Wandb and metric tracking
- `MetricLogger`: Tracks and logs metrics to wandb

**`state.py`**: Training state containers (being phased out in favor of NNX direct state)

### Training Scripts (`scripts/`)

Each script corresponds to one phase of the pipeline:
- `train_tokenizer.py`: Phase 1 (MAE training)
- `train_dynamics.py`: Phase 2 (dynamics model)
- `train_heads.py`: Phase 3 (BC + reward heads)
- `train_policy.py`: Phase 4 (RL in imagination) - original Linen-style
- `new_train_policy.py`: Phase 4 refactored using Flax NNX (preferred going forward)

All scripts follow the same pattern:
1. Hydra config loading
2. Initialize ParallelContext for multi-device
3. Load/restore checkpoints
4. Define JIT-compiled training step
5. Training loop with tqdm progress bar
6. Periodic evaluation and checkpoint saving

### Interactive Runtime (`reactor.py`)

- Real-time interactive world model using `reactor-runtime` library
- Loads trained dynamics + policy models
- Allows user to control agent via keyboard and see model imagination in real-time
- Maps keyboard input → actions via `input_to_action()`
- Uses `next_frame()` for fast autoregressive generation with KV caching

## Important Implementation Details

### Flax NNX vs Linen
This codebase uses **Flax NNX** (the new eager, object-oriented API), not Linen. Key differences:
- Direct module calls: `model(x)` instead of `model.apply(vars, x)`
- Stateful: Module state lives in the object, not passed as separate dict
- Use `nnx.Optimizer` instead of manual `optax.GradientTransformation` + opt_state
- Use `nnx.jit` and `nnx.value_and_grad` for automatic state handling
- Migration in progress: newer scripts use NNX patterns (e.g., `new_train_policy.py`)

### τ-ladder Diffusion (Shortcut Forcing)
- Signal level τ ∈ [0, 1] controls noise: τ=0 is pure noise, τ=1 is clean
- Training samples τ on discrete grids: K ∈ {1, 2, 4, ..., k_max=256}
- Step size d = 1/K determines grid granularity
- Bootstrap loss distills two half-steps into one full step
- Diffusion loss predicts clean signal from noisy input at sampled τ

### Checkpointing & Restoration
- All checkpoint paths are specified via command-line args (e.g., `tokenizer_ckpt=...`)
- `try_restore()` loads latest checkpoint from directory
- When composing models (e.g., dynamics requires tokenizer), pass checkpoint dirs to training scripts
- Frozen models (e.g., tokenizer during dynamics training) have requires_grad=False set

### Data Parallelism
- Batch dimension is sharded across devices automatically
- Create `ParallelContext` at start of training script
- Use `ctx.shard_data(batch)` to shard input batches
- Use `ctx.replicate(params)` to replicate model state
- JIT-compiled functions work transparently with sharding

### RNG Handling
JAX requires explicit random key management:
- All scripts maintain `rng` state and split it for each random operation
- Use `jax.random.split(rng, num)` to create independent keys
- Pass RNG keys to all stochastic operations (sampling, dropout, etc.)
- Recent improvements ensure reproducibility across runs

### Mixed Precision
- Computation dtype: `bfloat16` (default, set in `configs/common.yaml`)
- Parameter dtype: `float32` (maintains precision)
- Uses `to_jnp_dtype()` utility for dtype conversion
- Set via `dtype` and `param_dtype` config fields

## Dataset Notes

**Bouncing Square (default):**
- Synthetic dataset with agent-controlled square
- Action space: {up, down, left, right, null} (5 discrete actions)
- Reward: proximity to center of image
- Generated on-the-fly during training

**CoinRun:**
- Procgen environment episodes stored as ArrayRecord
- Set `dataset.source=custom` and point to `array_record_path`
- Action space: 15 discrete actions (diagonals + cardinals + no-op)
- Requires `procgen-mirror` and `grain` for data loading

## Common Patterns

### Loading a pretrained model:
```python
from dreamer.utils import init_tokenizer, try_restore, make_manager

# Initialize model
tokenizer = init_tokenizer(config, rngs)

# Create checkpoint manager
manager = make_manager(checkpoint_dir)

# Restore latest checkpoint
restored_state = try_restore(manager, {"model": tokenizer})
if restored_state:
    tokenizer = restored_state["model"]
```

### JIT-compiling a training step:
```python
@partial(jax.jit, static_argnames=("config",))
def train_step(model, opt_state, batch, rng, config):
    def loss_fn(params):
        # compute loss
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(model.parameters())
    # apply gradients
    return loss, metrics
```

### Sharding data across devices:
```python
ctx = ParallelContext.create(batch_size=config.dataset.B)

for batch in data_iter:
    batch = ctx.shard_data(batch)  # Shard along batch dimension
    loss, metrics = train_step(model, batch)
    metrics = ctx.to_host_scalar(metrics)  # Convert back to Python scalars
```

## Development Workflow

1. **Iterative development**: Edit code → run training script → check wandb logs
2. **Debugging**: Use `ipdb` (installed in dev dependencies) for breakpoints
3. **Visualization**: Training scripts log sample videos/images to `logs/{run_name}/`
4. **Config experiments**: Create sweep configs in `configs/sweep/` for hyperparameter search
5. **Checkpoint management**: Old checkpoints auto-deleted based on `ckpt_max_to_keep`

## Code Style

- Type hints used throughout (from `typing` and `jax.Array`)
- Docstrings focus on Args/Returns and implementation notes
- Shape comments: `# (B, T, H, W, C)` indicate tensor dimensions
- Prefer pure functions for JIT compilation
- Use `partial(jax.jit, static_argnames=(...))` for static config args
