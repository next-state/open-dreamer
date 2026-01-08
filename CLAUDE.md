# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements **Dreamer 4** - a world model agent that learns to solve control tasks through reinforcement learning inside a fast and accurate world model. The implementation is based on the paper "Training Agents Inside of Scalable World Models" and is written in pure JAX using the Flax NNX API.

The project aims to reproduce the Dreamer 4 architecture as described in `docs/main.txt`, focusing on:
- Action-conditioned video prediction via shortcut forcing
- Causal tokenization with masked autoencoding
- Interactive dynamics modeling with efficient transformers
- Imagination-based RL training (PMPO-style policy optimization)

## Development Philosophy

**Role**: You are an expert consultant specializing in Flax NNX, tasked with helping research teams build and modernize state-of-the-art machine learning models. Your mandate is to transform codebases into high-performance, ultra-efficient, and production-ready systems. You value elegant, readable, configurable, and maintainable code above all.

**Development Context & Authority**:
- The client is in an active development stage. Backward compatibility is explicitly not a concern.
- You have been instructed to avoid all defensive coding patterns and legacy support. Your goal is to eliminate technical debt at its root.
- You possess full authority to critique, break, or completely redesign the client's existing approaches. Preserving suboptimal code is seen as more harmful than rewriting it.

**Methodology**:
1. **Top-Down Analysis First**: Before examining the client's code, you will deeply familiarize yourself with the core task and the latest Flax NNX tools and patterns. Use available tools to search documentation and official examples.
2. **Holistic Design**: Develop a comprehensive, MECE (Mutually Exclusive, Collectively Exhaustive) understanding of the problem. Architect a modular, decoupled, and DRY solution.
3. **Critical Code Review**: Only after forming your independent design plan do you examine the client's code. Evaluate it against your optimal architecture.
4. **Decisive Execution**: If the existing code aligns with your optimal plan, proceed with implementation. If not, architect and execute a refactor to make the codebase fit the superior design. Do not compromise the design to accommodate existing code.

**Critical Technical Directive**:
- The Flax/JAX ecosystem evolves rapidly. You must not rely on cached or outdated knowledge.
- Your north star is always the latest stable version of Flax NNX. Be aware that documentation for older paradigms (e.g., Linen) may be prevalent and contradictory.
- For every significant decision, especially where doubt exists, consult the latest official NNX documentation, source examples, or literature using provided tools. Verify patterns actively.
- Primary Tools: Use the Context7 MCP tool or search functions to access and cross-reference the latest documentation, API references, and canonical examples.

## Running Commands

All scripts should be run using `uv run`. The repository uses `uv` for dependency management (see `pyproject.toml`).

### Environment Setup

```bash
uv sync                      # Create .venv and install packages
source .venv/bin/activate    # Activate virtual environment
uv pip install -e .          # Install project as editable package
```

### Training Pipeline

Dreamer 4 follows a **4-phase training pipeline**:

#### Phase 1: Train Causal Tokenizer
```bash
uv run python scripts/train_tokenizer.py
```
- Trains a causal tokenizer using masked autoencoding (MAE)
- Compresses video frames into continuous latent representations
- Config: `configs/tokenizer.yaml`
- Target: ~40 PSNR reconstruction quality

#### Phase 2: Train Interactive Dynamics Model
```bash
uv run python scripts/train_dynamics.py tokenizer_ckpt=./logs/tokenizer/checkpoints
```
- Trains action-conditioned dynamics model in latent space
- Uses shortcut forcing objective for fast interactive inference
- Config: `configs/dynamics.yaml`
- Target: ~30 PSNR autoregressive generation (4 sampling steps)

#### Phase 3: Train BC/Reward Heads
```bash
uv run python scripts/train_heads.py \
    tokenizer_ckpt=./logs/tokenizer/checkpoints \
    dynamics_ckpt=./logs/dynamics/checkpoints
```
- Adds agent tokens, behavior cloning head, and reward prediction head
- Finetunes on top of frozen tokenizer + dynamics
- Config: `configs/heads.yaml`

#### Phase 4: Train Policy via Imagination RL
```bash
uv run python scripts/train_policy.py bc_rew_ckpt=./logs/bc_rew/checkpoints
```
- Trains policy purely in imagination using PMPO objective
- No environment interaction (offline RL)
- Config: `configs/policy.yaml`

### Configuration Management

The project uses **Hydra** for configuration management. Configs are in `configs/`:
- `common.yaml` - Shared settings (wandb, dtype, checkpointing)
- `tokenizer.yaml` - Tokenizer architecture and training
- `dynamics.yaml` - Dynamics model architecture and training
- `heads.yaml` - BC/reward head configuration
- `policy.yaml` - RL policy training configuration

Override config values via command line:
```bash
uv run python scripts/train_tokenizer.py use_wandb=True max_steps=500000
```

### Logging and Checkpoints

- Checkpoints saved to `logs/{run_name}/checkpoints/` (uses Orbax)
- Hydra outputs to `logs/{run_name}/`
- Enable wandb logging: `use_wandb=True wandb_entity=<your_entity>`
- Default checkpoint retention: `ckpt_max_to_keep=2`

## Architecture

### Core Components (`dreamer/`)

- **`models.py`** - Main model definitions:
  - `Tokenizer` - Causal tokenizer (encoder + decoder with latent bottleneck)
  - `Dynamics` - Interactive dynamics model with shortcut forcing
  - `PolicyHeadMTP` - Multi-token prediction policy head
  - `RewardHead`, `ValueHead` - Reward and value prediction heads
  - Efficient transformer with space-time axial attention, GQA, RoPE

- **`data.py`** - Dataset and environment implementations:
  - Bouncing square synthetic dataset (default for testing)
  - CoinRun dataset support via `coinrun_data/` module
  - Action space utilities and data iterators

- **`generation.py`** - Sampling and generation utilities:
  - `DenoiseSchedule` - τ-ladder denoising schedule for shortcut forcing
  - `next_frame()` - Frame-by-frame generation with KV cache
  - Shortcut forcing inference (typically 4 steps per frame)

- **`training.py`** - Training loops and loss functions:
  - Shortcut forcing losses (flow matching + bootstrap)
  - Behavior cloning and reward modeling losses
  - PMPO policy gradient objective
  - Lambda-returns for value learning

- **`utils.py`** - Checkpointing and utilities:
  - Orbax checkpoint management
  - Normalization/denormalization helpers
  - Learning rate schedules (constant, warmup-stable-decay, cosine)
  - Model initialization from checkpoints

- **`parallel.py`** - Data/model parallelism setup (FSDP sharding)

- **`configs.py`** - Dataclass configuration definitions

### Key Architectural Concepts

#### Shortcut Forcing
The dynamics model uses "shortcut forcing" - a combination of diffusion forcing and shortcut models:
- Trains on multiple denoising step sizes (powers of 2)
- Allows variable-step inference at test time (typically K=4)
- Uses x-space prediction (not v-space) to prevent error accumulation
- Ramp loss weight: linearly increases with signal level σ

#### Efficient Transformer
- **Space-time axial attention**: Separate space-only and time-only attention layers
- **Temporal sparsity**: Time attention only every 4 layers
- **GQA (Grouped Query Attention)**: Multiple query heads share KV heads
- **KV cache**: Ring buffer for efficient autoregressive generation
- **Causal masking**: Tokens attend to current frame + past

#### Agent Tokens
- Agent tokens are interleaved with observation/action tokens
- Agent tokens attend to all modalities; other modalities cannot attend back
- Prevents causal confusion: future predictions only influenced by actions
- Used for policy, reward, and value predictions

#### PMPO (Policy Optimization)
- Sign-based advantage weighting (magnitude-free)
- Balances positive/negative feedback equally (α=0.5)
- Behavioral prior KL regularization (β=0.3)
- Robust across different return scales (no normalization needed)

## Reactor Runtime

`reactor.py` implements an interactive inference server using `reactor-runtime`:
- Real-time interactive world model inference
- Supports keyboard/controller input → action mapping
- Can use learned policy or human control
- Used for CoinRun visualization and testing

For information on how the reactor runtime works, see https://docs.reactor.inc

## Dataset Notes

The codebase currently works with:
- **Bouncing Square**: Synthetic dataset for testing (WASD control)
- **CoinRun**: Procgen environment (via `coinrun_data/` and `procgen-mirror`)
- **Custom datasets**: Via ArrayRecord format (see `dataset.array_record_path` config)

Dataset statistics (mean/std) are computed and stored in configs for normalization.

## Important Implementation Details

### JAX/Flax NNX Usage
- Models use Flax NNX (next-gen Flax API)
- Explicit state management with `nnx.State` and `nnx.GraphDef`
- Checkpointing via Orbax with NNX integration
- FSDP sharding for distributed training

### Precision
- Default mixed precision: `bfloat16` compute, `float32` parameters
- Controlled via `dtype` and `param_dtype` config fields
- Important for stability on TPU/GPU

### Batch Handling
- Alternating short/long batch lengths during dynamics training
- Prevents overfitting to always seeing start frames
- Enables length generalization at inference

### Multi-Token Prediction (MTP)
- Policy and reward heads predict L steps into the future
- Improves sample efficiency and temporal consistency
- Default L=8 for BC/reward, L=2 for RL policy

## Development Workflow

1. **Start with tokenizer training** - Verify reconstruction quality
2. **Train dynamics** - Check autoregressive generation quality
3. **Add agent heads** - Ensure BC/reward losses decrease
4. **RL in imagination** - Watch returns improve without env interaction

Each phase builds on the previous checkpoint. Do not skip phases.

## Common Pitfalls

- **Checkpoint paths**: Scripts require explicit checkpoint paths from previous phases
- **Hydra working directory**: Logs go to `logs/{run_name}/`, not current directory
- **Dataset paths**: ArrayRecord datasets must be pre-generated (not auto-downloaded)
- **Memory**: Large models may require gradient checkpointing or smaller batch sizes
- **XLA compilation**: First iteration is slow due to JIT compilation

## References

- Paper: "Training Agents Inside of Scalable World Models" (docs/main.txt)
- Dreamer 4 website: https://danijar.com/project/dreamer4/ (all of the informaion is in the main.txt, you hardly have any reason to look at this website)
- Jasmine codebase: https://github.com/p-doom/jasmine (reference implementation)
