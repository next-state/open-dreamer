"""State management containers for RL training.

This module provides hierarchical state containers to organize the many
models, parameters, and training components needed for RL policy training.
Using structured containers dramatically reduces function argument counts
from 20+ to just 3-5.
"""

from dataclasses import dataclass
from typing import Any
from flax import struct
import optax
import jax.numpy as jnp

from dreamer.models import (
    Encoder,
    Decoder,
    Dynamics,
    TaskEmbedder,
    PolicyHeadMTP,
    RewardHeadMTP,
    ValueHead,
)


# ---------------------------
# Frozen State (Static, never updated during training)
# ---------------------------

@dataclass(frozen=True)
class FrozenModels:
    """Container for all pretrained (frozen) model instances.
    
    These models are loaded from checkpoints and never updated during
    RL training. They provide the world model and behavioral prior.
    """
    encoder: Encoder
    decoder: Decoder
    dynamics: Dynamics
    task_embedder: TaskEmbedder
    policy_bc: PolicyHeadMTP  # Behavioral cloning policy (frozen prior)
    reward_head: RewardHeadMTP
    tokenizer: Any = None  # Optional: full tokenizer for convenience (contains encoder+decoder)


@dataclass(frozen=True)
class FrozenVars:
    """Container for all frozen model variables.
    
    These are the parameters and constants for frozen models.
    Using frozen=True allows these to be JIT static arguments.
    """
    enc: dict
    dec: dict
    dyn: dict  # Contains both params and constants
    task: dict
    pi_bc: dict
    rew: dict
    mae_key: jax.Array  # Fixed random key for MAE dropout during encoding


# ---------------------------
# Trainable State (Mutable, updated each step)
# ---------------------------

@struct.dataclass
class TrainableParams:
    """Trainable parameters (JAX pytree).
    
    Only the policy and value heads are trained during RL training.
    Using flax.struct.dataclass makes this a proper JAX pytree for
    gradient computation and optimizer state management.
    """
    pi: dict  # Policy head params
    val: dict  # Value head params


@struct.dataclass
class TrainableState:
    """Mutable training state (JAX pytree for scan/jit).
    
    This contains all the state that changes during training:
    - params: Trainable parameters
    - opt_state: Optimizer state (momentum, etc.)
    - rng: Random number generator key
    - step: Training step counter
    
    Using flax.struct.dataclass makes this a JAX pytree that can
    be efficiently passed through JIT-compiled functions.
    """
    params: TrainableParams
    opt_state: optax.OptState
    rng: jax.Array
    step: int


# ---------------------------
# Complete Training System
# ---------------------------

@dataclass(frozen=True)
class RLTrainingSystem:
    """Complete RL training system with all components.
    
    This is the top-level container that holds everything needed for
    training. It separates frozen components (models, config) from
    mutable state (params, optimizer).
    
    By making this frozen=True, it can be a JIT static argument,
    which is crucial for performance.
    
    Usage:
        @partial(jax.jit, static_argnames=("system",))
        def train_step(system: RLTrainingSystem, state: TrainableState, batch: dict):
            # Now train_step has just 3 arguments instead of 20+!
            ...
    """
    # Frozen components (static args to JIT)
    frozen_models: FrozenModels
    frozen_vars: FrozenVars
    
    # Trainable models (static args to JIT)
    policy_head: PolicyHeadMTP
    value_head: ValueHead
    
    # Training infrastructure (static)
    tx: optax.GradientTransformation
    schedule: Any  # DenoiseSchedule
    cfg: Any  # RLConfig or similar config object
