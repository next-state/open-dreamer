"""Data types shared across the dreamer package."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from flax import nnx


@dataclass
class Actions(nnx.Pytree):
    """Container for multi-modal action data.

    Attributes:
        binary: (B, T, num_binary_actions) int32 with values 0 or 1
        categorical: (B, T) int32 categorical indices
        continuous: (B, T, continuous_action_dim) float32 continuous values
    """
    binary: jnp.ndarray | None = None
    categorical: jnp.ndarray | None = None
    continuous: jnp.ndarray | None = None

    def __getitem__(self, key):
        """Slice Actions along batch/time dimensions."""
        return jax.tree.map(lambda x: x[key] if x is not None else None, self)
