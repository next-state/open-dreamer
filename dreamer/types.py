"""
Data types shared across the dreamer package.

most of the information about the data is taken from: 
https://github.com/openai/Video-Pre-Training/blob/095519fbd4ee0e9281d19f19601e45629de9ac3f/run_inverse_dynamics_model.py
"""
from __future__ import annotations
import warnings

from typing import Any, Final

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from dataclasses import dataclass
from flax import nnx

from .configs import DatasetConfig


@dataclass
class Actions(nnx.Pytree):
    """Container for multi-modal action data.

    Attributes:
        binary: (B, T, num_binary_actions) int32 with values 0 or 1
        categorical: (B, T) int32 categorical indices
        continuous: (B, T, continuous_action_dim) float32 continuous values
    """
    binary: Array | None = None
    categorical: Array | None = None
    continuous: Array | None = None

    def __getitem__(self, key) -> Actions:
        """Slice Actions along batch/time dimensions."""
        return jax.tree.map(lambda x: x[key] if x is not None else None, self)


def get_noop_action_like(template: Actions, cfg: DatasetConfig) -> Actions:
    """Creates a (B, 1, ...) no-op start action."""

    def _create_action(arr, fill_value):
        if arr is None: return None
        return jnp.full_like(arr[:, 0:1], fill_value)

    return Actions(
        binary     = _create_action(template.binary, 0),
        categorical = _create_action(template.categorical, cfg.categorical_action_dim - 1),
        continuous  = _create_action(template.continuous, 0.0)
    )


def shift_actions(actions: Actions, cfg: DatasetConfig) -> Actions:
    """Shift actions right by 1, preprend noop action."""

    noop_action = get_noop_action_like(actions, cfg)
    
    def _shift(current_arr, start_arr):
        if current_arr is None: return None
        return jnp.concatenate([start_arr, current_arr[:, :-1]], axis=1)

    return Actions(
        binary      = _shift(actions.binary, noop_action.binary),
        categorical = _shift(actions.categorical, noop_action.categorical),
        continuous  = _shift(actions.continuous, noop_action.continuous)
    )


# sources:
# https://github.com/garrettallen14/JEPA-Image-World-Model/blob/main/video_dataset.py 
# https://github.com/microsoft/imitation_learning_in_modern_video_games/blob/main/pixelbc/data/minerl_actions.py
# https://github.com/openai/Video-Pre-Training/blob/main/lib/actions.py
# key_to_index to be used with idx = key_to_index.get(key,22)
# TODO: add a warning in case the ouput is 22
key_to_index: Final[dict[str, int]] = {
    "key.keyboard.w": 0,
    "key.keyboard.a": 1,
    "key.keyboard.s": 2,
    "key.keyboard.d": 3,
    "key.keyboard.space": 4,
    "key.keyboard.left.shift": 5,
    "key.keyboard.left.control": 6,
    "key.keyboard.e": 7,
    "key.keyboard.q": 8,
    "key.keyboard.escape": 9,
    "key.keyboard.f": 10,
    "key.keyboard.1": 11,
    "key.keyboard.2": 12,
    "key.keyboard.3": 13,
    "key.keyboard.4": 14,
    "key.keyboard.5": 15,
    "key.keyboard.6": 16,
    "key.keyboard.7": 17,
    "key.keyboard.8": 18,
    "key.keyboard.9": 19,
    "mouse.0": 20,
    "mouse.1": 21,
    "unknown": 22,
}
# In the dreamer paper it says that the actions are 23.
# The actions from 0 to 21 in this list affect the game. everything else should go here.



# source:
# https://github.com/openai/Video-Pre-Training/blob/main/lib/actions.py
# Uses mu-law foveated discretization as described in VPT paper
# 11x11 = 121 categorical classes for camera actions
# some interesting data about the actions https://github.com/openai/Video-Pre-Training/issues/54
CAMERA_SCALER = 360.0 / 2400.0
CAMERA_MAXVAL = 30.
CAMERA_MU = 5.0 # taking the default from    https://github.com/openai/Video-Pre-Training/blob/main/lib/actions.py#L80
NUM_CAMERA_BINS = 11  # per axis
NUM_CAMERA_CLASSES = NUM_CAMERA_BINS * NUM_CAMERA_BINS 


def mu_law_encode(x: Array, mu: float = CAMERA_MU) -> Array:
    """Apply mu-law compression for foveated discretization."""
    return jnp.sign(x) * jnp.log(1.0 + mu * jnp.abs(x)) / jnp.log(1.0 + mu)


def mouse_movement_to_categorical(dx: Array, dy: Array) -> Array:
    """Convert continuous mouse movement to categorical action index.
    
    Args:
        dx: Raw mouse x delta (before scaling), any shape
        dy: Raw mouse y delta (before scaling), same shape as dx
    
    Returns:
        Categorical index in [0, 120] representing the 11x11 camera action grid.
        Index is computed as: bin_y * 11 + bin_x. Same shape as input.
    """
    dxy = jnp.stack([dx, dy], axis=-1)
    
    # Scale to degrees and clip to valid range
    dxy_deg = jnp.clip(dxy * CAMERA_SCALER, -CAMERA_MAXVAL, CAMERA_MAXVAL)
    
    # Normalize to [-1, 1] and apply mu-law encoding
    dxy_norm = dxy_deg / CAMERA_MAXVAL
    dxy_encoded = mu_law_encode(dxy_norm)
    
    # Map from [-1, 1] to bin indices [0, 10]
    bins = jnp.round((dxy_encoded + 1.0) * (NUM_CAMERA_BINS - 1) / 2.0).astype(jnp.int32)
    bins = jnp.clip(bins, 0, NUM_CAMERA_BINS - 1)
    
    return bins[..., 1] * NUM_CAMERA_BINS + bins[..., 0]


NUM_BINARY_ACTIONS: Final[int] = 23  # keyboard (22) + unknown (1)


def parse_action_dicts(action_dicts: list[dict[str, Any]]) -> Actions:
    """Convert a list of VPT action dictionaries to an Actions pytree.
    
    Args:
        action_dicts: List of action dicts from VPT JSONL format. Each dict has:
            - mouse: {dx, dy, buttons, newButtons, ...}
            - keyboard: {keys: ["key.keyboard.w", ...], newKeys: [...]}
            - hotbar: int (0-8)
            - isGuiOpen: bool
            
    Returns:
        Actions pytree with:
            - binary: (T, 23) int32 array of keyboard/mouse button states
            - categorical: (T,) int32 array of camera action indices [0, 120]
    """
    T = len(action_dicts)
    
    # Initialize arrays
    binary = np.zeros((T, NUM_BINARY_ACTIONS), dtype=np.int32)
    camera_dx = np.zeros(T, dtype=np.float32)
    camera_dy = np.zeros(T, dtype=np.float32)
    
    for t, action in enumerate(action_dicts):
        # Parse keyboard keys
        keyboard = action.get("keyboard", {})
        keys = keyboard.get("keys", [])
        for key in keys:
            idx = key_to_index.get(key, 22)  # 22 = unknown
            if idx == 22: warnings.warn(f"Unknown key: {key}")
            binary[t, idx] = 1
        
        # Parse mouse buttons (attack=mouse.0, use=mouse.1)
        mouse = action.get("mouse", {})
        buttons = mouse.get("buttons", [])
        for btn in buttons:
            if btn == 0:
                binary[t, key_to_index["mouse.0"]] = 1
            elif btn == 1:
                binary[t, key_to_index["mouse.1"]] = 1
        
        # Parse camera movement
        camera_dx[t] = mouse.get("dx", 0.0)
        camera_dy[t] = mouse.get("dy", 0.0)
    
    # Convert camera to categorical using mu-law foveated discretization
    categorical = mouse_movement_to_categorical(
        jnp.array(camera_dx), 
        jnp.array(camera_dy)
    )
    
    return Actions(
        binary=jnp.array(binary),
        categorical=categorical,
    )
