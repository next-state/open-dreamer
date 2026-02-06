from __future__ import annotations

from typing import Any, Final

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from flax import struct


@struct.dataclass
class Actions:
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

    def to_dict(self) -> dict[str, Array | None]:
        """Flatten Actions to a dict of arrays for serialization."""
        actions = {"binary": self.binary, "categorical": self.categorical, "continuous": self.continuous}
        return actions

    @classmethod
    def from_dict(cls, d: dict[str, Array | None]) -> Actions:
        """Reconstruct Actions from a flattened dict. Raise KeyError if a key is missing."""
        return cls(binary=d["binary"], categorical=d["categorical"], continuous=d["continuous"])


def create_noop_action_like(template: Actions, categorical_action_dim: int) -> Actions:
    """Creates a (B, 1, ...) no-op start action."""

    def _create_action(arr, fill_value):
        if arr is None: return None
        return jnp.full_like(arr[:, 0:1], fill_value)

    return Actions(
        binary     = _create_action(template.binary, 0),
        categorical = _create_action(template.categorical, categorical_action_dim - 1),
        continuous  = _create_action(template.continuous, 0.0)
    )


def shift_actions(actions: Actions, categorical_action_dim: int) -> Actions:
    """Shift actions right by 1, preprend noop action."""

    noop_action = create_noop_action_like(actions, categorical_action_dim)
    
    def _shift(current_arr, start_arr):
        if current_arr is None: return None
        return jnp.concatenate([start_arr, current_arr[:, :-1]], axis=1)

    return Actions(
        binary      = _shift(actions.binary, noop_action.binary),
        categorical = _shift(actions.categorical, noop_action.categorical),
        continuous  = _shift(actions.continuous, noop_action.continuous)
    )


# ------------------------------------------------------------
# VPT action space
# ------------------------------------------------------------

# source:
# https://github.com/openai/Video-Pre-Training/blob/095519fbd4ee0e9281d19f19601e45629de9ac3f/run_inverse_dynamics_model.py
key_to_index: Final[dict[str, int]] = {
    # Keyboard actions (matching KEYBOARD_BUTTON_MAPPING order from reference)
    "key.keyboard.w": 0,              # forward
    "key.keyboard.a": 1,              # left
    "key.keyboard.s": 2,              # back
    "key.keyboard.d": 3,              # right
    "key.keyboard.space": 4,          # jump
    "key.keyboard.left.shift": 5,     # sneak
    "key.keyboard.left.control": 6,   # sprint
    "key.keyboard.e": 7,              # inventory
    "key.keyboard.q": 8,              # drop
    "key.keyboard.escape": 9,         # ESC
    "key.keyboard.f": 10,             # swapHands
    "key.keyboard.1": 11,             # hotbar.1
    "key.keyboard.2": 12,             # hotbar.2
    "key.keyboard.3": 13,             # hotbar.3
    "key.keyboard.4": 14,             # hotbar.4
    "key.keyboard.5": 15,             # hotbar.5
    "key.keyboard.6": 16,             # hotbar.6
    "key.keyboard.7": 17,             # hotbar.7
    "key.keyboard.8": 18,             # hotbar.8
    "key.keyboard.9": 19,             # hotbar.9
    "mouse.0": 20,                    # attack
    "mouse.1": 21,                    # use
    "mouse.2": 22,                    # pickItem
}


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
    return np.sign(x) * np.log(1.0 + mu * np.abs(x)) / np.log(1.0 + mu)


def mouse_movement_to_categorical(dx: Array, dy: Array) -> Array:
    """Convert continuous mouse movement to categorical action index.
    
    Args:
        dx: Raw mouse x delta (before scaling), any shape
        dy: Raw mouse y delta (before scaling), same shape as dx
    
    Returns:
        Categorical index in [0, 120] representing the 11x11 camera action grid.
        Index is computed as: bin_y * 11 + bin_x. Same shape as input.
    """
    dxy = np.stack([dx, dy], axis=-1)
    
    # Scale to degrees and clip to valid range
    dxy_deg = np.clip(dxy * CAMERA_SCALER, -CAMERA_MAXVAL, CAMERA_MAXVAL)
    
    # Normalize to [-1, 1] and apply mu-law encoding
    dxy_norm = dxy_deg / CAMERA_MAXVAL
    dxy_encoded = mu_law_encode(dxy_norm)
    
    # Map from [-1, 1] to bin indices [0, 10]
    bins = np.round((dxy_encoded + 1.0) * (NUM_CAMERA_BINS - 1) / 2.0).astype(np.int32)
    bins = np.clip(bins, 0, NUM_CAMERA_BINS - 1)
    
    return bins[..., 1] * NUM_CAMERA_BINS + bins[..., 0]


NUM_BINARY_ACTIONS: Final[int] = 23  # keyboard (20) + mouse buttons (3)


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
            idx = key_to_index.get(key, 21)
            binary[t, idx] = 1
        
        # Parse mouse buttons
        mouse = action.get("mouse", {})
        buttons = mouse.get("buttons", [])
        for btn in buttons:
            if btn == 0:
                binary[t, key_to_index["mouse.0"]] = 1  # attack
            elif btn == 1:
                binary[t, key_to_index["mouse.1"]] = 1  # use
            elif btn == 2:
                binary[t, key_to_index["mouse.2"]] = 1  # pickItem
        
        # Parse camera movement
        camera_dx[t] = mouse.get("dx", 0.0)
        camera_dy[t] = mouse.get("dy", 0.0)
    
    # Convert camera to categorical using mu-law foveated discretization
    categorical = mouse_movement_to_categorical(
        np.array(camera_dx), 
        np.array(camera_dy)
    )
    
    return Actions(
        binary=np.array(binary),
        categorical=categorical,
    )
