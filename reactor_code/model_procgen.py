"""
MineRL Reactor Integration
===========================

This is a test version that uses the actual MineRL Minecraft environment.
Use this to verify that the Reactor integration works correctly before testing with the model.

User Input → input_to_action() → Action Dict
                                      ↓
                            MineRL Environment Step
                                      ↓
                              Numpy Frame (H,W,C)
                                      ↓
                            emit_frame() → User sees result

Action space matches the VPT action space defined in dreamer/actions.py:
  - 23 binary actions (keyboard keys + mouse buttons)
  - 121 categorical camera classes (11x11 mu-law discretized mouse dx/dy)
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

os.environ.setdefault("DISPLAY", ":0")

import cv2
import gym
import minerl
import numpy as np
from omegaconf import DictConfig, OmegaConf
from reactor_runtime import VideoModel, command, get_ctx
from reactor_runtime.model_api import model

# Configure logging to show DEBUG level messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MineRLReactorConfig:
    """Configuration for MineRL reactor runtime."""

    # Environment configuration
    env_name: str = "MineRLBasaltFindCave-v0"

    # Video settings
    fps: int = 20
    height: int = 360
    width: int = 640

    # Camera mapping (raw mouse delta -> MineRL camera degrees)
    camera_scaler: float = 360.0 / 2400.0
    camera_maxval: float = 30.0
    camera_deadzone_deg: float = 0.5
    invert_y: bool = False


def _camera_delta_to_degrees(
    raw_delta: Any,
    *,
    scaler: float,
    maxval: float,
    deadzone_deg: float,
) -> float:
    """Convert raw mouse deltas to stable camera degrees for MineRL."""
    try:
        delta = float(raw_delta)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(delta):
        return 0.0

    delta_deg = float(np.clip(delta * scaler, -maxval, maxval))
    if abs(delta_deg) < deadzone_deg:
        return 0.0
    return delta_deg


def input_to_action(
    env,
    controller_state: Dict[str, Any],
    mouse_state: Dict[str, Any],
    cfg: MineRLReactorConfig,
) -> Dict[str, Any]:
    """
    Convert keyboard/mouse input to MineRL action dictionary.

    Uses env.action_space.noop() as the base to ensure correct types and key
    ordering, then overwrites every field with the current input state.

    Mapping follows the VPT action space from dreamer/actions.py.

    Args:
        env: The MineRL gym environment (used for noop action base)
        controller_state: Dictionary with keyboard key states (bool values)
        mouse_state: Dictionary with mouse buttons and dx/dy deltas
        cfg: Runtime config containing camera scaling/filtering parameters

    Returns:
        MineRL action dictionary
    """
    action = env.action_space.no_op()

    # Movement keys (VPT indices 0-3)
    action["forward"] = 1 if controller_state.get("w") else 0
    action["left"] = 1 if controller_state.get("a") else 0
    action["back"] = 1 if controller_state.get("s") else 0
    action["right"] = 1 if controller_state.get("d") else 0

    # Movement modifier keys (VPT indices 4-6)
    action["jump"] = 1 if controller_state.get("space") else 0
    action["sneak"] = 1 if controller_state.get("shift") else 0
    action["sprint"] = 1 if controller_state.get("ctrl") else 0

    # Inventory/item keys (VPT indices 7-8, 10)
    action["inventory"] = 1 if controller_state.get("e") else 0
    action["drop"] = 1 if controller_state.get("q") else 0
    action["swapHands"] = 1 if controller_state.get("f") else 0

    # Hotbar keys (VPT indices 11-19)
    for slot in range(1, 10):
        action[f"hotbar.{slot}"] = 1 if controller_state.get(str(slot)) else 0

    # Mouse buttons (VPT indices 20-22)
    action["attack"] = 1 if mouse_state.get("left") else 0
    action["use"] = 1 if mouse_state.get("right") else 0
    action["pickItem"] = 1 if mouse_state.get("middle") else 0

    # Camera from mouse deltas — convert raw deltas to MineRL camera degrees.
    dx = _camera_delta_to_degrees(
        mouse_state.get("dx", 0.0),
        scaler=cfg.camera_scaler,
        maxval=cfg.camera_maxval,
        deadzone_deg=cfg.camera_deadzone_deg,
    )
    dy = _camera_delta_to_degrees(
        mouse_state.get("dy", 0.0) + 6, # TODO: this -6 and + 6 is an abomination but it seems to be needed to avoid drift
        scaler=cfg.camera_scaler,
        maxval=cfg.camera_maxval,
        deadzone_deg=cfg.camera_deadzone_deg,
    )
    if cfg.invert_y:
        dy = -dy
    action["camera"] = np.array([dy, dx], dtype=np.float32)

    # Never send ESC (VPT index 9)
    action["ESC"] = 0

    return action


# @model(name="minerl", config="configs/procgen.yaml")
class MineRLVideoModel(VideoModel):
    """
    MineRL environment integration with Reactor VideoModel interface.

    This version uses the actual Minecraft environment for testing the Reactor integration
    before deploying the full Dreamer model.

    The action space matches the VPT action space from dreamer/actions.py:
    - 23 binary actions: 10 keyboard keys + 9 hotbar keys + 3 mouse buttons + ESC
    - 121 categorical camera classes: 11x11 mu-law discretized mouse dx/dy
    """

    @command(
        "send_keyboard_state",
        description="Update keyboard state (WASD movement, Space/Shift/Ctrl modifiers, E/Q/F items, 1-9 hotbar)",
    )
    def send_keyboard_state(
        self,
        w: bool = False,
        a: bool = False,
        s: bool = False,
        d: bool = False,
        space: bool = False,
        shift: bool = False,
        ctrl: bool = False,
        e: bool = False,
        q: bool = False,
        f: bool = False,
        n1: bool = False,
        n2: bool = False,
        n3: bool = False,
        n4: bool = False,
        n5: bool = False,
        n6: bool = False,
        n7: bool = False,
        n8: bool = False,
        n9: bool = False,
    ):
        """
        Update keyboard state. Maps to VPT binary action indices from dreamer/actions.py.

        Args:
            w: W key (Forward) - VPT index 0
            a: A key (Strafe Left) - VPT index 1
            s: S key (Back) - VPT index 2
            d: D key (Strafe Right) - VPT index 3
            space: Space key (Jump) - VPT index 4
            shift: Shift key (Sneak) - VPT index 5
            ctrl: Ctrl key (Sprint) - VPT index 6
            e: E key (Inventory) - VPT index 7
            q: Q key (Drop) - VPT index 8
            f: F key (Swap Hands) - VPT index 10
            n1-n9: Number keys 1-9 (Hotbar slots) - VPT indices 11-19
        """
        self.controller_state = {
            "w": w, "a": a, "s": s, "d": d,
            "space": space, "shift": shift, "ctrl": ctrl,
            "e": e, "q": q, "f": f,
            "1": n1, "2": n2, "3": n3, "4": n4, "5": n5,
            "6": n6, "7": n7, "8": n8, "9": n9,
        }

    @command(
        "send_mouse_state",
        description="Update mouse state (buttons and camera movement)",
    )
    def send_mouse_state(
        self,
        left: bool = False,
        right: bool = False,
        middle: bool = False,
        dx: float = 0.0,
        dy: float = 0.0,
    ):
        """
        Update mouse button and camera movement state.
        Maps to VPT binary indices 20-22 and categorical camera action.

        Args:
            left: Left mouse button (Attack/Mine) - VPT index 20 (mouse.0)
            right: Right mouse button (Use/Place) - VPT index 21 (mouse.1)
            middle: Middle mouse button (Pick Item) - VPT index 22 (mouse.2)
            dx: Mouse X delta for camera yaw (raw pixels, will be mu-law encoded)
            dy: Mouse Y delta for camera pitch (raw pixels, will be mu-law encoded)
        """
        self.mouse_state["left"] = left
        self.mouse_state["right"] = right
        self.mouse_state["middle"] = middle
        # Accumulate deltas — multiple messages may arrive between game steps
        self.mouse_state["dx"] += dx
        self.mouse_state["dy"] += dy

    @command("reset_env", description="Reset the environment to a new world")
    def reset_environment(self):
        """Reset the environment to generate a new world."""
        if self.env is not None:
            obs = self.env.reset()
            self.current_obs = obs["pov"]
            logger.debug("Environment reset to new world")

    def __init__(self, config: DictConfig):
        """
        Initialize MineRL environment.

        Args:
            config: DictConfig loaded from configs/procgen.yaml and merged by Reactor
        """
        super().__init__()

        logger.info("Initializing MineRL...")
        print("DEBUG: Initializing MineRL...", flush=True)

        # Merge config with defaults from dataclass
        self.cfg = OmegaConf.structured(MineRLReactorConfig)
        self.cfg = OmegaConf.merge(self.cfg, config)

        self.fps = self.cfg.fps
        self.size = (self.cfg.height, self.cfg.width)

        # Create MineRL environment
        self.env = gym.make(self.cfg.env_name)

        # State variables — updated by command handlers, consumed by game loop
        self.controller_state = {}
        self.mouse_state = {"left": False, "right": False, "middle": False, "dx": 0.0, "dy": 0.0}
        self.current_obs = None

        logger.info("MineRL initialization complete")
        print("DEBUG: MineRL initialization complete", flush=True)

    def start_session(self) -> None:
        """
        Start the environment's main processing loop.

        This method runs the MineRL environment and emits frames to the Reactor runtime.
        """
        self._running = True

        logger.info("Starting MineRL session...")
        print("DEBUG: Starting MineRL session...", flush=True)

        # Reset input state
        self.controller_state = {}
        self.mouse_state = {"left": False, "right": False, "middle": False, "dx": 0.0, "dy": 0.0}

        # Reset environment to get initial observation
        obs = self.env.reset()
        # obs is a dict with key 'pov', shape: (360, 640, 3)
        self.current_obs = obs["pov"]

        logger.info(
            f"Environment initialized. Observation shape: {self.current_obs.shape}"
        )
        print(
            f"DEBUG: Environment initialized. Observation shape: {self.current_obs.shape}",
            flush=True,
        )
        logger.info("Session initialized, starting game loop...")
        print("DEBUG: Session initialized, starting game loop...", flush=True)

        # Calculate frame time based on FPS
        frame_time = 1.0 / self.fps
        logger.info(f"Running at {self.fps} FPS (frame time: {frame_time:.3f}s)")
        print(f"DEBUG: Running at {self.fps} FPS (frame time: {frame_time:.3f}s)", flush=True)

        try:
            last_frame_time = time.time()
            while not get_ctx().should_stop():
                # Consume accumulated mouse deltas and build action fresh
                # This avoids race conditions between command handlers and the game loop
                step_mouse = {
                    "left": self.mouse_state["left"],
                    "right": self.mouse_state["right"],
                    "middle": self.mouse_state["middle"],
                    "dx": self.mouse_state["dx"],
                    "dy": self.mouse_state["dy"],
                }
                self.mouse_state["dx"] = 0.0
                self.mouse_state["dy"] = 0.0

                current_action = input_to_action(
                    self.env, self.controller_state, step_mouse, self.cfg
                )

                # Step the environment
                obs, reward, done, info = self.env.step(current_action)

                # Extract frame from observation
                # obs is a dict with key 'pov', shape: (360, 640, 3)
                frame = obs["pov"]

                # Ensure frame is uint8
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                # Resize if configured size differs from native resolution
                if frame.shape[0] != self.cfg.height or frame.shape[1] != self.cfg.width:
                    frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))

                # Check if episode ended
                if done:
                    logger.info(f"Episode ended. Reward: {reward}")
                    print(f"DEBUG: Episode ended. Reward: {reward}", flush=True)
                    obs = self.env.reset()
                    frame = obs["pov"]
                    if frame.shape[0] != self.cfg.height or frame.shape[1] != self.cfg.width:
                        frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))

                # Update current observation
                self.current_obs = frame

                # Emit frame to reactor runtime
                get_ctx().emit_block(frame)

                # Sleep to maintain target FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = time.time()

        except Exception as e:
            logger.error(f"Error in session: {e}", exc_info=True)
            self._running = False
            time.sleep(2)
            raise e
        finally:
            self._running = False
            logger.info("MineRL session ended.")
            print("DEBUG: MineRL session ended.", flush=True)
