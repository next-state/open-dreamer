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
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

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

    # Camera sensitivity (degrees per tick when mouse/arrow keys are held)
    camera_sensitivity: float = 5.0


def input_to_action(
    env, mouse_pos: Tuple[int, int], controller_state: Dict[str, Any], camera_sensitivity: float = 5.0
) -> Dict[str, Any]:
    """
    Convert keyboard/mouse input to MineRL action dictionary.

    MineRL Action Space:
        forward:  Discrete(2) - W key
        back:     Discrete(2) - S key
        left:     Discrete(2) - A key
        right:    Discrete(2) - D key
        jump:     Discrete(2) - Space key
        sneak:    Discrete(2) - Shift key
        sprint:   Discrete(2) - Ctrl key
        attack:   Discrete(2) - Left click / F key
        use:      Discrete(2) - Right click / G key
        camera:   Box(-180, 180, shape=(2,)) - Mouse movement (pitch, yaw)
        ESC:      Discrete(2) - Never sent (ends episode)

    Camera is controlled via arrow keys or I/J/K/L:
        Arrow Up / I:    Look up (negative pitch)
        Arrow Down / K:  Look down (positive pitch)
        Arrow Left / J:  Look left (negative yaw)
        Arrow Right / L: Look right (positive yaw)

    Args:
        env: The MineRL gym environment (used for noop action)
        mouse_pos: (mouse_x, mouse_y) position (unused currently)
        controller_state: Dictionary with keyboard/mouse button states
        camera_sensitivity: Degrees of camera rotation per tick

    Returns:
        MineRL action dictionary
    """
    action = env.action_space.noop()

    # Movement keys
    action["forward"] = 1 if controller_state.get("w", False) else 0
    action["back"] = 1 if controller_state.get("s", False) else 0
    action["left"] = 1 if controller_state.get("a", False) else 0
    action["right"] = 1 if controller_state.get("d", False) else 0

    # Action keys
    action["jump"] = 1 if controller_state.get(" ", False) else 0
    action["sneak"] = 1 if controller_state.get("shift", False) else 0
    action["sprint"] = 1 if controller_state.get("ctrl", False) else 0
    action["attack"] = 1 if controller_state.get("f", False) else 0
    action["use"] = 1 if controller_state.get("g", False) else 0

    # Camera control via arrow keys or IJKL
    pitch = 0.0  # vertical: negative = look up, positive = look down
    yaw = 0.0    # horizontal: negative = look left, positive = look right

    if controller_state.get("arrowup", False) or controller_state.get("i", False):
        pitch -= camera_sensitivity
    if controller_state.get("arrowdown", False) or controller_state.get("k", False):
        pitch += camera_sensitivity
    if controller_state.get("arrowleft", False) or controller_state.get("j", False):
        yaw -= camera_sensitivity
    if controller_state.get("arrowright", False) or controller_state.get("l", False):
        yaw += camera_sensitivity

    action["camera"] = np.array([pitch, yaw], dtype=np.float32)

    # Never send ESC
    action["ESC"] = 0

    return action


@model(name="minerl", config="configs/procgen.yaml")
class MineRLVideoModel(VideoModel):
    """
    MineRL environment integration with Reactor VideoModel interface.

    This version uses the actual Minecraft environment for testing the Reactor integration
    before deploying the full Dreamer model.

    Demonstrates:
    - Method-based command system with automatic schema generation
    - Environment integration pattern
    - Async session management
    - Proper state reset between sessions
    """

    @command("send_keyboard_state", description="Update keyboard state (WASD + Space/Shift/Ctrl + IJKL camera + F attack + G use)")
    def send_keyboard_state(
        self,
        w: bool = False,
        a: bool = False,
        s: bool = False,
        d: bool = False,
        f: bool = False,
        g: bool = False,
        i: bool = False,
        j: bool = False,
        k: bool = False,
        l: bool = False,
        space: bool = False,
        shift: bool = False,
        ctrl: bool = False,
    ):
        """
        Update keyboard state and compute action from current key presses.

        Args:
            w: W key pressed (Forward)
            a: A key pressed (Strafe Left)
            s: S key pressed (Back)
            d: D key pressed (Strafe Right)
            f: F key pressed (Attack/Mine)
            g: G key pressed (Use/Place)
            i: I key pressed (Look Up)
            j: J key pressed (Look Left)
            k: K key pressed (Look Down)
            l: L key pressed (Look Right)
            space: Space key pressed (Jump)
            shift: Shift key pressed (Sneak)
            ctrl: Ctrl key pressed (Sprint)
        """
        self.controller_state = {
            "w": w, "a": a, "s": s, "d": d,
            "f": f, "g": g,
            "i": i, "j": j, "k": k, "l": l,
            " ": space, "shift": shift, "ctrl": ctrl,
        }
        self.current_action = input_to_action(
            self.env, (0, 0), self.controller_state, self.cfg.camera_sensitivity
        )

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

        # State variables
        self.current_action = None  # Will be set to noop on session start
        self.controller_state = {}
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

        # Reset state
        self.controller_state = {}

        # Reset environment to get initial observation
        obs = self.env.reset()
        # obs is a dict with key 'pov', shape: (360, 640, 3)
        self.current_obs = obs["pov"]
        self.current_action = self.env.action_space.noop()

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
                # Get current action from user input
                current_action = self.current_action

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
                    # Reset environment
                    obs = self.env.reset()
                    frame = obs["pov"]
                    if frame.shape[0] != self.cfg.height or frame.shape[1] != self.cfg.width:
                        frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))
                    self.current_action = self.env.action_space.noop()

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
