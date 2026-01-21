"""
Procgen CoinRun Reactor Integration
====================================

This is a test version that uses the actual CoinRun environment instead of the Dreamer model.
Use this to verify that the Reactor integration works correctly before testing with the model.

User Input → input_to_action() → Action Index
                                      ↓
                            CoinRun Environment Step
                                      ↓
                              Numpy Frame (H,W,C)
                                      ↓
                            emit_frame() → User sees result
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from procgen import ProcgenEnv
from reactor_runtime import VideoModel, command, get_ctx
from reactor_runtime.model_api import model

# Configure logging to show DEBUG level messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProcgenReactorConfig:
    """Configuration for Procgen reactor runtime."""

    # Environment configuration
    env_name: str = "coinrun"
    num_levels: int = 0  # 0 = infinite procedural levels
    start_level: int = 0
    distribution_mode: str = "easy"  # "easy", "hard", or "exploration"

    # Rendering
    render_mode: str = "rgb_array"

    # Action space configuration (CoinRun has 7 discrete actions)
    action_dim: int = 7

    # Video settings
    fps: int = 5
    height: int = 64
    width: int = 64


def input_to_action(
    mouse_pos: Tuple[int, int], controller_state: Dict[str, Any], action_dim: int = 7
) -> int:
    """
    Convert keyboard input to CoinRun action.
    Used https://github.com/openai/coinrun/blob/master/coinrun/coinrun.cpp for reference

    CoinRun Action Space (NUM_ACTIONS = 7):
        Action 0: No movement (dx=0, dy=0)
        Action 1: Right → (dx=+1, dy=0) - mapped to D key
        Action 2: Left ← (dx=-1, dy=0) - mapped to A key
        Action 3: Jump/Up ↑ (dx=0, dy=+1) - mapped to W key
        Action 4: Right-Jump ↗ (dx=+1, dy=+1) - mapped to E key
        Action 5: Left-Jump ↖ (dx=-1, dy=+1) - mapped to Q key
        Action 6: Down ↓ (dx=0, dy=-1) - mapped to S key

    Args:
        mouse_pos: (mouse_x, mouse_y) position (unused for CoinRun)
        controller_state: Dictionary with keyboard button states
        action_dim: Dimension of action space (7 for CoinRun)

    Returns:
        Integer action index
    """
    # Key to action mapping (priority order: diagonals first, then cardinals)
    # Uses indices from docstring: 0=none, 1=right, 2=left, 3=jump, 4=right-jump, 5=left-jump, 6=down
    key_map = [
        ("q", 5),  # Left-Jump ↖
        ("e", 4),  # Right-Jump ↗
        ("w", 3),  # Jump/Up ↑
        ("s", 6),  # Down ↓
        ("d", 1),  # Right →
        ("a", 2),  # Left ←
    ]

    action_idx = 0  # Default: no movement
    for key, action in key_map:
        if controller_state.get(key, False):
            action_idx = action
            break

    return action_idx


@model(name="procgen-coinrun", config="configs/procgen.yaml")
class ProcgenVideoModel(VideoModel):
    """
    Procgen environment integration with Reactor VideoModel interface.

    This version uses the actual CoinRun environment for testing the Reactor integration
    before deploying the full Dreamer model.

    Demonstrates:
    - Method-based command system with automatic schema generation
    - Environment integration pattern
    - Async session management
    - Proper state reset between sessions
    """

    @command("send_keyboard_state", description="Update keyboard state (WASD+QE keys)")
    def send_keyboard_state(
        self,
        w: bool = False,
        a: bool = False,
        s: bool = False,
        d: bool = False,
        q: bool = False,
        e: bool = False,
    ):
        """
        Update keyboard state and compute action from current key presses.

        Args:
            w: W key pressed (Jump/Up)
            a: A key pressed (Left)
            s: S key pressed (Down)
            d: D key pressed (Right)
            q: Q key pressed (Left-Jump)
            e: E key pressed (Right-Jump)
        """
        self.controller_state = {"w": w, "a": a, "s": s, "d": d, "q": q, "e": e}
        action = input_to_action((0, 0), self.controller_state, self.cfg.action_dim)
        self.current_action = action

    @command("reset_env", description="Reset the environment to a new level")
    def reset_environment(self):
        """Reset the environment to generate a new level."""
        if self.env is not None:
            obs = self.env.reset()
            logger.debug("Environment reset to new level")

    def __init__(self, config: DictConfig):
        """
        Initialize Procgen environment.

        Args:
            config: DictConfig loaded from configs/procgen.yaml and merged by Reactor
        """
        super().__init__()

        logger.info("Initializing Procgen CoinRun...")
        print("DEBUG: Initializing Procgen CoinRun...", flush=True)

        # Merge config with defaults from dataclass
        self.cfg = OmegaConf.structured(ProcgenReactorConfig)
        self.cfg = OmegaConf.merge(self.cfg, config)

        self.fps = self.cfg.fps
        self.size = (self.cfg.height, self.cfg.width)

        # Create Procgen environment
        # num_envs=1 for single environment
        self.env = ProcgenEnv(
            num_envs=1,
            env_name=self.cfg.env_name,
            num_levels=self.cfg.num_levels,
            start_level=self.cfg.start_level,
            distribution_mode=self.cfg.distribution_mode,
            render_mode=self.cfg.render_mode,
        )

        # State variables
        self.current_action = 0  # Default to no movement (action 0)
        self.controller_state = {}
        self.current_obs = None

        logger.info("Procgen CoinRun initialization complete")
        print("DEBUG: Procgen CoinRun initialization complete", flush=True)

    def start_session(self) -> None:
        """
        Start the environment's main processing loop.

        This method runs the CoinRun environment and emits frames to the Reactor runtime.
        """
        self._running = True

        logger.info("Starting Procgen session...")
        print("DEBUG: Starting Procgen session...", flush=True)

        # Reset environment and initialize state
        self.current_action = 0  # Default to no movement (action 0)
        self.controller_state = {}

        # Reset environment to get initial observation
        obs = self.env.reset()
        # obs is a dict with key 'rgb', shape: (1, 64, 64, 3)
        self.current_obs = obs["rgb"][0]  # Extract first batch element: (64, 64, 3)

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
                obs, reward, done, info = self.env.step(np.array([current_action]))

                # Extract frame from observation
                # obs is a dict with key 'rgb', shape: (1, 64, 64, 3)
                frame = obs["rgb"][0]  # Extract first batch element: (64, 64, 3)

                # Ensure frame is uint8
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                # Check if episode ended
                if done[0]:
                    logger.info(f"Episode ended. Reward: {reward[0]}")
                    print(f"DEBUG: Episode ended. Reward: {reward[0]}", flush=True)
                    # Reset environment
                    obs = self.env.reset()
                    frame = obs["rgb"][0]

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
            time.sleep(2)  # Fake machine resetting time
            raise e
        finally:
            self._running = False
            # Note: Don't close self.env here - it persists across sessions
            # Environment cleanup happens when the runtime shuts down
            logger.info("Procgen session ended.")
            print("DEBUG: Procgen session ended.", flush=True)
