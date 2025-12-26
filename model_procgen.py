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

import numpy as np
import logging
import time
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from procgen import ProcgenEnv
from reactor_runtime import VideoModel, command, get_ctx

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


def input_to_action(mouse_pos: Tuple[int, int], controller_state: Dict[str, Any], action_dim: int = 7) -> int:
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
    key_map = [
        ('q', 5),  # Left-Jump
        ('e', 4),  # Right-Jump
        ('w', 3),  # Jump/Up
        ('s', 6),  # Down
        ('d', 1),  # Right
        ('a', 2),  # Left
    ]
    
    action_idx = 0  # Default: no movement
    for key, action in key_map:
        if controller_state.get(key, False):
            action_idx = action
            break
    
    return action_idx


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

    name: str = "procgen-coinrun"

    @command("send_keyboard_state", description="Update keyboard state (WASD+QE keys)")
    def send_keyboard_state(self, w: bool = False, a: bool = False, s: bool = False, 
                           d: bool = False, q: bool = False, e: bool = False):
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
        self.controller_state = {'w': w, 'a': a, 's': s, 'd': d, 'q': q, 'e': e}
        action = input_to_action((0, 0), self.controller_state, self.cfg.action_dim)
        self.current_action = action
        
    @command("reset_env", description="Reset the environment to a new level")
    def reset_environment(self):
        """Reset the environment to generate a new level."""
        if self.env is not None:
            obs = self.env.reset()
            logger.debug("Environment reset to new level")

    def __init__(self, fps: int = 15, size: Tuple[int, int] = (64, 64), cfg: Optional[ProcgenReactorConfig] = None, **kwargs):
        """
        Initialize Procgen environment.
        
        Args:
            fps: Target frames per second (CoinRun typically runs at 15 FPS)
            size: (height, width) of output video (CoinRun native is 64x64)
            cfg: ProcgenReactorConfig with environment settings
            **kwargs: Additional arguments
        """
        super().__init__()
        
        logger.debug("Initializing Procgen CoinRun...")
        
        # Handle configuration
        if cfg is None:
            cfg = ProcgenReactorConfig()
        
        self.cfg = cfg
        self.fps = fps
        self.size = size
        
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
        
        logger.debug("Procgen CoinRun initialization complete")

    def start_session(self):
        # Emit a single frame
        frame = np.random.rand(480, 640, 3)  # (H, W, 3) in RGB
        get_ctx().emit_block(frame)
        
        # Emit multiple frames at once
        frames = np.random.rand(10, 480, 640, 3)  # (N, H, W, 3)
        get_ctx().emit_block(frames)
        
        # Send a black frame
        get_ctx().emit_block(None)

    # def start_session(self) -> None:
    #     """
    #     Start the environment's main processing loop.
        
    #     This method runs the CoinRun environment and emits frames to the Reactor runtime.
    #     """
    #     self._running = True
        
    #     logger.debug("Starting Procgen session...")
        
    #     # Reset environment and initialize state
    #     self.current_action = 0  # Default to no movement (action 0)
    #     self.controller_state = {}
        
    #     # Reset environment to get initial observation
    #     obs = self.env.reset()
    #     # obs is a dict with key 'rgb', shape: (1, 64, 64, 3)
    #     self.current_obs = obs['rgb'][0]  # Extract first batch element: (64, 64, 3)
        
    #     logger.debug(f"Environment initialized. Observation shape: {self.current_obs.shape}")
    #     logger.debug("Session initialized, starting game loop...")
        
    #     try:
    #         while get_ctx()._stop_evt.is_set() is False:
    #             # Get current action from user input
    #             current_action = self.current_action
                
    #             # Step the environment
    #             obs, reward, done, info = self.env.step(np.array([current_action]))
                
    #             # Extract frame from observation
    #             # obs is a dict with key 'rgb', shape: (1, 64, 64, 3)
    #             frame = obs['rgb'][0]  # Extract first batch element: (64, 64, 3)
                
    #             # Ensure frame is uint8
    #             if frame.dtype != np.uint8:
    #                 frame = np.clip(frame, 0, 255).astype(np.uint8)
                
    #             # Check if episode ended
    #             if done[0]:
    #                 logger.debug(f"Episode ended. Reward: {reward[0]}")
    #                 # Reset environment
    #                 obs = self.env.reset()
    #                 frame = obs['rgb'][0]
                
    #             # Update current observation
    #             self.current_obs = frame
                
    #             # Emit frame to reactor runtime
    #             self.emit_frame(frame)
                
    #     except Exception as e:
    #         logger.error(f"Error in session: {e}", exc_info=True)
    #         self._running = False
    #         time.sleep(2)  # Fake machine resetting time
    #         raise e
    #     finally:
    #         time.sleep(2)  # Fake machine resetting time
    #         self._running = False
    #         if self.env is not None:
    #             self.env.close()
    #         logger.debug("Procgen session ended.")
