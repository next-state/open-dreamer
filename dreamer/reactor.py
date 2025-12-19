import jax
from dreamer.models import Tokenizer, Dynamics
# 
# 
# 
def input_to_action(mouse_pos, controller_state):
    pass

def policy(model, state):
    pass

def new_frame(model, schedule, action, cache, rng):
    pass
    
# 
# 
#  
"""
Template showing how to integrate existing ML models into VideoModel interface.

"""
import time
from typing import Tuple
import numpy as np
import logging
from reactor_runtime import VideoModel, command, get_ctx

logger = logging.getLogger(__name__)

class DreamerVideoModel(VideoModel):
    """
    Template showing integration of existing ML models with VideoModel interface.
    
    Demonstrates:
    - Method-based command system with automatic schema generation
    - Model pipeline integration pattern
    - Async session management
    - Proper state reset between sessions
    """

    name: str = "template-video"

    @command("send_mouse_control", description="Send mouse control inputs")
    def get_inputs(self, mouse_x: int, mouse_y: int):
        self.current_mouse_pos = (mouse_x, mouse_y)
        action = input_to_action(self.current_mouse_pos, self.controller_state)
        self.current_action = action
        
    @command("use_agent", description="Switch to policy-based action selection")
    def switch_to_policy(self, use_agent: bool):
        self.use_agent = use_agent

    def __init__(self, fps: int = 30, size: Tuple[int, int] = (480, 640), **kwargs):
        """Initialize model. Heavy weight loading happens here during container startup, before any user connects."""
        logger.debug("Initializing Dreamer...")
        self.tokenizer, self.tokenizer_vars, self.tokenizer_cfg = Tokenizer.from_pretrained(cfg.tokenizer_ckpt)
        self.dynamics, self.dynamics_vars, self.dynamics_cfg = Dynamics.from_pretrained(cfg.dynamics_ckpt)
        # if cfg.policy_ckpt is not None:
        #    self.policy, self.policy_vars, self.policy_cfg = Policy.from_pretrained(cfg.policy_ckpt)
        
        self.dynamics_cache, self.tokenizer_cache = None, None
        self.rng = jax.random.PRNGKey(0)
        logger.debug("Dreamer initialization complete")


    def start_session(self) -> None:
        """
        Start the video model's main processing loop.
        
        This method should assume that the model was already loaded in memory, and it should simply
        start it's inner loop, also making available the emit_frame function for pushing frames.
        
        This demonstrates the most common integration pattern: delegating to an existing model pipeline
        that handles frame generation and emission.
        """
        self._running = True
        
        logger.debug("Starting user session...")
        dynamics_cache, tokenizer_cache = None, None
        
        try:
            while get_ctx()._stop_evt.is_set() is False:
                self.rng, key = jax.random.split(self.rng)
                # current_action = self.policy(cache) if self.use_agent else self.current_action
                latent, dynamics_cache = new_frame(self.dynamics, self.schedule, current_action, dynamics_cache, key)
                # TODO: add delay if 
                frame, tokenizer_cache = self.tokenizer.decode(latent, tokenizer_cache)
                self.emit_frame(frame)
                
        except Exception as e:
            self._running = False
            time.sleep(2) #fake machine resetting time
            raise e
        finally:
            time.sleep(2) #fake machine resetting time
            self._running = False
            logger.debug("Model session ended.")

