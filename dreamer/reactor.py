import jax
import jax.numpy as jnp
from dreamer.models import Tokenizer, Dynamics, PolicyHeadMTP
from dreamer.generation import DenoiseSchedule, next_latent
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import logging
import time
from reactor_runtime import VideoModel, command, get_ctx

logger = logging.getLogger(__name__)
"""
User Input → input_to_action() → Action Array
                                      ↓
                            new_frame() calls:
                            1. next_latent() - τ-ladder denoising
                            2. tokenizer.decode() - latent to pixels
                                      ↓
                              Numpy Frame (H,W,C)
                                      ↓
                            emit_frame() → User sees result

"""

@dataclass
class ReactorConfig:
    """Configuration for Dreamer reactor runtime."""
    tokenizer_ckpt: str
    dynamics_ckpt: str
    policy_ckpt: Optional[str] = None
    
    # Action space configuration
    action_dim: int = 121  # Discretized mouse actions (foveated)
    keyboard_dim: int = 23  # Binary keyboard actions
    
    # Denoising schedule
    num_steps: int = 4  # Number of denoising steps per frame
    k_max: int = 256
    tau_ctx: float = 0.9
    
    # Initial context
    context_length: int = 16  # Number of initial frames to use as context
    use_noise_init: bool = True  # Start from noise if no context provided
    
    # Batch size (usually 1 for interactive)
    batch_size: int = 1


def input_to_action(mouse_pos: Tuple[int, int], controller_state: Dict[str, Any], action_dim: int = 121) -> jax.Array:
    """
    Convert mouse position and controller state to action representation.
    
    Args:
        mouse_pos: (mouse_x, mouse_y) position
        controller_state: Dictionary with keyboard button states
        action_dim: Dimension of discretized mouse action space
        
    Returns:
        JAX array with shape (action_dim,) for mouse action
    """
    # For now, simple implementation: map mouse position to discrete action
    # In practice, you'd use foveated discretization like VPT
    # This is a placeholder - you should implement proper mouse action encoding
    mouse_x, mouse_y = mouse_pos
    
    # Simple discretization: map to action_dim bins
    # This is a simplified version - VPT uses more sophisticated foveated encoding
    action_idx = (mouse_x % action_dim)  # Placeholder logic
    
    action = jnp.zeros(action_dim)
    action = action.at[action_idx].set(1.0)
    
    return action


def policy(policy_model: PolicyHeadMTP, policy_vars: Dict, agent_tokens: jax.Array, rng: jax.Array) -> jax.Array:
    """
    Run policy model to select actions from agent tokens.
    
    Args:
        policy_model: Policy head model
        policy_vars: Policy model parameters
        agent_tokens: Agent token representations from dynamics (B, n_agent, d_model)
        rng: Random key for sampling
        
    Returns:
        Sampled action as JAX array
    """
    # Get action logits from policy head
    logits = policy_model.apply(policy_vars, agent_tokens, deterministic=False)
    
    # Sample action from categorical distribution
    action = jax.random.categorical(rng, logits)
    
    return action


def new_frame(
    tokenizer: Tokenizer,
    tokenizer_vars: Dict,
    dynamics: Dynamics,
    dynamics_vars: Dict,
    schedule: DenoiseSchedule,
    action: jax.Array,
    latent_shape: Tuple,
    dynamics_cache: Any,
    tokenizer_cache: Any,
    rng: jax.Array,
) -> Tuple[np.ndarray, Any, Any, jax.Array]:
    """
    Generate next frame using dynamics model and decode to pixels.
    
    Args:
        tokenizer: Tokenizer model for decoding
        tokenizer_vars: Tokenizer parameters
        dynamics: Dynamics model
        dynamics_vars: Dynamics parameters
        schedule: Denoising schedule
        action: Action to condition on (B, 1) or (B,)
        latent_shape: Shape of latent (B, 1, n_spatial, D_s)
        dynamics_cache: KV cache for dynamics model from previous steps
        tokenizer_cache: KV cache for tokenizer decoder from previous steps
        rng: Random key
        
    Returns:
        Tuple of (frame as numpy array, updated dynamics cache, updated tokenizer cache, updated rng)
    """
    # Generate next latent using τ-ladder denoising
    latent, h_last, dynamics_cache_updated, rng = next_latent(
        dynamics=dynamics,
        dyn_vars=dynamics_vars,
        schedule=schedule,
        action=action,
        latent_shape=latent_shape,
        rng=rng,
        agent_tokens=None,
        caches=dynamics_cache,
        latents_ctx=None,
        actions_ctx=None,
    )
    
    # Decode latent to frame
    # latent shape: (B, 1, n_spatial, D_s)
    frame, tokenizer_cache_updated = tokenizer.apply(
        tokenizer_vars,
        latent,
        packing_factor=dynamics.config.packing_factor,
        caches=tokenizer_cache,
        method=tokenizer.decode,
        deterministic=True,
    )
    
    # Convert to numpy and clip to valid range
    # frame shape: (B, 1, H, W, C)
    frame = jnp.clip(frame, 0, 255).astype(jnp.uint8)
    frame_np = np.array(frame[0, 0])  # Extract (H, W, C) from batch
    
    return frame_np, dynamics_cache_updated, tokenizer_cache_updated, rng

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
        action = input_to_action(self.current_mouse_pos, self.controller_state, self.cfg.action_dim)
        self.current_action = action
        
    @command("use_agent", description="Switch to policy-based action selection")
    def switch_to_policy(self, use_agent: bool):
        self.use_agent = use_agent

    def __init__(self, fps: int = 30, size: Tuple[int, int] = (480, 640), cfg: Optional[ReactorConfig] = None, **kwargs):
        """
        Initialize model. Heavy weight loading happens here during container startup, before any user connects.
        
        Args:
            fps: Target frames per second
            size: (height, width) of output video
            cfg: ReactorConfig with checkpoint paths and settings
            **kwargs: Additional arguments (can contain cfg parameters)
        """
        super().__init__(fps=fps, size=size, **kwargs)
        
        logger.debug("Initializing Dreamer...")
        
        # Handle configuration
        if cfg is None:
            # Try to construct from kwargs
            if 'tokenizer_ckpt' in kwargs and 'dynamics_ckpt' in kwargs:
                cfg = ReactorConfig(
                    tokenizer_ckpt=kwargs['tokenizer_ckpt'],
                    dynamics_ckpt=kwargs['dynamics_ckpt'],
                    policy_ckpt=kwargs.get('policy_ckpt', None),
                )
            else:
                raise ValueError("Must provide either cfg or tokenizer_ckpt/dynamics_ckpt in kwargs")
        
        self.cfg = cfg
        self.fps = fps
        self.size = size  # (H, W)
        
        # Load models from checkpoints
        logger.debug(f"Loading tokenizer from {cfg.tokenizer_ckpt}")
        self.tokenizer, self.tokenizer_vars, self.tokenizer_cfg = Tokenizer.from_pretrained(cfg.tokenizer_ckpt)
        
        logger.debug(f"Loading dynamics from {cfg.dynamics_ckpt}")
        self.dynamics, self.dynamics_vars, self.dynamics_cfg, _ = Dynamics.from_pretrained(cfg.dynamics_ckpt)
        
        # Load policy if checkpoint provided
        self.policy = None
        self.policy_vars = None
        if cfg.policy_ckpt is not None:
            raise NotImplementedError("Loading policy from checkpoint is not implemented yet")
            logger.debug(f"Loading policy from {cfg.policy_ckpt}")
            self.policy, self.policy_vars, self.policy_cfg = PolicyHeadMTP.from_pretrained(cfg.policy_ckpt)
        
        # Initialize denoising schedule
        self.schedule = DenoiseSchedule.init(
            num_steps=cfg.num_steps,
            k_max=cfg.k_max,
            tau_ctx=cfg.tau_ctx,
        )
        
        # Compute latent shape from dynamics config
        # This will be properly set when we know n_spatial from the actual image size
        self.latent_shape = None  # Will be set in start_session
        
        # State variables (will be initialized per session)
        self.dynamics_cache = None
        self.tokenizer_cache = None
        self.current_mouse_pos = (0, 0)
        self.current_action = None
        self.controller_state = {}
        self.use_agent = False
        
        # Random key
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
        
        # Initialize session state
        self.current_mouse_pos = (0, 0)
        self.current_action = jnp.zeros(self.cfg.action_dim)  # Default action
        self.controller_state = {}
        self.use_agent = False
        
        # Determine latent dimensions from the tokenizer and dynamics config
        H, W = self.size
        patch_size = self.tokenizer_cfg.patch_size
        packing_factor = self.dynamics.config.packing_factor
        
        # Calculate number of spatial tokens
        # After patching: (H/patch_size) * (W/patch_size) patches
        # After packing: patches // (packing_factor^2)
        patches_h = H // patch_size
        patches_w = W // patch_size
        n_patches = patches_h * patches_w
        n_spatial = n_patches // packing_factor ** 2
        
        # Get bottleneck dimension from encoder config
        D_s = self.tokenizer_cfg.encoder.d_bottleneck
        
        # Set latent shape: (batch_size, 1, n_spatial, D_s)
        self.latent_shape = (self.cfg.batch_size, 1, n_spatial, D_s)
        
        logger.debug(f"Latent shape: {self.latent_shape}")
        
        # Initialize KV caches for both dynamics and tokenizer
        # We need a large enough window for interactive generation
        window_size = 1024  # Large enough for continuous interactive use
        self.dynamics_cache = self.dynamics.create_static_caches(
            batch_size=self.cfg.batch_size,
            n_spatial=n_spatial,
            window_size=window_size,
        )
        
        self.tokenizer_cache = self.tokenizer.create_static_caches(
            batch_size=self.cfg.batch_size,
            window_size=window_size,
        )
        
        # Initialize with context if needed
        # For now, we start from noise and let the model generate from scratch
        # In a real implementation, you might want to:
        # 1. Show a starting frame to the user
        # 2. Encode it to get initial latents
        # 3. Warm up the cache with context
        
        logger.debug("Session initialized, starting generation loop...")
        
        try:
            while get_ctx()._stop_evt.is_set() is False:
                self.rng, key = jax.random.split(self.rng)
                
                # Determine action: either from policy or from user input
                if self.use_agent and self.policy is not None:
                    # Use policy to generate action
                    # Note: This requires agent_tokens from the last dynamics forward pass
                    # For simplicity, we'll use the current action if policy is enabled
                    # A full implementation would extract agent_tokens from dynamics
                    current_action = self.current_action
                else:
                    # Use current action from user input
                    current_action = self.current_action
                
                # Ensure action has correct shape (B,) or (B, 1)
                if current_action.ndim == 1:
                    current_action = current_action[None, :]  # Add batch dimension if needed
                
                # Generate next frame
                frame, self.dynamics_cache, self.tokenizer_cache, self.rng = new_frame(
                    tokenizer=self.tokenizer,
                    tokenizer_vars=self.tokenizer_vars,
                    dynamics=self.dynamics,
                    dynamics_vars=self.dynamics_vars,
                    schedule=self.schedule,
                    action=current_action,
                    latent_shape=self.latent_shape,
                    dynamics_cache=self.dynamics_cache,
                    tokenizer_cache=self.tokenizer_cache,
                    rng=key,
                )
                
                # Emit frame to reactor runtime
                self.emit_frame(frame)
                
        except Exception as e:
            self._running = False
            time.sleep(2)  # Fake machine resetting time
            raise e
        finally:
            time.sleep(2)  # Fake machine resetting time
            self._running = False
            self.dynamics_cache = None  # Clean up cache
            self.tokenizer_cache = None  # Clean up cache
            logger.debug("Model session ended.")

