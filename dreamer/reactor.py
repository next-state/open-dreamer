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


def input_to_action(mouse_pos: Tuple[int, int], controller_state: Dict[str, Any], action_dim: int = 7) -> jax.Array:
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
        JAX array - categorical integer action index
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
    
    # Return categorical integer (not one-hot)
    return jnp.array(action_idx, dtype=jnp.int32)


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
    task: jax.Array | None,
) -> Tuple[np.ndarray, Any, Any, Any, jax.Array]:
    """
    Generate next frame using dynamics model and decode to pixels.
    
    Args:
        tokenizer: Tokenizer model for decoding
        tokenizer_vars: Tokenizer parameters
        dynamics: Dynamics model
        dynamics_vars: Dynamics parameters
        schedule: Denoising schedule
        action: Action to condition on (B,) - categorical integer array
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
        action=action,  # Shape (1,) - categorical integer
        latent_shape=latent_shape,
        rng=rng,
        prefill_length=None,  # No prefill for interactive generation
        agent_tokens=None,  # Not using agent tokens in reactor mode
        caches=dynamics_cache,
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
    
    return frame_np, h_last, dynamics_cache_updated, tokenizer_cache_updated, rng

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
        self.use_agent = False  # User input overrides agent mode
        
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
                    dynamics_ckpt=kwargs['dynamics_ckpt'],
                    policy_ckpt=kwargs.get('policy_ckpt', None),
                )
            else:
                raise ValueError("Must provide either cfg or tokenizer_ckpt/dynamics_ckpt in kwargs")
        
        self.cfg, self.fps, self.size = cfg, fps, size
        
        # Load models from checkpoints
        logger.debug(f"Loading dynamics model and tokenizer from {cfg.dynamics_ckpt}")
        self.dynamics, self.dynamics_vars, self.dynamics_cfg, self.tokenizer, self.tokenizer_vars, self.tokenizer_cfg  = Dynamics.from_pretrained(cfg.dynamics_ckpt)
        
        # Load policy if checkpoint provided
        self.policy = None
        self.policy_vars = None
        if cfg.policy_ckpt is not None:
            raise NotImplementedError("Loading policy from checkpoint is not implemented yet")
            logger.debug(f"Loading policy from {cfg.policy_ckpt}")
            self.policy, self.policy_vars, self.policy_cfg = PolicyHeadMTP.from_pretrained(cfg.policy_ckpt)
        
        # Initialize denoising schedule
        self.schedule = DenoiseSchedule.init(num_steps=cfg.num_steps, k_max=cfg.k_max, tau_ctx=cfg.tau_ctx)
        
        # Compute latent shape from dynamics config
        H, W = self.size
        patch_size = self.tokenizer_cfg.patch_size
        packing_factor = self.dynamics.config.packing_factor
        
        # Calculate number of spatial tokens
        patches_h = H // patch_size
        patches_w = W // patch_size
        n_patches = patches_h * patches_w
        self.n_spatial = n_patches // packing_factor
        
        # Get bottleneck dimension from encoder config
        D_s = self.tokenizer_cfg.encoder.d_bottleneck
        
        # Set latent shape: (1, 1, n_spatial, D_s)
        self.latent_shape = (1, 1, self.n_spatial, D_s*packing_factor)
        
        # State variables (will be initialized per session)
        self.dynamics_cache = None
        self.tokenizer_cache = None
        self.current_action = jnp.array(0, dtype=jnp.int32)  # Default to no movement (action 0)
        self.controller_state = {}
        self.use_agent = self.policy is not None
        
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
        self.current_action = jnp.array(0, dtype=jnp.int32)  # Default to no movement (action 0)
        self.controller_state = {}
        
        # Determine latent dimensions from the tokenizer and dynamics config
        
        logger.debug(f"Latent shape: {self.latent_shape}")
        
        # Initialize KV caches for both dynamics and tokenizer
        # Window size: 1024 frames = ~34 seconds at 30 FPS. Cache will wrap after this period.
        window_size = 1024
        self.dynamics_cache = self.dynamics.create_static_caches(
            batch_size=self.cfg.batch_size,
            n_spatial=self.n_spatial,
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
                    raise NotImplementedError("Policy-based action generation not implemented")
                else:
                    # Use current action from user input
                    current_action = self.current_action
                
                # Ensure action has correct shape (B,) for dynamics model
                # Actions are categorical integers, shape should be (B,) or (B, 1)
                if current_action.ndim == 0:
                    # Scalar action, expand to (1,)
                    current_action = current_action[None]
                elif current_action.ndim == 2:
                    # (B, 1) -> (B,)
                    current_action = current_action.squeeze(axis=1)
                
                # Generate next frame
                frame, h, self.dynamics_cache, self.tokenizer_cache, self.rng = new_frame(
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
                    task=None,  # Not using task conditioning in reactor mode
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

