import jax
import jax.numpy as jnp
from flax import nnx
from dreamer.models import Dynamics, PolicyHeadMTP
from dreamer.generation import DenoiseSchedule, next_frame
from dreamer.parallel import create_data_model_parallel, MeshRules
from dreamer.types import Actions
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import logging
import time
from reactor_runtime import VideoModel, command, get_ctx

# Configure logging to show DEBUG level messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
    dynamics_ckpt: str = 'logs/dynamics/checkpoints'
    policy_ckpt: Optional[str] = None
    
    # Denoising schedule
    num_steps: int = 4  # Number of denoising steps per frame
    tau_ctx: float = .9
    
    # Batch size (usually 1 for interactive)
    batch_size: int = 1


def input_to_action(controller_state: Dict[str, Any]) -> Actions:
    """
    Convert keyboard input to CoinRun action.
    Used https://github.com/openai/coinrun/blob/master/coinrun/coinrun.cpp for reference


    Args:
        mouse_pos: (mouse_x, mouse_y) position (unused for CoinRun)
        controller_state: Dictionary with keyboard button states
        action_dim: Dimension of action space (7 for CoinRun)

    Returns:
        Actions object *without time dimension*: (B=1, *)
    """
    # Key to action mapping (priority order: diagonals first, then cardinals)
    key_map = [
        ("q", 2),  # Left-Jump # correct
        ("e", 8),  # Right-Jump
        ("w", 5),  # Jump/Up # correct
        ("s", 3),  # Down
        ("d", 7),  # Right # correct
        ("a", 1),  # Left #correct
    ]

    action_idx = 4  # Default: no movement
    for key, action in key_map:
        if controller_state.get(key, False):
            action_idx = action
            break
    
    return Actions(
        categorical=jnp.array([action_idx], dtype=jnp.int32)
    )


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

    # FIXME: the correct usage is policy_model.sample(h_t, greedy=True, rng=rng_policy)

    # # Get action logits from policy head
    # logits = policy_model.apply(policy_vars, agent_tokens, deterministic=False)
    
    # # Sample action from categorical distribution
    # action = jax.random.categorical(rng, logits)
    
    # return action

    raise NotImplementedError


class DreamerVideoModel(VideoModel):
    """
    Template showing integration of existing ML models with VideoModel interface.
    """

    name: str = "dreamer-coinrun"

    @command("send_keyboard_state", description="Update keyboard state (WASD+QE keys)")
    def send_keyboard_state(self, w: bool = False, a: bool = False, s: bool = False, 
                           d: bool = False, q: bool = False, e: bool = False):
        """
        Update keyboard state and compute action from current key presses.
        """
        self.controller_state = {'w': w, 'a': a, 's': s, 'd': d, 'q': q, 'e': e}
        action = input_to_action(self.controller_state)
        self.current_action = action
        self.use_agent = False  # User input overrides agent mode
        
    @command("use_agent", description="Switch to policy-based action selection")
    def switch_to_policy(self, use_agent: bool):
        self.use_agent = use_agent

    def __init__(self, fps: int = 30, size: Tuple[int, int] = (480, 640), cfg: Optional[ReactorConfig] = ReactorConfig()):
        """
        Initialize model. Heavy weight loading happens here during container startup, before any user connects.
        
        Args:
            fps: Target frames per second
            size: (height, width) of output video
            cfg: ReactorConfig with checkpoint paths and settings
            **kwargs: Additional arguments (can contain cfg parameters)
        """
        super().__init__()
        
        logger.info("Initializing Dreamer...")
        print("DEBUG: Initializing Dreamer...", flush=True)
        
        self.cfg, self.fps, self.size = cfg, fps, size
        
        # Load models from checkpoints
        assert isinstance(cfg, ReactorConfig)
        logger.info(f"Loading dynamics model and tokenizer from {cfg.dynamics_ckpt}")
        mesh, data_sharding = create_data_model_parallel(1, 1)
        mesh_rules = MeshRules(embed=None, mlp='model', attn='model', data='data')
        with jax.set_mesh(mesh):
            self.dynamics, self.tokenizer = Dynamics.from_pretrained(
                cfg.dynamics_ckpt, 
                mesh_rules=mesh_rules,
                rngs=nnx.Rngs(0)
            )
            self.dynamics_cfg = self.dynamics.cfg
            self.tokenizer_cfg = self.tokenizer.cfg
            
            # Load policy if checkpoint provided
            self.policy = None
            self.policy_vars = None
            if cfg.policy_ckpt is not None:
                raise NotImplementedError("Loading policy from checkpoint is not implemented yet")
                logger.info(f"Loading policy from {cfg.policy_ckpt}")
                self.policy, self.policy_vars, self.policy_cfg = PolicyHeadMTP.from_pretrained(cfg.policy_ckpt)
            
            # Initialize denoising schedule
            self.schedule = DenoiseSchedule.init(num_steps=cfg.num_steps, k_max=self.dynamics_cfg.k_max, tau_ctx=cfg.tau_ctx)
            
            # Compute latent shape from dynamics config
            H, W = self.size
            packing_factor = self.dynamics.cfg.packing_factor
            n_latents = self.tokenizer_cfg.decoder.n_latents
            
            # Calculate number of spatial tokens
            self.n_spatial = n_latents // packing_factor
            self.window_size = self.tokenizer_cfg.dataset.T//packing_factor
            
            # Get bottleneck dimension from encoder config
            D_s = self.tokenizer_cfg.encoder.d_bottleneck
            
            # Set latent shape: (1, 1, n_spatial, D_s)
            self.latent_shape = (1, 1, n_latents//packing_factor, D_s*packing_factor)
            
            # Initialize use_agent flag
            self.use_agent = self.policy is not None
            
            # Random key
            self.rng = jax.random.PRNGKey(0)
            
            # Initialize KV caches for both dynamics and tokenizer
            logger.info("Initializing KV caches...")
            self.initial_dynamics_cache = self.dynamics.create_static_caches(
                batch_size=1,
                n_spatial=self.n_spatial,
                window_size=self.window_size,
                dtype=self.dynamics_cfg.dtype,
            )
            
            self.initial_tokenizer_cache = self.tokenizer.create_static_caches(
                batch_size=1,
                window_size=self.window_size,
                dtype=self.tokenizer_cfg.decoder.dtype,
            )
            
            # Active caches (will be reset per session)
            self.dynamics_cache = None
            self.tokenizer_cache = None
            
            # Create JIT-compiled version of next_frame
            # Use jax.tree_util.Partial to properly handle non-hashable objects like Flax modules
            logger.info("Compiling next_frame function...")
            from jax.tree_util import Partial
            
            next_frame_partial = Partial(
                next_frame,
                tokenizer=self.tokenizer,
                dynamics=self.dynamics,
                schedule=self.schedule,
                latent_shape=self.latent_shape,
            )
            self.next_frame_compiled = jax.jit(next_frame_partial)
            
            logger.info("Dreamer initialization complete")
            print("DEBUG: Dreamer initialization complete", flush=True)

    def start_session(self) -> None:
        """
        Start the video model's main processing loop.
        
        This method should assume that the model was already loaded in memory, and it should simply
        start it's inner loop, also making available the emit_frame function for pushing frames.
        
        This demonstrates the most common integration pattern: delegating to an existing model pipeline
        that handles frame generation and emission.
        """
        self._running = True
        
        rng_warmup, rng_encoder, self.rng = jax.random.split(self.rng, num = 3)
        logger.info("Starting Dreamer session...")
        print("DEBUG: Starting Dreamer session...", flush=True)
        
        # Initialize session state
        self.current_action = jnp.full((1,1), 4, dtype=jnp.int32)  # Default to no movement (action 4)
        self.controller_state = {}
        
        logger.info(f"Latent shape: {self.latent_shape}")
        print(f"DEBUG: Latent shape: {self.latent_shape}", flush=True)
        
        # Reset caches to initial empty state
        self.dynamics_cache = self.initial_dynamics_cache
        self.tokenizer_cache = self.initial_tokenizer_cache
        
        # Initialize with context from canonical starting frames
        logger.info("Loading canonical starting frames...")
        init_frames = np.load('assets/start_frames/level_000.npy')  # Shape: (16, 64, 64, 3)
        init_frames_jax = jnp.array(init_frames)[None, :1]  # Add batch dim, take first frame -> (1, 1, 64, 64, 3)
        
        # Encode frames to latents
        logger.info("Encoding frames to latents...")
        init_latents, _ = self.tokenizer.encode(
            init_frames_jax,
            deterministic=True,
            packing_factor=self.dynamics.cfg.packing_factor,
            rngs=nnx.Rngs(mae=rng_encoder),
        )  # Shape: (1, T//packing, n_spatial, D_s*packing)
        
        # Warm up dynamics cache by processing context latents
        logger.info("Warming up dynamics cache...")
        assert isinstance(init_latents, jax.Array)
        B, T_ctx = init_latents.shape[0], init_latents.shape[1]
        
        # Create dummy actions for context (no-op actions)
        actions_ctx = jnp.full((B, T_ctx), 4, dtype=jnp.int32)  # Action 4 = no movement
        step_indices = jnp.full((B, T_ctx), self.schedule.step_idx_ctx, dtype=jnp.int32)
        tau_indices = jnp.full((B, T_ctx), self.schedule.tau_idx_ctx, dtype=jnp.int32)
        
        # Add slight noise to context latents as per tau_ctx
        latents_noised = init_latents*self.schedule.tau_ctx + (1-self.schedule.tau_ctx)*jax.random.normal(rng_warmup, shape=init_latents.shape, dtype=init_latents.dtype)
        
        # Run through dynamics to warm up cache
        _, (_, self.dynamics_cache) = self.dynamics(
            actions_ctx,
            step_indices,
            tau_indices,
            latents_noised,
            agent_tokens=None,
            deterministic=True,
            caches=self.dynamics_cache,
        )
        
        # Warm up tokenizer cache by decoding context frames
        logger.info("Warming up tokenizer cache...")
        _, self.tokenizer_cache = self.tokenizer.decode(
            init_latents,
            packing_factor=self.dynamics.cfg.packing_factor,
            caches=self.tokenizer_cache,
            deterministic=True,
            rngs=None,
        )
        
        # Show the last context frame to the user
        logger.info("Showing initial frame to user...")
        initial_frame = np.array(init_frames[-1])  # Last frame of context
        get_ctx().emit_block(initial_frame)
        
        logger.info("Session initialized, starting generation loop...")
        print("DEBUG: Session initialized, starting generation loop...", flush=True)
        
        # Calculate frame time based on FPS
        frame_time = 1.0 / self.fps
        logger.info(f"Running at {self.fps} FPS (frame time: {frame_time:.3f}s)")
        print(f"DEBUG: Running at {self.fps} FPS (frame time: {frame_time:.3f}s)", flush=True)
        
        try:
            last_frame_time = time.time()
            while get_ctx()._stop_evt.is_set() is False:
                self.rng, key = jax.random.split(self.rng)
                
                # Determine action: either from policy or from user input
                if self.use_agent:
                    # Use policy to generate action
                    raise NotImplementedError("Policy-based action generation not implemented")
                else:
                    # Use current action from user input
                    current_action = self.current_action
                
                # Generate next frame
                frame_jax, h, self.dynamics_cache, self.tokenizer_cache, self.rng = self.next_frame_compiled(
                    action=current_action,
                    dynamics_cache=self.dynamics_cache,
                    tokenizer_cache=self.tokenizer_cache,
                    rng=key,
                    task=None,  # Not using task conditioning in reactor mode
                )
                
                # Convert to numpy outside JIT boundary
                frame = np.array(frame_jax[0, 0])  # Extract (H, W, C) from batch
                
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
            self._running = False
            time.sleep(2)  # Fake machine resetting time
            raise e
        finally:
            time.sleep(2)  # Fake machine resetting time
            self._running = False
            self.dynamics_cache = None  # Clean up cache
            self.tokenizer_cache = None  # Clean up cache
            logger.info("Dreamer session ended.")
            print("DEBUG: Dreamer session ended.", flush=True)

