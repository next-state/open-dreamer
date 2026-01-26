"""
Hybrid Reactor: Procgen + World Model Imagination
==================================================

This reactor combines the Procgen CoinRun environment with the Dreamer world model.
It starts in "reality mode" showing frames from the actual game, while maintaining
the world model's KV caches in the background. A toggle command allows seamless
switching to "imagination mode" where the world model generates frames.

Data Flow:

REALITY MODE:
    Procgen env.step() -> emit_block(frame) -> User sees real game
                       -> update_caches() (background) -> KV caches updated

IMAGINATION MODE:
    next_frame_compiled() -> emit_block(frame) -> User sees imagination
"""

import jax
import jax.numpy as jnp
from flax import nnx
from dreamer.models import Dynamics, PolicyHeadMTP, TaskEmbedder, Tokenizer
from dreamer.generation import DenoiseSchedule, next_frame
from dreamer.parallel import create_data_model_parallel, MeshRules
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
import logging
import time
from procgen import ProcgenEnv
from reactor_runtime import VideoModel, command, get_ctx
from jax.tree_util import Partial

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FrameGenMode(Enum):
    """Operating mode for the frame generation reactor."""
    REALITY = "reality"          # Procgen provides frames
    IMAGINATION = "imagination"  # World model generates frames

class ActionGenMode(Enum):
    USER_INPUT = "user_input"
    POLICY = "policy"


@dataclass
class HybridReactorConfig:
    """Configuration for Hybrid Procgen + Dreamer reactor."""
    # Dynamics model checkpoint
    dynamics_ckpt: str = 'logs/dynamics/checkpoints'
    policy_ckpt: Optional[str] = None

    # Agent settings (other params loaded from checkpoint)
    task_id: int = 0

    # Denoising schedule
    num_steps: int = 4  # Number of denoising steps per frame
    tau_ctx: float = 0.9

    # Environment configuration
    env_name: str = "coinrun"
    num_levels: int = 0  # 0 = infinite procedural levels
    start_level: int = 0
    distribution_mode: str = "hard"

    # Batch size (always 1 for interactive)
    batch_size: int = 1


def input_to_action(controller_state: Dict[str, Any]) -> jax.Array:
    """
    Convert keyboard input to CoinRun action.
    
    Args:
        controller_state: Dictionary with keyboard button states
        
    Returns:
        Categorical integer action index of shape (1, 1)
    """
    # Key to action mapping (priority order: diagonals first, then cardinals)
    key_map = [
        ("q", 2),  # Left-Jump
        ("e", 8),  # Right-Jump
        ("w", 5),  # Jump/Up
        ("s", 3),  # Down
        ("d", 7),  # Right
        ("a", 1),  # Left
    ]

    action_idx = 4  # Default: no movement
    for key, action in key_map:
        if controller_state.get(key, False):
            action_idx = action
            break
    
    return jnp.full((1, 1), action_idx, dtype=jnp.int32)


def create_update_caches_fn(tokenizer:Tokenizer, dynamics: Dynamics, schedule:DenoiseSchedule, task_embeddings: None | jax.Array):
    """
    Create a function that updates KV caches by processing a frame through the model.

    This function encodes a frame to latent, processes it through dynamics,
    and decodes to update both caches. This keeps the world model synchronized
    with the procgen environment frames.
    """
    step_idx_ctx = schedule.step_idx_ctx
    tau_idx_ctx = schedule.tau_idx_ctx
    tau_ctx = schedule.tau_ctx

    def update_caches(
        frame: jax.Array,           # (1, 1, H, W, C)
        action: jax.Array,          # (1, 1)
        dynamics_cache,
        tokenizer_cache,
        rng: jax.Array,
    ):
        rng, enc_key, noise_key = jax.random.split(rng, 3)

        # 1. Encode frame to latent (returns unpacked)
        latent, _ = tokenizer.encode(
            frame,
            deterministic=True,
            rngs=nnx.Rngs(mae=enc_key),
        )  # Shape: (1, 1, n_latents, D_s)

        # 2. Add noise per tau_ctx (90% signal, 10% noise)
        noise = jax.random.normal(noise_key, latent.shape, dtype=latent.dtype)
        latent_noised = latent * tau_ctx + (1 - tau_ctx) * noise

        # 3. Process through dynamics to update dynamics cache
        step_indices = jnp.full((1, 1), step_idx_ctx, dtype=jnp.int32)
        tau_indices = jnp.full((1, 1), tau_idx_ctx, dtype=jnp.int32)

        _, (h_new, dynamics_cache_new) = dynamics(
            action,
            step_indices,
            tau_indices,
            latent_noised,
            task_embeddings=task_embeddings,
            deterministic=True,
            caches=dynamics_cache,
        )

        # 4. Update tokenizer cache by decoding
        _, tokenizer_cache_new = tokenizer.decode(
            latent,
            caches=tokenizer_cache,
            deterministic=True,
            rngs=None,
        )

        return h_new, dynamics_cache_new, tokenizer_cache_new, rng
    
    return update_caches


class HybridVideoModel(VideoModel):
    """
    Hybrid Procgen + Dreamer VideoModel.
    
    Combines real Procgen CoinRun gameplay with Dreamer world model imagination.
    Starts in reality mode with procgen frames, maintains world model caches
    in background, and can switch to imagination mode seamlessly.
    """

    name: str = "hybrid-coinrun"

    @command("send_keyboard_state", description="Update keyboard state (WASD+QE keys)")
    def send_keyboard_state(self, w: bool = False, a: bool = False, s: bool = False,
                           d: bool = False, q: bool = False, e: bool = False):
        """Update keyboard state - works in both reality and imagination modes."""
        self.controller_state = {'w': w, 'a': a, 's': s, 'd': d, 'q': q, 'e': e}
        self.current_action = input_to_action(self.controller_state)

    @command("switch_to_imagination", description="Switch between reality and imagination modes")
    def switch_to_imagination(self, enable: bool = True):
        """
        Toggle imagination mode.
        
        When enabled, switches from procgen frames to world model generation.
        The world model continues from the current cache state (built from procgen frames).
        """
        if enable and self.framegen_mode == FrameGenMode.REALITY:
            logger.info("Switching to IMAGINATION mode...")
            print("DEBUG: Switching to IMAGINATION mode...", flush=True)
            self.framegen_mode = FrameGenMode.IMAGINATION
        elif not enable and self.framegen_mode == FrameGenMode.IMAGINATION:
            logger.info("Switching to REALITY mode...")
            print("DEBUG: Switching to REALITY mode...", flush=True)
            # Reset caches and environment when returning to reality
            self._reset_caches()
            obs = self.env.reset()
            self.current_procgen_frame = obs["rgb"][0]
            self._warmup_with_frame(self.current_procgen_frame, 4)
            self.framegen_mode = FrameGenMode.REALITY

    @command("switch_to_policy", description="Switch between user input and policy control")
    def switch_to_policy(self, enable: bool = True):
        """
        Toggle policy mode.
        
        When enabled, the policy (if loaded) will take control of the agent.
        When disabled, control returns to user input.
        """
        if enable and self.actiongen_mode == ActionGenMode.USER_INPUT:
            if self.policy_head is None:
                logger.warning("No policy loaded. Cannot switch to policy mode.")
                print("DEBUG: No policy loaded. Cannot switch to policy mode.", flush=True)
                return

            logger.info("Switching to POLICY mode...")
            print("DEBUG: Switching to POLICY mode...", flush=True)
            self.actiongen_mode = ActionGenMode.POLICY
            
        elif not enable and self.actiongen_mode == ActionGenMode.POLICY:
            logger.info("Switching to USER_INPUT mode...")
            print("DEBUG: Switching to USER_INPUT mode...", flush=True)
            self.actiongen_mode = ActionGenMode.USER_INPUT

    @command("reset_env", description="Reset environment and caches")
    def reset_environment(self):
        """Reset environment and reinitialize caches."""
        logger.info("Resetting environment...")
        self._reset_caches()
        obs = self.env.reset()
        self.current_procgen_frame = obs["rgb"][0]
        self._warmup_with_frame(self.current_procgen_frame, 4)
        self.framegen_mode = FrameGenMode.REALITY

    def __init__(self, fps: int = 15, size: Tuple[int, int] = (64, 64),
                 cfg: Optional[HybridReactorConfig] = None):
        """
        Initialize hybrid reactor with both Procgen environment and Dreamer models.
        
        Args:
            fps: Target frames per second
            size: (height, width) of output video
            cfg: HybridReactorConfig with all settings
        """
        super().__init__()
        
        if cfg is None:
            cfg = HybridReactorConfig()
        
        logger.info("Initializing Hybrid Reactor...")
        print("DEBUG: Initializing Hybrid Reactor...", flush=True)
        
        self.cfg = cfg
        self.fps = fps
        self.size = size
        
        # === Procgen Environment Setup ===
        logger.info(f"Creating Procgen environment: {cfg.env_name}")
        self.env = ProcgenEnv(
            num_envs=1,
            env_name=cfg.env_name,
            num_levels=cfg.num_levels,
            start_level=cfg.start_level,
            distribution_mode=cfg.distribution_mode,
            render_mode="rgb_array",
        )
        
        # === Dreamer Model Setup ===
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
            
            # Initialize denoising schedule
            self.schedule = DenoiseSchedule.init(
                num_steps=cfg.num_steps,
                k_max=self.dynamics_cfg.k_max,
                tau_ctx=cfg.tau_ctx
            )
            
            # Compute latent shape from configs
            n_latents = self.tokenizer_cfg.decoder.n_latents
            D_s = self.tokenizer_cfg.encoder.d_bottleneck

            # Latents are now unpacked
            self.latent_shape = (1, 1, n_latents, D_s)
            self.window_size = self.tokenizer_cfg.dataset.T

            # Random key
            self.rng = jax.random.PRNGKey(0)
            
            # Load agent components if use_agent=True
            self.task_embedder = None
            self.policy_head = None
            self.task_embeddings = None

            if cfg.policy_ckpt is not None:

                logger.info(f"Loading agent from {cfg.policy_ckpt}")
                rng_key, task_key, pol_key = jax.random.split(jax.random.PRNGKey(42), 3)

                self.task_embedder = TaskEmbedder.from_pretrained(cfg.policy_ckpt, mesh_rules, nnx.Rngs(task_key))
                self.policy_head = PolicyHeadMTP.from_pretrained(cfg.policy_ckpt, mesh_rules, nnx.Rngs(pol_key))
                logger.info("Agent loaded successfully")

                self.task_id = cfg.task_id
                self.n_agent = self.task_embedder.n_agent
            
            
                # Create agent tokens
                task = jnp.full((1,), self.task_id, dtype=jnp.int32)
                self.task_embeddings = self.task_embedder(task=task, B=1, T=1)


            # JIT compile next_frame function for imagination mode
            logger.info("Compiling next_frame function...")
            self.next_frame_partial = jax.jit(Partial(
                next_frame,
                tokenizer=self.tokenizer,
                dynamics=self.dynamics,
                schedule=self.schedule,
                latent_shape=self.latent_shape,
            ))

            # JIT compile cache update function
            logger.info("Compiling update_caches function...")
            update_caches_fn = create_update_caches_fn(
                self.tokenizer, self.dynamics, self.schedule, self.task_embeddings
            )
            self.update_caches_compiled = jax.jit(update_caches_fn)

            # Initialize KV caches (with n_agent if using policy)
            logger.info("Initializing KV caches...")
            n_agent = 0 if self.policy_head is None else self.policy_head.L
            self.initial_dynamics_cache = self.dynamics.create_static_caches(
                batch_size=1,
                n_latents=n_latents,
                window_size=self.window_size,
                n_agent=n_agent,
                dtype=self.dynamics_cfg.dtype,
            )
            
            self.initial_tokenizer_cache = self.tokenizer.create_static_caches(
                batch_size=1,
                window_size=self.window_size,
                dtype=self.tokenizer_cfg.decoder.dtype,
            )
            
            # Active caches (reset per session)
            self.dynamics_cache = None
            self.tokenizer_cache = None
            
        # Mode and state
        self.framegen_mode = FrameGenMode.REALITY
        self.current_action = jnp.full((1, 1), 4, dtype=jnp.int32)  # No movement
        self.controller_state = {}
        self.current_procgen_frame = None
        self.h_last = None  # Hidden state for policy

        logger.info("Hybrid Reactor initialization complete")
        print("DEBUG: Hybrid Reactor initialization complete", flush=True)

    def _reset_caches(self):
        """Reset KV caches to initial empty state."""
        self.dynamics_cache = self.initial_dynamics_cache
        self.tokenizer_cache = self.initial_tokenizer_cache
        self.h_last = None

    def _warmup_with_frame(self, frame: np.ndarray, action: int):
        """
        Warm up caches with a single frame.
        
        Args:
            frame: Numpy array (H, W, C) in [0, 255]
            action: Integer action taken
        """
        frame_jax = jnp.array(frame)[None, None]  # (1, 1, H, W, C)
        action_jax = jnp.array([[action]], dtype=jnp.int32)
        
        self.rng, key = jax.random.split(self.rng)
        self.h_last, self.dynamics_cache, self.tokenizer_cache, self.rng = self.update_caches_compiled(
            frame_jax,
            action_jax,
            self.dynamics_cache,
            self.tokenizer_cache,
            key,
        )

    @property
    def current_action_int(self) -> int:
        """Get current action as integer for procgen."""
        if isinstance(self.current_action, jax.Array):
            return int(self.current_action[0, 0])
        return int(self.current_action)

    def start_session(self) -> None:
        """
        Start the hybrid reactor's main processing loop.
        
        Handles both reality mode (procgen frames with background cache updates)
        and imagination mode (world model generation).
        """
        self._running = True
        
        logger.info("Starting Hybrid session...")
        print("DEBUG: Starting Hybrid session...", flush=True)
        
        # Reset state
        self.framegen_mode = FrameGenMode.REALITY
        self.actiongen_mode = ActionGenMode.USER_INPUT
        self.current_action = jnp.full((1, 1), 4, dtype=jnp.int32)
        self.controller_state = {}
        
        # Reset caches
        self._reset_caches()
        
        # Reset environment
        obs = self.env.reset()
        self.current_procgen_frame = obs["rgb"][0]  # (64, 64, 3)
        
        # Warm up caches with initial frame
        logger.info("Warming up caches with initial frame...")
        self._warmup_with_frame(self.current_procgen_frame, 4)
        
        # Pre-compile next_frame function with dummy call
        logger.info("Pre-compiling next_frame function with dummy call...")
        print("DEBUG: Pre-compiling next_frame function...", flush=True)
        self.rng, compile_key = jax.random.split(self.rng)
        dummy_action = jnp.full((1, 1), 4, dtype=jnp.int32)
        
        # Make dummy call to trigger JIT compilation
        _, _, self.dynamics_cache, self.tokenizer_cache, self.rng = self.next_frame_compiled(
            action=dummy_action,
            dynamics_cache=self.dynamics_cache,
            tokenizer_cache=self.tokenizer_cache,
            rng=compile_key,
            task=None,
        )
        
        # Block until compilation completes (wait for result)
        # The JIT compilation happens during the first call
        logger.info("next_frame compilation complete")
        print("DEBUG: next_frame compilation complete", flush=True)
        
        # Emit initial frame
        get_ctx().emit_block(self.current_procgen_frame)
        
        logger.info(f"Session initialized. Mode: {self.framegen_mode.value}")
        print(f"DEBUG: Session initialized. Mode: {self.framegen_mode.value}", flush=True)
        
        # Calculate frame time based on FPS
        frame_time = 1.0 / self.fps
        logger.info(f"Running at {self.fps} FPS (frame time: {frame_time:.3f}s)")
        
        try:
            last_frame_time = time.time()
            while get_ctx()._stop_evt.is_set() is False:
                self.rng, key = jax.random.split(self.rng)
                
                if self.actiongen_mode == ActionGenMode.POLICY and isinstance(self.policy_head, PolicyHeadMTP) and self.h_last is not None:
                    logits = self.policy_head(self.h_last, deterministic=True)  # (1, 1, L, A)
                    action = jax.random.categorical(key, logits[:, 0, 0, :])  # (1,)
                    action = action[:, None].astype(jnp.int32)  # (1, 1)
                else:
                    action = self.current_action
                    
                if self.framegen_mode == FrameGenMode.REALITY:
                    # === REALITY MODE ===
                    # 1. Step procgen environment
                    action_int = self.current_action_int
                    obs, reward, done, info = self.env.step(np.array([action_int]))
                    frame = obs["rgb"][0]  # (64, 64, 3)
                    
                    # Handle episode end
                    if done[0]:
                        logger.info(f"Episode ended. Reward: {reward[0]}")
                        print(f"DEBUG: Episode ended. Reward: {reward[0]}", flush=True)
                        # Reset environment and caches
                        self._reset_caches()
                        obs = self.env.reset()
                        frame = obs["rgb"][0]
                        self._warmup_with_frame(frame, 4)
                    
                    # 2. Emit procgen frame to user
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    get_ctx().emit_block(frame)
                    
                    # 3. Background: update caches with this frame
                    frame_jax = jnp.array(frame)[None, None]  # (1, 1, H, W, C)
                    action_jax = jnp.array([[action_int]], dtype=jnp.int32)
                    
                    self.h_last, self.dynamics_cache, self.tokenizer_cache, self.rng = self.update_caches_compiled(
                        frame_jax,
                        action_jax,
                        self.dynamics_cache,
                        self.tokenizer_cache,
                        key,
                    )
                    
                    self.current_procgen_frame = frame
                    
                else:
                    # === IMAGINATION MODE ===
                    frame_jax, self.h_last, self.dynamics_cache, self.tokenizer_cache, self.rng = \
                        self.next_frame_compiled(
                            action=action,
                            dynamics_cache=self.dynamics_cache,
                            tokenizer_cache=self.tokenizer_cache,
                            rng=key,
                            task=None,
                            agent_tokens = self.task_embeddings
                        )

                    frame = np.array(frame_jax[0, 0])  # (H, W, C)
                    get_ctx().emit_block(frame)
                
                # FPS timing
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
            time.sleep(2)
            self._running = False
            self.dynamics_cache = None
            self.tokenizer_cache = None
            if self.env is not None:
                self.env.close()
            logger.info("Hybrid session ended.")
            print("DEBUG: Hybrid session ended.", flush=True)
