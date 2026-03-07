"""
Hybrid Reactor: MineRL + World Model Imagination
=================================================

This reactor combines real MineRL gameplay with Dreamer world model imagination.
It starts in reality mode, keeps world model KV caches in sync from real frames,
and can switch to imagination mode for autoregressive generation.
"""
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import cv2
import gym
import jax
import jax.numpy as jnp
import minerl
import numpy as np
from flax import nnx
from jax.tree_util import Partial
from omegaconf import DictConfig, OmegaConf
from reactor_runtime import VideoModel, command, get_ctx
from reactor_runtime.model_api import model

from dreamer.actions import Actions, NUM_BINARY_ACTIONS, NUM_CAMERA_CLASSES, mouse_movement_to_categorical
from dreamer.checkpointing import DynamicsCheckpointBundle, HeadsCheckpointBundle
from dreamer.generation import DenoiseSchedule, next_frame
from dreamer.models import PolicyHeadMTP, TaskEmbedder
from dreamer.parallel import build_parallel
from dreamer.utils import normalize_latents


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _ensure_display_with_xvfb() -> None:
    """Restart under xvfb-run when running headless."""
    if os.environ.get("DISPLAY"):
        return

    if os.environ.get("REACTOR_XVFB_WRAPPED") == "1":
        raise RuntimeError(
            "DISPLAY is still unset after restarting with xvfb-run."
        )

    xvfb_run = shutil.which("xvfb-run")
    if xvfb_run is None:
        raise RuntimeError(
            "DISPLAY is unset and xvfb-run is not installed. "
            "Install xvfb or launch manually with xvfb-run."
        )

    cmd = [xvfb_run, "-a", "-s", "-screen 0 1024x768x24", sys.executable, *sys.argv]
    env = os.environ.copy()
    env["REACTOR_XVFB_WRAPPED"] = "1"
    logger.info("DISPLAY is unset; restarting process under xvfb-run")
    os.execvpe(cmd[0], cmd, env)


class FrameGenMode(Enum):
    REALITY = "reality"
    IMAGINATION = "imagination"


class ActionGenMode(Enum):
    USER_INPUT = "user_input"
    POLICY = "policy"


@dataclass
class MineRLHybridConfig:
    # Checkpoints
    dynamics_ckpt: str = "logs/dynamics/checkpoints"
    policy_ckpt: str | None = None

    # Agent
    task_id: int = 0

    # Denoising
    num_steps: int = 4
    tau_ctx_target: float = 0.9

    # Environment
    env_name: str = "MineRLBasaltFindCave-v0"

    # Video
    fps: int = 20
    height: int = 360
    width: int = 640


def input_to_env_action(
    env, controller_state: Dict[str, Any], mouse_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Map user input state to MineRL environment action dict."""
    action = env.action_space.no_op()

    action["forward"] = 1 if controller_state.get("w", False) else 0
    action["left"] = 1 if controller_state.get("a", False) else 0
    action["back"] = 1 if controller_state.get("s", False) else 0
    action["right"] = 1 if controller_state.get("d", False) else 0

    action["jump"] = 1 if controller_state.get("space", False) else 0
    action["sneak"] = 1 if controller_state.get("shift", False) else 0
    action["sprint"] = 1 if controller_state.get("ctrl", False) else 0

    action["inventory"] = 1 if controller_state.get("e", False) else 0
    action["drop"] = 1 if controller_state.get("q", False) else 0
    action["swapHands"] = 1 if controller_state.get("f", False) else 0

    for slot in range(1, 10):
        action[f"hotbar.{slot}"] = 1 if controller_state.get(str(slot), False) else 0

    action["attack"] = 1 if mouse_state.get("left", False) else 0
    action["use"] = 1 if mouse_state.get("right", False) else 0
    action["pickItem"] = 1 if mouse_state.get("middle", False) else 0

    dx = float(mouse_state.get("dx", 0.0))
    dy = float(mouse_state.get("dy", 0.0))
    action["camera"] = np.array([dy, dx], dtype=np.float32)  # [pitch, yaw]
    action["ESC"] = 0

    return action


def input_to_wm_action(
    controller_state: Dict[str, Any], mouse_state: Dict[str, Any], with_time_dim: bool
) -> Actions:
    """Map user input state to Dreamer VPT-format action tensors."""
    binary = np.zeros((NUM_BINARY_ACTIONS,), dtype=np.int32)

    key_to_idx = {
        "w": 0,
        "a": 1,
        "s": 2,
        "d": 3,
        "space": 4,
        "shift": 5,
        "ctrl": 6,
        "e": 7,
        "q": 8,
        "f": 10,
        "1": 11,
        "2": 12,
        "3": 13,
        "4": 14,
        "5": 15,
        "6": 16,
        "7": 17,
        "8": 18,
        "9": 19,
    }
    for key, idx in key_to_idx.items():
        if controller_state.get(key, False):
            binary[idx] = 1

    if mouse_state.get("left", False):
        binary[20] = 1
    if mouse_state.get("right", False):
        binary[21] = 1
    if mouse_state.get("middle", False):
        binary[22] = 1

    dx = np.array([float(mouse_state.get("dx", 0.0))], dtype=np.float32)
    dy = np.array([float(mouse_state.get("dy", 0.0))], dtype=np.float32)
    categorical = mouse_movement_to_categorical(dx, dy).astype(np.int32)  # (1,)

    if with_time_dim:
        binary = jnp.asarray(binary, dtype=jnp.int32)[None, None, :]
        categorical = jnp.asarray(categorical, dtype=jnp.int32)[None, :]
    else:
        binary = jnp.asarray(binary, dtype=jnp.int32)[None, :]
        categorical = jnp.asarray(categorical, dtype=jnp.int32)

    return Actions(binary=binary, categorical=categorical, continuous=None)


def create_noop_wm_action(with_time_dim: bool) -> Actions:
    """Create a no-op action for cache warmup/reset."""
    if with_time_dim:
        return Actions(
            binary=jnp.zeros((1, 1, NUM_BINARY_ACTIONS), dtype=jnp.int32),
            categorical=jnp.full((1, 1), NUM_CAMERA_CLASSES // 2, dtype=jnp.int32),
            continuous=None,
        )
    return Actions(
        binary=jnp.zeros((1, NUM_BINARY_ACTIONS), dtype=jnp.int32),
        categorical=jnp.full((1,), NUM_CAMERA_CLASSES // 2, dtype=jnp.int32),
        continuous=None,
    )


def create_update_caches_fn(tokenizer, dynamics, schedule: DenoiseSchedule, task_embedding):
    """Update dynamics/tokenizer caches from a real observed frame and action."""
    emax = schedule.emax
    k_max = schedule.k_max

    def update_caches(
        frame: jax.Array,  # (1, 1, H, W, C)
        action: Actions,   # (1, 1, ...)
        dynamics_cache,
        tokenizer_cache,
        rng: jax.Array,
    ):
        rng, enc_key = jax.random.split(rng)

        latent, _ = tokenizer.encode(
            frame,
            deterministic=True,
            rngs=nnx.Rngs(mae=enc_key),
        )  # (1, 1, n_latents, D_s)

        latent_norm = normalize_latents(latent, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)

        # tau=1.0 (clean), step_idx=emax — matches training prefill for ground truth frames
        step_indices = jnp.full((1, 1), emax, dtype=jnp.int32)
        tau_indices = jnp.full((1, 1), k_max, dtype=jnp.int32)

        _, (h_new, dynamics_cache_new) = dynamics(
            action,
            step_indices,
            tau_indices,
            latent_norm,
            task_embeddings=task_embedding,
            deterministic=True,
            caches=dynamics_cache,
        )

        _, tokenizer_cache_new = tokenizer.decode(
            latent,
            caches=tokenizer_cache,
            deterministic=True,
            rngs=None,
        )

        return h_new, dynamics_cache_new, tokenizer_cache_new, rng

    return update_caches


# @model(name="minerl_hybrid", config="configs/minerl_hybrid.yaml")
class MineRLHybridVideoModel(VideoModel):
    @command(
        "send_keyboard_state",
        description="Update keyboard state (WASD, Space/Shift/Ctrl, E/Q/F, 1-9)",
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
        self.controller_state = {
            "w": w, "a": a, "s": s, "d": d,
            "space": space, "shift": shift, "ctrl": ctrl,
            "e": e, "q": q, "f": f,
            "1": n1, "2": n2, "3": n3, "4": n4, "5": n5,
            "6": n6, "7": n7, "8": n8, "9": n9,
        }

    @command(
        "send_mouse_state",
        description="Update mouse buttons and camera movement",
    )
    def send_mouse_state(
        self,
        left: bool = False,
        right: bool = False,
        middle: bool = False,
        dx: float = 0.0,
        dy: float = 0.0,
    ):
        self.mouse_state["left"] = left
        self.mouse_state["right"] = right
        self.mouse_state["middle"] = middle
        self.mouse_state["dx"] += dx
        self.mouse_state["dy"] += dy

    @command("switch_to_imagination", description="Switch between reality and imagination modes")
    def switch_to_imagination(self, enable: bool = True):
        if enable and self.framegen_mode == FrameGenMode.REALITY:
            logger.info("Switching to IMAGINATION mode")
            self.framegen_mode = FrameGenMode.IMAGINATION
        elif not enable and self.framegen_mode == FrameGenMode.IMAGINATION:
            logger.info("Switching to REALITY mode")
            self._reset_caches()
            obs = self.env.reset()
            frame = self._to_display_frame(obs["pov"])
            self.current_obs = frame
            self._warmup_with_frame(frame)
            self.framegen_mode = FrameGenMode.REALITY

    @command("switch_to_policy", description="Switch between user input and policy control")
    def switch_to_policy(self, enable: bool = True):
        if enable and self.actiongen_mode == ActionGenMode.USER_INPUT:
            if self.policy_head is None:
                logger.warning("No policy loaded; cannot switch to policy mode")
                return
            logger.info("Switching to POLICY mode")
            self.actiongen_mode = ActionGenMode.POLICY
        elif not enable and self.actiongen_mode == ActionGenMode.POLICY:
            logger.info("Switching to USER_INPUT mode")
            self.actiongen_mode = ActionGenMode.USER_INPUT

    @command("reset_env", description="Reset environment and caches")
    def reset_environment(self):
        logger.info("Resetting environment and caches")
        self._reset_caches()
        obs = self.env.reset()
        frame = self._to_display_frame(obs["pov"])
        self.current_obs = frame
        self._warmup_with_frame(frame)
        self.framegen_mode = FrameGenMode.REALITY

    def __init__(self, config: DictConfig):
        super().__init__()
        _ensure_display_with_xvfb()

        self.cfg = OmegaConf.structured(MineRLHybridConfig)
        self.cfg = OmegaConf.merge(self.cfg, config)

        self.fps = self.cfg.fps
        self.size = (self.cfg.height, self.cfg.width)

        logger.info("Creating MineRL environment: %s", self.cfg.env_name)
        self.env = gym.make(self.cfg.env_name)

        logger.info("Loading world model bundle from: %s", self.cfg.dynamics_ckpt)
        mesh, _, mesh_rules = build_parallel("data")
        with jax.set_mesh(mesh):
            dyn_bundle = DynamicsCheckpointBundle.from_pretrained(
                self.cfg.dynamics_ckpt,
                mesh_rules=mesh_rules,
                rngs=nnx.Rngs(0),
                model_names={"dynamics_ema", "tokenizer"},
            )

            self.dynamics = dyn_bundle.dynamics_ema
            self.tokenizer = dyn_bundle.tokenizer
            if self.dynamics is None or self.tokenizer is None:
                raise ValueError("Failed to load dynamics_ema/tokenizer from dynamics bundle")

            self.dynamics_cfg = self.dynamics.cfg
            self.tokenizer_cfg = self.tokenizer.cfg

            self.schedule = DenoiseSchedule.init(
                num_steps=self.cfg.num_steps,
                k_max=self.dynamics_cfg.k_max,
                tau_ctx_target=self.cfg.tau_ctx_target,
            )

            n_latents = self.tokenizer_cfg.decoder.n_latents
            d_bottleneck = self.tokenizer_cfg.encoder.d_bottleneck
            self.latent_shape = (1, 1, n_latents, d_bottleneck)

            # Prefer explicit model context lengths; keep compatibility with older checkpoint meta.
            window_candidates = [
                getattr(self.tokenizer_cfg.decoder, "context_length", None),
                getattr(self.tokenizer_cfg.encoder, "context_length", None),
                getattr(self.dynamics_cfg, "context_length", None),
            ]
            legacy_dataset_cfg = getattr(self.tokenizer_cfg, "dataset", None)
            if legacy_dataset_cfg is not None:
                window_candidates.append(getattr(legacy_dataset_cfg, "T", None))

            valid_windows = [w for w in window_candidates if isinstance(w, int) and w > 0]
            self.window_size = max(valid_windows) if valid_windows else 128

            logger.info("Using cache window_size=%d", self.window_size)
            self.rng = jax.random.PRNGKey(0)

            self.task_embedder: TaskEmbedder | None = None
            self.policy_head: PolicyHeadMTP | None = None
            self.task_embedding = None

            self.model_height = int(getattr(self.tokenizer_cfg.decoder, "H", self.cfg.height))
            self.model_width = int(getattr(self.tokenizer_cfg.decoder, "W", self.cfg.width))
            logger.info(
                "Frame sizing: display=%dx%d, tokenizer_input=%dx%d",
                self.cfg.height,
                self.cfg.width,
                self.model_height,
                self.model_width,
            )

            if self.cfg.policy_ckpt is not None:
                logger.info("Loading policy bundle from: %s", self.cfg.policy_ckpt)
                policy_bundle = HeadsCheckpointBundle.from_pretrained(
                    self.cfg.policy_ckpt,
                    mesh_rules=mesh_rules,
                    rngs=nnx.Rngs(42),
                    model_names={"task_embedder", "policy_head"},
                )
                self.task_embedder = policy_bundle.task_embedder
                self.policy_head = policy_bundle.policy_head
                if self.task_embedder is None or self.policy_head is None:
                    raise ValueError("Failed to load task_embedder/policy_head from heads bundle")
                task = jnp.full((1,), self.cfg.task_id, dtype=jnp.int32)
                self.task_embedding = self.task_embedder(task=task, B=1, T=1)

            self.next_frame_compiled = jax.jit(
                Partial(
                    next_frame,
                    tokenizer=self.tokenizer,
                    dynamics=self.dynamics,
                    schedule=self.schedule,
                    latent_shape=self.latent_shape,
                )
            )

            update_caches_fn = create_update_caches_fn(
                self.tokenizer, self.dynamics, self.schedule, self.task_embedding
            )
            self.update_caches_compiled = jax.jit(update_caches_fn)

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

        self.dynamics_cache = None
        self.tokenizer_cache = None
        self.h_last = None

        self.framegen_mode = FrameGenMode.REALITY
        self.actiongen_mode = ActionGenMode.USER_INPUT
        self.controller_state = {}
        self.mouse_state = {"left": False, "right": False, "middle": False, "dx": 0.0, "dy": 0.0}
        self.current_obs = None
        self.current_env_action = self.env.action_space.no_op()

        logger.info("MineRL hybrid initialization complete")

    def _reset_caches(self):
        self.dynamics_cache = self.initial_dynamics_cache
        self.tokenizer_cache = self.initial_tokenizer_cache
        self.h_last = None

    @staticmethod
    def _to_uint8(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _to_display_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = self._to_uint8(frame)
        if frame.shape[0] == self.cfg.height and frame.shape[1] == self.cfg.width:
            return frame

        h, w = frame.shape[:2]
        if h >= self.cfg.height and w >= self.cfg.width:
            y0 = (h - self.cfg.height) // 2
            x0 = (w - self.cfg.width) // 2
            cropped = frame[y0:y0 + self.cfg.height, x0:x0 + self.cfg.width]
            if cropped.shape[0] == self.cfg.height and cropped.shape[1] == self.cfg.width:
                return cropped

        return cv2.resize(frame, (self.cfg.width, self.cfg.height))

    def _to_model_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = self._to_display_frame(frame)
        if frame.shape[0] == self.model_height and frame.shape[1] == self.model_width:
            return frame

        h, w = frame.shape[:2]
        dh = self.model_height - h
        dw = self.model_width - w

        if dh >= 0 and dw >= 0:
            top = dh // 2
            bottom = dh - top
            left = dw // 2
            right = dw - left
            return np.pad(
                frame,
                ((top, bottom), (left, right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        if h >= self.model_height and w >= self.model_width:
            y0 = (h - self.model_height) // 2
            x0 = (w - self.model_width) // 2
            cropped = frame[y0:y0 + self.model_height, x0:x0 + self.model_width]
            if cropped.shape[0] == self.model_height and cropped.shape[1] == self.model_width:
                return cropped

        return cv2.resize(frame, (self.model_width, self.model_height))

    def _warmup_with_frame(self, frame: np.ndarray):
        frame_model = self._to_model_frame(frame)
        frame_jax = jnp.asarray(frame_model)[None, None]
        noop_action = create_noop_wm_action(with_time_dim=True)
        self.rng, key = jax.random.split(self.rng)
        self.h_last, self.dynamics_cache, self.tokenizer_cache, self.rng = self.update_caches_compiled(
            frame_jax,
            noop_action,
            self.dynamics_cache,
            self.tokenizer_cache,
            key,
        )

    def _current_wm_action(self, with_time_dim: bool) -> Actions:
        return input_to_wm_action(self.controller_state, self.mouse_state, with_time_dim=with_time_dim)

    def _current_env_action(self) -> Dict[str, Any]:
        return input_to_env_action(self.env, self.controller_state, self.mouse_state)

    def start_session(self) -> None:
        self._running = True
        self.framegen_mode = FrameGenMode.REALITY
        self.actiongen_mode = ActionGenMode.USER_INPUT
        self.controller_state = {}
        self.mouse_state = {"left": False, "right": False, "middle": False, "dx": 0.0, "dy": 0.0}
        self.current_env_action = self.env.action_space.no_op()

        self._reset_caches()

        obs = self.env.reset()
        print("Environment reset")
        frame = self._to_display_frame(obs["pov"])
        self.current_obs = frame
        self._warmup_with_frame(frame)
        print("first frame warmup complete")
        
        self.rng, compile_key = jax.random.split(self.rng)
        dummy_action = create_noop_wm_action(with_time_dim=False)
        self.next_frame_compiled(
            action=dummy_action,
            dynamics_cache=self.dynamics_cache,
            tokenizer_cache=self.tokenizer_cache,
            rng=compile_key,
            task_embedding=self.task_embedding,
        )

        print("Warmup complete")
        get_ctx().get_track().emit(frame)
        frame_time = 1.0 / self.fps

        try:
            last_frame_time = time.time()
            while not get_ctx().should_stop():
                self.rng, key, policy_key = jax.random.split(self.rng, 3)

                if (
                    self.actiongen_mode == ActionGenMode.POLICY
                    and isinstance(self.policy_head, PolicyHeadMTP)
                    and self.h_last is not None
                ):
                    sampled = self.policy_head.sample(
                        self.h_last, deterministic=False, rng=policy_key
                    )
                    wm_action_no_time = Actions(
                        binary=sampled.binary[:, 0, 0, :] if sampled.binary is not None else None,
                        categorical=sampled.categorical[:, 0, 0] if sampled.categorical is not None else None,
                        continuous=sampled.continuous[:, 0, 0, :] if sampled.continuous is not None else None,
                    )
                else:
                    wm_action_no_time = self._current_wm_action(with_time_dim=False)

                if self.framegen_mode == FrameGenMode.REALITY:
                    env_action = self._current_env_action()
                    obs, reward, done, _ = self.env.step(env_action)
                    frame = self._to_display_frame(obs["pov"])

                    if done:
                        logger.info("Episode ended. Reward: %s", reward)
                        self._reset_caches()
                        obs = self.env.reset()
                        frame = self._to_display_frame(obs["pov"])
                        self._warmup_with_frame(frame)

                    self.current_obs = frame
                    get_ctx().get_track().emit(frame)

                    frame_jax = jnp.asarray(self._to_model_frame(frame))[None, None]
                    wm_action_with_time = self._current_wm_action(with_time_dim=True)
                    self.h_last, self.dynamics_cache, self.tokenizer_cache, self.rng = self.update_caches_compiled(
                        frame_jax,
                        wm_action_with_time,
                        self.dynamics_cache,
                        self.tokenizer_cache,
                        key,
                    )
                else:
                    frame_jax, self.h_last, self.dynamics_cache, self.tokenizer_cache, self.rng = self.next_frame_compiled(
                        action=wm_action_no_time,
                        dynamics_cache=self.dynamics_cache,
                        tokenizer_cache=self.tokenizer_cache,
                        rng=key,
                        task_embedding=self.task_embedding,
                    )
                    frame = self._to_display_frame(np.asarray(frame_jax[0, 0]))
                    get_ctx().get_track().emit(frame)

                self.mouse_state["dx"] = 0.0
                self.mouse_state["dy"] = 0.0

                current_time = time.time()
                elapsed = current_time - last_frame_time
                sleep_time = max(0.0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = time.time()

        except Exception as e:
            logger.error("Error in session: %s", e, exc_info=True)
            self._running = False
            raise
        finally:
            self._running = False
            self.dynamics_cache = None
            self.tokenizer_cache = None
            if self.env is not None:
                self.env.close()
            logger.info("MineRL hybrid session ended")
