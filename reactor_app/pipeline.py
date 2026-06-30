"""
Hybrid MineRL + Dreamer world-model Reactor pipeline.

The session starts in the real MineRL environment. Each real frame is encoded
and used to advance the Dreamer dynamics and tokenizer KV caches. The existing
frontend toggle then switches generation to the warmed world model.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dreamer.models import Tokenizer, TokenizerCaches

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from reactor_runtime import get_weights_path
from reactor_runtime.interface import (
    InputState,
    Output,
    ReactorPipeline,
    Video,
    connected,
    event,
)

from dreamer.actions import (
    Actions,
    NUM_BINARY_ACTIONS,
    NUM_CAMERA_CLASSES,
    mouse_movement_to_categorical,
)
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.generation import DenoiseSchedule, next_frame
from dreamer.parallel import build_parallel
from dreamer.utils import normalize_latents


@dataclass
class WorldModelOutput(Output):
    main_video: Video


@dataclass
class WorldModelState(InputState):
    _keyboard: Any = None
    _mouse: Any = None
    # False means real MineRL mode; True means Dreamer rollout mode.
    _use_policy: bool = False
    _seed: int = 0
    _reset_requested: bool = False


_KEY_TO_BINARY_IDX = {"w": 0, "a": 1, "s": 2, "d": 3, "space": 4, "shift": 5, "ctrl": 6, "e": 7, "q": 8, "f": 10, "1": 11, "2": 12, "3": 13, "4": 14, "5": 15, "6": 16, "7": 17, "8": 18, "9": 19, "f3": 25}


def _build_world_model_action(keyboard: dict[str, Any] | None, mouse: dict[str, Any] | None) -> Actions:
    """Map current frontend input to the VPT-format action tensor."""
    keyboard = keyboard or {}
    mouse = mouse or {}

    binary = np.zeros((NUM_BINARY_ACTIONS,), dtype=np.int32)
    for key, idx in _KEY_TO_BINARY_IDX.items():
        if keyboard.get(key, False):
            binary[idx] = 1
    if mouse.get("left", False):
        binary[20] = 1
    if mouse.get("right", False):
        binary[21] = 1
    if mouse.get("middle", False):
        binary[22] = 1

    dwheel = float(mouse.get("dwheel", 0.0))
    if dwheel < 0:
        binary[23] = 1
    elif dwheel > 0:
        binary[24] = 1

    dx = np.array([float(mouse.get("dx", 0.0))], dtype=np.float32)
    dy = np.array([float(mouse.get("dy", 0.0))], dtype=np.float32)
    categorical = mouse_movement_to_categorical(dx, dy).astype(np.int32)

    return Actions(binary=jnp.asarray(binary, dtype=jnp.int32)[None, :], categorical=jnp.asarray(categorical, dtype=jnp.int32), continuous=None)


def _noop_action() -> Actions:
    return Actions(binary=jnp.zeros((1, NUM_BINARY_ACTIONS), dtype=jnp.int32), categorical=jnp.full((1,), NUM_CAMERA_CLASSES // 2, dtype=jnp.int32), continuous=None)


def _mesh_context(mesh: Any) -> AbstractContextManager[Any]:
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    return mesh


def _observe_frame(
    tokenizer,
    dynamics,
    schedule: DenoiseSchedule,
    frame: jax.Array,
    action: Actions,
    dynamics_cache: Any,
    tokenizer_cache: TokenizerCaches,
    rng: jax.Array,
) -> tuple[Any, TokenizerCaches, jax.Array]:
    """Advance dynamics and tokenizer encoder/decoder caches from one real observed frame."""
    frame = jnp.asarray(frame, dtype=jnp.float32)[None, None, ...]
    latent, _, encoder_cache = tokenizer.encode(frame, deterministic=True, caches=tokenizer_cache.encoder)
    latent_norm = normalize_latents(latent, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)

    batch_size = latent_norm.shape[0]
    action = action[:, None, ...]
    step_indices = jnp.full((batch_size, 1), schedule.emax, dtype=jnp.int32)
    tau_indices = jnp.full((batch_size, 1), schedule.k_max, dtype=jnp.int32)

    _, (_, dynamics_cache_updated) = dynamics(action, step_indices, tau_indices, latent_norm, deterministic=True, caches=dynamics_cache)
    _, decoder_cache = tokenizer.decode(latent, caches=tokenizer_cache.decoder, deterministic=True)

    return dynamics_cache_updated, TokenizerCaches(encoder=encoder_cache, decoder=decoder_cache), rng


class WorldModelPipeline(ReactorPipeline):
    state: WorldModelState

    def load(self, config: dict[str, Any]) -> None:
        self._env_id = str(config.get("env_id", "MineRLBasaltFindCave-v0"))
        self._fps = float(config.get("fps", 20.0))
        if self._fps <= 0:
            raise ValueError("fps must be greater than 0")
        self._frame_interval = 1.0 / self._fps
        self._camera_sensitivity = float(config.get("camera_sensitivity", 0.15))
        self._max_camera_degrees = float(config.get("max_camera_degrees", 20.0))
        self._minerl_runtime_dir = str(config.get("minerl_runtime_dir", "/tmp/reactor_minerl"))
        self._allow_frame_resize = bool(config.get("allow_frame_resize", False))
        self._warned_frame_resize = False
        self._env = None
        self._obs = None

        self._load_world_model(config)

        if bool(config.get("warmup_env", True)):
            self._env = self._make_env()
            self._obs = self._reset_env(self._env, int.from_bytes(os.urandom(4), "big"))

    def _load_world_model(self, config: dict[str, Any]) -> None:
        ckpt_path = str(get_weights_path())
        print(f"[reactor_app] JAX backend={jax.default_backend()} devices={jax.devices()}")
        if jax.default_backend() == "cpu":
            raise RuntimeError(f"needs to run on GPU, currenlty jax.default_backend() == cpu")

        mesh, _data_sharding, mesh_rules = build_parallel("data")
        self._mesh = mesh
        with _mesh_context(mesh):
            bundle = DynamicsCheckpointBundle.from_pretrained(ckpt_path, mesh_rules=mesh_rules, rngs=nnx.Rngs(0), model_names={"dynamics_ema", "tokenizer"})
            if bundle.dynamics_ema is None or bundle.tokenizer is None:
                raise RuntimeError(f"Missing dynamics_ema/tokenizer in checkpoint at {ckpt_path}")
            self._dynamics = bundle.dynamics_ema
            self._tokenizer = bundle.tokenizer

            dyn_cfg = self._dynamics.cfg
            tok_cfg = self._tokenizer.cfg

            self._schedule = DenoiseSchedule.init(num_steps=int(config.get("num_steps", 4)), k_max=dyn_cfg.k_max, tau_ctx_target=float(config.get("tau_ctx_target", 0.9)))

            n_latents = tok_cfg.decoder.n_latents
            d_bottleneck = tok_cfg.encoder.d_bottleneck
            self._latent_shape = (1, 1, n_latents, d_bottleneck)
            self._model_frame_shape = (int(tok_cfg.decoder.H), int(tok_cfg.decoder.W), 3)

            assert isinstance(dyn_cfg.context_length, int) and dyn_cfg.context_length > 0
            assert isinstance(tok_cfg.decoder.context_length, int) and tok_cfg.decoder.context_length > 0

            self._empty_dynamics_cache = self._dynamics.create_static_caches(batch_size=1, n_latents=n_latents, window_size=dyn_cfg.context_length, n_agent=0, dtype=dyn_cfg.dtype)
            self._empty_tokenizer_cache = self._tokenizer.create_static_caches(
                batch_size=1,
                H=int(tok_cfg.decoder.H),
                W=int(tok_cfg.decoder.W),
                window_size=tok_cfg.decoder.context_length,
                dtype=tok_cfg.decoder.dtype,
            )

            schedule = self._schedule

            def _next_frame_fn(tokenizer, dynamics, action, latent_shape, dynamics_cache, tokenizer_cache, rng, task_embedding=None):
                frame, h, dynamics_cache, decoder_cache, rng = next_frame(tokenizer, dynamics, schedule, action, latent_shape, dynamics_cache, tokenizer_cache.decoder, rng, task_embedding)
                return frame, h, dynamics_cache, TokenizerCaches(encoder=tokenizer_cache.encoder, decoder=decoder_cache), rng

            def _observe_frame_fn(tokenizer, dynamics, frame, action, dynamics_cache, tokenizer_cache, rng):
                return _observe_frame(tokenizer, dynamics, schedule, frame, action, dynamics_cache, tokenizer_cache, rng)

            self._next_frame_jit = nnx.jit(_next_frame_fn, static_argnames=("latent_shape",))
            self._observe_frame_jit = nnx.jit(_observe_frame_fn)

            rng = jax.random.PRNGKey(0)
            warmup_steps = max(2, int(config.get("world_model_warmup_steps", 2)))
            warmup_dynamics_cache = self._empty_dynamics_cache
            warmup_tokenizer_cache = self._empty_tokenizer_cache
            for _ in range(warmup_steps):
                rng, key = jax.random.split(rng)
                _frame, _h, warmup_dynamics_cache, warmup_tokenizer_cache, rng = self._next_frame_jit(self._tokenizer, self._dynamics, _noop_action(), self._latent_shape, warmup_dynamics_cache, warmup_tokenizer_cache, key)
                jax.block_until_ready((_frame, warmup_dynamics_cache, warmup_tokenizer_cache, rng))
            zero_frame = jnp.zeros(self._model_frame_shape, dtype=jnp.uint8)
            _dc2, _tc2, rng = self._observe_frame_jit(self._tokenizer, self._dynamics, zero_frame, _noop_action(), self._empty_dynamics_cache, self._empty_tokenizer_cache, rng)
            jax.block_until_ready((_dc2, _tc2))
            self._warmup_rng = rng

    @connected
    def on_connect(self) -> None:
        # Per-session input state. Each new client starts with no held keys,
        # a zeroed mouse delta, and a fresh random seed (so different sessions
        # generate different scenes).
        self.state._keyboard = {}
        self.state._mouse = {"left": False, "right": False, "middle": False, "dx": 0.0, "dy": 0.0, "dwheel": 0.0}
        self.state._seed = int.from_bytes(os.urandom(4), "big")
        self.state._reset_requested = False

    def inference(self):
        if self._env is None:
            self._env = self._make_env()
        if self._obs is None:
            self._obs = self._reset_env(self._env, self.state._seed)

        rng = jax.random.PRNGKey(self.state._seed)
        dynamics_cache = self._empty_dynamics_cache
        tokenizer_cache = self._empty_tokenizer_cache
        sent_initial_frame = False
        was_world_model = False
        last_frame_at = 0.0

        with _mesh_context(self._mesh):
            while True:
                if self.state._reset_requested:
                    rng = jax.random.PRNGKey(self.state._seed)
                    dynamics_cache = self._empty_dynamics_cache
                    tokenizer_cache = self._empty_tokenizer_cache
                    self.state._use_policy = False
                    self.state._reset_requested = False
                    self._obs = self._reset_env(self._env, self.state._seed)
                    sent_initial_frame = False
                    was_world_model = False
                
                use_world_model = bool(self.state._use_policy)

                action = _build_world_model_action(self.state._keyboard, self.state._mouse)
                if not use_world_model:
                    # rng = jax.random.PRNGKey(self.state._seed)
                    minerl_action = self._build_minerl_action(self._env)
                    self._obs, _reward, terminated, truncated, _info = self._step_env(self._env, minerl_action)
                    frame = self._frame_from_obs(self._obs)
                    dynamics_cache, tokenizer_cache, rng = self._observe_real_frame(frame, action, dynamics_cache, tokenizer_cache, rng)


                if use_world_model:
                    rng, key = jax.random.split(rng)
                    frame_jax, _h, dynamics_cache, tokenizer_cache, rng = self._next_frame_jit(
                        self._tokenizer, self._dynamics, action,
                        self._latent_shape, dynamics_cache, tokenizer_cache, key,
                    )

                    frame = np.asarray(frame_jax[0, 0])
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)

                # consume accumulated mouse delta + scroll-wheel pulse
                self.state._mouse["dx"] = 0.0
                self.state._mouse["dy"] = 0.0
                self.state._mouse["dwheel"] = 0.0

                yield WorldModelOutput(main_video=frame)

    def _observe_real_frame(
        self,
        frame: np.ndarray,
        action: Actions,
        dynamics_cache: Any,
        tokenizer_cache: TokenizerCaches,
        rng: jax.Array,
    ):
        model_frame = self._resize_frame_for_model(frame)
        return self._observe_frame_jit(self._tokenizer, self._dynamics, jnp.asarray(model_frame), action, dynamics_cache, tokenizer_cache, rng)

    def _resize_frame_for_model(self, frame: np.ndarray) -> np.ndarray:
        target_h, target_w, target_c = self._model_frame_shape
        if frame.shape == self._model_frame_shape:
            return np.ascontiguousarray(frame)
        if frame.ndim != 3 or frame.shape[-1] != target_c:
            raise RuntimeError(f"Cannot adapt MineRL frame shape {frame.shape} to model shape {self._model_frame_shape}")

        height, width, _channels = frame.shape
        pad_h = target_h - height
        pad_w = target_w - width
        if width == target_w and pad_w == 0 and 0 < pad_h <= 32:
            top = pad_h // 2
            bottom = pad_h - top
            padded = np.pad(frame, ((top, bottom), (0, 0), (0, 0)), mode="constant", constant_values=0)
            return np.ascontiguousarray(padded.astype(np.uint8, copy=False))

        if not self._allow_frame_resize:
            raise RuntimeError(
                f"MineRL frame shape {frame.shape} is incompatible with model shape {self._model_frame_shape}. "
                "The world model was trained on high-resolution VPT frames padded to the tokenizer size; "
                "silently resizing low-resolution MineRL frames corrupts the KV cache. Use a VPT-compatible "
                "high-resolution observation source, or set allow_frame_resize=true only for diagnostics."
            )

        try:
            import cv2

            if not self._warned_frame_resize:
                print(f"[reactor_app] WARNING resizing observed frames from {frame.shape} to {self._model_frame_shape}; this is likely out-of-distribution for cache warmup")
                self._warned_frame_resize = True
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return np.ascontiguousarray(resized.astype(np.uint8, copy=False))
        except ModuleNotFoundError:
            if not self._warned_frame_resize:
                print(f"[reactor_app] WARNING resizing observed frames from {frame.shape} to {self._model_frame_shape}; this is likely out-of-distribution for cache warmup")
                self._warned_frame_resize = True
            resized = jax.image.resize(jnp.asarray(frame, dtype=jnp.float32), (target_h, target_w, target_c), method="bilinear")
            return np.ascontiguousarray(np.clip(np.asarray(resized), 0, 255).astype(np.uint8))

    def _sleep_until_next_frame(self, last_frame_at: float) -> float:
        elapsed = time.monotonic() - last_frame_at
        if last_frame_at > 0.0 and elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)
        return time.monotonic()

    def _make_env(self):
        try:
            import gym
            import minerl  # noqa: F401 - importing registers MineRL env IDs.
        except ModuleNotFoundError as exc:
            missing = exc.name or "minerl"
            raise RuntimeError(f"Missing dependency {missing!r}. Install the Reactor app dependencies from `reactor_app/pyproject.toml` into the Python environment used to launch Reactor. MineRL also needs a JDK available before it can build/install.") from exc

        self._prepare_minerl_runtime_dir()
        return gym.make(self._env_id)

    def _prepare_minerl_runtime_dir(self) -> None:
        runtime_dir = getattr(self, "_minerl_runtime_dir", "/tmp/reactor_minerl")
        os.makedirs(runtime_dir, exist_ok=True)
        os.chdir(runtime_dir)
        os.makedirs("logs", exist_ok=True)

    def _reset_env(self, env: Any, seed: int):
        try:
            reset_result = env.reset(seed=int(seed))
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(int(seed))
            reset_result = env.reset()

        if isinstance(reset_result, tuple):
            return reset_result[0]
        return reset_result

    def _step_env(self, env: Any, action: dict[str, Any]):
        step_result = env.step(action)
        if len(step_result) == 5:
            return step_result
        obs, reward, done, info = step_result
        return obs, reward, bool(done), False, info

    def _build_minerl_action(self, env: Any) -> dict[str, Any]:
        action = env.action_space.no_op()
        keyboard = self.state._keyboard or {}
        mouse = self.state._mouse or {}

        self._set_action(action, "forward", keyboard.get("w", False))
        self._set_action(action, "back", keyboard.get("s", False))
        self._set_action(action, "left", keyboard.get("a", False))
        self._set_action(action, "right", keyboard.get("d", False))
        self._set_action(action, "jump", keyboard.get("space", False))
        self._set_action(action, "sneak", keyboard.get("shift", False))
        self._set_action(action, "sprint", keyboard.get("ctrl", False))
        self._set_action(action, "inventory", keyboard.get("e", False))
        self._set_action(action, "drop", keyboard.get("q", False))
        self._set_action(action, "swapHands", keyboard.get("f", False))
        self._set_action(action, "attack", mouse.get("left", False))
        self._set_action(action, "use", mouse.get("right", False))
        self._set_action(action, "pickItem", mouse.get("middle", False))

        self._apply_hotbar(action, keyboard, mouse)
        self._apply_camera(action, mouse)
        return action

    def _apply_hotbar(self, action: dict[str, Any], keyboard: dict[str, Any], mouse: dict[str, Any]) -> None:
        for index in range(1, 10):
            self._set_action(action, f"hotbar.{index}", keyboard.get(str(index), False))

        dwheel = float(mouse.get("dwheel", 0.0))
        if dwheel < 0:
            self._set_action(action, "hotbarNext", True)
        elif dwheel > 0:
            self._set_action(action, "hotbarPrev", True)

    def _apply_camera(self, action: dict[str, Any], mouse: dict[str, Any]) -> None:
        dx = float(mouse.get("dx", 0.0)) * self._camera_sensitivity
        dy = float(mouse.get("dy", 0.0)) * self._camera_sensitivity
        camera = np.asarray([np.clip(dy, -self._max_camera_degrees, self._max_camera_degrees), np.clip(dx, -self._max_camera_degrees, self._max_camera_degrees)], dtype=np.float32)

        if "camera" in action:
            action["camera"] = camera

    def _set_action(self, action: dict[str, Any], key: str, enabled: bool) -> None:
        if key in action:
            action[key] = int(bool(enabled))

    def _consume_pulsed_inputs(self) -> None:
        mouse = self.state._mouse or {}
        mouse["dx"] = 0.0
        mouse["dy"] = 0.0
        mouse["dwheel"] = 0.0
        self.state._mouse = mouse

    def _frame_from_obs(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict) and "pov" in obs:
            frame = np.asarray(obs["pov"])
        elif isinstance(obs, dict) and "rgb" in obs:
            frame = np.asarray(obs["rgb"])
        else:
            frame = np.asarray(obs)

        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise RuntimeError(f"MineRL observation does not contain an RGB frame: shape={frame.shape}")
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    @event(name="send_keyboard_state", description="Set currently-held keys")
    def send_keyboard_state(self, w: bool = False, a: bool = False, s: bool = False, d: bool = False, space: bool = False, shift: bool = False, ctrl: bool = False, e: bool = False, q: bool = False, f: bool = False, n1: bool = False, n2: bool = False, n3: bool = False, n4: bool = False, n5: bool = False, n6: bool = False, n7: bool = False, n8: bool = False, n9: bool = False, f3: bool = False):
        self.state._keyboard = {"w": w, "a": a, "s": s, "d": d, "space": space, "shift": shift, "ctrl": ctrl, "e": e, "q": q, "f": f, "f3": f3, "1": n1, "2": n2, "3": n3, "4": n4, "5": n5, "6": n6, "7": n7, "8": n8, "9": n9}

    @event(name="send_mouse_state", description="Set mouse buttons; accumulate camera delta")
    def send_mouse_state(
        self,
        left: bool = False,
        right: bool = False,
        middle: bool = False,
        dx: float = 0.0,
        dy: float = 0.0,
    ):
        prev = self.state._mouse or {}
        self.state._mouse = {
            "left": left,
            "right": right,
            "middle": middle,
            "dx": prev.get("dx", 0.0) + dx,
            "dy": prev.get("dy", 0.0) + dy,
            "dwheel": prev.get("dwheel", 0.0),
        }

    @event(name="send_mouse_wheel", description="Accumulate scroll-wheel ticks")
    def send_mouse_wheel(self, dwheel: float = 0.0):
        mouse = self.state._mouse or {}
        mouse["dwheel"] = mouse.get("dwheel", 0.0) + float(dwheel)
        self.state._mouse = mouse

    @event(name="switch_to_policy", description="Toggle between MineRL and world-model rollout")
    def switch_to_policy(self, enable: bool = True):
        self.state._use_policy = bool(enable)

    @event(name="new_scene", description="Reset MineRL, KV caches, and mode")
    def new_scene(self, seed: int = -1):
        self.state._seed = int(seed) if seed >= 0 else int.from_bytes(os.urandom(4), "big")
        self.state._use_policy = False
        self.state._reset_requested = True


HybridMineRLWorldModelPipeline = WorldModelPipeline
