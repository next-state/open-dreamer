"""
World Model — Reactor pipeline.

Dynamics-only autoregressive next-frame generation with action conditioning.
No prefill, no dataset: each client session starts from empty caches and
generates frames driven entirely by live keyboard + mouse input.
"""
from dataclasses import dataclass
from typing import Any

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


# ----------------------------------------------------------------------------
# Tracks: what we send back to the client.
# ----------------------------------------------------------------------------
@dataclass
class WorldModelOutput(Output):
    main_video: Video


# ----------------------------------------------------------------------------
# Per-client state. Public fields auto-generate `set_<field>` events; private
# fields (leading `_`) don't and are cleared on disconnect. Mutable defaults
# go in @connected below (mixing field(default_factory=...) with InputField
# breaks dataclass field ordering).
# ----------------------------------------------------------------------------
@dataclass
class WorldModelState(InputState):
    _keyboard: Any = None
    _mouse: Any = None
    # Set by the `switch_to_policy` event; left private to suppress the
    # auto-generated `set_use_policy` event.
    _use_policy: bool = False


# ----------------------------------------------------------------------------
# Input helpers — translate raw keyboard/mouse dicts into the VPT-format
# Actions tensor expected by the dynamics model.
# ----------------------------------------------------------------------------
_KEY_TO_BINARY_IDX = {
    "w": 0, "a": 1, "s": 2, "d": 3,
    "space": 4, "shift": 5, "ctrl": 6,
    "e": 7, "q": 8, "f": 10,
    "1": 11, "2": 12, "3": 13, "4": 14, "5": 15,
    "6": 16, "7": 17, "8": 18, "9": 19,
}


def _build_action(keyboard: dict, mouse: dict) -> Actions:
    """Map current keyboard/mouse state to a single-step Actions tensor."""
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

    dx = np.array([float(mouse.get("dx", 0.0))], dtype=np.float32)
    dy = np.array([float(mouse.get("dy", 0.0))], dtype=np.float32)
    categorical = mouse_movement_to_categorical(dx, dy).astype(np.int32)

    return Actions(
        binary=jnp.asarray(binary, dtype=jnp.int32)[None, :],
        categorical=jnp.asarray(categorical, dtype=jnp.int32),
        continuous=None,
    )


def _noop_action() -> Actions:
    return Actions(
        binary=jnp.zeros((1, NUM_BINARY_ACTIONS), dtype=jnp.int32),
        categorical=jnp.full((1,), NUM_CAMERA_CLASSES // 2, dtype=jnp.int32),
        continuous=None,
    )


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
class WorldModelPipeline(ReactorPipeline):
    state: WorldModelState

    def load(self, config: dict[str, Any]) -> None:
        # Output sizing
        self._fps = int(config.get("fps", 20))
        self._display_h = int(config.get("height", 360))
        self._display_w = int(config.get("width", 640))

        # Mesh + checkpoint
        weights_root = get_weights_path()
        subdir = config.get("dynamics_ckpt_subdir", "")
        ckpt_path = str(weights_root / subdir) if subdir else str(weights_root)

        mesh, _data_sharding, mesh_rules = build_parallel("data")
        self._mesh = mesh
        with jax.set_mesh(mesh):
            bundle = DynamicsCheckpointBundle.from_pretrained(
                ckpt_path,
                mesh_rules=mesh_rules,
                rngs=nnx.Rngs(0),
                model_names={"dynamics_ema", "tokenizer"},
            )
            if bundle.dynamics_ema is None or bundle.tokenizer is None:
                raise RuntimeError(f"Missing dynamics_ema/tokenizer in checkpoint at {ckpt_path}")
            self._dynamics = bundle.dynamics_ema
            self._tokenizer = bundle.tokenizer

            dyn_cfg = self._dynamics.cfg
            tok_cfg = self._tokenizer.cfg

            self._schedule = DenoiseSchedule.init(
                num_steps=int(config.get("num_steps", 4)),
                k_max=dyn_cfg.k_max,
                tau_ctx_target=float(config.get("tau_ctx_target", 0.9)),
            )

            n_latents = tok_cfg.decoder.n_latents
            d_bottleneck = tok_cfg.encoder.d_bottleneck
            self._latent_shape = (1, 1, n_latents, d_bottleneck)

            # Cache window: largest context_length declared anywhere, fallback 128.
            window_candidates = [
                getattr(tok_cfg.decoder, "context_length", None),
                getattr(tok_cfg.encoder, "context_length", None),
                getattr(dyn_cfg, "context_length", None),
            ]
            valid = [w for w in window_candidates if isinstance(w, int) and w > 0]
            self._window_size = max(valid) if valid else 128

            # Static (empty) KV caches — generation starts from these with no
            # context frames.
            self._empty_dynamics_cache = self._dynamics.create_static_caches(
                batch_size=1,
                n_latents=n_latents,
                window_size=self._window_size,
                n_agent=0,
                dtype=dyn_cfg.dtype,
            )
            _, self._empty_tokenizer_cache = self._tokenizer.create_static_caches(
                batch_size=1,
                window_size=self._window_size,
                dtype=tok_cfg.decoder.dtype,
            )

            # JIT next_frame with the schedule captured as a closure (tiny
            # arrays become constants; model weights flow through nnx.jit as
            # traced args).
            schedule = self._schedule

            def _next_frame_fn(tokenizer, dynamics, action, latent_shape,
                               dynamics_cache, tokenizer_cache, rng,
                               task_embedding=None):
                return next_frame(
                    tokenizer, dynamics, schedule, action, latent_shape,
                    dynamics_cache, tokenizer_cache, rng, task_embedding,
                )

            self._next_frame_jit = nnx.jit(_next_frame_fn, static_argnames=("latent_shape",))

            # Warmup: compile + execute once so the first user frame is fast.
            rng = jax.random.PRNGKey(0)
            _frame, _h, _dc, _tc, rng = self._next_frame_jit(
                self._tokenizer, self._dynamics, _noop_action(),
                self._latent_shape, self._empty_dynamics_cache,
                self._empty_tokenizer_cache, rng,
            )
            jax.block_until_ready((_dc, _tc))
            self._warmup_rng = rng

    @connected
    async def on_connect(self) -> None:
        # Per-session input state. Each new client starts with no held keys
        # and a zeroed mouse delta.
        self.state._keyboard = {}
        self.state._mouse = {"left": False, "right": False, "middle": False, "dx": 0.0, "dy": 0.0}

    def inference(self):
        rng = self._warmup_rng
        dynamics_cache = self._empty_dynamics_cache
        tokenizer_cache = self._empty_tokenizer_cache

        with jax.set_mesh(self._mesh):
            while True:
                rng, key = jax.random.split(rng)
                action = _build_action(self.state._keyboard, self.state._mouse)

                frame_jax, _h, dynamics_cache, tokenizer_cache, rng = self._next_frame_jit(
                    self._tokenizer, self._dynamics, action,
                    self._latent_shape, dynamics_cache, tokenizer_cache, key,
                )

                # consume accumulated mouse delta
                self.state._mouse["dx"] = 0.0
                self.state._mouse["dy"] = 0.0

                frame = np.asarray(frame_jax[0, 0])
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                yield WorldModelOutput(main_video=frame)

    # ------------------------------------------------------------------------
    # Client events. Names match what the existing frontend sends via
    # sendCommand(...); the data dict maps onto keyword args.
    # ------------------------------------------------------------------------
    @event(name="send_keyboard_state", description="Set currently-held keys")
    def send_keyboard_state(
        self,
        w: bool = False, a: bool = False, s: bool = False, d: bool = False,
        space: bool = False, shift: bool = False, ctrl: bool = False,
        e: bool = False, q: bool = False, f: bool = False,
        n1: bool = False, n2: bool = False, n3: bool = False, n4: bool = False,
        n5: bool = False, n6: bool = False, n7: bool = False, n8: bool = False,
        n9: bool = False,
    ):
        self.state._keyboard = {
            "w": w, "a": a, "s": s, "d": d,
            "space": space, "shift": shift, "ctrl": ctrl,
            "e": e, "q": q, "f": f,
            "1": n1, "2": n2, "3": n3, "4": n4, "5": n5,
            "6": n6, "7": n7, "8": n8, "9": n9,
        }

    @event(name="send_mouse_state", description="Set mouse buttons; accumulate camera delta")
    def send_mouse_state(
        self,
        left: bool = False, right: bool = False, middle: bool = False,
        dx: float = 0.0, dy: float = 0.0,
    ):
        # Accumulate dx/dy between inference steps so we don't lose deltas
        # that arrive faster than the model's tick rate.
        prev = self.state._mouse or {}
        self.state._mouse = {
            "left": left,
            "right": right,
            "middle": middle,
            "dx": prev.get("dx", 0.0) + dx,
            "dy": prev.get("dy", 0.0) + dy,
        }

    @event(name="switch_to_policy", description="Toggle policy/manual control (no-op until policy head wired)")
    def switch_to_policy(self, enable: bool = True):
        self.state._use_policy = enable
