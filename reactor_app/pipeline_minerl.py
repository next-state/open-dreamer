"""
MineRL Reactor pipeline.

Streams live frames from a MineRL Minecraft environment and drives it with the
same keyboard and mouse events used by the previous world-model frontend.
"""
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from reactor_runtime.interface import (
    InputState,
    Output,
    ReactorPipeline,
    Video,
    connected,
    event,
)


@dataclass
class MineRLOutput(Output):
    main_video: Video


@dataclass
class MineRLState(InputState):
    _keyboard: Any = None
    _mouse: Any = None
    _use_policy: bool = False
    _seed: int = 0
    _reset_requested: bool = False


class MineRLPipeline(ReactorPipeline):
    state: MineRLState

    def load(self, config: dict[str, Any]) -> None:
        self._env_id = str(config.get("env_id", "MineRLBasaltFindCave-v0"))
        self._fps = float(config.get("fps", 20.0))
        if self._fps <= 0:
            raise ValueError("fps must be greater than 0")
        self._frame_interval = 1.0 / self._fps
        self._camera_sensitivity = float(config.get("camera_sensitivity", 0.15))
        self._max_camera_degrees = float(config.get("max_camera_degrees", 20.0))
        self._env = None
        self._obs = None

        if bool(config.get("warmup_env", True)):
            self._env = self._make_env()
            self._obs = self._reset_env(self._env, int.from_bytes(os.urandom(4), "big"))

    @connected
    async def on_connect(self) -> None:
        self.state._keyboard = {}
        self.state._mouse = {
            "left": False,
            "right": False,
            "middle": False,
            "dx": 0.0,
            "dy": 0.0,
            "dwheel": 0.0,
        }
        self.state._seed = int.from_bytes(os.urandom(4), "big")
        self.state._reset_requested = False
        self.state._use_policy = False

    def inference(self):
        if self._env is None:
            self._env = self._make_env()
        if self._obs is None:
            self._obs = self._reset_env(self._env, self.state._seed)

        last_frame_at = 0.0
        sent_initial_frame = False

        while True:
            if self.state._reset_requested:
                self._obs = self._reset_env(self._env, self.state._seed)
                self.state._reset_requested = False
                sent_initial_frame = True
                yield MineRLOutput(main_video=self._frame_from_obs(self._obs))
                last_frame_at = time.monotonic()
                continue

            if not sent_initial_frame:
                sent_initial_frame = True
                yield MineRLOutput(main_video=self._frame_from_obs(self._obs))
                last_frame_at = time.monotonic()
                continue

            action = self._build_action(self._env)
            self._obs, _reward, terminated, truncated, _info = self._step_env(self._env, action)

            self._consume_pulsed_inputs()

            if terminated or truncated:
                self._obs = self._reset_env(self._env, self.state._seed)
                self.state._reset_requested = False

            elapsed = time.monotonic() - last_frame_at
            if elapsed < self._frame_interval:
                time.sleep(self._frame_interval - elapsed)
            last_frame_at = time.monotonic()

            yield MineRLOutput(main_video=self._frame_from_obs(self._obs))

    def _make_env(self):
        try:
            import gym
            import minerl  # noqa: F401 - importing registers MineRL env IDs.
        except ModuleNotFoundError as exc:
            missing = exc.name or "minerl"
            raise RuntimeError(
                f"Missing dependency {missing!r}. Install `reactor_app/requirements.txt` "
                "into the Python environment used to launch Reactor. MineRL also needs "
                "a JDK available before it can build/install."
            ) from exc

        return gym.make(self._env_id)

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

    def _build_action(self, env: Any) -> dict[str, Any]:
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
        camera = np.asarray(
            [
                np.clip(dy, -self._max_camera_degrees, self._max_camera_degrees),
                np.clip(dx, -self._max_camera_degrees, self._max_camera_degrees),
            ],
            dtype=np.float32,
        )

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
    def send_keyboard_state(
        self,
        w: bool = False, a: bool = False, s: bool = False, d: bool = False,
        space: bool = False, shift: bool = False, ctrl: bool = False,
        e: bool = False, q: bool = False, f: bool = False,
        n1: bool = False, n2: bool = False, n3: bool = False, n4: bool = False,
        n5: bool = False, n6: bool = False, n7: bool = False, n8: bool = False,
        n9: bool = False, f3: bool = False,
    ):
        self.state._keyboard = {
            "w": w,
            "a": a,
            "s": s,
            "d": d,
            "space": space,
            "shift": shift,
            "ctrl": ctrl,
            "e": e,
            "q": q,
            "f": f,
            "f3": f3,
            "1": n1,
            "2": n2,
            "3": n3,
            "4": n4,
            "5": n5,
            "6": n6,
            "7": n7,
            "8": n8,
            "9": n9,
        }

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

    @event(name="send_mouse_wheel", description="Accumulate scroll-wheel ticks (sign = direction)")
    def send_mouse_wheel(self, dwheel: float = 0.0):
        mouse = self.state._mouse or {}
        mouse["dwheel"] = mouse.get("dwheel", 0.0) + float(dwheel)
        self.state._mouse = mouse

    @event(name="switch_to_policy", description="Toggle policy/manual control")
    def switch_to_policy(self, enable: bool = True):
        self.state._use_policy = enable

    @event(name="new_scene", description="Reset MineRL environment")
    def new_scene(self, seed: int = -1):
        self.state._seed = int(seed) if seed >= 0 else int.from_bytes(os.urandom(4), "big")
        self.state._reset_requested = True
