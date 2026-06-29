import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reactor_app"))

from pipeline_minerl import MineRLPipeline, MineRLState  # noqa: E402


class _FakeActionSpace:
    def __init__(self):
        self.actions = [
            "forward",
            "back",
            "left",
            "right",
            "jump",
            "sneak",
            "sprint",
            "inventory",
            "drop",
            "swapHands",
            "attack",
            "use",
            "pickItem",
            "hotbar.1",
            "hotbarNext",
            "hotbarPrev",
            "camera",
        ]

    def no_op(self):
        action = {name: 0 for name in self.actions}
        action["camera"] = np.zeros(2, dtype=np.float32)
        return action


class _FakeEnv:
    action_space = _FakeActionSpace()

    def __init__(self):
        self.actions = []
        self.closed = False
        self.reset_count = 0

    def reset(self, seed=None):
        self.reset_count += 1
        return {"pov": np.zeros((4, 5, 3), dtype=np.uint8)}, {"seed": seed}

    def step(self, action):
        self.actions.append(action)
        frame = np.full((4, 5, 3), len(self.actions), dtype=np.uint8)
        return {"pov": frame}, 0.0, False, False, {}

    def close(self):
        self.closed = True


def test_build_action_maps_reactor_input_to_minerl_action():
    pipeline = MineRLPipeline.__new__(MineRLPipeline)
    pipeline.state = MineRLState()
    pipeline.load({"camera_sensitivity": 0.5, "max_camera_degrees": 5, "warmup_env": False})
    pipeline.send_keyboard_state(w=True, space=True, ctrl=True, n1=True)
    pipeline.send_mouse_state(left=True, right=True, dx=20, dy=-20)
    pipeline.send_mouse_wheel(dwheel=-1)

    action = pipeline._build_action(_FakeEnv())

    assert action["forward"] == 1
    assert action["jump"] == 1
    assert action["sprint"] == 1
    assert action["hotbar.1"] == 1
    assert action["attack"] == 1
    assert action["use"] == 1
    assert action["hotbarNext"] == 1
    np.testing.assert_array_equal(action["camera"], np.array([-5, 5], dtype=np.float32))


def test_inference_resets_steps_yields_frames_and_keeps_env_warm():
    env = _FakeEnv()
    pipeline = MineRLPipeline.__new__(MineRLPipeline)
    pipeline.state = MineRLState(
        _keyboard={},
        _mouse={"dx": 0.0, "dy": 0.0, "dwheel": 0.0},
        _seed=123,
        _reset_requested=False,
    )
    pipeline.load({"fps": 1000, "warmup_env": False})
    pipeline._make_env = lambda: env

    generator = pipeline.inference()

    first = next(generator)
    second = next(generator)
    generator.close()

    assert first.main_video.shape == (4, 5, 3)
    assert second.main_video[0, 0, 0] == 1
    assert env.reset_count == 1
    assert len(env.actions) == 1
    assert env.closed is False
