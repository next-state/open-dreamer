import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reactor_app"))

import pipeline as hybrid_pipeline  # noqa: E402
from pipeline import WorldModelPipeline, WorldModelState  # noqa: E402


class _FakeDynamics:
    cfg = SimpleNamespace(latent_mean=None, latent_std=None)

    def __call__(
        self,
        _action,
        _step_indices,
        _tau_indices,
        latent_noised,
        *,
        deterministic=True,
        caches=None,
    ):
        return latent_noised, (None, caches + ("dynamics",))


class _Tokenizer:
    def encode(self, frame, *, deterministic=True):
        assert deterministic is True
        assert frame.shape == (1, 1, 4, 5, 3)
        return jnp.ones((1, 1, 2, 2), dtype=jnp.float32), None

    def decode(self, latent, *, caches=None, deterministic=True):
        assert deterministic is True
        assert latent.shape == (1, 1, 2, 2)
        assert caches == ("tok",)
        return latent, caches + ("decoded",)


def _observe_test_inputs():
    schedule = SimpleNamespace(
        step_idx_ctx=1,
        tau_idx_ctx=2,
        tau_ctx=jnp.array(1.0, dtype=jnp.float32),
    )
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    rng = hybrid_pipeline.jax.random.PRNGKey(0)
    return schedule, frame, hybrid_pipeline._noop_action(), rng


def test_observe_frame_uses_uncached_tokenizer_encode_and_cached_decode():
    schedule, frame, action, rng = _observe_test_inputs()

    dynamics_cache, tokenizer_cache, _rng = hybrid_pipeline._observe_frame(
        _Tokenizer(),
        _FakeDynamics(),
        schedule,
        frame,
        action,
        ("dyn",),
        ("tok",),
        rng,
    )

    assert dynamics_cache == ("dyn", "dynamics")
    assert tokenizer_cache == ("tok", "decoded")


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
        self.reset_count = 0

    def reset(self, seed=None):
        self.reset_count += 1
        return {"pov": np.zeros((4, 5, 3), dtype=np.uint8)}, {"seed": seed}

    def step(self, action):
        self.actions.append(action)
        frame = np.full((4, 5, 3), len(self.actions), dtype=np.uint8)
        return {"pov": frame}, 0.0, False, False, {}


def _make_pipeline(monkeypatch, env):
    pipeline = WorldModelPipeline.__new__(WorldModelPipeline)
    pipeline.state = WorldModelState(
        _keyboard={},
        _mouse={"left": False, "right": False, "middle": False, "dx": 0.0, "dy": 0.0, "dwheel": 0.0},
        _use_policy=False,
        _seed=123,
        _reset_requested=False,
    )
    pipeline._env = env
    pipeline._obs = None
    pipeline._mesh = object()
    pipeline._frame_interval = 0.0
    pipeline._camera_sensitivity = 0.5
    pipeline._max_camera_degrees = 5.0
    pipeline._empty_dynamics_cache = ("empty_dyn",)
    pipeline._empty_tokenizer_cache = ("empty_tok",)
    pipeline._tokenizer = object()
    pipeline._dynamics = object()
    pipeline._latent_shape = (1, 1, 1, 1)
    pipeline._model_frame_shape = (4, 5, 3)
    pipeline._make_env = lambda: env

    observed_dynamics_caches = []
    generated_dynamics_caches = []
    generated_tokenizer_caches = []
    generated_rngs = []

    def fake_observe(_tokenizer, _dynamics, _frame, _action, dynamics_cache, tokenizer_cache, rng):
        observed_dynamics_caches.append(dynamics_cache)
        return dynamics_cache + ("observed",), tokenizer_cache + ("observed",), hybrid_pipeline.jax.random.PRNGKey(999)

    def fake_next(_tokenizer, _dynamics, _action, _latent_shape, dynamics_cache, tokenizer_cache, rng):
        generated_dynamics_caches.append(dynamics_cache)
        generated_tokenizer_caches.append(tokenizer_cache)
        generated_rngs.append(rng)
        frame = np.full((1, 1, 4, 5, 3), 9, dtype=np.uint8)
        return frame, None, dynamics_cache + ("generated",), tokenizer_cache + ("generated",), rng

    pipeline._observe_frame_jit = fake_observe
    pipeline._next_frame_jit = fake_next
    monkeypatch.setattr(hybrid_pipeline.jax, "set_mesh", lambda _mesh: nullcontext())
    monkeypatch.setattr(hybrid_pipeline.jax, "block_until_ready", lambda value: value)

    return pipeline, observed_dynamics_caches, generated_dynamics_caches, generated_tokenizer_caches, generated_rngs


def test_hybrid_starts_in_minerl_and_generates_from_old_empty_world_model_inputs(monkeypatch):
    env = _FakeEnv()
    pipeline, observed_caches, generated_caches, generated_tokenizer_caches, generated_rngs = _make_pipeline(monkeypatch, env)
    generator = pipeline.inference()

    first = next(generator)
    pipeline.switch_to_policy(True)
    second = next(generator)
    generator.close()

    assert first.main_video.shape == (4, 5, 3)
    assert second.main_video[0, 0, 0] == 9
    assert env.reset_count == 1
    assert env.actions == []
    assert observed_caches == [("empty_dyn",)]
    assert generated_caches == [("empty_dyn",)]
    assert generated_tokenizer_caches == [("empty_tok",)]
    np.testing.assert_array_equal(generated_rngs[0], hybrid_pipeline.jax.random.split(hybrid_pipeline.jax.random.PRNGKey(123))[1])


def test_hybrid_toggle_back_resumes_minerl_from_real_env(monkeypatch):
    env = _FakeEnv()
    pipeline, observed_caches, _generated_caches, _generated_tokenizer_caches, _generated_rngs = _make_pipeline(monkeypatch, env)
    generator = pipeline.inference()

    next(generator)
    pipeline.switch_to_policy(True)
    next(generator)
    pipeline.switch_to_policy(False)
    resumed = next(generator)
    stepped = next(generator)
    generator.close()

    assert resumed.main_video[0, 0, 0] == 0
    assert stepped.main_video[0, 0, 0] == 1
    assert len(env.actions) == 1
    assert observed_caches[-2:] == [("empty_dyn",), ("empty_dyn", "observed")]


def test_new_scene_resets_mode_and_requests_reset():
    pipeline = WorldModelPipeline.__new__(WorldModelPipeline)
    pipeline.state = WorldModelState(_use_policy=True, _reset_requested=False)

    pipeline.new_scene(seed=42)

    assert pipeline.state._seed == 42
    assert pipeline.state._use_policy is False
    assert pipeline.state._reset_requested is True


def test_prepare_minerl_runtime_dir_moves_logs_to_writable_path(tmp_path):
    pipeline = WorldModelPipeline.__new__(WorldModelPipeline)
    pipeline._minerl_runtime_dir = str(tmp_path / "minerl")
    original_cwd = Path.cwd()

    try:
        pipeline._prepare_minerl_runtime_dir()

        assert Path.cwd() == tmp_path / "minerl"
        assert (tmp_path / "minerl" / "logs").is_dir()
    finally:
        hybrid_pipeline.os.chdir(original_cwd)
