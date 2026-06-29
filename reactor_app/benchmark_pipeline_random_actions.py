from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import WorldModelPipeline, WorldModelState  # noqa: E402


class _FakeActionSpace:
    def __init__(self):
        self.actions = ["forward", "back", "left", "right", "jump", "sneak", "sprint", "inventory", "drop", "swapHands", "attack", "use", "pickItem", "hotbar.1", "hotbarNext", "hotbarPrev", "camera"]

    def no_op(self):
        action = {name: 0 for name in self.actions}
        action["camera"] = np.zeros(2, dtype=np.float32)
        return action


class _FakeEnv:
    action_space = _FakeActionSpace()

    def __init__(self, frame_shape: tuple[int, int, int]):
        self._frame_shape = frame_shape
        self._index = 0

    def reset(self, seed=None):
        self._index = 0
        return {"pov": np.zeros(self._frame_shape, dtype=np.uint8)}, {"seed": seed}

    def step(self, action):
        self._index += 1
        frame = np.full(self._frame_shape, self._index % 256, dtype=np.uint8)
        return {"pov": frame}, 0.0, False, False, {}


def _random_inputs(rng: np.random.Generator) -> tuple[dict[str, bool], dict[str, float | bool]]:
    keys = ("w", "a", "s", "d", "space", "shift", "ctrl", "e", "q", "f", "1", "2", "3", "4", "5", "6", "7", "8", "9", "f3")
    keyboard = {key: bool(rng.random() < 0.08) for key in keys}
    mouse = {"left": bool(rng.random() < 0.08), "right": bool(rng.random() < 0.05), "middle": bool(rng.random() < 0.02), "dx": float(rng.normal(0.0, 8.0)), "dy": float(rng.normal(0.0, 8.0)), "dwheel": float(rng.choice([-1.0, 0.0, 1.0], p=[0.03, 0.94, 0.03]))}
    return keyboard, mouse


def _apply_random_inputs(pipeline: WorldModelPipeline, rng: np.random.Generator) -> None:
    keyboard, mouse = _random_inputs(rng)
    pipeline.state._keyboard = keyboard
    pipeline.state._mouse = mouse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark WorldModelPipeline FPS with random actions, fake MineRL env frames, KV-cache updates, and world-model rollout.")
    parser.add_argument("--frames", type=int, default=120, help="Measured world-model frames after the env-to-world switch.")
    parser.add_argument("--env-frames", type=int, default=120, help="Measured fake MineRL frames before the env-to-world switch.")
    parser.add_argument("--env-warmup-frames", type=int, default=4)
    parser.add_argument("--warmup-frames", type=int, default=4)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--target-fps", type=float, default=0.0, help="Backward-compatible alias for --target-world-fps.")
    parser.add_argument("--target-world-fps", type=float, default=0.0)
    parser.add_argument("--target-env-cache-fps", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--tau-ctx-target", type=float, default=0.9)
    parser.add_argument("--require-accelerator", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    if args.env_frames <= 0:
        raise ValueError("--env-frames must be > 0")
    if args.env_warmup_frames < 0:
        raise ValueError("--env-warmup-frames must be >= 0")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if args.target_fps < 0 or args.target_world_fps < 0 or args.target_env_cache_fps < 0:
        raise ValueError("target FPS values must be >= 0")
    backend = jax.default_backend()
    print(f"jax_version={jax.__version__}")
    print(f"backend={backend}")
    print(f"devices={jax.devices()}")
    if args.require_accelerator and backend == "cpu":
        print("target_pass=false")
        print("target_reason=backend is cpu")
        raise SystemExit(3)

    rng = np.random.default_rng(args.seed)
    config = {"fps": args.fps, "num_steps": args.num_steps, "tau_ctx_target": args.tau_ctx_target, "warmup_env": False, "world_model_warmup_steps": 2}
    pipeline = WorldModelPipeline.__new__(WorldModelPipeline)
    pipeline.state = WorldModelState(_keyboard={}, _mouse={"left": False, "right": False, "middle": False, "dx": 0.0, "dy": 0.0, "dwheel": 0.0}, _use_policy=False, _seed=args.seed, _reset_requested=False)

    print(f"config={json.dumps(config, sort_keys=True)}")
    load_started = time.perf_counter()
    pipeline.load(config)
    print(f"load_seconds={time.perf_counter() - load_started:.3f}")

    fake_env = _FakeEnv(pipeline._model_frame_shape)
    pipeline._env = fake_env
    pipeline._obs = None
    pipeline._make_env = lambda: fake_env
    generator = pipeline.inference()

    try:
        _apply_random_inputs(pipeline, rng)
        first = next(generator)
        print(f"initial_shape={first.main_video.shape}")

        for _ in range(args.env_warmup_frames):
            _apply_random_inputs(pipeline, rng)
            next(generator)

        env_frame_times = []
        env_started = time.perf_counter()
        for index in range(args.env_frames):
            frame_started = time.perf_counter()
            _apply_random_inputs(pipeline, rng)
            output = next(generator)
            frame_seconds = time.perf_counter() - frame_started
            env_frame_times.append(frame_seconds)
            if (index + 1) % max(1, args.env_frames // 5) == 0:
                print(f"env_progress={index + 1} latest_shape={output.main_video.shape} frame_seconds={frame_seconds:.4f}")
        env_seconds = time.perf_counter() - env_started

        switch_started = time.perf_counter()
        pipeline.switch_to_policy(True)
        first_world = next(generator)
        switch_seconds = time.perf_counter() - switch_started
        print(f"first_world_shape={first_world.main_video.shape}")
        print(f"switch_world_seconds={switch_seconds:.4f}")

        for _ in range(args.warmup_frames):
            _apply_random_inputs(pipeline, rng)
            next(generator)

        world_frame_times = []
        world_started = time.perf_counter()
        for index in range(args.frames):
            frame_started = time.perf_counter()
            _apply_random_inputs(pipeline, rng)
            output = next(generator)
            frame_seconds = time.perf_counter() - frame_started
            world_frame_times.append(frame_seconds)
            if (index + 1) % max(1, args.frames // 5) == 0:
                print(f"world_progress={index + 1} latest_shape={output.main_video.shape} frame_seconds={frame_seconds:.4f}")
        world_seconds = time.perf_counter() - world_started
    finally:
        generator.close()

    env_fps = args.env_frames / env_seconds
    env_cache_seconds = env_seconds + switch_seconds
    env_cache_fps = args.env_frames / env_cache_seconds
    world_fps = args.frames / world_seconds
    env_median_frame_seconds = float(np.median(np.asarray(env_frame_times)))
    world_median_frame_seconds = float(np.median(np.asarray(world_frame_times)))
    target_world_fps = args.target_world_fps if args.target_world_fps > 0 else args.target_fps
    env_pass = args.target_env_cache_fps <= 0 or env_cache_fps >= args.target_env_cache_fps
    world_pass = target_world_fps <= 0 or world_fps >= target_world_fps
    result = {"env_frames": args.env_frames, "env_seconds": env_seconds, "env_fps": env_fps, "switch_world_seconds": switch_seconds, "env_cache_seconds": env_cache_seconds, "env_cache_fps": env_cache_fps, "env_median_frame_seconds": env_median_frame_seconds, "world_frames": args.frames, "world_seconds": world_seconds, "world_fps": world_fps, "world_median_frame_seconds": world_median_frame_seconds, "target_env_cache_fps": args.target_env_cache_fps, "target_world_fps": target_world_fps, "env_pass": env_pass, "world_pass": world_pass, "target_pass": env_pass and world_pass}
    print("results_json=" + json.dumps(result, sort_keys=True))
    print(f"env_fps={env_fps:.3f}")
    print(f"env_cache_fps={env_cache_fps:.3f}")
    print(f"env_median_frame_seconds={env_median_frame_seconds:.4f}")
    print(f"world_fps={world_fps:.3f}")
    print(f"world_median_frame_seconds={world_median_frame_seconds:.4f}")
    print(f"target_env_cache_fps={args.target_env_cache_fps:.3f}")
    print(f"target_world_fps={target_world_fps:.3f}")
    print(f"env_pass={str(env_pass).lower()}")
    print(f"world_pass={str(world_pass).lower()}")
    print(f"target_pass={str(result['target_pass']).lower()}")
    if not result["target_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
