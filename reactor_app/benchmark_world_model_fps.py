from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFAULT_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from dreamer.actions import Actions, NUM_BINARY_ACTIONS, NUM_CAMERA_CLASSES


if not hasattr(jax, "set_mesh"):
    jax.set_mesh = lambda mesh: mesh  # type: ignore[attr-defined]


@contextmanager
def mesh_context(mesh: Any):
    with jax.set_mesh(mesh):  # type: ignore[attr-defined]
        yield


def load_current_module():
    sys.modules.pop("pipeline", None)
    return importlib.import_module("pipeline")


def load_original_module(repo_root: Path, original_ref: str):
    source = subprocess.check_output(["git", "-c", f"safe.directory={repo_root}", "-C", str(repo_root), "show", f"{original_ref}:reactor_app/pipeline.py"], text=True)
    module = types.ModuleType(f"pipeline_original_{original_ref.replace('/', '_')}")
    module.__file__ = f"git:{original_ref}:reactor_app/pipeline.py"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def random_action(rng: np.random.Generator) -> Actions:
    binary = (rng.random(NUM_BINARY_ACTIONS) < 0.08).astype(np.int32)
    binary[23] = int(rng.random() < 0.03)
    binary[24] = int(rng.random() < 0.03)
    categorical = np.asarray([rng.integers(0, NUM_CAMERA_CLASSES)], dtype=np.int32)
    return Actions(binary=jnp.asarray(binary, dtype=jnp.int32)[None, :], categorical=jnp.asarray(categorical, dtype=jnp.int32), continuous=None)


def make_actions(module: Any, frames: int, seed: int, use_random_actions: bool) -> list[Actions]:
    if not use_random_actions:
        return [module._noop_action() for _ in range(frames)]
    rng = np.random.default_rng(seed)
    return [random_action(rng) for _ in range(frames)]


def benchmark_module(label: str, module: Any, config: dict[str, Any], frames: int, skip: int, seed: int, use_random_actions: bool) -> dict[str, Any]:
    pipeline = module.WorldModelPipeline.__new__(module.WorldModelPipeline)
    pipeline.state = module.WorldModelState(_keyboard={}, _mouse={"left": False, "right": False, "middle": False, "dx": 0.0, "dy": 0.0, "dwheel": 0.0}, _use_policy=True, _seed=123, _reset_requested=False)

    print(f"[{label}] loading")
    load_started = time.perf_counter()
    pipeline.load(config)
    load_seconds = time.perf_counter() - load_started
    print(f"[{label}] load_seconds={load_seconds:.3f}")

    rng = jax.random.PRNGKey(123)
    dynamics_cache = pipeline._empty_dynamics_cache
    tokenizer_cache = pipeline._empty_tokenizer_cache
    actions = make_actions(module, frames, seed, use_random_actions)
    frame_times = []

    with mesh_context(pipeline._mesh):
        for index in range(frames):
            frame_started = time.perf_counter()
            rng, key = jax.random.split(rng)
            frame_jax, _h, dynamics_cache, tokenizer_cache, rng = pipeline._next_frame_jit(pipeline._tokenizer, pipeline._dynamics, actions[index], pipeline._latent_shape, dynamics_cache, tokenizer_cache, key)
            jax.block_until_ready((frame_jax, dynamics_cache, tokenizer_cache, rng))
            frame = np.asarray(frame_jax[0, 0])
            frame_seconds = time.perf_counter() - frame_started
            frame_times.append(frame_seconds)
            print(f"[{label}] frame={index + 1} seconds={frame_seconds:.3f} shape={frame.shape} dtype={frame.dtype}")

    steady_times = frame_times[skip:] if skip < len(frame_times) else frame_times
    result = {
        "label": label,
        "load_seconds": load_seconds,
        "frames": frames,
        "skip": skip,
        "total_seconds": sum(frame_times),
        "fps_all": len(frame_times) / sum(frame_times),
        "steady_seconds": sum(steady_times),
        "steady_fps": len(steady_times) / sum(steady_times),
        "median_frame_seconds": float(np.median(np.asarray(steady_times))),
    }
    print(f"[{label}] total_seconds={result['total_seconds']:.3f}")
    print(f"[{label}] fps_all={result['fps_all']:.3f}")
    print(f"[{label}] steady_seconds={result['steady_seconds']:.3f}")
    print(f"[{label}] steady_fps={result['steady_fps']:.3f}")
    print(f"[{label}] median_frame_seconds={result['median_frame_seconds']:.3f}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Reactor Dreamer world-model next-frame FPS without starting MineRL.")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--tau-ctx-target", type=float, default=0.9)
    parser.add_argument("--world-model-warmup-steps", type=int, default=2)
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--original-ref", default="6d1dec0")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--target-fps", type=float, default=0.0)
    parser.add_argument("--target-label", default="current")
    parser.add_argument("--min-current-original-ratio", type=float, default=0.0)
    parser.add_argument("--require-accelerator", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    if args.skip < 0:
        raise ValueError("--skip must be >= 0")
    if args.target_fps < 0:
        raise ValueError("--target-fps must be >= 0")
    if args.min_current_original_ratio < 0:
        raise ValueError("--min-current-original-ratio must be >= 0")

    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "reactor_app"))

    config = {"fps": 1000, "num_steps": args.num_steps, "tau_ctx_target": args.tau_ctx_target, "warmup_env": False, "world_model_warmup_steps": args.world_model_warmup_steps}

    print(f"jax_version={jax.__version__}")
    backend = jax.default_backend()
    print(f"backend={backend}")
    print(f"devices={jax.devices()}")
    print(f"random_actions={str(args.random_actions).lower()}")
    print(f"config={json.dumps(config, sort_keys=True)}")
    if args.require_accelerator and backend == "cpu":
        print("target_pass=false")
        print("target_reason=backend is cpu")
        raise SystemExit(3)

    results = []
    if args.include_original:
        results.append(benchmark_module(f"original_{args.original_ref}", load_original_module(repo_root, args.original_ref), config, args.frames, args.skip, args.seed, args.random_actions))
    results.append(benchmark_module("current", load_current_module(), config, args.frames, args.skip, args.seed, args.random_actions))
    print("results_json=" + json.dumps(results, sort_keys=True))
    if args.min_current_original_ratio > 0:
        original = next((result for result in results if result["label"] == f"original_{args.original_ref}"), None)
        current = next((result for result in results if result["label"] == "current"), None)
        if original is None or current is None:
            print("ratio_pass=false")
            print("ratio_reason=missing original/current result")
            raise SystemExit(5)
        required_fps = original["steady_fps"] * args.min_current_original_ratio
        ratio = current["steady_fps"] / original["steady_fps"] if original["steady_fps"] > 0 else 0.0
        ratio_pass = current["steady_fps"] >= required_fps
        print(f"baseline_label=original_{args.original_ref}")
        print(f"baseline_fps={original['steady_fps']:.3f}")
        print(f"current_fps={current['steady_fps']:.3f}")
        print(f"min_current_original_ratio={args.min_current_original_ratio:.3f}")
        print(f"current_original_ratio={ratio:.3f}")
        print(f"ratio_pass={str(ratio_pass).lower()}")
        if not ratio_pass:
            raise SystemExit(6)
    if args.target_fps > 0:
        matching = [result for result in results if result["label"] == args.target_label]
        if not matching:
            print("target_pass=false")
            print(f"target_reason=missing label {args.target_label!r}")
            raise SystemExit(4)
        target_result = matching[-1]
        target_pass = target_result["steady_fps"] >= args.target_fps
        print(f"target_label={args.target_label}")
        print(f"target_fps={args.target_fps:.3f}")
        print(f"target_measured_fps={target_result['steady_fps']:.3f}")
        print(f"target_pass={str(target_pass).lower()}")
        if not target_pass:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
