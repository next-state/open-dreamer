"""Offline generator for cached Minecraft world saves.

Runs MineRL to procedurally generate a Minecraft world, lets it settle for a
few frames, then copies the resulting world save folder into
``reactor_app/worlds/<name>/`` so it can be committed and shipped in the image.
At serve time ``pipeline.py`` loads these saves via MineRL's
``FileWorldGenerator`` (a world-file copy + load, seconds) instead of
regenerating a world from scratch.

This script must run where MineRL can launch Minecraft (JDK 8 + a software-GL
display; the app image provides both). It only renders Minecraft (no model
inference), so run it WITHOUT ``--gpus`` to use Mesa/llvmpipe software GL
cleanly (the NVIDIA GL stack crashes Xvfb's GLX). Example:

    docker run --rm \
        -e DISPLAY=:99 -e LIBGL_ALWAYS_SOFTWARE=1 \
        -v "$(pwd)":/workspace -w /workspace/reactor_app \
        --entrypoint bash reactor-local/reactor_app:dev -c \
        'Xvfb :99 -screen 0 1024x768x24 +extension GLX +render -nolisten tcp & \
         sleep 3; python generate_worlds.py --name plains --seed 1 --steps 60'

The world is fixed by ``--seed`` (the seed only matters during generation). At
serve time the pipeline loads the resulting save WITHOUT a seed, so the exact
saved terrain is loaded rather than regenerated.

Save discovery is heuristic (newest folder containing a fresh ``level.dat``
under the MineRL instance dirs). Pass ``--saves-dir`` to point at the exact
Minecraft ``saves/`` directory if auto-discovery picks the wrong world.

Note: the pinned MineRL always joins the agent in the overworld -- there is no
chat/teleport action and the agent ignores the saved player dimension -- so all
cached worlds are overworld scenes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _make_env(env_id: str, seed: int):
    import gym
    import minerl  # noqa: F401 - importing registers MineRL env IDs.

    env = gym.make(env_id)
    try:
        env.reset(seed=int(seed))
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(int(seed))
        env.reset()
    return env


def _settle(env, steps: int) -> None:
    """Step no-op actions so terrain/chunks around spawn are generated."""
    for _ in range(max(0, steps)):
        action = env.action_space.no_op()
        step_result = env.step(action)
        done = step_result[2] if len(step_result) == 4 else (step_result[2] or step_result[3])
        if done:
            break


def _candidate_saves_roots() -> list[Path]:
    """Best-effort list of directories that may contain Minecraft saves."""
    roots: list[Path] = []
    try:
        import minerl

        roots.append(Path(minerl.__file__).resolve().parent)
    except Exception:
        pass
    roots.extend([Path.cwd(), Path(os.environ.get("MINERL_DATA_ROOT", "/tmp")), Path("/tmp")])
    return [r for r in roots if r.exists()]


def _find_world_dir(after_ts: float, saves_dir: str | None) -> Path | None:
    """Locate the world save folder generated after ``after_ts``.

    A Minecraft world folder is a directory containing a ``level.dat``. Pick the
    one whose ``level.dat`` was modified most recently after the reset.
    """
    search_roots = [Path(saves_dir)] if saves_dir else _candidate_saves_roots()

    best: tuple[float, Path] | None = None
    for root in search_roots:
        for level_dat in root.rglob("level.dat"):
            try:
                mtime = level_dat.stat().st_mtime
            except OSError:
                continue
            if mtime < after_ts - 5.0:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, level_dat.parent)
    return best[1] if best else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="World name (folder under worlds/)")
    parser.add_argument("--seed", type=int, default=0, help="World generation seed")
    parser.add_argument("--env-id", default="MineRLBasaltFindCave-v0")
    parser.add_argument("--steps", type=int, default=60, help="No-op steps to settle terrain")
    parser.add_argument("--saves-dir", default=None, help="Explicit Minecraft saves/ dir")
    parser.add_argument("--out-dir", default=str(_HERE / "worlds"), help="Output worlds/ dir")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / args.name

    started = time.time()
    print(f"[generate_worlds] generating {args.name!r} (seed={args.seed}) via {args.env_id}")
    env = _make_env(args.env_id, args.seed)
    _settle(env, args.steps)

    # Close the env so Malmo flushes the integrated-server world to disk.
    try:
        env.close()
    except Exception as exc:  # noqa: BLE001 - close failures shouldn't abort discovery
        print(f"[generate_worlds] WARNING env.close() raised {exc!r}")

    time.sleep(2.0)

    world_dir = _find_world_dir(started, args.saves_dir)
    if world_dir is None:
        print(
            "[generate_worlds] ERROR could not locate a generated world save. "
            "Pass --saves-dir pointing at the Minecraft saves/ directory.",
            file=sys.stderr,
        )
        return 1
    print(f"[generate_worlds] found world save at {world_dir}")

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(world_dir, target)
    print(f"[generate_worlds] copied -> {target}")
    print(f"[generate_worlds] done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
