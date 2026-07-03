# reactor_app

A Reactor Runtime app that starts in MineRL and can switch into a Dreamer
world-model rollout after the model cache has been warmed from real frames.

## Run it locally

MineRL builds and launches Minecraft/Malmo, so the serving environment needs a
JDK before `minerl` is installed. On headless Linux, run under Xvfb or use the
included Dockerfile, which installs Java/Xvfb and starts Reactor with
`xvfb-run`.

```bash
cd /home/ubuntu/dreamer4-jax-private
reactor_app/build.sh --progress=plain
docker run --rm --network host \
  -e WEBRTC_PORT_RANGE="40000:40050" \
  -v /home/ubuntu/dreamer4-jax-private/jelly:/app/jelly:ro \
  reactor-local/reactor_app:dev \
  run --host 0.0.0.0 --port 8080 --webrtc-port-range 40000:40050
```

The runtime listens on `http://localhost:8080`. Host networking is intentional:
WebRTC ICE candidates need to advertise host-reachable addresses and UDP ports.
The local frontend connects to the `world-model` Reactor model name.

The image installs the Reactor app dependencies from `reactor_app/pyproject.toml`
with `uv`, including CUDA JAX. On a host with the NVIDIA container runtime,
build and run with GPU access:

```bash
reactor_app/build.sh --progress=plain
docker run --rm --gpus all --network host \
  -e WEBRTC_PORT_RANGE="40000:40050" \
  -v /home/ubuntu/dreamer4-jax-private/jelly:/app/jelly:ro \
  reactor-local/reactor_app:dev \
  run --host 0.0.0.0 --port 8080 --webrtc-port-range 40000:40050
```

Do not set `JAX_PLATFORMS=cpu` when measuring GPU performance. Startup logs
print the JAX backend and visible devices.

## Benchmarking FPS

Use the random-action pipeline benchmark for the app path. It loads the real
Dreamer checkpoint, uses a fake fast MineRL env, observes real frames into the
cache synchronously, switches into world-model mode, sends random
keyboard/mouse actions, and measures the foreground
`WorldModelPipeline.inference()` generator.

```bash
docker run --rm -i \
  -e JAX_PLATFORMS=cpu \
  -e REACTOR_WEIGHTS_PATH=/workspace/jelly \
  -v /home/ubuntu/dreamer4-jax-private:/workspace:ro \
  -w /workspace/reactor_app \
  --entrypoint /usr/bin/tini \
  reactor-local/reactor_app:dev \
  -- python -u benchmark_pipeline_random_actions.py \
    --frames 40 \
    --warmup-frames 3 \
    --fps 20
```

Add `--target-world-fps` and/or `--target-env-cache-fps` when you want the
benchmark to act as a gate.

Use the raw world-model benchmark only when you want to measure the exact
`next_frame` compute path and compare against the committed original pipeline.
Mount the repo when comparing against the committed original pipeline, because
the script reads the old file through `git show`.

```bash
docker run --rm -i \
  -e JAX_PLATFORMS=cpu \
  -e REACTOR_WEIGHTS_PATH=/workspace/jelly \
  -v /home/ubuntu/dreamer4-jax-private:/workspace:ro \
  -w /workspace/reactor_app \
  --entrypoint /usr/bin/tini \
  reactor-local/reactor_app:dev \
  -- python -u benchmark_world_model_fps.py \
    --include-original \
    --original-ref 6d1dec0 \
    --frames 20 \
    --skip 2 \
    --repo-root /workspace
```

For a GPU run, run with `--gpus all` and remove `JAX_PLATFORMS=cpu`:

```bash
reactor_app/build.sh --progress=plain
docker run --rm -i --gpus all \
  -e REACTOR_WEIGHTS_PATH=/workspace/jelly \
  -v /home/ubuntu/dreamer4-jax-private:/workspace:ro \
  -w /workspace/reactor_app \
  --entrypoint /usr/bin/tini \
  reactor-local/reactor_app:dev \
  -- python -u benchmark_world_model_fps.py \
    --include-original \
    --original-ref 6d1dec0 \
    --frames 20 \
    --skip 2 \
    --repo-root /workspace \
    --require-accelerator
```

## Project structure

| File | Purpose |
|------|---------|
| `pipeline.py` | Hybrid **ReactorPipeline**: MineRL env, Dreamer cache warmup, cached worlds, and world-model rollout |
| `pipeline_minerl.py` | Standalone MineRL-only pipeline kept for debugging |
| `generate_worlds.py` | Offline generator for the cached Minecraft world saves under `worlds/` |
| `worlds/` | Committed pre-generated Minecraft world saves, selected at runtime by index |
| `benchmark_pipeline_random_actions.py` | Random-action foreground FPS gate for `WorldModelPipeline.inference()` |
| `benchmark_world_model_fps.py` | Raw Dreamer `next_frame` FPS diagnostic and original/current comparison |
| `reactor.yaml` | Combined model registration spec (`model:`) and runtime entry point (`runtime:`) |
| `config.yaml` | Model hyperparameters (passed to `load()` as a dict) |
| `pyproject.toml` | Reactor app dependency spec installed by `uv` inside the image |
| `Dockerfile` | Workspace image (`reactor-runtime-base` + Reactor app deps + your code) |
| `.dockerignore` | Files excluded from the image build context |

The runtime itself ships pre-installed in the `reactor-runtime-base` image.
The Dockerfile installs app dependencies into that existing venv with
`uv pip install`, without pruning the base image's Reactor/GStreamer packages.

## Iterating

Rebuild after editing `pipeline.py`, `reactor_app/pyproject.toml`, or Docker-baked files:

```bash
reactor_app/build.sh --progress=plain
```

For a direct host run, use a MineRL-compatible Python environment after Java is
available and MineRL is installed. Do not install MineRL into the root JAX
environment unless you are okay with its old NumPy/Gym pins replacing the JAX
stack's versions. The hybrid pipeline also needs the Dreamer checkpoint
configured by `reactor.yaml` (`weights_path: ../jelly`). Then launch:

```bash
WEBRTC_PORT_RANGE="40000:40050" uv run python -m reactor_runtime.serve run --path reactor_app
```

## How it works

- **`WorldModelOutput`** declares the video track sent to clients.
- **`WorldModelState`** stores keyboard, mouse, seed, reset, and mode state.
- **`inference()`** starts in MineRL, observes each real frame into the Dreamer
  dynamics/decoder caches synchronously, then uses `switch_to_policy` to
  continue from the world model with the warmed caches.
- **`new_scene`** resets MineRL, clears caches, and returns to MineRL mode.

## Cached worlds

Generating a brand-new random Minecraft world is slow: MineRL's
`DefaultWorldGenerator(force_reset=True)` regenerates the whole world on every
reset. To make scene switches fast, the app ships **pre-generated world saves**
and loads them via MineRL's `FileWorldGenerator` (a world-file copy + load,
seconds).

Worlds are listed, in order, in `config.yaml`:

```yaml
worlds:
  - name: plains
    path: worlds/plains
  - name: nether
    path: worlds/nether
```

The **index** into this list is the switch handle. On connect the model sends a
`worlds_available` message (`{ "worlds": ["plains", "nether"] }`) so the client
can build a picker, and the client switches worlds with the `load_world` event:

```jsonc
{ "type": "load_world", "data": { "index": 1 } }   // -> nether
```

`load_world` clears the Dreamer KV caches and returns to real-MineRL mode with
the chosen world loaded; the frontend then toggles `switch_to_policy` to hand
off to the world model as usual. `new_scene` still generates a fresh procedural
world (index `-1`). Missing world folders are skipped at load time, so the model
keeps serving procedural worlds when a save has not been shipped yet.

### How the load works (important)

MineRL only loads a saved world when the mission carries **no seed** — a seed in
the mission token makes Malmo procedurally regenerate the world and silently
ignore the `FileWorldGenerator`. The pipeline therefore resets **without a
seed** when a cached world is active (and with a seed only for procedural
`new_scene` worlds). The cached spec also drops `PreferredSpawnBiome` when
loading a save, because a biome search would relocate the agent and regenerate
chunks. The env is created once and switches worlds by re-resetting the same
instance.

### What a cached world controls

Everything the Minecraft save file carries: terrain, structures, spawn point,
time of day, weather, and inventory. The world is fixed by the generation seed
and then frozen into the save.

Nether (and other non-overworld) starts are **not** supported at the pinned
MineRL commit: there is no chat/teleport action, and the Malmo agent always
joins in the overworld regardless of the saved player dimension. All cached
worlds are overworld scenes. A true nether start would need a newer MineRL
(with `ChatAction`) or a custom Malmo dimension handler.

### Regenerating worlds

`generate_worlds.py` runs MineRL, generates a world from a seed, and copies the
save into `worlds/<name>/`. It only renders Minecraft (no model inference), so
run it **without `--gpus`** (the NVIDIA GL stack crashes Xvfb's GLX; Mesa
software GL is used instead):

```bash
reactor_app/build.sh --progress=plain
docker run --rm \
  -e DISPLAY=:99 -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v /home/ubuntu/davide/dreamer4-jax-private:/workspace \
  -w /workspace/reactor_app \
  --entrypoint bash reactor-local/reactor_app:dev -c \
  'Xvfb :99 -screen 0 1024x768x24 +extension GLX +render -nolisten tcp & \
   sleep 3; python generate_worlds.py --name plains --seed 1 --steps 60'
```

Commit the resulting `worlds/<name>/` folders (a few MB each) and add them to
`config.yaml`; the Dockerfile bakes them into the image via `COPY reactor_app`.

## Next steps

- Add `Input` tracks to receive webcam video from the client
- Add `@event` handlers for custom logic (e.g. encoding a prompt)
- Add `ModelMessage` subclasses to send structured data back to the client
- Register and publish the model: `reactor model register --model-file reactor.yaml`
