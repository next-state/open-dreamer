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

The default image installs CPU JAX. On this machine the committed original
world-model loop measured `0.124 FPS` steady-state on CPU, and the current
hybrid loop measured `0.126 FPS` for the same direct world-model path. To test
GPU throughput on a host with the NVIDIA container runtime, build and run with
CUDA JAX instead:

```bash
reactor_app/build.sh --progress=plain --build-arg 'JAX_PACKAGE=jax[cuda12]>=0.4.38,<0.5'
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

For a GPU run, rebuild with CUDA JAX, run with `--gpus all`, and remove
`JAX_PLATFORMS=cpu`:

```bash
reactor_app/build.sh --progress=plain --build-arg 'JAX_PACKAGE=jax[cuda12]>=0.4.38,<0.5'
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
| `pipeline.py` | Hybrid **ReactorPipeline**: MineRL env, Dreamer cache warmup, and world-model rollout |
| `pipeline_minerl.py` | Standalone MineRL-only pipeline kept for debugging |
| `benchmark_pipeline_random_actions.py` | Random-action foreground FPS gate for `WorldModelPipeline.inference()` |
| `benchmark_world_model_fps.py` | Raw Dreamer `next_frame` FPS diagnostic and original/current comparison |
| `reactor.yaml` | Combined model registration spec (`model:`) and runtime entry point (`runtime:`) |
| `config.yaml` | Model hyperparameters (passed to `load()` as a dict) |
| `requirements.txt` | Extra Python dependencies installed on top of the runtime base image |
| `Dockerfile` | Workspace image (`reactor-runtime-base` + your `requirements.txt` + your code) |
| `.dockerignore` | Files excluded from the image build context |

The runtime itself ships pre-installed in the `reactor-runtime-base`
image — `requirements.txt` is just for any extra deps your model needs
on top.

## Iterating

Rebuild after editing `pipeline.py`, `requirements.txt`, or Docker-baked files:

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

## Next steps

- Add `Input` tracks to receive webcam video from the client
- Add `@event` handlers for custom logic (e.g. encoding a prompt)
- Add `ModelMessage` subclasses to send structured data back to the client
- Register and publish the model: `reactor model register --model-file reactor.yaml`
