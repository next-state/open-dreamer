# reactor_app

A real-time video model built with Reactor Runtime.

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
  reactor-local/reactor_app:dev \
  run --host 0.0.0.0 --port 8080 --webrtc-port-range 40000:40050
```

The runtime listens on `http://localhost:8080`. Host networking is intentional:
WebRTC ICE candidates need to advertise host-reachable addresses and UDP ports.

## Project structure

| File | Purpose |
|------|---------|
| `pipeline.py` | **ReactorPipeline** — your model: tracks, state, and the `inference()` generator |
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
available. Do not install MineRL into the root JAX environment unless you are
okay with its old NumPy/Gym pins replacing the JAX stack's versions. Then
launch:

```bash
WEBRTC_PORT_RANGE="40000:40050" python -m reactor_runtime.serve run --path reactor_app
```

## How it works

- **`MineRLOutput`** declares the video track sent to clients.
- **`MineRLState`** stores per-client keyboard, mouse, seed, and reset state.
- **`inference()`** owns a MineRL Gym environment, steps it with live input, and yields `obs["pov"]` frames.
- **`load()`** reads the env id, frame rate, and camera tuning from `config.yaml`.

## Next steps

- Add `Input` tracks to receive webcam video from the client
- Add `@event` handlers for custom logic (e.g. encoding a prompt)
- Add `ModelMessage` subclasses to send structured data back to the client
- Register and publish the model: `reactor model register --model-file reactor.yaml`
