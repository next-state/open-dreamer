# reactor_app

A real-time video model built with Reactor Runtime.

## Run it locally

```bash
reactor run
```

`reactor run` builds the workspace's `Dockerfile` (if no cached image
exists) and starts the container locally. The runtime listens on
`http://localhost:8080`; connect with a WebRTC client at the URL printed
on startup. Pass `--rebuild` to force a fresh `docker build`, or
`--port 9000` to map the runtime onto a different host port.

To forward flags to the runtime entrypoint inside the container, append
them after the model — `reactor run` passes them through verbatim:

```bash
reactor run --runtime _redis --model.batch_size=8
```

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

`reactor run` reuses cached images by default, so a re-run after editing
`pipeline.py` will skip the build step. Force a rebuild after changing
`requirements.txt` or anything else baked into the image:

```bash
reactor run --rebuild
```

## How it works

- **`MyOutput`** declares the video track the model sends to clients.
- **`MyState`** declares parameters clients can change in real-time. Each field auto-generates a `set_<field>` event — no handler code needed.
- **`inference()`** is a generator that yields frames in batches. Read `self.state` to pick up the latest client values.
- **`load()`** runs once at startup — put weight loading here.

## Next steps

- Add `Input` tracks to receive webcam video from the client
- Add `@event` handlers for custom logic (e.g. encoding a prompt)
- Add `ModelMessage` subclasses to send structured data back to the client
- Register and publish the model: `reactor model register --model-file reactor.yaml`
