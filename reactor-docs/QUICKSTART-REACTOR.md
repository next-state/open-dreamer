# Quick Start: Running with Reactor

## Step 1: Test Locally (Optional but Recommended)

Before running with Reactor, verify the environment works:

```bash
uv run python test_procgen_local.py
```

You should see:
```
INFO - Test completed successfully!
```

## Step 2: Set Up Manifest

Copy the Procgen manifest:

```bash
cp manifest-procgen.json manifest.json
```

The manifest tells Reactor which class to load:
```json
{
  "class": "dreamer.procgen_reactor:ProcgenVideoModel",
  "model_name": "procgen-coinrun",
  "args": {
    "fps": 15,
    "size": [64, 64]
  }
}
```

## Step 3: Start the Reactor Server

Run in debug mode (recommended for testing):

```bash
uv run reactor run --debug --log-level DEBUG
```

Or run in deployment mode:

```bash
uv run reactor run --deploy --port 8081
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8081
```

## Step 4: Connect a Client

The Reactor server is now running and waiting for client connections. You can connect using:

### Option A: Reactor Web Client
Navigate to the Reactor dashboard and connect to your running instance.

### Option B: Custom Client (TypeScript/JavaScript)
```typescript
import { Reactor } from '@reactor/client';

const reactor = new Reactor({
  insecureApiKey: process.env.REACTOR_API_KEY,
  modelName: "procgen-coinrun",
});

await reactor.connect();

// Send keyboard input
reactor.sendMessage({
  type: "control",
  command: "send_keyboard_state",
  args: { w: true }  // Jump
});
```

## Available Commands

### `send_keyboard_state(w, a, s, d, q, e)`
Control the CoinRun character:
- `w`: Jump/Up
- `a`: Left
- `s`: Down
- `d`: Right
- `q`: Left-Jump
- `e`: Right-Jump

### `reset_env()`
Reset to a new procedurally generated level.

## Troubleshooting

### "Module not found" error
```bash
uv sync  # Reinstall dependencies
```

### Port already in use
```bash
uv run reactor run --debug --port 8082
```

### No video stream
- Check that the server is running (`uv run reactor run --debug`)
- Check logs for errors (`--log-level DEBUG`)
- Verify frame emission in logs: "Emitting frame..."

## Next Steps: Running Dreamer Model

Once Procgen works, switch to the Dreamer model:

1. Update `manifest.json`:
   ```json
   {
     "class": "dreamer.reactor:DreamerVideoModel",
     "args": {
       "dynamics_ckpt": "/path/to/checkpoint"
     }
   }
   ```

2. Run the same way:
   ```bash
   uv run reactor run --debug
   ```

The workflow is identical - just the model changes!
