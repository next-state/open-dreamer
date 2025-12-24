# Running Reactor Models

## Procgen CoinRun Test

To test the Reactor integration with the actual CoinRun environment:

```bash
# Copy the procgen manifest
cp manifest-procgen.json manifest.json

# Run the reactor server in debug mode
uv run reactor run --debug --log-level DEBUG

# Or run in deployment mode
uv run reactor run --deploy --port 8081
```

The server will:
1. Load `ProcgenVideoModel` from `dreamer/procgen_reactor.py`
2. Start a FastAPI server on port 8081 (default)
3. Wait for client connections

### Client Connection

Connect using the Reactor client SDK:

```typescript
const reactor = new Reactor({
  insecureApiKey: process.env.REACTOR_API_KEY,
  modelName: "procgen-coinrun",
});

await reactor.connect();

// Send keyboard input
reactor.sendMessage({
  type: "control",
  command: "send_keyboard_state",
  args: { w: true, a: false, s: false, d: false, q: false, e: false }
});
```

### Available Commands

- `send_keyboard_state(w, a, s, d, q, e)` - Send keyboard state (WASD+QE keys)
- `reset_env()` - Reset environment to a new level

## Dreamer Model

To run the full Dreamer model instead:

```bash
# Use the main manifest (update it first with correct checkpoint paths)
uv run reactor run --debug --log-level DEBUG
```

Make sure to update `manifest.json` with:
- `"class": "dreamer.reactor:DreamerVideoModel"`
- Add checkpoint paths to `args` or `weights`

## Troubleshooting

### Port already in use
```bash
# Use a different port
uv run reactor run --debug --port 8082
```

### Model not loading
Check that:
1. The class path in manifest.json is correct
2. All dependencies are installed (`uv sync`)
3. Log level is DEBUG to see detailed errors

### Video not streaming
- Check that `emit_frame()` is being called in the game loop
- Verify frame shape is (H, W, 3) uint8
- Check browser console for client-side errors
