#!/usr/bin/env bash
set -euo pipefail

# One-shot macOS setup for local development.
# - Keeps repo `pyproject.toml` unchanged (GPU-VM oriented deps).
# - Creates/refreshes `.venv` with Python 3.11 (required by this repo).
# - Installs this repo editable WITHOUT pulling pyproject deps (CUDA/procgen/etc.).
# - Installs a minimal dependency set sufficient to exercise the CoinRun dataloader
#   path used by `dreamer.data.make_iterator(... source="custom" ...)`.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYVER="${PYVER:-3.11}"

echo "[setup-mac] Ensuring Python ${PYVER} is available via uv..."
uv python install "${PYVER}" >/dev/null

echo "[setup-mac] Creating/refreshing venv at ${ROOT}/.venv ..."
UV_VENV_CLEAR=1 uv venv --python "${PYVER}" --clear .venv >/dev/null

PY="${ROOT}/.venv/bin/python"

echo "[setup-mac] Installing this repo editable (no deps)..."
uv pip install --python "$PY" -e . --no-deps

echo "[setup-mac] Installing minimal runtime deps for dataloader smoke-tests..."
uv pip install --python "$PY" \
  "jax" \
  "numpy" \
  "einops" \
  "imageio[ffmpeg]" \
  "array-record>=0.8.3" \
  "grain>=0.2.15"

echo
echo "[setup-mac] Done."
echo "[setup-mac] Activate with: source .venv/bin/activate"

