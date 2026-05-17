#!/usr/bin/env bash
# Build the workspace image with the project root as Docker context so the
# Dockerfile can include `dreamer/` and `pyproject.toml`. Tag matches what
# `reactor run` resolves to (`reactor-local/<workspace-dir>:dev`), so a
# subsequent `reactor run` from inside reactor_app/ picks this image up.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
exec docker build -f "$HERE/Dockerfile" -t "reactor-local/$(basename "$HERE"):dev" "$ROOT" "$@"
