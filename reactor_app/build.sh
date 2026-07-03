#!/usr/bin/env bash
# Build the workspace image with the project root as Docker context so the
# Dockerfile can include `dreamer/` and `pyproject.toml`. Tag matches what
# `reactor run` resolves to (`reactor-local/<workspace-dir>:dev`), so a
# subsequent `reactor run` from inside reactor_app/ picks this image up.
#
# RUNTIME_VERSION selects the reactor-runtime-base tag (override to track a
# different reactor-models release). IMAGE_VERSION is surfaced in runtime
# metrics/traces.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

RUNTIME_VERSION="${RUNTIME_VERSION:-2.7.1-0}"
IMAGE_VERSION="${IMAGE_VERSION:-dev}"

exec docker build \
    -f "$HERE/Dockerfile" \
    --build-arg "RUNTIME_VERSION=$RUNTIME_VERSION" \
    --build-arg "IMAGE_VERSION=$IMAGE_VERSION" \
    -t "reactor-local/$(basename "$HERE"):dev" \
    "$@" "$ROOT"
