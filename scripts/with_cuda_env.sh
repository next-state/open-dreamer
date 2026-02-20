#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing venv at ${VENV_DIR}. Run 'uv sync' first." >&2
  exit 1
fi

# Prefer CUDA libraries installed via nvidia-* wheels inside the venv.
CUDA_LIB_DIRS="$(find "${VENV_DIR}/lib" -type d -path '*/site-packages/nvidia/*/lib' | sort | paste -sd: -)"
if [[ -z "${CUDA_LIB_DIRS}" ]]; then
  echo "No CUDA wheel libraries found under ${VENV_DIR}/lib/.../site-packages/nvidia/*/lib" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 2
fi

export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$@"
