# syntax=docker/dockerfile:1
# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
# check=skip=FromPlatformFlagConstDisallowed
#
# Hybrid MineRL + Dreamer world-model Reactor image.
#
# Built on the reactor-models base image (`reactor-runtime-base`, which ships
# GStreamer + a pre-baked venv at /opt/venv with the runtime pre-installed).
# The runtime tag is threaded as a build arg (`RUNTIME_VERSION`) so it tracks
# the reactor-models release the rest of the platform runs.
#
# NOTE: unlike the torch-based reactor-models images, this image does NOT
# overlay onto an `nvidia/cuda` base. MineRL/Malmo (pinned commit) compiles a
# Minecraft/Forge runtime that requires JDK 8, which is not available on the
# ubuntu24.04 CUDA bases; and JAX ships its own CUDA via the `jax[cuda12]` pip
# wheels, so system CUDA is unnecessary. Keeping the runtime-base OS preserves
# a working JDK 8 for MineRL.
#
# Build from the repository root (the app imports the parent `dreamer`
# package). `reactor_app/build.sh` wires the context and args:
#   reactor_app/build.sh --progress=plain

ARG RUNTIME_VERSION=2.7.1-0

FROM --platform=linux/amd64 reactortechnologies/reactor-runtime-base:${RUNTIME_VERSION}

WORKDIR /app

ENV JAVA_HOME=/opt/java
ENV PATH=/opt/java/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV UV_CACHE_DIR=/tmp/uv-cache

# Minecraft (LWJGL/GLFW) needs an OpenGL context. Under Xvfb there is no GPU
# display, so force Mesa/llvmpipe software GL. On a GPU host (JAX runs with
# --gpus all) the NVIDIA EGL vendor otherwise crashes Xvfb's GLX init, so mask
# GLVND to the Mesa EGL vendor only. JAX uses CUDA (not EGL/GL), so this does
# not affect model compute.
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV __GLX_VENDOR_LIBRARY_NAME=mesa
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

# --- MineRL system/runtime dependencies ---------------------------------
# MineRL builds and launches a Minecraft/Malmo runtime. Its installer needs a
# JDK (8 preferred for the Forge build), the runtime needs a display (Xvfb),
# and Minecraft needs a GL driver (Mesa DRI provides the llvmpipe software
# rasteriser).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git patch xauth xvfb build-essential \
        libgl1-mesa-dri libglu1-mesa mesa-utils \
    && (apt-get install -y --no-install-recommends openjdk-8-jdk \
        || apt-get install -y --no-install-recommends openjdk-11-jdk \
        || apt-get install -y --no-install-recommends openjdk-17-jdk \
        || apt-get install -y --no-install-recommends default-jdk) \
    && detected_java_home="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")" \
    && ln -s "$detected_java_home" "$JAVA_HOME" \
    && java -version \
    && javac -version \
    && rm -rf /var/lib/apt/lists/*

# --- App install ---------------------------------------------------------
# Install Reactor app deps into the runtime-base venv without pruning the
# pre-baked Reactor/GStreamer packages that ship with the base image.
COPY pyproject.toml README.md /app/
COPY dreamer /app/dreamer
COPY reactor_app/pyproject.toml /app/reactor_app/pyproject.toml
RUN pip install --no-cache-dir uv \
    && uv pip install --python /opt/venv/bin/python --no-cache \
        --requirements /app/reactor_app/pyproject.toml \
    && uv pip install --python /opt/venv/bin/python --no-cache --no-deps \
        --editable /app \
    && uv pip install --python /opt/venv/bin/python --no-cache --no-deps \
        "minerl @ git+https://github.com/minerllabs/minerl@cdeae668c2f334e3c9117adf651b5a94436b45f8"

# --- Reactor workspace ---------------------------------------------------
# Includes pipeline.py, reactor.yaml, config.yaml, and the committed cached
# world saves under reactor_app/worlds/.
COPY reactor_app /app/reactor_app

# IMAGE_VERSION is surfaced in the runtime's machine metrics / traces.
ARG IMAGE_VERSION
ENV IMAGE_VERSION=${IMAGE_VERSION}

# reactor.yaml is in the workspace subdir, so cd there before serving.
WORKDIR /app/reactor_app

# --- Entrypoint ----------------------------------------------------------
# The Xvfb screen must advertise GLX so Minecraft can create its GL context.
ENTRYPOINT ["/usr/bin/tini", "--", "xvfb-run", "-a", "-s", "-screen 0 1024x768x24 +extension GLX +render -nolisten tcp", "python", "-m", "reactor_runtime.serve"]
CMD ["run"]
