#!/bin/bash
# Idempotent bootstrap for a SkyPilot RunPod pod.
# Fast-path: if the WASM artifacts already exist, skip the whole install/build.
# This handles pod resume (stop → launch) cheaply — SkyPilot runs `setup:` at
# cluster creation only, but we keep this guard as belt-and-suspenders.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f build/jspi/bin/bench.wasm ] && [ -f build/asyncify/bin/bench.wasm ]; then
    echo "=== setup.sh: WASM artifacts already present, skipping ==="
    exit 0
fi

echo "=== setup.sh: installing system packages ==="
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y \
    xvfb \
    ninja-build \
    cmake \
    build-essential \
    git \
    curl \
    unzip \
    ca-certificates

echo "=== setup.sh: installing Node.js 20 ==="
if ! command -v node >/dev/null 2>&1 || [ "$(node -p 'parseInt(process.versions.node)')" -lt 20 ]; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
node --version
npm --version

echo "=== setup.sh: npm ci ==="
npm ci

echo "=== setup.sh: installing Playwright browsers ==="
npx playwright install --with-deps chromium

# llama.cpp is excluded from workdir sync (.skyignore) to cut upload size.
# Clone it fresh on the pod, at the exact commit the operator passed via
# LLAMA_CPP_COMMIT so cloud builds match the laptop.
if [ ! -d llama.cpp/.git ]; then
    echo "=== setup.sh: cloning llama.cpp ==="
    rm -rf llama.cpp
    git clone https://github.com/ggerganov/llama.cpp.git llama.cpp
fi
if [ -n "${LLAMA_CPP_COMMIT:-}" ]; then
    echo "=== setup.sh: checking out llama.cpp @ $LLAMA_CPP_COMMIT ==="
    git -C llama.cpp fetch origin "$LLAMA_CPP_COMMIT" 2>/dev/null || git -C llama.cpp fetch origin
    git -C llama.cpp checkout "$LLAMA_CPP_COMMIT"
fi

echo "=== setup.sh: building WASM (build.sh handles emsdk + emdawnwebgpu) ==="
bash build.sh

echo "=== setup.sh: done ==="
