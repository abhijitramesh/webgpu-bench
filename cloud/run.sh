#!/bin/bash
# Entry point for a SkyPilot RunPod pod. Runs the benchmark under xvfb so headed
# browsers (runner.js keeps `headless: false`) get a virtual display. Headless
# mode is intentionally avoided — WebGPU adapter selection differs between
# headed and headless Chromium, so staying headed keeps cloud results
# comparable to the laptop runs.

set -euo pipefail

cd "$(dirname "$0")/.."

: "${BENCH_FLAGS:=--quick --browsers=chromium --resume}"

echo "=== run.sh ==="
echo "MACHINE_SLUG=${MACHINE_SLUG:-<unset>}"
echo "LLAMA_CPP_COMMIT=${LLAMA_CPP_COMMIT:-<unset>}"
echo "BENCH_FLAGS=${BENCH_FLAGS}"
echo ""

exec xvfb-run -a --server-args="-screen 0 1920x1080x24" \
    node runner.js ${BENCH_FLAGS}
