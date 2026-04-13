#!/bin/bash
# Build llama.cpp -> WASM with WebGPU support
# Produces two variants: JSPI (Chrome) and Asyncify (Safari/Firefox)
# Only external dependency: Emscripten SDK (emsdk)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMSDK_DIR="${EMSDK_DIR:-$SCRIPT_DIR/../emsdk}"
BUILD_DIR="$SCRIPT_DIR/build"

# emdawnwebgpu version (matches llama.cpp CI)
DAWN_TAG="v20260317.182325"
EMDAWN_PKG="emdawnwebgpu_pkg-${DAWN_TAG}.zip"
EMDAWN_DIR="$BUILD_DIR/emdawnwebgpu_pkg"

echo "=== WebGPU Bench WASM Build ==="
echo "Emscripten: $EMSDK_DIR"
echo ""

# Install emsdk if not present
if [ ! -f "$EMSDK_DIR/emsdk_env.sh" ]; then
    echo "=== emsdk not found, fetching latest ==="
    if [ -d "$EMSDK_DIR" ]; then
        # Directory exists but is incomplete — update it
        git -C "$EMSDK_DIR" pull
    else
        git clone https://github.com/emscripten-core/emsdk.git "$EMSDK_DIR"
    fi
    "$EMSDK_DIR/emsdk" install latest
    "$EMSDK_DIR/emsdk" activate latest
    echo "=== emsdk setup complete ==="
    echo ""
fi

# Activate Emscripten
source "$EMSDK_DIR/emsdk_env.sh"

# Get CPU count for parallel build
if command -v nproc &>/dev/null; then
    NCPU=$(nproc)
elif sysctl -n hw.ncpu &>/dev/null 2>&1; then
    NCPU=$(sysctl -n hw.ncpu)
else
    NCPU=4
fi

# Download emdawnwebgpu if not present
mkdir -p "$BUILD_DIR"
if [ ! -d "$EMDAWN_DIR" ]; then
    echo "=== Downloading emdawnwebgpu ${DAWN_TAG} ==="
    curl -L -o "$BUILD_DIR/emdawn.zip" \
        "https://github.com/google/dawn/releases/download/${DAWN_TAG}/${EMDAWN_PKG}"
    unzip -o "$BUILD_DIR/emdawn.zip" -d "$BUILD_DIR"
    rm "$BUILD_DIR/emdawn.zip"
    echo "emdawnwebgpu downloaded to $EMDAWN_DIR"
fi

# Common cmake flags
COMMON_FLAGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_WEBGPU=ON
    -DLLAMA_OPENSSL=OFF
    -DEMDAWNWEBGPU_DIR="$EMDAWN_DIR"
)

build_variant() {
    local variant_name="$1"
    local use_jspi="$2"
    local jspi_cmake="$3"
    local build_dir="$BUILD_DIR/$variant_name"

    echo ""
    echo "=== Building $variant_name variant ==="

    emcmake cmake -B "$build_dir" "$SCRIPT_DIR" \
        "${COMMON_FLAGS[@]}" \
        -DBENCH_USE_JSPI="$use_jspi" \
        -DGGML_WEBGPU_JSPI="$jspi_cmake"

    cmake --build "$build_dir" --target bench -j "$NCPU"

    if [ -f "$build_dir/bin/bench.js" ] && [ -f "$build_dir/bin/bench.wasm" ]; then
        ls -lh "$build_dir/bin/bench.js" "$build_dir/bin/bench.wasm"
        echo "$variant_name build OK"
    else
        echo "ERROR: $variant_name build failed - output files not found!"
        exit 1
    fi
}

# Build JSPI variant (for Chrome)
build_variant "jspi" "ON" "ON"

# Build Asyncify variant (for Safari/Firefox)
build_variant "asyncify" "OFF" "OFF"

echo ""
echo "=== All builds complete ==="
echo "  JSPI:     $BUILD_DIR/jspi/bin/bench.js + bench.wasm"
echo "  Asyncify: $BUILD_DIR/asyncify/bin/bench.js + bench.wasm"
