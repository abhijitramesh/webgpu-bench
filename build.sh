#!/bin/bash
# Build llama.cpp -> WASM with WebGPU support
# Produces two variants: JSPI (Chrome) and Asyncify (Safari/Firefox)
# Only external dependency: Emscripten SDK (emsdk)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMSDK_DIR="${EMSDK_DIR:-$SCRIPT_DIR/../emsdk}"
BUILD_DIR="$SCRIPT_DIR/build"
LLAMA_CPP_DIR="$SCRIPT_DIR/llama.cpp"

# emdawnwebgpu version (matches llama.cpp CI)
DAWN_TAG="v20260317.182325"
EMDAWN_PKG="emdawnwebgpu_pkg-${DAWN_TAG}.zip"
EMDAWN_DIR="$BUILD_DIR/emdawnwebgpu_pkg"

echo "=== WebGPU Bench WASM Build ==="
echo "Emscripten: $EMSDK_DIR"
echo ""

# Ensure llama.cpp submodule is initialized at the recorded SHA. We pin to a
# PR-head commit (currently PR #22497, which adds 32-bit-WASM mmap support
# for >2GB models — required for loading large GGUFs in browser). PR-head
# refs aren't part of the default fetch refspec, so a vanilla
# `git submodule update --init` would fail to find the SHA. We fix this by
# (re-)cloning manually with `+refs/pull/*/head:refs/remotes/origin/pr/*`
# added so any PR commit on upstream is fetchable.
ensure_llama_cpp_submodule() {
    local expected_sha
    expected_sha=$(git -C "$SCRIPT_DIR" ls-tree HEAD llama.cpp 2>/dev/null | awk '{print $3}')
    if [ -z "$expected_sha" ]; then
        echo "ERROR: could not read submodule pointer for llama.cpp from parent repo"
        exit 1
    fi

    if [ ! -d "$LLAMA_CPP_DIR/.git" ] && [ ! -f "$LLAMA_CPP_DIR/.git" ]; then
        # Submodule directory missing or empty — clone fresh with PR refs.
        local submodule_url
        submodule_url=$(git -C "$SCRIPT_DIR" config --file .gitmodules --get submodule.llama.cpp.url)
        echo "=== llama.cpp submodule not initialized — cloning from $submodule_url ==="
        rm -rf "$LLAMA_CPP_DIR"
        git clone --no-checkout "$submodule_url" "$LLAMA_CPP_DIR"
        git -C "$LLAMA_CPP_DIR" config --add remote.origin.fetch '+refs/pull/*/head:refs/remotes/origin/pr/*'
        git -C "$LLAMA_CPP_DIR" fetch origin --quiet
    elif ! git -C "$LLAMA_CPP_DIR" cat-file -e "$expected_sha" 2>/dev/null; then
        # Clone exists but the recorded SHA isn't fetched (e.g. submodule
        # bumped to a PR-head). Add the PR-ref fetch refspec idempotently
        # and pull everything.
        echo "=== llama.cpp submodule pointer ($expected_sha) not in clone — fetching PR refs ==="
        git -C "$LLAMA_CPP_DIR" config --get-all remote.origin.fetch | grep -q 'refs/pull/' \
            || git -C "$LLAMA_CPP_DIR" config --add remote.origin.fetch '+refs/pull/*/head:refs/remotes/origin/pr/*'
        git -C "$LLAMA_CPP_DIR" fetch origin --quiet
    fi

    local current_sha
    current_sha=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD 2>/dev/null || echo "")
    if [ "$current_sha" != "$expected_sha" ]; then
        echo "=== Checking out llama.cpp at $expected_sha ==="
        git -C "$LLAMA_CPP_DIR" checkout --detach "$expected_sha"
    fi
}

ensure_llama_cpp_submodule
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

    # Stamp the llama.cpp commit + describe + dawn tag next to the wasm so the
    # browser can fetch it at runtime and attach it to every benchmark record.
    # Lets us compare performance across llama.cpp versions later.
    write_build_info "$build_dir/bin"
}

write_build_info() {
    local out_dir="$1"
    local llama_dir="$SCRIPT_DIR/llama.cpp"
    local llama_commit=""
    local llama_describe=""
    if [ -d "$llama_dir/.git" ] || [ -f "$llama_dir/.git" ]; then
        llama_commit="$(git -C "$llama_dir" rev-parse HEAD 2>/dev/null || echo '')"
        llama_describe="$(git -C "$llama_dir" describe --tags --always 2>/dev/null || echo '')"
    fi
    local built_at
    built_at="$(date -u +%FT%TZ)"
    cat > "$out_dir/build-info.json" <<EOF
{
  "llamaCppCommit": "${llama_commit}",
  "llamaCppDescribe": "${llama_describe}",
  "dawnTag": "${DAWN_TAG}",
  "builtAt": "${built_at}"
}
EOF
    echo "Wrote $out_dir/build-info.json (llama.cpp ${llama_describe:-unknown})"
}

# Build JSPI variant (for Chrome)
build_variant "jspi" "ON" "ON"

# Build Asyncify variant (for Safari/Firefox)
build_variant "asyncify" "OFF" "OFF"

echo ""
echo "=== All builds complete ==="
echo "  JSPI:     $BUILD_DIR/jspi/bin/bench.js + bench.wasm"
echo "  Asyncify: $BUILD_DIR/asyncify/bin/bench.js + bench.wasm"
