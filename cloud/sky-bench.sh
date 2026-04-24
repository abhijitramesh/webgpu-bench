#!/bin/bash
# cloud/sky-bench.sh — orchestrate the CPU+GPU pod split for one target id.
#
# Short ids: 4090 | 5090 | mi300x
#
# Verbs:
#   launch-cpu <id> [--consistency]   launch the CPU-only pod
#   launch-gpu <id> [--consistency]   launch the GPU pod (auto-syncs baselines in consistency mode)
#   launch     <id> [--consistency]   launch both (CPU first so baselines exist)
#   sync-baselines <id>               rsync cpu_baselines.json: CPU pod → GPU pod
#   fetch      <id> [--consistency]   rsync results from both pods, merge into results-cloud/<id>/results.json
#   stop       <id>                   sky stop both pods (pauses; resume skips setup)
#   down       <id>                   sky down both pods (destroys)
#   status     <id>                   sky status for both pods
#
# Example end-to-end on RTX 4090:
#   ./cloud/sky-bench.sh launch 4090
#   ./cloud/sky-bench.sh fetch  4090
#   cp results-cloud/4090/results.json results/results.json
#   HF_TOKEN=… HF_DATASET_REPO=owner/webgpu-bench-leaderboard npm run submit
#   ./cloud/sky-bench.sh down   4090
#
# `npm run submit` now pushes to the HF dataset by default. Use
#   npm run submit -- --legacy-file-only
# if you want the old write-to-data/machines + git-PR flow instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# id → (GPU_TYPE, MACHINE_SLUG)
# ---------------------------------------------------------------------------
id_to_gpu_type() {
    case "$1" in
        4090)   echo "RTX4090" ;;
        5090)   echo "RTX5090" ;;
        mi300x) echo "MI300X" ;;
        *) echo "unknown id: $1 (expected 4090|5090|mi300x)" >&2; return 1 ;;
    esac
}

id_to_slug() {
    case "$1" in
        4090)   echo "runpod-rtx4090" ;;
        5090)   echo "runpod-rtx5090" ;;
        mi300x) echo "runpod-mi300x" ;;
        *) echo "unknown id: $1" >&2; return 1 ;;
    esac
}

cpu_cluster() { echo "webgpu-bench-cpu-$1"; }
gpu_cluster() { echo "webgpu-bench-gpu-$1"; }

# Read llama.cpp commit locally so both pods record the same hash.
get_llama_commit() {
    if [ -d "$REPO_ROOT/llama.cpp/.git" ] || [ -f "$REPO_ROOT/llama.cpp/.git" ]; then
        git -C "$REPO_ROOT/llama.cpp" rev-parse HEAD 2>/dev/null || echo ""
    else
        echo ""
    fi
}

# ---------------------------------------------------------------------------
# Argument parsing: extract optional --consistency / --smoke flags.
# ---------------------------------------------------------------------------
CONSISTENCY=0
SMOKE=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --consistency) CONSISTENCY=1 ;;
        --smoke)       SMOKE=1 ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done
set -- "${POSITIONAL[@]}"

VERB="${1:-}"
ID="${2:-}"

if [ -z "$VERB" ]; then
    echo "usage: $0 <verb> <id> [--consistency]" >&2
    echo "verbs: launch-cpu, launch-gpu, launch, sync-baselines, fetch, stop, down, status" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Flag composition for the two pods based on mode.
# ---------------------------------------------------------------------------
# Smoke mode: one model × one quantization. Used to verify the setup/build
# and the full launch+fetch pipeline end-to-end before spending time on --quick.
# --quick is mutually exclusive with --variants in runner.js, so we drop it.
SMOKE_MODEL="Llama-3.2-1B-Instruct"
SMOKE_VARIANT="Q4_K_M"

cpu_flags() {
    if [ "$SMOKE" = "1" ]; then
        echo "--no-webgpu --no-cache --models=$SMOKE_MODEL --variants=$SMOKE_VARIANT --browsers=chromium --resume"
    elif [ "$CONSISTENCY" = "1" ]; then
        # Consistency needs the WebGPU attempt path to run so runner.js records
        # baselines. The failed WebGPU rows from the CPU pod are filtered out
        # during merge.
        echo "--quick --consistency --no-cache --browsers=chromium --resume"
    else
        echo "--quick --no-webgpu --no-cache --browsers=chromium --resume"
    fi
}

gpu_flags() {
    if [ "$SMOKE" = "1" ]; then
        echo "--no-cache --models=$SMOKE_MODEL --variants=$SMOKE_VARIANT --browsers=chromium --resume"
    elif [ "$CONSISTENCY" = "1" ]; then
        echo "--quick --consistency --no-cache --browsers=chromium --resume"
    else
        echo "--quick --no-cache --browsers=chromium --resume"
    fi
}

# ---------------------------------------------------------------------------
# Launch helpers.
# ---------------------------------------------------------------------------
launch_cpu() {
    local id="$1"
    local slug; slug=$(id_to_slug "$id")
    local cluster; cluster=$(cpu_cluster "$id")
    local gpu_type; gpu_type=$(id_to_gpu_type "$id")
    local llama; llama=$(get_llama_commit)
    local flags; flags=$(cpu_flags)

    echo "=== launch-cpu: cluster=$cluster slug=$slug flags='$flags' ==="
    sky launch -y -c "$cluster" \
        --env "MACHINE_SLUG=$slug" \
        --env "LLAMA_CPP_COMMIT=$llama" \
        --env "BENCH_FLAGS=$flags" \
        "$SCRIPT_DIR/task-cpu.yaml"
}

launch_gpu() {
    local id="$1"
    local slug; slug=$(id_to_slug "$id")
    local cluster; cluster=$(gpu_cluster "$id")
    local gpu_type; gpu_type=$(id_to_gpu_type "$id")
    local llama; llama=$(get_llama_commit)
    local flags; flags=$(gpu_flags)

    echo "=== launch-gpu: cluster=$cluster gpu=$gpu_type slug=$slug flags='$flags' ==="

    # In consistency mode, push the CPU pod's baseline file into the GPU pod
    # *before* the benchmark starts so runner.js finds cached baselines and
    # skips regeneration. Requires the CPU pod to already be up.
    if [ "$CONSISTENCY" = "1" ]; then
        echo "=== consistency mode: syncing cpu_baselines.json before GPU launch ==="
        sync_baselines "$id"
    fi

    sky launch -y -c "$cluster" \
        --gpus "$gpu_type" \
        --env "MACHINE_SLUG=$slug" \
        --env "LLAMA_CPP_COMMIT=$llama" \
        --env "BENCH_FLAGS=$flags" \
        "$SCRIPT_DIR/task-gpu.yaml"
}

# ---------------------------------------------------------------------------
# File transfer via plain rsync + SkyPilot's SSH config.
# SkyPilot 0.12 dropped `sky rsync_up`/`sky rsync_down`; the supported path is
# `rsync <cluster>:path local/` where <cluster> resolves via entries SkyPilot
# writes into ~/.sky/generated/ssh/ (Include'd from ~/.ssh/config). We call
# `sky status <cluster>` first to refresh those entries in case the cluster
# was restarted since the last ssh config update.
# ---------------------------------------------------------------------------
refresh_ssh_config() {
    sky status "$1" >/dev/null 2>&1 || true
}

# sync-baselines: CPU pod → laptop → GPU pod
sync_baselines() {
    local id="$1"
    local cpu; cpu=$(cpu_cluster "$id")
    local gpu; gpu=$(gpu_cluster "$id")
    local tmp="/tmp/cpu_baselines-$id.json"

    refresh_ssh_config "$cpu"
    refresh_ssh_config "$gpu"

    echo "--- rsync down $cpu:sky_workdir/results/cpu_baselines.json → $tmp"
    rsync -Pavz "$cpu":sky_workdir/results/cpu_baselines.json "$tmp"

    # Ensure results/ exists on the GPU pod, then push the file.
    echo "--- ensuring sky_workdir/results on $gpu and uploading baselines"
    ssh "$gpu" "mkdir -p ~/sky_workdir/results"
    rsync -Pavz "$tmp" "$gpu":sky_workdir/results/cpu_baselines.json
}

# fetch: rsync both pods' results/ down, merge into results-cloud/<id>/results.json
fetch() {
    local id="$1"
    local cpu; cpu=$(cpu_cluster "$id")
    local gpu; gpu=$(gpu_cluster "$id")
    local out_dir="$REPO_ROOT/results-cloud/$id"

    refresh_ssh_config "$cpu"
    refresh_ssh_config "$gpu"

    mkdir -p "$out_dir/cpu" "$out_dir/gpu"

    echo "--- fetching CPU pod results"
    rsync -Pavz "$cpu":sky_workdir/results/ "$out_dir/cpu/" || \
        echo "warning: CPU pod fetch failed (cluster may be stopped/missing)"

    echo "--- fetching GPU pod results"
    rsync -Pavz "$gpu":sky_workdir/results/ "$out_dir/gpu/" || \
        echo "warning: GPU pod fetch failed (cluster may be stopped/missing)"

    echo "--- merging results.json arrays → $out_dir/results.json"
    node - "$out_dir" <<'EOF'
const fs = require('node:fs');
const path = require('node:path');

const outDir = process.argv[2];
const cpuFile = path.join(outDir, 'cpu', 'results.json');
const gpuFile = path.join(outDir, 'gpu', 'results.json');
const mergedFile = path.join(outDir, 'results.json');

const readJson = (f) => fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf-8')) : [];

const cpuRows = readJson(cpuFile);
const gpuRows = readJson(gpuFile);

// Keep only CPU-baseline performance rows from the CPU pod (nGpuLayers=0,
// successful). Discards the failed-WebGPU rows that show up in consistency mode.
const cpuFiltered = cpuRows.filter(
    r => r.nGpuLayers === 0 && r.status === 'done' && r.metrics
);

const merged = [...cpuFiltered, ...gpuRows];
fs.writeFileSync(mergedFile, JSON.stringify(merged, null, 2));

const passed = merged.filter(r => r.status === 'done').length;
console.log(`merged: ${merged.length} rows (${cpuFiltered.length} cpu-baseline + ${gpuRows.length} gpu) — ${passed} passed`);
EOF
}

# ---------------------------------------------------------------------------
# Lifecycle: stop, down, status
# ---------------------------------------------------------------------------
stop_both() {
    local id="$1"
    sky stop -y "$(cpu_cluster "$id")" || true
    sky stop -y "$(gpu_cluster "$id")" || true
}

down_both() {
    local id="$1"
    sky down -y "$(cpu_cluster "$id")" || true
    sky down -y "$(gpu_cluster "$id")" || true
}

status_both() {
    local id="$1"
    sky status "$(cpu_cluster "$id")" "$(gpu_cluster "$id")" || true
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
require_id() {
    if [ -z "$ID" ]; then
        echo "missing <id> (4090|5090|mi300x)" >&2
        exit 1
    fi
    # validate
    id_to_gpu_type "$ID" >/dev/null
}

case "$VERB" in
    launch-cpu)    require_id; launch_cpu "$ID" ;;
    launch-gpu)    require_id; launch_gpu "$ID" ;;
    launch)        require_id; launch_cpu "$ID"; launch_gpu "$ID" ;;
    sync-baselines) require_id; sync_baselines "$ID" ;;
    fetch)         require_id; fetch "$ID" ;;
    stop)          require_id; stop_both "$ID" ;;
    down)          require_id; down_both "$ID" ;;
    status)        require_id; status_both "$ID" ;;
    *) echo "unknown verb: $VERB" >&2; exit 1 ;;
esac
