# Cloud benchmark runs (SkyPilot + RunPod)

Run the WebGPU benchmark on RunPod GPU pods (RTX 4090, RTX 5090, MI300X proxy)
without owning the hardware. Each target splits into **two pods**: a cheap
CPU-only pod for the CPU baseline and a GPU pod for the WebGPU portion. The
wrapper merges their results locally before `npm run submit`.

## Prerequisites

```bash
# Venv (optional but recommended)
uv venv cloud/.venv && source cloud/.venv/bin/activate

uv pip install 'skypilot[runpod]' 'runpod>=1.6.1'

# Grab an API key from https://www.runpod.io/console/user/settings then:
runpod config                       # writes ~/.runpod/config.toml
sky check runpod                    # expect "RunPod: enabled"

# Confirm all three GPU names exist in the catalog:
sky gpus list --infra runpod -a | grep -E 'RTX4090|RTX5090|MI300X'
```

The last command must list all three names. If one is missing, that target
is unavailable in your SkyPilot release.

### Patch SkyPilot's RunPod CPU provisioner

SkyPilot 0.12.0 (and current master) crashes on CPU-only RunPod pods with
`KeyError: 'internal_ip'` right after provisioning. Upstream bug, no fix
released yet. One-line patch in the venv's installed package:

```bash
# File: cloud/.venv/lib/python3.13/site-packages/sky/provision/runpod/instance.py
# Change line ~205 from:
#     internal_ip=instance_info['internal_ip'],
# to:
#     internal_ip=instance_info.get('internal_ip', instance_info.get('external_ip')),
```

Re-apply after any `pip install -U skypilot`.

## Short ids

| id       | GPU         | MACHINE_SLUG          |
| -------- | ----------- | --------------------- |
| `4090`   | RTX 4090    | `runpod-rtx4090`      |
| `5090`   | RTX 5090    | `runpod-rtx5090`      |
| `mi300x` | MI300X      | `runpod-mi300x`       |

## Usage

All verbs live in `cloud/sky-bench.sh`. Run from the repo root. Activate the
venv first: `source cloud/.venv/bin/activate`.

### Smoke test first (always)

Before a full run, verify the entire pipeline (setup → build → browser →
fetch) with one model × one quantization × two browsers. Takes ~5 min per
pod, costs ~$0.20 on 4090:

```bash
./cloud/sky-bench.sh launch-cpu 4090 --smoke
./cloud/sky-bench.sh launch-gpu 4090 --smoke
./cloud/sky-bench.sh fetch      4090
# Inspect results-cloud/4090/results.json — expect 4 rows, all status:done,
# prefill/decode tok/s populated.
./cloud/sky-bench.sh down       4090
```

Smoke target: `Llama-3.2-1B-Instruct` × `Q4_K_M`, Chromium only. Only
when this passes should you start full runs.

### Perf mode (default)

```bash
./cloud/sky-bench.sh launch 4090     # spin up both CPU + GPU pods
./cloud/sky-bench.sh status 4090     # check readiness
./cloud/sky-bench.sh fetch  4090     # rsync + merge → results-cloud/4090/results.json
cp results-cloud/4090/results.json results/results.json
npm run submit                        # writes data/machines/runpod-rtx4090.json
./cloud/sky-bench.sh down   4090     # destroy both pods
```

### Consistency mode

```bash
./cloud/sky-bench.sh launch-cpu 4090 --consistency   # CPU pod generates cpu_baselines.json
./cloud/sky-bench.sh launch-gpu 4090 --consistency   # wrapper auto-rsyncs baselines, then launches
./cloud/sky-bench.sh fetch      4090                 # merge
./cloud/sky-bench.sh down       4090
```

The GPU pod reads the cached baselines (byte-portable because both pods build
the same `LLAMA_CPP_COMMIT`), skips regeneration, and runs forced-decoding.

### Individual verbs

All verbs accept optional `--consistency` and/or `--smoke`.

- `launch-cpu <id>` — just the CPU pod.
- `launch-gpu <id>` — just the GPU pod (auto-syncs baselines in consistency mode).
- `launch <id>` — both pods (CPU first so baselines exist when GPU runs).
- `sync-baselines <id>` — manually rsync `cpu_baselines.json` from the CPU pod to the GPU pod.
- `fetch <id>` — pull results from both pods, merge into `results-cloud/<id>/results.json`.
- `stop <id>` — pause both pods (cheap; resume skips setup).
- `down <id>` — destroy both pods.
- `status <id>` — `sky status` for both clusters.

### Flag reference

| Flag            | Effect                                                                       |
| --------------- | ---------------------------------------------------------------------------- |
| `--smoke`       | One model × one quant × 2 browsers (`Llama-3.2-1B-Instruct` × `Q4_K_M`).     |
| `--consistency` | Enable CPU-baseline + forced-decoding top-1 token agreement on GPU.          |

Both can combine (`--smoke --consistency` verifies the consistency plumbing).

### Running across all three targets

After the 4090 smoke and full runs pass, fan out. Parallel launch (6 pods
concurrently):

```bash
for id in 4090 5090 mi300x; do
    ./cloud/sky-bench.sh launch $id > /tmp/sky-$id.log 2>&1 &
done
wait
for id in 4090 5090 mi300x; do ./cloud/sky-bench.sh fetch $id; done
# Verify each results-cloud/<id>/results.json, then:
for id in 4090 5090 mi300x; do ./cloud/sky-bench.sh down $id; done
```

Tail progress with `tail -f /tmp/sky-*.log` or `sky logs webgpu-bench-gpu-<id> --follow`.

Sequential (simpler, ~1.5h total):

```bash
for id in 4090 5090 mi300x; do
    ./cloud/sky-bench.sh launch $id && \
    ./cloud/sky-bench.sh fetch  $id && \
    ./cloud/sky-bench.sh down   $id
done
```

## What happens on merge

The CPU pod's `results.json` contains CPU-baseline rows and, in consistency
mode, failed WebGPU-attempt rows (the CPU pod has no GPU). The wrapper's
`fetch` filters the CPU pod to keep only `nGpuLayers=0` + `status=='done'` +
non-null metrics rows, then concatenates with everything from the GPU pod.
Both pods share the same `MACHINE_SLUG`, so `npm run submit` writes a single
`data/machines/runpod-<id>.json`.

## Expected costs

RunPod "secure" 1x pricing from `sky gpus list --infra runpod -a`:

| Pod               | On-demand | Spot     | vCPU | RAM    |
| ----------------- | --------- | -------- | ---- | ------ |
| CPU `cpu3c-16-32` | $0.48/h   | —        | 16   | 32 GB  |
| RTX 4090          | $0.69/h   | $0.29/h  | 5    | 29 GB  |
| RTX 5090          | $0.99/h   | $0.53/h  | 6    | 46 GB  |
| MI300X            | $1.99/h   | $1.49/h  | 24   | 283 GB |

A single `--quick` end-to-end on RTX 4090 (≈15 min CPU + ≈15 min GPU) lands
under $0.30 on-demand. Consistency mode roughly doubles GPU wall time.

The 1x RTX 4090 pod ships with only 5 vCPU / 29 GB — this is why the CPU
baseline runs on a separate 16-vCPU CPU-only pod. The `cpus: "16+"`
constraint in `task-cpu.yaml` does **not** apply to the GPU pod.

## Known risks

- **MI300X WebGPU adapter detection** — ROCm is present on RunPod MI300X base
  images, but the Vulkan/Mesa layers needed for Chromium's WebGPU under xvfb
  are untested. Expect `webgpuAvailable: false` as a plausible outcome. Treat
  as a probe, not a gated requirement.
- **RTX 5090 catalog availability** — 5090 is recent; confirm
  `sky gpus list --infra runpod -a` lists it before launching. Otherwise wait
  for a newer SkyPilot release.
- **SkyPilot RunPod CPU crash** — `KeyError: 'internal_ip'` on CPU pod setup
  is an upstream bug. See the one-line patch in Prerequisites above; re-apply
  after upgrading SkyPilot.
- **RunPod capacity** — specific SKUs go out of stock region-by-region
  (`cpu3c-16-32` and `2x_RTX4000-Ada_SECURE` both seen unavailable in CA/CZ/
  IS/NL/NO/RO/SE/US at the same time). SkyPilot retries across all regions
  automatically; if all fail, wait or pass `--retry-until-up` manually.
- **RunPod host CPU drift** — CPU and GPU pods may land on different EPYC
  SKUs. `MACHINE_SLUG` hides this in the filename, but the `machine.cpus`
  field inside results reflects whichever host was assigned. Exact CPU match
  between the two pods is not achievable on RunPod.
- **WASM bit-identity across pods (consistency only)** — `build.sh` installs
  the latest emsdk on each pod. Back-to-back launches are almost always
  identical; long gaps between CPU and GPU setup could produce drift. If it
  matters, pin emsdk in `build.sh`.

## Resume semantics

`sky stop <cluster>` pauses the pod (disk preserved). A subsequent
`sky launch -c <cluster>` resumes it and **does not re-run `setup:`**.
`cloud/setup.sh` also has a belt-and-suspenders guard that short-circuits if
`build/jspi/bin/bench.wasm` and `build/asyncify/bin/bench.wasm` both exist.

## Files in this directory

| File             | Purpose                                                    |
| ---------------- | ---------------------------------------------------------- |
| `PLAN.md`        | Original design doc (handoff plan this was built from).    |
| `sky-bench.sh`   | Wrapper CLI — the only entry point you should run.         |
| `task-cpu.yaml`  | SkyPilot task for the CPU-only pod.                        |
| `task-gpu.yaml`  | SkyPilot task for the GPU pod.                             |
| `setup.sh`       | Pod bootstrap (apt, Node 20, Playwright, WASM build).      |
| `run.sh`         | Pod entry: `xvfb-run node runner.js $BENCH_FLAGS`.         |
