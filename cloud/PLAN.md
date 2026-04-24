# Plan: SkyPilot + RunPod cloud benchmark automation

> **Handoff doc.** This plan was drafted in a prior Claude session. A fresh session with full permissions should execute it end-to-end. Nothing below has been implemented yet — all listed files/paths are targets to create or edit.

## Context

Today the WebGPU benchmark only runs on the operator's local machine. We want to extend it to ephemeral RunPod VMs so results can be collected on consumer-class NVIDIA (RTX 4090, RTX 5090) and a best-available AMD proxy (MI300X) without owning the hardware. The workflow per target: provision VM → install toolchain → build WASM → run `--quick` → rsync results back → pause/destroy. Resuming a paused VM must skip the rebuild (artifacts already on disk).

A second cost-driven requirement: the CPU portion of the benchmark (`--no-webgpu`) is pure CPU workload. Running it on a $0.34/h RTX pod wastes ~8× vs. a $0.04/h EPYC CPU-only pod. So the design splits each GPU target into **two pods** — one CPU-only for the CPU baseline, one GPU for the WebGPU benchmark — and merges their results on the laptop before `npm run submit`. The split also applies to `--consistency` mode: the CPU pod generates `cpu_baselines.json`, the wrapper rsyncs it to the GPU pod, and the GPU pod runs `--consistency` with the baseline already cached (skipping regen) so it only does the WebGPU + forced-decoding portion.

## Repo runtime facts (from exploration of current code)

- `server.js:144` binds implicit `localhost:3000` (no env var override); fine because the browser runs on the same VM.
- `config.js:130-136` reads **zero env vars**; `MACHINE` object = `{platform, arch, cpus[0].model, totalMemoryGB, hostname}` — **no GPU field**.
- `runner.js:20-26` `getLlamaCppCommit()` runs `git -C llama.cpp rev-parse HEAD`; result stored as `llamaCppCommit` per result.
- `runner.js:31` sets `headless: false` → xvfb-run required on Linux. **Keep headless false** (see design decision #8).
- `scripts/submit-results.js:20` generates slug as `${cpu}-${ram}gb-${platform}` → **two GPUs on same RunPod host CPU produce identical slugs = collision** unless overridden.
- `--quick` downloads ~40GB cumulative but peak on-disk is ~7.3GB (variant-first loop deletes cache per model).
- `build.sh` handles emsdk + emdawnwebgpu download internally; platform-agnostic via Emscripten.
- Only existing CI is `.github/workflows/deploy-dashboard.yml` — no benchmark automation pattern to align with.

## Architecture

```
laptop                                 RunPod via SkyPilot
──────                                 ───────────────────
cloud/sky-bench.sh  ──launch-cpu──►    webgpu-bench-cpu-<id>   (no accelerator)
                                          perf mode:   node runner.js --quick --no-webgpu
                                          consistency: node runner.js --quick --consistency
                                          writes: ~/sky_workdir/results/{results.json, cpu_baselines.json}

                    ──launch-gpu──►    webgpu-bench-gpu-<id>   (RTX4090 / RTX5090 / MI300X)
                                          perf mode:   node runner.js --quick
                                          consistency: (1) rsync cpu_baselines.json down from CPU pod
                                                       (2) rsync_up into GPU pod at ~/sky_workdir/results/
                                                       (3) node runner.js --quick --consistency
                                          writes: ~/sky_workdir/results/results.json

                    ──fetch──────►    rsync_down both → results-cloud/<id>/{cpu,gpu}/
                                       → merge results.json arrays into results-cloud/<id>/results.json
                    ──stop / down──►  pause or destroy both pods
```

`<id>` is the short GPU id: `4090`, `5090`, or `mi300x`.

Both pods set `MACHINE_SLUG=runpod-<id>` (e.g. `runpod-rtx4090`) so merged results land in a single `data/machines/runpod-rtx4090.json` after `npm run submit`.

**Consistency-mode flow detail:** In `--consistency` mode the CPU pod's `results/results.json` contains (a) the useful `cpu_baselines.json` side-product and (b) failed/no-WebGPU rows we discard during merge. The wrapper's `fetch` filters CPU-pod rows to keep only those with `nGpuLayers=0` and non-null metrics (CPU-baseline performance rows), discarding failed-WebGPU rows. For GPU-pod rows we keep everything. Merged `results.json` contains CPU-baseline perf + WebGPU perf + consistency data — all under the same `MACHINE_SLUG`.

## Files to create

| Path | Purpose |
|---|---|
| `cloud/task-gpu.yaml` | SkyPilot task: RunPod GPU pod; `resources.accelerators=${GPU_TYPE}`, `disk_size=120`; envs: `GPU_TYPE`, `MACHINE_SLUG`, `LLAMA_CPP_COMMIT`, `BENCH_FLAGS` (default `--quick --browsers=chromium,firefox --resume`); `setup: bash cloud/setup.sh`; `run: bash cloud/run.sh`. Wrapper sets `BENCH_FLAGS` to include `--consistency` when user passes the flag. |
| `cloud/task-cpu.yaml` | SkyPilot task: CPU-only RunPod pod; `resources: { cloud: runpod, cpus: "16+", memory: "64+", disk_size: 120 }`; same envs; `BENCH_FLAGS` default `--quick --no-webgpu --browsers=chromium,firefox --resume`. When wrapper invokes with `--consistency`, flags become `--quick --consistency --browsers=chromium,firefox --resume` (drops `--no-webgpu` since consistency needs the WebGPU attempt path to run; the CPU pod's WebGPU rows will be failures we discard on merge). |
| `cloud/setup.sh` | Idempotent bootstrap. Guard: `[ -f build/jspi/bin/bench.wasm ] && [ -f build/asyncify/bin/bench.wasm ] && exit 0` (fast-path resume). Otherwise: `apt-get install -y xvfb ninja-build cmake build-essential git curl unzip`; install Node 20 via NodeSource; `npm ci`; `npx playwright install --with-deps chromium firefox`; `bash build.sh` (handles emsdk + emdawnwebgpu internally). |
| `cloud/run.sh` | `cd ~/sky_workdir && xvfb-run -a --server-args="-screen 0 1920x1080x24" node runner.js ${BENCH_FLAGS}`. **Browsers remain headed** (runner.js:31 keeps `headless: false` unchanged); xvfb provides a virtual display so headed browsers run on display-less VMs. Headless mode is intentionally avoided to keep WebGPU adapter selection identical to desktop runs. |
| `cloud/sky-bench.sh` | Wrapper CLI with verbs `launch-cpu`, `launch-gpu`, `launch` (both), `sync-baselines`, `fetch`, `stop`, `down`, `status`. All verbs accept optional `--consistency` flag. Maps short id (`4090`/`5090`/`mi300x`) → `GPU_TYPE`, cluster names (`webgpu-bench-cpu-<id>`, `webgpu-bench-gpu-<id>`), and `MACHINE_SLUG=runpod-<id>`. Reads `git -C llama.cpp rev-parse HEAD` locally and passes as `LLAMA_CPP_COMMIT` env to both clusters. `sync-baselines <id>` is called automatically inside `launch-gpu --consistency` (and manually exposed): `sky rsync_down <cpu-cluster>:~/sky_workdir/results/cpu_baselines.json /tmp/cpu_baselines-<id>.json` then `sky rsync_up <gpu-cluster>:/tmp/cpu_baselines-<id>.json ~/sky_workdir/results/cpu_baselines.json`. `fetch` rsyncs both pods' `results/` into `results-cloud/<id>/{cpu,gpu}/` then merges their `results.json` arrays (filtering CPU-pod rows to keep only `nGpuLayers=0` successful rows) into `results-cloud/<id>/results.json` via a small Node one-liner. |
| `cloud/README.md` | Prereqs (`pip install 'skypilot[runpod]'`, RunPod API key, `sky check`), GPU-name verification (`sky show-gpus --cloud runpod`), per-verb usage, manual submit flow, MI300X caveat, expected costs. |
| `.skyignore` (repo root) | Exclude `.git`, `node_modules`, `site/node_modules`, `build/`, `cache/`, `results/`, `results-cloud/`, `data/machines/`, `.DS_Store`. **Must live at workdir root**, not in `cloud/`. |

## Files to edit

| Path | Change |
|---|---|
| `config.js` (~lines 130–136) | Add `slug: process.env.MACHINE_SLUG \|\| null` to the `MACHINE` object. Laptop usage untouched (env unset → null → existing auto-generate path). |
| `runner.js` (~lines 20–26) | `getLlamaCppCommit()`: env-first, git-fallback. Prepend `if (process.env.LLAMA_CPP_COMMIT) return process.env.LLAMA_CPP_COMMIT;` before the `try { execSync(...) }`. |
| `scripts/submit-results.js` (~line 20) | In `generateSlug()`: `return machine.slug \|\| <existing cpu-ram-platform composition>;`. Preserves laptop slug; cloud pods write explicit slug. |
| `.gitignore` | Add `results-cloud/`. |

## Key design decisions

1. **Slug strategy: full `MACHINE_SLUG` env override.** CPU + GPU pods must land in the same `data/machines/<slug>.json` to merge. A suffix-append scheme would still diverge across the two pods' auto-generated CPU prefixes. Override is cleanest.

2. **Consistency mode: CPU pod generates baselines, wrapper transfers to GPU pod, GPU pod skips regeneration.** Feasible because `cpu_baselines.json` is byte-portable if both pods build the same llama.cpp commit with the same Emscripten toolchain (the wrapper enforces same commit via `LLAMA_CPP_COMMIT`; same Emscripten version is a new assumption since build.sh installs latest emsdk on each pod — worth flagging but low risk since builds are temporally close). If baselines don't match what GPU pod would have generated, forced-decoding agreement rate will be lower than it should be but the benchmark still completes; results are not corrupted, just potentially misleading for that specific variant. Worst case: rerun the GPU pod without rsyncing baselines so it regenerates them locally.

3. **Git metadata: `LLAMA_CPP_COMMIT` env, not `.git/` sync.** The llama.cpp submodule alone is ~500MB of git history. Wrapper runs `git -C llama.cpp rev-parse HEAD` locally and passes via env; `runner.js` prefers env, falls back to git.

4. **Rebuild-on-first-launch, skip-on-resume: achieved via SkyPilot's built-in `setup:` semantics.** `setup:` runs once at cluster creation, not on subsequent `sky launch -c <cluster>` resumes. `setup.sh` additionally has a belt-and-suspenders guard (`[ -f build/jspi/bin/bench.wasm ] && exit 0`) in case SkyPilot re-runs setup after a workdir change.

5. **CPU pod specification: `cpus: "16+"`, `memory: "64+"`.** RunPod doesn't expose per-SKU CPU selection, so exact matching is impossible. 16 vCPU / 64GB is typical of what accompanies an RTX pod and ensures the CPU baseline is not artificially constrained.

6. **Disk size: 120GB.** `--quick` peak on-disk is ~7GB (variant-first loop deletes cache per model), but cumulative HuggingFace download is ~40GB, plus emsdk (~5GB), Playwright browsers (~1GB), llama.cpp build (~3GB). 120GB is safe.

7. **Wrapper > npm scripts.** A single bash wrapper (`cloud/sky-bench.sh`) centralizes the id→cluster-name→env-var mapping. npm scripts would require ~12 entries (3 ids × 4+ verbs) and can't derive env vars from the id cleanly.

8. **Results land in `results-cloud/<id>/`, never in `results/`.** Fetching directly to `results/` would clobber whatever the user is running locally. The wrapper's `fetch` creates `results-cloud/<id>/results.json` (merged CPU+GPU); user manually copies to `results/results.json` before `npm run submit`.

9. **Headed browsers via xvfb — never switch to headless.** `runner.js:31` keeps `headless: false`; `cloud/run.sh` wraps with `xvfb-run -a --server-args="-screen 0 1920x1080x24"` so headed browsers get a virtual display on display-less VMs. Chromium headless mode has inconsistent WebGPU adapter behavior, so staying headed keeps cloud results comparable to the existing laptop/Apple-M3 data. **Do not "simplify" by switching to headless.**

10. **Two task YAMLs, not one parameterized.** SkyPilot can't conditionally include/exclude `accelerators` from a single YAML. Two small files are clearer than clever YAML tricks.

## Verification

1. **Local lint-style check (no cost):**
   - `sky check` — confirms RunPod creds are set.
   - `sky validate cloud/task-gpu.yaml` and `sky validate cloud/task-cpu.yaml`.
   - `sky show-gpus --cloud runpod | grep -E 'RTX4090|RTX5090|MI300X'` — confirm SkyPilot's catalog has the names we'll use.
   - `bash -n cloud/setup.sh cloud/run.sh cloud/sky-bench.sh` — syntax check scripts.

2. **CPU-pod smoke test (~$0.04/h):**
   - `./cloud/sky-bench.sh launch-cpu 4090` (CPU-only, `--no-webgpu`).
   - Expect ~15 min runtime for `--quick --no-webgpu`.
   - `./cloud/sky-bench.sh fetch 4090` — confirm `results-cloud/4090/cpu/results.json` has the expected row count (30 variants × 2 browsers).
   - `./cloud/sky-bench.sh down 4090` — destroy CPU pod.

3. **GPU-pod smoke test — RTX 4090 (~$0.34/h):**
   - `./cloud/sky-bench.sh launch-gpu 4090` (no consistency).
   - Expect ~15 min runtime; monitor `sky logs webgpu-bench-gpu-4090 --follow`.
   - Inspect first result for `gpuAdapterInfo` populated with RTX 4090 details.
   - `./cloud/sky-bench.sh fetch 4090` — now `results-cloud/4090/results.json` is merged CPU+GPU.
   - `cp results-cloud/4090/results.json results/results.json && npm run submit` — confirm `data/machines/runpod-rtx4090.json` is created with both CPU-perf and WebGPU-perf rows and matching slug.

4. **Consistency-split end-to-end:**
   - `./cloud/sky-bench.sh launch-cpu 4090 --consistency` — CPU pod generates baselines; results.json will have mostly-failed WebGPU rows (expected, to be filtered on merge).
   - `./cloud/sky-bench.sh launch-gpu 4090 --consistency` — wrapper first rsyncs `cpu_baselines.json` from CPU cluster → GPU cluster; GPU pod runs, sees cached baselines, skips regen, does forced-decoding. Check `sky logs` for a "using cached CPU baselines" message from runner.js.
   - `./cloud/sky-bench.sh fetch 4090` — merged results include `consistency.agreement_rate` fields on GPU rows.
   - `./cloud/sky-bench.sh down 4090` — teardown.

5. **Resume semantics check:**
   - `./cloud/sky-bench.sh stop 4090` — pauses both pods.
   - `./cloud/sky-bench.sh launch-gpu 4090` — should resume in <60s without re-running setup (check `sky logs` — no "Running setup" line).
   - `./cloud/sky-bench.sh down 4090` — final teardown.

6. **MI300X exploratory run (do this last, highest risk):**
   - `./cloud/sky-bench.sh launch-gpu mi300x`.
   - If `webgpuAvailable: false` across all results → expected failure mode; document in README and move on.

## Known risks (runtime-verify-only, no plan fix)

1. **MI300X WebGPU adapter detection** — RunPod MI300X base images carry ROCm but Vulkan loader/Mesa layers for MI300X exposed to a headed Chromium under xvfb are untested. Chromium `--use-angle=vulkan` may find no adapter. Accept as a probe, not a gating requirement.
2. **RTX 5090 SkyPilot catalog availability** — 5090 is recent; `sky show-gpus --cloud runpod` must confirm the name is recognized. If not, wait for a SkyPilot release or fall back to 4090.
3. **Firefox WebGPU on Linux cloud** — Playwright installs stable Firefox; stable doesn't enable WebGPU. Expect `webgpuAvailable: false` for Firefox rows; runner.js handles it gracefully.
4. **RunPod host CPU drift** — RunPod may assign different EPYC SKUs to your CPU and GPU pods. The fixed `MACHINE_SLUG` hides this, but the `machine.cpus` field in results will reflect whichever host was assigned. If you need exact-match CPU between CPU-baseline and GPU runs, this is not achievable with RunPod and is not solved here.
5. **WASM bit-identity across pods (consistency mode only)** — baseline portability assumes both pods' `bench.wasm` is functionally identical. Both pods run `build.sh` fresh → install latest emsdk. If emsdk upstream releases between the two `setup:` runs, builds may diverge. Low probability for back-to-back launches; document as a known risk. Mitigation if it matters: pin emsdk version in build.sh (out of scope here).

## Critical files referenced

- `config.js:130-136` — `MACHINE` object construction (edit target)
- `runner.js:20-26` — `getLlamaCppCommit()` (edit target)
- `runner.js:31` — `headless: false` (do NOT change — xvfb handles the display)
- `scripts/submit-results.js:20` — `generateSlug()` (edit target)
- `build.sh` — invoked unchanged by `setup.sh`
- `server.js:144` — implicit `localhost:3000` binding (unchanged; works because browser is on same VM)
- `config.js:98` — server port constant (informational)

## Implementation order (suggested)

1. `.skyignore` at repo root, `.gitignore` add `results-cloud/`.
2. Three code patches: `config.js`, `runner.js`, `scripts/submit-results.js`.
3. `cloud/setup.sh`, `cloud/run.sh` — shell scripts with `bash -n` syntax check.
4. `cloud/task-cpu.yaml`, `cloud/task-gpu.yaml` — `sky validate` to lint.
5. `cloud/sky-bench.sh` — wrapper with all verbs.
6. `cloud/README.md` — operator docs.
7. Run Verification #1 (local lint-style, free).
8. Hand back to operator for Verification #2–#6 (incurs RunPod cost).
