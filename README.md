# webgpu-bench

Automated benchmarking framework for llama.cpp's WebGPU backend. Compiles llama.cpp to WebAssembly, runs GGUF models in real browsers via WebGPU, and measures inference performance and numerical correctness across Chrome, Firefox, and Safari.

Currently tracks **10 models** with **230 quantization variants** from HuggingFace.

## Prerequisites

| Dependency | Version | Notes |
|-----------|---------|-------|
| [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html) | Latest | Set `EMSDK_DIR` env var or place at `../emsdk/` |
| [Ninja](https://ninja-build.org/) | Any | `brew install ninja` / `apt install ninja-build` |
| [Node.js](https://nodejs.org/) | 18+ | |
| [CMake](https://cmake.org/) | 3.14+ | |
| [Playwright browsers](https://playwright.dev/) | Installed via npx | For Chrome and Firefox |
| Safari Remote Automation | macOS only | Safari > Settings > Advanced > "Allow Remote Automation" |

## Quick Start

```bash
# Clone with submodules (llama.cpp)
git clone --recurse-submodules <repo-url>
cd webgpu-bench

# Build WASM (downloads emdawnwebgpu automatically)
npm run build

# Install dependencies + Playwright browsers
npm install
npx playwright install chromium firefox

# Run a quick benchmark (3 quants, Chromium only)
node runner.js --quick --browsers=chromium

# View results
node report.js
```

## Build

Compiles llama.cpp (git submodule) to two WASM variants with WebGPU support:

| Variant | Browser | Mechanism |
|---------|---------|-----------|
| **JSPI** | Chrome | JavaScript Promise Integration (native async) |
| **Asyncify** | Firefox, Safari | Emscripten Asyncify (transform-based async) |

```bash
npm run build
# or: bash build.sh
```

The browser automatically detects JSPI support at runtime and loads the correct variant. [emdawnwebgpu](https://github.com/nicehash/nicehash-browser/wiki/emdawnwebgpu) (WebGPU bindings for Emscripten) is downloaded on first build.

Output:
```
build/jspi/bin/bench.js + bench.wasm
build/asyncify/bin/bench.js + bench.wasm
```

## Running Benchmarks

### CLI Options

```bash
node runner.js [options]
```

| Flag | Description | Example |
|------|-------------|---------|
| _(none)_ | All 230 variants on all 3 browsers | `node runner.js` |
| `--quick` | Only Q2_K, Q4_K_M, Q8_0 | `node runner.js --quick` |
| `--browsers=` | Comma-separated browser list | `--browsers=chromium,firefox` |
| `--variants=` | Specific quantization types | `--variants=Q4_K_M,Q8_0` |
| `--models=` | Filter by model name (substring match) | `--models=Llama-3.2-1B` |
| `--no-webgpu` | CPU-only mode (disable GPU offload) | `--no-webgpu` |
| `--consistency` | Measure WebGPU vs CPU numerical correctness | `--consistency` |
| `--resume` | Skip browser+variant+GPU-layer combos that already succeeded | `--resume` |

### Examples

```bash
# Quick smoke test on Chrome
node runner.js --quick --browsers=chromium

# All quants for a specific model
node runner.js --models=Qwen3-0.6B --browsers=chromium

# Full suite (expect 5-6 hours)
node runner.js

# Consistency check: how faithfully does WebGPU reproduce CPU results?
node runner.js --quick --consistency

# Resume a partial run (skips completed combos)
node runner.js --resume

# CPU-only baseline
node runner.js --quick --no-webgpu
```

### npm Scripts

| Script | Command |
|--------|---------|
| `npm run build` | Build WASM (both variants) |
| `npm run bench` | Run all benchmarks |
| `npm run bench:quick` | Quick benchmark (3 quants) |
| `npm run bench:chromium` | All quants, Chromium only |
| `npm run report` | Generate CSV from results |
| `npm run submit` | Prepare results for PR submission |
| `npm run build:site` | Build dashboard data |

### Model Download Caching

Models are downloaded from HuggingFace through a local caching proxy. The first download for each model file hits HuggingFace and saves to `cache/models/`. Subsequent runs serve from disk, eliminating download overhead entirely.

Cache files persist across runs. To clear the cache: `rm -rf cache/`.

## Results

Results are saved to `results/`:

| File | Description |
|------|-------------|
| `results.json` | Full benchmark data with all metrics |
| `summary.json` | Grouped by browser with pass/fail status |
| `results.csv` | Flat CSV for spreadsheets |
| `cpu_baselines.json` | CPU reference token sequences (from `--consistency`) |

Generate reports:
```bash
npm run report
```

### Performance Metrics

| Field | Description |
|-------|-------------|
| `prefill_tok_s` | Prompt processing speed (tokens/sec) |
| `decode_tok_s` | Token generation speed (tokens/sec) |
| `t_p_eval_ms` | Prefill time in milliseconds |
| `t_eval_ms` | Decode time in milliseconds |
| `n_p_eval` | Number of prompt tokens processed |
| `n_eval` | Number of tokens generated |
| `buildType` | `jspi` or `asyncify` |
| `webgpuAvailable` | Whether WebGPU was available |

### Consistency Metrics (with `--consistency`)

| Field | Description |
|-------|-------------|
| `consistency.agreement_rate` | Fraction of positions where GPU and CPU agree on top-1 token (0.0-1.0) |
| `consistency.n_agree` | Number of agreeing positions |
| `consistency.n_tokens` | Total positions evaluated |
| `consistency.first_disagreement` | Position of first divergence (-1 if perfect) |

## Consistency Measurement

The `--consistency` flag measures how faithfully the WebGPU backend reproduces CPU computation for each quantization type.

### How It Works

For each variant, two runs happen in the **same browser** (isolating the WebGPU backend precisely):

1. **CPU baseline** (`n_gpu_layers=0`): greedy-decodes 128 tokens and records the token ID sequence. Cached to `results/cpu_baselines.json` so subsequent runs skip this step.
2. **WebGPU run** (`n_gpu_layers=999`): runs the normal benchmark, then performs a **forced-decoding pass** -- feeds the CPU's token sequence one token at a time and checks whether the GPU backend independently predicts the same top-1 token at each position.

When benchmarking across multiple browsers, the CPU baseline is shared (collected once from the first browser) since CPU computation is browser-independent.

### Why Forced Decoding, Not Text Comparison

Naively comparing generated text suffers from **cascading divergence**: a single different token changes the KV cache for all subsequent tokens, making the rest statistically unrelated. A text match ratio of 24% might mean only one token actually diverged.

Forced decoding evaluates each position independently against the same reference context, giving a clean per-token accuracy signal.

### Interpreting Results

| `agreement_rate` | Interpretation |
|---|---|
| `1.00` | Numerically identical to CPU -- no precision issues |
| `0.95-0.99` | A few tokens differ due to near-equal logits -- expected for lower-precision quants |
| `< 0.90` | Systematic precision issues -- the GPU kernel may need investigation |
| `0.00` | First token wrong -- the quantization kernel is likely broken |

## Dashboard

A static dashboard visualizes benchmark results across machines and browsers. Deployed to GitHub Pages automatically when results are merged to `main`.

### Local Preview

```bash
npm run build:site
npx serve site
```

### Dashboard Features

- **Support Matrix** -- pass/fail for each model/quant/browser combination
- **Performance Charts** -- decode and prefill throughput, throughput vs model size
- **Machine Comparison** -- side-by-side results when multiple machines have data
- **Error Analysis** -- failures grouped by category (OOM, WASM Abort, Timeout)
- **Filtering** -- filter by machine, browser, model, status, quantization type

### Submitting Results from Your Machine

```bash
# 1. Run benchmarks
node runner.js --browsers=chromium,firefox

# 2. Prepare results for submission
npm run submit

# 3. Commit and open a PR
git checkout -b results/my-machine
git add data/machines/
git commit -m "Add benchmark results for <your machine>"
git push -u origin results/my-machine
gh pr create
```

On merge, GitHub Actions rebuilds the dashboard with the new data.

### Data Pipeline

1. `npm run submit` reads `results/results.json`, strips heavy fields (`token_ids`, `output`), and writes to `data/machines/{slug}.json` (e.g., `apple-m4-pro-48gb-darwin`).
2. `npm run build:site` merges all machine files into `site/data/combined.json`.
3. The dashboard at `site/` fetches `combined.json` and renders everything client-side.

## Adding Models

Edit `models.json` to add new models, repos, or quantization types.

### Add a quantization variant

Add an entry to the model's `variants` array:

```json
{
  "quant": "Q5_0",
  "filename": "Llama-3.2-1B-Instruct-Q5_0.gguf",
  "sizeMB": 870
}
```

### Add a new model

Add a new entry to the top-level `models` array:

```json
{
  "repo": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
  "name": "Qwen2.5-1.5B-Instruct",
  "variants": [
    { "quant": "Q4_K_M", "filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", "sizeMB": 1050 },
    { "quant": "Q8_0",   "filename": "Qwen2.5-1.5B-Instruct-Q8_0.gguf",   "sizeMB": 1680 }
  ]
}
```

### Change quick variants

Edit the `quickVariants` array at the bottom of `models.json`:

```json
"quickVariants": ["Q2_K", "Q4_K_M", "Q8_0"]
```

### Finding model filenames on HuggingFace

```bash
curl -s "https://huggingface.co/api/models/<owner>/<repo>/tree/main" \
  | python3 -c "
import sys, json
for f in json.load(sys.stdin):
    if f['path'].endswith('.gguf'):
        print(f'{f[\"path\"]:60s} {f[\"size\"]/(1024**2):8.1f} MB')
"
```

### Current Models

| Model | Repo | Variants |
|-------|------|----------|
| Llama-3.2-1B-Instruct | unsloth | 27 |
| Llama-3.2-3B-Instruct | unsloth | 27 |
| gemma-3-270m-it | unsloth | 24 |
| gemma-3-1b-it | unsloth | 26 |
| Qwen3-0.6B | unsloth | 26 |
| Qwen3-1.7B | unsloth | 26 |
| Ministral-3-3B-Instruct-2512 | unsloth | 26 |
| DeepSeek-R1-Distill-Qwen-1.5B | unsloth | 17 |
| LFM2.5-1.2B-Instruct | LiquidAI | 7 |
| SmolLM3-3B-128K | unsloth | 24 |

## Architecture

```
webgpu-bench/
  llama.cpp/             # Git submodule
  bench.cpp              # C++ wrapper exporting 5 WASM functions
  CMakeLists.txt         # CMake config (JSPI/Asyncify toggle)
  build.sh               # Builds both WASM variants

  harness.html/js        # Browser-side: downloads model, runs inference
  server.js              # Express server with caching proxy + CORS
  runner.js              # Playwright/WebDriverIO orchestrator
  config.js              # Reads models.json, parses CLI args
  models.json            # Model definitions (10 models, 230 variants)
  report.js              # Results aggregation (JSON/CSV)

  scripts/
    submit-results.js    # Prepare results for PR submission
    build-site.js        # Merge machine data into combined.json

  data/machines/         # Committed benchmark results (one file per machine)
  site/                  # Static dashboard (GitHub Pages)
  .github/workflows/     # CI: deploy dashboard on merge
```

### How It Works

1. `build.sh` compiles llama.cpp to WASM with WebGPU support (Emscripten + emdawnwebgpu)
2. `runner.js` starts a local Express server with a HuggingFace caching proxy
3. Playwright launches Chrome/Firefox; WebDriverIO launches real Safari (for actual WebGPU support)
4. Each browser navigates to `harness.html`, which detects JSPI support and loads the correct WASM variant
5. The model is downloaded from HuggingFace (or served from cache) inside the browser
6. Inference runs via WebGPU (or CPU fallback) using llama.cpp's C API
7. Performance metrics from `llama_perf_context()` are exposed to the test runner via `window.__BENCH`
8. Results are aggregated into JSON/CSV files

### Exported WASM Functions

| Function | Description |
|----------|-------------|
| `bench_init()` | Load all GGML backends |
| `bench_load(path, n_ctx, n_gpu_layers)` | Load a GGUF model |
| `bench_run(prompt, n_predict)` | Greedy-decode tokens, return metrics + token IDs as JSON |
| `bench_eval_tokens(prompt, ref_ids_csv)` | Forced-decoding consistency check against CPU reference |
| `bench_exit()` | Free model and context |

All functions use greedy sampling for deterministic output.

### Browser Support

| Browser | Automation | WASM Variant | WebGPU |
|---------|-----------|--------------|--------|
| Chrome | Playwright | JSPI | Yes (via Dawn) |
| Firefox | Playwright | Asyncify | Yes (nightly) / No (stable) |
| Safari | WebDriverIO | Asyncify | Yes (macOS native) |

Safari uses WebDriverIO instead of Playwright to access real Safari with native WebGPU support. Playwright's WebKit engine doesn't support WebGPU.

### Cross-Platform Notes

The framework detects the platform and sets Chromium GPU flags:
- **macOS**: `--use-angle=metal`
- **Linux**: `--use-angle=vulkan`

On Linux with NVIDIA GPU, use `xvfb-run` for headless:
```bash
xvfb-run node runner.js --quick
```

## Updating llama.cpp

```bash
cd llama.cpp
git pull origin master
cd ..
git add llama.cpp
git commit -m "Update llama.cpp submodule"
npm run build
```
