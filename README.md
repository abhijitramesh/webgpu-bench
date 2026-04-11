# webgpu-bench

Automated benchmarking framework for llama.cpp's WebGPU backend. Compiles llama.cpp to WebAssembly, runs GGUF models in the browser via WebGPU, and measures inference performance and numerical correctness across Chromium, Firefox, and Safari using Playwright.

## Prerequisites

- **Emscripten SDK** (emsdk) installed and accessible. Set `EMSDK_DIR` env var or place at `../emsdk/` relative to this project.
- **Ninja** build system (`brew install ninja` on macOS, `apt install ninja-build` on Linux)
- **Node.js** 18+
- **CMake** 3.14+

## Quick Start

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo-url>
cd webgpu-bench

# 2. Build WASM (downloads emdawnwebgpu automatically)
bash build.sh

# 3. Install Node dependencies + Playwright browsers
npm install
npx playwright install

# 4. Run quick benchmark (3 models, Chromium only)
node runner.js --quick --browsers=chromium

# 5. View results
node report.js
```

## Build

The build script compiles llama.cpp (included as a git submodule) to two WASM variants:

- **JSPI** — for browsers with JavaScript Promise Integration support (Chrome)
- **Asyncify** — fallback for browsers without JSPI (Safari, Firefox)

```bash
bash build.sh
```

Output:
```
build/jspi/bin/bench.js + bench.wasm
build/asyncify/bin/bench.js + bench.wasm
```

The browser automatically detects JSPI support and loads the correct variant.

emdawnwebgpu (WebGPU bindings for Emscripten) is downloaded automatically on first build.

### Build Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMSDK_DIR` | `../emsdk` | Path to Emscripten SDK |

## Running Benchmarks

```bash
# All 27 variants on all 3 browsers
node runner.js

# Quick: Q2_K, Q4_K_M, Q8_0 on all browsers
node runner.js --quick

# Specific browsers
node runner.js --browsers=chromium
node runner.js --browsers=chromium,firefox

# Specific quantization variants
node runner.js --variants=Q4_K_M,Q8_0

# Specific model (when multiple models in models.json)
node runner.js --models=Llama-3.2-1B-Instruct

# CPU-only (disable WebGPU)
node runner.js --no-webgpu

# Consistency check: measure WebGPU numerical correctness vs CPU baseline
node runner.js --quick --consistency
node runner.js --quick --browsers=chromium --consistency

# Combine flags
node runner.js --quick --browsers=chromium --no-webgpu
```

## Results

Results are saved to `results/`:

- `results.json` — full benchmark data
- `summary.json` — grouped by browser with pass/fail
- `results.csv` — flat CSV for spreadsheets
- `cpu_baselines.json` — cached CPU reference token sequences (created by `--consistency`)

Generate reports:
```bash
node report.js
```

### Performance Fields

| Field | Description |
|-------|-------------|
| `prefill_tok_s` | Prompt processing speed (tokens/sec) |
| `decode_tok_s` | Token generation speed (tokens/sec) |
| `t_p_eval_ms` | Prefill time in milliseconds |
| `t_eval_ms` | Decode time in milliseconds |
| `n_p_eval` | Number of prompt tokens processed |
| `n_eval` | Number of tokens generated |
| `buildType` | `jspi` or `asyncify` (which WASM variant was used) |
| `webgpuAvailable` | Whether WebGPU was available in the browser |

### Consistency Fields (with `--consistency`)

| Field | Description |
|-------|-------------|
| `consistency.agreement_rate` | Fraction of token positions where WebGPU and CPU independently agree on the top-1 token (0.0–1.0) |
| `consistency.n_agree` | Number of positions that agreed |
| `consistency.n_tokens` | Total positions evaluated |
| `consistency.first_disagreement` | Token position of first divergence (-1 if perfect agreement) |
| `consistency.matches` | Per-position agreement as a compact 0/1 array |

## Consistency Measurement

The `--consistency` flag measures how faithfully the WebGPU backend reproduces the CPU computation for each quantization type.

### How it works

For each variant being tested, two runs are performed in the same browser:

1. **CPU baseline** (`n_gpu_layers=0`): greedy-decodes 128 tokens and records the token ID sequence. Cached to `results/cpu_baselines.json` so subsequent runs skip this step.
2. **WebGPU run** (`n_gpu_layers=999`): performs the normal benchmark for performance metrics, then runs a **forced-decoding pass** — feeds the CPU's token sequence into the model one token at a time and checks whether the WebGPU backend independently predicts the same top-1 token at each position.

Using the same browser for both runs isolates the WebGPU backend precisely: JSPI builds are compared against JSPI CPU, Asyncify builds against Asyncify CPU.

### Why forced decoding, not text comparison

Naively comparing generated text suffers from **cascading divergence**: a single token difference changes the KV cache context for all subsequent tokens, making the rest of the output statistically unrelated. A text `matchRatio` of 24% might mean only one token actually diverged — the rest is noise.

Forced decoding avoids this entirely. Each of the 128 positions is evaluated independently against the same reference context, giving a clean per-token accuracy signal. This is the same "same top-1" metric used by llama.cpp's `perplexity` tool.

### Interpreting results

| `agreement_rate` | Interpretation |
|---|---|
| `1.00` | WebGPU backend is numerically identical to CPU for this quant — no precision issues |
| `0.95–0.99` | A few tokens differ due to near-equal logits flipping across backends — expected for lower-precision quants, not a bug |
| `< 0.90` | Systematic precision issues — the GPU kernel for this quantization type may need investigation |
| `0.00` | First token wrong — the quantization kernel is likely broken entirely |

Q8_0 typically achieves 100% agreement. Q4_K_M and Q2_K may have a small number of differing positions due to reduced numerical precision, which is expected and consistent with llama.cpp's own tolerance thresholds.

## Dashboard

A static dashboard visualizes benchmark results across machines and browsers. It is deployed to GitHub Pages automatically when results are merged to `main`.

### Viewing Results

Visit the GitHub Pages site for this repo, or preview locally:

```bash
npm run build:site
npx serve site
```

### Submitting Results from Your Machine

```bash
# 1. Run benchmarks
node runner.js --browsers=chromium,firefox

# 2. Prepare results for submission (strips heavy fields, generates machine slug)
npm run submit

# 3. Commit and open a PR
git checkout -b results/my-machine
git add data/machines/
git commit -m "Add benchmark results for <your CPU>"
git push -u origin results/my-machine
gh pr create
```

On merge, GitHub Actions rebuilds the dashboard with the new data.

### How the Data Pipeline Works

1. `npm run submit` reads `results/results.json`, strips large fields (`token_ids`, `output`), and writes a cleaned file to `data/machines/{slug}.json` where the slug is derived from your CPU, RAM, and platform (e.g., `apple-m4-pro-48gb-darwin`).
2. `npm run build:site` merges all machine files into `site/data/combined.json` with flattened metrics and metadata.
3. The static site at `site/` fetches `combined.json` at load time and renders all visualizations client-side.

### Dashboard Features

- **Support Matrix** — Which model/quant/browser combos pass or fail, color-coded
- **Detailed Results** — Full metrics table with decode tok/s, prefill tok/s, eval times, build type, WebGPU status
- **Performance Charts** — Grouped bar charts for decode and prefill throughput, throughput vs model size
- **Machine Comparison** — Side-by-side performance when multiple machines have data
- **Error Analysis** — Failures grouped by category (OOM, WASM Abort, Timeout)
- **Methodology** — How benchmarks work, metrics glossary, consistency measurement explanation

## Adding Models

Edit `models.json` to add new models, repositories, or quantization types.

### Add a new quantization variant to an existing model

Add an entry to the model's `variants` array:

```json
{
  "quant": "Q5_0",
  "filename": "Llama-3.2-1B-Instruct-Q5_0.gguf",
  "sizeMB": 870
}
```

### Add a new model from a different repository

Add a new entry to the `models` array:

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

### Change which variants run with `--quick`

Edit the `quickVariants` array at the bottom of `models.json`:

```json
"quickVariants": ["Q2_K", "Q4_K_M", "Q8_0"]
```

### Finding model filenames

Model files are hosted on HuggingFace. To find exact filenames:
```bash
# List GGUF files in a repo
curl -s https://huggingface.co/api/models/<owner>/<repo> | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('siblings', []):
    if s['rfilename'].endswith('.gguf'):
        print(s['rfilename'])
"
```

## Architecture

```
webgpu-bench/
  llama.cpp/          # Git submodule (llama.cpp master)
  bench.cpp           # Minimal C++ wrapper (4 exported functions)
  CMakeLists.txt      # Build config with JSPI/Asyncify option
  build.sh            # Builds both WASM variants locally

  harness.html/js     # Browser test page (downloads model, runs inference)
  server.js           # Express server with CORS headers
  runner.js           # Playwright orchestrator
  config.js           # Reads models.json, parses CLI args
  models.json         # Model definitions (edit this to add models)
  report.js           # Results aggregation

  scripts/
    submit-results.js # Prepare results for PR submission
    build-site.js     # Merge machine data into combined.json

  data/machines/      # Committed benchmark results (one file per machine)
  site/               # Static dashboard (deployed to GitHub Pages)
  .github/workflows/  # CI: build + deploy dashboard on merge
```

### How It Works

1. `build.sh` compiles llama.cpp to WASM with WebGPU support (via Emscripten + emdawnwebgpu)
2. `runner.js` launches Playwright browsers and navigates to `harness.html`
3. `harness.js` detects JSPI support, loads the correct WASM variant
4. The model is downloaded from HuggingFace directly in the browser
5. Inference runs via WebGPU (or CPU fallback) using llama.cpp's C API
6. Performance metrics are collected via `llama_perf_context()` and exposed to Playwright
7. Results are aggregated into JSON/CSV

### Exported WASM Functions

| Function | Description |
|----------|-------------|
| `bench_init()` | Load all GGML backends |
| `bench_load(path, n_ctx, n_gpu_layers)` | Load a GGUF model |
| `bench_run(prompt, n_predict)` | Greedy-decode `n_predict` tokens, return metrics + token IDs as JSON |
| `bench_eval_tokens(prompt, ref_ids_csv)` | Forced-decoding consistency check against a CPU reference token sequence |
| `bench_exit()` | Free model and context |

All functions use greedy sampling (`llama_sampler_init_greedy`) for deterministic, reproducible output.

### Cross-Platform

The framework detects the platform and adjusts Chromium flags:
- **macOS**: `--use-angle=metal`
- **Linux**: `--use-angle=vulkan`

On Linux with NVIDIA GPU, ensure WebGPU drivers are available and use `xvfb-run` for headless environments:
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
bash build.sh  # Rebuild WASM
```
