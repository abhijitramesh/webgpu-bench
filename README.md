# webgpu-bench

Automated benchmarking framework for llama.cpp's WebGPU backend. Compiles llama.cpp to WebAssembly, runs GGUF models in the browser via WebGPU, and measures inference performance across Chromium, Firefox, and Safari using Playwright.

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

# Combine flags
node runner.js --quick --browsers=chromium --no-webgpu
```

## Results

Results are saved to `results/`:

- `results.json` — full benchmark data
- `summary.json` — grouped by browser with pass/fail
- `results.csv` — flat CSV for spreadsheets

Generate reports:
```bash
node report.js
```

### Output Fields

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
```

### How It Works

1. `build.sh` compiles llama.cpp to WASM with WebGPU support (via Emscripten + emdawnwebgpu)
2. `runner.js` launches Playwright browsers and navigates to `harness.html`
3. `harness.js` detects JSPI support, loads the correct WASM variant
4. The model is downloaded from HuggingFace directly in the browser
5. Inference runs via WebGPU (or CPU fallback) using llama.cpp's C API
6. Performance metrics are collected via `llama_perf_context()` and exposed to Playwright
7. Results are aggregated into JSON/CSV

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
