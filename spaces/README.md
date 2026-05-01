---
title: WebGPU Benchmark
emoji: 🧪
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: One-click browser benchmark for GGUF models on WebGPU + WASM
hf_oauth: true
hf_oauth_scopes:
  - write-discussions
---

# WebGPU Benchmark — One Click

Open this page in any WebGPU-capable browser (Chrome, Safari Technology Preview, Firefox with WebGPU flags), pick the GGUF variants that fit on your device, click **Download**, then **Run**.

Results are cached locally in OPFS (no re-downloads on reload). When signed into Hugging Face, you can submit your results as a PR to the leaderboard dataset.

**Source**: this Space is auto-synced from the [main repo](https://github.com/abhijitramesh/webgpu-bench) on every push to `main`.

## What's measured
Per-variant: prefill tokens/sec, decode tokens/sec, wall clock, optional CPU-vs-GPU token agreement.

## Feature caveats
- **Granite 4.0 h-1b** variants need `SSM_SCAN` support in the vendored llama.cpp build — the UI tags the family with a warning badge.
- **Bonsai-1.7B Q1_0** needs `Q1_0` quantization support. The base (non-Q1_0) variant loads regardless.

## Privacy
No data is sent anywhere unless you click **Submit to leaderboard dataset**, which pushes to the dataset configured in `site/js/run/config.js` (`js/run/config.js` on the Space). Models and logs stay in your browser.
