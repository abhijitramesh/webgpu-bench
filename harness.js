// Thin adapter: reads URL params, calls runBenchmarkCore(), writes to
// window.__BENCH so runner.js (Playwright) can poll. Inference logic lives
// in bench-core.js, shared with the interactive bench-app.js page.

import { runBenchmarkCore } from './bench-core.js';
import { localSource } from './bench-source.js';

// Global error handlers — catch Emscripten abort() which may not throw.
window.addEventListener('error', (e) => {
  if (window.__BENCH && window.__BENCH.status !== 'done') {
    window.__BENCH.error = window.__BENCH.error || e.message || 'Uncaught error';
    window.__BENCH.status = 'error';
  }
});
window.addEventListener('unhandledrejection', (e) => {
  if (window.__BENCH && window.__BENCH.status !== 'done') {
    window.__BENCH.error = window.__BENCH.error || String(e.reason) || 'Unhandled rejection';
    window.__BENCH.status = 'error';
  }
});

(async function () {
  const params = new URLSearchParams(window.location.search);
  const modelFile   = params.get('model')        || '';
  const hfRepo      = params.get('hfRepo')       || 'unsloth/Llama-3.2-1B-Instruct-GGUF';
  const prompt      = params.get('prompt')       || 'Hello, how are you?';
  const nPredict    = parseInt(params.get('nPredict')   || '128', 10);
  const nCtx        = parseInt(params.get('nCtx')       || '2048', 10);
  const nGpuLayers  = parseInt(params.get('nGpuLayers') || '999', 10);
  const refTokenIds = params.get('refTokenIds') || null;

  const hasJspi = 'Suspending' in WebAssembly;

  window.__BENCH = {
    status: 'init',
    error: null,
    modelFile,
    buildType: hasJspi ? 'jspi' : 'asyncify',
    webgpuAvailable: !!navigator.gpu,
    gpuAdapterInfo: null,
    downloadProgress: 0,
    metrics: null,
    output: '',
  };

  const statusEl   = document.getElementById('status');
  const progressEl = document.getElementById('progress');
  const logEl      = document.getElementById('log');

  function onStatus(status, msg) {
    window.__BENCH.status = status;
    if (statusEl) {
      statusEl.textContent = msg || status;
      statusEl.className = status === 'error' ? 'err' : status === 'done' ? 'ok' : '';
    }
  }

  function onLog(msg) {
    const line = `[${new Date().toISOString().slice(11, 23)}] ${msg}`;
    console.log(line);
    if (logEl) logEl.textContent += line + '\n';
  }

  function onProgress(fraction, downloaded, total) {
    window.__BENCH.downloadProgress = fraction;
    if (progressEl && total > 0) {
      const pct = (fraction * 100).toFixed(1);
      progressEl.textContent =
        `Downloaded: ${(downloaded / (1024 * 1024)).toFixed(1)} MB / ` +
        `${(total / (1024 * 1024)).toFixed(1)} MB (${pct}%)`;
    }
  }

  const result = await runBenchmarkCore({
    source: localSource(),
    modelFile, hfRepo, prompt, nPredict, nCtx, nGpuLayers, refTokenIds,
    onStatus, onProgress, onLog,
  });

  // Merge core result fields into window.__BENCH. The downloadProgress we wrote
  // during the run is preserved (result has no `downloadProgress` field).
  Object.assign(window.__BENCH, result);
})();
