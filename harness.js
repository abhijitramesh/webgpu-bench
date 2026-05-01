// Thin adapter: reads URL params, calls runBenchmarkCore(), writes to
// window.__BENCH so runner.js (Playwright) can poll. Inference logic lives
// in site/js/run/core.js, shared with the interactive Run-tab controller.

import { runBenchmarkCore } from './js/run/core.js';
import { localSource } from './js/run/source.js';
import { CONSISTENCY_PROMPT } from './js/run/config.js';

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
  const modelFile           = params.get('model')         || '';
  const hfRepo              = params.get('hfRepo')        || 'unsloth/Llama-3.2-1B-Instruct-GGUF';
  const consistencyPrompt   = CONSISTENCY_PROMPT;
  const consistencyNPredict = parseInt(params.get('nPredict')   || '128', 10);
  const nPrompt             = parseInt(params.get('nPrompt')    || '512', 10);
  const nGen                = parseInt(params.get('nGen')       || '128', 10);
  const nReps               = parseInt(params.get('nReps')      || '5', 10);
  const nCtx                = parseInt(params.get('nCtx')       || '2048', 10);
  const nGpuLayers          = parseInt(params.get('nGpuLayers') || '999', 10);
  const refTokenIds         = params.get('refTokenIds') || null;
  // mode=perf → skip consistency entirely (e.g. for the GPU perf-only pass).
  // mode=consistency → skip perf (e.g. CPU baseline pass that just needs token_ids).
  // default 'both' runs both phases in one model load.
  const mode                = params.get('mode') || 'both';
  const runConsistency      = mode !== 'perf';
  const runPerf             = mode !== 'consistency';

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
    modelFile, hfRepo,
    consistencyPrompt, consistencyNPredict, refTokenIds,
    runConsistency,
    nPrompt: runPerf ? nPrompt : 0,
    nGen:    runPerf ? nGen    : 0,
    nReps,
    nCtx, nGpuLayers,
    onStatus, onProgress, onLog,
  });

  // Merge core result fields into window.__BENCH. The downloadProgress we wrote
  // during the run is preserved (result has no `downloadProgress` field).
  Object.assign(window.__BENCH, result);
})();
