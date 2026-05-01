// Thin adapter for runner.js (Playwright). Reads URL params, downloads the
// model into OPFS, hands it to bench-worker.js, and forwards the worker's
// progress/result onto window.__BENCH so the runner can poll. Inference
// orchestration lives in site/js/run/bench-worker.js — same worker the
// interactive Run page uses.

import { localSource } from './js/run/source.js';
import { OPFS_ROOT_NAME } from './js/run/source.js';
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
  const buildType = hasJspi ? 'jspi' : 'asyncify';

  window.__BENCH = {
    status: 'init',
    error: null,
    modelFile,
    buildType,
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

  // Stage 1: download into OPFS on the main thread (sync access handles
  // are worker-only, but the downloading half runs fine here).
  let size;
  try {
    onStatus('downloading', `Downloading ${modelFile}...`);
    onLog(`Fetching ${hfRepo}/${modelFile} into OPFS`);
    const r = await localSource().opfsHandleForModel(hfRepo, modelFile, onProgress);
    size = r.size;
  } catch (err) {
    window.__BENCH.error = `opfsHandleForModel failed: ${err.message}`;
    window.__BENCH.status = 'error';
    onStatus('error', window.__BENCH.error);
    onLog(`ERROR: ${window.__BENCH.error}`);
    return;
  }

  // Stage 2: hand the OPFS layout key to the worker. The worker re-resolves
  // the FileHandle locally (FileHandles don't structured-clone reliably on
  // iOS Safari) and opens a sync access handle inside its own thread.
  const result = await new Promise((resolve) => {
    let worker;
    try {
      worker = new Worker(new URL('./js/run/bench-worker.js', import.meta.url));
    } catch (err) {
      resolve({ status: 'error', error: `worker construct failed: ${err.message}` });
      return;
    }

    let settled = false;
    const finish = (record) => {
      if (settled) return;
      settled = true;
      try { worker.terminate(); } catch { /* noop */ }
      resolve(record);
    };

    worker.onmessage = (e) => {
      const msg = e.data || {};
      if (msg.type === 'status') onStatus(msg.status, msg.msg);
      else if (msg.type === 'progress') onProgress(msg.fraction, msg.downloaded, msg.total);
      else if (msg.type === 'log') onLog(msg.line);
      else if (msg.type === 'result') finish(msg.record);
    };
    worker.onerror = (err) => {
      finish({ status: 'error', error: err?.message || 'worker error' });
    };
    worker.onmessageerror = () => {
      finish({ status: 'error', error: 'worker message deserialization failed' });
    };

    worker.postMessage({
      type: 'run',
      params: {
        buildType,
        nCtx,
        nGpuLayers,
        consistencyPrompt: runConsistency ? consistencyPrompt : '',
        consistencyNPredict,
        refTokenIds,
        nPrompt: runPerf ? nPrompt : 0,
        nGen:    runPerf ? nGen    : 0,
        nReps,
        noWarmup: false,
      },
      opfsPath: { rootDir: OPFS_ROOT_NAME, repo: hfRepo, filename: modelFile },
    });
  });

  // Merge worker result into window.__BENCH. downloadProgress was set
  // during stage 1 and is preserved.
  Object.assign(window.__BENCH, result);
  window.__BENCH._opfsSize = size;
})();
