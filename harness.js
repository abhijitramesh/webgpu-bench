// Browser-side benchmark harness.
// Downloads a GGUF model from HuggingFace, runs inference via our WASM module,
// and exposes results on window.__BENCH for Playwright to read.

// Global error handlers — catch Emscripten abort() which may not throw
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

(async function() {
  // Read config from URL params
  const params = new URLSearchParams(window.location.search);
  const modelFile   = params.get('model')       || '';
  const hfRepo      = params.get('hfRepo')      || 'unsloth/Llama-3.2-1B-Instruct-GGUF';
  const prompt       = params.get('prompt')      || 'Hello, how are you?';
  const nPredict    = parseInt(params.get('nPredict') || '128', 10);
  const nCtx        = parseInt(params.get('nCtx')     || '2048', 10);
  const nGpuLayers  = parseInt(params.get('nGpuLayers') || '999', 10);

  // Detect JSPI support (following wllama's pattern: src/wllama.ts:602)
  const hasJspi = 'Suspending' in WebAssembly;

  // State exposed to Playwright
  window.__BENCH = {
    status: 'init',
    error: null,
    modelFile: modelFile,
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

  function setStatus(status, msg) {
    window.__BENCH.status = status;
    statusEl.textContent = msg || status;
    statusEl.className = status === 'error' ? 'err' : status === 'done' ? 'ok' : '';
  }

  function log(msg) {
    const line = `[${new Date().toISOString().slice(11, 23)}] ${msg}`;
    console.log(line);
    if (logEl) logEl.textContent += line + '\n';
  }

  try {
    if (!modelFile) {
      throw new Error('No model file specified. Use ?model=filename.gguf');
    }

    // Check WebGPU
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          window.__BENCH.gpuAdapterInfo = adapter.info || null;
          log(`WebGPU adapter: ${JSON.stringify(adapter.info || 'no info')}`);
        } else {
          log('WebGPU: adapter request returned null');
        }
      } catch (e) {
        log(`WebGPU adapter error: ${e.message}`);
      }
    } else {
      log('WebGPU: not available in this browser');
    }

    // Load WASM module — pick JSPI or Asyncify variant based on browser support
    const buildVariant = hasJspi ? 'jspi' : 'asyncify';
    setStatus('loading_wasm', `Loading WASM module (${buildVariant})...`);
    log(`JSPI supported: ${hasJspi} — using ${buildVariant} variant`);

    const script = document.createElement('script');
    script.src = `/build/${buildVariant}/bench.js`;
    await new Promise((resolve, reject) => {
      script.onload = resolve;
      script.onerror = () => reject(new Error(`Failed to load bench.js (${buildVariant})`));
      document.head.appendChild(script);
    });

    const Module = await createBenchModule({
      print: (text) => log(`[wasm] ${text}`),
      printErr: (text) => log(`[wasm:err] ${text}`),
      // Catch Emscripten abort() — Firefox can abort during Asyncify init
      onAbort: (reason) => {
        const msg = `WASM aborted: ${reason}`;
        window.__BENCH.error = msg;
        window.__BENCH.status = 'error';
        log(`ERROR: ${msg}`);
      },
    });
    log('WASM module loaded');

    // Download model from HuggingFace
    setStatus('downloading', `Downloading ${modelFile}...`);
    log(`Downloading: https://huggingface.co/${hfRepo}/resolve/main/${modelFile}`);

    const modelUrl = `https://huggingface.co/${hfRepo}/resolve/main/${modelFile}`;
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`Download failed: ${response.status} ${response.statusText}`);
    }

    const contentLength = parseInt(response.headers.get('content-length') || '0', 10);
    log(`Model size: ${(contentLength / (1024 * 1024)).toFixed(1)} MB`);

    // Stream download with progress
    const reader = response.body.getReader();
    const chunks = [];
    let downloaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      downloaded += value.length;
      const pct = contentLength ? ((downloaded / contentLength) * 100).toFixed(1) : '?';
      window.__BENCH.downloadProgress = contentLength ? downloaded / contentLength : 0;
      progressEl.textContent = `Downloaded: ${(downloaded / (1024 * 1024)).toFixed(1)} MB / ${(contentLength / (1024 * 1024)).toFixed(1)} MB (${pct}%)`;
    }

    // Concatenate chunks and write to Emscripten FS
    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
    const modelData = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      modelData.set(chunk, offset);
      offset += chunk.length;
    }

    log(`Writing ${(totalLength / (1024 * 1024)).toFixed(1)} MB to virtual FS...`);
    Module.FS.writeFile('/model.gguf', modelData);
    log('Model written to /model.gguf');

    // Initialize backend
    // Note: {async: true} is required for JSPI-exported functions
    setStatus('initializing', 'Initializing llama.cpp backends...');
    log('Calling bench_init()...');
    const initResult = await Module.ccall('bench_init', 'number', [], [], {async: true});
    if (initResult !== 0) {
      throw new Error(`bench_init failed: ${initResult}`);
    }
    log('Backends initialized');

    // Load model
    setStatus('loading_model', `Loading model (ctx=${nCtx}, gpu_layers=${nGpuLayers})...`);
    log(`Calling bench_load("/model.gguf", ${nCtx}, ${nGpuLayers})...`);
    const loadResult = await Module.ccall('bench_load', 'number',
      ['string', 'number', 'number'],
      ['/model.gguf', nCtx, nGpuLayers],
      {async: true}
    );
    if (loadResult !== 0) {
      throw new Error(`bench_load failed: ${loadResult}`);
    }
    log('Model loaded');

    // Run inference
    setStatus('running', 'Running inference...');
    log(`Calling bench_run(prompt, ${nPredict})...`);
    const resultJson = await Module.ccall('bench_run', 'string',
      ['string', 'number'],
      [prompt, nPredict],
      {async: true}
    );
    log(`bench_run returned: ${String(resultJson).substring(0, 200)}`);

    const result = JSON.parse(resultJson);
    if (result.error) {
      throw new Error(`Inference error: ${result.error}`);
    }

    // Compute derived metrics
    const prefillTokS = result.t_p_eval_ms > 0
      ? (result.n_p_eval / (result.t_p_eval_ms / 1000)).toFixed(2)
      : 'N/A';
    const decodeTokS = result.t_eval_ms > 0
      ? (result.n_eval / (result.t_eval_ms / 1000)).toFixed(2)
      : 'N/A';

    window.__BENCH.metrics = {
      ...result,
      prefill_tok_s: parseFloat(prefillTokS) || 0,
      decode_tok_s: parseFloat(decodeTokS) || 0,
    };
    window.__BENCH.output = result.output || '';

    // Cleanup
    log('Calling bench_exit()...');
    await Module.ccall('bench_exit', null, [], [], {async: true});

    setStatus('done', `Done! Prefill: ${prefillTokS} tok/s | Decode: ${decodeTokS} tok/s`);
    log(`Prefill: ${prefillTokS} tok/s (${result.n_p_eval} tokens in ${result.t_p_eval_ms.toFixed(0)} ms)`);
    log(`Decode:  ${decodeTokS} tok/s (${result.n_eval} tokens in ${result.t_eval_ms.toFixed(0)} ms)`);
    log(`Output: ${(result.output || '').substring(0, 200)}`);

  } catch (err) {
    window.__BENCH.error = err.message || String(err);
    setStatus('error', `Error: ${err.message}`);
    log(`ERROR: ${err.message}`);
    log(err.stack || '');
  }
})();
