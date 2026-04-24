// Benchmark core: load GGUF via a source adapter, init llama.cpp WASM,
// run inference, collect metrics. Used by harness.js (URL-param driven, for
// runner.js) and by bench-app.js (UI driven).

const DEFAULT_PROMPT = 'Hello, how are you?';

async function loadBenchScriptOnce(buildType) {
  if (typeof globalThis.createBenchModule === 'function') return;
  const script = document.createElement('script');
  script.src = `/build/${buildType}/bench.js`;
  await new Promise((resolve, reject) => {
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load bench.js (${buildType})`));
    document.head.appendChild(script);
  });
  if (typeof globalThis.createBenchModule !== 'function') {
    throw new Error(`createBenchModule not defined after loading /build/${buildType}/bench.js`);
  }
}

export async function runBenchmarkCore({
  source,
  modelFile,
  hfRepo,
  prompt = DEFAULT_PROMPT,
  nPredict = 128,
  nCtx = 2048,
  nGpuLayers = 999,
  refTokenIds = null,
  onStatus = () => {},
  onProgress = () => {},
  onLog = () => {},
}) {
  if (!source) throw new Error('No source provided (see bench-source.js).');
  if (!modelFile) throw new Error('No model file specified.');
  if (!hfRepo) throw new Error('No hfRepo specified.');

  const hasJspi = 'Suspending' in WebAssembly;
  const buildType = hasJspi ? 'jspi' : 'asyncify';

  const result = {
    status: 'init',
    error: null,
    modelFile,
    buildType,
    webgpuAvailable: !!navigator.gpu,
    gpuAdapterInfo: null,
    metrics: null,
    output: '',
  };

  try {
    // WebGPU adapter probe — informational only.
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          result.gpuAdapterInfo = adapter.info || null;
          onLog(`WebGPU adapter: ${JSON.stringify(adapter.info || 'no info')}`);
        } else {
          onLog('WebGPU: adapter request returned null');
        }
      } catch (e) {
        onLog(`WebGPU adapter error: ${e.message}`);
      }
    } else {
      onLog('WebGPU: not available in this browser');
    }

    // Load the Emscripten glue script once per page.
    onStatus('loading_wasm', `Loading WASM module (${buildType})...`);
    onLog(`JSPI supported: ${hasJspi} — using ${buildType} variant`);
    await loadBenchScriptOnce(buildType);

    const Module = await globalThis.createBenchModule({
      print: (text) => onLog(`[wasm] ${text}`),
      printErr: (text) => onLog(`[wasm:err] ${text}`),
      // Catch Emscripten abort() — Firefox can abort during Asyncify init.
      onAbort: (reason) => {
        const msg = `WASM aborted: ${reason}`;
        result.error = msg;
        result.status = 'error';
        onLog(`ERROR: ${msg}`);
      },
    });
    onLog('WASM module loaded');

    // Download model via the injected source adapter.
    onStatus('downloading', `Downloading ${modelFile}...`);
    onLog(`Fetching model via source: ${hfRepo}/${modelFile}`);
    const { stream, contentLength } = await source.fetchModel(hfRepo, modelFile);
    onLog(`Model size: ${
      contentLength ? `${(contentLength / (1024 * 1024)).toFixed(1)} MB` : 'unknown'
    }`);

    // Stream directly into MEMFS to avoid holding the full model in JS memory.
    // Pre-allocate so MEMFS doesn't realloc on every chunk.
    if (contentLength > 0) {
      Module.FS.writeFile('/model.gguf', new Uint8Array(0));
      Module.FS.truncate('/model.gguf', contentLength);
    }
    const memfsHandle = Module.FS.open('/model.gguf', 'w');
    const reader = stream.getReader();
    let downloaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      Module.FS.write(memfsHandle, value, 0, value.length, downloaded);
      downloaded += value.length;
      const fraction = contentLength ? downloaded / contentLength : 0;
      onProgress(fraction, downloaded, contentLength);
    }
    Module.FS.close(memfsHandle);
    onLog(`Model written to /model.gguf (${(downloaded / (1024 * 1024)).toFixed(1)} MB)`);

    // Init backend.
    onStatus('initializing', 'Initializing llama.cpp backends...');
    onLog('Calling bench_init()...');
    const initResult = await Module.ccall('bench_init', 'number', [], [], { async: true });
    if (initResult !== 0) throw new Error(`bench_init failed: ${initResult}`);
    onLog('Backends initialized');

    // Load model.
    onStatus('loading_model', `Loading model (ctx=${nCtx}, gpu_layers=${nGpuLayers})...`);
    onLog(`Calling bench_load("/model.gguf", ${nCtx}, ${nGpuLayers})...`);
    const loadResult = await Module.ccall(
      'bench_load',
      'number',
      ['string', 'number', 'number'],
      ['/model.gguf', nCtx, nGpuLayers],
      { async: true },
    );
    if (loadResult !== 0) throw new Error(`bench_load failed: ${loadResult}`);
    onLog('Model loaded');

    // llama.cpp has copied the model into WASM heap — free the MEMFS copy.
    try {
      Module.FS.unlink('/model.gguf');
      onLog('Freed model file from virtual FS');
    } catch (e) {
      onLog(`Warning: could not remove model file from FS: ${e.message}`);
    }

    // Run inference.
    onStatus('running', 'Running inference...');
    onLog(`Calling bench_run(prompt, ${nPredict})...`);
    const resultJson = await Module.ccall(
      'bench_run',
      'string',
      ['string', 'number'],
      [prompt, nPredict],
      { async: true },
    );
    onLog(`bench_run returned: ${String(resultJson).substring(0, 200)}`);

    const inferResult = JSON.parse(resultJson);
    if (inferResult.error) throw new Error(`Inference error: ${inferResult.error}`);

    const prefillTokS = inferResult.t_p_eval_ms > 0
      ? (inferResult.n_p_eval / (inferResult.t_p_eval_ms / 1000)).toFixed(2)
      : 'N/A';
    const decodeTokS = inferResult.t_eval_ms > 0
      ? (inferResult.n_eval / (inferResult.t_eval_ms / 1000)).toFixed(2)
      : 'N/A';

    result.metrics = {
      ...inferResult,
      prefill_tok_s: parseFloat(prefillTokS) || 0,
      decode_tok_s: parseFloat(decodeTokS) || 0,
    };
    result.output = inferResult.output || '';

    // Forced-decoding consistency check against a CPU reference token sequence.
    if (refTokenIds && nGpuLayers > 0 && inferResult.token_ids?.length > 0) {
      onLog('Running forced-decoding consistency check...');
      const evalJson = await Module.ccall(
        'bench_eval_tokens',
        'string',
        ['string', 'string'],
        [prompt, refTokenIds],
        { async: true },
      );
      const evalResult = JSON.parse(evalJson);
      if (evalResult.error) {
        onLog(`Consistency check error: ${evalResult.error}`);
      } else {
        result.consistency = evalResult;
        onLog(
          `Consistency: ${(evalResult.agreement_rate * 100).toFixed(1)}% top-1 agreement (` +
          `${evalResult.n_agree}/${evalResult.n_tokens} tokens)`,
        );
        if (evalResult.first_disagreement >= 0) {
          onLog(`First disagreement at token position ${evalResult.first_disagreement}`);
        }
      }
    }

    onLog('Calling bench_exit()...');
    await Module.ccall('bench_exit', null, [], [], { async: true });

    result.status = 'done';
    onStatus('done', `Done! Prefill: ${prefillTokS} tok/s | Decode: ${decodeTokS} tok/s`);
    onLog(
      `Prefill: ${prefillTokS} tok/s (${inferResult.n_p_eval} tokens in ` +
      `${inferResult.t_p_eval_ms.toFixed(0)} ms)`,
    );
    onLog(
      `Decode:  ${decodeTokS} tok/s (${inferResult.n_eval} tokens in ` +
      `${inferResult.t_eval_ms.toFixed(0)} ms)`,
    );
    onLog(`Output: ${(inferResult.output || '').substring(0, 200)}`);
    return result;
  } catch (err) {
    result.error = err.message || String(err);
    result.status = 'error';
    onStatus('error', `Error: ${err.message}`);
    onLog(`ERROR: ${err.message}`);
    if (err.stack) onLog(err.stack);
    return result;
  }
}
