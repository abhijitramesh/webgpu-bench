// Dedicated Worker that runs a single llama.cpp inference pass. Loaded by
// controller.js with { type: 'classic' } so we can importScripts() the
// Emscripten-emitted bench.js (which is a classic, non-module script).
//
// Protocol (all messages use { type, ... } tag):
//
//   main → worker: {
//     type: 'run',
//     params: { buildType, prompt, nPredict, nCtx, nGpuLayers, refTokenIds,
//               contentLength },
//     stream: ReadableStream<Uint8Array>  // TRANSFERRED via postMessage
//   }
//
//   worker → main: { type: 'status', status, msg }
//   worker → main: { type: 'progress', fraction, downloaded, total }
//   worker → main: { type: 'log', line }
//   worker → main: { type: 'result', record }   // terminal
//
// Abort: main thread calls worker.terminate(). No cooperative abort — JSPI
// decode loops ignore signals, and termination is the only reliable way to
// stop an in-flight WASM call.
//
// NOTE ON DUPLICATION: the orchestration below mirrors runBenchmarkCore()
// in site/js/run/core.js. core.js stays the authoritative main-thread path
// (used by harness.js + runner.js Playwright harness). When changing one,
// change the other.

const post = (msg) => self.postMessage(msg);
const log = (line) => post({ type: 'log', line });
const status = (s, msg) => post({ type: 'status', status: s, msg });

self.onmessage = async (e) => {
  const { type } = e.data || {};
  if (type !== 'run') {
    log(`worker: ignoring unknown message type "${type}"`);
    return;
  }
  try {
    const record = await runOne(e.data);
    post({ type: 'result', record });
  } catch (err) {
    post({
      type: 'result',
      record: {
        status: 'error',
        error: err?.message || String(err),
        metrics: null,
      },
    });
  }
};

async function runOne({ params, stream }) {
  const {
    buildType,
    prompt,
    nPredict,
    nCtx,
    nGpuLayers,
    refTokenIds,
    contentLength,
  } = params;

  const result = {
    status: 'init',
    error: null,
    buildType,
    webgpuAvailable: !!self.navigator?.gpu,
    gpuAdapterInfo: null,
    metrics: null,
    consistency: null,
    output: '',
  };

  // ─── WebGPU adapter probe (informational) ───
  if (self.navigator?.gpu) {
    try {
      const adapter = await self.navigator.gpu.requestAdapter();
      if (adapter) {
        // GPUAdapterInfo is a host object — structured-clone can't serialize
        // it across postMessage. Copy the fields we care about into a plain
        // object before storing on result.
        const info = adapter.info;
        result.gpuAdapterInfo = info ? {
          vendor: info.vendor || '',
          architecture: info.architecture || '',
          device: info.device || '',
          description: info.description || '',
        } : null;
        log(`WebGPU adapter: ${JSON.stringify(result.gpuAdapterInfo || 'no info')}`);
      } else {
        log('WebGPU: adapter request returned null');
      }
    } catch (err) {
      log(`WebGPU adapter error: ${err.message}`);
    }
  } else {
    log('WebGPU: not available in this worker');
  }

  // ─── Load the Emscripten glue once per worker ───
  status('loading_wasm', `Loading WASM module (${buildType})...`);
  try {
    self.importScripts(`/build/${buildType}/bench.js`);
  } catch (err) {
    throw new Error(`importScripts /build/${buildType}/bench.js failed: ${err.message}`);
  }
  if (typeof self.createBenchModule !== 'function') {
    throw new Error('createBenchModule not defined after importScripts');
  }

  const Module = await self.createBenchModule({
    // In a worker loaded via importScripts(), Emscripten can't infer the
    // script's directory and falls back to self.location (this worker's
    // own URL), which makes it look for bench.wasm next to bench-worker.js.
    // Pin the lookup to the build directory so it grabs the right file.
    locateFile: (filename) => `/build/${buildType}/${filename}`,
    print: (text) => log(`[wasm] ${text}`),
    printErr: (text) => log(`[wasm:err] ${text}`),
    onAbort: (reason) => {
      const msg = `WASM aborted: ${reason}`;
      result.error = msg;
      result.status = 'error';
      log(`ERROR: ${msg}`);
    },
  });
  log('WASM module loaded');

  // ─── Stream the model into MEMFS ───
  status('downloading', 'Streaming model into WASM FS...');
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
    post({ type: 'progress', fraction, downloaded, total: contentLength });
  }
  Module.FS.close(memfsHandle);
  log(`Model written to /model.gguf (${(downloaded / (1024 * 1024)).toFixed(1)} MB)`);

  // ─── Init backend ───
  status('initializing', 'Initializing llama.cpp backends...');
  const initResult = await Module.ccall('bench_init', 'number', [], [], { async: true });
  if (initResult !== 0) throw new Error(`bench_init failed: ${initResult}`);
  log('Backends initialized');

  // ─── Load model ───
  status('loading_model', `Loading model (ctx=${nCtx}, gpu_layers=${nGpuLayers})...`);
  const loadResult = await Module.ccall(
    'bench_load',
    'number',
    ['string', 'number', 'number'],
    ['/model.gguf', nCtx, nGpuLayers],
    { async: true },
  );
  if (loadResult !== 0) throw new Error(`bench_load failed: ${loadResult}`);
  log('Model loaded');

  // Free MEMFS copy — llama.cpp has mapped weights into its own heap by now.
  try {
    Module.FS.unlink('/model.gguf');
    log('Freed model file from virtual FS');
  } catch (err) {
    log(`Warning: could not remove model file from FS: ${err.message}`);
  }

  // ─── Inference ───
  status('running', 'Running inference...');
  const resultJson = await Module.ccall(
    'bench_run',
    'string',
    ['string', 'number'],
    [prompt, nPredict],
    { async: true },
  );
  log(`bench_run returned: ${String(resultJson).substring(0, 200)}`);

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

  // ─── Consistency check ───
  if (refTokenIds && nGpuLayers > 0 && inferResult.token_ids?.length > 0) {
    log('Running forced-decoding consistency check...');
    const evalJson = await Module.ccall(
      'bench_eval_tokens',
      'string',
      ['string', 'string'],
      [prompt, refTokenIds],
      { async: true },
    );
    const evalResult = JSON.parse(evalJson);
    if (evalResult.error) {
      log(`Consistency check error: ${evalResult.error}`);
    } else {
      result.consistency = evalResult;
      log(
        `Consistency: ${(evalResult.agreement_rate * 100).toFixed(1)}% top-1 agreement (` +
        `${evalResult.n_agree}/${evalResult.n_tokens} tokens)`,
      );
      if (evalResult.first_disagreement >= 0) {
        log(`First disagreement at token position ${evalResult.first_disagreement}`);
      }
    }
  }

  await Module.ccall('bench_exit', null, [], [], { async: true });

  result.status = 'done';
  status('done', `Done! Prefill: ${prefillTokS} tok/s | Decode: ${decodeTokS} tok/s`);
  log(
    `Prefill: ${prefillTokS} tok/s (${inferResult.n_p_eval} tokens in ` +
    `${inferResult.t_p_eval_ms.toFixed(0)} ms)`,
  );
  log(
    `Decode:  ${decodeTokS} tok/s (${inferResult.n_eval} tokens in ` +
    `${inferResult.t_eval_ms.toFixed(0)} ms)`,
  );
  log(`Output: ${(inferResult.output || '').substring(0, 200)}`);
  return result;
}
