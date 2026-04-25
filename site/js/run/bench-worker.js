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
//     // Exactly one of these — depends on whether the runtime supports
//     // transferable ReadableStreams (most desktops do; iOS Safari and some
//     // mobile Chrome configs don't, in which case the main thread drains
//     // the stream into an ArrayBuffer and transfers the buffer instead):
//     stream?: ReadableStream<Uint8Array>,  // TRANSFERRED
//     buffer?: ArrayBuffer                  // TRANSFERRED (mobile fallback)
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

async function runOne({ params, stream, buffer }) {
  const {
    buildType,
    prompt,
    nPredict,
    nCtx,
    nGpuLayers,
    refTokenIds,
    contentLength,
  } = params;
  if (!stream && !buffer) {
    throw new Error('runOne: exactly one of `stream` or `buffer` must be provided');
  }

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

  // ─── Stream the model into the WASM heap (HeapFS-style) ───
  // Avoid the JS-side MEMFS staging buffer by allocating space inside the
  // WASM heap with _malloc and writing chunks directly via HEAPU8.set. Then
  // register the file with MEMFS using a Uint8Array view backed by the heap
  // region, so llama.cpp's mmap can take the zero-copy branch in MEMFS.mmap
  // (which fires when contents.buffer === HEAP8.buffer).
  //
  // Heap growth during bench_init/bench_load detaches old views, so we
  // override node.contents with a getter that always rebuilds the view
  // from the saved pointer + length against the current Module.HEAPU8.
  if (!(contentLength > 0)) {
    throw new Error('content-length is required for streaming into WASM heap');
  }
  status('downloading', 'Streaming model into WASM heap...');

  let modelPtr = Module._malloc(contentLength);
  if (!modelPtr) {
    throw new Error(
      `_malloc(${(contentLength / (1024 * 1024)).toFixed(0)} MB) failed — wasm heap exhausted`
    );
  }

  try {
    let downloaded = 0;
    if (stream) {
      const reader = stream.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        Module.HEAPU8.set(value, modelPtr + downloaded);
        downloaded += value.length;
        post({ type: 'progress', fraction: downloaded / contentLength, downloaded, total: contentLength });
      }
    } else {
      // Buffered path (mobile fallback): the whole file is already in
      // memory. Copy it into the WASM heap in one shot. Progress was
      // emitted on the main thread while buffering, so we just report 100%
      // here for the loading phase.
      const view = new Uint8Array(buffer);
      if (view.byteLength !== contentLength) {
        log(`warning: buffer size ${view.byteLength} != content-length ${contentLength}`);
      }
      Module.HEAPU8.set(view, modelPtr);
      downloaded = view.byteLength;
      post({ type: 'progress', fraction: 1, downloaded, total: contentLength });
    }
    log(`Model written to WASM heap @ 0x${modelPtr.toString(16)} (${(downloaded / (1024 * 1024)).toFixed(1)} MB)`);

    // Register as a MEMFS file with a heap-backed view. canOwn=true so MEMFS
    // doesn't make its own copy.
    const view = new Uint8Array(Module.HEAPU8.buffer, modelPtr, contentLength);
    Module.FS.createDataFile('/', 'model.gguf', view, true, false, true);

    // Replace contents with a getter — heap growth (e.g. when llama.cpp
    // allocates KV cache) replaces Module.HEAPU8.buffer, which would
    // detach our static view. The getter rebuilds against the live buffer.
    const node = Module.FS.lookupPath('/model.gguf').node;
    Object.defineProperty(node, 'contents', {
      get: () => new Uint8Array(Module.HEAPU8.buffer, modelPtr, contentLength),
      set: () => { /* read-only file */ },
      configurable: true,
    });
    // usedBytes is read by MEMFS for stat() — keep it accurate.
    node.usedBytes = contentLength;
  } catch (err) {
    Module._free(modelPtr);
    modelPtr = 0;
    throw err;
  }

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

  // Drop the MEMFS node — the bytes themselves stay alive in the WASM heap
  // because llama.cpp's mmap captured a pointer into our _malloc'd region.
  // We free that region after bench_exit.
  try {
    Module.FS.unlink('/model.gguf');
  } catch (err) {
    log(`Warning: could not remove model FS node: ${err.message}`);
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

  // Free the heap-resident model bytes now that llama.cpp has unmapped.
  if (modelPtr) {
    Module._free(modelPtr);
    modelPtr = 0;
  }

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
