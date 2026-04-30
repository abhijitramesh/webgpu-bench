// Dedicated Worker that runs a single llama.cpp inference pass. Loaded by
// controller.js with { type: 'classic' } so we can importScripts() the
// Emscripten-emitted bench.js (which is a classic, non-module script).
//
// Protocol (all messages use { type, ... } tag):
//
//   main → worker: {
//     type: 'run',
//     params: {
//       buildType, contentLength,
//       // model load
//       nCtx, nGpuLayers,
//       // consistency phase (set consistencyPrompt to '' to skip)
//       consistencyPrompt, consistencyNPredict, refTokenIds,
//       // perf phase
//       nPrompt, nGen, nReps, noWarmup,
//     },
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
// NOTE ON DUPLICATION: the orchestration below mirrors runBenchmarkCore() +
// runBenchActions() in site/js/run/core.js. core.js stays the authoritative
// main-thread path (used by harness.js + runner.js Playwright harness). When
// changing one, change the other.

const post = (msg) => self.postMessage(msg);
const log = (line) => post({ type: 'log', line });
const status = (s, msg) => post({ type: 'status', status: s, msg });

// Below this many compared tokens, the consistency agreement rate is
// statistical noise (e.g. early-EOS models that produce 1 token always
// report 100%). Mirror of CONSISTENCY_MIN_TOKENS in core.js.
const CONSISTENCY_MIN_TOKENS = 8;

// llama.cpp/ggml emit info, warnings, AND errors all to stderr. Tag only the
// actually-bad lines as :err so real failures stand out. Mirror in core.js.
function classifyWasmStderr(text) {
  return /\b(error|abort(ed)?|failed|fatal|panic|assert)\b|GGML_ASSERT/i.test(text)
    ? '[wasm:err]' : '[wasm]';
}

// ─── OPFS-backed model loading (wllama-style) ───
// For >2GB GGUFs we can't put the whole file on the WASM heap (TypedArray
// length limits, and it eats the heap budget that KV cache + working memory
// need). Instead, we open a FileSystemSyncAccessHandle on the OPFS file in
// this worker, register a zero-byte stub in MEMFS, and patch MEMFS's
// stream_ops so reads delegate to syncHandle.read(). llama.cpp then loads
// the model via fread (use_mmap=false), which calls the patched stream_ops
// — never copying the bytes through the WASM heap.
//
// Mirrors wllama's src/workers-code/llama-cpp.js (patchMEMFS / opfsAlloc /
// opfsFreeAll). Worker-only: sync access handles aren't available on the
// main thread.

const opfsHandles = {}; // map MEMFS-name → { syncHandle, size }

function patchMEMFS(Module) {
  const m = Module;
  // Idempotent — only install the patches once per Module.
  if (m.MEMFS.stream_ops._read) return;
  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;
  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;
  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;

  m.MEMFS.stream_ops.read = function (stream, buffer, offset, length, position) {
    const name = stream.node.name;
    if (opfsHandles[name]) {
      const { syncHandle, size } = opfsHandles[name];
      const toRead = Math.min(length, size - position);
      if (toRead <= 0) return 0;
      const view = new Uint8Array(buffer.buffer, buffer.byteOffset + offset, toRead);
      return syncHandle.read(view, { at: position });
    }
    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);
  };
  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;

  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {
    const name = stream.node.name;
    if (opfsHandles[name]) {
      const { size } = opfsHandles[name];
      let newPos = offset;
      if (whence === 1) newPos += stream.position;  // SEEK_CUR
      if (whence === 2) newPos += size;             // SEEK_END
      if (newPos < 0) throw new Error('SEEK before start of file');
      stream.position = newPos;
      return newPos;
    }
    return m.MEMFS.stream_ops._llseek(stream, offset, whence);
  };
  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;

  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {
    const name = stream.node.name;
    if (opfsHandles[name]) {
      // OPFS-backed files must never be mmap'd — that would force MEMFS to
      // copy the file into the WASM heap, defeating the OPFS path. The C++
      // side passes use_mmap=0 to avoid this. If we ever land here, the
      // caller forgot to disable mmap.
      throw new Error(`[OPFS] mmap called on "${name}" — bench_load was not invoked with use_mmap=0`);
    }
    return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);
  };
  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;
}

// Resolve an OPFS path (rootDir + repo segments + filename) to a
// FileSystemFileHandle inside this worker. Works around the iOS Safari
// limitation that FileSystemFileHandle isn't structured-cloneable across
// postMessage — main thread sends the layout key, worker opens the
// handle locally.
async function resolveOpfsHandle({ rootDir, repo, filename }) {
  if (!self.navigator?.storage?.getDirectory) {
    throw new Error('OPFS not available in this worker');
  }
  let dir = await self.navigator.storage.getDirectory();
  dir = await dir.getDirectoryHandle(rootDir, { create: false });
  for (const seg of String(repo).split('/').filter(Boolean)) {
    dir = await dir.getDirectoryHandle(seg, { create: false });
  }
  return dir.getFileHandle(filename, { create: false });
}

async function opfsAlloc(Module, name, fileHandle) {
  // createSyncAccessHandle is worker-only and exclusive — only one writer
  // per OPFS file at a time. Caller must ensure no createWritable session
  // is open when we land here.
  const syncHandle = await fileHandle.createSyncAccessHandle();
  const size = syncHandle.getSize();
  opfsHandles[name] = { syncHandle, size };
  // Zero-byte placeholder so llama.cpp's fopen() finds the path.
  Module.FS.createDataFile('/', name, new Uint8Array(0), true, false, true);
  // Set usedBytes so fstat()/seek-end report the real file size — our
  // patched llseek consults size, but other code (e.g. llama.cpp's GGUF
  // reader sanity-checking the file length) goes through stat first.
  Module.FS.lookupPath('/' + name).node.usedBytes = size;
  return size;
}

function opfsFreeAll(Module) {
  for (const [name, { syncHandle }] of Object.entries(opfsHandles)) {
    try { syncHandle.close(); } catch { /* already closed */ }
    try { Module.FS.unlink('/' + name); } catch { /* already gone */ }
    delete opfsHandles[name];
  }
}

// Aggregate raw nanosecond samples into the llama-bench result shape.
// Mirrors core.js buildTest — keep them identical.
function buildTest(name, n_prompt, n_gen, samples_ns) {
  const n = samples_ns.length;
  if (n === 0) {
    return { name, n_prompt, n_gen, avg_ns: 0, stddev_ns: 0, avg_ts: 0, stddev_ts: 0, samples_ns: [], samples_ts: [] };
  }
  const avg_ns = samples_ns.reduce((a, b) => a + b, 0) / n;
  const var_ns = n > 1
    ? samples_ns.reduce((a, b) => a + (b - avg_ns) * (b - avg_ns), 0) / (n - 1)
    : 0;
  const stddev_ns = Math.sqrt(var_ns);
  const n_tokens = n_prompt + n_gen;
  const samples_ts = samples_ns.map(t => t > 0 ? (1e9 * n_tokens) / t : 0);
  const avg_ts = samples_ts.reduce((a, b) => a + b, 0) / n;
  const var_ts = n > 1
    ? samples_ts.reduce((a, b) => a + (b - avg_ts) * (b - avg_ts), 0) / (n - 1)
    : 0;
  const stddev_ts = Math.sqrt(var_ts);
  const round2 = x => Math.round(x * 100) / 100;
  return {
    name,
    n_prompt,
    n_gen,
    avg_ns: Math.round(avg_ns),
    stddev_ns: Math.round(stddev_ns),
    avg_ts: round2(avg_ts),
    stddev_ts: round2(stddev_ts),
    samples_ns: samples_ns.map(Math.round),
    samples_ts: samples_ts.map(round2),
  };
}

function parseBenchResult(label, raw) {
  let r;
  try { r = JSON.parse(raw); } catch (e) {
    throw new Error(`${label}: invalid JSON from C (${e.message})`);
  }
  if (r.error) throw new Error(`${label}: ${r.error}`);
  return r;
}

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

async function runOne({ params, stream, buffer, opfsPath }) {
  const {
    buildType,
    contentLength,
    nCtx,
    nGpuLayers,
    // consistency
    consistencyPrompt,
    consistencyNPredict,
    refTokenIds,
    // perf
    nPrompt,
    nGen,
    nReps,
    noWarmup,
  } = params;
  // Three input modes are supported:
  //   opfsPath    → wllama-style OPFS-streaming load (preferred for >2GB).
  //                  Resolved to a FileSystemFileHandle inside the worker
  //                  via navigator.storage.getDirectory() — FileHandles
  //                  themselves don't structured-clone reliably (iOS Safari).
  //   stream      → heap-stream mode (zero-copy WASM-heap, transferable)
  //   buffer      → buffered fallback for browsers without transferable streams
  // Exactly one must be provided.
  const inputCount = (opfsPath ? 1 : 0) + (stream ? 1 : 0) + (buffer ? 1 : 0);
  if (inputCount !== 1) {
    throw new Error('runOne: exactly one of `opfsPath`, `stream`, or `buffer` must be provided');
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
    locateFile: (filename) => `/build/${buildType}/${filename}`,
    print: (text) => log(`[wasm] ${text}`),
    printErr: (text) => log(`${classifyWasmStderr(text)} ${text}`),
    onAbort: (reason) => {
      const msg = `WASM aborted: ${reason}`;
      result.error = msg;
      result.status = 'error';
      log(`ERROR: ${msg}`);
    },
  });
  log('WASM module loaded');

  // ─── Make the model visible to the WASM filesystem ───
  // Two paths:
  //   useOpfsPath: leave the bytes on disk (OPFS) and route reads through
  //               a sync access handle via patched MEMFS stream_ops. No
  //               heap copy, supports >2GB.
  //   else:       _malloc the full file on the WASM heap, write the stream
  //               in, register a heap-backed MEMFS file. Faster (mmap'd
  //               zero-copy at load time) but caps at ~2GB.
  let modelPtr = 0;  // tracks heap-path allocation for cleanup
  const useOpfsPath = !!opfsPath;

  if (useOpfsPath) {
    status('opfs', 'Linking OPFS-backed model into MEMFS...');
    const fileHandle = await resolveOpfsHandle(opfsPath);
    patchMEMFS(Module);
    const size = await opfsAlloc(Module, 'model.gguf', fileHandle);
    log(`OPFS-backed model.gguf registered (${(size / (1024 * 1024)).toFixed(1)} MB)`);
    // Report 100% to keep the existing progress UI happy — the actual
    // download to OPFS happened before the worker spawn.
    post({ type: 'progress', fraction: 1, downloaded: size, total: size });
  } else {
    if (!(contentLength > 0)) {
      throw new Error('content-length is required for streaming into WASM heap');
    }
    status('downloading', 'Streaming model into WASM heap...');

    modelPtr = Module._malloc(contentLength);
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
        const view = new Uint8Array(buffer);
        if (view.byteLength !== contentLength) {
          log(`warning: buffer size ${view.byteLength} != content-length ${contentLength}`);
        }
        Module.HEAPU8.set(view, modelPtr);
        downloaded = view.byteLength;
        post({ type: 'progress', fraction: 1, downloaded, total: contentLength });
      }
      log(`Model written to WASM heap @ 0x${modelPtr.toString(16)} (${(downloaded / (1024 * 1024)).toFixed(1)} MB)`);

      const view = new Uint8Array(Module.HEAPU8.buffer, modelPtr, contentLength);
      Module.FS.createDataFile('/', 'model.gguf', view, true, false, true);

      const node = Module.FS.lookupPath('/model.gguf').node;
      Object.defineProperty(node, 'contents', {
        get: () => new Uint8Array(Module.HEAPU8.buffer, modelPtr, contentLength),
        set: () => { /* read-only file */ },
        configurable: true,
      });
      node.usedBytes = contentLength;
    } catch (err) {
      Module._free(modelPtr);
      modelPtr = 0;
      throw err;
    }
  }

  // ─── Init backend ───
  status('initializing', 'Initializing llama.cpp backends...');
  const initResult = await Module.ccall('bench_init', 'number', [], [], { async: true });
  if (initResult !== 0) throw new Error(`bench_init failed: ${initResult}`);
  log('Backends initialized');

  // ─── Load model ───
  // OPFS path requires use_mmap=0 — the patched mmap throws to surface bugs
  // if it's accidentally invoked. Heap path uses mmap=1 to take MEMFS's
  // zero-copy mmap fast path against our HEAPU8-backed file.
  const useMmap = useOpfsPath ? 0 : 1;
  status('loading_model', `Loading model (ctx=${nCtx}, gpu_layers=${nGpuLayers}, mmap=${useMmap})...`);
  const loadResult = await Module.ccall(
    'bench_load',
    'number',
    ['string', 'number', 'number', 'number'],
    ['/model.gguf', nCtx, nGpuLayers, useMmap],
    { async: true },
  );
  if (loadResult !== 0) throw new Error(`bench_load failed: ${loadResult}`);
  log('Model loaded');

  if (!useOpfsPath) {
    // Heap path: drop the MEMFS node now that llama.cpp's mmap captured a
    // pointer into our _malloc'd region. Bytes stay alive in the heap until
    // bench_exit + _free.
    try {
      Module.FS.unlink('/model.gguf');
    } catch (err) {
      log(`Warning: could not remove model FS node: ${err.message}`);
    }
  }

  // ─── Consistency phase ───
  // Soft-fail: a failure here logs and falls through to the perf phase
  // rather than aborting the whole run. Some devices/models can't survive
  // bench_run (e.g. unsupported op, OOM mid-decode) but can still produce
  // useful pp/tg numbers via synthetic-token paths.
  if (consistencyPrompt) {
    try {
      status('consistency', 'Running consistency check...');
      log(`bench_run("...", ${consistencyNPredict}) — consistency phase`);
      const raw = await Module.ccall(
        'bench_run', 'string',
        ['string', 'number'],
        [consistencyPrompt, consistencyNPredict],
        { async: true },
      );
      const r = parseBenchResult('bench_run', raw);
      result.output = r.output || '';
      result.consistency = { token_ids: r.token_ids || [] };

      if (refTokenIds) {
        log('bench_eval_tokens — forced-decode vs CPU baseline');
        const evalRaw = await Module.ccall(
          'bench_eval_tokens', 'string',
          ['string', 'string'],
          [consistencyPrompt, refTokenIds],
          { async: true },
        );
        const ev = parseBenchResult('bench_eval_tokens', evalRaw);
        result.consistency = { ...result.consistency, ...ev };
        if (ev.n_tokens < CONSISTENCY_MIN_TOKENS) {
          log(
            `Consistency: insufficient samples (${ev.n_tokens} token` +
            `${ev.n_tokens === 1 ? '' : 's'} before EOS) — agreement rate not meaningful`
          );
        } else {
          log(
            `Consistency: ${(ev.agreement_rate * 100).toFixed(1)}% top-1 agreement (` +
            `${ev.n_agree}/${ev.n_tokens})` +
            (ev.first_disagreement >= 0 ? ` — first diverge @ ${ev.first_disagreement}` : '')
          );
        }
      }
    } catch (err) {
      log(`Consistency phase failed: ${err.message} — continuing to perf phase`);
    }
  }

  // ─── Perf phase (llama-bench style) ───
  // Each test (pp, tg) is wrapped independently so a failure in one doesn't
  // skip the other. Empty samples_ns produces a buildTest with avg_ts=0,
  // which the dashboard renders as a dash.
  const wantPp = nPrompt > 0;
  const wantTg = nGen > 0;
  if (wantPp || wantTg) {
    const tests = [];

    if (wantPp) {
      try {
        if (!noWarmup) {
          status('perf', `warmup pp${nPrompt}`);
          log(`bench_pp(${nPrompt}) — warmup`);
          const raw = await Module.ccall('bench_pp', 'string', ['number'], [nPrompt], { async: true });
          parseBenchResult('bench_pp warmup', raw);
        }
        const samples_ns = [];
        for (let i = 0; i < nReps; i++) {
          status('perf', `pp${nPrompt} ${i + 1}/${nReps}`);
          const t0 = performance.now();
          const raw = await Module.ccall('bench_pp', 'string', ['number'], [nPrompt], { async: true });
          const t_ns = (performance.now() - t0) * 1e6;
          parseBenchResult('bench_pp', raw);
          samples_ns.push(t_ns);
          log(`pp${nPrompt} run ${i + 1}/${nReps}: ${(t_ns / 1e6).toFixed(1)} ms (${(1e9 * nPrompt / t_ns).toFixed(1)} t/s)`);
        }
        tests.push(buildTest(`pp${nPrompt}`, nPrompt, 0, samples_ns));
      } catch (err) {
        log(`pp test failed: ${err.message}`);
      }
    }

    if (wantTg) {
      try {
        if (!noWarmup) {
          status('perf', `warmup tg`);
          log('bench_tg(1) — warmup');
          const raw = await Module.ccall('bench_tg', 'string', ['number'], [1], { async: true });
          parseBenchResult('bench_tg warmup', raw);
        }
        const samples_ns = [];
        for (let i = 0; i < nReps; i++) {
          status('perf', `tg${nGen} ${i + 1}/${nReps}`);
          const t0 = performance.now();
          const raw = await Module.ccall('bench_tg', 'string', ['number'], [nGen], { async: true });
          const t_ns = (performance.now() - t0) * 1e6;
          parseBenchResult('bench_tg', raw);
          samples_ns.push(t_ns);
          log(`tg${nGen} run ${i + 1}/${nReps}: ${(t_ns / 1e6).toFixed(1)} ms (${(1e9 * nGen / t_ns).toFixed(1)} t/s)`);
        }
        tests.push(buildTest(`tg${nGen}`, 0, nGen, samples_ns));
      } catch (err) {
        log(`tg test failed: ${err.message}`);
      }
    }

    if (tests.length > 0) {
      result.metrics = {
        tests,
        n_prompt: wantPp ? nPrompt : 0,
        n_gen: wantTg ? nGen : 0,
        n_reps: nReps,
      };
    }
  }

  await Module.ccall('bench_exit', null, [], [], { async: true });

  if (useOpfsPath) {
    // Close the sync handle so OPFS can release its lock on the file (and
    // so a subsequent run can open a fresh handle without colliding).
    opfsFreeAll(Module);
  } else if (modelPtr) {
    Module._free(modelPtr);
    modelPtr = 0;
  }

  result.status = 'done';
  const summary = result.metrics?.tests
    ?.map(t => `${t.name}: ${t.avg_ts.toFixed(2)} ± ${t.stddev_ts.toFixed(2)} t/s`)
    .join(' | ') || 'no perf';
  status('done', `Done! ${summary}`);
  return result;
}
