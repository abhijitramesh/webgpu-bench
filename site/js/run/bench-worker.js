// Dedicated Worker that runs a single llama.cpp inference pass. Loaded by
// controller.js and harness.js so we can importScripts() the
// Emscripten-emitted bench.js (which is a classic, non-module script).
//
// Protocol (all messages use { type, ... } tag):
//
//   main → worker: {
//     type: 'run',
//     params: {
//       buildType,
//       // model load
//       nCtx, nGpuLayers,
//       // consistency phase (set consistencyPrompt to '' to skip)
//       consistencyPrompt, consistencyNPredict, refTokenIds,
//       // perf phase
//       nPrompt, nGen, nReps, nDepth, noWarmup,
//     },
//     opfsPath: { rootDir, repo, filename }
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

const post = (msg) => self.postMessage(msg);
const log = (line) => post({ type: 'log', line });
// sinceMs: optional epoch ms. Forwarded to controller so the row ticks an
// elapsed counter while a long-running ccall (warmup, big-model rep) is in
// flight — JSPI doesn't yield often enough on CPU paths to drive ticks here.
const status = (s, msg, sinceMs) => post({ type: 'status', status: s, msg, sinceMs });

// Capture the GPUDevice that llama.cpp's WebGPU backend creates so we can
// destroy() it before the worker terminates. Without this, iOS Safari holds
// Metal allocations from prior runs long enough for the next model load in a
// study sweep to push the tab over its memory limit and trigger Jetsam.
// Installed at module scope so the capture is in place before the bench.js
// glue is importScripts'd and before any C-side requestAdapter/requestDevice
// calls run. The wrapper is one-shot per device: if the backend ever
// re-requests, the latest reference wins.
let capturedGpuDevice = null;
if (self.navigator?.gpu && typeof self.navigator.gpu.requestAdapter === 'function') {
  const origRequestAdapter = self.navigator.gpu.requestAdapter.bind(self.navigator.gpu);
  self.navigator.gpu.requestAdapter = async (...args) => {
    const adapter = await origRequestAdapter(...args);
    if (adapter && typeof adapter.requestDevice === 'function' && !adapter.__deviceWrapped) {
      const origRequestDevice = adapter.requestDevice.bind(adapter);
      adapter.requestDevice = async (...devArgs) => {
        const device = await origRequestDevice(...devArgs);
        capturedGpuDevice = device;
        return device;
      };
      adapter.__deviceWrapped = true;
    }
    return adapter;
  };
}

// Below this many compared tokens, the consistency agreement rate is
// statistical noise (e.g. early-EOS models that produce 1 token always
// report 100%).
const CONSISTENCY_MIN_TOKENS = 8;

// Sleep between perf reps so the GPU clock state can recover. Without
// this, sustained tg decode reps showed monotonic decay (rep 1 fastest,
// rep N slowest) — looks like Apple's GPU power-state cooldown.
const REP_COOLDOWN_MS = 1000;
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// llama.cpp/ggml emit info, warnings, AND errors all to stderr. Tag only the
// actually-bad lines as :err so real failures stand out.
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
// llama-bench reports avg_ts = (n_tokens * 1e9) / avg_ns and stddev_ts as
// the std of per-sample t/s, computed independently rather than propagated
// from stddev_ns (the mapping isn't linear).
//
// `n_depth` carries through unchanged so downstream consumers can label
// e.g. "pp512 @ d2048" the way llama-bench does (line 1984 of
// llama.cpp/tools/llama-bench/llama-bench.cpp).
function buildTest(name, n_prompt, n_gen, n_depth, samples_ns) {
  const n = samples_ns.length;
  if (n === 0) {
    return { name, n_prompt, n_gen, n_depth, avg_ns: 0, stddev_ns: 0, avg_ts: 0, stddev_ts: 0, samples_ns: [], samples_ts: [] };
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
    n_depth,
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

function describeError(err) {
  if (err == null) return '';
  if (typeof err === 'string') return err;
  if (typeof err === 'number' || typeof err === 'boolean') return String(err);
  if (err instanceof Error) return err.message || String(err);
  if (typeof err === 'object') {
    const parts = [];
    if (typeof err.name === 'string' && err.name) parts.push(err.name);
    if (typeof err.type === 'string' && err.type) parts.push(`type=${err.type}`);
    if (typeof err.message === 'string' && err.message) parts.push(err.message);
    if (typeof err.reason === 'string' && err.reason) parts.push(`reason=${err.reason}`);
    if (typeof err.filename === 'string' && err.filename) parts.push(`file=${err.filename}`);
    if (typeof err.lineno === 'number' && err.lineno > 0) parts.push(`line=${err.lineno}`);
    if (typeof err.colno === 'number' && err.colno > 0) parts.push(`col=${err.colno}`);
    if (typeof err.error === 'string' && err.error) parts.push(`error=${err.error}`);
    else if (err.error instanceof Error && err.error.message) parts.push(`error=${err.error.message}`);
    if (parts.length > 0) return parts.join(' | ');
    try {
      const own = {};
      for (const key of Object.getOwnPropertyNames(err)) {
        own[key] = err[key];
      }
      const json = JSON.stringify(own);
      if (json && json !== '{}') return json;
    } catch {
      // fall through
    }
    const tag = Object.prototype.toString.call(err);
    if (tag && tag !== '[object Object]') return tag;
    return 'unknown structured error';
  }
  return String(err);
}

function formatPhaseError(phase, err) {
  const detail = describeError(err);
  return detail ? `${phase} threw WASM exception (${detail})` : `${phase} threw WASM exception`;
}

async function ccallPhase(Module, phase, returnType, argTypes, args) {
  try {
    return await Module.ccall(phase, returnType, argTypes, args, { async: true });
  } catch (err) {
    throw new Error(formatPhaseError(phase, err));
  }
}

async function ccallPhaseLabel(Module, phaseLabel, exportName, returnType, argTypes, args) {
  try {
    return await Module.ccall(exportName, returnType, argTypes, args, { async: true });
  } catch (err) {
    throw new Error(formatPhaseError(phaseLabel, err));
  }
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
        error: describeError(err),
        metrics: null,
      },
    });
  }
};

async function runOne({ params, opfsPath }) {
  const {
    buildType,
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
    nDepth = 0,
    noWarmup,
  } = params;
  // The worker only loads via OPFS now: main thread downloads to OPFS,
  // we open a FileSystemSyncAccessHandle here, and patched MEMFS
  // stream_ops route llama.cpp's fread through the sync handle. That
  // path scales past the WASM heap budget and shares one numerical
  // implementation across every surface.
  if (!opfsPath) {
    throw new Error('runOne: opfsPath is required (heap/buffer paths removed)');
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
  // Open a FileSystemSyncAccessHandle on the OPFS file, register a
  // zero-byte placeholder in MEMFS, and patch MEMFS stream_ops so
  // llama.cpp's fread is routed to the sync handle. Bytes never touch
  // the WASM heap, so the model size is bounded by OPFS quota, not heap.
  status('opfs', 'Linking OPFS-backed model into MEMFS...');
  const fileHandle = await resolveOpfsHandle(opfsPath);
  patchMEMFS(Module);
  const size = await opfsAlloc(Module, 'model.gguf', fileHandle);
  log(`OPFS-backed model.gguf registered (${(size / (1024 * 1024)).toFixed(1)} MB)`);
  // Report 100% to keep the existing progress UI happy — the actual
  // download to OPFS happened before the worker spawn.
  post({ type: 'progress', fraction: 1, downloaded: size, total: size });

  // ─── Init backend ───
  status('initializing', 'Initializing llama.cpp backends...');
  const initResult = await ccallPhase(Module, 'bench_init', 'number', [], []);
  if (initResult !== 0) throw new Error(`bench_init failed: ${initResult}`);
  log('Backends initialized');

  // ─── Load model ───
  // use_mmap=0 — the patched MEMFS mmap throws explicitly, so any
  // accidental mmap attempt surfaces as a clear error rather than a
  // silent heap copy.
  status('loading_model', `Loading model (ctx=${nCtx}, gpu_layers=${nGpuLayers}, mmap=0)...`);
  const loadResult = await ccallPhase(
    Module,
    'bench_load',
    'number',
    ['string', 'number', 'number', 'number'],
    ['/model.gguf', nCtx, nGpuLayers, 0],
  );
  if (loadResult !== 0) throw new Error(`bench_load failed: ${loadResult}`);
  log('Model loaded');

  // ─── Memory snapshot from llama.cpp ───
  // Captured immediately after bench_load so model_size reflects the loaded
  // model and per-device free counters reflect post-allocation state. Wrapped
  // in try/catch — if the C side or a backend errors, the run can still
  // produce perf numbers, just without memoryInfo on the record.
  try {
    const raw = await ccallPhase(Module, 'bench_memory_info', 'string', [], []);
    result.memoryInfo = parseBenchResult('bench_memory_info', raw);
    const dev = (result.memoryInfo.devices || [])
      .map(d => `${d.name}(${d.type}) free=${(d.free / (1024 * 1024)).toFixed(0)}MB total=${(d.total / (1024 * 1024)).toFixed(0)}MB`)
      .join(' | ') || 'none';
    log(`Memory: model=${(result.memoryInfo.model_size / (1024 * 1024)).toFixed(0)}MB state=${(result.memoryInfo.state_size / (1024 * 1024)).toFixed(0)}MB | ${dev}`);
  } catch (err) {
    log(`bench_memory_info failed: ${err.message} — continuing without memoryInfo`);
  }

  // ─── Consistency phase ───
  // Soft-fail: a failure here logs and falls through to the perf phase
  // rather than aborting the whole run. Some devices/models can't survive
  // bench_run (e.g. unsupported op, OOM mid-decode) but can still produce
  // useful pp/tg numbers via synthetic-token paths.
  if (consistencyPrompt) {
    try {
      status('consistency', 'Running consistency check...', Date.now());
      log(`bench_run("...", ${consistencyNPredict}) — consistency phase`);
      const raw = await ccallPhase(Module, 'bench_run', 'string',
        ['string', 'number'],
        [consistencyPrompt, consistencyNPredict]);
      const r = parseBenchResult('bench_run', raw);
      result.output = r.output || '';
      result.consistency = { token_ids: r.token_ids || [] };

      if (refTokenIds) {
        log('bench_eval_tokens — forced-decode vs CPU baseline');
        const evalRaw = await ccallPhase(Module, 'bench_eval_tokens', 'string',
          ['string', 'string'],
          [consistencyPrompt, refTokenIds]);
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
  // Test name suffix mirroring llama-bench (e.g. "pp512 @ d2048").
  const depthSuffix = nDepth > 0 ? ` @ d${nDepth}` : '';
  // Each timed rep is preceded by an untimed bench_set_depth call so the KV
  // cache is in a known state. The C side caches the post-prefill snapshot,
  // so reps 2..N at the same depth restore from snapshot instead of
  // re-running the prefill (mirroring llama-bench's `cstate` reuse).
  const setDepth = async (label) => {
    const raw = await ccallPhaseLabel(Module, `bench_set_depth(${nDepth}) ${label}`, 'bench_set_depth', 'string', ['number'], [nDepth]);
    const r = parseBenchResult(`bench_set_depth(${nDepth}) ${label}`, raw);
    if (nDepth > 0) {
      log(`bench_set_depth(${nDepth}) ${label}: ${r.cached ? 'restored snapshot' : 'prefilled'}`);
    }
  };
  if (wantPp || wantTg) {
    const tests = [];

    if (wantPp) {
      try {
        if (!noWarmup) {
          status('perf', `warmup pp${nPrompt}${depthSuffix}`, Date.now());
          await setDepth('pp warmup');
          log(`bench_pp(${nPrompt})${depthSuffix} — warmup`);
          const raw = await ccallPhaseLabel(Module, `bench_pp warmup (${nPrompt}${depthSuffix})`, 'bench_pp', 'string', ['number'], [nPrompt]);
          parseBenchResult('bench_pp warmup', raw);
        }
        const samples_ns = [];
        for (let i = 0; i < nReps; i++) {
          status('perf', `pp${nPrompt}${depthSuffix} ${i + 1}/${nReps}`, Date.now());
          await setDepth(`pp rep ${i + 1}/${nReps}`);
          const t0 = performance.now();
          const raw = await ccallPhaseLabel(Module, `bench_pp rep ${i + 1}/${nReps} (${nPrompt}${depthSuffix})`, 'bench_pp', 'string', ['number'], [nPrompt]);
          const t_ns = (performance.now() - t0) * 1e6;
          parseBenchResult('bench_pp', raw);
          samples_ns.push(t_ns);
          log(`pp${nPrompt}${depthSuffix} run ${i + 1}/${nReps}: ${(t_ns / 1e6).toFixed(1)} ms (${(1e9 * nPrompt / t_ns).toFixed(1)} t/s)`);
          if (i + 1 < nReps) await sleep(REP_COOLDOWN_MS);
        }
        tests.push(buildTest(`pp${nPrompt}${depthSuffix}`, nPrompt, 0, nDepth, samples_ns));
      } catch (err) {
        log(`pp test failed: ${err.message}`);
      }
    }

    if (wantTg) {
      try {
        if (!noWarmup) {
          // Run the full nGen-token decode loop as warmup (was bench_tg(1)).
          // A 1-token warmup exercises the decode kernel once, which leaves
          // the first timed rep absorbing pipeline-cache / shader-specialize
          // cost on every subsequent step.
          status('perf', `warmup tg${nGen}${depthSuffix}`, Date.now());
          await setDepth('tg warmup');
          log(`bench_tg(${nGen})${depthSuffix} — warmup`);
          const raw = await ccallPhaseLabel(Module, `bench_tg warmup (${nGen}${depthSuffix})`, 'bench_tg', 'string', ['number'], [nGen]);
          parseBenchResult('bench_tg warmup', raw);
        }
        const samples_ns = [];
        for (let i = 0; i < nReps; i++) {
          status('perf', `tg${nGen}${depthSuffix} ${i + 1}/${nReps}`, Date.now());
          await setDepth(`tg rep ${i + 1}/${nReps}`);
          const t0 = performance.now();
          const raw = await ccallPhaseLabel(Module, `bench_tg rep ${i + 1}/${nReps} (${nGen}${depthSuffix})`, 'bench_tg', 'string', ['number'], [nGen]);
          const t_ns = (performance.now() - t0) * 1e6;
          parseBenchResult('bench_tg', raw);
          samples_ns.push(t_ns);
          log(`tg${nGen}${depthSuffix} run ${i + 1}/${nReps}: ${(t_ns / 1e6).toFixed(1)} ms (${(1e9 * nGen / t_ns).toFixed(1)} t/s)`);
          if (i + 1 < nReps) await sleep(REP_COOLDOWN_MS);
        }
        tests.push(buildTest(`tg${nGen}${depthSuffix}`, 0, nGen, nDepth, samples_ns));
      } catch (err) {
        log(`tg test failed: ${err.message}`);
      }
    }

    if (tests.length > 0) {
      result.metrics = {
        tests,
        n_prompt: wantPp ? nPrompt : 0,
        n_gen: wantTg ? nGen : 0,
        n_depth: nDepth,
        n_reps: nReps,
      };
    }
  }

  await ccallPhase(Module, 'bench_exit', null, [], []);

  // Close the sync handle so OPFS can release its lock on the file (and
  // so a subsequent run can open a fresh handle without colliding).
  opfsFreeAll(Module);

  // Eagerly drop GPU buffers. worker.terminate() alone leaves Metal
  // allocations alive on iOS Safari long enough for the next study run to
  // hit Jetsam — destroy() returns the memory synchronously.
  if (capturedGpuDevice) {
    try {
      capturedGpuDevice.destroy();
    } catch (err) {
      log(`device.destroy() failed: ${err.message}`);
    }
    capturedGpuDevice = null;
  }

  result.status = 'done';
  const summary = result.metrics?.tests
    ?.map(t => `${t.name}: ${t.avg_ts.toFixed(2)} ± ${t.stddev_ts.toFixed(2)} t/s`)
    .join(' | ') || 'no perf';
  status('done', `Done! ${summary}`);
  return result;
}
