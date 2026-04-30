// Device-fit helpers for the interactive bench page.
//
// Two budget probes drive the per-variant fit decision:
//
//   getDeviceBudgetMB() — empirical WASM heap probe. Grows a
//     WebAssembly.Memory page-by-page in a worker until it fails. Caps
//     the working set (KV cache + compute scratch + JS heap headroom)
//     llama.cpp consumes during inference.
//
//   probeGpuBudgetMB() — empirical WebGPU memory probe. Allocates real
//     buffers with mappedAtCreation=true on the actual adapter until OOM.
//     Caps the size of model weights llama.cpp can hold in GPU buffers,
//     since OPFS-streaming keeps model bytes off the WASM heap.
//
// variantFits() then checks both: model size + GPU overhead ≤ GPU budget,
// AND heap working-set floor ≤ heap budget. wllama doesn't probe at all
// — they let load attempts fail naturally — but our auto-select buttons
// ("All fit", "Run study") need a fit predicate, so we err on the side
// of measuring rather than guessing.
//
// On wasm32 the linear memory caps at 4 GiB no matter how much physical
// RAM the device has, so heap probe results above 4096 MB cannot exist.

const DEFAULT_BUDGET_MB = 2 * 1024;
const HOSTED_QUOTA_FRACTION = 0.4;
const HOSTED_QUOTA_CAP_MB = 8 * 1024;

// Hard ceiling on mobile WASM heap regardless of probe result. iOS/Android
// can reap the tab under system memory pressure without raising a JS error
// the probe could observe, so an "ok at 4 GiB" result is not safe to trust
// on a phone.
//
// Empirically iOS Safari tabs get reaped well below the WebAssembly.Memory
// engine cap (~1 GiB on iPhone), and Android Chrome on mid-range devices
// behaves similarly. Below 500 MB heap usage tends to be safe across
// modern phones; above that we start seeing tab kills mid-run. The OPFS-
// streaming model load means model bytes no longer live on the WASM heap,
// so this budget caps the per-step working set, not the model file.
const MOBILE_HEAP_CEILING_MB = 500;

// Hard ceiling on mobile GPU memory probe result. Even when the probe
// succeeds at higher numbers, the OS may evict the GPU process or the tab
// before we can actually use it. iPhone WebGPU (Metal-3 under the hood)
// typically gives a tab 1.5–3 GB usable depending on device class; cap at
// 3 GB as a conservative ceiling that won't reject anything reasonable.
const MOBILE_GPU_CEILING_MB = 3 * 1024;

const PROBE_TIMEOUT_MS = 15_000;
const GPU_PROBE_STEP_MB = 256;
const GPU_PROBE_MAX_MB = 8 * 1024;
const GPU_PROBE_TIMEOUT_MS = 8_000;

// Working-set floor in the WASM heap. KV cache + compute buffers + JS heap
// headroom for a typical 1B model at n_ctx=2048 add up to ~400 MB; we
// require 500 to leave a margin. Bigger contexts scale this up — not
// modeled yet (worth revisiting if we benchmark at n_ctx >> 2048).
const HEAP_WORKING_SET_FLOOR_MB = 500;

// Per-variant overhead added on top of the model file size when checking
// GPU fit. Covers compute buffers, alignment padding, and the KV cache
// mirror that the WebGPU backend keeps. A flat 200 MB is a conservative
// approximation; in practice it scales somewhat with model + context size.
const GPU_VARIANT_OVERHEAD_MB = 200;

export function isMobileDevice() {
  if (typeof navigator === 'undefined') return false;
  if (navigator.userAgentData?.mobile === true) return true;
  const ua = navigator.userAgent || '';
  return /iPhone|iPad|iPod|Android.*Mobile/.test(ua);
}

// ──────────────── WASM heap probe ────────────────

// Spawn the probe worker, wait for a result, clean up. Returns
// { probedMB } on success, or { probedMB: 0, error } on any failure mode
// (timeout, worker construct error, worker onerror — typically the probe
// itself ran the engine out of memory).
export function probeHeapBudgetMB({ stepPages, maxPages, timeoutMs = PROBE_TIMEOUT_MS } = {}) {
  return new Promise((resolve) => {
    let worker;
    try {
      worker = new Worker(new URL('./memory-probe.js', import.meta.url));
    } catch (err) {
      resolve({ probedMB: 0, error: `worker construct failed: ${err.message}` });
      return;
    }

    const timer = setTimeout(() => {
      try { worker.terminate(); } catch { /* noop */ }
      resolve({ probedMB: 0, error: 'probe timeout' });
    }, timeoutMs);

    worker.onmessage = (e) => {
      clearTimeout(timer);
      const { committedMB = 0 } = e.data || {};
      try { worker.terminate(); } catch { /* noop */ }
      resolve({ probedMB: committedMB });
    };
    worker.onerror = (err) => {
      clearTimeout(timer);
      try { worker.terminate(); } catch { /* noop */ }
      resolve({ probedMB: 0, error: err.message || 'worker error' });
    };

    worker.postMessage({ stepPages, maxPages });
  });
}

// ──────────────── GPU memory probe ────────────────

// Allocate WebGPU buffers in stepMB increments until OOM, return the
// total committed bytes as the GPU memory budget. Uses
// mappedAtCreation=true to force real memory commit (some drivers lazy-
// allocate until first use otherwise) and captures OOM via the
// 'out-of-memory' error scope, with device.lost as a backstop.
//
// Caveats:
//  - The GPU process is shared with other tabs. If they're holding GPU
//    memory the probe undercounts. (Same as wllama's heap probe — best
//    we can do without a richer browser API.)
//  - Some drivers (notably iOS Metal under WebKit) lazy-fail at dispatch
//    time rather than at createBuffer; this probe's number is therefore
//    an upper bound, not a guarantee. Mobile cap below mitigates.
export async function probeGpuBudgetMB({
  stepMB = GPU_PROBE_STEP_MB,
  maxMB = GPU_PROBE_MAX_MB,
  timeoutMs = GPU_PROBE_TIMEOUT_MS,
} = {}) {
  if (!navigator.gpu) {
    return { probedMB: 0, error: 'WebGPU not available' };
  }

  let adapter, device;
  try {
    adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { probedMB: 0, error: 'no WebGPU adapter' };
    // Request the maximum the adapter can give us; defaults are often
    // smaller than what the hardware supports.
    const requiredLimits = {};
    const cap = (k) => {
      const v = adapter.limits?.[k];
      if (typeof v === 'number') requiredLimits[k] = v;
    };
    cap('maxBufferSize');
    cap('maxStorageBufferBindingSize');
    device = await adapter.requestDevice({ requiredLimits });
  } catch (err) {
    return { probedMB: 0, error: `adapter/device init failed: ${err.message}` };
  }

  let deviceLost = false;
  device.lost.then(() => { deviceLost = true; }).catch(() => {});

  const buffers = [];
  const stepBytes = stepMB * 1024 * 1024;
  let totalBytes = 0;
  const start = performance.now();

  try {
    while (totalBytes + stepBytes <= maxMB * 1024 * 1024) {
      if (deviceLost) break;
      if (performance.now() - start > timeoutMs) break;

      device.pushErrorScope('out-of-memory');
      let buffer;
      try {
        buffer = device.createBuffer({
          size: stepBytes,
          usage: GPUBufferUsage.STORAGE,
          mappedAtCreation: true,
        });
        // Touch the start of the mapped range to force a real commit.
        // Drivers can lazy-back the allocation until first write, which
        // would fool the probe into thinking it has more headroom than it
        // really does.
        const touchBytes = Math.min(stepBytes, 64 * 1024);
        new Uint8Array(buffer.getMappedRange(0, touchBytes))[0] = 1;
        buffer.unmap();
      } catch (err) {
        await device.popErrorScope().catch(() => null);
        break;
      }

      const error = await device.popErrorScope().catch(() => null);
      if (error) {
        try { buffer.destroy(); } catch { /* noop */ }
        break;
      }

      buffers.push(buffer);
      totalBytes += stepBytes;

      // Yield so we don't starve the main thread / GC.
      await new Promise((r) => setTimeout(r, 0));
    }
  } finally {
    for (const b of buffers) {
      try { b.destroy(); } catch { /* noop */ }
    }
    try { device.destroy(); } catch { /* noop */ }
  }

  return { probedMB: Math.floor(totalBytes / (1024 * 1024)) };
}

// ──────────────── public budget API ────────────────

// Cache the full budget for the lifetime of the page load. Both probes
// take 1–8 s; we don't want to pay that twice for the same surface.
let _budgetPromise = null;

export async function getDeviceBudgetMB() {
  if (_budgetPromise) return _budgetPromise;
  _budgetPromise = _computeBudget();
  return _budgetPromise;
}

async function _computeBudget() {
  const memGB = typeof navigator.deviceMemory === 'number' ? navigator.deviceMemory : null;
  let quotaMB = null;
  try {
    const est = await navigator.storage?.estimate?.();
    if (est?.quota) quotaMB = est.quota / (1024 * 1024);
  } catch {
    // some browsers throw on storage.estimate in non-secure contexts
  }

  const isMobile = isMobileDevice();

  // The heap probe is always safe — it runs inside a dedicated worker so
  // a runaway grow() kills the worker, not the tab.
  //
  // The GPU probe is NOT safe on mobile. It allocates real WebGPU buffers
  // in the GPU process, which is shared with the main tab. On phones,
  // pushing 1–2 GB of GPU buffers triggers OOM behavior in iOS Safari
  // that reaps the tab — which then reloads, reprobes, and reaps again,
  // an infinite refresh loop. Skip the probe on mobile and substitute a
  // heuristic based on navigator.deviceMemory; on desktop the GPU process
  // has enough headroom for the probe to be useful.
  const heapProbe = await probeHeapBudgetMB();
  const gpuProbe = isMobile
    ? { probedMB: 0, error: 'skipped on mobile (would OOM the tab)' }
    : await probeGpuBudgetMB();

  // ── Heap budget ──
  let heapBudgetMB;
  let heapSource;
  if (heapProbe.probedMB > 0) {
    heapBudgetMB = heapProbe.probedMB;
    heapSource = `probe (WASM heap, ${heapProbe.probedMB} MB committed)`;
  } else if (memGB !== null) {
    heapBudgetMB = memGB * 1024 * 0.6;
    heapSource = 'navigator.deviceMemory (heap probe failed)';
  } else if (quotaMB !== null) {
    heapBudgetMB = Math.min(quotaMB * HOSTED_QUOTA_FRACTION, HOSTED_QUOTA_CAP_MB);
    heapSource = 'navigator.storage.estimate().quota (heap probe failed)';
  } else {
    heapBudgetMB = DEFAULT_BUDGET_MB;
    heapSource = 'default (heap probe failed)';
  }
  if (isMobile && heapBudgetMB > MOBILE_HEAP_CEILING_MB) {
    heapBudgetMB = MOBILE_HEAP_CEILING_MB;
    heapSource += ' → mobile-capped';
  }

  // ── GPU budget ──
  let gpuBudgetMB;
  let gpuSource;
  if (isMobile) {
    // Heuristic since the probe would OOM the tab. Quarter of the
    // device's reported RAM, clamped to a sensible range. iPhone WebGPU
    // typically gives a tab 1.5–2 GB of usable GPU memory before things
    // start failing; this estimate undershoots slightly to leave margin.
    const memMB = (memGB || 4) * 1024;
    gpuBudgetMB = Math.max(512, Math.min(memMB * 0.25, MOBILE_GPU_CEILING_MB));
    gpuSource = `mobile heuristic (deviceMemory ${memGB ?? '?'} GB × 0.25, clamped 512 MB–${MOBILE_GPU_CEILING_MB} MB)`;
  } else if (gpuProbe.probedMB > 0) {
    gpuBudgetMB = gpuProbe.probedMB;
    gpuSource = `probe (WebGPU buffers, ${gpuProbe.probedMB} MB allocated)`;
  } else {
    gpuBudgetMB = 0;
    gpuSource = `probe failed: ${gpuProbe.error || 'unknown'}`;
  }

  return {
    // Combined headline budget — what the UI shows as "Max model size".
    // GPU memory is now the constraint that varies per device; heap
    // budget is a separate floor check.
    budgetMB: gpuBudgetMB,
    gpuBudgetMB,
    heapBudgetMB,
    memGB,
    quotaMB,
    probedMB: heapProbe.probedMB,
    gpuProbedMB: gpuProbe.probedMB,
    probeError: heapProbe.error || null,
    gpuProbeError: gpuProbe.error || null,
    isMobile,
    // Two-line source string so the UI stays compact while still
    // surfacing both probes in the device card tooltip.
    source: gpuSource,
    heapSource,
  };
}

// variantFits decides whether a model file of `sizeMB` bytes can be
// loaded and run on this device. Two checks must pass:
//
//   1. sizeMB + GPU_VARIANT_OVERHEAD_MB ≤ gpuBudgetMB
//        Model weights live in WebGPU buffers (since OPFS streaming
//        keeps them off the WASM heap). The overhead covers compute
//        scratch + alignment + KV cache mirror.
//
//   2. heapBudgetMB ≥ HEAP_WORKING_SET_FLOOR_MB
//        The WASM heap still has to fit the working set: KV cache,
//        ggml compute buffers, and JS heap headroom. Roughly constant
//        per inference regardless of model size at fixed n_ctx.
//
// Backwards-compat: if the second arg is a plain number, treat it as
// the legacy heap-only budget and apply the prior 1.5× sizeMB overhead.
// New callers should pass { gpuBudgetMB, heapBudgetMB }.
export function variantFits(sizeMB, budget) {
  if (typeof sizeMB !== 'number' || sizeMB <= 0) return false;

  if (typeof budget === 'number') {
    return budget > 0 && sizeMB * 1.5 <= budget;
  }
  if (!budget || typeof budget !== 'object') return false;

  const { gpuBudgetMB, heapBudgetMB } = budget;
  if (typeof gpuBudgetMB !== 'number' || sizeMB + GPU_VARIANT_OVERHEAD_MB > gpuBudgetMB) {
    return false;
  }
  if (typeof heapBudgetMB !== 'number' || heapBudgetMB < HEAP_WORKING_SET_FLOOR_MB) {
    return false;
  }
  return true;
}

export async function describeDevice() {
  const budget = await getDeviceBudgetMB();
  let gpu = null;
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) gpu = adapter.info || { vendor: 'unknown' };
    } catch {
      gpu = null;
    }
  }

  // UA Client Hints: high-entropy values give us the real architecture
  // and OS, neither of which `navigator.platform` reports correctly on
  // Apple Silicon Macs (it returns "MacIntel" forever for back-compat).
  let uaArch = null;
  let uaPlatform = null;
  let uaPlatformVersion = null;
  // `fullVersionList` is the high-entropy version of `brands` and gives
  // us the full dotted version (e.g. "147.0.7390.107") instead of just
  // the major. The default `brands` is major-only.
  let fullVersionList = null;
  try {
    const uad = navigator.userAgentData;
    if (uad?.getHighEntropyValues) {
      const hev = await uad.getHighEntropyValues([
        'architecture', 'platform', 'platformVersion', 'fullVersionList',
      ]);
      uaArch = hev.architecture || null;
      uaPlatform = hev.platform || null;
      uaPlatformVersion = hev.platformVersion || null;
      fullVersionList = hev.fullVersionList || null;
    }
  } catch { /* not Chromium or denied */ }

  // UA-CH brands give us a clean { brand, version } pair without parsing
  // the userAgent string. Filter out the "Not(A:Brand)" decoy entries.
  // Prefer `fullVersionList` (full dotted version) over the major-only
  // default `brands` list.
  const brandSource = fullVersionList || navigator.userAgentData?.brands || [];
  const brands = brandSource
    .filter(b => b && !/not[^\w]*a[^\w]*brand/i.test(b.brand));

  return {
    ...budget,
    webgpu: !!navigator.gpu,
    gpu,
    userAgent: navigator.userAgent,
    platform: navigator.platform ?? null,
    uaArch,
    uaPlatform,
    uaPlatformVersion,
    uaBrands: brands,
  };
}
