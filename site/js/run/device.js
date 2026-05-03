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

// Mobile per-device budgets. Two independent caps, mirroring the desktop
// path — model weights stream from OPFS into WebGPU buffers (see
// bench-worker.js:patchMEMFS / opfsAlloc), so the model size constrains
// `gpuBudgetMB`, not `heapBudgetMB`. The WASM heap only has to hold the
// working set (KV cache + ggml compute scratch + JS heap headroom).
//
// Earlier we collapsed both into a single tab budget on the theory that
// iOS Jetsam treats the whole tab process as one pool, so any allocation
// counts the same. That's true for Jetsam — but it conflates *where* the
// memory lives with *how much* the platform can hand out: the WASM heap
// has a much tighter practical ceiling than the GPU side, and counting
// model bytes against the heap ceiling rejected models that load fine
// via OPFS streaming.
//
// Numbers come from public reports / Apple docs:
//
//  - iPhone WASM practical limit: 300–450 MB → heap budget
//      lapcatsoftware.com/articles/2026/1/7.html
//      news.ycombinator.com/item?id=39039593
//      github.com/emscripten-core/emscripten/issues/19374
//      github.com/godotengine/godot/issues/70621
//
//  - iOS Safari WebGPU maxBufferSize: 256 MB on iPhone 6 / older,
//    993 MB on iPad Pro M-class. Per-buffer cap, not total.
//      Apple WWDC 2025 "Unlock GPU computing with WebGPU"
//
//  - iPhone 12 Pro reports tab OOM around 1.5–3 GB; Jetsam intervenes
//    earlier under pressure. We undershoot the lower bound for headroom.
//      developer.apple.com/forums/thread/761666
//
// Heap budgets = WASM heap practical limits.
const IPHONE_HEAP_BUDGET_MB  = 450;
const IPAD_HEAP_BUDGET_MB    = 1500;
const ANDROID_HEAP_BUDGET_MB = 800;

// GPU budgets = available GPU-buffer capacity for model weights + KV
// mirror, sized below the Jetsam tab ceiling minus working-set headroom.
// These are *fallback* values. On mobile we run a bounded GPU probe
// (capped well below the Jetsam ceiling, with yields between steps) and
// only fall back to the static value when the probe trips, returns less
// than the static floor, or maxBufferSize is too small to bother.
//
// iPhone: empirical — 1200 MB caused tab reloads on first variant of a
// Run study (Llama-3.2-1B Q2_K, 554 MB) on iPhone 17 Pro Max. 700 MB
// keeps Llama-1B variants out of variantFits while still allowing the
// 250–500 MB tier (gemma-3-270m Q8, Qwen3-0.6B Q4, etc.) — the band
// that was missing under the old 450 MB shared cap.
const IPHONE_GPU_BUDGET_MB  = 700;
const IPAD_GPU_BUDGET_MB    = 2500;
const ANDROID_GPU_BUDGET_MB = 1500;

// Bounded mobile GPU probe — small steps + yields keep allocation rate
// below the spike threshold that triggers Jetsam, and a tier-based hard
// cap keeps the probe ceiling well below the device's known crash point.
const MOBILE_PROBE_STEP_MB = 128;
const MOBILE_PROBE_TIMEOUT_MS = 10_000;
const MOBILE_PROBE_YIELD_MS = 50;
const MOBILE_PROBE_SAFETY_MARGIN_MB = 150;

// SessionStorage sentinel: written before the probe, cleared after. If
// we see it on the next page load, the previous probe crashed the tab —
// skip probing and use the static fallback so we don't loop forever
// with the user staring at a tab that keeps reloading. Cleared at end
// of probe so subsequent loads in the same session re-probe normally.
const MOBILE_PROBE_SENTINEL_KEY = 'webgpu-bench:mobile-gpu-probe-in-progress';

// Probe ceiling per family × maxBufferSize tier. Caps are deliberately
// conservative — a probe that completes successfully gives `cap - margin`,
// while a probe that OOMs partway gives `probed - margin`. We never
// exceed `cap`, so even a successful probe sits below the empirical
// crash point on the worst-case device we've seen for that tier.
function getMobileProbeCapMB(family, maxBufferSizeMB) {
  if (family === 'iphone') {
    if (maxBufferSizeMB >= 900) return 1000;
    if (maxBufferSizeMB >= 500) return 800;
    return 400;
  }
  if (family === 'ipad') {
    if (maxBufferSizeMB >= 900) return 3000;
    if (maxBufferSizeMB >= 500) return 1800;
    return 1000;
  }
  if (family === 'android') {
    if (maxBufferSizeMB >= 900) return 2000;
    if (maxBufferSizeMB >= 500) return 1500;
    return 800;
  }
  return 700;
}

function detectMobileFamily() {
  if (typeof navigator === 'undefined') return null;
  const ua = navigator.userAgent || '';
  // iPadOS 13+ reports "Macintosh" UA but exposes touch; that's the
  // standard iPad-detection workaround.
  if (/iPad/.test(ua)) return 'ipad';
  if (navigator.maxTouchPoints > 1 && /Mac/.test(navigator.platform || '')) return 'ipad';
  if (/iPhone|iPod/.test(ua)) return 'iphone';
  if (/Android.*Mobile/.test(ua)) return 'android';
  if (navigator.userAgentData?.mobile === true) return 'android';
  return null;
}

function getMobileBudgetMB(family) {
  if (family === 'ipad')    return { heap: IPAD_HEAP_BUDGET_MB,    gpu: IPAD_GPU_BUDGET_MB };
  if (family === 'iphone')  return { heap: IPHONE_HEAP_BUDGET_MB,  gpu: IPHONE_GPU_BUDGET_MB };
  if (family === 'android') return { heap: ANDROID_HEAP_BUDGET_MB, gpu: ANDROID_GPU_BUDGET_MB };
  return { heap: IPHONE_HEAP_BUDGET_MB, gpu: IPHONE_GPU_BUDGET_MB }; // safest default
}

const PROBE_TIMEOUT_MS = 15_000;
const GPU_PROBE_STEP_MB = 256;
const GPU_PROBE_MAX_MB = 8 * 1024;
const GPU_PROBE_TIMEOUT_MS = 8_000;

// Working-set floor in the WASM heap. KV cache + compute buffers + JS
// heap headroom for a typical 1B model at n_ctx=2048 add up to a few
// hundred MB. Floor at 256 so an absurdly-tiny heap (or a probe failure
// that returned 0) doesn't pass variantFits.
const HEAP_WORKING_SET_FLOOR_MB = 256;

// Per-variant overhead added on top of the model file size when checking
// GPU fit. Covers compute buffers, alignment padding, and the KV cache
// mirror that the WebGPU backend keeps. A flat 200 MB is a conservative
// approximation; in practice it scales somewhat with model + context size.
const GPU_VARIANT_OVERHEAD_MB = 200;

export function isMobileDevice() {
  return detectMobileFamily() !== null;
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
  yieldMs = 0,
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

      // Yield so we don't starve the main thread / GC. On mobile a
      // longer yield also gives the OS a chance to update its memory
      // accounting between steps so a fast burst doesn't look like a
      // spike to Jetsam.
      await new Promise((r) => setTimeout(r, yieldMs));
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

  const mobileFamily = detectMobileFamily();
  const isMobile = mobileFamily !== null;

  // ── Mobile path: static heap budget, bounded GPU probe ──
  //
  // Heap stays static — the heap probe itself can trip Jetsam (commit
  // 6f33b5d), and the working-set floor matters more than a precise
  // number anyway.
  //
  // GPU runs a *bounded* probe: we read maxBufferSize from the adapter
  // (free, no allocation), pick a per-tier hard cap from
  // getMobileProbeCapMB, then probe with small 128 MB steps and 50 ms
  // yields up to that cap. This gives us a real measurement on capable
  // devices (e.g. iPhone 17 Pro Max gets ~850 MB instead of the 700 MB
  // static fallback) without risking the unbounded behavior that tripped
  // Jetsam in commit 4f567a5. If the probe OOMs partway, we use
  // `probed - margin`. If it returns less than the static fallback or
  // fails entirely, we use the static fallback.
  if (isMobile) {
    const { heap: heapBudgetMB, gpu: staticGpuBudgetMB } = getMobileBudgetMB(mobileFamily);

    // Read adapter limits without allocating a device buffer.
    let maxBufferSizeMB = 0;
    let adapterReadError = null;
    try {
      if (navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        const lim = adapter?.limits?.maxBufferSize;
        if (typeof lim === 'number') {
          maxBufferSizeMB = Math.floor(lim / (1024 * 1024));
        }
      } else {
        adapterReadError = 'WebGPU not available';
      }
    } catch (err) {
      adapterReadError = err.message;
    }

    const probeCap = getMobileProbeCapMB(mobileFamily, maxBufferSizeMB);
    const probeBestCase = probeCap - MOBILE_PROBE_SAFETY_MARGIN_MB;

    // Skip the probe if even a successful run can't beat the static
    // fallback — allocating ~probeCap of GPU buffers on a low-RAM iPhone
    // (e.g. iPhone 13 with 6 GB) can itself trip Jetsam, and there's
    // no payoff if we'd have used staticGpuBudgetMB regardless.
    const probeWorthIt = probeBestCase > staticGpuBudgetMB;

    // Crash-loop guard: if a previous probe in this session crashed the
    // tab, we never made it back to the post-probe sentinel clear, so the
    // sentinel is still set on this load. Skip the probe until the user
    // closes the tab (clears sessionStorage).
    let prevProbeCrashed = false;
    try {
      prevProbeCrashed = sessionStorage.getItem(MOBILE_PROBE_SENTINEL_KEY) === '1';
    } catch { /* sessionStorage may be disabled */ }

    let gpuProbe = { probedMB: 0, error: null };
    let probeSkipReason = null;
    if (prevProbeCrashed) {
      probeSkipReason = 'previous probe crashed tab';
    } else if (!probeWorthIt) {
      probeSkipReason = `probe ceiling ${probeBestCase} MB ≤ static ${staticGpuBudgetMB} MB`;
    } else {
      try { sessionStorage.setItem(MOBILE_PROBE_SENTINEL_KEY, '1'); } catch { /* noop */ }
      gpuProbe = await probeGpuBudgetMB({
        stepMB: MOBILE_PROBE_STEP_MB,
        maxMB: probeCap,
        timeoutMs: MOBILE_PROBE_TIMEOUT_MS,
        yieldMs: MOBILE_PROBE_YIELD_MS,
      });
      try { sessionStorage.removeItem(MOBILE_PROBE_SENTINEL_KEY); } catch { /* noop */ }
    }

    const margined = gpuProbe.probedMB - MOBILE_PROBE_SAFETY_MARGIN_MB;
    let gpuBudgetMB;
    let source;
    if (probeSkipReason) {
      gpuBudgetMB = staticGpuBudgetMB;
      source = `mobile probe skipped (${probeSkipReason}), using static ${staticGpuBudgetMB} MB for ${mobileFamily}`;
    } else if (gpuProbe.probedMB > 0 && margined > staticGpuBudgetMB) {
      gpuBudgetMB = margined;
      const hitCap = gpuProbe.probedMB + MOBILE_PROBE_STEP_MB > probeCap;
      const detail = hitCap
        ? `hit cap ${probeCap} MB`
        : `stopped at ${gpuProbe.probedMB} MB (OOM)`;
      source = `mobile probe — ${mobileFamily}, ${detail}, using ${gpuBudgetMB} MB (− ${MOBILE_PROBE_SAFETY_MARGIN_MB} MB margin)`;
    } else {
      gpuBudgetMB = staticGpuBudgetMB;
      if (gpuProbe.probedMB > 0) {
        source = `mobile probe — ${mobileFamily}, only ${gpuProbe.probedMB} MB measured (below static floor), using static ${staticGpuBudgetMB} MB`;
      } else {
        source = `mobile probe failed (${gpuProbe.error || 'unknown'}), using static ${staticGpuBudgetMB} MB for ${mobileFamily}`;
      }
    }

    const adapterDetail = adapterReadError
      ? ` (adapter read failed: ${adapterReadError})`
      : maxBufferSizeMB > 0
        ? ` (maxBufferSize ${maxBufferSizeMB} MB → probe cap ${probeCap} MB)`
        : '';

    return {
      budgetMB: gpuBudgetMB,
      gpuBudgetMB,
      heapBudgetMB,
      memGB,
      quotaMB,
      probedMB: 0,
      gpuProbedMB: gpuProbe.probedMB,
      probeError: 'skipped on mobile (heap probe can trip Jetsam)',
      gpuProbeError: gpuProbe.error || (probeSkipReason ? `skipped: ${probeSkipReason}` : null),
      isMobile: true,
      mobileFamily,
      source: source + adapterDetail,
      heapSource: `mobile static budget — ${mobileFamily} (WASM heap ${heapBudgetMB} MB for KV + compute scratch)`,
    };
  }

  // ── Desktop path: real probes ──
  const [heapProbe, gpuProbe] = await Promise.all([
    probeHeapBudgetMB(),
    probeGpuBudgetMB(),
  ]);

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

  let gpuBudgetMB;
  let gpuSource;
  if (gpuProbe.probedMB > 0) {
    gpuBudgetMB = gpuProbe.probedMB;
    gpuSource = `probe (WebGPU buffers, ${gpuProbe.probedMB} MB allocated)`;
  } else {
    gpuBudgetMB = 0;
    gpuSource = `probe failed: ${gpuProbe.error || 'unknown'}`;
  }

  return {
    budgetMB: gpuBudgetMB,
    gpuBudgetMB,
    heapBudgetMB,
    memGB,
    quotaMB,
    probedMB: heapProbe.probedMB,
    gpuProbedMB: gpuProbe.probedMB,
    probeError: heapProbe.error || null,
    gpuProbeError: gpuProbe.error || null,
    isMobile: false,
    mobileFamily: null,
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
