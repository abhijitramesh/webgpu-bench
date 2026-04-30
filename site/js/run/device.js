// Device-fit helpers for the interactive bench page.
//
// getDeviceBudgetMB() empirically probes the WASM heap's actual maximum
// growth on this device, mirroring how llama.cpp itself allocates (a single
// WebAssembly.Memory grown in pages). The probe runs in a worker so an
// allocation failure dies harmlessly. We fall back to deviceMemory /
// storage.estimate heuristics if the probe can't run.
//
// The probed value is the budget — no extra safety factor. variantFits()
// already multiplies the GGUF size by 1.5× to cover the WASM heap +
// activations + KV cache + WebGPU staging beyond the file size.
//
// On wasm32 the linear memory caps at 4 GiB no matter how much physical
// RAM the device has, so probe results above 4096 MB cannot exist.

const DEFAULT_BUDGET_MB = 2 * 1024;
const HOSTED_QUOTA_FRACTION = 0.4;
const HOSTED_QUOTA_CAP_MB = 8 * 1024;

// Hard ceiling on mobile regardless of probe result. iOS/Android can reap
// the tab under system memory pressure without raising a JS error the
// probe could observe, so an "ok at 4 GiB" result is not safe to trust on
// a phone — the OS gets the last word.
//
// Empirically iOS Safari tabs get reaped well below the WebAssembly.Memory
// engine cap (~1 GiB on iPhone), and Android Chrome on mid-range devices
// behaves similarly. Below 500 MB heap usage tends to be safe across
// modern phones; above that we start seeing tab kills mid-run. The OPFS-
// streaming model load means model bytes no longer live on the WASM heap,
// so this budget now caps KV cache + compute scratch + JS heap headroom,
// not the model file itself — fitting variants no longer means "model
// fits in memory" but "the per-step working set fits". variantFits
// applies the per-variant 1.5× factor on top of this for safety margin.
const MOBILE_BUDGET_CEILING_MB = 500;

const PROBE_TIMEOUT_MS = 15_000;

export function isMobileDevice() {
  if (typeof navigator === 'undefined') return false;
  if (navigator.userAgentData?.mobile === true) return true;
  const ua = navigator.userAgent || '';
  return /iPhone|iPad|iPod|Android.*Mobile/.test(ua);
}

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

// Cache the budget for the lifetime of the page load. controller.js calls
// both getDeviceBudgetMB() and describeDevice() at mount, and we don't want
// to run the 1–2 s probe twice.
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

  // Primary: empirical probe. The worker grows a WebAssembly.Memory page
  // by page and reports how far it got — that's literally the WASM heap
  // ceiling on this device, capped at wasm32's 4 GiB. We trust it directly;
  // variantFits() applies the per-variant 1.5× overhead.
  const probe = await probeHeapBudgetMB();
  const probedMB = probe.probedMB;

  let budgetMB;
  let source;
  if (probedMB > 0) {
    budgetMB = probedMB;
    source = `probe (WASM heap, ${probedMB} MB committed)`;
  } else if (memGB !== null) {
    budgetMB = memGB * 1024 * 0.6;
    source = 'navigator.deviceMemory (probe failed)';
  } else if (quotaMB !== null) {
    budgetMB = Math.min(quotaMB * HOSTED_QUOTA_FRACTION, HOSTED_QUOTA_CAP_MB);
    source = 'navigator.storage.estimate().quota (probe failed)';
  } else {
    budgetMB = DEFAULT_BUDGET_MB;
    source = 'default (probe failed)';
  }

  if (isMobile && budgetMB > MOBILE_BUDGET_CEILING_MB) {
    budgetMB = MOBILE_BUDGET_CEILING_MB;
    source += ' → mobile-capped';
  }

  return {
    budgetMB,
    memGB,
    quotaMB,
    probedMB,
    probeError: probe.error || null,
    isMobile,
    source,
  };
}

export function variantFits(sizeMB, budgetMB, overhead = 1.5) {
  if (typeof sizeMB !== 'number' || sizeMB <= 0) return false;
  if (typeof budgetMB !== 'number' || budgetMB <= 0) return false;
  return sizeMB * overhead <= budgetMB;
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
