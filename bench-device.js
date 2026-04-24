// Device-fit helpers for the interactive bench page.
//
// getDeviceBudgetMB() returns an estimate of how much RAM we can reasonably
// dedicate to a GGUF file in this browser, alongside the raw numbers so the
// UI can explain the calculation to the user.
//
// variantFits() is the per-variant predicate used to auto-uncheck rows that
// won't fit. Overhead default of 1.5× covers the WASM heap + activations
// buffer beyond the raw file size.

const DEFAULT_BUDGET_MB = 4 * 1024;
const HOSTED_QUOTA_FRACTION = 0.4;
const HOSTED_QUOTA_CAP_MB = 8 * 1024;

export async function getDeviceBudgetMB() {
  const memGB = typeof navigator.deviceMemory === 'number' ? navigator.deviceMemory : null;
  let quotaMB = null;
  try {
    const est = await navigator.storage?.estimate?.();
    if (est?.quota) quotaMB = est.quota / (1024 * 1024);
  } catch {
    // fall through — some browsers throw on storage.estimate in non-secure contexts
  }

  let budgetMB;
  let source;
  if (memGB !== null) {
    // Chromium path. deviceMemory is rounded to {0.25, 0.5, 1, 2, 4, 8} GB.
    budgetMB = memGB * 1024 * 0.6;
    source = 'navigator.deviceMemory';
  } else if (quotaMB !== null) {
    budgetMB = Math.min(quotaMB * HOSTED_QUOTA_FRACTION, HOSTED_QUOTA_CAP_MB);
    source = 'navigator.storage.estimate().quota';
  } else {
    budgetMB = DEFAULT_BUDGET_MB;
    source = 'default';
  }

  return { budgetMB, memGB, quotaMB, source };
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
  return {
    ...budget,
    webgpu: !!navigator.gpu,
    gpu,
    userAgent: navigator.userAgent,
    platform: navigator.platform ?? null,
  };
}
