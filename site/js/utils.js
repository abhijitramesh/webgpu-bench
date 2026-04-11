export const BROWSER_COLORS = {
  chromium: '#60a5fa',
  firefox: '#fb923c',
  webkit: '#a78bfa',
};

// Quantization types sorted by approximate bit-width (low -> high)
export const QUANT_ORDER = [
  'IQ1_S', 'IQ1_M',
  'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ2_M', 'Q2_K', 'Q2_K_S',
  'IQ3_XXS', 'IQ3_XS', 'IQ3_S', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L',
  'IQ4_NL', 'IQ4_XS', 'Q4_0', 'Q4_K_S', 'Q4_K_M',
  'Q5_0', 'Q5_K_S', 'Q5_K_M',
  'Q6_K',
  'Q8_0',
  'F16',
  'F32',
  'BF16',
];

export function quantSortKey(q) {
  const idx = QUANT_ORDER.indexOf(q);
  return idx >= 0 ? idx : QUANT_ORDER.length;
}

export function formatTokS(v) {
  if (v == null || isNaN(v)) return '\u2014';
  return v.toFixed(1);
}

export function formatMs(v) {
  if (v == null || isNaN(v)) return '\u2014';
  return v.toFixed(1);
}

export function categorizeError(err) {
  if (!err) return null;
  const e = err.toLowerCase();
  if (e.includes('out of memory') || e.includes('oom') || e.includes('memory allocation')) return 'OOM';
  if (e.includes('wasm') || e.includes('abort') || e.includes('unreachable')) return 'WASM Abort';
  if (e.includes('timeout') || e.includes('timed out')) return 'Timeout';
  if (e.includes('download') || e.includes('fetch') || e.includes('404') || e.includes('network')) return 'Download Failed';
  return 'Other';
}

export function groupBy(arr, keyFn) {
  const map = {};
  for (const item of arr) {
    const key = typeof keyFn === 'function' ? keyFn(item) : item[keyFn];
    if (!map[key]) map[key] = [];
    map[key].push(item);
  }
  return map;
}
