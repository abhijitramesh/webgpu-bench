// Empirical WASM heap probe. Runs in a Dedicated Worker so if growing the
// WebAssembly.Memory hits the engine's hard limit and aborts, the worker
// dies instead of the tab. The main thread treats worker death as "probe
// failed" and falls back to heuristics.
//
// We measure the same allocator that llama.cpp's WASM build uses for its
// linear memory: a single WebAssembly.Memory object, grown one chunk at a
// time. This matches what wllama does and answers the right question —
// "how big can the WASM heap actually become on this device?" — without
// any JS-side proxy (ArrayBuffer is not the same allocator).
//
// Protocol:
//   main → worker: { stepPages?: number, maxPages?: number }
//                  defaults: 2048 pages (128 MiB) and 65536 pages (4 GiB)
//   worker → main: { committedMB: number, failedAtPagesGrowth?: number }
//
// The wasm32 linear memory is capped at 4 GiB by the engine regardless of
// how much physical RAM the device has, so probing past 65536 pages is
// pointless. On iOS Safari the cap will land much lower (~1 GiB) and the
// grow() throws.

const PAGE_BYTES = 64 * 1024;
const DEFAULT_STEP_PAGES = 2048;     // 128 MiB
const DEFAULT_MAX_PAGES = 65536;     // 4 GiB (wasm32 hard cap)

self.onmessage = async (e) => {
  const stepPages = Number(e.data?.stepPages) || DEFAULT_STEP_PAGES;
  const maxPages = Number(e.data?.maxPages) || DEFAULT_MAX_PAGES;

  let memory;
  try {
    memory = new WebAssembly.Memory({ initial: 1, maximum: maxPages });
  } catch (err) {
    // Engine couldn't even reserve the address space. Treat as 64 KiB
    // (the initial page) and let main thread fall back.
    self.postMessage({ committedMB: 0, error: `Memory ctor failed: ${err.message}` });
    return;
  }

  let committedPages = 1;
  let failedAt = null;

  while (committedPages + stepPages <= maxPages) {
    try {
      memory.grow(stepPages);
      committedPages += stepPages;
      // Touch the last byte of the just-grown region to force a real commit
      // — engines can be lazy about backing pages with physical memory until
      // first write, and we want the probe to reflect actual capacity.
      const view = new Uint8Array(memory.buffer);
      view[committedPages * PAGE_BYTES - 1] = 1;
    } catch (err) {
      failedAt = stepPages;
      break;
    }
    // Yield so the main thread / GC can breathe.
    await new Promise((r) => setTimeout(r, 5));
  }

  // Try one final smaller step in case we have headroom under the last
  // failure but above committedPages. Halve until we either succeed or hit
  // the noise floor.
  if (failedAt !== null) {
    let halfStep = Math.floor(stepPages / 2);
    while (halfStep >= 16 && committedPages + halfStep <= maxPages) {
      try {
        memory.grow(halfStep);
        committedPages += halfStep;
      } catch {
        halfStep = Math.floor(halfStep / 2);
      }
    }
  }

  const committedMB = Math.floor((committedPages * PAGE_BYTES) / (1024 * 1024));
  self.postMessage({ committedMB, failedAtPagesGrowth: failedAt });
};
