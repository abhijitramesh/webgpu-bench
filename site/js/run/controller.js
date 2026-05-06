// Run-tab controller. Mounts into the existing #run-section subtree and
// drives the one-click benchmark UI using the dashboard's design-system
// classes. Detects `surface` (localhost / space / pages) to gate the
// server save checkbox and the HF hub sign-in/submit row.

import { ggufSource, inventoryOpfs, purgeOpfs, OPFS_ROOT_NAME } from './source.js';
import { getDeviceBudgetMB, variantFits, describeDevice, isMobileDevice } from './device.js';
import {
  resumeHFSession, beginHFSignIn, signOutHF, submitResultsToDataset,
  HF_OAUTH_PENDING_KEY,
} from './hub.js';
import { isHubConfigured, HF_DATASET_REPO, CONSISTENCY_PROMPT } from './config.js';

const RUN_INTENT_STORAGE_KEY = 'webgpu-bench:runIntent';
const USER_REPORTED_STORAGE_KEY = 'webgpu-bench:userReported';
const CRASH_STALE_MS = 10_000;

const DEFAULT_N_PREDICT = 128;
const DEFAULT_N_CTX = 2048;
const DEFAULT_N_GPU_LAYERS = 999;
const YIELD_BETWEEN_RUNS_MS = 500;
// iOS Safari needs much longer to actually release Metal/WebGPU buffer
// allocations after worker.terminate() — back-to-back runs at the desktop
// 500 ms cadence trip Jetsam and Safari reloads the tab. 4 s gives the
// GPU process room to drain. Android Chromium is more forgiving but
// shares the same code path here.
const MOBILE_YIELD_BETWEEN_RUNS_MS = 4_000;
// llama-bench defaults: -p 512 -n 128 -r 5
const DEFAULT_N_PROMPT = 512;
const DEFAULT_N_GEN = 128;
const DEFAULT_N_DEPTH = 2048;
const DEFAULT_ITERATIONS = 5;
const MIN_ITERATIONS_FOR_SUBMIT = 5;

const state = {
  surface: 'pages',    // 'localhost' | 'space' | 'pages' | 'file'
  source: null,        // ggufSource() — single OPFS-backed source
  models: null,        // parsed models.json
  budget: null,        // { budgetMB, memGB, quotaMB, probedMB, isMobile, source }
  device: null,        // describeDevice() output
  cacheStatus: {},     // { 'repo/file': { cachedBytes } }
  variants: [],        // flat variant rows with metadata
  running: false,
  aborted: false,
  results: [],         // result records from the current session
  hfSession: null,     // { accessToken, expiresAt, userName } when signed in
  iterations: DEFAULT_ITERATIONS,
  nPrompt: DEFAULT_N_PROMPT,
  nGen: DEFAULT_N_GEN,
  nDepth: DEFAULT_N_DEPTH,
  // True while a Run Study is in flight (or a restored study session).
  // Drives the progress table layout: study mode renders pp/tg as
  // d=0 / d=N column pairs so both passes' numbers stay visible
  // instead of the d=N pass overwriting d=0.
  studyMode: false,
  // User-controlled phase toggles. Both default OFF — a Run (or Run Study)
  // does GPU perf only unless the user explicitly opts in to the CPU
  // baseline. The CPU pass is the slowest step on most devices and most
  // submissions don't need its consistency / comparison output, so making
  // it opt-in keeps the default experience fast.
  runConsistency: false,
  runCpuPerf: false,
  mounted: false,
  // Tracks variants the Run pipeline downloaded this session (as opposed to
  // the standalone Download button or pre-existing cache). Only these are
  // candidates for post-run eviction when the user has opted in.
  sessionDownloads: new Set(),
  // Handle to the currently-running worker, so Abort can terminate it.
  currentWorker: null,
  // Set of fns that abort an in-flight async op (worker terminate, fetch
  // signal abort). Multiple concurrent ops register here — Run study has a
  // worker running variant i AND a prefetch downloading variant i+1, both
  // of which need to be cancellable. Abort handler iterates the whole set.
  abortHandlers: new Set(),
  // Build metadata fetched from `build/<variant>/build-info.json`. Stamped
  // onto every result record so we can compare performance across llama.cpp
  // versions. JSPI and Asyncify variants are built from the same source
  // tree, so a single fetch is enough; both files would be identical.
  buildInfo: null,
  // User-reported machine identity (Machine Name / GPU Name / Browser /
  // OS). Filled by the "Your machine" form on the Run page, persisted to
  // localStorage between visits, and stamped onto every result record so
  // the leaderboard can attribute submissions even when UA / WebGPU
  // adapter info is missing or wrong. machineName/browser/os are required
  // before submission; gpuName is optional.
  userReported: { machineName: '', gpuName: '', browser: '', os: '' },
};

const USER_REPORTED_REQUIRED = ['machineName', 'browser', 'os'];

function loadUserReported() {
  try {
    const raw = localStorage.getItem(USER_REPORTED_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object') return parsed;
  } catch { /* corrupt storage */ }
  return null;
}

function saveUserReported() {
  try {
    localStorage.setItem(USER_REPORTED_STORAGE_KEY, JSON.stringify(state.userReported));
  } catch { /* quota / disabled */ }
}

// Register an abort callback for an in-flight async op (worker terminate,
// fetch signal abort, etc.). Returns an unregister fn the caller MUST
// invoke when the op settles, so we don't accumulate stale handlers across
// runs. Abort handler iterates state.abortHandlers and calls every fn.
function registerAbort(fn) {
  state.abortHandlers.add(fn);
  return () => state.abortHandlers.delete(fn);
}

async function loadBuildInfo() {
  // Try jspi first (Chrome path), fall back to asyncify (Safari/Firefox path).
  // Either contains the same llama.cpp commit/describe.
  const candidates = ['./build/jspi/build-info.json', './build/asyncify/build-info.json'];
  for (const url of candidates) {
    try {
      const r = await fetch(url, { cache: 'no-cache' });
      if (!r.ok) continue;
      const data = await r.json();
      if (data && (data.llamaCppCommit || data.llamaCppDescribe)) return data;
    } catch { /* try next */ }
  }
  return null;
}

// ──────────────── surface detection ────────────────

async function detectSurface() {
  const params = new URLSearchParams(location.search);
  if (params.get('mode') === 'local') return 'localhost';
  if (params.get('mode') === 'hosted') return 'space';
  if (/\.static\.hf\.space$/.test(location.hostname)) return 'space';
  if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') {
    try {
      const r = await fetch('/api/models', { method: 'HEAD' });
      if (r.ok) return 'localhost';
    } catch { /* no backend */ }
  }
  if (location.protocol === 'file:') return 'file';
  // Fallback for any other hosted location (mirror, preview deploy, etc.).
  // Read-only: Submit hidden, no backend save.
  return 'pages';
}

function canSubmit() {
  return state.surface === 'localhost'
    || (state.surface === 'space' && isHubConfigured());
}

// ──────────────── data loading ────────────────

async function loadModels() {
  // Page lives at /site/run.html locally and /run.html on the HF Space
  // (flattened root). Sibling `./models.json` works in both; `/api/models`
  // is the Express backend only.
  const candidates = state.surface === 'localhost'
    ? ['/api/models', './models.json', '/models.json']
    : ['./models.json', '/models.json'];
  let lastErr = null;
  for (const url of candidates) {
    try {
      const r = await fetch(url);
      if (r.ok) return await r.json();
      lastErr = new Error(`${url} → ${r.status}`);
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error('Could not load models.json');
}

async function loadCacheStatus() {
  // Cache lives in OPFS on every surface — ggufSource writes through
  // the same `opfsHandleForModel` path everywhere.
  try {
    return await inventoryOpfs();
  } catch (err) {
    console.warn('OPFS inventory failed:', err.message);
    return {};
  }
}

// ──────────────── variant bookkeeping ────────────────

function flattenVariants(models) {
  const out = [];
  for (const m of models.models || []) {
    for (const v of m.variants || []) {
      out.push({
        modelName: m.name,
        repo: m.repo,
        quant: v.quant,
        filename: v.filename,
        sizeMB: typeof v.sizeMB === 'number' ? v.sizeMB : 0,
        warnings: computeWarnings(m.name, v.quant),
      });
    }
  }
  return out;
}

function getQuickVariantSet() {
  const list = state.models?.quickVariants;
  return new Set(Array.isArray(list) && list.length ? list : ['Q2_K', 'Q4_K_M', 'Q8_0']);
}

function isQuickVariant(v) {
  return getQuickVariantSet().has(v.quant);
}

function computeWarnings(modelName, quant) {
  // SSM_SCAN and Q1_0 are both supported in the bundled llama.cpp
  // (ggml-webgpu.cpp). granite-4 ran cleanly in the apr-30 run; Q1_0 is
  // wired into the fast-path dequant table. No warnings to surface today.
  return [];
}

function cacheKey(v) { return `${v.repo}/${v.filename}`; }
function variantFitsDevice(v) {
  // New variantFits signature: pass both budgets so the predicate can
  // check (a) model fits in GPU memory + small overhead, and (b) WASM
  // heap can hold the working set. See device.js for the rationale.
  return variantFits(v.sizeMB, {
    gpuBudgetMB: state.budget.gpuBudgetMB,
    heapBudgetMB: state.budget.heapBudgetMB,
  });
}
function isCached(v) {
  const entry = state.cacheStatus[cacheKey(v)];
  return !!entry && entry.cachedBytes > 0;
}

function groupByFamily(variants) {
  const map = new Map();
  for (const v of variants) {
    if (!map.has(v.modelName)) map.set(v.modelName, []);
    map.get(v.modelName).push(v);
  }
  return map;
}

// ──────────────── rendering ────────────────

function $(id) { return document.getElementById(id); }

/* Pretty browser name + version. Prefers UA Client Hints (clean
   { brand, version } pairs) over UA-string regex parsing. The brand list
   is ordered Chromium-favoured, so pick the most-specific brand the user
   actually has (Edg → Chrome → Chromium). */
function formatBrowser(d) {
  const preferred = ['Microsoft Edge', 'Edg', 'Opera', 'Brave', 'Arc', 'Vivaldi',
                     'Google Chrome', 'Chromium'];
  const brands = d.uaBrands || [];
  for (const name of preferred) {
    const hit = brands.find(b => b.brand === name);
    if (hit) return `${hit.brand} ${hit.version}`;
  }
  if (brands.length > 0) return `${brands[0].brand} ${brands[0].version}`;

  // Non-Chromium fallback: regex on userAgent. Capture brand + version
  // separately so the slash isn't visible.
  const m = (d.userAgent || '').match(/(Firefox|FxiOS|Edg|CriOS|Chrome|Version)\/([\d.]+)/);
  if (!m) return 'browser';
  const brand = m[1] === 'Version' ? 'Safari' : (m[1] === 'CriOS' ? 'Chrome iOS' : (m[1] === 'FxiOS' ? 'Firefox iOS' : m[1]));
  return `${brand} ${m[2]}`;
}

/* Pretty OS + architecture. `navigator.platform` is unreliable on Apple
   Silicon (it returns "MacIntel" for back-compat); prefer UA-CH and fall
   back to the WebGPU vendor as a strong arm64 signal on Macs. */
function formatPlatform(d) {
  const ua = d.userAgent || '';
  const platHint = (d.uaPlatform || d.platform || '').toLowerCase();
  let os;
  if (platHint.includes('mac') || /Mac/.test(ua)) os = 'macOS';
  else if (platHint.includes('win') || /Win/.test(ua)) os = 'Windows';
  else if (/iPhone|iPad|iPod/.test(ua) || platHint.includes('ios')) os = 'iOS';
  else if (/Android/.test(ua) || platHint.includes('android')) os = 'Android';
  else if (platHint.includes('linux') || /Linux/.test(ua)) os = 'Linux';
  else os = d.uaPlatform || d.platform || 'unknown';

  let arch = '';
  if (d.uaArch === 'arm') arch = 'arm64';
  else if (d.uaArch === 'x86') arch = 'x86_64';
  else if (d.uaArch) arch = d.uaArch;
  else if (os === 'macOS' && d.gpu?.vendor === 'apple') arch = 'arm64';
  else if (os === 'iOS') arch = 'arm64';
  else if (/arm|aarch/i.test(ua)) arch = 'arm64';
  else if (/x86_64|Win64;|x64/i.test(ua)) arch = 'x86_64';

  return arch ? `${os} · ${arch}` : os;
}

function renderHeader() {
  const d = state.device;
  const b = state.budget;

  const badge = $('run-mode-badge');
  if (badge) {
    const labels = {
      localhost: 'Local dev',
      space: 'Hosted · Hugging Face',
      pages: 'Read-only preview',
      file: 'Local file',
    };
    badge.textContent = labels[state.surface] || state.surface;
    badge.className = `badge run-mode-badge run-mode-${state.surface}`;
  }

  const browserStr = formatBrowser(d);
  const platformStr = formatPlatform(d);
  const gpuStr = d.gpu
    ? [d.gpu.vendor, d.gpu.architecture, d.gpu.device].filter(Boolean).join(' ').trim()
    : '';

  $('device-browser').textContent = browserStr;
  $('device-platform').textContent = platformStr;
  $('device-gpu').textContent = gpuStr || (d.webgpu ? 'WebGPU (no info)' : 'no WebGPU');

  const memStr = b.memGB !== null ? `${b.memGB} GB` : '—';
  $('device-memory').textContent = memStr;

  // budgetMB is now the GPU-memory budget (per device.js _computeBudget),
  // since with OPFS streaming the model lives in WebGPU buffers, not the
  // WASM heap. We surface the heap budget separately in the source line so
  // a curious reader can see both probes' results.
  const budgetGB = (b.budgetMB / 1024).toFixed(1);
  const heapGB = (b.heapBudgetMB / 1024).toFixed(1);
  $('device-budget').textContent = `${budgetGB} GB`;
  $('device-budget-source').textContent = `GPU memory · WASM heap: ${heapGB} GB`;

  const webgpuCell = $('device-webgpu');
  if (webgpuCell) {
    webgpuCell.textContent = d.webgpu ? 'yes' : 'no';
    webgpuCell.classList.toggle('text-success', d.webgpu);
    webgpuCell.classList.toggle('text-error', !d.webgpu);
  }

  const llamaCell = $('device-llamacpp');
  if (llamaCell) {
    const bi = state.buildInfo;
    if (bi?.llamaCppCommit) {
      const label = bi.llamaCppDescribe || bi.llamaCppCommit.slice(0, 10);
      llamaCell.innerHTML = '';
      const a = document.createElement('a');
      a.href = `https://github.com/ggml-org/llama.cpp/commit/${bi.llamaCppCommit}`;
      a.target = '_blank';
      a.rel = 'noopener';
      a.className = 'mono';
      a.textContent = label;
      llamaCell.appendChild(a);
    } else {
      llamaCell.textContent = '—';
    }
  }

  // Surface-dependent UI gating.
  const hubRow = $('hub-row');
  if (hubRow) hubRow.hidden = state.surface !== 'space';

  const saveLocalRow = $('save-local-row');
  if (saveLocalRow) saveLocalRow.hidden = state.surface !== 'localhost';

  const pagesBanner = $('run-pages-banner');
  if (pagesBanner) pagesBanner.hidden = state.surface !== 'pages';

  const mobileBanner = $('run-mobile-banner');
  if (mobileBanner) mobileBanner.hidden = !state.budget?.isMobile;

  const purgeBtn = $('btn-purge');
  // Cache lives in OPFS on every surface now, so the Purge button is
  // always meaningful. Was hidden on localhost back when the disk-cache
  // path lived on the server.
  if (purgeBtn) purgeBtn.hidden = false;

  renderHfSection();
}

function renderHfSection() {
  if (state.surface !== 'space') return;
  const signinBtn = $('btn-signin');
  const submitBtn = $('btn-submit');
  const userEl = $('hf-user');
  if (!signinBtn || !submitBtn || !userEl) return;

  if (!isHubConfigured()) {
    signinBtn.disabled = true;
    signinBtn.textContent = 'HF hub not configured';
    signinBtn.title = 'Set HF_DATASET_REPO in site/js/run/config.js';
    submitBtn.hidden = true;
    userEl.textContent = '';
    return;
  }

  if (state.hfSession) {
    signinBtn.textContent = 'Sign out';
    // Sign-out itself is fine mid-run, but stay consistent with the disabled
    // sign-in state so the row doesn't toggle look mid-run.
    signinBtn.disabled = state.running;
    submitBtn.hidden = false;
    const eligible = submittableResults();
    submitBtn.disabled = state.running || eligible.length === 0;
    submitBtn.title = state.running
      ? 'Wait for the benchmark to finish before submitting'
      : (eligible.length === 0 && state.results.length > 0
        ? `Need at least ${MIN_ITERATIONS_FOR_SUBMIT} successful iterations per variant to submit`
        : '');
    const who = state.hfSession.userName ? `@${state.hfSession.userName}` : 'signed in';
    const hint = eligible.length > 0
      ? ` · ${eligible.length}/${state.results.length} variants eligible`
      : '';
    userEl.textContent = `${who} · → ${HF_DATASET_REPO}${hint}`;
  } else {
    signinBtn.textContent = 'Sign in with Hugging Face';
    // Sign-in triggers a full-page redirect, which would kill an in-flight
    // worker. Disable the button while the benchmark is running so the user
    // can't accidentally lose their run; results are saved progressively to
    // localStorage and restored on the next mount, so finishing the run and
    // signing in afterwards still lets them submit.
    signinBtn.disabled = state.running;
    signinBtn.title = state.running
      ? 'Wait for the benchmark to finish before signing in'
      : '';
    submitBtn.hidden = true;
    userEl.textContent = '';
  }
}

function renderModels() {
  const panel = $('run-models');
  panel.innerHTML = '';

  const groups = groupByFamily(state.variants);
  for (const [family, variants] of groups) {
    const fitsCount = variants.filter(variantFitsDevice).length;
    const quickFitCount = variants.filter(v => isQuickVariant(v) && variantFitsDevice(v)).length;

    // Card wrapper (not <details>, to avoid nested-interactive with the
    // family-level checkbox). A dedicated toggle button expands/collapses
    // the variant list.
    const familyEl = document.createElement('section');
    familyEl.className = 'run-family card';
    familyEl.dataset.family = family;

    const header = document.createElement('div');
    header.className = 'run-family-summary';

    const toggleBtn = document.createElement('button');
    toggleBtn.type = 'button';
    toggleBtn.className = 'run-family-toggle';
    toggleBtn.setAttribute('aria-expanded', 'false');
    toggleBtn.setAttribute('aria-label', `Expand ${family}`);
    toggleBtn.innerHTML = '<span class="run-family-chevron" aria-hidden="true"></span>';

    const selectAllId = `run-family-all-${family.replace(/[^a-z0-9]/gi, '-')}`;
    const selectAll = document.createElement('input');
    selectAll.type = 'checkbox';
    selectAll.className = 'run-family-select-all';
    selectAll.dataset.family = family;
    selectAll.id = selectAllId;
    selectAll.setAttribute('aria-label', `Select all variants in ${family}`);

    const nameLabel = document.createElement('label');
    nameLabel.className = 'run-family-name';
    nameLabel.htmlFor = selectAllId;
    nameLabel.textContent = family;

    const paramChip = document.createElement('span');
    paramChip.className = 'run-family-params';
    const params = parseParamSize(family);
    if (params) paramChip.textContent = params;
    else paramChip.hidden = true;

    const stats = document.createElement('span');
    stats.className = 'run-family-stats';
    stats.textContent = `${variants.length} variants · ${fitsCount} fit · ${quickFitCount} quick`;

    header.append(toggleBtn, selectAll, nameLabel, paramChip, stats);
    familyEl.appendChild(header);

    const list = document.createElement('div');
    list.className = 'run-variant-list';
    list.hidden = true;

    for (const v of variants) {
      const row = document.createElement('label');
      row.className = 'run-variant-row';
      if (!variantFitsDevice(v)) row.classList.add('is-non-fit');
      row.dataset.key = cacheKey(v);

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.className = 'run-variant-select';
      cb.dataset.key = cacheKey(v);
      cb.checked = isQuickVariant(v) && variantFitsDevice(v);

      const quant = document.createElement('span');
      quant.className = 'run-variant-quant';
      quant.textContent = v.quant;

      const filename = document.createElement('code');
      filename.className = 'run-variant-file';
      filename.textContent = v.filename;

      const size = document.createElement('span');
      size.className = 'run-variant-size';
      size.textContent = v.sizeMB > 0 ? formatSize(v.sizeMB) : '?';

      const badges = document.createElement('span');
      badges.className = 'run-variant-badges';
      updateBadgesForVariant(badges, v);

      row.append(cb, quant, filename, size, badges);
      list.appendChild(row);
    }
    familyEl.appendChild(list);
    panel.appendChild(familyEl);

    updateFamilySelectAllState(family);
  }
}

function updateFamilySelectAllState(family) {
  const panel = $('run-models');
  if (!panel) return;
  const familyEl = panel.querySelector(
    `.run-family[data-family="${cssEscape(family)}"]`,
  );
  if (!familyEl) return;
  // Only count fit variants — the parent checkbox is intentionally limited
  // to toggling fits (non-fits would OOM). If we counted non-fits here too,
  // the parent could never reach "all checked" for any mixed family, which
  // wedges its underlying `checked` at false and turns subsequent clicks
  // into no-ops (see SmolLM3-3B: 21 fit / 24 variants).
  const rows = familyEl.querySelectorAll('.run-variant-row:not(.is-non-fit) .run-variant-select');
  const all = rows.length;
  const checked = [...rows].filter(cb => cb.checked).length;
  const selectAll = familyEl.querySelector('.run-family-select-all');
  if (!selectAll) return;
  selectAll.checked = checked === all && all > 0;
  selectAll.indeterminate = checked > 0 && checked < all;
}

function updateBadgesForVariant(badgesEl, v) {
  badgesEl.innerHTML = '';
  if (isCached(v)) badgesEl.appendChild(makeBadge('cached', 'badge--cached'));
  for (const w of v.warnings) badgesEl.appendChild(makeBadge(w, 'badge--warn'));
}

function refreshCacheBadge(v) {
  const row = document.querySelector(`.run-variant-row[data-key="${cssEscape(cacheKey(v))}"]`);
  if (!row) return;
  const badges = row.querySelector('.run-variant-badges');
  if (badges) updateBadgesForVariant(badges, v);
}

function makeBadge(text, cls) {
  const el = document.createElement('span');
  el.className = `badge ${cls}`;
  el.textContent = text;
  return el;
}

function formatSize(mb) {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(0)} MB`;
}

/* Pull a parameter-count hint (e.g. "1B", "270M", "0.6B") from a family
   name. Most family names embed this near the end (Llama-3.2-1B-Instruct,
   gemma-3-270m-it). Returns the LAST `<digits>[Bb|Mm]` token in the name,
   uppercased. Returns null if no match — chip is then hidden. */
function parseParamSize(name) {
  if (!name) return null;
  const matches = String(name).match(/(\d+\.?\d*)\s*[BbMm](?![A-Za-z])/g);
  if (!matches?.length) return null;
  const last = matches[matches.length - 1];
  return last.toUpperCase().replace(/\s+/g, '');
}

function escapeText(s) {
  return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}
function escapeAttr(s) { return escapeText(s).replace(/"/g, '&quot;'); }
function cssEscape(s) {
  if (window.CSS?.escape) return CSS.escape(s);
  return String(s).replace(/[^\w-]/g, ch => `\\${ch}`);
}

// ──────────────── selection / filters ────────────────

function wireSelectionHandlers() {
  const panel = $('run-models');
  panel.addEventListener('change', (e) => {
    const t = e.target;
    if (t.classList?.contains('run-family-select-all')) {
      const family = t.dataset.family;
      const rows = panel.querySelectorAll(
        `.run-family[data-family="${cssEscape(family)}"] .run-variant-row`,
      );
      // Only affect fit variants — checking non-fit can cause OOM on the
      // user's device, which is actively dangerous.
      rows.forEach(row => {
        if (row.classList.contains('is-non-fit')) return;
        const cb = row.querySelector('.run-variant-select');
        if (cb) cb.checked = t.checked;
      });
      updateFamilySelectAllState(family);
      updateButtons();
    } else if (t.classList?.contains('run-variant-select')) {
      const familyEl = t.closest('.run-family');
      if (familyEl) updateFamilySelectAllState(familyEl.dataset.family);
      updateButtons();
    }
  });
  panel.addEventListener('click', (e) => {
    // Clicks on the select-all checkbox or name label must not toggle
    // expansion — they have their own semantics.
    if (e.target.closest('.run-family-select-all, .run-family-name, .run-variant-list, .run-variant-row')) {
      return;
    }
    const header = e.target.closest?.('.run-family-summary');
    if (!header) return;
    const familyEl = header.closest('.run-family');
    const list = familyEl?.querySelector('.run-variant-list');
    const toggle = familyEl?.querySelector('.run-family-toggle');
    if (!list || !toggle) return;
    const expanded = !list.hidden;
    list.hidden = expanded;
    toggle.setAttribute('aria-expanded', String(!expanded));
    familyEl.classList.toggle('is-open', !expanded);
  });
}

function wireFilters() {
  ['hide-ud', 'hide-iq', 'hide-hifp'].forEach(id => {
    const el = $(id);
    if (el) el.addEventListener('change', applyFilters);
  });
}

function wireFamilySearch() {
  const input = $('family-search');
  if (!input) return;
  // Live-filter family cards on input. Match against the lowercased family
  // name; auto-expand any family that matches a non-empty query so the user
  // sees the relevant variants without an extra click.
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    document.querySelectorAll('.run-family').forEach(el => {
      const family = (el.dataset.family || '').toLowerCase();
      const match = q === '' || family.includes(q);
      el.hidden = !match;
      // Expand on match-with-query so variants are visible without a click.
      if (q !== '' && match) {
        const list = el.querySelector('.run-variant-list');
        const toggle = el.querySelector('.run-family-toggle');
        if (list && toggle) {
          list.hidden = false;
          toggle.setAttribute('aria-expanded', 'true');
          el.classList.add('is-open');
        }
      }
    });
  });
}

function wireBatchSelect() {
  const apply = (pred) => {
    document.querySelectorAll('.run-variant-select').forEach(cb => {
      const v = state.variants.find(x => cacheKey(x) === cb.dataset.key);
      cb.checked = pred(v);
    });
    document.querySelectorAll('.run-family').forEach(el => {
      if (el.dataset.family) updateFamilySelectAllState(el.dataset.family);
    });
    updateButtons();
  };
  $('btn-select-quick')?.addEventListener('click', () => {
    apply(v => !!v && isQuickVariant(v) && variantFitsDevice(v));
  });
  $('btn-select-fit')?.addEventListener('click', () => {
    apply(v => !!v && variantFitsDevice(v));
  });
  $('btn-select-none')?.addEventListener('click', () => {
    apply(() => false);
  });
}

function wirePerfInputs() {
  const reps = $('iterations-input');
  if (reps) {
    reps.value = String(state.iterations);
    reps.addEventListener('change', () => {
      const n = Math.max(1, Math.min(50, parseInt(reps.value, 10) || DEFAULT_ITERATIONS));
      state.iterations = n;
      reps.value = String(n);
    });
  }
  const np = $('n-prompt-input');
  if (np) {
    np.value = String(state.nPrompt);
    np.addEventListener('change', () => {
      const n = Math.max(0, Math.min(4096, parseInt(np.value, 10)));
      state.nPrompt = Number.isFinite(n) ? n : DEFAULT_N_PROMPT;
      np.value = String(state.nPrompt);
    });
  }
  const ng = $('n-gen-input');
  if (ng) {
    ng.value = String(state.nGen);
    ng.addEventListener('change', () => {
      const n = Math.max(0, Math.min(4096, parseInt(ng.value, 10)));
      state.nGen = Number.isFinite(n) ? n : DEFAULT_N_GEN;
      ng.value = String(state.nGen);
    });
  }
  const nd = $('n-depth-input');
  if (nd) {
    nd.value = String(state.nDepth);
    nd.addEventListener('change', () => {
      const n = Math.max(0, Math.min(32768, parseInt(nd.value, 10)));
      state.nDepth = Number.isFinite(n) ? n : DEFAULT_N_DEPTH;
      nd.value = String(state.nDepth);
    });
  }
  const runCons = $('run-consistency');
  if (runCons) {
    runCons.checked = state.runConsistency;
    runCons.addEventListener('change', () => {
      state.runConsistency = runCons.checked;
    });
  }
  const runCpu = $('run-cpu-perf');
  if (runCpu) {
    runCpu.checked = state.runCpuPerf;
    runCpu.addEventListener('change', () => {
      state.runCpuPerf = runCpu.checked;
    });
  }
}

function submittableResults() {
  return state.results.filter(r =>
    r.status === 'done' && (r.metrics?.iterations || 0) >= MIN_ITERATIONS_FOR_SUBMIT,
  );
}

function applyFilters() {
  const hideUd = $('hide-ud')?.checked;
  const hideIq = $('hide-iq')?.checked;
  const hideHifp = $('hide-hifp')?.checked;
  const hiddenByFamily = new Map();
  document.querySelectorAll('.run-variant-row').forEach(row => {
    const v = state.variants.find(x => cacheKey(x) === row.dataset.key);
    if (!v) return;
    const isUd = v.quant.startsWith('UD-');
    const isIq = /^IQ/.test(v.quant) || /^UD-IQ/.test(v.quant);
    const isHifp = /^(BF16|F16|bf16|f16)$/.test(v.quant);
    const hide = (hideUd && isUd) || (hideIq && isIq) || (hideHifp && isHifp);
    row.style.display = hide ? 'none' : '';
    if (hide) hiddenByFamily.set(v.modelName, (hiddenByFamily.get(v.modelName) || 0) + 1);
  });
  // Refresh the per-family stats line so users see hidden filter impact.
  document.querySelectorAll('.run-family').forEach(familyEl => {
    const family = familyEl.dataset.family;
    const all = [...familyEl.querySelectorAll('.run-variant-row')];
    const visible = all.filter(r => r.style.display !== 'none').length;
    const fit = all.filter(r => !r.classList.contains('is-non-fit') && r.style.display !== 'none').length;
    const quick = all.filter(r => {
      if (r.style.display === 'none' || r.classList.contains('is-non-fit')) return false;
      const v = state.variants.find(x => cacheKey(x) === r.dataset.key);
      return v && isQuickVariant(v);
    }).length;
    const stats = familyEl.querySelector('.run-family-stats');
    if (!stats) return;
    const hiddenCount = hiddenByFamily.get(family) || 0;
    const base = `${visible} variants · ${fit} fit · ${quick} quick`;
    stats.textContent = hiddenCount > 0 ? `${base} · ${hiddenCount} hidden` : base;
  });
  // A selected-but-now-hidden variant is a footgun; re-count the queue.
  updateButtons();
}

function getCheckedVariants() {
  return Array.from(document.querySelectorAll('.run-variant-select:checked'))
    .map(cb => state.variants.find(v => cacheKey(v) === cb.dataset.key))
    .filter(Boolean);
}

function updateButtons() {
  const checked = getCheckedVariants();
  const cachedChecked = checked.filter(isCached);
  const dl = $('btn-download'); if (dl) dl.disabled = state.running || checked.length === 0;
  // Run is now allowed even when nothing is cached — the pipeline downloads
  // on demand. (Download button remains for the "pre-cache without running"
  // workflow.)
  const rn = $('btn-run'); if (rn) rn.disabled = state.running || checked.length === 0;
  const study = $('btn-run-study'); if (study) study.disabled = state.running;
  const ab = $('btn-abort'); if (ab) { ab.disabled = !state.running; ab.hidden = !state.running; }
  renderBudgetMeter(checked, cachedChecked);
  // Keep the Sign in / Submit buttons in sync with the running flag — they
  // depend on it so the user can't kick off a redirect mid-run.
  renderHfSection();
}

/* Show selected size as a fill bar against the device's max model size.
   Three states drive the fill color: under (signal green), nearing (amber
   ≥ 70%), over (red ≥ 100%). When nothing is selected, hide the whole
   widget so the action bar isn't dominated by an empty meter. */
function renderBudgetMeter(checked, cachedChecked) {
  const widget = $('run-budget');
  const fill = $('run-budget-fill');
  const text = $('run-budget-text');
  const meta = $('run-budget-meta');
  if (!widget || !fill || !text || !meta) return;

  if (checked.length === 0) {
    widget.hidden = true;
    return;
  }
  widget.hidden = false;

  const totalMB = checked.reduce((a, v) => a + (v.sizeMB || 0), 0);
  const toDownload = checked.filter(v => !isCached(v));
  const dlMB = toDownload.reduce((a, v) => a + (v.sizeMB || 0), 0);
  const budgetMB = state.budget?.budgetMB || 0;

  // Largest single model is what really matters for the device — total is
  // download size, not peak memory. Show both.
  const largest = checked.reduce((m, v) => Math.max(m, v.sizeMB || 0), 0);
  const pct = budgetMB > 0 ? Math.min(100, (largest / budgetMB) * 100) : 0;

  fill.style.width = `${pct}%`;
  let tone = 'ok';
  if (budgetMB > 0 && largest > budgetMB) tone = 'over';
  else if (budgetMB > 0 && largest / budgetMB >= 0.7) tone = 'warn';
  widget.dataset.tone = tone;

  text.innerHTML = `<strong>${checked.length}</strong> selected · <span class="run-budget-size">${formatSize(totalMB)}</span> total`;
  const metaParts = [];
  if (largest > 0 && budgetMB > 0) {
    metaParts.push(`largest ${formatSize(largest)} / budget ${formatSize(budgetMB)}`);
  }
  if (cachedChecked.length > 0) metaParts.push(`${cachedChecked.length} cached`);
  if (dlMB > 0) metaParts.push(`~${formatSize(dlMB)} to download`);
  meta.textContent = metaParts.join(' · ');
}

// ──────────────── progress table ────────────────

function ensureProgressTable() {
  const wrap = $('run-progress-wrapper');
  if (!wrap) return null;
  // Reveal the progress card + its header — they are hidden by default on
  // mount so the user doesn't see an empty "Progress" scaffold, but we must
  // un-hide them as soon as the first row (download or run) appears.
  const card = wrap.closest('.table-card');
  if (card) card.hidden = false;
  const header = card?.previousElementSibling;
  if (header?.classList?.contains('section-header')) header.hidden = false;
  // Layout key — 'study' means pp/tg are split into d=0 and d=N columns,
  // 'plain' means a single column each. If the existing table doesn't
  // match the current state, drop it: state.results + the run loop are the
  // source of truth, the progress table is just a visual scaffold.
  const wantedLayout = state.studyMode ? 'study' : 'plain';
  let table = wrap.querySelector('table');
  if (table && table.dataset.layout !== wantedLayout) {
    table.remove();
    table = null;
  }
  if (!table) {
    table = document.createElement('table');
    table.className = 'results-table run-progress-table';
    table.dataset.layout = wantedLayout;
    const dN = state.nDepth || 0;
    const ppHead = state.studyMode
      ? `<th class="num" title="Prompt processing throughput at empty cache (avg \u00b1 stddev t/s)">pp tok/s @ d0</th>
         <th class="num" title="Prompt processing throughput at depth ${dN} (avg \u00b1 stddev t/s)">pp tok/s @ d${dN}</th>`
      : `<th class="num" title="Prompt processing throughput (avg \u00b1 stddev t/s)">pp tok/s</th>`;
    const tgHead = state.studyMode
      ? `<th class="num" title="Text generation throughput at empty cache (avg \u00b1 stddev t/s)">tg tok/s @ d0</th>
         <th class="num" title="Text generation throughput at depth ${dN} (avg \u00b1 stddev t/s)">tg tok/s @ d${dN}</th>`
      : `<th class="num" title="Text generation throughput (avg \u00b1 stddev t/s)">tg tok/s</th>`;
    table.innerHTML = `
      <thead>
        <tr>
          <th>Model</th>
          <th>Variant</th>
          <th>Status</th>
          ${ppHead}
          ${tgHead}
          <th class="num">Wall s</th>
          <th>Error</th>
        </tr>
      </thead>
      <tbody></tbody>
    `;
    wrap.appendChild(table);
  }
  return table;
}

function progressRowFor(v) {
  const key = cacheKey(v);
  const table = ensureProgressTable();
  const tbody = table.querySelector('tbody');
  let tr = tbody.querySelector(`tr[data-key="${cssEscape(key)}"]`);
  if (!tr) {
    tr = document.createElement('tr');
    tr.dataset.key = key;
    tr.className = 'run-row-queued';
    // pp/tg cells gain a depth-suffixed class in study mode so
    // fillFromRecord can route each record to its own column. Plain mode
    // still uses a single .prefill-dn / .decode-dn cell — pre-study (or
    // single-pass) records all go there regardless of nDepth.
    const ppCells = state.studyMode
      ? '<td class="num prefill prefill-d0">—</td><td class="num prefill prefill-dn">—</td>'
      : '<td class="num prefill prefill-dn">—</td>';
    const tgCells = state.studyMode
      ? '<td class="num decode decode-d0">—</td><td class="num decode decode-dn">—</td>'
      : '<td class="num decode decode-dn">—</td>';
    tr.innerHTML = `
      <td>${escapeText(v.modelName)}</td>
      <td>${escapeText(v.quant)}</td>
      <td class="status">queued</td>
      ${ppCells}
      ${tgCells}
      <td class="num wall">—</td>
      <td class="err"></td>
    `;
    tbody.appendChild(tr);
  }
  let tickInterval = null;
  const stopTicker = () => {
    if (tickInterval !== null) { clearInterval(tickInterval); tickInterval = null; }
  };
  return {
    // sinceMs: optional epoch ms. When set, the cell ticks once a second so
    // long-running phases (CPU pp512 warmup, big-model rep calls) show
    // wall-clock progress instead of looking hung. Cleared on next setStatus.
    setStatus(status, msg, sinceMs) {
      stopTicker();
      tr.className = `run-row-${rowClassFor(status)}`;
      const cell = tr.querySelector('.status');
      const render = () => {
        const base = msg ? `${status} — ${msg}` : status;
        cell.textContent = sinceMs
          ? `${base} (${Math.floor((Date.now() - sinceMs) / 1000)}s)`
          : base;
      };
      render();
      if (sinceMs) tickInterval = setInterval(render, 1000);
    },
    setProgress(fraction, downloaded, total) {
      stopTicker();
      const pct = (fraction * 100).toFixed(1);
      const detail = total > 0
        ? `${pct}% (${formatSize(downloaded / (1024 * 1024))} / ${formatSize(total / (1024 * 1024))})`
        : '';
      tr.querySelector('.status').textContent = detail ? `downloading ${detail}` : 'downloading';
    },
    fillFromRecord(record) {
      stopTicker();
      tr.className = `run-row-${record.status === 'done' ? 'ok' : 'error'}`;
      tr.querySelector('.status').textContent = record.status;
      // Format llama-bench style: "avg \u00b1 stddev" with the test name as
      // the cell tooltip so users see the exact pp/tg N that was measured.
      const tests = record.metrics?.tests || [];
      const pp = tests.find(t => t.name?.startsWith('pp'));
      const tg = tests.find(t => t.name?.startsWith('tg'));
      const fmt = (t) => t ? `${t.avg_ts.toFixed(2)} \u00b1 ${t.stddev_ts.toFixed(2)}` : '\u2014';
      // In study mode pick d=0 vs d=N based on the record's nDepth so the
      // first pass doesn't get clobbered by the second. Plain mode only
      // ever has the .prefill-dn / .decode-dn cells.
      const isD0 = state.studyMode && (record.nDepth ?? 0) === 0;
      const ppSel = isD0 ? '.prefill-d0' : '.prefill-dn';
      const tgSel = isD0 ? '.decode-d0' : '.decode-dn';
      const ppCell = tr.querySelector(ppSel);
      const tgCell = tr.querySelector(tgSel);
      if (ppCell) {
        ppCell.textContent = fmt(pp);
        if (pp) ppCell.title = pp.name;
      }
      if (tgCell) {
        tgCell.textContent = fmt(tg);
        if (tg) tgCell.title = tg.name;
      }
      // Wall cell accumulates across depth passes in study mode so the
      // user sees total time per variant. Plain mode is a single-shot
      // assignment as before.
      const wallSec = record.wallTimeMs ? record.wallTimeMs / 1000 : 0;
      const wallEl = tr.querySelector('.wall');
      if (state.studyMode) {
        const prev = parseFloat(wallEl.dataset.totalSec || '0') || 0;
        const total = prev + wallSec;
        wallEl.dataset.totalSec = String(total);
        wallEl.textContent = total > 0 ? total.toFixed(1) : '\u2014';
      } else {
        wallEl.textContent = wallSec > 0 ? wallSec.toFixed(1) : '\u2014';
      }
      tr.querySelector('.err').textContent = record.error || '';
    },
  };
}

function rowClassFor(status) {
  if (status === 'done' || status === 'ok' || status === 'cached') return 'ok';
  if (status === 'error') return 'error';
  if (status === 'queued' || !status) return 'queued';
  return 'running';
}

// ──────────────── logging ────────────────

function logLine(msg) {
  const pre = $('log-output');
  if (!pre) return;
  const line = `[${new Date().toISOString().slice(11, 23)}] ${msg}\n`;
  pre.textContent += line;
  pre.scrollTop = pre.scrollHeight;
}

// ──────────────── machine / browser info ────────────────

function browserInfo() {
  const ua = navigator.userAgent;
  if (/Firefox\/(\d+)/.test(ua)) return `firefox-${RegExp.$1}`;
  if (/Edg\/(\d+)/.test(ua)) return `edge-${RegExp.$1}`;
  if (/Chrome\/(\d+)/.test(ua)) return `chromium-${RegExp.$1}`;
  if (/Version\/(\d+).*Safari/.test(ua)) return `webkit-${RegExp.$1}`;
  return 'browser-unknown';
}

function slugify(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'unknown';
}

// ──────────────── user-reported submission fields ────────────────

// Best-effort default for the four user-reported inputs, derived from the
// auto-detected device + browser data. The user is expected to edit these
// before running — defaults exist only so the form isn't empty on first
// visit. Returns { machineName, gpuName, browser, os }.
function autoDetectedUserReported() {
  const d = state.device || {};
  const gpu = d.gpu || {};
  const gpuStr = [gpu.vendor, gpu.architecture, gpu.device, gpu.description]
    .filter(Boolean).join(' ').trim();
  const memGB = state.budget?.memGB;
  const browser = formatBrowser(d);
  const os = formatPlatform(d);
  // machineName default: "<gpu> · <memGB> GB" if both known, else either,
  // else the OS string. The user is encouraged to replace with a friendly
  // label like "MacBook Pro M3 16GB".
  let machineName = '';
  if (gpuStr && memGB) machineName = `${gpuStr} · ${memGB} GB`;
  else if (gpuStr) machineName = gpuStr;
  else if (memGB) machineName = `${memGB} GB device`;
  else machineName = os;
  return { machineName, gpuName: gpuStr, browser, os };
}

function readUserReportedFromInputs() {
  return {
    machineName: ($('ur-machine-name')?.value ?? '').trim(),
    gpuName:     ($('ur-gpu-name')?.value     ?? '').trim(),
    browser:     ($('ur-browser')?.value      ?? '').trim(),
    os:          ($('ur-os')?.value           ?? '').trim(),
  };
}

function refreshUserReportedValidation() {
  const hint = $('ur-hint');
  const missing = USER_REPORTED_REQUIRED.filter(k => !state.userReported[k]);
  for (const k of USER_REPORTED_REQUIRED) {
    const id = { machineName: 'ur-machine-name', browser: 'ur-browser', os: 'ur-os' }[k];
    const el = $(id);
    if (el) el.classList.toggle('is-missing', !state.userReported[k]);
  }
  if (hint) {
    if (missing.length === 0) {
      hint.textContent = 'Looks good — these labels will be attached to every result you submit.';
      hint.classList.remove('is-warn');
    } else {
      hint.textContent = `Required: ${missing.join(', ')}. We'll still let you run, but submissions need these filled in.`;
      hint.classList.add('is-warn');
    }
  }
}

function wireUserReported() {
  // Pre-fill: stored values win, fall back to auto-detected defaults so
  // first-time users see something rather than an empty form.
  const stored = loadUserReported();
  const auto = autoDetectedUserReported();
  state.userReported = {
    machineName: stored?.machineName?.trim() || auto.machineName,
    gpuName:     stored?.gpuName?.trim()     || auto.gpuName,
    browser:     stored?.browser?.trim()     || auto.browser,
    os:          stored?.os?.trim()          || auto.os,
  };
  for (const [id, key] of [
    ['ur-machine-name', 'machineName'],
    ['ur-gpu-name',     'gpuName'],
    ['ur-browser',      'browser'],
    ['ur-os',           'os'],
  ]) {
    const el = $(id);
    if (!el) continue;
    el.value = state.userReported[key] || '';
    el.addEventListener('input', () => {
      state.userReported = readUserReportedFromInputs();
      saveUserReported();
      refreshUserReportedValidation();
    });
  }
  // Persist whatever the auto-detect filled in so the user doesn't lose
  // it on reload before they touch anything.
  saveUserReported();
  refreshUserReportedValidation();
}

async function machineInfo() {
  const ua = navigator.userAgent;
  const platform = /Mac/.test(ua) ? 'darwin'
    : /Win/.test(ua) ? 'win32'
    : /Linux/.test(ua) ? 'linux'
    : /iPhone|iPad|iOS/.test(ua) ? 'ios'
    : /Android/.test(ua) ? 'android'
    : 'unknown';
  let arch = 'unknown';
  let platformVersion = '';
  try {
    const uad = navigator.userAgentData;
    if (uad?.getHighEntropyValues) {
      const hev = await uad.getHighEntropyValues(['architecture', 'platformVersion']);
      arch = hev.architecture || arch;
      platformVersion = hev.platformVersion || '';
    }
  } catch { /* non-UA-Data browsers */ }
  if (arch === 'unknown') {
    arch = /arm/i.test(ua) ? 'arm64'
      : /x86_64|Win64|x64/i.test(ua) ? 'x64'
      : 'unknown';
  }
  const gpu = state.device?.gpu;
  const gpuStr = gpu
    ? [gpu.vendor, gpu.architecture, gpu.device, gpu.description].filter(Boolean).join(' ').trim()
    : '';
  const cpus = gpuStr || 'browser';
  const totalMemoryGB = navigator.deviceMemory || 0;
  return {
    slug: slugify(`${cpus}-${totalMemoryGB}gb-${platform}`),
    platform,
    platformVersion,
    arch,
    cpus,
    totalMemoryGB,
    userAgent: ua,
  };
}

// ──────────────── Download ────────────────

async function onDownloadClick() {
  const variants = getCheckedVariants();
  if (variants.length === 0) return;
  state.running = true;
  state.aborted = false;
  updateButtons();

  for (const v of variants) {
    if (state.aborted) break;
    const row = progressRowFor(v);
    row.setStatus('downloading', '');
    const ac = new AbortController();
    const unregister = registerAbort(() => ac.abort());
    try {
      const { size } = await state.source.opfsHandleForModel(
        v.repo, v.filename,
        (fr, downloaded, total) => row.setProgress(fr, downloaded, total),
        ac.signal,
      );
      if (!ac.signal.aborted) {
        state.cacheStatus[cacheKey(v)] = { cachedBytes: size };
        refreshCacheBadge(v);
        row.setStatus('cached', formatSize(size / (1024 * 1024)));
      } else {
        row.setStatus('aborted', '');
      }
    } catch (err) {
      if (ac.signal.aborted) { row.setStatus('aborted', ''); }
      else { row.setStatus('error', err.message); logLine(`Download failed: ${v.filename}: ${err.message}`); }
    } finally {
      unregister();
    }
  }

  // Refresh cache inventory to reconcile any partial downloads.
  state.cacheStatus = await loadCacheStatus();
  document.querySelectorAll('.run-variant-row').forEach(row => {
    const v = state.variants.find(x => cacheKey(x) === row.dataset.key);
    if (v) refreshCacheBadge(v);
  });

  state.running = false;
  updateButtons();
}

// ──────────────── Run ────────────────

// Curated leaderboard study: focus model at several quants for a quant
// sweep, plus every other model at the standard quant as a single
// representative point. Selection rule lives in models.json
// (`studySelection`) so the CLI's --study flag and this button stay in
// sync. Variants that don't fit the device's memory budget are dropped
// silently — same rule the "All fit" button enforces.
function isStudyVariant(v) {
  if (!v) return false;
  const sel = state.models?.studySelection;
  if (!sel) return false;
  if ((sel.extras || []).some(e => e.model === v.modelName && e.quant === v.quant)) return true;
  if (v.modelName === sel.focusModel) return (sel.focusQuants || []).includes(v.quant);
  return v.quant === sel.standardQuant;
}

async function onRunStudyClick() {
  if (state.running) return;

  // Apply the study selection — same DOM/state plumbing as wireBatchSelect.
  document.querySelectorAll('.run-variant-select').forEach(cb => {
    const v = state.variants.find(x => cacheKey(x) === cb.dataset.key);
    cb.checked = !!v && isStudyVariant(v) && variantFitsDevice(v);
  });
  document.querySelectorAll('.run-family').forEach(el => {
    if (el.dataset.family) updateFamilySelectAllState(el.dataset.family);
  });
  updateButtons();

  const checked = getCheckedVariants();
  if (checked.length === 0) {
    logLine('Run study: no variants matched (none of the study quants fit this device).');
    return;
  }
  logLine(`Run study: selected ${checked.length} variants — starting run.`);
  // studyMode flips on the depth-pairing branch in runVariantWithIterations
  // so each variant produces both d=0 and d=N_DEPTH records (matches the
  // CLI runner's --study behavior).
  await onRunClick({ studyMode: true });
}

async function onRunClick({ studyMode = false } = {}) {
  // Run accepts any checked variant — uncached ones download just-in-time.
  const variants = getCheckedVariants();
  if (variants.length === 0) return;

  state.running = true;
  state.aborted = false;
  state.results = [];
  state.sessionDownloads = new Set();
  // Drive progress-table layout: study mode splits pp/tg into d=0 / d=N
  // columns so both depth passes' numbers stay visible.
  state.studyMode = !!studyMode;
  updateButtons();

  if (isMobileDevice()) {
    logLine(
      'Mobile device — sequential downloads (no parallel prefetch), ' +
      'forced eviction after each variant, ' +
      `${(MOBILE_YIELD_BETWEEN_RUNS_MS / 1000).toFixed(1)} s cooldown between runs ` +
      'so iOS can release WebGPU buffers before the next load.',
    );
    if (state.budget?.source) {
      logLine(`GPU budget: ${state.budget.source}`);
    }
  }

  const machine = await machineInfo();
  const browser = browserInfo();
  // Mobile forces eviction regardless of the checkbox: keeping multiple
  // ~700 MB GGUFs in OPFS while the GPU process retains buffers from the
  // just-finished run is the fastest path to a Jetsam tab kill on iOS.
  const evictAfter = isMobileDevice() || !!$('evict-after-run')?.checked;

  // One-ahead prefetch: while variant i runs, we may have variant i+1
  // downloading. Only one prefetch in flight at a time.
  // On mobile, the overlap is a measurement hazard — concurrent download
  // contends with inference for SoC power, memory bandwidth, and OPFS
  // write queues. Skip the prefetch entirely; runBenchmarkInWorker's
  // opfsHandleForModel does the download inline (with the same progress
  // events the prefetch row would have shown).
  const skipPrefetch = isMobileDevice();
  const prefetchFor = async (v) => {
    if (!v || isCached(v)) return;
    if (skipPrefetch) return;
    const row = progressRowFor(v);
    row.setStatus('prefetching', '');
    const ac = new AbortController();
    const unregister = registerAbort(() => ac.abort());
    try {
      const { size } = await state.source.opfsHandleForModel(
        v.repo, v.filename,
        (fr, downloaded, total) => row.setProgress(fr, downloaded, total),
        ac.signal,
      );
      state.cacheStatus[cacheKey(v)] = { cachedBytes: size };
      state.sessionDownloads.add(cacheKey(v));
      refreshCacheBadge(v);
      row.setStatus('cached', formatSize(size / (1024 * 1024)));
    } catch (err) {
      if (ac.signal.aborted) {
        row.setStatus('aborted', '');
        return;
      }
      row.setStatus('error', `prefetch: ${err.message}`);
      logLine(`Prefetch failed: ${v.filename}: ${err.message}`);
    } finally {
      unregister();
    }
  };

  // Seed the first prefetch before the loop so variant 0 starts downloading
  // while we set up. The loop awaits each prefetch completion before running.
  let prefetchPromise = prefetchFor(variants[0]);

  for (let i = 0; i < variants.length; i++) {
    if (state.aborted) break;
    const v = variants[i];
    const row = progressRowFor(v);

    // Wait for variant i to be cached (either via prefetch or pre-existing).
    await prefetchPromise;
    if (state.aborted) break;
    // When skipPrefetch is on (mobile), variants arrive uncached and
    // runBenchmarkInWorker → opfsHandleForModel handles the inline
    // download. Skip the cache-check error path in that case.
    if (!skipPrefetch && !isCached(v)) {
      row.setStatus('error', 'not cached after prefetch');
      prefetchPromise = prefetchFor(variants[i + 1]);
      continue;
    }

    // Kick off prefetch of i+1 in parallel with the run of i.
    prefetchPromise = prefetchFor(variants[i + 1]);

    // Persist run intent so a tab crash leaves a breadcrumb.
    writeRunIntent(v);

    row.setStatus('running', '');

    // Depth schedule for this variant. Study mode pairs d=0 with the
    // configured d=N so the dashboard can compare cold-cache against
    // depth-loaded numbers; non-study runs do a single pass at the user's
    // configured depth (default 2048). Mirrors the runner.js depth loop.
    const baseDepth = Math.max(0, state.nDepth ?? DEFAULT_N_DEPTH);
    const depthsToRun = (studyMode && baseDepth > 0) ? [0, baseDepth] : [baseDepth];

    let sharedCpu = null;
    for (const nDepth of depthsToRun) {
      if (state.aborted) break;
      const start = performance.now();
      const variantResult = await runVariantWithIterations(v, row, {
        nDepth,
        cpuResult: sharedCpu,
      });
      const wallTimeMs = performance.now() - start;

      const record = makeRecord(v, variantResult, machine, browser, wallTimeMs);
      state.results.push(record);
      row.fillFromRecord(record);

      // Cache the CPU pass from the first depth so subsequent depth runs
      // skip it (CPU baseline is depth-independent).
      if (!sharedCpu && variantResult.cpu?.status === 'done') {
        sharedCpu = variantResult.cpu;
      }

      try {
        // sessionStorage so results survive in-tab navigations (the OAuth
        // sign-in redirect in particular) but reset when the user actually
        // closes the tab — they don't want stale results on a fresh visit.
        sessionStorage.setItem(RESULTS_STORAGE_KEY, JSON.stringify(state.results));
      } catch { /* quota */ }

      // Mobile: drop per-rep raw arrays from the in-memory record after
      // sessionStorage has the full copy. The dashboard only reads the
      // aggregates (avg_ts, stddev_ts) and on iOS Safari every byte that
      // isn't reclaimed between variants edges the tab toward Jetsam.
      // Trade-off: an HF submission in the same session loses per-rep
      // samples; a fresh page-load rehydrates from sessionStorage and
      // recovers them.
      if (isMobileDevice()) {
        if (record.metrics) {
          delete record.metrics.prefill_samples;
          delete record.metrics.decode_samples;
          for (const t of record.metrics.tests || []) {
            delete t.samples_ts;
            delete t.samples_ns;
          }
        }
        if (record.consistency) delete record.consistency.token_ids;
        record.output = '';
      }

      if (state.surface === 'localhost' && $('save-local')?.checked) {
        fetch('/api/results', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(record),
        }).catch(err => logLine(`POST /api/results failed: ${err.message}`));
      }
    }

    clearRunIntent();

    // Evict if enabled and this variant was downloaded this session. Files
    // the user had cached before the run are always preserved.
    if (evictAfter && state.sessionDownloads.has(cacheKey(v))) {
      try {
        const res = await state.source.evictModel(v.repo, v.filename);
        if (res.ok) {
          logLine(`Evicted ${v.filename} (${formatSize(res.bytesFreed / (1024 * 1024))})`);
          delete state.cacheStatus[cacheKey(v)];
          state.sessionDownloads.delete(cacheKey(v));
          refreshCacheBadge(v);
        } else {
          logLine(`Eviction skipped (${v.filename}): ${res.reason}`);
        }
      } catch (err) {
        logLine(`Eviction error (${v.filename}): ${err.message}`);
      }
    }

    await sleep(isMobileDevice() ? MOBILE_YIELD_BETWEEN_RUNS_MS : YIELD_BETWEEN_RUNS_MS);
  }

  // Queue ended or aborted: make sure we don't leave a prefetch running.
  try { await prefetchPromise; } catch { /* already logged */ }

  renderOutput();
  state.running = false;
  updateButtons();
  renderHfSection();
}

// Spawn a dedicated worker, transfer the stream + params, relay events back
// into the provided callbacks, resolve with the worker's final record.
// The worker is terminated (and state.currentWorker cleared) when done.
function runInWorker({
  params,
  opfsPath,
  onStatus,
  onProgress,
  onLog,
}) {
  return new Promise((resolve) => {
    let worker;
    try {
      worker = new Worker(new URL('./bench-worker.js', import.meta.url));
    } catch (err) {
      resolve({ status: 'error', error: `worker construct failed: ${err.message}` });
      return;
    }

    state.currentWorker = worker;
    let settled = false;
    let unregister = () => {};
    const finish = (record) => {
      if (settled) return;
      settled = true;
      try { worker.terminate(); } catch { /* noop */ }
      if (state.currentWorker === worker) state.currentWorker = null;
      unregister();
      resolve(record);
    };
    unregister = registerAbort(() => finish({ status: 'aborted', error: 'aborted by user' }));

    worker.onmessage = (e) => {
      const msg = e.data || {};
      if (msg.type === 'status') onStatus?.(msg.status, msg.msg, msg.sinceMs);
      else if (msg.type === 'progress') onProgress?.(msg.fraction, msg.downloaded, msg.total);
      else if (msg.type === 'log') onLog?.(msg.line);
      else if (msg.type === 'result') finish(msg.record);
    };
    worker.onerror = (err) => {
      finish({
        status: 'error',
        error: err?.message || 'worker error (tab likely out of memory)',
      });
    };
    worker.onmessageerror = () => {
      finish({ status: 'error', error: 'worker message deserialization failed' });
    };

    // OPFS path is the only transport. We send the layout key only
    // (rootDir + repo + filename); the worker re-resolves to a
    // FileSystemFileHandle via navigator.storage.getDirectory() itself,
    // since FileSystemFileHandle structured-clone is missing on iOS Safari.
    try {
      worker.postMessage({ type: 'run', params, opfsPath });
    } catch (err) {
      finish({ status: 'error', error: `postMessage(opfsPath) failed: ${err.message}` });
    }
  });
}

// Download to OPFS on the main thread, then hand the OPFS layout key to a
// freshly-spawned worker. The worker opens a FileSystemSyncAccessHandle
// and routes MEMFS reads through it (use_mmap=0), never copying the model
// into the WASM heap. Supports models larger than the WASM heap budget.
async function runBenchmarkInWorker(v, params, callbacks) {
  const baseParams = {
    buildType: 'Suspending' in WebAssembly ? 'jspi' : 'asyncify',
    // Model load
    nCtx: params.nCtx,
    nGpuLayers: params.nGpuLayers,
    // Consistency phase — empty consistencyPrompt skips it
    consistencyPrompt: params.consistencyPrompt || '',
    consistencyNPredict: params.consistencyNPredict || DEFAULT_N_PREDICT,
    refTokenIds: params.refTokenIds || null,
    // Perf phase — set both to 0 to skip
    nPrompt: params.nPrompt ?? 0,
    nGen:    params.nGen    ?? 0,
    nReps:   params.nReps   ?? DEFAULT_ITERATIONS,
    nDepth:  params.nDepth  ?? 0,
    noWarmup: !!params.noWarmup,
  };

  const ac = new AbortController();
  const unregister = registerAbort(() => ac.abort());
  try {
    callbacks.onStatus?.('downloading', 'Downloading model to OPFS...');
    const r = await state.source.opfsHandleForModel(
      v.repo, v.filename,
      callbacks.onProgress,
      ac.signal,
    );
    // When the prefetch is skipped (mobile path), the inline download
    // above is the variant's first arrival in OPFS. Mark it as
    // session-downloaded so the post-run eviction logic frees it before
    // the next variant starts — keeping disk usage flat.
    if (r.wasDownloaded) {
      state.sessionDownloads.add(cacheKey(v));
      state.cacheStatus[cacheKey(v)] = { cachedBytes: r.size };
      refreshCacheBadge(v);
    }
  } catch (err) {
    if (ac.signal.aborted) {
      return { status: 'aborted', error: 'aborted by user' };
    }
    return { status: 'error', error: `opfsHandleForModel failed: ${err.message}` };
  } finally {
    unregister();
  }
  if (state.aborted) {
    return { status: 'aborted', error: 'aborted by user' };
  }
  // Pass the OPFS layout key (rootDir + repo + filename), not a
  // FileSystemFileHandle. iOS Safari can't structured-clone FileHandles,
  // so the worker re-resolves it locally via navigator.storage.getDirectory().
  return runInWorker({
    params: baseParams,
    opfsPath: { rootDir: OPFS_ROOT_NAME, repo: v.repo, filename: v.filename },
    onStatus: callbacks.onStatus,
    onProgress: callbacks.onProgress,
    onLog: callbacks.onLog,
  });
}

// Runs one variant: CPU consistency baseline (one model load, generates
// reference token IDs via bench_run), then GPU pass (one model load that
// does both consistency forced-decoding and the llama-bench-style perf
// sweep — pp + tg with warmup + nReps timed reps each).
// Returns an aggregate that makeRecord consumes.
//
// `opts.nDepth` overrides state.nDepth so the caller can sweep multiple
// depths per variant (study mode pairs d=0 with d=N).
// `opts.cpuResult` when provided short-circuits the CPU baseline phase —
// study mode runs CPU once on the d=0 pass and reuses it for d=N, since
// reference tokens and the 1-rep CPU comparator are depth-independent.
async function runVariantWithIterations(v, row, opts = {}) {
  const nReps = Math.max(1, state.iterations || DEFAULT_ITERATIONS);
  const nPrompt = Math.max(0, state.nPrompt ?? DEFAULT_N_PROMPT);
  const nGen = Math.max(0, state.nGen ?? DEFAULT_N_GEN);
  const nDepth = Math.max(0, opts.nDepth ?? state.nDepth ?? DEFAULT_N_DEPTH);
  const reuseCpu = opts.cpuResult || null;
  // Per-test n_ctx mirrors llama-bench (line 1211 of
  // tools/llama-bench/llama-bench.cpp): sized to fit prompt+gen+depth so a
  // raised depth doesn't silently overflow the cache.
  const nCtxFor = (depth) => Math.max(DEFAULT_N_CTX, nPrompt + nGen + depth);
  // Phase toggles from the run page. Both default OFF; combined effect:
  //   neither (default)  → only GPU perf, no CPU pass at all
  //   run CPU perf       → CPU perf baseline + GPU perf, no token-id check
  //   run consistency    → CPU consistency tokens + GPU consistency + GPU perf
  //   both               → full CPU baseline (consistency + 1-rep perf) +
  //                        GPU consistency + GPU perf
  const runConsistency = !!state.runConsistency;
  const runCpuPerf = !!state.runCpuPerf;
  const needCpuPass = runConsistency || runCpuPerf;

  // ─── CPU baseline ───
  // Skipped entirely if both toggles disable it OR caller provided a cached
  // result from an earlier depth pass. Otherwise the pass mixes and matches:
  // consistency_run captures token_ids; perf phase runs at nReps=1 (single
  // warmup+timed rep — enough to populate the dashboard's CPU/GPU comparison
  // without doubling CPU runtime).
  let cpuResult;
  if (reuseCpu) {
    cpuResult = reuseCpu;
  } else if (needCpuPass) {
    const phaseLabel = runConsistency && runCpuPerf ? 'reference tokens + 1-rep perf'
      : runConsistency ? 'reference tokens'
      : '1-rep perf';
    row.setStatus('cpu-baseline', phaseLabel);
    try {
      cpuResult = await runBenchmarkInWorker(v, {
        consistencyPrompt: runConsistency ? CONSISTENCY_PROMPT : '',
        consistencyNPredict: DEFAULT_N_PREDICT,
        refTokenIds: null,
        nPrompt: runCpuPerf ? nPrompt : 0,
        nGen:    runCpuPerf ? nGen    : 0,
        // CPU baseline keeps depth=0 — its job is reference-token capture
        // and a single-rep perf comparator, not depth-loaded sweeping.
        nDepth: 0,
        nReps: 1,
        nCtx: nCtxFor(0),
        nGpuLayers: 0,
      }, {
        onStatus: (status, msg, sinceMs) => row.setStatus(`cpu/${status}`, msg, sinceMs),
        onProgress: (fr, downloaded, total) => row.setProgress(fr, downloaded, total),
        onLog: logLine,
      });
    } catch (err) {
      cpuResult = { status: 'error', error: err.message || String(err) };
    }
  } else {
    cpuResult = { status: 'skipped' };
  }

  // CPU pass is best-effort. Failures (OOM, slow device, missing op) don't
  // block the GPU run — the user opted into resilience implicitly by the
  // phase being best-effort, and explicitly via the skip checkboxes.
  const cpuOk = cpuResult.status === 'done';
  if (cpuResult.status === 'error') {
    logLine(`CPU baseline failed (${cpuResult.error || 'unknown'}) — proceeding with GPU run.`);
    row.setStatus('cpu-skipped', 'continuing with GPU only');
  }

  // refTokenIds is the GPU pass's input for forced-decode consistency. Only
  // pass when we actually have tokens (consistency was requested AND CPU
  // produced tokens).
  const refTokenIds = (cpuOk && runConsistency && cpuResult.consistency?.token_ids?.length)
    ? cpuResult.consistency.token_ids.join(',')
    : '';

  if (state.aborted) {
    return { status: 'error', error: 'aborted', cpu: cpuResult, gpu: null };
  }

  // ─── GPU pass: consistency (when not skipped) + perf in one model load ───
  row.setStatus('gpu-run', 'loading model');
  let gpuResult;
  try {
    gpuResult = await runBenchmarkInWorker(v, {
      consistencyPrompt: runConsistency ? CONSISTENCY_PROMPT : '',
      consistencyNPredict: DEFAULT_N_PREDICT,
      refTokenIds: refTokenIds || null,
      nPrompt,
      nGen,
      nDepth,
      nReps,
      nCtx: nCtxFor(nDepth),
      nGpuLayers: DEFAULT_N_GPU_LAYERS,
    }, {
      onStatus: (s, m, sinceMs) => row.setStatus(`gpu/${s}`, m, sinceMs),
      onProgress: (fr, d, t) => row.setProgress(fr, d, t),
      onLog: logLine,
    });
  } catch (err) {
    gpuResult = { status: 'error', error: err.message || String(err) };
  }

  return {
    status: gpuResult.status === 'done' ? 'done' : 'error',
    error: gpuResult.status === 'done' ? null : (gpuResult.error || 'GPU run failed'),
    cpu: cpuResult,
    gpu: gpuResult,
  };
}

function round2(n) { return Number.isFinite(n) ? parseFloat(n.toFixed(2)) : 0; }

// Pull pp/tg test results out of a metrics.tests array. Returns null if the
// requested test wasn't run (e.g. nPrompt=0 means no pp test).
function findTest(tests, prefix) {
  if (!Array.isArray(tests)) return null;
  return tests.find(t => typeof t.name === 'string' && t.name.startsWith(prefix)) || null;
}

function makeRecord(v, vr, machine, browser, wallTimeMs) {
  const gpu = vr.gpu;
  const tests = gpu?.metrics?.tests || null;
  const pp = findTest(tests, 'pp');
  const tg = findTest(tests, 'tg');

  // Llama-bench shape lives under metrics.tests; flat prefill_tok_s /
  // decode_tok_s are kept for backward compat with the existing dashboard
  // table cells until those are migrated to read from tests directly.
  const metrics = tests ? {
    tests,
    n_prompt: gpu.metrics.n_prompt,
    n_gen: gpu.metrics.n_gen,
    n_reps: gpu.metrics.n_reps,
    iterations: gpu.metrics.n_reps,
    prefill_tok_s: pp ? round2(pp.avg_ts) : 0,
    decode_tok_s:  tg ? round2(tg.avg_ts) : 0,
    prefill_tok_s_stdev: pp ? round2(pp.stddev_ts) : 0,
    decode_tok_s_stdev:  tg ? round2(tg.stddev_ts) : 0,
    prefill_samples: pp ? pp.samples_ts : [],
    decode_samples:  tg ? tg.samples_ts : [],
    n_p_eval: pp ? pp.n_prompt : 0,
    n_eval:   tg ? tg.n_gen    : 0,
    t_p_eval_ms: pp ? round2(pp.avg_ns / 1e6) : 0,
    t_eval_ms:   tg ? round2(tg.avg_ns / 1e6) : 0,
  } : null;

  // CPU baseline now runs a 1-rep perf sweep alongside the consistency
  // pass, so we have CPU-vs-GPU numbers to compare on the dashboard.
  // n=1 means no stddev, so the dashboard cell renders just the avg.
  const cpuTests = vr.cpu?.metrics?.tests;
  const cpuPp = cpuTests?.find(t => t.name?.startsWith('pp')) || null;
  const cpuTg = cpuTests?.find(t => t.name?.startsWith('tg')) || null;
  const cpuBaseline = vr.cpu?.status === 'done' ? {
    prefill_tok_s: cpuPp ? round2(cpuPp.avg_ts) : null,
    decode_tok_s:  cpuTg ? round2(cpuTg.avg_ts) : null,
  } : null;

  return {
    status: vr.status,
    error: vr.error || null,
    model: v.modelName,
    variant: v.quant,
    filename: v.filename,
    repo: v.repo,
    sizeMB: v.sizeMB,
    browser,
    nCtx: DEFAULT_N_CTX,
    nPredict: DEFAULT_N_PREDICT,
    nPrompt: gpu?.metrics?.n_prompt ?? 0,
    nGen: gpu?.metrics?.n_gen ?? 0,
    nDepth: gpu?.metrics?.n_depth ?? 0,
    nReps: gpu?.metrics?.n_reps ?? 0,
    nGpuLayers: DEFAULT_N_GPU_LAYERS,
    timestamp: new Date().toISOString(),
    wallTimeMs,
    webgpuAvailable: gpu?.webgpuAvailable ?? !!navigator.gpu,
    gpuAdapterInfo: gpu?.gpuAdapterInfo ?? null,
    buildType: gpu?.buildType ?? null,
    // llama.cpp version stamped from build-info.json. Lets us correlate
    // result drift with llama.cpp upgrades over time.
    llamaCppCommit: state.buildInfo?.llamaCppCommit ?? null,
    llamaCppDescribe: state.buildInfo?.llamaCppDescribe ?? null,
    dawnTag: state.buildInfo?.dawnTag ?? null,
    metrics,
    consistency: gpu?.consistency ?? null,
    cpu_baseline: cpuBaseline,
    output: gpu?.output || '',
    machine,
    // Memory snapshot llama.cpp captured immediately after bench_load —
    // model_size, state_size, and per-device {free,total} from every ggml
    // backend. Useful for spotting memory-pressured runs and for sanity-
    // checking GPU memory headroom across machines.
    memoryInfo: gpu?.memoryInfo ?? null,
    // User-typed labels that override (or supplement) the auto-detected
    // machine/browser fields. Auto-detection is unreliable across UA-string
    // anonymization, deviceMemory rounding, and missing WebGPU adapter info.
    userReported: { ...state.userReported },
    source: `webgpu-bench/site (${state.surface})`,
  };
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ──────────────── crash-recovery trail ────────────────
//
// Mobile tabs often get reaped mid-run without warning — WebKit reloads the
// page and the user sees a silent reset. We stamp localStorage before each
// variant; if a stamp is present on page load and we can't match it against
// a successful result in lastRun, we assume a crash and surface a banner.

function writeRunIntent(v) {
  try {
    localStorage.setItem(RUN_INTENT_STORAGE_KEY, JSON.stringify({
      model: v.modelName,
      quant: v.quant,
      filename: v.filename,
      sizeMB: v.sizeMB,
      when: Date.now(),
    }));
  } catch { /* quota / disabled */ }
}

function clearRunIntent() {
  try { localStorage.removeItem(RUN_INTENT_STORAGE_KEY); } catch {}
}

function maybeShowCrashBanner() {
  const banner = $('run-crash-banner');
  const text = $('run-crash-banner-text');
  const dismiss = $('run-crash-banner-dismiss');
  if (!banner || !text || !dismiss) return;

  let intent;
  try {
    const raw = localStorage.getItem(RUN_INTENT_STORAGE_KEY);
    if (!raw) return;
    intent = JSON.parse(raw);
  } catch {
    clearRunIntent();
    return;
  }
  if (!intent || typeof intent.when !== 'number') {
    clearRunIntent();
    return;
  }
  if (Date.now() - intent.when < CRASH_STALE_MS) {
    // Too fresh — another tab might still be running. Leave it alone.
    return;
  }

  // Intent survived the page reload and is stale: the run almost certainly
  // didn't finish cleanly (we clear the intent on success).
  const size = intent.sizeMB ? formatSize(intent.sizeMB) : 'unknown size';
  text.textContent =
    `A previous run on "${intent.model} ${intent.quant}" (${size}) did not complete — the tab was likely reaped by the OS (low memory). Try a smaller quant.`;
  banner.hidden = false;

  dismiss.addEventListener('click', () => {
    banner.hidden = true;
    clearRunIntent();
  }, { once: true });
}

// ──────────────── Output ────────────────

function renderOutput() {
  const ta = $('output-textarea');
  if (ta) ta.value = generateMarkdown(state.results);
  // Reflect emptiness: collapse the textarea, disable copy/download.
  const hasContent = !!ta?.value;
  const outputCard = document.querySelector('.run-output');
  if (outputCard) outputCard.classList.toggle('is-empty', !hasContent);
  const copyBtn = $('btn-copy');
  const dlJson = $('btn-download-json');
  if (copyBtn) copyBtn.disabled = !hasContent;
  if (dlJson) dlJson.disabled = !hasContent;
}

/* Hide the Progress scaffolding at mount so we don't show an empty
   placeholder. `ensureProgressTable` un-hides it the moment a download or
   run row appears. */
function hideProgressUntilFirstRow() {
  const wrap = $('run-progress-wrapper');
  if (!wrap) return;
  const card = wrap.closest('.table-card');
  if (card) card.hidden = true;
  const header = card?.previousElementSibling;
  if (header?.classList?.contains('section-header')) header.hidden = true;
}

function generateMarkdown(results) {
  if (results.length === 0) return '';
  const m = results[0].machine || {};
  const header = [
    `# WebGPU Benchmark Results`,
    ``,
    `- Machine: \`${m.cpus || 'unknown'}\` · ${m.totalMemoryGB || 0} GB · ${m.platform || 'unknown'} (${m.arch || '?'})`,
    `- Browser: \`${results[0].browser}\``,
    `- Build: \`${results[0].buildType || '?'}\``,
    `- WebGPU: ${results[0].webgpuAvailable ? 'yes' : 'no'}`,
    `- Timestamp: ${new Date().toISOString()}`,
    `- Variants run: ${results.length}`,
    '',
  ].join('\n');

  const passed = results.filter(r => r.status === 'done');
  const failed = results.filter(r => r.status !== 'done');

  let body = '';
  if (passed.length) {
    body += `## Passed (${passed.length})\n\n`;
    // llama-bench-style markdown: separate pp / tg columns with avg \u00b1 stddev.
    body += `| Model | Variant | Size | pp tok/s | tg tok/s | Wall s |\n`;
    body += `|---|---|---:|---:|---:|---:|\n`;
    const fmtTest = (tests, prefix) => {
      const t = tests?.find(x => x.name?.startsWith(prefix));
      return t ? `${t.avg_ts.toFixed(2)} \u00b1 ${t.stddev_ts.toFixed(2)} (${t.name})` : '\u2014';
    };
    for (const r of passed) {
      body += `| ${r.model} | ${r.variant} | ${formatSize(r.sizeMB)} | ${
        fmtTest(r.metrics?.tests, 'pp')} | ${fmtTest(r.metrics?.tests, 'tg')} | ${
        (r.wallTimeMs / 1000).toFixed(1)} |\n`;
    }
    body += `\n`;
  }
  if (failed.length) {
    body += `## Failed (${failed.length})\n\n`;
    for (const r of failed) {
      body += `- **${r.model}** ${r.variant}: \`${r.error || 'unknown error'}\`\n`;
    }
    body += `\n`;
  }

  const json = JSON.stringify(results, null, 2);
  body += `<details>\n<summary>Raw JSON (click to expand)</summary>\n\n\`\`\`json\n${json}\n\`\`\`\n</details>\n`;

  return header + body;
}

function wireOutputHandlers() {
  $('btn-copy')?.addEventListener('click', async () => {
    const text = $('output-textarea').value;
    try {
      await navigator.clipboard.writeText(text);
      flashButton($('btn-copy'), 'Copied!');
    } catch {
      $('output-textarea').select();
      try { document.execCommand('copy'); flashButton($('btn-copy'), 'Copied!'); } catch {}
    }
  });

  $('btn-download-json')?.addEventListener('click', () => {
    if (state.results.length === 0) return;
    const blob = new Blob([JSON.stringify(state.results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const stamp = new Date().toISOString().replace(/[:T.]/g, '-').slice(0, 19);
    a.download = `webgpu-bench-${stamp}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}

function flashButton(el, msg) {
  const original = el.textContent;
  el.textContent = msg;
  setTimeout(() => { el.textContent = original; }, 1200);
}

// ──────────────── Abort / Purge / Hub ────────────────

function wireAbortHandler() {
  $('btn-abort')?.addEventListener('click', () => {
    state.aborted = true;
    const ab = $('btn-abort');
    if (ab) ab.disabled = true;
    // Iterate every registered op (worker terminate, fetch AbortController):
    // worker.terminate() alone leaves the Promise pending forever, and
    // fetch without a signal can hang on slow connections. Each fn is
    // expected to also resolve / reject its own awaiting promise.
    const n = state.abortHandlers.size;
    for (const fn of state.abortHandlers) {
      try { fn(); } catch { /* keep iterating */ }
    }
    state.abortHandlers.clear();
    logLine(n > 0
      ? `Abort requested — cancelled ${n} in-flight op${n === 1 ? '' : 's'}.`
      : 'Abort requested — will stop between variants.');
  });
}

function wirePurgeHandler() {
  const btn = $('btn-purge');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    if (!confirm('Delete all cached GGUF files from OPFS? This frees browser storage but re-downloads will be needed.')) return;
    try {
      await purgeOpfs();
      state.cacheStatus = {};
      document.querySelectorAll('.run-variant-row').forEach(row => {
        const v = state.variants.find(x => cacheKey(x) === row.dataset.key);
        if (v) refreshCacheBadge(v);
      });
      updateButtons();
      logLine('OPFS cache purged.');
    } catch (err) {
      logLine(`Purge failed: ${err.message}`);
    }
  });
}

function wireHubHandlers() {
  const signinBtn = $('btn-signin');
  const submitBtn = $('btn-submit');
  if (signinBtn) {
    signinBtn.addEventListener('click', async () => {
      // Sign in / Sign out is disabled while a run is in flight; this guard
      // catches a stale-event-during-state-change race and keeps results safe.
      if (state.running) return;
      try {
        if (state.hfSession) {
          signOutHF();
          state.hfSession = null;
          renderHfSection();
          return;
        }
        await beginHFSignIn();
        // beginHFSignIn redirects — unreachable after.
      } catch (err) {
        logLine(`Sign-in failed: ${err.message}`);
      }
    });
  }

  if (submitBtn) {
    submitBtn.addEventListener('click', async () => {
      if (!state.hfSession) return;
      const eligible = submittableResults();
      if (eligible.length === 0) return;
      // Required user-reported fields gate the submission so the leaderboard
      // doesn't accumulate anonymous rows. The Run buttons stay enabled
      // even when these are blank — we only block at submit time.
      const missing = USER_REPORTED_REQUIRED.filter(k => !state.userReported[k]);
      if (missing.length > 0) {
        const card = $('user-reported-card');
        if (card) { card.open = true; card.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
        refreshUserReportedValidation();
        logLine(`Submit blocked: fill in ${missing.join(', ')} in "Your machine".`);
        return;
      }
      submitBtn.disabled = true;
      const original = submitBtn.textContent;
      submitBtn.textContent = 'Submitting…';
      try {
        const first = eligible[0];
        const res = await submitResultsToDataset(eligible, {
          token: state.hfSession.accessToken,
          machineSlug: first.machine?.slug || 'unknown',
          browser: first.browser || 'unknown-browser',
          submittedBy: state.hfSession.userName ? {
            name: state.hfSession.userName,
            hubId: state.hfSession.hubId || null,
            avatarUrl: state.hfSession.avatarUrl || null,
          } : null,
        });
        const link = res.pullRequestUrl
          || `https://huggingface.co/datasets/${HF_DATASET_REPO}/discussions`;
        logLine(`Opened PR with ${eligible.length} variant(s): ${link}`);
        // Restore the real label before flashing so the post-flash revert
        // doesn't snap back to "Submitting…".
        submitBtn.textContent = original;
        flashButton(submitBtn, 'Submitted!');
      } catch (err) {
        logLine(`Submit failed: ${err.message}`);
        submitBtn.textContent = original;
      } finally {
        submitBtn.disabled = submittableResults().length === 0;
      }
    });
  }
}

function wireRunHandlers() {
  $('btn-download')?.addEventListener('click', onDownloadClick);
  $('btn-run')?.addEventListener('click', onRunClick);
  $('btn-run-study')?.addEventListener('click', onRunStudyClick);
}

// ──────────────── Public API ────────────────

export async function mountRunSection() {
  if (state.mounted) return;
  state.mounted = true;

  state.surface = await detectSurface();
  state.source = ggufSource();
  state.budget = await getDeviceBudgetMB();
  state.device = await describeDevice();
  // Don't block mount on the build-info fetch — it's non-critical and the
  // first record will pick it up on the next render once it resolves.
  loadBuildInfo().then(info => {
    state.buildInfo = info;
    renderHeader();
  }).catch(() => { /* keep buildInfo null */ });

  try {
    state.models = await loadModels();
  } catch (err) {
    const panel = $('run-models');
    if (panel) panel.innerHTML = `<div class="empty-state">Could not load models.json — ${escapeText(err.message)}</div>`;
    console.error(err);
    return;
  }

  state.cacheStatus = await loadCacheStatus();
  state.variants = flattenVariants(state.models);

  if (state.surface === 'space') {
    try { state.hfSession = await resumeHFSession(); } catch { /* ignore */ }
  }

  // Evict-after-run default depends on surface: hosted OPFS quota is tight
  // and worth clawing back between runs; localhost's cache/models/ is
  // commonly shared with CLI workflows, so leaving it populated is helpful.
  const evictCheckbox = $('evict-after-run');
  if (evictCheckbox) {
    evictCheckbox.checked = state.surface === 'space';
  }

  renderHeader();
  renderModels();
  wireSelectionHandlers();
  wireFilters();
  wireFamilySearch();
  wireBatchSelect();
  wirePerfInputs();
  wireRunHandlers();
  wireAbortHandler();
  wirePurgeHandler();
  wireHubHandlers();
  wireOutputHandlers();
  wireUserReported();
  // Restore the last completed run from localStorage so it survives a page
  // reload — including the OAuth redirect taking the user to HF and back.
  // Must run before updateButtons/renderOutput/hideProgress so they pick up
  // the rehydrated state.results.
  restoreSavedResults();
  updateButtons();
  renderOutput();
  if (state.results.length === 0) hideProgressUntilFirstRow();
  maybeShowCrashBanner();
}

const RESULTS_STORAGE_KEY = 'webgpu-bench:lastRun';

function restoreSavedResults() {
  // Clean up the pre-migration localStorage entry — earlier builds wrote
  // results there, which made them persist across full tab closes. The
  // canonical location is now sessionStorage.
  try { localStorage.removeItem(RESULTS_STORAGE_KEY); } catch { /* noop */ }

  // Only restore when we just round-tripped through HF for sign-in
  // (beginHFSignIn() sets HF_OAUTH_PENDING_KEY immediately before the
  // redirect). A plain refresh has no such marker and should land on a
  // clean progress table — old runs sticking around was the bug.
  let oauthPending = false;
  try { oauthPending = !!sessionStorage.getItem(HF_OAUTH_PENDING_KEY); } catch { /* noop */ }
  if (!oauthPending) {
    try { sessionStorage.removeItem(RESULTS_STORAGE_KEY); } catch { /* noop */ }
    return;
  }
  // Consume the marker now so the next plain refresh doesn't restore again.
  try { sessionStorage.removeItem(HF_OAUTH_PENDING_KEY); } catch { /* noop */ }

  let saved;
  try {
    const raw = sessionStorage.getItem(RESULTS_STORAGE_KEY);
    if (!raw) return;
    saved = JSON.parse(raw);
  } catch { return; }
  if (!Array.isArray(saved) || saved.length === 0) return;

  state.results = saved;
  // Detect study mode from the saved records: if any (model, variant) cell
  // has both nDepth=0 and nDepth>0 entries, the OAuth-round-tripped run
  // was a Run Study and should restore into the depth-split layout.
  const depthsByCell = new Map();
  for (const r of saved) {
    const k = `${r.model}::${r.variant}`;
    if (!depthsByCell.has(k)) depthsByCell.set(k, new Set());
    depthsByCell.get(k).add(r.nDepth ?? 0);
  }
  state.studyMode = [...depthsByCell.values()].some(s => s.has(0) && [...s].some(d => d > 0));
  for (const record of saved) {
    const v = state.variants.find(x => x.repo === record.repo && x.filename === record.filename);
    if (!v) continue;
    progressRowFor(v).fillFromRecord(record);
  }
}

export function teardownRunSection() {
  // Placeholder — no explicit teardown today. Future: abort in-flight runs,
  // detach listeners. For now the Run tab just sits idle.
  state.aborted = true;
}
