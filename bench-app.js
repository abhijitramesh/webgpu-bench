// Interactive benchmark page controller.
// Mode detection (local vs hosted), model list rendering with device-fit
// check + filter chips, Download / Run / Abort orchestration, results output.
// Inference logic lives in bench-core.js — this file drives the UI and
// sequences one runBenchmarkCore() call per selected variant.

import { runBenchmarkCore } from './bench-core.js';
import { localSource, hostedSource, inventoryOpfs, purgeOpfs } from './bench-source.js';
import { getDeviceBudgetMB, variantFits, describeDevice } from './bench-device.js';
import {
  resumeHFSession, beginHFSignIn, signOutHF, submitResultsToDataset,
} from './bench-hub.js';
import { isHubConfigured, HF_DATASET_REPO } from './bench-config.js';

const OVERHEAD = 1.5;
const DEFAULT_PROMPT =
  'Explain quantum computing to a software engineer in four concise paragraphs. ' +
  'Cover superposition, entanglement, quantum gates, and one practical use case.';
const DEFAULT_N_PREDICT = 128;
const DEFAULT_N_CTX = 2048;
const DEFAULT_N_GPU_LAYERS = 999;
const YIELD_BETWEEN_RUNS_MS = 500;
const YIELD_BETWEEN_ITERATIONS_MS = 200;
const DEFAULT_ITERATIONS = 5;
const MIN_ITERATIONS_FOR_SUBMIT = 5;

const state = {
  mode: 'local',        // 'local' | 'hosted'
  source: null,         // localSource() | hostedSource()
  models: null,         // parsed models.json
  budget: null,         // { budgetMB, memGB, quotaMB, source }
  device: null,         // describeDevice() output
  cacheStatus: {},      // { 'repo/file': { cachedBytes } }
  variants: [],         // flat variant rows with metadata
  running: false,
  aborted: false,
  results: [],          // result records from the current session
  hfSession: null,      // { accessToken, expiresAt, userName } when signed in
  iterations: DEFAULT_ITERATIONS,
};

// ──────────────── mode / data loading ────────────────

async function detectMode() {
  const params = new URLSearchParams(location.search);
  if (params.get('mode') === 'hosted') return 'hosted';
  if (params.get('mode') === 'local') return 'local';
  try {
    const r = await fetch('/api/models', { method: 'HEAD' });
    if (r.ok) return 'local';
  } catch { /* fall through */ }
  return 'hosted';
}

async function loadModels() {
  const url = state.mode === 'local' ? '/api/models' : './models.json';
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return r.json();
}

async function loadCacheStatus() {
  if (state.mode === 'local') {
    try {
      const r = await fetch('/api/cache-status');
      if (r.ok) return r.json();
    } catch { /* no server */ }
    return {};
  }
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

function computeWarnings(modelName, quant) {
  const w = [];
  if (/^granite-4/i.test(modelName)) w.push('needs SSM_SCAN');
  if (quant === 'Q1_0') w.push('needs Q1_0');
  return w;
}

function cacheKey(v) { return `${v.repo}/${v.filename}`; }
function variantFitsDevice(v) { return variantFits(v.sizeMB, state.budget.budgetMB, OVERHEAD); }
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

function renderHeader() {
  const d = state.device;
  const b = state.budget;
  $('mode-badge').textContent = state.mode;

  const uaShort = d.userAgent.match(/(Firefox|Chrome|CriOS|Edg|Safari)\/[\d.]+/)?.[0] || 'browser';
  const gpuStr = d.gpu
    ? [d.gpu.vendor, d.gpu.architecture, d.gpu.device].filter(Boolean).join(' ').trim()
    : 'no WebGPU';
  $('device-line').textContent = `${uaShort} · ${d.platform || 'unknown'} · ${gpuStr || 'WebGPU info unavailable'}`;

  const pieces = [];
  if (b.memGB !== null) pieces.push(`deviceMemory ${b.memGB} GB`);
  if (b.quotaMB !== null) pieces.push(`storage quota ${(b.quotaMB / 1024).toFixed(1)} GB`);
  const budgetGB = (b.budgetMB / 1024).toFixed(1);
  $('budget-line').textContent =
    `Model budget: ~${budgetGB} GB · ${pieces.join(' · ') || 'using default'} · source: ${b.source}`;

  $('output-actions-local').hidden = state.mode !== 'local';
  const hubSection = $('hub-section');
  if (hubSection) hubSection.hidden = state.mode !== 'hosted';
  const purgeBtn = $('btn-purge');
  if (purgeBtn) purgeBtn.hidden = state.mode !== 'hosted';
  renderHfSection();
}

function renderHfSection() {
  if (state.mode !== 'hosted') return;
  const signinBtn = $('btn-signin');
  const submitBtn = $('btn-submit');
  const userEl = $('hf-user');
  if (!signinBtn || !submitBtn || !userEl) return;

  if (!isHubConfigured()) {
    signinBtn.disabled = true;
    signinBtn.textContent = 'HF hub not configured';
    signinBtn.title = 'Set HF_DATASET_REPO in bench-config.js';
    submitBtn.hidden = true;
    userEl.textContent = '';
    return;
  }

  if (state.hfSession) {
    signinBtn.textContent = 'Sign out';
    signinBtn.disabled = false;
    submitBtn.hidden = false;
    const eligible = submittableResults();
    submitBtn.disabled = eligible.length === 0;
    submitBtn.title = eligible.length === 0 && state.results.length > 0
      ? `Need at least ${MIN_ITERATIONS_FOR_SUBMIT} successful iterations per variant to submit`
      : '';
    const who = state.hfSession.userName ? `@${state.hfSession.userName}` : 'signed in';
    const hint = eligible.length > 0
      ? ` · ${eligible.length}/${state.results.length} variants eligible`
      : '';
    userEl.textContent = `${who} · → ${HF_DATASET_REPO}${hint}`;
  } else {
    signinBtn.textContent = 'Sign in with Hugging Face';
    signinBtn.disabled = false;
    submitBtn.hidden = true;
    userEl.textContent = '';
  }
}

function renderModels() {
  const panel = $('models-panel');
  panel.innerHTML = '';

  const groups = groupByFamily(state.variants);
  for (const [family, variants] of groups) {
    const fitsCount = variants.filter(variantFitsDevice).length;
    const allFit = fitsCount === variants.length;

    const familyEl = document.createElement('details');
    familyEl.className = 'family';
    familyEl.dataset.family = family;
    familyEl.open = true;

    const summary = document.createElement('summary');
    summary.innerHTML = `
      <input type="checkbox" class="family-select-all" data-family="${escapeAttr(family)}"${allFit ? ' checked' : ''}>
      <span class="family-name">${escapeText(family)}</span>
      <span class="family-stats">(${variants.length} variants, ${fitsCount} fit)</span>
    `;
    if (/^granite-4/i.test(family)) {
      const w = document.createElement('span');
      w.className = 'family-warnings';
      w.textContent = '⚠ needs SSM_SCAN in llama.cpp';
      summary.appendChild(w);
    }
    familyEl.appendChild(summary);

    for (const v of variants) {
      const row = document.createElement('label');
      row.className = 'variant-row';
      if (!variantFitsDevice(v)) row.classList.add('non-fit');
      row.dataset.key = cacheKey(v);

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.className = 'variant-select';
      cb.dataset.key = cacheKey(v);
      cb.checked = variantFitsDevice(v);

      const label = document.createElement('span');
      label.className = 'variant-label';
      label.innerHTML = `<b>${escapeText(v.quant)}</b> · <code>${escapeText(v.filename)}</code>`;

      const size = document.createElement('span');
      size.className = 'size';
      size.textContent = v.sizeMB > 0 ? formatSize(v.sizeMB) : '?';

      const badges = document.createElement('span');
      badges.className = 'badges';
      updateBadgesForVariant(badges, v);

      row.append(cb, label, size, badges);
      familyEl.appendChild(row);
    }
    panel.appendChild(familyEl);
  }
}

function updateBadgesForVariant(badgesEl, v) {
  badgesEl.innerHTML = '';
  if (isCached(v)) badgesEl.appendChild(makeBadge('cached', 'cache-badge'));
  for (const w of v.warnings) badgesEl.appendChild(makeBadge(w, 'warn-badge'));
}

function refreshCacheBadge(v) {
  const row = document.querySelector(`.variant-row[data-key="${cssEscape(cacheKey(v))}"]`);
  if (!row) return;
  const badges = row.querySelector('.badges');
  if (badges) updateBadgesForVariant(badges, v);
}

function makeBadge(text, cls) {
  const el = document.createElement('span');
  el.className = cls;
  el.textContent = text;
  return el;
}

function formatSize(mb) {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(0)} MB`;
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
  document.querySelectorAll('.family-select-all').forEach(el => {
    el.addEventListener('change', () => {
      const family = el.dataset.family;
      const rows = document.querySelectorAll(
        `details.family[data-family="${cssEscape(family)}"] .variant-select`,
      );
      rows.forEach(cb => { cb.checked = el.checked; });
      updateButtons();
    });
    el.addEventListener('click', e => e.stopPropagation());
  });
  document.querySelectorAll('.variant-select').forEach(el => {
    el.addEventListener('change', updateButtons);
  });
}

function wireFilters() {
  ['hide-ud', 'hide-iq', 'hide-hifp'].forEach(id => {
    $(id).addEventListener('change', applyFilters);
  });
}

function wireIterationsInput() {
  const el = $('iterations-input');
  if (!el) return;
  el.value = String(state.iterations);
  el.addEventListener('change', () => {
    const n = Math.max(1, Math.min(50, parseInt(el.value, 10) || DEFAULT_ITERATIONS));
    state.iterations = n;
    el.value = String(n);
  });
}

function submittableResults() {
  return state.results.filter(r =>
    r.status === 'done' && (r.metrics?.iterations || 0) >= MIN_ITERATIONS_FOR_SUBMIT,
  );
}

function applyFilters() {
  const hideUd = $('hide-ud').checked;
  const hideIq = $('hide-iq').checked;
  const hideHifp = $('hide-hifp').checked;
  document.querySelectorAll('.variant-row').forEach(row => {
    const v = state.variants.find(x => cacheKey(x) === row.dataset.key);
    if (!v) return;
    const isUd = v.quant.startsWith('UD-');
    const isIq = /^IQ/.test(v.quant) || /^UD-IQ/.test(v.quant);
    const isHifp = /^(BF16|F16|bf16|f16)$/.test(v.quant);
    const hide = (hideUd && isUd) || (hideIq && isIq) || (hideHifp && isHifp);
    row.style.display = hide ? 'none' : '';
  });
}

function getCheckedVariants() {
  return Array.from(document.querySelectorAll('.variant-select:checked'))
    .map(cb => state.variants.find(v => cacheKey(v) === cb.dataset.key))
    .filter(Boolean);
}

function updateButtons() {
  const checked = getCheckedVariants();
  const cachedChecked = checked.filter(isCached);
  $('btn-download').disabled = state.running || checked.length === 0;
  $('btn-run').disabled = state.running || cachedChecked.length === 0;
  $('btn-abort').disabled = !state.running;
  $('queue-status').textContent = checked.length
    ? `${checked.length} selected · ${cachedChecked.length} cached`
    : '';
}

// ──────────────── progress table ────────────────

function showProgressPanel() { $('progress-panel').hidden = false; }

function progressRowFor(v) {
  const key = cacheKey(v);
  const tbody = $('progress-table').querySelector('tbody');
  let tr = tbody.querySelector(`tr[data-key="${cssEscape(key)}"]`);
  if (!tr) {
    tr = document.createElement('tr');
    tr.dataset.key = key;
    tr.className = 'row-queued';
    tr.innerHTML = `
      <td>${escapeText(v.modelName)}</td>
      <td>${escapeText(v.quant)}</td>
      <td class="status">queued</td>
      <td class="num prefill">—</td>
      <td class="num decode">—</td>
      <td class="num wall">—</td>
      <td class="err"></td>
    `;
    tbody.appendChild(tr);
  }
  return {
    setStatus(status, msg) {
      tr.className = `row-${rowClassFor(status)}`;
      tr.querySelector('.status').textContent = msg ? `${status} — ${msg}` : status;
    },
    setProgress(fraction, downloaded, total) {
      const pct = (fraction * 100).toFixed(1);
      const detail = total > 0
        ? `${pct}% (${formatSize(downloaded / (1024 * 1024))} / ${formatSize(total / (1024 * 1024))})`
        : '';
      tr.querySelector('.status').textContent = detail ? `downloading ${detail}` : 'downloading';
    },
    fillFromRecord(record) {
      tr.className = `row-${record.status === 'done' ? 'ok' : 'error'}`;
      tr.querySelector('.status').textContent = record.status;
      tr.querySelector('.prefill').textContent = record.metrics?.prefill_tok_s ?? '—';
      tr.querySelector('.decode').textContent = record.metrics?.decode_tok_s ?? '—';
      tr.querySelector('.wall').textContent = record.wallTimeMs
        ? (record.wallTimeMs / 1000).toFixed(1)
        : '—';
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
  showProgressPanel();

  for (const v of variants) {
    if (state.aborted) break;
    const row = progressRowFor(v);
    row.setStatus('downloading', '');
    try {
      const { stream, contentLength } = await state.source.fetchModel(v.repo, v.filename);
      const reader = stream.getReader();
      let read = 0;
      while (true) {
        if (state.aborted) { try { reader.cancel(); } catch {} break; }
        const { done, value } = await reader.read();
        if (done) break;
        read += value.length;
        row.setProgress(contentLength ? read / contentLength : 0, read, contentLength);
      }
      if (!state.aborted) {
        state.cacheStatus[cacheKey(v)] = { cachedBytes: read };
        refreshCacheBadge(v);
        row.setStatus('cached', formatSize(read / (1024 * 1024)));
      }
    } catch (err) {
      row.setStatus('error', err.message);
      logLine(`Download failed: ${v.filename}: ${err.message}`);
    }
  }

  // Refresh cache inventory from server to reconcile any partial downloads.
  state.cacheStatus = await loadCacheStatus();
  document.querySelectorAll('.variant-row').forEach(row => {
    const v = state.variants.find(x => cacheKey(x) === row.dataset.key);
    if (v) refreshCacheBadge(v);
  });

  state.running = false;
  updateButtons();
}

// ──────────────── Run ────────────────

async function onRunClick() {
  const variants = getCheckedVariants().filter(isCached);
  if (variants.length === 0) return;

  state.running = true;
  state.aborted = false;
  state.results = [];
  updateButtons();
  showProgressPanel();

  const machine = await machineInfo();
  const browser = browserInfo();

  for (const v of variants) {
    if (state.aborted) break;
    const row = progressRowFor(v);
    row.setStatus('running', '');
    const start = performance.now();

    const variantResult = await runVariantWithIterations(v, row);

    const wallTimeMs = performance.now() - start;
    const record = makeRecord(v, variantResult, machine, browser, wallTimeMs);
    state.results.push(record);
    row.fillFromRecord(record);

    try {
      localStorage.setItem('webgpu-bench:lastRun', JSON.stringify(state.results));
    } catch { /* quota */ }

    if (state.mode === 'local' && $('save-local')?.checked) {
      fetch('/api/results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(record),
      }).catch(err => logLine(`POST /api/results failed: ${err.message}`));
    }

    await sleep(YIELD_BETWEEN_RUNS_MS);
  }

  renderOutput();
  state.running = false;
  updateButtons();
  renderHfSection();
}

// Runs one variant: CPU baseline (1x, for reference token IDs + consistency),
// then N GPU iterations (consistency check on the first only to save time).
// Returns an aggregate that makeRecord consumes.
async function runVariantWithIterations(v, row) {
  const iterations = Math.max(1, state.iterations || DEFAULT_ITERATIONS);

  // ─── CPU baseline ───
  row.setStatus('cpu-baseline', 'generating reference tokens');
  let cpuResult;
  try {
    cpuResult = await runBenchmarkCore({
      source: state.source,
      modelFile: v.filename,
      hfRepo: v.repo,
      prompt: DEFAULT_PROMPT,
      nPredict: DEFAULT_N_PREDICT,
      nCtx: DEFAULT_N_CTX,
      nGpuLayers: 0,
      refTokenIds: null,
      onStatus: (status, msg) => row.setStatus(`cpu/${status}`, msg),
      onProgress: (fr, downloaded, total) => row.setProgress(fr, downloaded, total),
      onLog: logLine,
    });
  } catch (err) {
    cpuResult = { status: 'error', error: err.message || String(err) };
  }

  if (cpuResult.status !== 'done') {
    return {
      status: 'error',
      error: `CPU baseline failed: ${cpuResult.error || 'unknown'}`,
      iterations: 0,
      cpu: cpuResult,
      gpuSamples: [],
      consistency: null,
      gpuCore: null,
    };
  }

  const refTokenIds = (cpuResult.metrics?.token_ids || []).join(',');

  // ─── GPU iterations ───
  const gpuSamples = [];
  let consistency = null;
  let gpuCore = null;

  for (let i = 0; i < iterations; i++) {
    if (state.aborted) break;
    row.setStatus('gpu-run', `iteration ${i + 1}/${iterations}`);
    let gpuResult;
    try {
      gpuResult = await runBenchmarkCore({
        source: state.source,
        modelFile: v.filename,
        hfRepo: v.repo,
        prompt: DEFAULT_PROMPT,
        nPredict: DEFAULT_N_PREDICT,
        nCtx: DEFAULT_N_CTX,
        nGpuLayers: DEFAULT_N_GPU_LAYERS,
        // Only the first iteration runs the consistency check — the result
        // is deterministic with greedy decoding, so subsequent iterations
        // would just repeat the same check.
        refTokenIds: i === 0 ? (refTokenIds || null) : null,
        onStatus: (s, m) => row.setStatus(`gpu${i + 1}/${s}`, m),
        onProgress: (fr, d, t) => row.setProgress(fr, d, t),
        onLog: logLine,
      });
    } catch (err) {
      gpuResult = { status: 'error', error: err.message || String(err) };
    }

    if (gpuResult.status !== 'done') {
      return {
        status: 'error',
        error: `GPU iteration ${i + 1} failed: ${gpuResult.error || 'unknown'}`,
        iterations: gpuSamples.length,
        cpu: cpuResult,
        gpuSamples,
        consistency,
        gpuCore: gpuCore || gpuResult,
      };
    }

    gpuSamples.push({
      prefill_tok_s: gpuResult.metrics?.prefill_tok_s ?? 0,
      decode_tok_s: gpuResult.metrics?.decode_tok_s ?? 0,
      n_p_eval: gpuResult.metrics?.n_p_eval ?? 0,
      n_eval: gpuResult.metrics?.n_eval ?? 0,
      t_p_eval_ms: gpuResult.metrics?.t_p_eval_ms ?? 0,
      t_eval_ms: gpuResult.metrics?.t_eval_ms ?? 0,
    });
    if (i === 0) {
      consistency = gpuResult.consistency || null;
      gpuCore = gpuResult;
    }

    await sleep(YIELD_BETWEEN_ITERATIONS_MS);
  }

  return {
    status: gpuSamples.length > 0 ? 'done' : 'error',
    error: gpuSamples.length === 0 ? 'no GPU iterations completed' : null,
    iterations: gpuSamples.length,
    cpu: cpuResult,
    gpuSamples,
    consistency,
    gpuCore,
  };
}

function mean(arr, key) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, x) => a + (x[key] || 0), 0) / arr.length;
}
function stdev(arr, key) {
  if (arr.length < 2) return 0;
  const m = mean(arr, key);
  return Math.sqrt(arr.reduce((a, x) => a + ((x[key] || 0) - m) ** 2, 0) / arr.length);
}
function round2(n) { return Number.isFinite(n) ? parseFloat(n.toFixed(2)) : 0; }

function makeRecord(v, vr, machine, browser, wallTimeMs) {
  const first = vr.gpuSamples[0] || {};
  const metrics = vr.gpuSamples.length > 0 ? {
    prefill_tok_s: round2(mean(vr.gpuSamples, 'prefill_tok_s')),
    decode_tok_s: round2(mean(vr.gpuSamples, 'decode_tok_s')),
    prefill_tok_s_stdev: round2(stdev(vr.gpuSamples, 'prefill_tok_s')),
    decode_tok_s_stdev: round2(stdev(vr.gpuSamples, 'decode_tok_s')),
    prefill_samples: vr.gpuSamples.map(s => round2(s.prefill_tok_s)),
    decode_samples: vr.gpuSamples.map(s => round2(s.decode_tok_s)),
    iterations: vr.iterations,
    // Retain first-iteration detail for backward-compat with dashboard tables.
    n_p_eval: first.n_p_eval,
    n_eval: first.n_eval,
    t_p_eval_ms: first.t_p_eval_ms,
    t_eval_ms: first.t_eval_ms,
  } : null;

  const cpuBaseline = vr.cpu?.status === 'done' && vr.cpu.metrics ? {
    prefill_tok_s: vr.cpu.metrics.prefill_tok_s,
    decode_tok_s: vr.cpu.metrics.decode_tok_s,
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
    nGpuLayers: DEFAULT_N_GPU_LAYERS,
    timestamp: new Date().toISOString(),
    wallTimeMs,
    webgpuAvailable: vr.gpuCore?.webgpuAvailable ?? !!navigator.gpu,
    gpuAdapterInfo: vr.gpuCore?.gpuAdapterInfo ?? null,
    buildType: vr.gpuCore?.buildType ?? null,
    metrics,
    consistency: vr.consistency ?? null,
    cpu_baseline: cpuBaseline,
    output: vr.gpuCore?.output || '',
    machine,
    source: 'bench.html',
  };
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ──────────────── Output ────────────────

function renderOutput() {
  $('output-textarea').value = generateMarkdown(state.results);
  $('output-panel').hidden = false;
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
    body += `| Model | Variant | Size | Prefill tok/s | Decode tok/s | Wall s |\n`;
    body += `|---|---|---:|---:|---:|---:|\n`;
    for (const r of passed) {
      body += `| ${r.model} | ${r.variant} | ${formatSize(r.sizeMB)} | ${
        r.metrics?.prefill_tok_s ?? '—'} | ${r.metrics?.decode_tok_s ?? '—'} | ${
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
  $('btn-copy').addEventListener('click', async () => {
    const text = $('output-textarea').value;
    try {
      await navigator.clipboard.writeText(text);
      flashButton($('btn-copy'), 'Copied!');
    } catch {
      $('output-textarea').select();
      try { document.execCommand('copy'); flashButton($('btn-copy'), 'Copied!'); } catch {}
    }
  });

  $('btn-download-json').addEventListener('click', () => {
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

// ──────────────── Abort ────────────────

function wireAbortHandler() {
  $('btn-abort').addEventListener('click', () => {
    state.aborted = true;
    $('btn-abort').disabled = true;
    logLine('Abort requested — will stop between variants.');
  });
}

function wirePurgeHandler() {
  const btn = $('btn-purge');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    if (state.mode !== 'hosted') return;
    if (!confirm('Delete all cached GGUF files from OPFS? This frees browser storage but re-downloads will be needed.')) return;
    try {
      await purgeOpfs();
      state.cacheStatus = {};
      document.querySelectorAll('.variant-row').forEach(row => {
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
      submitBtn.disabled = true;
      const original = submitBtn.textContent;
      submitBtn.textContent = 'Submitting…';
      try {
        const first = eligible[0];
        const res = await submitResultsToDataset(eligible, {
          token: state.hfSession.accessToken,
          machineSlug: first.machine?.slug || 'unknown',
          browser: first.browser || 'unknown-browser',
        });
        const link = res.commitUrl
          || `https://huggingface.co/datasets/${HF_DATASET_REPO}/blob/main/${res.path}`;
        logLine(`Submitted ${eligible.length} variant(s): ${link}`);
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
  $('btn-download').addEventListener('click', onDownloadClick);
  $('btn-run').addEventListener('click', onRunClick);
}

// ──────────────── Init ────────────────

async function init() {
  state.mode = await detectMode();
  state.source = state.mode === 'local' ? localSource() : hostedSource();
  state.budget = await getDeviceBudgetMB();
  state.device = await describeDevice();
  state.models = await loadModels();
  state.cacheStatus = await loadCacheStatus();
  state.variants = flattenVariants(state.models);

  // Resume any existing HF OAuth session / complete redirect flow.
  if (state.mode === 'hosted') {
    try { state.hfSession = await resumeHFSession(); } catch { /* ignore */ }
  }

  renderHeader();
  renderModels();
  wireSelectionHandlers();
  wireFilters();
  wireIterationsInput();
  wireRunHandlers();
  wireAbortHandler();
  wirePurgeHandler();
  wireHubHandlers();
  wireOutputHandlers();
  updateButtons();
}

init().catch(err => {
  const el = $('models-loading');
  if (el) el.textContent = `Error loading models: ${err.message}`;
  console.error(err);
});
