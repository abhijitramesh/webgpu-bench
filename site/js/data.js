import { fetchAllRuns } from './dataset.js';
import { HF_DATASET_REPO } from './run/config.js';

// In-memory cache for the current page session.
let cachedData = null;
// sessionStorage cache so a refresh-within-a-minute doesn't re-fetch the
// entire dataset. Short TTL — submissions land continuously and the
// dashboard is the surface where we actually want freshness.
const SESSION_CACHE_KEY = 'webgpu-bench:dashboard-data';
const SESSION_CACHE_TTL_MS = 60 * 1000;

export async function loadData() {
  if (cachedData) return cachedData;

  const fromSession = readSessionCache();
  if (fromSession) {
    cachedData = fromSession;
    return cachedData;
  }

  // Single source of truth: the HF dataset repo. No static baseline. A new
  // dashboard with zero submissions shows an empty state until something is
  // submitted.
  const empty = makeEmptyDataset();
  try {
    const { records, machines, fileCount } = await fetchAllRuns(HF_DATASET_REPO);
    if (fileCount > 0) {
      mergeRecords(empty, records, machines);
    }
    cachedData = empty;
    writeSessionCache(cachedData);
  } catch (err) {
    console.warn(`Live dataset load failed: ${err.message}`);
    cachedData = empty;
  }
  return cachedData;
}

function makeEmptyDataset() {
  return {
    meta: {
      machines: [],
      models: [],
      browsers: [],
      generatedAt: new Date().toISOString(),
    },
    results: [],
  };
}

/* Append records into an empty payload and recompute the meta lookups. Same
   shape the old combined.json had, so all downstream consumers (charts,
   tables, machine cards) work unchanged. */
function mergeRecords(payload, records, machines) {
  if (records.length === 0) return;

  payload.results.push(...records);

  const modelsSet = new Set(payload.meta.models || []);
  const browsersSet = new Set(payload.meta.browsers || []);
  for (const r of records) {
    if (r.model) modelsSet.add(r.model);
    if (r.browser) browsersSet.add(r.browser);
  }
  payload.meta.models = [...modelsSet].sort();
  payload.meta.browsers = [...browsersSet].sort();

  const machineMap = new Map((payload.meta.machines || []).map(m => [m.slug, m]));
  for (const m of machines) {
    if (!machineMap.has(m.slug)) machineMap.set(m.slug, { ...m });
  }
  for (const m of machineMap.values()) {
    m.resultCount = 0;
    m.passCount = 0;
  }

  // Per-machine submitter aggregation — counts contributions and tracks the
  // most-recent submission so the machine card can render a stacked-avatar
  // row sorted by activity.
  const submitterAccumulator = new Map(); // slug → Map(key → {profile, count, latestAt})
  for (const r of payload.results) {
    const m = machineMap.get(r.machineSlug);
    if (!m) continue;
    m.resultCount += 1;
    if (r.status === 'done') m.passCount += 1;
    const sb = r.submittedBy;
    if (!sb?.name) continue;
    const key = sb.hubId || sb.name;
    if (!submitterAccumulator.has(r.machineSlug)) submitterAccumulator.set(r.machineSlug, new Map());
    const inner = submitterAccumulator.get(r.machineSlug);
    const cur = inner.get(key);
    if (!cur) {
      inner.set(key, { profile: sb, count: 1, latestAt: r.timestamp || '' });
    } else {
      cur.count += 1;
      if (r.timestamp && r.timestamp > cur.latestAt) {
        cur.profile = sb;
        cur.latestAt = r.timestamp;
      }
    }
  }
  for (const [slug, inner] of submitterAccumulator) {
    const m = machineMap.get(slug);
    if (!m) continue;
    m.submitters = [...inner.values()]
      .map(({ profile, count, latestAt }) => ({ ...profile, count, latestAt }))
      .sort((a, b) => b.count - a.count || (b.latestAt || '').localeCompare(a.latestAt || ''));
  }
  payload.meta.machines = [...machineMap.values()];
  payload.meta.generatedAt = new Date().toISOString();
}

function readSessionCache() {
  try {
    const raw = sessionStorage.getItem(SESSION_CACHE_KEY);
    if (!raw) return null;
    const { ts, data } = JSON.parse(raw);
    if (typeof ts !== 'number' || (Date.now() - ts) > SESSION_CACHE_TTL_MS) return null;
    return data;
  } catch {
    return null;
  }
}

function writeSessionCache(data) {
  try {
    sessionStorage.setItem(SESSION_CACHE_KEY, JSON.stringify({ ts: Date.now(), data }));
  } catch { /* quota or disabled */ }
}

/* Fold the (d=0, d=N) GPU record pair that Run Study emits per variant
   into a single dashboard row. The d=N record stays canonical
   (`decode_tok_s` / `prefill_tok_s` keep the depth-loaded numbers) so
   existing chart/table consumers keep working unchanged; a new pair of
   `_d0` / `_dN` suffix fields lets depth-aware code pick a specific pass.

   CPU records are pinned to d=0 by the runner, so they pass through
   untouched. Cells with only one half of the pair (plain Run, pre-study
   data, or a partial study) lift their values into the suffix field on
   the side that exists, leaving the other side null — so consumers can
   render `—` without having to know the record's history.

   Within each cell we also tie-break duplicate records per depth bucket
   (same iteration / latest timestamp wins, mirroring selectBestResults)
   so multiple study runs of the same variant collapse cleanly.

   Run AFTER attachCpuBaselineFromCpuRecords (which keys on the
   depth-independent (machine, browser, model, variant) tuple) and
   BEFORE selectBestResults (CPU rows still need cell-dedup; GPU rows
   are already deduped here). */
export function mergeDepthPairs(records) {
  const cells = new Map();
  const cpuRows = [];
  for (const r of records) {
    if (r.nGpuLayers === 0) {
      cpuRows.push(r);
      continue;
    }
    const cellKey = `${r.machineSlug}|${r.browser}|${r.model}|${r.variant}`;
    const bucket = (r.n_depth ?? 0) === 0 ? 'd0' : 'dN';
    const slot = cells.get(cellKey) || { d0: null, dN: null };
    if (!slot[bucket] || isStrongerRecord(r, slot[bucket])) slot[bucket] = r;
    cells.set(cellKey, slot);
  }
  const merged = [...cpuRows];
  for (const { d0, dN } of cells.values()) {
    if (d0 && dN) merged.push(joinDepthPair(d0, dN));
    else if (dN) merged.push(liftSingleDepth(dN, 'dN'));
    else if (d0) merged.push(liftSingleDepth(d0, 'd0'));
  }
  return merged;
}

function isStrongerRecord(a, b) {
  const ai = a.iterations ?? 0;
  const bi = b.iterations ?? 0;
  if (ai !== bi) return ai > bi;
  return (a.timestamp || '') > (b.timestamp || '');
}

const DEPTH_PERF_FIELDS = [
  'decode_tok_s', 'prefill_tok_s',
  'decode_stddev_ts', 'prefill_stddev_ts',
  'pp_test_name', 'tg_test_name',
];

function joinDepthPair(d0, dN) {
  const out = { ...dN };
  for (const f of DEPTH_PERF_FIELDS) {
    out[`${f}_d0`] = d0[f] ?? null;
    out[`${f}_dN`] = dN[f] ?? null;
  }
  out.n_depth_dN = dN.n_depth ?? null;
  return out;
}

function liftSingleDepth(r, bucket) {
  const out = { ...r };
  for (const f of DEPTH_PERF_FIELDS) {
    out[`${f}_d0`] = bucket === 'd0' ? (r[f] ?? null) : null;
    out[`${f}_dN`] = bucket === 'dN' ? (r[f] ?? null) : null;
  }
  out.n_depth_dN = bucket === 'dN' ? (r.n_depth ?? null) : null;
  return out;
}

/* Reduce a flat result set down to one canonical row per
   (machineSlug, browser, model, variant, backend) cell. Picks the row with
   the most iterations; ties break on latest timestamp. This is the
   leaderboard view — "best representative number per cell" — and is what
   the dashboard renders in the table, charts, and stat cards.

   `backend` (CPU vs GPU, derived from nGpuLayers) is part of the key so
   CLI CPU+GPU pairs and browser-flow synthetic CPU rows don't collapse
   into the GPU row. */
export function selectBestResults(records) {
  const bestByCell = new Map();
  for (const r of records) {
    const backend = r.nGpuLayers === 0 ? 'cpu' : 'gpu';
    const key = `${r.machineSlug}|${r.browser}|${r.model}|${r.variant}|${backend}`;
    const cur = bestByCell.get(key);
    if (!cur) {
      bestByCell.set(key, r);
      continue;
    }
    const curIter = cur.iterations ?? 0;
    const newIter = r.iterations ?? 0;
    if (newIter > curIter) {
      bestByCell.set(key, r);
    } else if (newIter === curIter) {
      const curTs = cur.timestamp || '';
      const newTs = r.timestamp || '';
      if (newTs > curTs) bestByCell.set(key, r);
    }
  }
  return [...bestByCell.values()];
}

/* For CLI-flow records that ship CPU and GPU as separate dataset entries,
   look up each GPU record's matching CPU companion (same machine, browser,
   model, variant) and copy its perf into cpu_baseline_*. After this pass,
   GPU records from both submission paths (browser, CLI) carry their CPU
   baseline inline, so the main table can render a single row per cell with
   both numbers side-by-side. No-op on records that already have
   cpu_baseline_* (e.g. browser-flow records, where controller.makeRecord
   embeds it at write time). */
export function attachCpuBaselineFromCpuRecords(results) {
  const cpuByCell = new Map();
  for (const r of results) {
    if (r.nGpuLayers === 0 && r.status === 'done' && (r.decode_tok_s != null || r.prefill_tok_s != null)) {
      const key = `${r.machineSlug}|${r.browser}|${r.model}|${r.variant}`;
      const cur = cpuByCell.get(key);
      // Most-recent wins on tiebreak — matches selectBestResults() semantics.
      if (!cur || (r.timestamp || '') > (cur.timestamp || '')) {
        cpuByCell.set(key, r);
      }
    }
  }
  return results.map(r => {
    if (r.nGpuLayers === 0) return r;
    if (r.cpu_baseline_decode_tok_s != null || r.cpu_baseline_prefill_tok_s != null) return r;
    const key = `${r.machineSlug}|${r.browser}|${r.model}|${r.variant}`;
    const cpu = cpuByCell.get(key);
    if (!cpu) return r;
    return {
      ...r,
      cpu_baseline_decode_tok_s: cpu.decode_tok_s ?? null,
      cpu_baseline_prefill_tok_s: cpu.prefill_tok_s ?? null,
    };
  });
}

/* Synthesize a CPU row for every browser-flow GPU record (the in-page
   bench measures one CPU pass per variant alongside the GPU iterations
   and stamps the result on the same record via cpu_baseline_*). Returns
   only CPU rows — combine real (nGpuLayers === 0) and synthetic ones.
   Used by the CPU-vs-GPU views which want the CPU subset only. */
export function expandCpuRows(results) {
  const real = results.filter(r => r.nGpuLayers === 0);
  const synthetic = synthesizeCpuRowsFromBaseline(results);
  return [...real, ...synthetic];
}

/* Same synthesis as expandCpuRows but returns the originals plus the
   synthesized CPU rows — for the main results table where we want both
   GPU and CPU rows visible. */
export function withSyntheticCpuRows(results) {
  return [...results, ...synthesizeCpuRowsFromBaseline(results)];
}

function synthesizeCpuRowsFromBaseline(results) {
  return results
    .filter(r => r.nGpuLayers !== 0
      && (r.cpu_baseline_decode_tok_s != null || r.cpu_baseline_prefill_tok_s != null))
    .map(r => ({
      ...r,
      decode_tok_s: r.cpu_baseline_decode_tok_s,
      prefill_tok_s: r.cpu_baseline_prefill_tok_s,
      // The CPU baseline is a single-rep measurement (warmup + 1 timed),
      // so it has no stddev. Null out the stddev fields the spread above
      // inherited from the GPU row — otherwise the table renders the
      // CPU avg with the GPU's stddev attached, which is nonsensical.
      decode_stddev_ts: null,
      prefill_stddev_ts: null,
      // CPU baseline runs have no t_eval / n_eval breakdowns — null those
      // out so the table doesn't show stale GPU numbers in CPU rows.
      n_eval: null,
      t_eval_ms: null,
      n_p_eval: null,
      t_p_eval_ms: null,
      // Strip the embedded baseline from synthetic CPU rows so the
      // "CPU decode tok/s" column doesn't duplicate the row's own metric.
      cpu_baseline_decode_tok_s: null,
      cpu_baseline_prefill_tok_s: null,
      cpu_baseline: null,
      nGpuLayers: 0,
    }));
}

export function filterResults(results, filters) {
  return results.filter(r => {
    if (filters.machine && filters.machine !== 'all' && r.machineSlug !== filters.machine) return false;
    if (filters.browser && filters.browser !== 'all' && r.browser !== filters.browser) return false;
    if (filters.model && filters.model !== 'all' && r.model !== filters.model) return false;
    if (filters.backend && filters.backend !== 'all') {
      if (filters.backend === 'cpu' && r.nGpuLayers !== 0) return false;
      if (filters.backend === 'webgpu' && r.nGpuLayers === 0) return false;
    }
    if (filters.status && filters.status !== 'all') {
      if (filters.status === 'pass' && r.status !== 'done') return false;
      if (filters.status === 'fail' && r.status === 'done') return false;
    }
    if (filters.quants && filters.quants.size > 0 && !filters.quants.has(r.variant)) return false;
    return true;
  });
}
