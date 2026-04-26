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

/* Reduce a flat result set down to one canonical row per
   (machineSlug, browser, model, variant) cell. Picks the row with the most
   iterations; ties break on latest timestamp. This is the leaderboard view —
   "best representative number per cell" — and is what the dashboard renders
   in the table, charts, and stat cards. */
export function selectBestResults(records) {
  const bestByCell = new Map();
  for (const r of records) {
    const key = `${r.machineSlug}|${r.browser}|${r.model}|${r.variant}`;
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

/* Synthesize "CPU only" rows for the CPU-vs-GPU views.
   Two record sources contribute:
     1) Real CPU runs (nGpuLayers === 0) — produced by the CLI runner which
        alternates CPU and GPU passes.
     2) The cpu_baseline_* fields on every browser-flow record — the in-page
        bench measures one CPU pass per variant alongside the GPU iterations
        and stamps the result on the same record. We turn each of those into
        a synthetic CPU row so the comparison view sees both data shapes.
*/
export function expandCpuRows(results) {
  const real = results.filter(r => r.nGpuLayers === 0);
  const synthetic = results
    .filter(r => r.nGpuLayers !== 0
      && (r.cpu_baseline_decode_tok_s != null || r.cpu_baseline_prefill_tok_s != null))
    .map(r => ({
      ...r,
      decode_tok_s: r.cpu_baseline_decode_tok_s,
      prefill_tok_s: r.cpu_baseline_prefill_tok_s,
      nGpuLayers: 0,
    }));
  return [...real, ...synthetic];
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
