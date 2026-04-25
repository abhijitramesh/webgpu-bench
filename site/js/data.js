import { fetchRecentRuns } from './dataset.js';
import { HF_DATASET_REPO } from './run/config.js';

let cachedData = null;

export async function loadData() {
  if (cachedData) return cachedData;

  // Static baseline — pre-built by CI from the dataset on the last code push.
  const baselineResp = await fetch('data/combined.json');
  const baseline = await baselineResp.json();

  // Live delta — pull recent submissions from the HF dataset that landed
  // since baseline.meta.generatedAt. Best-effort: a CORS / network / parse
  // failure here just leaves the dashboard on the static baseline rather
  // than blocking the whole page.
  try {
    const since = baseline.meta?.generatedAt;
    const { records, machines, fileCount } = await fetchRecentRuns(HF_DATASET_REPO, since);
    if (fileCount > 0) {
      mergeLiveDelta(baseline, records, machines);
    }
  } catch (err) {
    console.warn(`Live dataset sync skipped: ${err.message}`);
  }

  cachedData = baseline;
  return cachedData;
}

/* Append fresh records into the baseline and recompute the meta lookups so
   the dashboard's filters/charts see them as first-class data. Records that
   already exist in the baseline (deduped by a composite key) are dropped —
   if CI ran recently, the static combined.json already contains them. */
function mergeLiveDelta(baseline, records, machines) {
  const baselineKeys = new Set(baseline.results.map(keyOf));
  const fresh = records.filter(r => !baselineKeys.has(keyOf(r)));
  if (fresh.length === 0) return;

  baseline.results.push(...fresh);

  // meta.models / meta.browsers union.
  const modelsSet = new Set(baseline.meta.models || []);
  const browsersSet = new Set(baseline.meta.browsers || []);
  for (const r of fresh) {
    if (r.model) modelsSet.add(r.model);
    if (r.browser) browsersSet.add(r.browser);
  }
  baseline.meta.models = [...modelsSet].sort();
  baseline.meta.browsers = [...browsersSet].sort();

  // meta.machines: keep existing entries, add any new slugs from the live
  // delta, then recompute resultCount/passCount across baseline+fresh.
  const machineMap = new Map((baseline.meta.machines || []).map(m => [m.slug, m]));
  for (const m of machines) {
    if (!machineMap.has(m.slug)) machineMap.set(m.slug, { ...m });
  }
  for (const m of machineMap.values()) {
    m.resultCount = 0;
    m.passCount = 0;
  }
  // Per-machine submitter aggregation — counts contributions and tracks the
  // most recent submission so the dashboard can render a stacked-avatar row
  // sorted by activity. Mirrors scripts/build-site.js logic so live and
  // baseline data stay byte-equivalent.
  const submitterAccumulator = new Map(); // slug → Map(key → {profile, count, latestAt})
  for (const r of baseline.results) {
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
  baseline.meta.machines = [...machineMap.values()];
  baseline.meta.generatedAt = new Date().toISOString();
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

/* Composite key for dedupe. The submit pipeline writes one record per
   (machine, browser, model, variant, timestamp) tuple, so this is unique
   in practice. */
function keyOf(r) {
  return `${r.machineSlug}|${r.browser}|${r.model}|${r.variant}|${r.timestamp}`;
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
