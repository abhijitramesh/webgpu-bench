// Live read of recent submissions from the HF leaderboard dataset.
//
// The dashboard's static `data/combined.json` is rebuilt only when CI runs,
// so freshly-submitted results don't appear until the next code push. This
// module fetches files written to the dataset since `combined.json` was
// last generated and merges them into the dashboard at load time.
//
// HF endpoints used (no auth, public dataset):
//   GET /api/datasets/<repo>/tree/main/runs?recursive=1   → file listing
//   GET /datasets/<repo>/resolve/main/<path>              → file content
//
// Both endpoints support CORS for public datasets so we can call them
// directly from the dashboard.

const HF = 'https://huggingface.co';

// Safety window for clock skew between the dataset commit timestamps and
// the `meta.generatedAt` we compare against. 10 minutes should be more
// than enough — the cost of overshooting is just a few extra files that
// dedupe out anyway.
const CLOCK_SKEW_MS = 10 * 60 * 1000;

// Cap on parallel/total file fetches per dashboard load. The dashboard now
// pulls the entire dataset live (no static baseline), so this cap is the
// upper bound on how many run files the page will download at once. 1000
// is conservative — actual bench submissions are typically ≤ 1 KB each so
// the bandwidth ceiling is well under a megabyte even at the cap.
const MAX_FETCH = 1000;

/* Fetch the runs/ tree from the dataset. Returns the file entries that
   look newer than `sinceISO` (with a clock-skew buffer applied). On any
   network/CORS/parse failure, returns an empty array — the dashboard then
   silently falls back to the static combined.json baseline. */
export async function listRecentRunFiles(datasetRepo, sinceISO) {
  if (!datasetRepo) return [];
  // Cache-bust the listing — HF's CDN can serve a stale tree response, and
  // we specifically care about reading-our-own-write after a submit.
  const url = `${HF}/api/datasets/${datasetRepo}/tree/main/runs?recursive=1&_=${Date.now()}`;
  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) {
    throw new Error(`tree listing ${resp.status} ${resp.statusText}`);
  }
  const tree = await resp.json();
  if (!Array.isArray(tree)) return [];

  const cutoff = sinceISO ? new Date(sinceISO).getTime() - CLOCK_SKEW_MS : 0;
  const files = tree
    .filter(it => it.type === 'file' && it.path.endsWith('.json'))
    .filter(it => {
      if (!cutoff) return true;
      const t = it.lastCommit?.date ? new Date(it.lastCommit.date).getTime() : NaN;
      // Files with no commit timestamp pass through — better to over-include
      // than miss the user's own freshly-pushed submission.
      return Number.isNaN(t) ? true : t > cutoff;
    });

  return files.slice(0, MAX_FETCH);
}

async function fetchRunFile(datasetRepo, filePath) {
  const url = `${HF}/datasets/${datasetRepo}/resolve/main/${filePath}`;
  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) throw new Error(`fetch ${filePath}: ${resp.status}`);
  return resp.json();
}

/* List the dataset tree and download every file in `runs/`. Pure-live
   variant of fetchRecentRuns — no cutoff, returns the entire dataset.
   Caller is responsible for rate-limiting/caching. */
export async function fetchAllRuns(datasetRepo) {
  return fetchRunsBatch(datasetRepo, await listRecentRunFiles(datasetRepo, null));
}

/* List the dataset tree and download every file that's newer than the
   baseline's generatedAt. Kept for callers that still want a delta view. */
export async function fetchRecentRuns(datasetRepo, sinceISO) {
  return fetchRunsBatch(datasetRepo, await listRecentRunFiles(datasetRepo, sinceISO));
}

async function fetchRunsBatch(datasetRepo, files) {
  if (files.length === 0) return { records: [], machines: [], fileCount: 0 };

  const records = [];
  const machinesBySlug = new Map();

  // Fetch in parallel — HF's CDN handles concurrent reads fine.
  const results = await Promise.allSettled(
    files.map(f => fetchRunFile(datasetRepo, f.path)),
  );

  for (const res of results) {
    if (res.status !== 'fulfilled' || !Array.isArray(res.value)) continue;
    const arr = res.value;
    for (const r of arr) {
      const slug = generateSlug(r.machine);
      records.push(flattenForDashboard(r, slug));
      if (!machinesBySlug.has(slug) && r.machine) {
        machinesBySlug.set(slug, {
          slug,
          cpus: r.machine.cpus || 'unknown',
          platform: r.machine.platform || 'unknown',
          arch: r.machine.arch || 'unknown',
          totalMemoryGB: r.machine.totalMemoryGB || 0,
          submittedAt: r.timestamp || new Date().toISOString(),
          // Per-machine resultCount/passCount get computed by the caller
          // after the merge — leaving them as 0 here is a placeholder.
          resultCount: 0,
          passCount: 0,
          llamaCppCommit: r.llamaCppCommit ?? null,
          llamaCppDescribe: r.llamaCppDescribe ?? null,
        });
      }
    }
  }

  return { records, machines: [...machinesBySlug.values()], fileCount: files.length };
}

/* Flatten a raw dataset record into the same shape `scripts/build-site.js`
   produces. Keep field-for-field aligned with build-site.js so the merged
   results are indistinguishable from the baseline. */
function flattenForDashboard(r, slug) {
  // New-format records have metrics.tests = [{name:'pp512',...},{name:'tg128',...}].
  // Old-format records have flat metrics.prefill_tok_s / decode_tok_s only.
  // Surface both shapes so the table can render llama-bench-style "avg \u00b1 stddev"
  // when stddev is available without breaking on older rows.
  const tests = Array.isArray(r.metrics?.tests) ? r.metrics.tests : null;
  const pp = tests?.find(t => t.name?.startsWith('pp')) || null;
  const tg = tests?.find(t => t.name?.startsWith('tg')) || null;
  return {
    machineSlug: slug,
    timestamp: r.timestamp,
    browser: r.browser,
    model: r.model,
    repo: r.repo,
    variant: r.variant,
    filename: r.filename,
    sizeMB: r.sizeMB,
    status: r.status,
    error: r.error,
    buildType: r.buildType,
    webgpuAvailable: r.webgpuAvailable,
    nGpuLayers: r.nGpuLayers ?? null,
    wallTimeMs: r.wallTimeMs,
    prefill_tok_s: r.metrics?.prefill_tok_s ?? null,
    decode_tok_s: r.metrics?.decode_tok_s ?? null,
    // llama-bench shape: per-test stddev + the test labels (pp{N} / tg{N})
    prefill_stddev_ts: pp?.stddev_ts ?? r.metrics?.prefill_tok_s_stdev ?? null,
    decode_stddev_ts:  tg?.stddev_ts ?? r.metrics?.decode_tok_s_stdev  ?? null,
    pp_test_name: pp?.name ?? null,
    tg_test_name: tg?.name ?? null,
    pp_n_prompt: pp?.n_prompt ?? r.nPrompt ?? null,
    tg_n_gen:    tg?.n_gen    ?? r.nGen    ?? null,
    // KV-cache depth the timed reps ran at. Mirrors llama-bench's `-d` and
    // is per-test in metrics.tests; record-level r.nDepth is the
    // study/runner-set value, used as a fallback for older exports.
    n_depth: pp?.n_depth ?? tg?.n_depth ?? r.nDepth ?? 0,
    n_p_eval: r.metrics?.n_p_eval ?? null,
    t_p_eval_ms: r.metrics?.t_p_eval_ms ?? null,
    n_eval: r.metrics?.n_eval ?? null,
    t_eval_ms: r.metrics?.t_eval_ms ?? null,
    consistency_rate: r.consistency?.agreement_rate ?? null,
    consistency_first_disagree: r.consistency?.first_disagreement ?? null,
    // Keep these in sync with scripts/build-site.js — the dashboard merges
    // baseline (combined.json) and live (here) records into one table.
    cpu_baseline_prefill_tok_s: r.cpu_baseline?.prefill_tok_s ?? null,
    cpu_baseline_decode_tok_s: r.cpu_baseline?.decode_tok_s ?? null,
    llamaCppCommit: r.llamaCppCommit ?? null,
    llamaCppDescribe: r.llamaCppDescribe ?? null,
    dawnTag: r.dawnTag ?? null,
    submittedBy: r.submittedBy ?? null,
    iterations: r.metrics?.iterations ?? null,
  };
}

// Mirror of scripts/_hub.mjs:generateSlug — keep in sync.
function generateSlug(machine) {
  if (machine?.slug) return machine.slug;
  const cpu = slugify(machine?.cpus || 'unknown');
  const ram = machine?.totalMemoryGB || 0;
  const platform = machine?.platform || 'unknown';
  return `${cpu}-${ram}gb-${platform}`;
}

function slugify(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}
