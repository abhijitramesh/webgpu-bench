// Shared helpers for the push / sync / bootstrap dataset scripts.

export const DEFAULT_DATASET_REPO = process.env.HF_DATASET_REPO || '';

export function requireDatasetRepo() {
  if (!DEFAULT_DATASET_REPO) {
    throw new Error(
      'HF_DATASET_REPO environment variable is not set. ' +
      'Set it to e.g. "owner/webgpu-bench-leaderboard" before running dataset scripts.',
    );
  }
  return DEFAULT_DATASET_REPO;
}

export function requireToken() {
  const t = process.env.HF_TOKEN;
  if (!t) {
    throw new Error(
      'HF_TOKEN environment variable is not set. ' +
      'Create a write token at https://huggingface.co/settings/tokens and export it before running.',
    );
  }
  return t;
}

// Same strip logic as scripts/submit-results.js — keep in sync.
export function stripResult(r) {
  const s = { ...r };
  if (s.metrics) {
    s.metrics = { ...s.metrics };
    delete s.metrics.token_ids;
    delete s.metrics.output;
  }
  delete s.output;
  if (s.consistency) {
    s.consistency = { ...s.consistency };
    delete s.consistency.matches;
  }
  return s;
}

export function slugify(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

export function generateSlug(machine) {
  if (machine?.slug) return machine.slug;
  const cpu = slugify(machine?.cpus || 'unknown');
  const ram = machine?.totalMemoryGB || 0;
  const platform = machine?.platform || 'unknown';
  return `${cpu}-${ram}gb-${platform}`;
}

export function datestamp(d = new Date()) {
  return d.toISOString().slice(0, 10);
}

export function runPath({ date, slug, browser, tag = '', epoch = Date.now() }) {
  const suffix = tag ? `-${tag}` : '';
  return `runs/${date}/${slug}-${browser}${suffix}-${epoch}.json`;
}

// Group an array by a key function.
export function groupBy(arr, keyFn) {
  const m = new Map();
  for (const item of arr) {
    const k = keyFn(item);
    if (!m.has(k)) m.set(k, []);
    m.get(k).push(item);
  }
  return m;
}
