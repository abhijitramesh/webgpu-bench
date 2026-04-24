// Push results/results.json to the HF leaderboard dataset.
// Groups records by (machine.slug, browser) and uploads each group as
// runs/{date}/{slug}-{browser}-{epoch}.json with one commit per group.
//
// Usage (CLI):
//   HF_TOKEN=… HF_DATASET_REPO=owner/webgpu-bench-leaderboard \
//     node scripts/push-to-dataset.mjs
//
// Also importable as a library: `pushResultsToDataset({ datasetRepo, token, records })`.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { uploadFile, whoAmI } from '@huggingface/hub';
import {
  requireDatasetRepo, requireToken, stripResult, generateSlug,
  datestamp, runPath, groupBy,
} from './_hub.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const RESULTS_FILE = path.join(ROOT, 'results', 'results.json');

/**
 * Push an array of benchmark result records (already stripped or raw) to the
 * dataset. Returns `{ uploads, files: [{ path, commit }] }`.
 */
export async function pushResultsToDataset({ datasetRepo, token, records, strip = true }) {
  if (!Array.isArray(records) || records.length === 0) {
    throw new Error('No records to push.');
  }
  const processed = strip ? records.map(stripResult) : records;

  const me = await whoAmI({ credentials: { accessToken: token } }).catch(() => null);
  if (me?.name) console.log(`Authenticated as: ${me.name}`);

  const groups = groupBy(processed, r => {
    const slug = generateSlug(r.machine);
    const browser = r.browser || 'unknown-browser';
    return `${slug}::${browser}`;
  });

  const date = datestamp();
  const epoch = Date.now();
  const files = [];

  for (const [key, items] of groups) {
    const [slug, browser] = key.split('::');
    const body = JSON.stringify(items, null, 2);
    const filePath = runPath({ date, slug, browser, epoch });
    const blob = new Blob([body], { type: 'application/json' });
    const commitTitle = `bench: ${slug} / ${browser} / ${items.length} variants`;

    console.log(`→ ${filePath} (${items.length} records, ${(body.length / 1024).toFixed(1)} KB)`);

    const res = await uploadFile({
      repo: { type: 'dataset', name: datasetRepo },
      credentials: { accessToken: token },
      file: { path: filePath, content: blob },
      commitTitle,
    });
    files.push({ path: filePath, commit: res?.commit?.oid || null });
    if (res?.commit?.oid) console.log(`  commit: ${res.commit.oid.slice(0, 10)}`);
  }

  return { uploads: files.length, files };
}

async function cli() {
  const datasetRepo = requireDatasetRepo();
  const token = requireToken();

  if (!fs.existsSync(RESULTS_FILE)) {
    console.error(`No ${path.relative(ROOT, RESULTS_FILE)}; run benchmarks first.`);
    process.exit(1);
  }
  const raw = JSON.parse(fs.readFileSync(RESULTS_FILE, 'utf-8'));
  if (!Array.isArray(raw) || raw.length === 0) {
    console.error('results.json is empty.');
    process.exit(1);
  }

  const { uploads } = await pushResultsToDataset({ datasetRepo, token, records: raw });
  const url = `https://huggingface.co/datasets/${datasetRepo}`;
  console.log(`\nPushed ${uploads} file${uploads === 1 ? '' : 's'} to ${url}`);
}

// Only run CLI when invoked directly, not when imported.
const invokedDirectly = (() => {
  const thisFile = fileURLToPath(import.meta.url);
  return process.argv[1] && path.resolve(process.argv[1]) === thisFile;
})();

if (invokedDirectly) {
  cli().catch(err => {
    console.error(`push failed: ${err.message}`);
    if (err.stack) console.error(err.stack);
    process.exit(1);
  });
}
