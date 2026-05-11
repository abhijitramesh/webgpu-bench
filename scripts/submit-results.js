// Submit benchmark results to the HF leaderboard dataset.
// Reads results/results.json, strips heavy fields, and pushes one commit per
// (machine slug, browser) group via scripts/push-to-dataset.mjs.
//
// Usage:
//   HF_TOKEN=… HF_DATASET_REPO=owner/webgpu-bench-leaderboard npm run submit

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pushResultsToDataset } from './push-to-dataset.mjs';
import { requireDatasetRepo, requireToken } from './_hub.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const RESULTS_FILE = path.join(ROOT, 'results', 'results.json');

function readResults() {
  if (!fs.existsSync(RESULTS_FILE)) {
    console.error('No results/results.json found. Run benchmarks first: node runner.js');
    process.exit(1);
  }
  const raw = JSON.parse(fs.readFileSync(RESULTS_FILE, 'utf-8'));
  if (!Array.isArray(raw) || raw.length === 0) {
    console.error('results.json is empty.');
    process.exit(1);
  }
  return raw;
}

async function main() {
  const raw = readResults();

  let datasetRepo, token;
  try {
    datasetRepo = requireDatasetRepo();
    token = requireToken();
  } catch (err) {
    console.error(`\n${err.message}\n`);
    console.error('Export HF_TOKEN and HF_DATASET_REPO, then re-run: npm run submit');
    process.exit(1);
  }

  const { uploads, files } = await pushResultsToDataset({
    datasetRepo, token, records: raw,
  });

  const url = `https://huggingface.co/datasets/${datasetRepo}`;
  console.log(`\nPushed ${uploads} file${uploads === 1 ? '' : 's'} to ${url}`);
  for (const f of files) {
    console.log(`  ${url}/blob/main/${f.path}${f.commit ? ` (${f.commit.slice(0, 10)})` : ''}`);
  }
  console.log('\nDashboard CI will pick these up on its next run (or via manual dispatch).');
}

main().catch(err => {
  console.error(`submit failed: ${err.message}`);
  if (err.stack) console.error(err.stack);
  process.exit(1);
});
