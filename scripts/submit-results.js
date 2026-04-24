// Submit benchmark results to the HF leaderboard dataset.
// Reads results/results.json, strips heavy fields, and pushes one commit per
// (machine slug, browser) group via scripts/push-to-dataset.mjs.
//
// Legacy behavior (write data/machines/{slug}.json and emit PR instructions)
// is preserved behind --legacy-file-only for transition + local debug.
//
// Usage:
//   HF_TOKEN=… HF_DATASET_REPO=owner/webgpu-bench-leaderboard npm run submit
//   npm run submit -- --legacy-file-only

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pushResultsToDataset } from './push-to-dataset.mjs';
import {
  requireDatasetRepo, requireToken, stripResult, generateSlug,
} from './_hub.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const RESULTS_FILE = path.join(ROOT, 'results', 'results.json');
const MACHINES_DIR = path.join(ROOT, 'data', 'machines');

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

function legacyWriteFile(raw) {
  const machine = raw[0].machine;
  if (!machine) {
    console.error('No machine info found in results.');
    process.exit(1);
  }
  const slug = generateSlug(machine);
  const benchmarkConfig = {
    nCtx: raw[0].nCtx,
    nPredict: raw[0].nPredict,
    nGpuLayers: raw[0].nGpuLayers,
  };
  const submission = {
    machine: {
      slug,
      platform: machine.platform,
      arch: machine.arch,
      cpus: machine.cpus,
      totalMemoryGB: machine.totalMemoryGB,
    },
    submittedAt: new Date().toISOString(),
    benchmarkConfig,
    results: raw.map(stripResult),
  };
  fs.mkdirSync(MACHINES_DIR, { recursive: true });
  const outFile = path.join(MACHINES_DIR, `${slug}.json`);
  fs.writeFileSync(outFile, JSON.stringify(submission, null, 2));

  console.log(`Created ${path.relative(ROOT, outFile)}`);
  console.log(`  Machine: ${machine.cpus} (${machine.platform}/${machine.arch}, ${machine.totalMemoryGB}GB)`);
  console.log(`  Slug:    ${slug}`);
  console.log(`  Results: ${raw.length} benchmarks (${raw.filter(r => r.status === 'done').length} passed)`);
  console.log('');
  console.log('Next steps (legacy file path):');
  console.log(`  git add data/machines/${slug}.json`);
  console.log('  git commit -m "Add benchmark results for ' + machine.cpus + '"');
  console.log('  # Then open a PR');
  console.log('');
  console.log('Tip: the default path is now HF dataset push — set HF_TOKEN + HF_DATASET_REPO and drop --legacy-file-only.');
}

async function datasetPush(raw) {
  let datasetRepo, token;
  try {
    datasetRepo = requireDatasetRepo();
    token = requireToken();
  } catch (err) {
    console.error(`\n${err.message}\n`);
    console.error('Either:');
    console.error('  1. Export HF_TOKEN and HF_DATASET_REPO, then re-run: npm run submit');
    console.error('  2. Use the legacy PR flow: npm run submit -- --legacy-file-only');
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

async function main() {
  const args = new Set(process.argv.slice(2));
  const legacyFileOnly = args.has('--legacy-file-only');
  const raw = readResults();

  if (legacyFileOnly) {
    legacyWriteFile(raw);
  } else {
    await datasetPush(raw);
  }
}

main().catch(err => {
  console.error(`submit failed: ${err.message}`);
  if (err.stack) console.error(err.stack);
  process.exit(1);
});
