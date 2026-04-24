// Pull every runs/**/*.json from the HF leaderboard dataset and regroup them
// into data/machines/{slug}.json so scripts/build-site.js can consume the
// dashboard input in its existing format.
//
// Usage:
//   HF_DATASET_REPO=owner/webgpu-bench-leaderboard node scripts/sync-from-dataset.mjs
//
// HF_TOKEN is only needed for private datasets; public datasets sync anonymously.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { listFiles, downloadFile } from '@huggingface/hub';
import {
  requireDatasetRepo, stripResult, generateSlug,
} from './_hub.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const MACHINES_DIR = path.join(ROOT, 'data', 'machines');

async function main() {
  const datasetRepo = requireDatasetRepo();
  const token = process.env.HF_TOKEN || null;
  const credentials = token ? { accessToken: token } : undefined;

  console.log(`Listing runs/ in dataset ${datasetRepo}…`);
  const runFiles = [];
  for await (const file of listFiles({
    repo: { type: 'dataset', name: datasetRepo },
    path: 'runs',
    recursive: true,
    credentials,
  })) {
    if (file.type === 'file' && file.path.endsWith('.json')) {
      runFiles.push(file);
    }
  }
  console.log(`Found ${runFiles.length} run file${runFiles.length === 1 ? '' : 's'}.`);
  if (runFiles.length === 0) {
    console.log('Nothing to sync.');
    return;
  }

  // Download and parse each file.
  const allRecords = [];
  for (const file of runFiles) {
    const resp = await downloadFile({
      repo: { type: 'dataset', name: datasetRepo },
      path: file.path,
      credentials,
    });
    if (!resp) {
      console.warn(`  missing: ${file.path}`);
      continue;
    }
    const body = await resp.text();
    let records;
    try {
      records = JSON.parse(body);
    } catch (err) {
      console.warn(`  skipping (invalid JSON): ${file.path}`);
      continue;
    }
    if (!Array.isArray(records)) {
      console.warn(`  skipping (not an array): ${file.path}`);
      continue;
    }
    for (const r of records) allRecords.push(stripResult(r));
  }
  console.log(`Downloaded ${allRecords.length} record${allRecords.length === 1 ? '' : 's'}.`);

  // Group by machine slug and write data/machines/{slug}.json.
  const bySlug = new Map();
  for (const r of allRecords) {
    const slug = generateSlug(r.machine);
    if (!bySlug.has(slug)) bySlug.set(slug, []);
    bySlug.get(slug).push(r);
  }

  fs.mkdirSync(MACHINES_DIR, { recursive: true });

  const priorFiles = new Set(
    fs.existsSync(MACHINES_DIR)
      ? fs.readdirSync(MACHINES_DIR).filter(f => f.endsWith('.json'))
      : [],
  );

  for (const [slug, records] of bySlug) {
    // Summary fields — take from the first record's machine block + earliest config.
    const first = records[0];
    const submission = {
      machine: {
        slug,
        platform: first.machine?.platform || 'unknown',
        arch: first.machine?.arch || 'unknown',
        cpus: first.machine?.cpus || 'unknown',
        totalMemoryGB: first.machine?.totalMemoryGB || 0,
      },
      submittedAt: new Date().toISOString(),
      benchmarkConfig: {
        nCtx: first.nCtx,
        nPredict: first.nPredict,
        nGpuLayers: first.nGpuLayers,
      },
      results: records,
    };
    const outFile = path.join(MACHINES_DIR, `${slug}.json`);
    fs.writeFileSync(outFile, JSON.stringify(submission, null, 2) + '\n');
    priorFiles.delete(`${slug}.json`);
    console.log(`  wrote data/machines/${slug}.json (${records.length} records)`);
  }

  if (priorFiles.size > 0) {
    console.log('\nLegacy files in data/machines/ not covered by the dataset (left untouched):');
    for (const f of priorFiles) console.log(`  ${f}`);
  }
}

main().catch(err => {
  console.error(`sync failed: ${err.message}`);
  if (err.stack) console.error(err.stack);
  process.exit(1);
});
