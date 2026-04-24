// One-shot seeding: take every existing data/machines/*.json submission and
// upload its `results` array to the leaderboard dataset. One commit per
// (machine slug, browser) pair, tagged "bootstrap" so these can be
// distinguished from future interactive submissions.
//
// Usage:
//   HF_TOKEN=… HF_DATASET_REPO=owner/webgpu-bench-leaderboard \
//     node scripts/bootstrap-dataset.mjs

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
const MACHINES_DIR = path.join(ROOT, 'data', 'machines');

async function main() {
  const datasetRepo = requireDatasetRepo();
  const token = requireToken();

  if (!fs.existsSync(MACHINES_DIR)) {
    console.error(`No ${path.relative(ROOT, MACHINES_DIR)} directory — nothing to bootstrap.`);
    process.exit(1);
  }
  const files = fs.readdirSync(MACHINES_DIR).filter(f => f.endsWith('.json'));
  if (files.length === 0) {
    console.error('No machine files found to bootstrap.');
    process.exit(1);
  }

  const me = await whoAmI({ credentials: { accessToken: token } }).catch(() => null);
  if (me?.name) console.log(`Authenticated as: ${me.name}`);

  const epochBase = Date.now();
  let totalUploads = 0;

  for (const fname of files) {
    const raw = JSON.parse(fs.readFileSync(path.join(MACHINES_DIR, fname), 'utf-8'));
    const records = (raw.results || []).map(stripResult);
    if (records.length === 0) {
      console.log(`  skip ${fname}: no records`);
      continue;
    }
    const submittedAt = raw.submittedAt ? new Date(raw.submittedAt) : new Date();
    const date = datestamp(submittedAt);

    const groups = groupBy(records, r => {
      const slug = raw.machine?.slug || generateSlug(r.machine) || 'unknown';
      const browser = r.browser || 'unknown-browser';
      return `${slug}::${browser}`;
    });

    for (const [key, groupRecords] of groups) {
      const [slug, browser] = key.split('::');
      const body = JSON.stringify(groupRecords, null, 2);
      const blob = new Blob([body], { type: 'application/json' });
      const filePath = runPath({
        date, slug, browser, tag: 'bootstrap', epoch: epochBase + totalUploads,
      });
      const commitTitle = `bootstrap: ${slug} / ${browser} / ${groupRecords.length} variants`;
      console.log(`→ ${filePath} (${groupRecords.length} records)`);
      const res = await uploadFile({
        repo: { type: 'dataset', name: datasetRepo },
        credentials: { accessToken: token },
        file: { path: filePath, content: blob },
        commitTitle,
      });
      totalUploads++;
      if (res?.commit?.oid) {
        console.log(`  commit: ${res.commit.oid.slice(0, 10)}`);
      }
    }
  }

  const url = `https://huggingface.co/datasets/${datasetRepo}`;
  console.log(`\nBootstrap pushed ${totalUploads} file${totalUploads === 1 ? '' : 's'} to ${url}`);
}

main().catch(err => {
  console.error(`bootstrap failed: ${err.message}`);
  if (err.stack) console.error(err.stack);
  process.exit(1);
});
