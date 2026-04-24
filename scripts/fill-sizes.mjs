// scripts/fill-sizes.mjs
// HEAD each Hugging Face GGUF URL in models.json and fill `sizeMB` based on
// the resolved Content-Length. Skips variants that already have a non-zero
// sizeMB. Use `--force` to refresh all sizes regardless.
//
// Usage:
//   node scripts/fill-sizes.mjs [--force] [--concurrency=8]

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const MODELS_FILE = path.join(ROOT, 'models.json');

const args = new Set(process.argv.slice(2));
const force = args.has('--force');
const concurrencyArg = [...args].find(a => a.startsWith('--concurrency='));
const concurrency = concurrencyArg ? parseInt(concurrencyArg.split('=')[1], 10) : 8;

async function headSize(repo, filename) {
  const url = `https://huggingface.co/${repo}/resolve/main/${filename}`;
  const resp = await fetch(url, { method: 'HEAD', redirect: 'follow' });
  if (!resp.ok) throw new Error(`HEAD ${url} → ${resp.status}`);
  const cl = resp.headers.get('content-length');
  if (!cl) throw new Error(`no content-length for ${url}`);
  return parseInt(cl, 10);
}

async function pool(items, limit, worker) {
  const queue = items.slice();
  const results = [];
  const runners = Array.from({ length: limit }, async () => {
    while (queue.length) {
      const item = queue.shift();
      try {
        const out = await worker(item);
        results.push({ item, out });
      } catch (err) {
        results.push({ item, error: err });
      }
    }
  });
  await Promise.all(runners);
  return results;
}

const data = JSON.parse(fs.readFileSync(MODELS_FILE, 'utf-8'));

const jobs = [];
for (const m of data.models) {
  for (const v of m.variants) {
    if (!force && typeof v.sizeMB === 'number' && v.sizeMB > 0) continue;
    jobs.push({ model: m, variant: v });
  }
}

if (jobs.length === 0) {
  console.log('Nothing to fill — every variant already has a sizeMB.');
  process.exit(0);
}

console.log(`Filling ${jobs.length} size${jobs.length === 1 ? '' : 's'} (concurrency=${concurrency})…`);

const results = await pool(jobs, concurrency, async ({ model, variant }) => {
  const bytes = await headSize(model.repo, variant.filename);
  const sizeMB = parseFloat((bytes / 1048576).toFixed(2));
  variant.sizeMB = sizeMB;
  return sizeMB;
});

let ok = 0;
let failed = 0;
for (const r of results) {
  if (r.error) {
    failed++;
    console.error(`  FAILED: ${r.item.model.repo}/${r.item.variant.filename}: ${r.error.message}`);
  } else {
    ok++;
  }
}

fs.writeFileSync(MODELS_FILE, JSON.stringify(data, null, 2) + '\n');
console.log(`\nFilled ${ok} sizes, ${failed} failures. Wrote ${path.relative(ROOT, MODELS_FILE)}.`);
if (failed) process.exit(1);
