// Merge all machine result files into site/data/combined.json for the dashboard.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const MACHINES_DIR = path.join(ROOT, 'data', 'machines');
const OUT_FILE = path.join(ROOT, 'site', 'data', 'combined.json');
const MODELS_SRC = path.join(ROOT, 'models.json');
const MODELS_DST = path.join(ROOT, 'site', 'models.json');

function main() {
  if (!fs.existsSync(MACHINES_DIR)) {
    console.error('No data/machines/ directory found. Run `npm run submit` first.');
    process.exit(1);
  }

  const files = fs.readdirSync(MACHINES_DIR).filter(f => f.endsWith('.json'));
  if (files.length === 0) {
    console.error('No machine files found in data/machines/.');
    process.exit(1);
  }

  const machines = [];
  const allResults = [];
  const modelsSet = new Set();
  const browsersSet = new Set();

  for (const file of files) {
    const filePath = path.join(MACHINES_DIR, file);
    const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    const slug = data.machine.slug;

    // Extract llama.cpp commit(s) used in this machine's results
    const llamaCommits = [...new Set(data.results.map(r => r.llamaCppCommit).filter(Boolean))];

    machines.push({
      slug,
      cpus: data.machine.cpus,
      platform: data.machine.platform,
      arch: data.machine.arch,
      totalMemoryGB: data.machine.totalMemoryGB,
      submittedAt: data.submittedAt,
      resultCount: data.results.length,
      passCount: data.results.filter(r => r.status === 'done').length,
      llamaCppCommit: llamaCommits[0] || null,
    });

    for (const r of data.results) {
      // Flatten metrics into the result object
      const flat = {
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
        n_p_eval: r.metrics?.n_p_eval ?? null,
        t_p_eval_ms: r.metrics?.t_p_eval_ms ?? null,
        n_eval: r.metrics?.n_eval ?? null,
        t_eval_ms: r.metrics?.t_eval_ms ?? null,
        consistency_rate: r.consistency?.agreement_rate ?? null,
        consistency_first_disagree: r.consistency?.first_disagreement ?? null,
        llamaCppCommit: r.llamaCppCommit ?? null,
      };

      allResults.push(flat);

      if (r.model) modelsSet.add(r.model);
      if (r.browser) browsersSet.add(r.browser);
    }

    console.log(`  ${file}: ${data.results.length} results (${slug})`);
  }

  const combined = {
    meta: {
      machines,
      models: [...modelsSet].sort(),
      browsers: [...browsersSet].sort(),
      generatedAt: new Date().toISOString(),
    },
    results: allResults,
  };

  fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
  fs.writeFileSync(OUT_FILE, JSON.stringify(combined, null, 2));

  console.log(`\nGenerated ${path.relative(ROOT, OUT_FILE)}`);
  console.log(`  Machines: ${machines.length}`);
  console.log(`  Results:  ${allResults.length}`);
  console.log(`  Models:   ${[...modelsSet].join(', ')}`);
  console.log(`  Browsers: ${[...browsersSet].join(', ')}`);

  // Mirror models.json into site/ so the Run tab can fetch it on static hosts
  // (GH Pages, HF Space) via `./models.json`.
  if (fs.existsSync(MODELS_SRC)) {
    fs.copyFileSync(MODELS_SRC, MODELS_DST);
    console.log(`  Copied ${path.relative(ROOT, MODELS_SRC)} → ${path.relative(ROOT, MODELS_DST)}`);
  }
}

main();
