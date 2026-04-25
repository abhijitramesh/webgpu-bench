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
    // Pair the surfaced commit with its describe (when present) so the
    // machine card can show the readable "b8708-12-gd12cc3d1c" label.
    const firstCommit = llamaCommits[0] || null;
    const llamaDescribe = firstCommit
      ? data.results.find(r => r.llamaCppCommit === firstCommit && r.llamaCppDescribe)?.llamaCppDescribe || null
      : null;

    // Aggregate every submitter who's contributed results on this machine,
    // ranked by submission count (descending) so the most-active contributor
    // surfaces first on the dashboard. Ties on count fall back to most-recent
    // submission. Stays per-machine because each card is per-machine.
    const submitterCounts = new Map();
    const submitterLatest = new Map();
    for (const r of data.results) {
      const sb = r.submittedBy;
      if (!sb?.name) continue;
      const key = sb.hubId || sb.name;
      submitterCounts.set(key, (submitterCounts.get(key) || 0) + 1);
      if (!submitterLatest.has(key) || r.timestamp > submitterLatest.get(key).timestamp) {
        submitterLatest.set(key, { profile: sb, timestamp: r.timestamp || '' });
      }
    }
    const submitters = [...submitterCounts.entries()]
      .map(([key, count]) => ({
        ...submitterLatest.get(key).profile,
        count,
        latestAt: submitterLatest.get(key).timestamp,
      }))
      .sort((a, b) => b.count - a.count || (b.latestAt || '').localeCompare(a.latestAt || ''));

    machines.push({
      slug,
      cpus: data.machine.cpus,
      platform: data.machine.platform,
      arch: data.machine.arch,
      totalMemoryGB: data.machine.totalMemoryGB,
      submittedAt: data.submittedAt,
      resultCount: data.results.length,
      passCount: data.results.filter(r => r.status === 'done').length,
      llamaCppCommit: firstCommit,
      llamaCppDescribe: llamaDescribe,
      submitters,
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
        // CPU baseline: stamped by the worker as a sanity check for the GPU
        // numbers. Surface alongside the GPU rates so reviewers can spot a
        // regression where the GPU path silently fell back to CPU.
        cpu_baseline_prefill_tok_s: r.cpu_baseline?.prefill_tok_s ?? null,
        cpu_baseline_decode_tok_s: r.cpu_baseline?.decode_tok_s ?? null,
        llamaCppCommit: r.llamaCppCommit ?? null,
        // Human-readable git describe of the llama.cpp build (e.g.
        // "b8708-12-gd12cc3d1c"). Falls back to short commit when missing.
        llamaCppDescribe: r.llamaCppDescribe ?? null,
        dawnTag: r.dawnTag ?? null,
        submittedBy: r.submittedBy ?? null,
        // Iterations — primary tiebreak when multiple submissions cover the
        // same (machine, browser, model, variant) cell. The dashboard
        // canonicalizes to the row with most iterations (then latest).
        iterations: r.metrics?.iterations ?? null,
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

  // Mirror models.json into site/ so the Run page can fetch it via
  // `./models.json` on the HF Space (flattened root) and any other static host.
  if (fs.existsSync(MODELS_SRC)) {
    fs.copyFileSync(MODELS_SRC, MODELS_DST);
    console.log(`  Copied ${path.relative(ROOT, MODELS_SRC)} → ${path.relative(ROOT, MODELS_DST)}`);
  }
}

main();
