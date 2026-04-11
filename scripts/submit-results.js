// Prepare benchmark results for PR submission.
// Reads results/results.json, strips heavy fields, writes to data/machines/{slug}.json.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const RESULTS_FILE = path.join(ROOT, 'results', 'results.json');
const MACHINES_DIR = path.join(ROOT, 'data', 'machines');

function slugify(str) {
  return str
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function generateSlug(machine) {
  const cpu = slugify(machine.cpus || 'unknown');
  const ram = machine.totalMemoryGB || 0;
  const platform = machine.platform || 'unknown';
  return `${cpu}-${ram}gb-${platform}`;
}

function stripResult(r) {
  const stripped = { ...r };

  // Strip heavy fields from metrics
  if (stripped.metrics) {
    stripped.metrics = { ...stripped.metrics };
    delete stripped.metrics.token_ids;
    delete stripped.metrics.output;
  }

  // Strip output text (already truncated but not needed for dashboard)
  delete stripped.output;

  // Strip consistency matches array if present
  if (stripped.consistency) {
    stripped.consistency = { ...stripped.consistency };
    delete stripped.consistency.matches;
  }

  return stripped;
}

function main() {
  if (!fs.existsSync(RESULTS_FILE)) {
    console.error('No results/results.json found. Run benchmarks first: node runner.js');
    process.exit(1);
  }

  const results = JSON.parse(fs.readFileSync(RESULTS_FILE, 'utf-8'));
  if (results.length === 0) {
    console.error('results.json is empty.');
    process.exit(1);
  }

  // Extract machine info from first result
  const machine = results[0].machine;
  if (!machine) {
    console.error('No machine info found in results.');
    process.exit(1);
  }

  const slug = generateSlug(machine);

  // Extract benchmark config from first result
  const benchmarkConfig = {
    nCtx: results[0].nCtx,
    nPredict: results[0].nPredict,
    nGpuLayers: results[0].nGpuLayers,
  };

  // Strip heavy fields from all results
  const strippedResults = results.map(stripResult);

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
    results: strippedResults,
  };

  // Write machine file
  fs.mkdirSync(MACHINES_DIR, { recursive: true });
  const outFile = path.join(MACHINES_DIR, `${slug}.json`);
  fs.writeFileSync(outFile, JSON.stringify(submission, null, 2));

  console.log(`Created ${path.relative(ROOT, outFile)}`);
  console.log(`  Machine: ${machine.cpus} (${machine.platform}/${machine.arch}, ${machine.totalMemoryGB}GB)`);
  console.log(`  Slug:    ${slug}`);
  console.log(`  Results: ${results.length} benchmarks (${results.filter(r => r.status === 'done').length} passed)`);
  console.log('');
  console.log('Next steps:');
  console.log(`  git add data/machines/${slug}.json`);
  console.log('  git commit -m "Add benchmark results for ' + machine.cpus + '"');
  console.log('  # Then open a PR');
}

main();
