// Results aggregation: reads results.json, produces summary + CSV.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const RESULTS_DIR = path.join(__dirname, 'results');

function main() {
  const resultsFile = path.join(RESULTS_DIR, 'results.json');
  if (!fs.existsSync(resultsFile)) {
    console.error('No results.json found. Run benchmarks first: node runner.js');
    process.exit(1);
  }

  const results = JSON.parse(fs.readFileSync(resultsFile, 'utf-8'));
  console.log(`Loaded ${results.length} benchmark results\n`);

  // Generate CSV
  const csvHeader = [
    'browser', 'variant', 'size_mb', 'status', 'webgpu_available',
    'n_gpu_layers', 'prefill_tok_s', 'decode_tok_s',
    'n_p_eval', 't_p_eval_ms', 'n_eval', 't_eval_ms',
    'wall_time_s', 'error',
  ].join(',');

  const csvRows = results.map(r => {
    const m = r.metrics || {};
    return [
      r.browser,
      r.variant,
      r.sizeMB,
      r.status,
      r.webgpuAvailable,
      r.nGpuLayers,
      m.prefill_tok_s || '',
      m.decode_tok_s || '',
      m.n_p_eval || '',
      m.t_p_eval_ms ? m.t_p_eval_ms.toFixed(2) : '',
      m.n_eval || '',
      m.t_eval_ms ? m.t_eval_ms.toFixed(2) : '',
      (r.wallTimeMs / 1000).toFixed(1),
      (r.error || '').replace(/,/g, ';').replace(/\n/g, ' '),
    ].join(',');
  });

  const csv = [csvHeader, ...csvRows].join('\n');
  const csvFile = path.join(RESULTS_DIR, 'results.csv');
  fs.writeFileSync(csvFile, csv);
  console.log(`CSV written to ${csvFile}`);

  // Generate summary grouped by browser
  const summary = {};
  for (const r of results) {
    if (!summary[r.browser]) summary[r.browser] = { passed: [], failed: [] };
    if (r.status === 'done') {
      summary[r.browser].passed.push({
        variant: r.variant,
        prefill_tok_s: r.metrics?.prefill_tok_s,
        decode_tok_s: r.metrics?.decode_tok_s,
        wall_time_s: (r.wallTimeMs / 1000).toFixed(1),
      });
    } else {
      summary[r.browser].failed.push({
        variant: r.variant,
        error: r.error,
      });
    }
  }

  const summaryFile = path.join(RESULTS_DIR, 'summary.json');
  fs.writeFileSync(summaryFile, JSON.stringify({
    timestamp: results[0]?.timestamp,
    machine: results[0]?.machine,
    summary,
  }, null, 2));
  console.log(`Summary written to ${summaryFile}\n`);

  // Console table
  for (const browser of Object.keys(summary)) {
    console.log(`=== ${browser} ===`);

    if (summary[browser].passed.length > 0) {
      console.log('  Passed:');
      console.log('  ' + 'Variant'.padEnd(16) + 'Prefill (tok/s)'.padEnd(18) + 'Decode (tok/s)'.padEnd(18) + 'Wall (s)');
      console.log('  ' + '-'.repeat(66));
      for (const r of summary[browser].passed) {
        console.log(
          '  ' +
          r.variant.padEnd(16) +
          String(r.prefill_tok_s || 'N/A').padEnd(18) +
          String(r.decode_tok_s || 'N/A').padEnd(18) +
          r.wall_time_s
        );
      }
    }

    if (summary[browser].failed.length > 0) {
      console.log('  Failed:');
      for (const r of summary[browser].failed) {
        console.log(`    ${r.variant}: ${r.error}`);
      }
    }
    console.log('');
  }
}

main();
