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

  const hasConsistency = results.some(r => r.consistency !== undefined);

  // Generate CSV
  const csvHeader = [
    'browser', 'model', 'variant', 'size_mb', 'status', 'webgpu_available',
    'n_gpu_layers', 'prefill_tok_s', 'decode_tok_s',
    'n_p_eval', 't_p_eval_ms', 'n_eval', 't_eval_ms',
    'wall_time_s', 'error',
    ...(hasConsistency ? ['consistency_agreement_rate', 'consistency_n_agree', 'consistency_n_tokens', 'consistency_first_disagreement'] : []),
  ].join(',');

  const csvRows = results.map(r => {
    const m = r.metrics || {};
    const c = r.consistency;
    return [
      r.browser,
      r.model || '',
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
      ...(hasConsistency ? [
        c == null ? '' : c.agreement_rate,
        c == null ? '' : c.n_agree,
        c == null ? '' : c.n_tokens,
        c == null ? '' : c.first_disagreement,
      ] : []),
    ].join(',');
  });

  const csv = [csvHeader, ...csvRows].join('\n');
  const csvFile = path.join(RESULTS_DIR, 'results.csv');
  fs.writeFileSync(csvFile, csv);
  console.log(`CSV written to ${csvFile}`);

  // Generate summary grouped by browser, then model
  const summary = {};
  for (const r of results) {
    const model = r.model || 'unknown';
    if (!summary[r.browser]) summary[r.browser] = {};
    if (!summary[r.browser][model]) summary[r.browser][model] = { passed: [], failed: [] };
    if (r.status === 'done') {
      summary[r.browser][model].passed.push({
        variant: r.variant,
        prefill_tok_s: r.metrics?.prefill_tok_s,
        decode_tok_s: r.metrics?.decode_tok_s,
        wall_time_s: (r.wallTimeMs / 1000).toFixed(1),
        consistency: r.consistency ?? null,
      });
    } else {
      summary[r.browser][model].failed.push({
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

    for (const model of Object.keys(summary[browser])) {
      const modelData = summary[browser][model];
      console.log(`  --- ${model} ---`);

      if (modelData.passed.length > 0) {
        console.log('    Passed:');
        if (hasConsistency) {
          console.log('    ' + 'Variant'.padEnd(16) + 'Prefill (tok/s)'.padEnd(18) + 'Decode (tok/s)'.padEnd(18) + 'Wall (s)'.padEnd(12) + 'CPU match');
          console.log('    ' + '-'.repeat(80));
          for (const r of modelData.passed) {
            const c = r.consistency;
            const matchLabel = c == null ? 'no baseline'
              : c.agreement_rate === 1.0 ? '100% top-1'
              : `${(c.agreement_rate * 100).toFixed(1)}% top-1 (diverge@${c.first_disagreement})`;
            console.log(
              '    ' +
              r.variant.padEnd(16) +
              String(r.prefill_tok_s || 'N/A').padEnd(18) +
              String(r.decode_tok_s || 'N/A').padEnd(18) +
              r.wall_time_s.padEnd(12) +
              matchLabel
            );
          }
        } else {
          console.log('    ' + 'Variant'.padEnd(16) + 'Prefill (tok/s)'.padEnd(18) + 'Decode (tok/s)'.padEnd(18) + 'Wall (s)');
          console.log('    ' + '-'.repeat(66));
          for (const r of modelData.passed) {
            console.log(
              '    ' +
              r.variant.padEnd(16) +
              String(r.prefill_tok_s || 'N/A').padEnd(18) +
              String(r.decode_tok_s || 'N/A').padEnd(18) +
              r.wall_time_s
            );
          }
        }
      }

      if (modelData.failed.length > 0) {
        console.log('    Failed:');
        for (const r of modelData.failed) {
          console.log(`      ${r.variant}: ${r.error}`);
        }
      }
    }
    console.log('');
  }

  if (hasConsistency) {
    console.log('=== Consistency Issues ===');
    const issues = results.filter(r => r.consistency && r.consistency.agreement_rate < 1.0);
    if (issues.length === 0) {
      console.log('  All variants agree 100% with CPU baseline on top-1 token.\n');
    } else {
      for (const r of issues) {
        const c = r.consistency;
        console.log(`  ${r.browser} / ${r.variant}: ${(c.agreement_rate * 100).toFixed(1)}% (${c.n_agree}/${c.n_tokens} tokens agree, first diverge @ token ${c.first_disagreement})`);
      }
      console.log('');
    }
  }
}

main();
