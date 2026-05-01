import { formatTokS, formatMs, categorizeError, groupBy, quantSortKey } from './utils.js';
import { expandCpuRows } from './data.js';

let lastResults = [];
let sortState = { key: null, dir: 'asc' };

const NUM_KEYS = new Set([
  'sizeMB',
  'decode_tok_s', 'prefill_tok_s',
  'decode_tok_s_d0', 'decode_tok_s_dN',
  'prefill_tok_s_d0', 'prefill_tok_s_dN',
  'cpu_baseline_decode_tok_s', 'cpu_baseline_prefill_tok_s',
  'n_eval', 't_eval_ms',
  'n_p_eval', 't_p_eval_ms', 'wallTimeMs', 'consistency_rate',
]);

function sortResults(results, key, dir) {
  const isNum = NUM_KEYS.has(key);
  return [...results].sort((a, b) => {
    let va = a[key], vb = b[key];
    // Submitter is an object — collapse to its name for comparison and let
    // the null-handling below treat unattributed rows as the lowest.
    if (key === 'submittedBy') {
      va = va?.name || null;
      vb = vb?.name || null;
    }
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;

    let cmp;
    if (isNum) {
      cmp = Number(va) - Number(vb);
    } else if (key === 'webgpuAvailable') {
      cmp = (va === vb) ? 0 : va ? -1 : 1;
    } else {
      cmp = String(va).localeCompare(String(vb));
    }
    return dir === 'desc' ? -cmp : cmp;
  });
}

function handleSort(key) {
  if (sortState.key === key) {
    sortState.dir = sortState.dir === 'asc' ? 'desc' : 'asc';
  } else {
    sortState.key = key;
    // Default to descending for performance metrics
    sortState.dir = NUM_KEYS.has(key) ? 'desc' : 'asc';
  }
  renderResultsTable(lastResults);
}

export function renderResultsTable(results) {
  lastResults = results;
  const container = document.getElementById('results-table');
  if (!container) return;

  if (results.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <p>No results match the current filters.</p>
        <p class="empty-state-sub">Try resetting filters above, or <a href="run.html">run the benchmark</a> on your own machine to contribute data.</p>
      </div>`;
    return;
  }

  const sorted = sortState.key ? sortResults(results, sortState.key, sortState.dir) : results;

  /* priority: 1 = always show; 2 = hide below 640px; 3 = hide below 900px */
  const cols = [
    { key: 'machineSlug', label: 'Machine', priority: 1 },
    { key: 'model', label: 'Model', priority: 1 },
    { key: 'variant', label: 'Quant', priority: 1 },
    { key: 'sizeMB', label: 'Size (MB)', priority: 3 },
    { key: 'browser', label: 'Browser', priority: 2 },
    { key: 'submittedBy', label: 'Submitter', priority: 2 },
    { key: 'status', label: 'Status', priority: 1 },
    { key: 'buildType', label: 'Build', priority: 3 },
    { key: 'webgpuAvailable', label: 'WebGPU', priority: 3 },
    // tg / pp split into cold-cache (d=0) and depth-loaded (d=N) columns
    // so Run Study's depth-pair shows as side-by-side numbers instead of
    // overwriting one with the other. Pre-study and plain-Run records
    // populate only the side they actually measured; the other reads `—`.
    { key: 'decode_tok_s_d0', label: 'tg @ d0', priority: 1 },
    { key: 'decode_tok_s_dN', label: 'tg @ dN', priority: 1 },
    { key: 'prefill_tok_s_d0', label: 'pp @ d0', priority: 3 },
    { key: 'prefill_tok_s_dN', label: 'pp @ dN', priority: 3 },
    { key: 'cpu_baseline_decode_tok_s', label: 'CPU tg tok/s', priority: 2 },
    { key: 'cpu_baseline_prefill_tok_s', label: 'CPU pp tok/s', priority: 3 },
    { key: 'n_eval', label: 'n_eval', priority: 3 },
    { key: 't_eval_ms', label: 't_eval (ms)', priority: 3 },
    { key: 'n_p_eval', label: 'n_p_eval', priority: 3 },
    { key: 't_p_eval_ms', label: 't_p_eval (ms)', priority: 3 },
    { key: 'wallTimeMs', label: 'Wall (s)', priority: 3 },
    { key: 'consistency_rate', label: 'CPU Match', priority: 2 },
    { key: 'llamaCppCommit', label: 'llama.cpp', priority: 3 },
    { key: 'error', label: 'Error', priority: 2 },
  ];

  let html = '<table class="results-table"><thead><tr>';
  cols.forEach((col, i) => {
    const isActive = sortState.key === col.key;
    const ariaSort = isActive ? (sortState.dir === 'asc' ? 'ascending' : 'descending') : 'none';
    const arrowChar = isActive ? (sortState.dir === 'asc' ? '\u2191' : '\u2193') : '\u2195';
    const pin = i === 0 ? ' col-pin col-pin-1' : (i === 1 ? ' col-pin col-pin-2' : '');
    const prio = col.priority >= 3 ? ' col-p3' : (col.priority === 2 ? ' col-p2' : '');
    const cls = `sortable${isActive ? ' sorted' : ''}${pin}${prio}`;
    html += `<th data-key="${col.key}" class="${cls}" aria-sort="${ariaSort}" scope="col" tabindex="0"><span class="th-label">${col.label}</span><span class="th-sort-indicator" aria-hidden="true">${arrowChar}</span></th>`;
  });
  html += '</tr></thead><tbody>';

  for (const r of sorted) {
    const rowClass = r.status === 'done' ? 'row-pass' : 'row-fail';
    html += `<tr class="${rowClass}">`;
    cols.forEach((col, i) => {
      const pin = i === 0 ? 'col-pin col-pin-1' : (i === 1 ? 'col-pin col-pin-2' : '');
      const prio = col.priority >= 3 ? 'col-p3' : (col.priority === 2 ? 'col-p2' : '');
      const parts = [pin, prio].filter(Boolean);
      const cls = parts.length ? ` class="${parts.join(' ')}"` : '';
      html += `<td${cls}>`;
      switch (col.key) {
        case 'status':
          html += r.status === 'done'
            ? '<span class="badge badge--pass">PASS</span>'
            : '<span class="badge badge--fail">FAIL</span>';
          break;
        case 'webgpuAvailable':
          html += r.webgpuAvailable
            ? '<span class="badge badge--yes">Yes</span>'
            : '<span class="badge badge--no">No</span>';
          break;
        case 'decode_tok_s':
        case 'prefill_tok_s':
        case 'decode_tok_s_d0':
        case 'decode_tok_s_dN':
        case 'prefill_tok_s_d0':
        case 'prefill_tok_s_dN':
        case 'cpu_baseline_decode_tok_s':
        case 'cpu_baseline_prefill_tok_s': {
          // llama-bench style "avg \u00b1 stddev" with the pp{N} / tg{N} test
          // label as a tooltip when the new schema is present. Older records
          // without stddev fall back to the bare avg from formatTokS.
          // Depth-suffixed keys read from the matching `_d0` / `_dN`
          // stddev + test_name fields produced by mergeDepthPairs.
          let stddev = null;
          let testName = null;
          switch (col.key) {
            case 'decode_tok_s':       stddev = r.decode_stddev_ts;     testName = r.tg_test_name;       break;
            case 'prefill_tok_s':      stddev = r.prefill_stddev_ts;    testName = r.pp_test_name;       break;
            case 'decode_tok_s_d0':    stddev = r.decode_stddev_ts_d0;  testName = r.tg_test_name_d0;    break;
            case 'decode_tok_s_dN':    stddev = r.decode_stddev_ts_dN;  testName = r.tg_test_name_dN;    break;
            case 'prefill_tok_s_d0':   stddev = r.prefill_stddev_ts_d0; testName = r.pp_test_name_d0;    break;
            case 'prefill_tok_s_dN':   stddev = r.prefill_stddev_ts_dN; testName = r.pp_test_name_dN;    break;
          }
          const avg = r[col.key];
          let cell;
          if (avg != null && stddev != null) {
            cell = `${formatTokS(avg)} \u00b1 ${formatTokS(stddev)}`;
          } else {
            cell = formatTokS(avg);
          }
          const titleAttr = testName ? ` title="${escapeHtml(testName)}"` : '';
          html += `<span class="mono"${titleAttr}>${cell}</span>`;
          break;
        }
        case 't_eval_ms':
        case 't_p_eval_ms':
          html += `<span class="mono">${formatMs(r[col.key])}</span>`;
          break;
        case 'wallTimeMs':
          html += `<span class="mono">${r.wallTimeMs != null ? (r.wallTimeMs / 1000).toFixed(1) : '\u2014'}</span>`;
          break;
        case 'consistency_rate':
          if (r.consistency_rate != null) {
            const pct = (r.consistency_rate * 100).toFixed(1);
            const cls = r.consistency_rate >= 0.95 ? 'text-success' : r.consistency_rate >= 0.90 ? '' : 'text-error';
            const diverge = r.consistency_first_disagree >= 0 ? ` (diverge@${r.consistency_first_disagree})` : '';
            html += `<span class="mono ${cls}">${pct}%${diverge}</span>`;
          } else {
            html += '<span class="text-muted">\u2014</span>';
          }
          break;
        case 'submittedBy':
          html += renderSubmitterCell(r.submittedBy);
          break;
        case 'llamaCppCommit':
          if (r.llamaCppCommit) {
            // Prefer the human-readable git describe when present (e.g.
            // "b8708-12-gd12cc3d1c"); fall back to a short commit hash.
            const label = r.llamaCppDescribe || r.llamaCppCommit.slice(0, 10);
            html += `<a class="mono" href="https://github.com/ggml-org/llama.cpp/commit/${r.llamaCppCommit}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`;
          } else {
            html += '<span class="text-muted">\u2014</span>';
          }
          break;
        case 'error':
          if (r.error) {
            const cat = categorizeError(r.error);
            const short = r.error.length > 60 ? r.error.slice(0, 60) + '\u2026' : r.error;
            html += `<span class="error-cell" title="${escapeHtml(r.error)}"><span class="error-cat">${cat}</span>${escapeHtml(short)}</span>`;
          } else {
            html += '<span class="text-muted">\u2014</span>';
          }
          break;
        case 'sizeMB':
        case 'n_eval':
        case 'n_p_eval':
          html += `<span class="mono">${r[col.key] != null ? r[col.key] : '\u2014'}</span>`;
          break;
        default:
          html += escapeHtml(String(r[col.key] ?? '\u2014'));
      }
      html += '</td>';
    });
    html += '</tr>';
  }

  html += '</tbody></table>';
  container.innerHTML = html;

  // Wire sort click + keyboard handlers
  container.querySelectorAll('th[data-key]').forEach(th => {
    th.addEventListener('click', () => handleSort(th.dataset.key));
    th.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleSort(th.dataset.key);
      }
    });
  });
}

export function renderErrorTable(results) {
  const container = document.getElementById('error-table');
  if (!container) return;

  const errors = results.filter(r => r.status !== 'done' && r.error);
  if (errors.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <p>No errors in the current filter.</p>
        <p class="empty-state-sub">Either every benchmark passed, or no results are in scope — try widening the filter.</p>
      </div>`;
    return;
  }

  const grouped = groupBy(errors, r => categorizeError(r.error));

  let html = '<div class="table-card"><table class="data-table"><thead><tr><th>Category</th><th>Count</th><th>Variants</th><th>Browsers</th></tr></thead><tbody>';
  for (const [cat, items] of Object.entries(grouped).sort((a, b) => b[1].length - a[1].length)) {
    const variants = [...new Set(items.map(i => i.variant))].join(', ');
    const browsers = [...new Set(items.map(i => i.browser))].join(', ');
    html += `<tr><td><span class="error-cat">${cat}</span></td><td><span class="mono">${items.length}</span></td><td>${variants}</td><td>${browsers}</td></tr>`;
  }
  html += '</tbody></table></div>';
  container.innerHTML = html;
}

export function renderMachineInfo(machines) {
  const container = document.getElementById('machine-info');
  if (!container) return;

  const addYourMachineCard = `
    <a class="machine-card machine-card-add" href="run.html">
      <div class="machine-card-header">
        <svg class="machine-card-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
        <h3>Add your machine</h3>
      </div>
      <p class="machine-card-add-blurb">Run benchmarks directly in your browser. Results post to the leaderboard.</p>
      <code class="machine-card-add-cmd">npm run bench:quick</code>
      <span class="machine-card-add-cta">
        Open Run page
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
      </span>
    </a>`;

  if (machines.length === 0) {
    container.innerHTML = `<div class="machine-grid">${addYourMachineCard}</div>`;
    return;
  }

  let html = '<div class="machine-grid">';
  for (const m of machines) {
    const failCount = m.resultCount - m.passCount;
    html += `
      <div class="machine-card">
        <div class="machine-card-header">
          <svg class="machine-card-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
          <h3>${escapeHtml(m.cpus)}</h3>
        </div>
        <div class="machine-card-specs">
          <div class="spec-row"><span class="spec-label">Platform</span><span class="spec-value">${m.platform}</span></div>
          <div class="spec-row"><span class="spec-label">Arch</span><span class="spec-value">${m.arch}</span></div>
          <div class="spec-row"><span class="spec-label">RAM</span><span class="spec-value">${m.totalMemoryGB} GB</span></div>
          <div class="spec-row"><span class="spec-label">Results</span><span class="spec-value">${m.resultCount}</span></div>
          <div class="spec-row"><span class="spec-label">Passed</span><span class="spec-value text-success">${m.passCount}</span></div>
          <div class="spec-row"><span class="spec-label">Failed</span><span class="spec-value text-error">${failCount}</span></div>
          ${m.llamaCppCommit ? `<div class="spec-row"><span class="spec-label">llama.cpp</span><span class="spec-value"><a href="https://github.com/ggml-org/llama.cpp/commit/${m.llamaCppCommit}" target="_blank" rel="noopener">${escapeHtml(m.llamaCppDescribe || m.llamaCppCommit.slice(0, 10))}</a></span></div>` : ''}
        </div>
      </div>`;
  }
  html += addYourMachineCard;
  html += '</div>';
  container.innerHTML = html;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/* Render a single submitter's avatar + @username link for the Results
   table column. Falls back to an em-dash if attribution is unknown. */
function renderSubmitterCell(sb) {
  if (!sb?.name) return '<span class="text-muted">\u2014</span>';
  const avatar = sb.avatarUrl
    ? `<img class="submitter-avatar" src="${escapeHtml(sb.avatarUrl)}" alt="" width="18" height="18" loading="lazy">`
    : '<span class="submitter-avatar submitter-avatar--placeholder" aria-hidden="true"></span>';
  return `<a class="submitter-link" href="https://huggingface.co/${escapeHtml(sb.name)}" target="_blank" rel="noopener" title="View @${escapeHtml(sb.name)} on Hugging Face">${avatar}<span class="submitter-name">@${escapeHtml(sb.name)}</span></a>`;
}

export function renderCpuGpuTable(results) {
  const container = document.getElementById('cpu-gpu-table');
  if (!container) return;

  // CPU is pinned to d=0 by the runner, so the comparison must read GPU's
  // d=0 number for an apples-to-apples ratio. Plain-Run records that only
  // measured d=N have null `_d0` and silently drop out of the comparison
  // — that's the right call: without a cold-cache GPU sample the speedup
  // ratio would be measuring different workloads.
  const METRICS = [
    { cpuField: 'decode_tok_s',  gpuField: 'decode_tok_s_d0',  label: 'Decode tok/s @ d0' },
    { cpuField: 'prefill_tok_s', gpuField: 'prefill_tok_s_d0', label: 'Prefill tok/s @ d0' },
  ];

  const passed = results.filter(r => r.status === 'done');
  // CPU side aggregates standalone CPU runs (nGpuLayers === 0) plus
  // synthetic rows derived from the cpu_baseline_* fields on browser-flow
  // GPU records. See expandCpuRows() in data.js.
  const cpuResults = expandCpuRows(passed);
  const gpuResults = passed.filter(r => r.nGpuLayers !== 0);

  if (cpuResults.length === 0 || gpuResults.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>Select "All Backends" to see CPU vs GPU comparison.</p></div>';
    return;
  }

  function avg(items, field) {
    const vals = items.map(r => r[field]).filter(v => v != null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  }

  const gpuBrowsers = [...new Set(gpuResults.map(r => r.browser))].sort();

  const cpuByModelVariant = groupBy(cpuResults, r => `${r.model}::${r.variant}`);
  const gpuByModelVariant = groupBy(gpuResults, r => `${r.model}::${r.variant}`);

  const keys = [...new Set([...Object.keys(cpuByModelVariant), ...Object.keys(gpuByModelVariant)])]
    .filter(k => cpuByModelVariant[k] && gpuByModelVariant[k]);

  if (keys.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No matching model+variant pairs between CPU and GPU results.</p></div>';
    return;
  }

  keys.sort((a, b) => {
    const [aModel, aVar] = a.split('::');
    const [bModel, bVar] = b.split('::');
    if (aModel !== bModel) return aModel.localeCompare(bModel);
    return quantSortKey(aVar) - quantSortKey(bVar);
  });

  // Two-row grouped header: row1 = group labels (CPU, Chromium, …), row2 = metric sub-labels
  // CPU gets colspan = METRICS.length, each GPU browser gets colspan = METRICS.length * 2 (value + speedup per metric)
  const gpuColspan = METRICS.length * 2;
  // CPU side reads cpuField; GPU side reads gpuField (_d0 for apples-to-
  // apples). Both labels match the metric's display label.
  let html = '<div class="table-card"><div class="results-wrapper"><table class="results-table"><thead>';

  // Row 1: group headers
  html += '<tr>';
  html += '<th rowspan="2" class="th-group-border">Model</th><th rowspan="2" class="th-group-border">Quant</th>';
  html += `<th colspan="${METRICS.length}" class="th-group th-group-border">CPU</th>`;
  for (const b of gpuBrowsers) {
    html += `<th colspan="${gpuColspan}" class="th-group th-group-border">${escapeHtml(b.charAt(0).toUpperCase() + b.slice(1))}</th>`;
  }
  html += '</tr>';

  // Row 2: metric sub-headers
  html += '<tr>';
  for (const m of METRICS) {
    html += `<th class="th-sub">${m.label}</th>`;
  }
  for (const b of gpuBrowsers) {
    for (const m of METRICS) {
      html += `<th class="th-sub">${m.label}</th><th class="th-sub">Speedup</th>`;
    }
  }
  html += '</tr></thead><tbody>';

  for (const key of keys) {
    const [model, variant] = key.split('::');
    const cpuItems = cpuByModelVariant[key] || [];
    const gpuByBrowser = groupBy(gpuByModelVariant[key] || [], 'browser');

    html += '<tr>';
    html += `<td>${escapeHtml(model)}</td>`;
    html += `<td><span class="mono">${escapeHtml(variant)}</span></td>`;

    // CPU columns
    for (const m of METRICS) {
      const val = avg(cpuItems, m.cpuField);
      html += `<td><span class="mono">${formatTokS(val)}</span></td>`;
    }

    // GPU columns per browser
    for (const b of gpuBrowsers) {
      const gpuItems = gpuByBrowser[b] || [];
      for (const m of METRICS) {
        const cpuVal = avg(cpuItems, m.cpuField);
        const gpuVal = avg(gpuItems, m.gpuField);
        const speedup = cpuVal && gpuVal ? gpuVal / cpuVal : null;
        const cls = speedup == null ? '' : speedup >= 3 ? 'text-success' : speedup >= 1.5 ? '' : speedup >= 1 ? 'text-muted' : 'text-error';
        html += `<td><span class="mono">${formatTokS(gpuVal)}</span></td>`;
        html += `<td><span class="mono ${cls}">${speedup != null ? speedup.toFixed(2) + '\u00d7' : '\u2014'}</span></td>`;
      }
    }

    html += '</tr>';
  }

  html += '</tbody></table></div></div>';
  container.innerHTML = html;
}
