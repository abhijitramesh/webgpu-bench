import { formatTokS, formatMs, categorizeError, groupBy, quantSortKey } from './utils.js';

let lastResults = [];
let sortState = { key: null, dir: 'asc' };

const NUM_KEYS = new Set([
  'sizeMB', 'decode_tok_s', 'prefill_tok_s', 'n_eval', 't_eval_ms',
  'n_p_eval', 't_p_eval_ms', 'wallTimeMs', 'consistency_rate', 'nGpuLayers',
]);

function sortResults(results, key, dir) {
  const isNum = NUM_KEYS.has(key);
  return [...results].sort((a, b) => {
    let va = a[key], vb = b[key];
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
    container.innerHTML = '<div class="empty-state"><p>No results match the current filters.</p></div>';
    return;
  }

  const sorted = sortState.key ? sortResults(results, sortState.key, sortState.dir) : results;

  const cols = [
    { key: 'machineSlug', label: 'Machine' },
    { key: 'model', label: 'Model' },
    { key: 'variant', label: 'Quant' },
    { key: 'sizeMB', label: 'Size (MB)' },
    { key: 'browser', label: 'Browser' },
    { key: 'nGpuLayers', label: 'Backend' },
    { key: 'status', label: 'Status' },
    { key: 'buildType', label: 'Build' },
    { key: 'webgpuAvailable', label: 'WebGPU' },
    { key: 'decode_tok_s', label: 'Decode tok/s' },
    { key: 'prefill_tok_s', label: 'Prefill tok/s' },
    { key: 'n_eval', label: 'n_eval' },
    { key: 't_eval_ms', label: 't_eval (ms)' },
    { key: 'n_p_eval', label: 'n_p_eval' },
    { key: 't_p_eval_ms', label: 't_p_eval (ms)' },
    { key: 'wallTimeMs', label: 'Wall (s)' },
    { key: 'consistency_rate', label: 'CPU Match' },
    { key: 'llamaCppCommit', label: 'llama.cpp' },
    { key: 'error', label: 'Error' },
  ];

  let html = '<table class="results-table"><thead><tr>';
  for (const col of cols) {
    const isActive = sortState.key === col.key;
    const arrow = isActive ? (sortState.dir === 'asc' ? ' \u2191' : ' \u2193') : '';
    const cls = isActive ? ' class="sorted"' : '';
    html += `<th data-key="${col.key}"${cls}>${col.label}${arrow}</th>`;
  }
  html += '</tr></thead><tbody>';

  for (const r of sorted) {
    const rowClass = r.status === 'done' ? 'row-pass' : 'row-fail';
    html += `<tr class="${rowClass}">`;
    for (const col of cols) {
      html += '<td>';
      switch (col.key) {
        case 'status':
          html += r.status === 'done'
            ? '<span class="badge badge--pass">PASS</span>'
            : '<span class="badge badge--fail">FAIL</span>';
          break;
        case 'nGpuLayers':
          if (r.nGpuLayers != null) {
            const isCpu = r.nGpuLayers === 0;
            html += `<span class="badge ${isCpu ? 'badge--cpu' : 'badge--webgpu'}">${isCpu ? 'CPU' : 'WebGPU'}</span>`;
          } else {
            html += '<span class="text-muted">\u2014</span>';
          }
          break;
        case 'webgpuAvailable':
          html += r.webgpuAvailable
            ? '<span class="badge badge--yes">Yes</span>'
            : '<span class="badge badge--no">No</span>';
          break;
        case 'decode_tok_s':
        case 'prefill_tok_s':
          html += `<span class="mono">${formatTokS(r[col.key])}</span>`;
          break;
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
        case 'llamaCppCommit':
          if (r.llamaCppCommit) {
            const short = r.llamaCppCommit.slice(0, 10);
            html += `<a class="mono" href="https://github.com/ggml-org/llama.cpp/commit/${r.llamaCppCommit}" target="_blank" rel="noopener">${short}</a>`;
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
    }
    html += '</tr>';
  }

  html += '</tbody></table>';
  container.innerHTML = html;

  // Wire sort click handlers
  container.querySelectorAll('th[data-key]').forEach(th => {
    th.addEventListener('click', () => handleSort(th.dataset.key));
  });
}

export function renderErrorTable(results) {
  const container = document.getElementById('error-table');
  if (!container) return;

  const errors = results.filter(r => r.status !== 'done' && r.error);
  if (errors.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No errors found.</p></div>';
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

  if (machines.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No machine data.</p></div>';
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
          ${m.llamaCppCommit ? `<div class="spec-row"><span class="spec-label">llama.cpp</span><span class="spec-value"><a href="https://github.com/ggml-org/llama.cpp/commit/${m.llamaCppCommit}" target="_blank" rel="noopener">${m.llamaCppCommit.slice(0, 10)}</a></span></div>` : ''}
        </div>
      </div>`;
  }
  html += '</div>';
  container.innerHTML = html;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

export function renderCpuGpuTable(results) {
  const container = document.getElementById('cpu-gpu-table');
  if (!container) return;

  const METRICS = [
    { field: 'decode_tok_s', label: 'Decode tok/s' },
    { field: 'prefill_tok_s', label: 'Prefill tok/s' },
  ];

  const passed = results.filter(r => r.status === 'done');
  const cpuResults = passed.filter(r => r.nGpuLayers === 0);
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
      const val = avg(cpuItems, m.field);
      html += `<td><span class="mono">${formatTokS(val)}</span></td>`;
    }

    // GPU columns per browser
    for (const b of gpuBrowsers) {
      const gpuItems = gpuByBrowser[b] || [];
      for (const m of METRICS) {
        const cpuVal = avg(cpuItems, m.field);
        const gpuVal = avg(gpuItems, m.field);
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
