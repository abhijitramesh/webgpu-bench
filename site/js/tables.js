import { formatTokS, formatMs, categorizeError, groupBy } from './utils.js';

export function renderResultsTable(results) {
  const container = document.getElementById('results-table');
  if (!container) return;

  if (results.length === 0) {
    container.innerHTML = '<p class="empty-msg">No results match the current filters.</p>';
    return;
  }

  const cols = [
    { key: 'machineSlug', label: 'Machine' },
    { key: 'model', label: 'Model' },
    { key: 'variant', label: 'Quant' },
    { key: 'sizeMB', label: 'Size (MB)' },
    { key: 'browser', label: 'Browser' },
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
    { key: 'error', label: 'Error' },
  ];

  let html = '<table class="results"><thead><tr>';
  for (const col of cols) {
    html += `<th data-key="${col.key}">${col.label}</th>`;
  }
  html += '</tr></thead><tbody>';

  for (const r of results) {
    const rowClass = r.status === 'done' ? 'row-pass' : 'row-fail';
    html += `<tr class="${rowClass}">`;
    for (const col of cols) {
      html += '<td>';
      switch (col.key) {
        case 'status':
          html += r.status === 'done'
            ? '<span class="badge pass">PASS</span>'
            : '<span class="badge fail">FAIL</span>';
          break;
        case 'webgpuAvailable':
          html += r.webgpuAvailable ? 'Yes' : 'No';
          break;
        case 'decode_tok_s':
        case 'prefill_tok_s':
          html += formatTokS(r[col.key]);
          break;
        case 't_eval_ms':
        case 't_p_eval_ms':
          html += formatMs(r[col.key]);
          break;
        case 'wallTimeMs':
          html += r.wallTimeMs != null ? (r.wallTimeMs / 1000).toFixed(1) : '—';
          break;
        case 'consistency_rate':
          if (r.consistency_rate != null) {
            const pct = (r.consistency_rate * 100).toFixed(1);
            const cls = r.consistency_rate >= 0.95 ? 'text-green' : r.consistency_rate >= 0.90 ? '' : 'text-red';
            const diverge = r.consistency_first_disagree >= 0 ? ` (diverge@${r.consistency_first_disagree})` : '';
            html += `<span class="${cls}">${pct}%${diverge}</span>`;
          } else {
            html += '—';
          }
          break;
        case 'error':
          if (r.error) {
            const cat = categorizeError(r.error);
            const short = r.error.length > 60 ? r.error.slice(0, 60) + '...' : r.error;
            html += `<span class="error-cell" title="${escapeHtml(r.error)}">${cat}: ${escapeHtml(short)}</span>`;
          } else {
            html += '—';
          }
          break;
        case 'sizeMB':
        case 'n_eval':
        case 'n_p_eval':
          html += r[col.key] != null ? r[col.key] : '—';
          break;
        default:
          html += escapeHtml(String(r[col.key] ?? '—'));
      }
      html += '</td>';
    }
    html += '</tr>';
  }

  html += '</tbody></table>';
  container.innerHTML = html;
}

export function renderErrorTable(results) {
  const container = document.getElementById('error-table');
  if (!container) return;

  const errors = results.filter(r => r.status !== 'done' && r.error);
  if (errors.length === 0) {
    container.innerHTML = '<p class="empty-msg">No errors.</p>';
    return;
  }

  const grouped = groupBy(errors, r => categorizeError(r.error));

  let html = '<table class="errors"><thead><tr><th>Category</th><th>Count</th><th>Variants</th><th>Browsers</th></tr></thead><tbody>';
  for (const [cat, items] of Object.entries(grouped).sort((a, b) => b[1].length - a[1].length)) {
    const variants = [...new Set(items.map(i => i.variant))].join(', ');
    const browsers = [...new Set(items.map(i => i.browser))].join(', ');
    html += `<tr><td>${cat}</td><td>${items.length}</td><td>${variants}</td><td>${browsers}</td></tr>`;
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}

export function renderMachineInfo(machines) {
  const container = document.getElementById('machine-info');
  if (!container) return;

  if (machines.length === 0) {
    container.innerHTML = '<p class="empty-msg">No machine data.</p>';
    return;
  }

  let html = '<div class="machine-cards">';
  for (const m of machines) {
    html += `
      <div class="card machine-card">
        <h3>${escapeHtml(m.cpus)}</h3>
        <div class="machine-specs">
          <div><strong>Platform:</strong> ${m.platform}</div>
          <div><strong>Arch:</strong> ${m.arch}</div>
          <div><strong>RAM:</strong> ${m.totalMemoryGB} GB</div>
          <div><strong>Results:</strong> ${m.resultCount}</div>
          <div><strong>Passed:</strong> <span class="text-green">${m.passCount}</span></div>
          <div><strong>Failed:</strong> <span class="text-red">${m.resultCount - m.passCount}</span></div>
        </div>
      </div>
    `;
  }
  html += '</div>';
  container.innerHTML = html;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
