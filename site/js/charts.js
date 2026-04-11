import { BROWSER_COLORS, quantSortKey, groupBy, formatTokS } from './utils.js';

// Global Chart.js theme
Chart.defaults.font.family = "'Manrope', system-ui, -apple-system, sans-serif";
Chart.defaults.color = '#a1a1aa';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 15, 18, 0.95)';
Chart.defaults.plugins.tooltip.borderColor = '#27272a';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.plugins.tooltip.padding = { top: 8, bottom: 8, left: 12, right: 12 };
Chart.defaults.plugins.tooltip.titleFont = { weight: '600', size: 13 };
Chart.defaults.plugins.tooltip.bodyFont = { family: "'JetBrains Mono', monospace", size: 12 };
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.boxHeight = 12;
Chart.defaults.plugins.legend.labels.padding = 16;

const chartInstances = new Map();

function destroyChart(id) {
  if (chartInstances.has(id)) {
    chartInstances.get(id).destroy();
    chartInstances.delete(id);
  }
}

const DARK_GRID = 'rgba(255,255,255,0.06)';
const DARK_TEXT = '#a1a1aa';
const TITLE_TEXT = '#e4e4e7';

function darkScales(xTitle, yTitle) {
  return {
    x: {
      ticks: { color: DARK_TEXT },
      grid: { color: DARK_GRID },
      title: xTitle ? { display: true, text: xTitle, color: DARK_TEXT } : undefined,
    },
    y: {
      ticks: { color: DARK_TEXT },
      grid: { color: DARK_GRID },
      title: yTitle ? { display: true, text: yTitle, color: DARK_TEXT } : undefined,
      beginAtZero: true,
    },
  };
}

function darkLegend() {
  return { labels: { color: DARK_TEXT, usePointStyle: true, pointStyle: 'circle' } };
}

function titleConfig(text) {
  return { display: true, text, color: TITLE_TEXT, font: { size: 14, weight: '600' } };
}

export function renderDecodeChart(results) {
  const canvasId = 'chart-decode';
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const passed = results.filter(r => r.status === 'done' && r.decode_tok_s != null);
  if (passed.length === 0) {
    canvas.parentElement.querySelector('.chart-empty')?.remove();
    const msg = document.createElement('div');
    msg.className = 'chart-empty';
    msg.textContent = 'No data';
    canvas.parentElement.appendChild(msg);
    return;
  }
  canvas.parentElement.querySelector('.chart-empty')?.remove();

  const byBrowser = groupBy(passed, 'browser');
  const allQuants = [...new Set(passed.map(r => r.variant))].sort((a, b) => quantSortKey(a) - quantSortKey(b));

  const datasets = Object.entries(byBrowser).map(([browser, items]) => {
    const byQuant = groupBy(items, 'variant');
    return {
      label: browser,
      backgroundColor: BROWSER_COLORS[browser] || '#888',
      data: allQuants.map(q => {
        const group = byQuant[q];
        if (!group) return null;
        const vals = group.map(r => r.decode_tok_s).filter(v => v != null);
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      }),
    };
  });

  chartInstances.set(canvasId, new Chart(canvas, {
    type: 'bar',
    data: { labels: allQuants, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: titleConfig('Decode Throughput by Quantization'),
        legend: darkLegend(),
        tooltip: {
          callbacks: { label: ctx => `${ctx.dataset.label}: ${formatTokS(ctx.raw)} tok/s` },
        },
      },
      scales: darkScales('Quantization', 'Decode tok/s'),
    },
  }));
}

export function renderPrefillChart(results) {
  const canvasId = 'chart-prefill';
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const passed = results.filter(r => r.status === 'done' && r.prefill_tok_s != null);
  if (passed.length === 0) {
    canvas.parentElement.querySelector('.chart-empty')?.remove();
    const msg = document.createElement('div');
    msg.className = 'chart-empty';
    msg.textContent = 'No data';
    canvas.parentElement.appendChild(msg);
    return;
  }
  canvas.parentElement.querySelector('.chart-empty')?.remove();

  const byBrowser = groupBy(passed, 'browser');
  const allQuants = [...new Set(passed.map(r => r.variant))].sort((a, b) => quantSortKey(a) - quantSortKey(b));

  const datasets = Object.entries(byBrowser).map(([browser, items]) => {
    const byQuant = groupBy(items, 'variant');
    return {
      label: browser,
      backgroundColor: BROWSER_COLORS[browser] || '#888',
      data: allQuants.map(q => {
        const group = byQuant[q];
        if (!group) return null;
        const vals = group.map(r => r.prefill_tok_s).filter(v => v != null);
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      }),
    };
  });

  chartInstances.set(canvasId, new Chart(canvas, {
    type: 'bar',
    data: { labels: allQuants, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: titleConfig('Prefill Throughput by Quantization'),
        legend: darkLegend(),
        tooltip: {
          callbacks: { label: ctx => `${ctx.dataset.label}: ${formatTokS(ctx.raw)} tok/s` },
        },
      },
      scales: darkScales('Quantization', 'Prefill tok/s'),
    },
  }));
}

export function renderSizeChart(results) {
  const canvasId = 'chart-size';
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const passed = results.filter(r => r.status === 'done' && r.decode_tok_s != null && r.sizeMB);
  if (passed.length === 0) {
    canvas.parentElement.querySelector('.chart-empty')?.remove();
    const msg = document.createElement('div');
    msg.className = 'chart-empty';
    msg.textContent = 'No data';
    canvas.parentElement.appendChild(msg);
    return;
  }
  canvas.parentElement.querySelector('.chart-empty')?.remove();

  const byBrowser = groupBy(passed, 'browser');

  const datasets = Object.entries(byBrowser).map(([browser, items]) => {
    const sorted = [...items].sort((a, b) => a.sizeMB - b.sizeMB);
    return {
      label: browser,
      borderColor: BROWSER_COLORS[browser] || '#888',
      backgroundColor: BROWSER_COLORS[browser] || '#888',
      data: sorted.map(r => ({ x: r.sizeMB, y: r.decode_tok_s })),
      showLine: true,
      pointRadius: 4,
      tension: 0.2,
    };
  });

  chartInstances.set(canvasId, new Chart(canvas, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: titleConfig('Throughput vs Model Size'),
        legend: darkLegend(),
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${ctx.parsed.x}MB \u2192 ${formatTokS(ctx.parsed.y)} tok/s`,
          },
        },
      },
      scales: darkScales('Model Size (MB)', 'Decode tok/s'),
    },
  }));
}

export function renderMachineChart(results, machines) {
  const canvasId = 'chart-machine';
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  const container = document.getElementById('machine-chart-section');
  if (!canvas || !container) return;

  if (machines.length <= 1) {
    container.style.display = 'none';
    return;
  }
  container.style.display = '';

  const passed = results.filter(r => r.status === 'done' && r.decode_tok_s != null);
  const quantCounts = {};
  for (const r of passed) quantCounts[r.variant] = (quantCounts[r.variant] || 0) + 1;
  const targetQuant = quantCounts['Q4_K_M'] ? 'Q4_K_M' : Object.keys(quantCounts).sort((a, b) => quantCounts[b] - quantCounts[a])[0];

  if (!targetQuant) {
    container.style.display = 'none';
    return;
  }

  const forQuant = passed.filter(r => r.variant === targetQuant);
  const byMachine = groupBy(forQuant, 'machineSlug');
  const machineLabels = Object.keys(byMachine);
  const browsers = [...new Set(forQuant.map(r => r.browser))].sort();

  const datasets = browsers.map(browser => ({
    label: browser,
    backgroundColor: BROWSER_COLORS[browser] || '#888',
    data: machineLabels.map(slug => {
      const items = byMachine[slug].filter(r => r.browser === browser);
      if (!items.length) return null;
      return items.reduce((s, r) => s + r.decode_tok_s, 0) / items.length;
    }),
  }));

  chartInstances.set(canvasId, new Chart(canvas, {
    type: 'bar',
    data: { labels: machineLabels, datasets },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: titleConfig(`Machine Comparison (${targetQuant})`),
        legend: darkLegend(),
        tooltip: {
          callbacks: { label: ctx => `${ctx.dataset.label}: ${formatTokS(ctx.raw)} tok/s` },
        },
      },
      scales: darkScales('Decode tok/s', 'Machine'),
    },
  }));
}
