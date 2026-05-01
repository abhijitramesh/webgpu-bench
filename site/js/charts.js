import { BROWSER_COLORS, quantSortKey, groupBy, formatTokS } from './utils.js';
import { expandCpuRows } from './data.js';

// Global Chart.js theme — uses the site's font tokens and a calm tooltip
// silhouette. Colors are pulled from CSS variables at render time so the
// theme toggle works without rebuilding chart instances.
Chart.defaults.font.family = "'Bricolage Grotesque', system-ui, -apple-system, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = '#a1a1aa';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 15, 18, 0.95)';
Chart.defaults.plugins.tooltip.borderColor = '#27272a';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.plugins.tooltip.padding = { top: 8, bottom: 8, left: 12, right: 12 };
Chart.defaults.plugins.tooltip.titleFont = { weight: '600', size: 12, family: "'Bricolage Grotesque', system-ui, sans-serif" };
Chart.defaults.plugins.tooltip.bodyFont = { family: "'Geist Mono', 'SF Mono', monospace", size: 12 };
Chart.defaults.plugins.tooltip.displayColors = true;
Chart.defaults.plugins.tooltip.boxPadding = 6;
Chart.defaults.plugins.legend.labels.boxWidth = 8;
Chart.defaults.plugins.legend.labels.boxHeight = 8;
Chart.defaults.plugins.legend.labels.padding = 16;
Chart.defaults.plugins.legend.labels.font = { family: "'Geist Mono', monospace", size: 11 };
Chart.defaults.elements.bar.borderRadius = 4;
Chart.defaults.elements.bar.borderSkipped = false;

const chartInstances = new Map();

function destroyChart(id) {
  if (chartInstances.has(id)) {
    chartInstances.get(id).destroy();
    chartInstances.delete(id);
  }
}

function themeColors() {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    grid:  dark ? 'rgba(255,255,255,0.04)' : 'rgba(15, 23, 42, 0.05)',
    text:  dark ? '#a1a1aa' : '#71717a',
    title: dark ? '#e4e4e7' : '#09090b',
    signal: dark ? '#22e09a' : '#0fa968',
  };
}

function darkScales(xTitle, yTitle) {
  const c = themeColors();
  return {
    x: {
      ticks: { color: c.text },
      grid: { color: c.grid },
      title: xTitle ? { display: true, text: xTitle, color: c.text } : undefined,
    },
    y: {
      ticks: { color: c.text },
      grid: { color: c.grid },
      title: yTitle ? { display: true, text: yTitle, color: c.text } : undefined,
      beginAtZero: true,
    },
  };
}

function darkLegend() {
  const c = themeColors();
  return { labels: { color: c.text, usePointStyle: true, pointStyle: 'circle' } };
}

function titleConfig(text) {
  const c = themeColors();
  return { display: true, text, color: c.title, font: { size: 14, weight: '600' } };
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

const CPU_COLOR = 'rgba(245, 158, 11, 0.75)';

function showEmptyState(canvas, msg) {
  canvas.parentElement.querySelector('.chart-empty')?.remove();
  const el = document.createElement('div');
  el.className = 'chart-empty';
  el.textContent = msg || 'No data';
  canvas.parentElement.appendChild(el);
}

function clearEmptyState(canvas) {
  canvas.parentElement.querySelector('.chart-empty')?.remove();
}

const METRIC_LABELS = {
  decode_tok_s: 'Decode tok/s',
  prefill_tok_s: 'Prefill tok/s',
};

function avgBy(items, field) {
  const vals = items.map(r => r[field]).filter(v => v != null);
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
}

// CPU is pinned to d=0 by the runner, so apples-to-apples means reading
// GPU's d=0 number. The CPU side keeps its bare metric (CPU records are
// depth-pinned to 0 either way); GPU reads `<metric>_d0`. Plain-Run
// records that only measured d=N have null `_d0` and silently drop out.
function gpuDepthField(metric) { return `${metric}_d0`; }

export function renderCpuGpuChart(results, metric = 'decode_tok_s') {
  const canvasId = 'chart-cpu-gpu';
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const gpuMetric = gpuDepthField(metric);
  const passed = results.filter(r => r.status === 'done');
  // expandCpuRows folds in cpu_baseline_* from browser-flow GPU records.
  const cpuResults = expandCpuRows(passed).filter(r => r[metric] != null);
  const gpuResults = passed.filter(r => r.nGpuLayers !== 0 && r[gpuMetric] != null);

  if (cpuResults.length === 0 || gpuResults.length === 0) {
    showEmptyState(canvas, cpuResults.length === 0 ? 'No CPU baseline data in current filter' : 'No GPU data in current filter');
    return;
  }
  clearEmptyState(canvas);

  const cpuVariants = new Set(cpuResults.map(r => r.variant));
  const allQuants = [...new Set(gpuResults.map(r => r.variant))]
    .filter(q => cpuVariants.has(q))
    .sort((a, b) => quantSortKey(a) - quantSortKey(b));

  if (allQuants.length === 0) {
    showEmptyState(canvas, 'No overlapping variants between CPU and GPU');
    return;
  }

  const cpuByVariant = groupBy(cpuResults, 'variant');
  const cpuData = allQuants.map(q => avgBy(cpuByVariant[q] || [], metric));

  const gpuByBrowser = groupBy(gpuResults, 'browser');
  const gpuDatasets = Object.entries(gpuByBrowser).map(([browser, items]) => {
    const byVariant = groupBy(items, 'variant');
    return {
      label: browser,
      backgroundColor: BROWSER_COLORS[browser] || '#888',
      data: allQuants.map(q => avgBy(byVariant[q] || [], gpuMetric)),
    };
  });

  const metricLabel = METRIC_LABELS[metric] || metric;
  chartInstances.set(canvasId, new Chart(canvas, {
    type: 'bar',
    data: { labels: allQuants, datasets: [{ label: 'CPU', backgroundColor: CPU_COLOR, data: cpuData }, ...gpuDatasets] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: titleConfig(`CPU vs WebGPU: ${metricLabel} @ d0`),
        legend: darkLegend(),
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${formatTokS(ctx.raw)} tok/s` } },
      },
      scales: darkScales('Quantization', metricLabel),
    },
  }));
}

export function renderSpeedupChart(results, metric = 'decode_tok_s') {
  const canvasId = 'chart-speedup';
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const gpuMetric = gpuDepthField(metric);
  const passed = results.filter(r => r.status === 'done');
  const cpuResults = expandCpuRows(passed).filter(r => r[metric] != null);
  const gpuResults = passed.filter(r => r.nGpuLayers !== 0 && r[gpuMetric] != null);

  if (cpuResults.length === 0 || gpuResults.length === 0) {
    showEmptyState(canvas, cpuResults.length === 0 ? 'No CPU baseline data in current filter' : 'No GPU data in current filter');
    return;
  }
  clearEmptyState(canvas);

  const cpuAvgByVariant = {};
  for (const [q, items] of Object.entries(groupBy(cpuResults, 'variant'))) {
    cpuAvgByVariant[q] = avgBy(items, metric);
  }

  const allQuants = [...new Set(gpuResults.map(r => r.variant))]
    .filter(q => cpuAvgByVariant[q] != null)
    .sort((a, b) => quantSortKey(a) - quantSortKey(b));

  if (allQuants.length === 0) {
    showEmptyState(canvas, 'No overlapping variants between CPU and GPU');
    return;
  }

  const gpuByBrowser = groupBy(gpuResults, 'browser');
  const barDatasets = Object.entries(gpuByBrowser).map(([browser, items]) => {
    const byVariant = groupBy(items, 'variant');
    return {
      label: browser,
      backgroundColor: BROWSER_COLORS[browser] || '#888',
      data: allQuants.map(q => {
        const cpuAvg = cpuAvgByVariant[q];
        const gpuAvg = avgBy(byVariant[q] || [], gpuMetric);
        return cpuAvg && gpuAvg ? gpuAvg / cpuAvg : null;
      }),
    };
  });

  const refLine = {
    label: '1\u00d7',
    type: 'line',
    data: allQuants.map(() => 1),
    borderColor: 'rgba(255,255,255,0.3)',
    borderDash: [4, 4],
    borderWidth: 1.5,
    pointRadius: 0,
    fill: false,
    order: -1,
  };

  const metricLabel = METRIC_LABELS[metric] || metric;
  chartInstances.set(canvasId, new Chart(canvas, {
    type: 'bar',
    data: { labels: allQuants, datasets: [...barDatasets, refLine] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: titleConfig(`WebGPU Speedup over CPU (${metricLabel} @ d0)`),
        legend: {
          ...darkLegend(),
          labels: { ...darkLegend().labels, filter: item => item.text !== '1\u00d7' },
        },
        tooltip: {
          filter: item => item.dataset.label !== '1\u00d7',
          callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.raw != null ? ctx.raw.toFixed(2) + '\u00d7' : '\u2014'}` },
        },
      },
      scales: {
        x: { ticks: { color: themeColors().text }, grid: { color: themeColors().grid }, title: { display: true, text: 'Quantization', color: themeColors().text } },
        y: {
          ticks: { color: themeColors().text, callback: v => `${v.toFixed(1)}\u00d7` },
          grid: { color: themeColors().grid },
          title: { display: true, text: 'Speedup (\u00d7)', color: themeColors().text },
          beginAtZero: true,
        },
      },
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
