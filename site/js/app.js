import { loadData, filterResults } from './data.js';
import { initFilters, populateQuantOptions, getFilters, resetFilters } from './filters.js';
import { renderDecodeChart, renderPrefillChart, renderSizeChart, renderMachineChart, renderCpuGpuChart, renderSpeedupChart } from './charts.js';
import { renderResultsTable, renderErrorTable, renderMachineInfo, renderCpuGpuTable } from './tables.js';

let appData = null;

async function init() {
  try {
    appData = await loadData();
  } catch (e) {
    const loading = document.getElementById('loading');
    loading.className = 'loading-state';
    loading.innerHTML = `
      <div class="loading-content">
        <p class="loading-error">Failed to load data</p>
        <p class="loading-hint">Run: <code>node scripts/build-site.js</code></p>
      </div>
    `;
    return;
  }

  // Hide loading, show dashboard with entrance animation
  const loading = document.getElementById('loading');
  const dashboard = document.getElementById('dashboard');
  loading.style.display = 'none';
  dashboard.style.display = '';
  requestAnimationFrame(() => dashboard.classList.add('animate-in'));

  // Populate quant options from actual data
  populateQuantOptions(appData.results);

  // Surface the dataset's last-updated time so users know data freshness.
  renderHeroMeta(appData);

  // Init filter dropdowns
  initFilters(appData.meta, () => render());

  // Wire theme toggle
  document.getElementById('theme-toggle')?.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    if (appData) render();
  });

  // Wire reset button
  const resetBtn = document.getElementById('filter-reset');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      resetFilters();
      render();
    });
  }

  // Wire metric selector for CPU vs GPU section
  const metricSelect = document.getElementById('cpu-gpu-metric');
  if (metricSelect) {
    metricSelect.addEventListener('change', () => render());
  }

  // Init section navigation
  initSectionNav();

  // Initial render
  render();
}

function render() {
  // Sync Chart.js defaults with current theme
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  Chart.defaults.color = isDark ? '#a1a1aa' : '#71717a';
  Chart.defaults.plugins.tooltip.backgroundColor = isDark ? 'rgba(15,15,18,0.95)' : 'rgba(255,255,255,0.95)';
  Chart.defaults.plugins.tooltip.borderColor = isDark ? '#27272a' : '#e4e4e7';
  Chart.defaults.plugins.tooltip.titleColor = isDark ? '#e4e4e7' : '#09090b';
  Chart.defaults.plugins.tooltip.bodyColor = isDark ? '#a1a1aa' : '#71717a';

  const filters = getFilters();
  const filtered = filterResults(appData.results, filters);

  // Summary cards
  const passed = filtered.filter(r => r.status === 'done');
  document.getElementById('stat-machines').textContent = appData.meta.machines.length;
  document.getElementById('stat-benchmarks').textContent = filtered.length;
  const passRate = filtered.length > 0 ? ((passed.length / filtered.length) * 100).toFixed(0) : '0';
  document.getElementById('stat-pass-rate').textContent = `${passRate}%`;

  // Results count
  const countEl = document.getElementById('results-count');
  if (countEl) {
    const total = appData.results.length;
    countEl.textContent = filtered.length === total
      ? `${total} total`
      : `${filtered.length} of ${total}`;
  }

  // Reset button — always visible so users understand filters can be
  // cleared; disabled when nothing to reset.
  const resetBtn = document.getElementById('filter-reset');
  if (resetBtn) {
    const activeCount = (filters.machine !== 'all' ? 1 : 0) + (filters.browser !== 'all' ? 1 : 0) +
      (filters.model !== 'all' ? 1 : 0) + (filters.backend !== 'all' ? 1 : 0) +
      (filters.status !== 'all' ? 1 : 0) + (filters.quants.size > 0 ? 1 : 0);
    resetBtn.disabled = activeCount === 0;
    resetBtn.style.display = '';
    const label = resetBtn.querySelector('.filter-reset-label') || resetBtn;
    if (label !== resetBtn) {
      label.textContent = activeCount ? `Reset (${activeCount})` : 'Reset';
    }
  }

  // Tables
  renderResultsTable(filtered);
  renderErrorTable(filtered);
  renderMachineInfo(appData.meta.machines);

  // Charts
  renderDecodeChart(filtered);
  renderPrefillChart(filtered);
  renderSizeChart(filtered);
  renderMachineChart(filtered, appData.meta.machines);

  // CPU vs GPU comparison
  const metric = document.getElementById('cpu-gpu-metric')?.value || 'decode_tok_s';
  renderCpuGpuSection(filtered, metric);
}

/* Consolidate the 3-part CPU-vs-GPU block (two charts + table). When there
   is no CPU baseline or no overlapping GPU data, render a single inline
   empty state and hide the charts+table so the user doesn't see the same
   message repeated three times. */
function renderCpuGpuSection(filtered, metric) {
  const chartsGrid = document.querySelector('#performance-section .charts-grid:nth-of-type(2)');
  const table = document.getElementById('cpu-gpu-table');
  const passed = filtered.filter(r => r.status === 'done');
  const cpuResults = passed.filter(r => r.nGpuLayers === 0);
  const gpuResults = passed.filter(r => r.nGpuLayers !== 0);

  if (!chartsGrid || !table) {
    renderCpuGpuChart(filtered, metric);
    renderSpeedupChart(filtered, metric);
    renderCpuGpuTable(filtered);
    return;
  }

  if (cpuResults.length === 0 || gpuResults.length === 0) {
    chartsGrid.hidden = true;
    const reason = cpuResults.length === 0
      ? 'No CPU baseline in the current filter. Select "All Backends" or enable CPU baselines when benchmarking with <code>--consistency</code>.'
      : 'No WebGPU runs in the current filter. Adjust the Backend filter to include WebGPU.';
    table.innerHTML = `<div class="empty-state"><p>${reason}</p></div>`;
    return;
  }

  chartsGrid.hidden = false;
  renderCpuGpuChart(filtered, metric);
  renderSpeedupChart(filtered, metric);
  renderCpuGpuTable(filtered);
}

function renderHeroMeta(data) {
  const el = document.getElementById('hero-meta');
  if (!el) return;
  const generated = data?.meta?.generatedAt;
  const machineCount = data?.meta?.machines?.length || 0;
  const resultCount = data?.results?.length || 0;
  const parts = [];
  if (machineCount > 0) parts.push(`${machineCount} machine${machineCount === 1 ? '' : 's'}`);
  if (resultCount > 0) parts.push(`${resultCount} benchmark${resultCount === 1 ? '' : 's'}`);
  if (generated) {
    const d = new Date(generated);
    const rel = formatRelativeTime(d);
    parts.push(`updated ${rel}`);
  }
  if (parts.length === 0) return;
  el.textContent = parts.join(' · ');
  el.hidden = false;
  if (generated) el.title = new Date(generated).toLocaleString();
}

function formatRelativeTime(date) {
  const now = Date.now();
  const diff = Math.max(0, now - date.getTime());
  const min = 60_000, hr = 60 * min, day = 24 * hr;
  if (diff < min) return 'just now';
  if (diff < hr) return `${Math.floor(diff / min)} min ago`;
  if (diff < day) return `${Math.floor(diff / hr)} h ago`;
  const days = Math.floor(diff / day);
  if (days < 30) return `${days} day${days === 1 ? '' : 's'} ago`;
  return date.toISOString().slice(0, 10);
}

function initSectionNav() {
  const nav = document.getElementById('section-nav');
  if (!nav) return;

  const buttons = nav.querySelectorAll('.section-nav-item');
  const sections = [];

  // Prefer the sticky wrapper height so the jumped-to section isn't
  // obscured by the sticky head.
  const stickyHead = document.querySelector('.sticky-head') || nav;

  buttons.forEach(btn => {
    const sectionId = btn.dataset.section;
    const section = document.getElementById(sectionId);
    if (section) sections.push({ btn, section });

    btn.addEventListener('click', (e) => {
      e.preventDefault();
      if (section) {
        const offset = stickyHead.offsetHeight + 8;
        const top = section.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top, behavior: 'smooth' });
      }
    });
  });

  // Scroll spy: instead of IntersectionObserver (which fires inconsistently
  // when multiple sections overlap the observer band), compute the
  // currently-active section on scroll by comparing each section's top to
  // the bottom of the sticky head. Cheaper and predictable.
  if (sections.length === 0) return;

  let ticking = false;
  const updateActive = () => {
    const anchor = stickyHead.offsetHeight + 16;
    let activeId = sections[0].section.id;
    for (const { section } of sections) {
      const top = section.getBoundingClientRect().top;
      if (top - anchor <= 0) activeId = section.id;
      else break;
    }
    buttons.forEach(b => b.classList.toggle('active', b.dataset.section === activeId));
  };
  const onScroll = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => { updateActive(); ticking = false; });
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  updateActive();
}

init();
