import { loadData, filterResults } from './data.js';
import { initFilters, populateQuantOptions, getFilters, resetFilters } from './filters.js';
import { renderDecodeChart, renderPrefillChart, renderSizeChart, renderMachineChart, renderCpuGpuChart, renderSpeedupChart } from './charts.js';
import { renderResultsTable, renderErrorTable, renderMachineInfo, renderCpuGpuTable } from './tables.js';

let appData = null;

async function init() {
  try {
    appData = await loadData();
  } catch (e) {
    document.getElementById('loading').innerHTML = `
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

  // Reset button visibility
  const resetBtn = document.getElementById('filter-reset');
  if (resetBtn) {
    const active = filters.machine !== 'all' || filters.browser !== 'all' ||
      filters.model !== 'all' || filters.backend !== 'all' ||
      filters.status !== 'all' || filters.quants.size > 0;
    resetBtn.style.display = active ? '' : 'none';
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
  renderCpuGpuChart(filtered, metric);
  renderSpeedupChart(filtered, metric);
  renderCpuGpuTable(filtered);
}

function initSectionNav() {
  const nav = document.getElementById('section-nav');
  if (!nav) return;

  const buttons = nav.querySelectorAll('.section-nav-item');
  const sections = [];

  buttons.forEach(btn => {
    const sectionId = btn.dataset.section;
    const section = document.getElementById(sectionId);
    if (section) sections.push({ btn, section });

    btn.addEventListener('click', (e) => {
      e.preventDefault();
      if (section) {
        const navHeight = nav.offsetHeight;
        const top = section.getBoundingClientRect().top + window.scrollY - navHeight;
        window.scrollTo({ top, behavior: 'smooth' });
      }
    });
  });

  // Scroll spy with IntersectionObserver
  if (sections.length === 0) return;

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          buttons.forEach(b => b.classList.toggle('active', b.dataset.section === id));
        }
      }
    },
    { rootMargin: '-20% 0px -60% 0px' }
  );

  sections.forEach(({ section }) => observer.observe(section));
}

init();
