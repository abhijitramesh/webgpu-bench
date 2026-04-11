import { loadData, filterResults } from './data.js';
import { initFilters, populateQuantOptions, getFilters, resetFilters } from './filters.js';
import { renderDecodeChart, renderPrefillChart, renderSizeChart, renderMachineChart } from './charts.js';
import { renderResultsTable, renderErrorTable, renderMachineInfo } from './tables.js';

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

  // Wire reset button
  const resetBtn = document.getElementById('filter-reset');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      resetFilters();
      render();
    });
  }

  // Init section navigation
  initSectionNav();

  // Initial render
  render();
}

function render() {
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
      filters.model !== 'all' || filters.status !== 'all' || filters.quants.size > 0;
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
