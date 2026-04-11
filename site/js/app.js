import { loadData, filterResults } from './data.js';
import { initFilters, populateQuantOptions, getFilters } from './filters.js';
import { renderDecodeChart, renderPrefillChart, renderSizeChart, renderMachineChart } from './charts.js';
import { renderResultsTable, renderErrorTable, renderMachineInfo } from './tables.js';

let appData = null;

async function init() {
  try {
    appData = await loadData();
  } catch (e) {
    document.getElementById('loading').textContent = 'Failed to load data. Run: node scripts/build-site.js';
    return;
  }

  document.getElementById('loading').style.display = 'none';
  document.getElementById('dashboard').style.display = '';

  // Populate quant options from actual data
  populateQuantOptions(appData.results);

  // Init filter dropdowns
  initFilters(appData.meta, () => render());

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

init();
