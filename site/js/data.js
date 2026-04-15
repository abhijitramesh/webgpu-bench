let cachedData = null;

export async function loadData() {
  if (cachedData) return cachedData;
  const resp = await fetch('data/combined.json');
  cachedData = await resp.json();
  return cachedData;
}

export function filterResults(results, filters) {
  return results.filter(r => {
    if (filters.machine && filters.machine !== 'all' && r.machineSlug !== filters.machine) return false;
    if (filters.browser && filters.browser !== 'all' && r.browser !== filters.browser) return false;
    if (filters.model && filters.model !== 'all' && r.model !== filters.model) return false;
    if (filters.backend && filters.backend !== 'all') {
      if (filters.backend === 'cpu' && r.nGpuLayers !== 0) return false;
      if (filters.backend === 'webgpu' && r.nGpuLayers === 0) return false;
    }
    if (filters.status && filters.status !== 'all') {
      if (filters.status === 'pass' && r.status !== 'done') return false;
      if (filters.status === 'fail' && r.status === 'done') return false;
    }
    if (filters.quants && filters.quants.size > 0 && !filters.quants.has(r.variant)) return false;
    return true;
  });
}
