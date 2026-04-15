import { QUANT_ORDER, quantSortKey } from './utils.js';

let selectedQuants = new Set();
let onChange = null;

export function initFilters(meta, onChangeCallback) {
  onChange = onChangeCallback;

  // Machine select
  const machineSelect = document.getElementById('filter-machine');
  machineSelect.innerHTML = '<option value="all">All Machines</option>';
  for (const m of meta.machines) {
    const opt = document.createElement('option');
    opt.value = m.slug;
    opt.textContent = `${m.cpus} (${m.totalMemoryGB}GB)`;
    machineSelect.appendChild(opt);
  }

  // Browser select
  const browserSelect = document.getElementById('filter-browser');
  browserSelect.innerHTML = '<option value="all">All Browsers</option>';
  for (const b of meta.browsers) {
    const opt = document.createElement('option');
    opt.value = b;
    opt.textContent = b.charAt(0).toUpperCase() + b.slice(1);
    browserSelect.appendChild(opt);
  }

  // Model select
  const modelSelect = document.getElementById('filter-model');
  modelSelect.innerHTML = '<option value="all">All Models</option>';
  for (const m of meta.models) {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    modelSelect.appendChild(opt);
  }

  // Backend select
  const backendSelect = document.getElementById('filter-backend');
  backendSelect.innerHTML = `
    <option value="all">All Backends</option>
    <option value="webgpu">WebGPU</option>
    <option value="cpu">CPU</option>
  `;

  // Status select
  const statusSelect = document.getElementById('filter-status');
  statusSelect.innerHTML = `
    <option value="all">All Status</option>
    <option value="pass">Pass</option>
    <option value="fail">Fail</option>
  `;

  // Quant multi-select
  initQuantMultiSelect(meta);

  // Wire up change events for single selects
  for (const sel of [machineSelect, browserSelect, modelSelect, backendSelect, statusSelect]) {
    sel.addEventListener('change', fireChange);
  }
}

function initQuantMultiSelect(meta) {
  const btn = document.getElementById('quant-dropdown-btn');
  const dropdown = document.getElementById('quant-dropdown');

  // Toggle dropdown
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdown.classList.toggle('open');
  });

  // Close on outside click
  document.addEventListener('click', (e) => {
    if (!dropdown.contains(e.target) && e.target !== btn) {
      dropdown.classList.remove('open');
    }
  });
}

export function populateQuantOptions(results) {
  const quantsInData = new Set(results.map(r => r.variant));
  const sorted = [...quantsInData].sort((a, b) => quantSortKey(a) - quantSortKey(b));

  const list = document.getElementById('quant-options');
  list.innerHTML = '';

  // Select All
  const selectAllLabel = document.createElement('label');
  selectAllLabel.className = 'quant-option select-all';
  const selectAllCb = document.createElement('input');
  selectAllCb.type = 'checkbox';
  selectAllCb.checked = true;
  selectAllCb.id = 'quant-select-all';
  selectAllLabel.appendChild(selectAllCb);
  selectAllLabel.appendChild(document.createTextNode(' Select All'));
  list.appendChild(selectAllLabel);

  const checkboxes = [];

  for (const q of sorted) {
    const label = document.createElement('label');
    label.className = 'quant-option';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.value = q;
    cb.className = 'quant-cb';
    label.appendChild(cb);
    label.appendChild(document.createTextNode(` ${q}`));
    list.appendChild(label);
    checkboxes.push(cb);

    cb.addEventListener('change', () => {
      updateQuantSelection(checkboxes, selectAllCb);
      fireChange();
    });
  }

  selectAllCb.addEventListener('change', () => {
    const checked = selectAllCb.checked;
    for (const cb of checkboxes) cb.checked = checked;
    updateQuantSelection(checkboxes, selectAllCb);
    fireChange();
  });

  // Update button text
  updateQuantButtonText(sorted.length, sorted.length);
}

function updateQuantSelection(checkboxes, selectAllCb) {
  selectedQuants = new Set();
  let checkedCount = 0;
  for (const cb of checkboxes) {
    if (cb.checked) {
      checkedCount++;
    }
  }

  if (checkedCount === checkboxes.length) {
    selectedQuants = new Set();
    selectAllCb.checked = true;
    selectAllCb.indeterminate = false;
  } else if (checkedCount === 0) {
    selectedQuants = new Set(['__none__']);
    selectAllCb.checked = false;
    selectAllCb.indeterminate = false;
  } else {
    selectedQuants = new Set();
    for (const cb of checkboxes) {
      if (cb.checked) selectedQuants.add(cb.value);
    }
    selectAllCb.checked = false;
    selectAllCb.indeterminate = true;
  }

  updateQuantButtonText(checkedCount, checkboxes.length);
}

function updateQuantButtonText(checked, total) {
  const textEl = document.getElementById('quant-dropdown-text');
  if (!textEl) return;
  if (checked === total) {
    textEl.textContent = 'All Quants';
  } else if (checked === 0) {
    textEl.textContent = 'No Quants';
  } else {
    textEl.textContent = `${checked}/${total} Quants`;
  }
}

function fireChange() {
  if (onChange) onChange(getFilters());
}

export function resetFilters() {
  // Reset all select dropdowns
  document.getElementById('filter-machine').value = 'all';
  document.getElementById('filter-browser').value = 'all';
  document.getElementById('filter-model').value = 'all';
  document.getElementById('filter-backend').value = 'all';
  document.getElementById('filter-status').value = 'all';

  // Reset quant checkboxes
  const selectAllCb = document.getElementById('quant-select-all');
  const checkboxes = [...document.querySelectorAll('#quant-options .quant-cb')];

  if (selectAllCb) {
    selectAllCb.checked = true;
    selectAllCb.indeterminate = false;
  }
  for (const cb of checkboxes) cb.checked = true;

  // Reset internal state
  selectedQuants = new Set();
  updateQuantButtonText(checkboxes.length, checkboxes.length);
}

export function getFilters() {
  return {
    machine: document.getElementById('filter-machine').value,
    browser: document.getElementById('filter-browser').value,
    model: document.getElementById('filter-model').value,
    backend: document.getElementById('filter-backend').value,
    status: document.getElementById('filter-status').value,
    quants: selectedQuants,
  };
}
