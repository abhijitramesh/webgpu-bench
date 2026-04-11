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
  for (const sel of [machineSelect, browserSelect, modelSelect, statusSelect]) {
    sel.addEventListener('change', fireChange);
  }
}

function initQuantMultiSelect(meta) {
  const allQuants = new Set();
  // We'll collect quants from the meta if available, or use QUANT_ORDER
  // Since meta doesn't list quants, we build from known QUANT_ORDER
  // and only show ones that appear in the data
  // For now, we'll populate after data loads — store ref for later

  const btn = document.getElementById('quant-dropdown-btn');
  const dropdown = document.getElementById('quant-dropdown');
  const list = document.getElementById('quant-options');

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
    } else {
      // We track unchecked - but actually we track checked as the selected set
    }
  }

  // If all are checked, selectedQuants stays empty (meaning "all")
  if (checkedCount === checkboxes.length) {
    selectedQuants = new Set();
    selectAllCb.checked = true;
    selectAllCb.indeterminate = false;
  } else if (checkedCount === 0) {
    // None checked — show nothing
    for (const cb of checkboxes) {
      selectedQuants.add(cb.value); // trick: add none, keep empty
    }
    selectedQuants = new Set(); // empty set but we need a way to show "none"
    // Actually: if none checked, we want no results. Use a sentinel.
    selectedQuants = new Set(['__none__']);
    selectAllCb.checked = false;
    selectAllCb.indeterminate = false;
  } else {
    // Some checked — selectedQuants = only checked ones
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
  const btn = document.getElementById('quant-dropdown-btn');
  if (checked === total) {
    btn.textContent = 'All Quants';
  } else if (checked === 0) {
    btn.textContent = 'No Quants';
  } else {
    btn.textContent = `${checked}/${total} Quants`;
  }
}

function fireChange() {
  if (onChange) onChange(getFilters());
}

export function getFilters() {
  return {
    machine: document.getElementById('filter-machine').value,
    browser: document.getElementById('filter-browser').value,
    model: document.getElementById('filter-model').value,
    status: document.getElementById('filter-status').value,
    quants: selectedQuants,
  };
}
