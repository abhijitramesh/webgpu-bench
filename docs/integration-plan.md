# Plan: unify the dashboard and one-click bench into a single app

## Context

The repo currently ships two separate web UIs on two separate deployments:

1. **Dashboard** — `site/index.html` + `site/css/style.css` (908 lines, shadcn-inspired, light/dark theme via `data-theme` on `<html>`) + `site/js/{app,data,filters,tables,charts,utils}.js`. Reads `site/data/combined.json` built from `data/machines/*.json` (which are regenerated from the HF dataset by `scripts/sync-from-dataset.mjs`). Section-nav: **Overview / Results / Performance / Errors / Machines**. Deployed to GitHub Pages via `.github/workflows/deploy-dashboard.yml` (cron + on-push).
2. **One-click bench** — root `bench.html` + `bench.css` (dark-mode only, its own design) + `bench-{app,core,source,device,hub,config}.js`. Device-fit check, variant checkboxes, Download → Run (CPU baseline + N GPU iterations per variant + consistency check on first iteration), output markdown with copy/download, HF OAuth + dataset submit. Deployed to an HF Space (Static) `abhijitramesh/webgpu-bench` via `.github/workflows/sync-to-hf-space.yml`. Reached at `https://abhijitramesh-webgpu-bench.static.hf.space/`.

The user wants a single app that honors the dashboard's design system. "Run" becomes a first-class section alongside the existing analysis tabs. Both deployments serve the same app.

## Goal

Single HTML entry point at `site/index.html`. Section-nav gains a **Run** tab. All one-click functionality moves into that section, restyled with the existing `site/css/style.css` tokens. The root-level `bench.html` / `bench.css` / `bench-*.js` files disappear (or are reduced to thin aliases).

### Mode matrix

| Origin | Dashboard | Run | Submit |
|---|---|---|---|
| `localhost:3000` (Express) | works (built locally via `npm run build:site`) | works (Express `/models/*` proxy + disk cache) | `POST /api/results` → `npm run submit` pushes to dataset |
| `*.static.hf.space` (HF Space) | works (`combined.json` bundled at deploy time) | works (direct HF fetch + OPFS cache) | HF Spaces OAuth → direct commit to `abhijitramesh/webgpu-bench-leaderboard` |
| `*.github.io` (GH Pages) | works (CI-built) | works read-only (direct HF fetch + OPFS) | **disabled** (no OAuth injection on GH Pages — UI hides Submit) |

Detection:
- `state.surface = 'localhost' | 'space' | 'pages' | 'file'`
- Driven by `location.origin` + presence of `/api/models` + presence of `window.huggingface?.variables?.OAUTH_CLIENT_ID`.
- `state.canSubmit = surface === 'localhost' || (surface === 'space' && hubConfigured)`.

## Target architecture

```
site/
├── index.html            (single entry; section-nav incl. Run)
├── methodology.html      (unchanged)
├── css/
│   └── style.css         (extended with Run-specific classes, using existing tokens)
├── data/
│   └── combined.json     (built by scripts/build-site.js, unchanged)
└── js/
    ├── app.js            (orchestrator; now also mounts Run section lazily)
    ├── data.js           (unchanged)
    ├── filters.js        (unchanged)
    ├── tables.js         (unchanged)
    ├── charts.js         (unchanged)
    ├── utils.js          (unchanged)
    └── run/
        ├── controller.js    (ex bench-app.js — section-scoped, uses style.css classes)
        ├── core.js          (ex bench-core.js)
        ├── source.js        (ex bench-source.js)
        ├── device.js        (ex bench-device.js)
        ├── hub.js           (ex bench-hub.js)
        └── config.js        (ex bench-config.js)
```

Root becomes:
```
webgpu-bench/
├── harness.html          (kept — runner.js uses this)
├── harness.js            (kept — still imports from site/js/run/core.js via relative path)
├── server.js             (extended: serves `site/` as default root, keeps `/api/*`, `/models/*`, `/build/*`, `/harness.html`)
├── models.json           (unchanged)
├── build/                (WASM, unchanged)
├── site/                 (merged app — see tree above)
├── spaces/README.md      (pointed at the merged site/)
├── scripts/              (unchanged, build-site.js extended to stage run assets)
└── .github/workflows/    (deploy-dashboard.yml unchanged; sync-to-hf-space.yml stages site/ instead of the flat bundle)
```

## File-by-file plan

### Phase 1 — Move and rename (pure moves, no behavior change)

| From | To | Notes |
|---|---|---|
| `bench-core.js` | `site/js/run/core.js` | Exports unchanged. |
| `bench-source.js` | `site/js/run/source.js` | |
| `bench-device.js` | `site/js/run/device.js` | |
| `bench-hub.js` | `site/js/run/hub.js` | |
| `bench-config.js` | `site/js/run/config.js` | |
| `bench-app.js` | `site/js/run/controller.js` | Refactor `init()` to mount into an existing DOM subtree (`#run-section`), not assume whole page. |

Update imports in each moved file (relative paths intra-subdir are unchanged).

Update `harness.js` import: `import { runBenchmarkCore } from './site/js/run/core.js'` and `import { localSource } from './site/js/run/source.js'`.

`server.js`: add `app.use('/site/js/run', express.static(path.join(__dirname, 'site', 'js', 'run')))` — or rely on the already-present `express.static(__dirname)` which serves everything.

### Phase 2 — Integrate Run into `site/index.html`

Add a new nav button and section. Keep existing markup exactly as-is; append:

```html
<!-- In the section-nav track, as the LAST item: -->
<button class="section-nav-item" data-section="run-section">Run</button>

<!-- After the existing #machines-section: -->
<section id="run-section" class="dash-section">
  <div class="container">
    <div class="section-header">
      <h2>Run a benchmark</h2>
      <span id="run-mode-badge" class="badge"></span>
    </div>

    <!-- Hub row (shown only when running on the HF Space) -->
    <div id="hub-row" class="card hub-row" hidden>
      <div class="hub-row-inner">
        <div class="hub-row-info">
          <span id="hf-user" class="muted"></span>
        </div>
        <div class="hub-row-actions">
          <button id="btn-signin" class="btn btn-secondary">Sign in with Hugging Face</button>
          <button id="btn-submit" class="btn btn-primary" disabled>Submit to leaderboard</button>
        </div>
      </div>
    </div>

    <!-- Device & budget card -->
    <div class="summary-grid run-device-grid">
      <div class="stat-card"> … deviceMemory / WebGPU / model budget … </div>
      <div class="stat-card"> … selected variants / cached / iterations … </div>
    </div>

    <!-- Variant filters + model catalogue -->
    <div class="filter-bar">
      <div class="filter-bar-inner run-filters">
        <!-- reuses .filter-group / .filter-select patterns -->
        <div class="filter-group"> Hide UD / IQ / BF16-F16 chips </div>
        <div class="filter-group">
          <label class="filter-label" for="iterations-input">Iterations per variant</label>
          <input type="number" id="iterations-input" class="filter-select" value="5" min="1" max="50">
        </div>
        <div class="filter-actions">
          <button class="btn btn-secondary" id="btn-download" disabled>Download selected</button>
          <button class="btn btn-primary" id="btn-run" disabled>Run benchmarks</button>
          <button class="btn btn-danger" id="btn-abort" hidden>Abort</button>
          <button class="btn btn-secondary" id="btn-purge" hidden>Purge OPFS cache</button>
        </div>
      </div>
    </div>

    <!-- Variant list: one .table-card per family with collapsible rows -->
    <div id="run-models" class="run-models-stack"></div>

    <!-- Progress table (same .table-card look as Results) -->
    <div class="section-header" style="margin-top: 32px;">
      <h3 class="subsection-title">Progress</h3>
    </div>
    <div class="table-card">
      <div id="run-progress-wrapper" class="results-wrapper"></div>
    </div>

    <!-- Output block -->
    <div class="section-header" style="margin-top: 32px;">
      <h3 class="subsection-title">Output</h3>
    </div>
    <div class="table-card run-output">
      <label id="save-local-row" class="run-output-toggle">
        <input type="checkbox" id="save-local" checked>
        Save to <code>results/results.json</code> on this server
      </label>
      <textarea id="output-textarea" readonly spellcheck="false" class="run-output-textarea"></textarea>
      <div class="run-output-buttons">
        <button class="btn btn-secondary" id="btn-copy">Copy</button>
        <button class="btn btn-secondary" id="btn-download-json">Download JSON</button>
      </div>
    </div>

    <!-- Log panel -->
    <details id="run-log" class="card run-log">
      <summary>Run log</summary>
      <pre id="log-output" class="run-log-pre"></pre>
    </details>
  </div>
</section>
```

### Phase 3 — Style mapping (`site/css/style.css`)

All classes use existing variables. Add a small run-specific block rather than rewriting `bench.css`. Key pieces:

- `.badge` — reuse existing; mode badge colors keyed off `--accent` / `--muted`
- `.hub-row` / `.hub-row-inner` / `.hub-row-actions` — flex row, inside a `.card`
- `.summary-grid.run-device-grid` — reuse `.summary-grid` (auto-fit grid of `.stat-card`s); the two Run-specific cards use the same `.stat-card` shell
- `.run-models-stack > details.family-card` — each family becomes a `.card` with a `<summary>` header
- `.variant-row` — a `.table-card` row mimic; use `font-variant-numeric: tabular-nums;` and reuse `.cell-code`-style spans for filenames
- `.cache-badge` / `.warn-badge` — reuse status badge styles (see existing `.status-pass` / `.status-fail` — add `.status-cache` and `.status-warn`)
- `.btn` / `.btn-primary` / `.btn-secondary` / `.btn-danger` — if the existing style.css doesn't name them this way, use whatever it already has (likely `.filter-reset-btn` pattern). Audit before coding.
- `#output-textarea` reuses `code`/`pre` token family + border-radius

Audit first: read `site/css/style.css` top-to-bottom and catalogue what exists. Reuse, don't recreate. The goal is a PR of maybe 200–300 added lines in style.css, not a rewrite.

### Phase 4 — Wire the controller into `site/js/app.js`

`site/js/app.js` currently loads data, wires section-nav, renders dashboard sections. Add:

```js
import { mountRunSection, teardownRunSection } from './run/controller.js';

// In the section-nav click handler:
if (activeSection === 'run-section' && !runMounted) {
  await mountRunSection();  // async because it fetches /api/models + describes device
  runMounted = true;
}
```

`site/js/run/controller.js` exports `mountRunSection()` that:
1. Detects `state.surface` (localhost / space / pages).
2. Loads models.json (prefers `/api/models`; falls back to `./models.json` on static hosts; for GH Pages falls back to `./data/models.json` if the build step copied it there).
3. Probes device (`describeDevice()`).
4. Loads cache inventory (local: `/api/cache-status`; hosted: `inventoryOpfs()`; pages: same as hosted).
5. Renders into `#run-models` + wires all the buttons, iterations input, hub auth, submit gate.

The inner loop (CPU baseline + N GPU iterations + consistency + aggregation + submit) is unchanged from `bench-app.js`; only the DOM selectors and class names change.

### Phase 5 — Build / deploy

1. `scripts/build-site.js` — read current implementation; ensure it copies / preserves the new `site/js/run/` tree. If it currently rewrites `site/data/combined.json` only, it already leaves the JS alone — likely a one-line audit.
2. `models.json` must be reachable from the static host. Two options:
   - **Recommended**: add to build-site.js a step that copies `models.json` to `site/models.json` so the hosted Run tab can `fetch('./models.json')`.
   - Alternative: keep models.json at root; have Run tab try `./models.json` first, then `/models.json`.
3. WASM paths: Run core loads `/build/${buildType}/bench.js`. On the HF Space these are served from `build/{jspi,asyncify}/bench.{js,wasm}` (already in place). On GH Pages the `build/` dir isn't published. So GH Pages Run tab has to either (a) show a "WASM not available on GH Pages — open the HF Space to run" banner, or (b) the build step copies WASM into `site/build/`. Decide at plan-review time.
4. `.github/workflows/sync-to-hf-space.yml` — stage `site/` as the Space root (rename `site/index.html` → `index.html` at the top of the Space, copy `site/js`, `site/css`, `site/data`, `site/models.json`, plus `harness.html`, `harness.js`, `build/{jspi,asyncify}/bench.{js,wasm}` if present, and `spaces/README.md` → `README.md`). The `--exclude='build/**'` + `--exclude='.gitattributes'` rules already in the workflow stay.
5. `spaces/README.md` frontmatter: set `app_file: index.html` (unchanged); optionally set `app_port` if needed.
6. `server.js` — add a one-line redirect from `/` and `/bench.html` to `/site/` (keep backwards compat, but the canonical entry is `http://localhost:3000/site/`).
7. `.github/workflows/deploy-dashboard.yml` — unchanged; it publishes `site/` as is.

### Phase 6 — Cleanup

Delete:
- `bench.html`
- `bench.css`
- root-level `bench-app.js`, `bench-core.js`, `bench-source.js`, `bench-device.js`, `bench-hub.js`, `bench-config.js` (all moved into `site/js/run/`)

Update `README.md`: the "One-click benchmark" section now points at `http://localhost:3000/site/` locally and the Space URL for hosted.

## Detection / mode behavior

```js
async function detectSurface() {
  const params = new URLSearchParams(location.search);
  if (params.get('mode') === 'local') return 'localhost';
  if (params.get('mode') === 'hosted') return 'space';
  if (/\.static\.hf\.space$/.test(location.hostname)) return 'space';
  if (/\.github\.io$/.test(location.hostname)) return 'pages';
  if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') {
    try {
      if ((await fetch('/api/models', { method: 'HEAD' })).ok) return 'localhost';
    } catch {}
  }
  return 'pages';  // conservative default — no backend, no OAuth
}
```

UI differences per surface:
- `localhost`: `#save-local-row` visible, `#hub-row` hidden.
- `space`: `#hub-row` visible, `#save-local-row` hidden. Sign-in + Submit enabled when OAuth variables are injected by HF.
- `pages`: `#hub-row` hidden (no OAuth), `#save-local-row` hidden. Run works read-only: benchmark locally, copy-paste output. Banner: "Publish to the leaderboard via the HF Space URL: …"

## Design system audit (do before coding)

Before writing any CSS, read `site/css/style.css` once end-to-end and write a one-paragraph note on the existing tokens (buttons, cards, badges, tables). This prevents redefining what already exists. Common gotchas the dashboard already solved:

- Theme switching via `data-theme="light|dark"` on `<html>` + a flicker-free `<head>` script — Run must not re-define its own palette.
- Font loading via Google Fonts (Manrope/JetBrains Mono).
- Chart.js loaded via CDN `<script>` in index.html, not via import.

## Runner.js / CLI impact

Runner.js still opens `http://localhost:3000/harness.html?model=…&hfRepo=…&…`. That page is a thin `harness.html` that imports `./site/js/run/core.js` and `./site/js/run/source.js`. The `window.__BENCH` contract is preserved exactly — `runner.js` doesn't know anything changed.

`scripts/submit-results.js` and cloud flow: unchanged. They only touch `results/results.json` + the dataset.

## Implementation order (safe, one commit per step)

1. **Move files into `site/js/run/`** — byte-for-byte move + import path updates. Smoke-test: open `http://localhost:3000/bench.html` (old entry) still works because `bench.html` now imports from the new paths. Runner.js still works.
2. **Style audit + minimal style.css additions** — add the Run-specific classes without touching bench.html yet. Verify no visual regressions in the existing dashboard.
3. **Add Run section to `site/index.html`** — new `<section>` + nav button + lazy mount from `site/js/app.js`. Existing sections untouched. Test locally at `http://localhost:3000/site/`.
4. **Surface detection + UI gating** — show/hide hub-row and save-local-row based on detectSurface(). Test on localhost.
5. **Deploy adjustments** — update `sync-to-hf-space.yml` to publish the whole `site/` tree (plus WASM) instead of the flat bundle. Push + verify the Space serves the merged app.
6. **Delete root bench.html / bench.css / bench-*.js** + final README update.

Between steps 1 and 6, the old bench.html and the new site/#run-section can coexist so you always have a fallback.

## Verification

- `http://localhost:3000/site/` → dashboard + Run tab both work. Run a tiny variant (gemma-3-270m-it Q2_K) through 5 iterations, save to results.json, `npm run submit` pushes to dataset.
- `https://abhijitramesh-webgpu-bench.static.hf.space/` → same UI; Run tab → Sign in with HF → submit direct commit to `abhijitramesh/webgpu-bench-leaderboard`.
- `https://abhijitramesh.github.io/webgpu-bench` → dashboard + Run tab; Run works read-only; Submit hidden; banner points at the HF Space for submission.
- `node runner.js --models=gemma-3-270m-it --variants=Q2_K --browsers=chromium --no-webgpu` → regression check (harness.html still drives the same core).

## Critical files to touch

- `site/index.html`  *(add nav item + Run section)*
- `site/css/style.css`  *(extend with run-specific classes — no palette changes)*
- `site/js/app.js`  *(lazy-mount Run controller)*
- `site/js/run/*.js`  *(new subdir; files moved from root bench-*.js + small DOM refactor)*
- `harness.js`  *(update import paths to site/js/run/)*
- `server.js`  *(optional `/` → `/site/` redirect)*
- `scripts/build-site.js`  *(copy models.json + optionally build/ into site/)*
- `.github/workflows/sync-to-hf-space.yml`  *(stage site/ as Space root)*
- `spaces/README.md`  *(unchanged or app_file tweak)*
- `README.md`  *(update URLs)*

## Open questions to answer at plan review

1. **Default active section on the HF Space**: should `Run` be the default landing tab on the Space (Space purpose is running), while `Overview` stays default on GH Pages (public is mostly readers)? Or identical everywhere for simplicity?
2. **WASM on GH Pages**: copy `build/` into `site/build/` at deploy time (so Run works read-only on GH Pages) or hide Run there entirely? Trade-off: site weight vs. feature.
3. **Local models.json access**: `/api/models` for local; for hosted, do we copy `models.json` → `site/models.json` at build, or move it permanently into `site/`? If moved, `config.js` / `runner.js` / `scripts/fill-sizes.mjs` update their paths.
4. **Submit gate & surface combo**: on GH Pages, should Run still compute results (for local copy-paste) even without Submit? Or should the whole Run tab be disabled?
5. **Theme**: Run section currently dark-only. Post-merge it inherits the dashboard's light/dark toggle. Verify every Run UI element renders well in light mode (stat-cards, progress table, badges).
6. **Section-nav overflow**: with a 6th tab, the nav track may need horizontal scroll on narrow screens. Test on iOS widths.

## Notes for the executing session

- Start from this document. Read `site/css/style.css` before touching any markup.
- Run `node server.js` during Phase 3–4; keep both `bench.html` (old) and `site/#run-section` (new) working until Phase 6.
- All `@huggingface/hub` browser imports go through the import map already in `bench.html`. When moving to `site/index.html`, port the `<script type="importmap">` block.
- The HF Space OAuth reads `window.huggingface.variables.OAUTH_CLIENT_ID` — HF injects this into the page served at the Space root. As long as `site/index.html` is served as the Space root, OAuth keeps working.
- Before deleting the old files in Phase 6, grep for any remaining references (`grep -rn "bench.html\|bench.css\|bench-app.js" .`).
