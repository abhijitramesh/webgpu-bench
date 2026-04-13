// Playwright orchestrator for WebGPU LLM benchmarks.
// Runs each model variant in each browser, collects metrics.
// Safari uses WebDriverIO (real Safari) to get actual WebGPU support.

import { chromium, firefox } from 'playwright';
import { remote } from 'webdriverio';
import fs from 'node:fs';
import path from 'node:path';
import { getConfig } from './config.js';
import { startServer, stopServer } from './server.js';

const config = getConfig();

// Browser launch args
function getBrowserLaunchArgs(browserName) {
  const common = { headless: false };

  if (browserName === 'chromium') {
    const args = [
      '--enable-features=Vulkan,WebGPU',
      '--enable-unsafe-webgpu',
    ];
    // Platform-specific GPU backend
    if (config.MACHINE.platform === 'darwin') {
      args.push('--use-angle=metal');
    } else {
      args.push('--use-angle=vulkan');
    }
    return { ...common, args };
  }

  // Firefox: no special args needed
  return common;
}

function getBrowserType(name) {
  switch (name) {
    case 'chromium': return chromium;
    case 'firefox':  return firefox;
    default: throw new Error(`Unknown browser: ${name}`);
  }
}

// Collect GPU info from a browser page
async function getGpuInfo(page) {
  try {
    return await page.evaluate(async () => {
      if (!navigator.gpu) return null;
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { available: false };
      return {
        available: true,
        info: adapter.info || null,
      };
    });
  } catch {
    return null;
  }
}

function buildHarnessUrl(serverUrl, variant, nGpuLayers, refTokenIds = null) {
  const harnessUrl = new URL('/harness.html', serverUrl);
  harnessUrl.searchParams.set('model', variant.filename);
  harnessUrl.searchParams.set('hfRepo', variant.repo);
  harnessUrl.searchParams.set('prompt', config.PROMPT);
  harnessUrl.searchParams.set('nPredict', String(config.N_PREDICT));
  harnessUrl.searchParams.set('nCtx', String(config.N_CTX));
  harnessUrl.searchParams.set('nGpuLayers', String(nGpuLayers));
  if (refTokenIds) harnessUrl.searchParams.set('refTokenIds', refTokenIds);
  return harnessUrl.toString();
}

function loadCpuBaselines() {
  const file = path.join(config.RESULTS_DIR, 'cpu_baselines.json');
  if (fs.existsSync(file)) {
    return JSON.parse(fs.readFileSync(file, 'utf-8'));
  }
  return {};
}

function saveCpuBaselines(baselines) {
  const file = path.join(config.RESULTS_DIR, 'cpu_baselines.json');
  fs.writeFileSync(file, JSON.stringify(baselines, null, 2));
}

// Convert stored token ID array to the CSV string expected by bench_eval_tokens
function tokenIdsToCsv(tokenIds) {
  return Array.isArray(tokenIds) ? tokenIds.join(',') : null;
}

// Look up CPU baseline for a variant. Uses shared "cpu" key when running
// multiple browsers, falls back to browser-specific key for backwards compat.
function getBaselineTokenIds(baselines, browserName, filename) {
  const sharedKey = config.BROWSERS.length > 1 ? 'cpu' : browserName;
  return baselines[sharedKey]?.[filename] ?? baselines[browserName]?.[filename] ?? null;
}

// Run a single benchmark via Playwright (chromium, firefox)
async function runBenchmark(browser, variant, serverUrl, nGpuLayers = config.N_GPU_LAYERS, refTokenIds = null) {
  const context = await browser.newContext();
  const page = await context.newPage();

  // Capture console output
  const consoleLogs = [];
  page.on('console', msg => {
    consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
  });

  const harnessUrl = buildHarnessUrl(serverUrl, variant, nGpuLayers, refTokenIds);

  try {
    await page.goto(harnessUrl, { timeout: 30_000 });

    // Wait for completion or error
    // Note: waitForFunction(fn, arg, options) - arg must be null when not needed
    await page.waitForFunction(
      () => window.__BENCH && (window.__BENCH.status === 'done' || window.__BENCH.status === 'error'),
      null,
      { timeout: config.TIMEOUTS.total }
    );

    const bench = await page.evaluate(() => window.__BENCH);
    return { bench, consoleLogs };
  } catch (err) {
    // Timeout or navigation error
    let bench;
    try {
      bench = await page.evaluate(() => window.__BENCH);
    } catch {
      bench = { status: 'error', error: 'Page became unresponsive' };
    }
    return {
      bench: {
        ...bench,
        status: 'error',
        error: bench?.error || `Runner error: ${err.message}`,
      },
      consoleLogs,
    };
  } finally {
    await context.close();
  }
}

// Run a single benchmark in an existing WebDriverIO Safari session
async function runBenchmarkSafari(browser, variant, serverUrl, nGpuLayers = config.N_GPU_LAYERS, refTokenIds = null) {
  const harnessUrl = buildHarnessUrl(serverUrl, variant, nGpuLayers, refTokenIds);

  try {
    await browser.url(harnessUrl);

    await browser.waitUntil(
      async () => {
        const status = await browser.execute(() => window.__BENCH?.status ?? null);
        return status === 'done' || status === 'error';
      },
      { timeout: config.TIMEOUTS.total, interval: 2000 }
    );

    const bench = await browser.execute(() => window.__BENCH);
    return { bench, consoleLogs: [] };
  } catch (err) {
    let bench;
    try {
      bench = await browser.execute(() => window.__BENCH);
    } catch {
      bench = { status: 'error', error: 'Page became unresponsive' };
    }
    return {
      bench: {
        ...bench,
        status: 'error',
        error: bench?.error || `Runner error: ${err.message}`,
      },
      consoleLogs: [],
    };
  }
}

// Format consistency result for console output
function formatConsistency(c) {
  if (!c) return 'no baseline';
  const pct = (c.agreement_rate * 100).toFixed(1);
  if (c.agreement_rate === 1.0) return `100% top-1 agreement`;
  return `${pct}% top-1 agreement (first diverge @ token ${c.first_disagreement})`;
}

// Collect CPU baselines (n_gpu_layers=0) for any variants missing from the cache.
// Stores token IDs from the CPU run — these are the reference sequence for forced eval.
// runFn: (variant, nGpuLayers, refTokenIds) => { bench }
// Mutates the baselines map and saves to disk after each run.
//
// When running multiple browsers, baselines are shared: the first browser collects
// them and subsequent browsers reuse the same reference sequences (CPU output is
// identical regardless of JSPI vs Asyncify — only the async wrapper differs).
// When running a single browser, behaviour is unchanged.
async function collectMissingBaselines(browserName, variants, runFn, baselines) {
  // Use shared "cpu" key when multiple browsers are being tested,
  // so baselines are collected once and reused across all browsers.
  const baselineKey = config.BROWSERS.length > 1 ? 'cpu' : browserName;

  if (!baselines[baselineKey]) baselines[baselineKey] = {};
  const sharedBaselines = baselines[baselineKey];

  const missing = variants.filter(v => !(v.filename in sharedBaselines));
  if (missing.length === 0) {
    const source = baselineKey === 'cpu' ? 'shared' : browserName;
    console.log(`  CPU baselines: all ${variants.length} cached (${source})`);
    return;
  }

  console.log(`  CPU baselines: collecting ${missing.length} (${variants.length - missing.length} cached)`);

  for (let i = 0; i < missing.length; i++) {
    const variant = missing[i];
    process.stdout.write(`    [${i + 1}/${missing.length}] ${variant.name} (CPU)... `);

    const { bench } = await runFn(variant, 0);

    if (bench.status === 'done' && bench.metrics?.token_ids?.length > 0) {
      // Store token IDs as the reference for forced-decoding consistency check
      sharedBaselines[variant.filename] = bench.metrics.token_ids;
      console.log(`OK (${bench.metrics.token_ids.length} tokens)`);
    } else {
      sharedBaselines[variant.filename] = null; // failed — record so we don't retry
      console.log(`FAIL (${bench.error || 'unknown'})`);
    }

    saveCpuBaselines(baselines);
  }
}

// Load previous results for resume mode
function loadPreviousResults() {
  const file = path.join(config.RESULTS_DIR, 'results.json');
  if (fs.existsSync(file)) {
    try {
      return JSON.parse(fs.readFileSync(file, 'utf-8'));
    } catch {
      return [];
    }
  }
  return [];
}

// Check if a browser+variant+gpu_layers combo already has a successful result
function alreadyCompleted(previousResults, browserName, filename, nGpuLayers) {
  return previousResults.some(
    r =>
      r.browser === browserName &&
      r.filename === filename &&
      r.nGpuLayers === nGpuLayers &&
      r.status === 'done'
  );
}

// Main
async function main() {
  console.log('=== WebGPU LLM Benchmark Runner ===');
  console.log(`Browsers: ${config.BROWSERS.join(', ')}`);
  console.log(`Variants: ${config.MODEL_VARIANTS.length} models`);
  console.log(`Machine:  ${config.MACHINE.platform}/${config.MACHINE.arch} - ${config.MACHINE.cpus}`);
  console.log(`GPU layers: ${config.N_GPU_LAYERS}`);
  if (config.NO_CACHE) console.log('Cache: OFF (models will be downloaded fresh each run)');
  if (config.CONSISTENCY) console.log('Consistency mode: ON (CPU baselines will be collected per browser)');
  if (config.RESUME) console.log('Resume mode: ON (skipping already-succeeded benchmarks)');
  console.log('');

  // Ensure results dir exists
  fs.mkdirSync(config.RESULTS_DIR, { recursive: true });

  // Start server
  const { server, url: serverUrl } = await startServer(config.PORT, { noCache: config.NO_CACHE });
  console.log(`Server: ${serverUrl}`);

  // Load persisted CPU baselines (allows resuming after crash)
  const cpuBaselines = config.CONSISTENCY ? loadCpuBaselines() : {};

  // Resume mode: load previous results, keep only successful ones, re-run failures
  const previousResults = config.RESUME ? loadPreviousResults() : [];
  const allResults = config.RESUME
    ? previousResults.filter(r => r.status === 'done')  // drop failed results so they get retried
    : [];
  if (config.RESUME && previousResults.length > 0) {
    const doneCount = allResults.length;
    const failedCount = previousResults.length - doneCount;
    console.log(`Resuming: ${doneCount} succeeded (kept), ${failedCount} failed (will retry)`);
  }
  const timestamp = new Date().toISOString();

  for (const browserName of config.BROWSERS) {
    console.log(`\n--- Browser: ${browserName} ---`);

    if (browserName === 'webkit') {
      // Real Safari via WebDriverIO — supports WebGPU natively on macOS.
      // Requires: Safari > Settings > Advanced > "Allow Remote Automation"
      //
      // Safari sessions die when the page crashes (GPU OOM on large models).
      // Unlike Playwright browsers, WebDriverIO can't launch per-variant easily,
      // so we detect dead sessions and restart them.

      async function createSafariSession() {
        return remote({ capabilities: { browserName: 'safari' }, logLevel: 'error' });
      }

      let safariSession;
      try {
        safariSession = await createSafariSession();
      } catch (err) {
        console.error(`  Failed to launch Safari: ${err.message}`);
        continue;
      }

      if (config.CONSISTENCY) {
        await collectMissingBaselines(
          browserName,
          config.MODEL_VARIANTS,
          async (variant, nGpuLayers) => {
            try {
              return await runBenchmarkSafari(safariSession, variant, serverUrl, nGpuLayers);
            } catch {
              // Session died during baseline collection — restart and retry once
              console.log('    (restarting Safari session...)');
              try { await safariSession.deleteSession(); } catch { /* already dead */ }
              safariSession = await createSafariSession();
              return runBenchmarkSafari(safariSession, variant, serverUrl, nGpuLayers);
            }
          },
          cpuBaselines,
        );
      }

      for (let i = 0; i < config.MODEL_VARIANTS.length; i++) {
        const variant = config.MODEL_VARIANTS[i];
        const progress = `[${i + 1}/${config.MODEL_VARIANTS.length}]`;

        // Resume: skip already-succeeded benchmarks
        if (config.RESUME && alreadyCompleted(previousResults, browserName, variant.filename, config.N_GPU_LAYERS)) {
          console.log(`  ${progress} ${variant.name} — skipped (already done)`);
          continue;
        }

        console.log(`  ${progress} ${variant.name} (${variant.sizeMB} MB)...`);

        const refTokenIds = config.CONSISTENCY
          ? tokenIdsToCsv(getBaselineTokenIds(cpuBaselines, browserName, variant.filename))
          : null;

        const startTime = Date.now();
        let { bench } = await runBenchmarkSafari(safariSession, variant, serverUrl, config.N_GPU_LAYERS, refTokenIds);
        const wallTimeMs = Date.now() - startTime;

        // If the page became unresponsive, the session is dead — restart it
        // so subsequent variants don't all fail with "invalid session id".
        if (bench.status === 'error' && bench.error === 'Page became unresponsive') {
          console.log('    (restarting Safari session for next variant...)');
          try { await safariSession.deleteSession(); } catch { /* already dead */ }
          try {
            safariSession = await createSafariSession();
          } catch (err) {
            console.error(`    Failed to restart Safari: ${err.message}`);
            // Record remaining variants as errors and break
            for (let j = i + 1; j < config.MODEL_VARIANTS.length; j++) {
              const v = config.MODEL_VARIANTS[j];
              allResults.push({
                timestamp, browser: browserName, model: v.modelName, repo: v.repo,
                variant: v.name, filename: v.filename, sizeMB: v.sizeMB,
                status: 'error', error: 'Safari session lost and could not restart',
                buildType: 'unknown', webgpuAvailable: false, gpuAdapterInfo: null,
                nGpuLayers: config.N_GPU_LAYERS, nCtx: config.N_CTX, nPredict: config.N_PREDICT,
                wallTimeMs: 0, metrics: null, output: '', machine: config.MACHINE, consistency: null,
              });
            }
            const intermediateFile = path.join(config.RESULTS_DIR, 'results.json');
            fs.writeFileSync(intermediateFile, JSON.stringify(allResults, null, 2));
            break;
          }
        }

        // consistency is set by harness.js via bench_eval_tokens (forced decoding)
        const consistency = bench.consistency ?? null;

        const result = {
          timestamp,
          browser: browserName,
          model: variant.modelName,
          repo: variant.repo,
          variant: variant.name,
          filename: variant.filename,
          sizeMB: variant.sizeMB,
          status: bench.status,
          error: bench.error || null,
          buildType: bench.buildType || 'unknown',
          webgpuAvailable: bench.webgpuAvailable || false,
          gpuAdapterInfo: bench.gpuAdapterInfo || null,
          nGpuLayers: config.N_GPU_LAYERS,
          nCtx: config.N_CTX,
          nPredict: config.N_PREDICT,
          wallTimeMs,
          metrics: bench.metrics || null,
          output: (bench.output || '').substring(0, 200),
          machine: config.MACHINE,
          consistency,
        };

        allResults.push(result);

        if (bench.status === 'done' && bench.metrics) {
          const m = bench.metrics;
          const consistencyLabel = config.CONSISTENCY ? ` | ${formatConsistency(consistency)}` : '';
          console.log(`    OK | prefill: ${m.prefill_tok_s} tok/s | decode: ${m.decode_tok_s} tok/s | wall: ${(wallTimeMs / 1000).toFixed(1)}s${consistencyLabel}`);
        } else {
          console.log(`    FAIL | ${bench.error || 'unknown error'}`);
        }

        const intermediateFile = path.join(config.RESULTS_DIR, 'results.json');
        fs.writeFileSync(intermediateFile, JSON.stringify(allResults, null, 2));
      }
      try { await safariSession.deleteSession(); } catch { /* may already be dead */ }
      continue;
    }

    // Chromium / Firefox via Playwright
    // Launch a fresh browser for each variant to prevent WASM memory from
    // accumulating across runs (Firefox is especially prone to OOM otherwise).
    const browserType = getBrowserType(browserName);
    const launchOpts = getBrowserLaunchArgs(browserName);

    if (config.CONSISTENCY) {
      // Skip browser launch if all baselines are already cached (shared across browsers)
      const baselineKey = config.BROWSERS.length > 1 ? 'cpu' : browserName;
      const cached = cpuBaselines[baselineKey] || {};
      const needsCollection = config.MODEL_VARIANTS.some(v => !(v.filename in cached));

      if (needsCollection) {
        let baselineBrowser;
        try {
          baselineBrowser = await browserType.launch(launchOpts);
        } catch (err) {
          console.error(`Failed to launch ${browserName} for baselines: ${err.message}`);
          continue;
        }
        await collectMissingBaselines(
          browserName,
          config.MODEL_VARIANTS,
          (variant, nGpuLayers) => runBenchmark(baselineBrowser, variant, serverUrl, nGpuLayers),
          cpuBaselines,
        );
        await baselineBrowser.close();
      } else {
        console.log(`  CPU baselines: all ${config.MODEL_VARIANTS.length} cached (shared)`);
      }
    }

    let gpuInfo = null;

    for (let i = 0; i < config.MODEL_VARIANTS.length; i++) {
      const variant = config.MODEL_VARIANTS[i];
      const progress = `[${i + 1}/${config.MODEL_VARIANTS.length}]`;

      // Resume: skip already-succeeded benchmarks
      if (config.RESUME && alreadyCompleted(previousResults, browserName, variant.filename, config.N_GPU_LAYERS)) {
        console.log(`  ${progress} ${variant.name} — skipped (already done)`);
        continue;
      }

      console.log(`  ${progress} ${variant.name} (${variant.sizeMB} MB)...`);

      // Launch a fresh browser for each variant to avoid OOM from WASM memory leaks
      let browser;
      try {
        browser = await browserType.launch(launchOpts);
      } catch (err) {
        console.error(`    Failed to launch ${browserName}: ${err.message}`);
        const result = {
          timestamp,
          browser: browserName,
          model: variant.modelName,
          repo: variant.repo,
          variant: variant.name,
          filename: variant.filename,
          sizeMB: variant.sizeMB,
          status: 'error',
          error: `Browser launch failed: ${err.message}`,
          buildType: 'unknown',
          webgpuAvailable: false,
          gpuAdapterInfo: gpuInfo || null,
          nGpuLayers: config.N_GPU_LAYERS,
          nCtx: config.N_CTX,
          nPredict: config.N_PREDICT,
          wallTimeMs: 0,
          metrics: null,
          output: '',
          machine: config.MACHINE,
          consistency: null,
        };
        allResults.push(result);
        const intermediateFile = path.join(config.RESULTS_DIR, 'results.json');
        fs.writeFileSync(intermediateFile, JSON.stringify(allResults, null, 2));
        continue;
      }

      const refTokenIds = config.CONSISTENCY
        ? tokenIdsToCsv(getBaselineTokenIds(cpuBaselines, browserName, variant.filename))
        : null;

      const startTime = Date.now();
      const { bench } = await runBenchmark(browser, variant, serverUrl, config.N_GPU_LAYERS, refTokenIds);
      const wallTimeMs = Date.now() - startTime;

      // Close browser immediately to free WASM memory before next variant
      await browser.close();

      // consistency is set by harness.js via bench_eval_tokens (forced decoding)
      const consistency = bench.consistency ?? null;

      const result = {
        timestamp,
        browser: browserName,
        model: variant.modelName,
        repo: variant.repo,
        variant: variant.name,
        filename: variant.filename,
        sizeMB: variant.sizeMB,
        status: bench.status,
        error: bench.error || null,
        buildType: bench.buildType || 'unknown',
        webgpuAvailable: bench.webgpuAvailable || false,
        gpuAdapterInfo: bench.gpuAdapterInfo || gpuInfo || null,
        nGpuLayers: config.N_GPU_LAYERS,
        nCtx: config.N_CTX,
        nPredict: config.N_PREDICT,
        wallTimeMs,
        metrics: bench.metrics || null,
        output: (bench.output || '').substring(0, 200),
        machine: config.MACHINE,
        consistency,
      };

      allResults.push(result);

      if (!gpuInfo && bench.gpuAdapterInfo) gpuInfo = bench.gpuAdapterInfo;

      if (bench.status === 'done' && bench.metrics) {
        const m = bench.metrics;
        const consistencyLabel = config.CONSISTENCY ? ` | ${formatConsistency(consistency)}` : '';
        console.log(`    OK | prefill: ${m.prefill_tok_s} tok/s | decode: ${m.decode_tok_s} tok/s | wall: ${(wallTimeMs / 1000).toFixed(1)}s${consistencyLabel}`);
      } else {
        console.log(`    FAIL | ${bench.error || 'unknown error'}`);
      }

      // Save intermediate results (crash resilience)
      const intermediateFile = path.join(config.RESULTS_DIR, 'results.json');
      fs.writeFileSync(intermediateFile, JSON.stringify(allResults, null, 2));
    }
  }

  // Final save
  const resultsFile = path.join(config.RESULTS_DIR, 'results.json');
  fs.writeFileSync(resultsFile, JSON.stringify(allResults, null, 2));
  console.log(`\nResults saved to ${resultsFile}`);
  console.log(`Total: ${allResults.length} benchmarks (${allResults.filter(r => r.status === 'done').length} passed, ${allResults.filter(r => r.status === 'error').length} failed)`);

  if (config.CONSISTENCY) {
    const withConsistency = allResults.filter(r => r.consistency);
    const perfect = withConsistency.filter(r => r.consistency.agreement_rate === 1.0).length;
    const partial = withConsistency.filter(r => r.consistency.agreement_rate < 1.0).length;
    const noBaseline = allResults.filter(r => r.status === 'done' && !r.consistency).length;
    console.log(`Consistency: ${perfect} perfect (100%), ${partial} partial, ${noBaseline} no baseline`);
  }

  await stopServer(server);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
