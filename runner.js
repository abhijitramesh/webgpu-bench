// Playwright orchestrator for WebGPU LLM benchmarks.
// Variant-first execution: downloads model once, tests all browsers, deletes cache.
// This minimises disk usage and avoids redundant downloads.
// Safari uses WebDriverIO (real Safari) to get actual WebGPU support.

import { chromium } from 'playwright';
import { remote } from 'webdriverio';
import { execSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { getConfig } from './config.js';
import { startServer, stopServer } from './server.js';
import { pushResultsToDataset } from './scripts/push-to-dataset.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const config = getConfig();

// Read llama.cpp submodule commit hash once at startup for result tracking
function getLlamaCppCommit() {
  if (process.env.LLAMA_CPP_COMMIT) return process.env.LLAMA_CPP_COMMIT;
  try {
    return execSync('git -C llama.cpp rev-parse HEAD', { encoding: 'utf-8' }).trim();
  } catch {
    return null;
  }
}
const LLAMA_CPP_COMMIT = getLlamaCppCommit();

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

  return common;
}

function getBrowserType(name) {
  switch (name) {
    case 'chromium': return chromium;
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

// Run a single benchmark via Playwright (chromium)
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
    try { await context.close(); } catch {}
  }
}

// Run a single benchmark in a WebDriverIO Safari session
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

// Build a result object from benchmark output
function makeResult(timestamp, browserName, variant, bench, nGpuLayers, wallTimeMs) {
  return {
    timestamp,
    browser: browserName,
    model: variant.modelName,
    repo: variant.repo,
    variant: variant.name,
    filename: variant.filename,
    sizeMB: variant.sizeMB,
    status: bench.status || 'error',
    error: bench.error || null,
    buildType: bench.buildType || 'unknown',
    webgpuAvailable: bench.webgpuAvailable || false,
    gpuAdapterInfo: bench.gpuAdapterInfo || null,
    nGpuLayers,
    nCtx: config.N_CTX,
    nPredict: config.N_PREDICT,
    wallTimeMs,
    metrics: bench.metrics || null,
    output: (bench.output || '').substring(0, 200),
    machine: config.MACHINE,
    consistency: bench.consistency ?? null,
    llamaCppCommit: LLAMA_CPP_COMMIT,
  };
}

// Managed safaridriver lifecycle.  WebDriverIO's internal safaridriver management
// doesn't clean up reliably between sessions (port conflicts even after deleteSession).
// We start safaridriver manually on a fixed port, giving us full control.
const SAFARI_PORT = 4444;

function killSafariDriver() {
  try { execSync('killall -9 safaridriver 2>/dev/null', { stdio: 'ignore' }); } catch {}
}

async function startSafariDriver() {
  killSafariDriver();
  await new Promise(resolve => setTimeout(resolve, 1000));
  const proc = spawn('safaridriver', ['--port', String(SAFARI_PORT)], {
    stdio: 'ignore', detached: true,
  });
  proc.unref();
  await new Promise(resolve => setTimeout(resolve, 1000));
  return proc;
}

async function createSafariSession() {
  return remote({
    capabilities: { browserName: 'safari' },
    hostname: 'localhost',
    port: SAFARI_PORT,
    logLevel: 'error',
  });
}

// Delete a cached model file to free disk space
function deleteModelCache(variant) {
  const cachePath = path.join(__dirname, 'cache', 'models', variant.repo, variant.filename);
  try {
    if (fs.existsSync(cachePath)) {
      fs.unlinkSync(cachePath);
    }
  } catch {}
}

// Run a variant in a Playwright browser (chromium) with a hard-kill timeout.
// Launches a fresh browser per variant to prevent WASM memory accumulation.
// If the browser hangs past the soft timeout, SIGKILL ensures we don't wait forever.
async function runVariantPlaywright(browserName, variant, serverUrl, timestamp, nGpuLayers, refTokenIds) {
  const browserType = getBrowserType(browserName);
  const launchOpts = getBrowserLaunchArgs(browserName);

  let browser;
  try {
    browser = await browserType.launch(launchOpts);
  } catch (err) {
    return makeResult(timestamp, browserName, variant, {
      status: 'error', error: `Browser launch failed: ${err.message}`,
    }, nGpuLayers, 0);
  }

  const startTime = Date.now();
  let hardKilled = false;

  // Hard timeout: SIGKILL the browser process if it hangs past the soft timeout.
  // This prevents hung browser processes from blocking the entire run for hours.
  const hardTimer = setTimeout(() => {
    hardKilled = true;
    const proc = browser.process();
    if (proc) {
      try { proc.kill('SIGKILL'); } catch {}
    }
  }, config.TIMEOUTS.total + 5000);

  try {
    const { bench } = await runBenchmark(browser, variant, serverUrl, nGpuLayers, refTokenIds);
    clearTimeout(hardTimer);
    const wallTimeMs = Date.now() - startTime;
    return makeResult(timestamp, browserName, variant, bench, nGpuLayers, wallTimeMs);
  } catch (err) {
    clearTimeout(hardTimer);
    const wallTimeMs = Date.now() - startTime;
    return makeResult(timestamp, browserName, variant, {
      status: 'error',
      error: hardKilled ? 'Hard timeout — browser killed' : `Runner error: ${err.message}`,
    }, nGpuLayers, wallTimeMs);
  } finally {
    clearTimeout(hardTimer);
    // Graceful close with a short deadline — don't let a dead browser block us
    try {
      await Promise.race([
        browser.close(),
        new Promise(resolve => setTimeout(resolve, 5000)),
      ]);
    } catch {}
  }
}

// Run a variant in Safari/WebKit with a hard-kill timeout.
// Starts a fresh safaridriver + session per variant so one crash never cascade-fails.
async function runVariantSafari(variant, serverUrl, timestamp, nGpuLayers, refTokenIds) {
  let driverProc;
  let session;
  try {
    driverProc = await startSafariDriver();
    session = await createSafariSession();
  } catch (err) {
    killSafariDriver();
    return makeResult(timestamp, 'webkit', variant, {
      status: 'error', error: `Safari launch failed: ${err.message}`,
    }, nGpuLayers, 0);
  }

  const startTime = Date.now();
  let hardKilled = false;

  const hardTimer = setTimeout(() => {
    hardKilled = true;
    killSafariDriver();
  }, config.TIMEOUTS.total + 5000);

  try {
    const { bench } = await runBenchmarkSafari(session, variant, serverUrl, nGpuLayers, refTokenIds);
    clearTimeout(hardTimer);
    const wallTimeMs = Date.now() - startTime;
    return makeResult(timestamp, 'webkit', variant, bench, nGpuLayers, wallTimeMs);
  } catch (err) {
    clearTimeout(hardTimer);
    const wallTimeMs = Date.now() - startTime;
    return makeResult(timestamp, 'webkit', variant, {
      status: 'error',
      error: hardKilled ? 'Hard timeout — Safari killed' : `Runner error: ${err.message}`,
    }, nGpuLayers, wallTimeMs);
  } finally {
    clearTimeout(hardTimer);
    try { await session.deleteSession(); } catch {}
    try { driverProc.kill('SIGTERM'); } catch {}
    killSafariDriver();
  }
}

// Dispatch a variant run to the appropriate browser driver
async function runVariantInBrowser(browserName, variant, serverUrl, timestamp, nGpuLayers, refTokenIds) {
  if (browserName === 'webkit') {
    return runVariantSafari(variant, serverUrl, timestamp, nGpuLayers, refTokenIds);
  }
  return runVariantPlaywright(browserName, variant, serverUrl, timestamp, nGpuLayers, refTokenIds);
}

// Main — variant-first loop: download model → CPU baseline → all browsers GPU → delete cache
async function main() {
  console.log('=== WebGPU LLM Benchmark Runner ===');
  console.log(`Browsers: ${config.BROWSERS.join(', ')}`);
  console.log(`Variants: ${config.MODEL_VARIANTS.length} models`);
  console.log(`Machine:  ${config.MACHINE.platform}/${config.MACHINE.arch} - ${config.MACHINE.cpus}`);
  console.log(`GPU layers: ${config.N_GPU_LAYERS}`);
  if (LLAMA_CPP_COMMIT) console.log(`llama.cpp:  ${LLAMA_CPP_COMMIT.slice(0, 10)}`);
  if (config.NO_CACHE) console.log('Cache: OFF (models will be downloaded fresh each run)');
  if (config.CONSISTENCY) console.log('Consistency mode: ON (CPU baseline + GPU per variant)');
  if (config.RESUME) console.log('Resume mode: ON (skipping already-succeeded benchmarks)');
  console.log('Execution: variant-first (download → all browsers → delete cache)');
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

  // Filter browsers: skip webkit on non-macOS
  const activeBrowsers = config.BROWSERS.filter(b => {
    if (b === 'webkit' && os.platform() !== 'darwin') {
      console.log('Skipping WebKit — Safari is only available on macOS');
      return false;
    }
    return true;
  });

  // Variant-first loop: each model is downloaded once, tested across all
  // browsers, then its cache entry is deleted to free disk space.
  for (let i = 0; i < config.MODEL_VARIANTS.length; i++) {
    const variant = config.MODEL_VARIANTS[i];
    console.log(`\n[${i + 1}/${config.MODEL_VARIANTS.length}] ${variant.modelName} / ${variant.name} (${variant.sizeMB} MB)`);

    // Phase 1: CPU baseline (if --consistency and not yet cached)
    if (config.CONSISTENCY) {
      const baselineKey = activeBrowsers.length > 1 ? 'cpu' : activeBrowsers[0];
      if (!cpuBaselines[baselineKey]) cpuBaselines[baselineKey] = {};

      if (!(variant.filename in cpuBaselines[baselineKey])) {
        // Use first Playwright-compatible browser for CPU baseline
        const blBrowser = activeBrowsers.find(b => b !== 'webkit') || activeBrowsers[0];
        process.stdout.write(`  cpu baseline (${blBrowser})... `);

        const cpuResult = await runVariantInBrowser(
          blBrowser, variant, serverUrl, timestamp, 0, null
        );

        if (cpuResult.status === 'done' && cpuResult.metrics?.token_ids?.length > 0) {
          cpuBaselines[baselineKey][variant.filename] = cpuResult.metrics.token_ids;
          const m = cpuResult.metrics;
          console.log(`OK | prefill: ${m.prefill_tok_s} tok/s | decode: ${m.decode_tok_s} tok/s | wall: ${(cpuResult.wallTimeMs / 1000).toFixed(1)}s`);
        } else {
          cpuBaselines[baselineKey][variant.filename] = null; // failed — don't retry
          console.log(`FAIL (${cpuResult.error || 'unknown'})`);
        }

        allResults.push(cpuResult);
        saveCpuBaselines(cpuBaselines);

        // Save intermediate results (crash resilience)
        fs.writeFileSync(path.join(config.RESULTS_DIR, 'results.json'), JSON.stringify(allResults, null, 2));
      } else {
        console.log('  cpu baseline: cached');
      }
    }

    // Phase 2: GPU (or configured nGpuLayers) run across all browsers
    for (const browserName of activeBrowsers) {
      // Resume: skip already-succeeded benchmarks
      if (config.RESUME && alreadyCompleted(previousResults, browserName, variant.filename, config.N_GPU_LAYERS)) {
        console.log(`  ${browserName}: skipped (already done)`);
        continue;
      }

      process.stdout.write(`  ${browserName}... `);

      const refTokenIds = config.CONSISTENCY
        ? tokenIdsToCsv(getBaselineTokenIds(cpuBaselines, browserName, variant.filename))
        : null;

      const result = await runVariantInBrowser(
        browserName, variant, serverUrl, timestamp, config.N_GPU_LAYERS, refTokenIds
      );

      allResults.push(result);

      if (result.status === 'done' && result.metrics) {
        const m = result.metrics;
        const consistencyLabel = config.CONSISTENCY ? ` | ${formatConsistency(result.consistency)}` : '';
        console.log(`OK | prefill: ${m.prefill_tok_s} tok/s | decode: ${m.decode_tok_s} tok/s | wall: ${(result.wallTimeMs / 1000).toFixed(1)}s${consistencyLabel}`);
      } else {
        console.log(`FAIL | ${result.error || 'unknown error'}`);
      }

      // Save intermediate results (crash resilience)
      fs.writeFileSync(path.join(config.RESULTS_DIR, 'results.json'), JSON.stringify(allResults, null, 2));
    }

    // Phase 3: Delete cached model to free disk space
    deleteModelCache(variant);
  }

  // Final save (local cache for retry / inspection).
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

  // Auto-push to the HF dataset — the dashboard reads from there as its
  // single source of truth. Without HF_TOKEN/HF_DATASET_REPO we leave the
  // local results.json in place and surface a hint so the dev can push
  // later via `npm run submit`.
  if (process.env.HF_TOKEN && process.env.HF_DATASET_REPO) {
    try {
      const submittable = allResults.filter(r => r.status === 'done');
      if (submittable.length > 0) {
        console.log('\nPushing to HF dataset…');
        const { uploads } = await pushResultsToDataset({
          datasetRepo: process.env.HF_DATASET_REPO,
          token: process.env.HF_TOKEN,
          records: submittable,
        });
        console.log(`Pushed ${uploads} file${uploads === 1 ? '' : 's'} to https://huggingface.co/datasets/${process.env.HF_DATASET_REPO}`);
      } else {
        console.log('\nNo successful runs to push.');
      }
    } catch (err) {
      console.warn(`\nPush to HF dataset failed: ${err.message}`);
      console.warn(`Local results preserved at ${resultsFile}; retry with: npm run submit`);
    }
  } else {
    console.log('\nSkipping HF dataset push (HF_TOKEN/HF_DATASET_REPO not set). Run `npm run submit` to push later.');
  }

  await stopServer(server);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
