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

// Build the URL the headless browser navigates to. `mode` selects which
// phases to run inside one model load:
//   'consistency' — only bench_run + bench_eval_tokens
//   'perf'        — only bench_pp + bench_tg
//   'both'        — both, in one model load (default for the GPU pass)
// `nRepsOverride` lets the caller pin a non-default rep count for this URL
// (used for the CPU baseline, which runs a single warmup+timed rep — enough
// for a sanity-check CPU/GPU comparison without a full sweep on CPU).
function buildHarnessUrl(serverUrl, variant, nGpuLayers, refTokenIds = null, mode = 'both', nRepsOverride = null) {
  const harnessUrl = new URL('/harness.html', serverUrl);
  harnessUrl.searchParams.set('model', variant.filename);
  harnessUrl.searchParams.set('hfRepo', variant.repo);
  harnessUrl.searchParams.set('nPredict', String(config.N_PREDICT));
  harnessUrl.searchParams.set('nPrompt', String(config.N_PROMPT));
  harnessUrl.searchParams.set('nGen',    String(config.N_GEN));
  harnessUrl.searchParams.set('nReps',   String(nRepsOverride ?? config.N_REPS));
  harnessUrl.searchParams.set('nCtx', String(config.N_CTX));
  harnessUrl.searchParams.set('nGpuLayers', String(nGpuLayers));
  harnessUrl.searchParams.set('mode', mode);
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
async function runBenchmark(browser, variant, serverUrl, nGpuLayers = config.N_GPU_LAYERS, refTokenIds = null, mode = 'both', nRepsOverride = null) {
  const context = await browser.newContext();
  const page = await context.newPage();

  // Capture console output
  const consoleLogs = [];
  page.on('console', msg => {
    consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
  });

  const harnessUrl = buildHarnessUrl(serverUrl, variant, nGpuLayers, refTokenIds, mode, nRepsOverride);

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
async function runBenchmarkSafari(browser, variant, serverUrl, nGpuLayers = config.N_GPU_LAYERS, refTokenIds = null, mode = 'both', nRepsOverride = null) {
  const harnessUrl = buildHarnessUrl(serverUrl, variant, nGpuLayers, refTokenIds, mode, nRepsOverride);

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

// Backward-compat: surface flat prefill_tok_s / decode_tok_s alongside the
// new metrics.tests array, so existing dashboard renderers keep working
// while we migrate them to read from tests directly.
function flattenMetrics(metrics) {
  if (!metrics) return null;
  const tests = Array.isArray(metrics.tests) ? metrics.tests : null;
  if (!tests) return metrics;
  const pp = tests.find(t => t.name?.startsWith('pp'));
  const tg = tests.find(t => t.name?.startsWith('tg'));
  return {
    ...metrics,
    iterations: metrics.n_reps || tests[0]?.samples_ns?.length || 0,
    prefill_tok_s: pp ? pp.avg_ts : 0,
    decode_tok_s:  tg ? tg.avg_ts : 0,
    prefill_tok_s_stdev: pp ? pp.stddev_ts : 0,
    decode_tok_s_stdev:  tg ? tg.stddev_ts : 0,
    prefill_samples: pp ? pp.samples_ts : [],
    decode_samples:  tg ? tg.samples_ts : [],
    n_p_eval: pp ? pp.n_prompt : 0,
    n_eval:   tg ? tg.n_gen    : 0,
    t_p_eval_ms: pp ? Math.round(pp.avg_ns / 1e3) / 1e3 : 0,
    t_eval_ms:   tg ? Math.round(tg.avg_ns / 1e3) / 1e3 : 0,
  };
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
    nPrompt: bench.metrics?.n_prompt ?? 0,
    nGen:    bench.metrics?.n_gen    ?? 0,
    nReps:   bench.metrics?.n_reps   ?? 0,
    wallTimeMs,
    metrics: flattenMetrics(bench.metrics),
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

// Run a variant in a Playwright browser (chromium) with a hard-kill timeout.
// Launches a fresh browser per variant to prevent WASM memory accumulation.
// If the browser hangs past the soft timeout, SIGKILL ensures we don't wait forever.
async function runVariantPlaywright(browserName, variant, serverUrl, timestamp, nGpuLayers, refTokenIds, mode = 'both', nRepsOverride = null) {
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
    const { bench } = await runBenchmark(browser, variant, serverUrl, nGpuLayers, refTokenIds, mode, nRepsOverride);
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
async function runVariantSafari(variant, serverUrl, timestamp, nGpuLayers, refTokenIds, mode = 'both', nRepsOverride = null) {
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
    const { bench } = await runBenchmarkSafari(session, variant, serverUrl, nGpuLayers, refTokenIds, mode, nRepsOverride);
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
async function runVariantInBrowser(browserName, variant, serverUrl, timestamp, nGpuLayers, refTokenIds, mode = 'both', nRepsOverride = null) {
  if (browserName === 'webkit') {
    return runVariantSafari(variant, serverUrl, timestamp, nGpuLayers, refTokenIds, mode, nRepsOverride);
  }
  return runVariantPlaywright(browserName, variant, serverUrl, timestamp, nGpuLayers, refTokenIds, mode, nRepsOverride);
}

// Main — variant-first loop: each variant downloads to OPFS in a fresh
// browser context, runs across all browsers, then the context tears down
// (which evicts OPFS automatically).
async function main() {
  console.log('=== WebGPU LLM Benchmark Runner ===');
  console.log(`Browsers: ${config.BROWSERS.join(', ')}`);
  console.log(`Variants: ${config.MODEL_VARIANTS.length} models`);
  console.log(`Machine:  ${config.MACHINE.platform}/${config.MACHINE.arch} - ${config.MACHINE.cpus}`);
  console.log(`GPU layers: ${config.N_GPU_LAYERS}`);
  console.log(`Perf:       -p ${config.N_PROMPT} -n ${config.N_GEN} -r ${config.N_REPS}${config.NO_WARMUP ? ' --no-warmup' : ''}`);
  if (LLAMA_CPP_COMMIT) console.log(`llama.cpp:  ${LLAMA_CPP_COMMIT.slice(0, 10)}`);
  if (config.CONSISTENCY) console.log('Consistency mode: ON (CPU baseline + GPU per variant)');
  if (config.RESUME) console.log('Resume mode: ON (skipping already-succeeded benchmarks)');
  console.log('');

  // Ensure results dir exists
  fs.mkdirSync(config.RESULTS_DIR, { recursive: true });

  const { server, url: serverUrl } = await startServer(config.PORT);
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

        // CPU pass produces token_ids for the GPU forced-decode comparison
        // AND a single-rep perf measurement (warmup + 1 timed) so the
        // dashboard has a CPU baseline to compare GPU numbers against.
        const cpuResult = await runVariantInBrowser(
          blBrowser, variant, serverUrl, timestamp, 0, null, 'both', /* nReps */ 1
        );

        const tokenIds = cpuResult.consistency?.token_ids;
        if (cpuResult.status === 'done' && tokenIds?.length > 0) {
          cpuBaselines[baselineKey][variant.filename] = tokenIds;
          const tests = cpuResult.metrics?.tests;
          const fmt = (prefix) => {
            const t = tests?.find(x => x.name?.startsWith(prefix));
            return t ? `${t.name}: ${t.avg_ts.toFixed(2)} t/s` : `${prefix}: \u2014`;
          };
          console.log(`OK | ${tokenIds.length} ref tokens | ${fmt('pp')} | ${fmt('tg')} | wall: ${(cpuResult.wallTimeMs / 1000).toFixed(1)}s`);
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

      // GPU pass: consistency forced-decode (if CPU baseline present) + perf
      // sweep, both within the same model load. mode='perf' if no consistency
      // wanted at all, otherwise 'both'.
      const gpuMode = config.CONSISTENCY ? 'both' : 'perf';
      const result = await runVariantInBrowser(
        browserName, variant, serverUrl, timestamp, config.N_GPU_LAYERS, refTokenIds, gpuMode
      );

      allResults.push(result);

      if (result.status === 'done' && result.metrics?.tests) {
        const tests = result.metrics.tests;
        const fmt = (prefix) => {
          const t = tests.find(x => x.name?.startsWith(prefix));
          return t ? `${t.name}: ${t.avg_ts.toFixed(2)} \u00b1 ${t.stddev_ts.toFixed(2)} t/s` : `${prefix}: \u2014`;
        };
        const consistencyLabel = config.CONSISTENCY ? ` | ${formatConsistency(result.consistency)}` : '';
        console.log(`OK | ${fmt('pp')} | ${fmt('tg')} | wall: ${(result.wallTimeMs / 1000).toFixed(1)}s${consistencyLabel}`);
      } else {
        console.log(`FAIL | ${result.error || 'unknown error'}`);
      }

      // Save intermediate results (crash resilience)
      fs.writeFileSync(path.join(config.RESULTS_DIR, 'results.json'), JSON.stringify(allResults, null, 2));
    }
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
