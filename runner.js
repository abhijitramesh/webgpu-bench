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
// Mutates the baselines[browserName] map and saves to disk after each run.
async function collectMissingBaselines(browserName, variants, runFn, baselines) {
  if (!baselines[browserName]) baselines[browserName] = {};
  const browserBaselines = baselines[browserName];

  const missing = variants.filter(v => !(v.filename in browserBaselines));
  if (missing.length === 0) {
    console.log(`  CPU baselines: all ${variants.length} cached`);
    return;
  }

  console.log(`  CPU baselines: collecting ${missing.length} (${variants.length - missing.length} cached)`);

  for (let i = 0; i < missing.length; i++) {
    const variant = missing[i];
    process.stdout.write(`    [${i + 1}/${missing.length}] ${variant.name} (CPU)... `);

    const { bench } = await runFn(variant, 0);

    if (bench.status === 'done' && bench.metrics?.token_ids?.length > 0) {
      // Store token IDs as the reference for forced-decoding consistency check
      browserBaselines[variant.filename] = bench.metrics.token_ids;
      console.log(`OK (${bench.metrics.token_ids.length} tokens)`);
    } else {
      browserBaselines[variant.filename] = null; // failed — record so we don't retry
      console.log(`FAIL (${bench.error || 'unknown'})`);
    }

    saveCpuBaselines(baselines);
  }
}

// Main
async function main() {
  console.log('=== WebGPU LLM Benchmark Runner ===');
  console.log(`Browsers: ${config.BROWSERS.join(', ')}`);
  console.log(`Variants: ${config.MODEL_VARIANTS.length} models`);
  console.log(`Machine:  ${config.MACHINE.platform}/${config.MACHINE.arch} - ${config.MACHINE.cpus}`);
  console.log(`GPU layers: ${config.N_GPU_LAYERS}`);
  if (config.CONSISTENCY) console.log('Consistency mode: ON (CPU baselines will be collected per browser)');
  console.log('');

  // Ensure results dir exists
  fs.mkdirSync(config.RESULTS_DIR, { recursive: true });

  // Start server
  const { server, url: serverUrl } = await startServer(config.PORT);
  console.log(`Server: ${serverUrl}`);

  // Load persisted CPU baselines (allows resuming after crash)
  const cpuBaselines = config.CONSISTENCY ? loadCpuBaselines() : {};

  const allResults = [];
  const timestamp = new Date().toISOString();

  for (const browserName of config.BROWSERS) {
    console.log(`\n--- Browser: ${browserName} ---`);

    if (browserName === 'webkit') {
      // Real Safari via WebDriverIO — supports WebGPU natively on macOS.
      // Requires: Safari > Settings > Advanced > "Allow Remote Automation"
      let safariSession;
      try {
        safariSession = await remote({ capabilities: { browserName: 'safari' }, logLevel: 'error' });
      } catch (err) {
        console.error(`  Failed to launch Safari: ${err.message}`);
        continue;
      }

      if (config.CONSISTENCY) {
        await collectMissingBaselines(
          browserName,
          config.MODEL_VARIANTS,
          (variant, nGpuLayers) => runBenchmarkSafari(safariSession, variant, serverUrl, nGpuLayers),
          cpuBaselines,
        );
      }

      for (let i = 0; i < config.MODEL_VARIANTS.length; i++) {
        const variant = config.MODEL_VARIANTS[i];
        const progress = `[${i + 1}/${config.MODEL_VARIANTS.length}]`;
        console.log(`  ${progress} ${variant.name} (${variant.sizeMB} MB)...`);

        const refTokenIds = config.CONSISTENCY
          ? tokenIdsToCsv(cpuBaselines[browserName]?.[variant.filename])
          : null;

        const startTime = Date.now();
        const { bench } = await runBenchmarkSafari(safariSession, variant, serverUrl, config.N_GPU_LAYERS, refTokenIds);
        const wallTimeMs = Date.now() - startTime;

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
      await safariSession.deleteSession();
      continue;
    }

    // Chromium / Firefox via Playwright
    const browserType = getBrowserType(browserName);
    const launchOpts = getBrowserLaunchArgs(browserName);

    let browser;
    try {
      browser = await browserType.launch(launchOpts);
    } catch (err) {
      console.error(`Failed to launch ${browserName}: ${err.message}`);
      continue;
    }

    if (config.CONSISTENCY) {
      await collectMissingBaselines(
        browserName,
        config.MODEL_VARIANTS,
        (variant, nGpuLayers) => runBenchmark(browser, variant, serverUrl, nGpuLayers),
        cpuBaselines,
      );
    }

    // Get GPU info once per browser
    let gpuInfo = null;
    try {
      const ctx = await browser.newContext();
      const page = await ctx.newPage();
      await page.goto(serverUrl + '/harness.html', { timeout: 10_000 });
      gpuInfo = await getGpuInfo(page);
      await ctx.close();
    } catch {
      console.warn('  Could not get GPU info');
    }

    for (let i = 0; i < config.MODEL_VARIANTS.length; i++) {
      const variant = config.MODEL_VARIANTS[i];
      const progress = `[${i + 1}/${config.MODEL_VARIANTS.length}]`;
      console.log(`  ${progress} ${variant.name} (${variant.sizeMB} MB)...`);

      const refTokenIds = config.CONSISTENCY
        ? tokenIdsToCsv(cpuBaselines[browserName]?.[variant.filename])
        : null;

      const startTime = Date.now();
      const { bench } = await runBenchmark(browser, variant, serverUrl, config.N_GPU_LAYERS, refTokenIds);
      const wallTimeMs = Date.now() - startTime;

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
        gpuAdapterInfo: bench.gpuAdapterInfo || gpuInfo?.info || null,
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

      // Save intermediate results (crash resilience)
      const intermediateFile = path.join(config.RESULTS_DIR, 'results.json');
      fs.writeFileSync(intermediateFile, JSON.stringify(allResults, null, 2));
    }

    await browser.close();
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
