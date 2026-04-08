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

function buildHarnessUrl(serverUrl, variant) {
  const harnessUrl = new URL('/harness.html', serverUrl);
  harnessUrl.searchParams.set('model', variant.filename);
  harnessUrl.searchParams.set('hfRepo', variant.repo);
  harnessUrl.searchParams.set('prompt', config.PROMPT);
  harnessUrl.searchParams.set('nPredict', String(config.N_PREDICT));
  harnessUrl.searchParams.set('nCtx', String(config.N_CTX));
  harnessUrl.searchParams.set('nGpuLayers', String(config.N_GPU_LAYERS));
  return harnessUrl.toString();
}

// Run a single benchmark via Playwright (chromium, firefox)
async function runBenchmark(browser, variant, serverUrl) {
  const context = await browser.newContext();
  const page = await context.newPage();

  // Capture console output
  const consoleLogs = [];
  page.on('console', msg => {
    consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
  });

  const harnessUrl = buildHarnessUrl(serverUrl, variant);

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
async function runBenchmarkSafari(browser, variant, serverUrl) {
  const harnessUrl = buildHarnessUrl(serverUrl, variant);

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

// Main
async function main() {
  console.log('=== WebGPU LLM Benchmark Runner ===');
  console.log(`Browsers: ${config.BROWSERS.join(', ')}`);
  console.log(`Variants: ${config.MODEL_VARIANTS.length} models`);
  console.log(`Machine:  ${config.MACHINE.platform}/${config.MACHINE.arch} - ${config.MACHINE.cpus}`);
  console.log(`GPU layers: ${config.N_GPU_LAYERS}`);
  console.log('');

  // Ensure results dir exists
  fs.mkdirSync(config.RESULTS_DIR, { recursive: true });

  // Start server
  const { server, url: serverUrl } = await startServer(config.PORT);
  console.log(`Server: ${serverUrl}`);

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

      for (let i = 0; i < config.MODEL_VARIANTS.length; i++) {
        const variant = config.MODEL_VARIANTS[i];
        const progress = `[${i + 1}/${config.MODEL_VARIANTS.length}]`;
        console.log(`  ${progress} ${variant.name} (${variant.sizeMB} MB)...`);

        const startTime = Date.now();
        const { bench, consoleLogs } = await runBenchmarkSafari(safariSession, variant, serverUrl);
        const wallTimeMs = Date.now() - startTime;

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
        };

        allResults.push(result);

        if (bench.status === 'done' && bench.metrics) {
          const m = bench.metrics;
          console.log(`    OK | prefill: ${m.prefill_tok_s} tok/s | decode: ${m.decode_tok_s} tok/s | wall: ${(wallTimeMs / 1000).toFixed(1)}s`);
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

      const startTime = Date.now();
      const { bench, consoleLogs } = await runBenchmark(browser, variant, serverUrl);
      const wallTimeMs = Date.now() - startTime;

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
      };

      allResults.push(result);

      // Print summary
      if (bench.status === 'done' && bench.metrics) {
        const m = bench.metrics;
        console.log(`    OK | prefill: ${m.prefill_tok_s} tok/s | decode: ${m.decode_tok_s} tok/s | wall: ${(wallTimeMs / 1000).toFixed(1)}s`);
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

  await stopServer(server);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
