import os from 'node:os';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Load models from models.json — edit that file to add new models/repos/quants
function loadModels() {
  const modelsFile = path.join(__dirname, 'models.json');
  const data = JSON.parse(fs.readFileSync(modelsFile, 'utf-8'));

  // Flatten: each model entry has a repo + variants array
  // Produce a flat list with repo attached to each variant
  const allVariants = [];
  for (const model of data.models) {
    for (const v of model.variants) {
      allVariants.push({
        name: v.quant,
        filename: v.filename,
        sizeMB: v.sizeMB,
        repo: model.repo,
        modelName: model.name,
      });
    }
  }
  return { allVariants, quickVariants: data.quickVariants || [] };
}

// Parse CLI arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const parsed = {
    quick: false,
    browsers: null,
    variants: null,
    models: null,
    noWebgpu: false,
    noCache: false,
    consistency: false,
    resume: false,
  };

  for (const arg of args) {
    if (arg === '--quick') {
      parsed.quick = true;
    } else if (arg === '--no-webgpu') {
      parsed.noWebgpu = true;
    } else if (arg === '--no-cache') {
      parsed.noCache = true;
    } else if (arg === '--consistency') {
      parsed.consistency = true;
    } else if (arg === '--resume') {
      parsed.resume = true;
    } else if (arg.startsWith('--browsers=')) {
      parsed.browsers = arg.split('=')[1].split(',');
    } else if (arg.startsWith('--variants=')) {
      parsed.variants = arg.split('=')[1].split(',');
    } else if (arg.startsWith('--models=')) {
      // Filter by model name (e.g., --models=Llama-3.2-1B-Instruct)
      parsed.models = arg.split('=')[1].split(',');
    }
  }

  return parsed;
}

export function getConfig() {
  const args = parseArgs();
  const { allVariants, quickVariants } = loadModels();

  let variants = allVariants;

  // Filter by model name
  if (args.models) {
    variants = variants.filter(v => args.models.some(m => v.modelName.includes(m)));
  }

  // Filter by quant name
  if (args.quick) {
    variants = variants.filter(v => quickVariants.includes(v.name));
  } else if (args.variants) {
    variants = variants.filter(v => args.variants.includes(v.name));
  }

  // Normalize browser aliases: "safari" → "webkit"
  // Default to webkit only on macOS — Safari is unavailable on Linux
  const defaultBrowsers = os.platform() === 'darwin'
    ? ['chromium', 'webkit']
    : ['chromium'];
  const browsers = (args.browsers || defaultBrowsers)
    .map(b => b === 'safari' ? 'webkit' : b);

  // Sort smallest first for faster feedback and early error detection
  variants.sort((a, b) => a.sizeMB - b.sizeMB);

  return {
    PORT: 3000,
    RESULTS_DIR: path.join(__dirname, 'results'),

    // Model config (repo is now per-variant from models.json)
    MODEL_VARIANTS: variants,

    // Inference config
    PROMPT: '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nExplain quantum computing in simple terms.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
    N_PREDICT: 128,
    N_CTX: 2048,
    N_GPU_LAYERS: args.noWebgpu ? 0 : 999,

    // Disable model download caching (always fetch from HuggingFace)
    NO_CACHE: args.noCache || false,

    // Consistency mode: run CPU baselines and compare
    CONSISTENCY: args.consistency || false,

    // Resume mode: skip browser+variant combos that already succeeded
    RESUME: args.resume || false,

    // Browser config
    BROWSERS: browsers,

    // Timeouts (ms)
    TIMEOUTS: {
      download: 600_000,   // 10 min for large model downloads
      inference: 300_000,  // 5 min for inference
      total: 900_000,      // 15 min total per variant
    },

    // Machine info
    MACHINE: {
      platform: os.platform(),
      arch: os.arch(),
      cpus: os.cpus()[0]?.model || 'unknown',
      totalMemoryGB: Math.round(os.totalmem() / (1024 ** 3)),
      hostname: os.hostname(),
      slug: process.env.MACHINE_SLUG || null,
    },
  };
}
