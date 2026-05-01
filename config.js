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
  return {
    allVariants,
    quickVariants: data.quickVariants || [],
    studySelection: data.studySelection || null,
  };
}

// Apply the study selection rule from models.json to a flat variant list.
// Same shape the interactive Run-page "Run study" button uses, so CLI and
// webapp run the exact same sweep.
function applyStudyFilter(variants, studySelection) {
  if (!studySelection) {
    throw new Error('--study requested but models.json has no studySelection block');
  }
  const focusModel = studySelection.focusModel;
  const focusQuants = new Set(studySelection.focusQuants || []);
  const standardQuant = studySelection.standardQuant;
  return variants.filter(v => {
    if (v.modelName === focusModel) return focusQuants.has(v.name);
    return v.name === standardQuant;
  });
}

// Parse CLI arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const parsed = {
    quick: false,
    study: false,
    browsers: null,
    variants: null,
    models: null,
    noWebgpu: false,
    consistency: false,
    resume: false,
    noWarmup: false,
    nPrompt: null,
    nGen: null,
    nReps: null,
    nDepth: null,
  };

  const intArg = (arg, name) => {
    const raw = arg.split('=')[1];
    const n = parseInt(raw, 10);
    if (!Number.isFinite(n) || n < 0) {
      throw new Error(`${name} must be a non-negative integer (got "${raw}")`);
    }
    return n;
  };

  for (const arg of args) {
    if (arg === '--quick') {
      parsed.quick = true;
    } else if (arg === '--study') {
      parsed.study = true;
    } else if (arg === '--no-webgpu') {
      parsed.noWebgpu = true;
    } else if (arg === '--consistency') {
      parsed.consistency = true;
    } else if (arg === '--resume') {
      parsed.resume = true;
    } else if (arg === '--no-warmup') {
      parsed.noWarmup = true;
    } else if (arg.startsWith('--n-prompt=')) {
      parsed.nPrompt = intArg(arg, '--n-prompt');
    } else if (arg.startsWith('--n-gen=')) {
      parsed.nGen = intArg(arg, '--n-gen');
    } else if (arg.startsWith('--n-reps=')) {
      parsed.nReps = intArg(arg, '--n-reps');
    } else if (arg.startsWith('--n-depth=')) {
      parsed.nDepth = intArg(arg, '--n-depth');
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
  const { allVariants, quickVariants, studySelection } = loadModels();

  let variants = allVariants;

  // Study selection runs first — it picks a curated set the interactive
  // "Run study" button uses (Llama-3.2-1B-Instruct at four quants for the
  // headline sweep + every other model at the standard quant). Any
  // subsequent --models / --variants / --quick narrows further within it.
  if (args.study) {
    variants = applyStudyFilter(variants, studySelection);
  }

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

    // Consistency-phase prompt is owned by site/js/run/config.js so the
    // interactive Run page and harness.js share a single value.
    N_PREDICT: 128,
    // Perf phase (synthetic-token llama-bench-style pp / tg)
    N_PROMPT: args.nPrompt ?? 512,
    N_GEN:    args.nGen    ?? 128,
    N_REPS:   args.nReps   ?? 5,
    // KV-cache prefill depth before each timed pp/tg rep (llama-bench `-d`).
    // Default 2048 mirrors the typical context-loaded use-case. Study mode
    // additionally pairs every variant with a d=0 baseline run.
    N_DEPTH:  args.nDepth  ?? 2048,
    NO_WARMUP: args.noWarmup || false,
    N_CTX: 2048,
    N_GPU_LAYERS: args.noWebgpu ? 0 : 999,

    // Consistency mode: run CPU baselines and compare
    CONSISTENCY: args.consistency || false,

    // Resume mode: skip browser+variant combos that already succeeded
    RESUME: args.resume || false,

    // Study mode: surfaced in the runner banner. The selection itself was
    // already applied to MODEL_VARIANTS above; this is just a label.
    STUDY: args.study || false,

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
