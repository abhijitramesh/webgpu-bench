import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function startServer(port = 3000) {
  return new Promise((resolve) => {
    const app = express();

    // CORS headers required for SharedArrayBuffer and WASM
    app.use((req, res, next) => {
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      next();
    });

    // JSON parser for POST /api/results.
    app.use(express.json({ limit: '10mb' }));

    // Root / → /site/  so the merged app is the canonical entry.
    app.get('/', (req, res, next) => {
      if (req.path === '/') return res.redirect(302, '/site/');
      next();
    });
    app.get('/bench.html', (req, res) => res.redirect(301, '/site/run.html'));

    // Alias so harness.html (served at root) can import Run-tab modules via
    // `./js/run/*.js` — the same relative path that works on the HF Space
    // where the merged app is flattened to the Space root.
    app.use('/js', express.static(path.join(__dirname, 'site', 'js'), {
      setHeaders: (res, filePath) => {
        if (filePath.endsWith('.js')) {
          res.setHeader('Content-Type', 'application/javascript');
        }
      },
    }));

    // Serve harness files from project root
    app.use(express.static(__dirname, {
      setHeaders: (res, filePath) => {
        if (filePath.endsWith('.wasm')) {
          res.setHeader('Content-Type', 'application/wasm');
        }
        if (filePath.endsWith('.js')) {
          res.setHeader('Content-Type', 'application/javascript');
        }
      },
    }));

    // Serve WASM build outputs — JSPI and Asyncify variants
    const wasmHeaders = (res, filePath) => {
      if (filePath.endsWith('.wasm')) {
        res.setHeader('Content-Type', 'application/wasm');
      }
    };
    app.use('/build/jspi', express.static(path.join(__dirname, 'build', 'jspi', 'bin'), {
      setHeaders: wasmHeaders,
    }));
    app.use('/build/asyncify', express.static(path.join(__dirname, 'build', 'asyncify', 'bin'), {
      setHeaders: wasmHeaders,
    }));

    const RESULTS_DIR = path.join(__dirname, 'results');
    const RESULTS_FILE = path.join(RESULTS_DIR, 'results.json');

    // Return parsed models.json for the interactive bench page.
    app.get('/api/models', (req, res) => {
      try {
        const modelsPath = path.join(__dirname, 'models.json');
        const models = JSON.parse(fs.readFileSync(modelsPath, 'utf-8'));
        res.json(models);
      } catch (err) {
        res.status(500).json({ error: err.message });
      }
    });

    // Append a benchmark result record to results/results.json. Body shape
    // matches what runner.js produces (one element of the array).
    app.post('/api/results', (req, res) => {
      const record = req.body;
      if (!record || typeof record !== 'object' || Array.isArray(record)) {
        res.status(400).json({ error: 'Expected a single JSON object' });
        return;
      }
      try {
        fs.mkdirSync(RESULTS_DIR, { recursive: true });
        let results = [];
        if (fs.existsSync(RESULTS_FILE)) {
          try {
            const parsed = JSON.parse(fs.readFileSync(RESULTS_FILE, 'utf-8'));
            if (Array.isArray(parsed)) results = parsed;
          } catch {
            // Corrupt file — start over rather than crash. The tmp+rename path
            // below makes this safe.
          }
        }
        results.push(record);
        const tmpFile = RESULTS_FILE + '.tmp';
        fs.writeFileSync(tmpFile, JSON.stringify(results, null, 2));
        fs.renameSync(tmpFile, RESULTS_FILE);
        res.json({ status: 'ok', count: results.length });
      } catch (err) {
        res.status(500).json({ error: err.message });
      }
    });

    const server = app.listen(port, () => {
      const url = `http://localhost:${port}`;
      console.log(`Server running at ${url}`);
      resolve({ server, url });
    });
  });
}

export function stopServer(server) {
  return new Promise((resolve) => {
    server.close(resolve);
  });
}

// When run directly (`node server.js`), start listening so the README
// workflow is self-consistent. When imported (e.g. from runner.js), the
// caller drives startServer() itself.
if (import.meta.url === `file://${process.argv[1]}`) {
  const port = parseInt(process.env.PORT || '3000', 10);
  startServer(port).catch((err) => {
    console.error('Failed to start server:', err);
    process.exit(1);
  });
}
