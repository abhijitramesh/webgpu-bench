import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function startServer(port = 3000, { noCache = false } = {}) {
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
    app.get('/bench.html', (req, res) => res.redirect(301, '/site/run/'));

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

    // Model download proxy — optionally caches to disk.
    // With caching (default): first request streams from HF to both client and disk.
    // Subsequent requests serve from disk. With --no-cache: always streams from HF.
    const CACHE_DIR = path.join(__dirname, 'cache', 'models');
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

    // Cache inventory. With ?path=<repo/file>, returns a single entry.
    // Without, returns a map of all cached files under cache/models/.
    app.get('/api/cache-status', (req, res) => {
      const singlePath = req.query.path;
      if (typeof singlePath === 'string' && singlePath.length > 0) {
        // Defend against path traversal by resolving and checking containment.
        const resolved = path.resolve(CACHE_DIR, singlePath);
        if (!resolved.startsWith(CACHE_DIR + path.sep) && resolved !== CACHE_DIR) {
          res.status(400).json({ error: 'Invalid path' });
          return;
        }
        if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) {
          res.json({ cachedBytes: fs.statSync(resolved).size });
        } else {
          res.json({ cachedBytes: 0 });
        }
        return;
      }

      const inventory = {};
      if (fs.existsSync(CACHE_DIR)) {
        const walk = (dir, relRoot) => {
          for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
            const full = path.join(dir, entry.name);
            const rel = path.posix.join(relRoot, entry.name);
            if (entry.isDirectory()) {
              walk(full, rel);
            } else if (entry.isFile() && !entry.name.endsWith('.tmp')) {
              inventory[rel] = { cachedBytes: fs.statSync(full).size };
            }
          }
        };
        walk(CACHE_DIR, '');
      }
      res.json(inventory);
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

    app.get('/models/*', async (req, res) => {
      const modelPath = req.params[0];
      const cachePath = path.join(CACHE_DIR, modelPath);

      // Serve from cache (unless --no-cache)
      if (!noCache && fs.existsSync(cachePath)) {
        const stat = fs.statSync(cachePath);
        res.setHeader('Content-Length', String(stat.size));
        res.setHeader('Content-Type', 'application/octet-stream');
        console.log(`Cache hit: ${modelPath} (${(stat.size / (1024 * 1024)).toFixed(0)} MB)`);
        fs.createReadStream(cachePath).pipe(res);
        return;
      }

      // Parse repo/filename from path: /models/<org>/<repo>/<file>
      const parts = modelPath.split('/');
      if (parts.length < 3) {
        res.status(400).send('Invalid model path — expected /models/<org>/<repo>/<file>');
        return;
      }
      const repo = parts.slice(0, 2).join('/');
      const filename = parts.slice(2).join('/');
      const hfUrl = `https://huggingface.co/${repo}/resolve/main/${filename}`;

      console.log(`${noCache ? 'No-cache' : 'Cache miss'}: downloading ${filename} from ${repo}`);

      try {
        const upstream = await fetch(hfUrl);
        if (!upstream.ok) {
          res.status(upstream.status).send(`HuggingFace: ${upstream.statusText}`);
          return;
        }

        const contentLength = upstream.headers.get('content-length');
        if (contentLength) res.setHeader('Content-Length', contentLength);
        res.setHeader('Content-Type', 'application/octet-stream');

        if (noCache) {
          // Stream directly to client, no disk writes
          const reader = upstream.body.getReader();
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              res.write(value);
            }
            res.end();
          } catch (err) {
            if (!res.headersSent) {
              res.status(502).send(`Download error: ${err.message}`);
            } else {
              res.end();
            }
          }
          return;
        }

        fs.mkdirSync(path.dirname(cachePath), { recursive: true });

        const tmpPath = cachePath + '.tmp';
        const fileStream = fs.createWriteStream(tmpPath);
        const reader = upstream.body.getReader();

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            fileStream.write(value);
            res.write(value);
          }
          fileStream.end();
          await new Promise((resolve, reject) => {
            fileStream.on('finish', resolve);
            fileStream.on('error', reject);
          });
          fs.renameSync(tmpPath, cachePath);
          res.end();
          console.log(`Cached: ${modelPath} (${(parseInt(contentLength || '0') / (1024 * 1024)).toFixed(0)} MB)`);
        } catch (err) {
          fileStream.end();
          try { fs.unlinkSync(tmpPath); } catch {}
          if (!res.headersSent) {
            res.status(502).send(`Download error: ${err.message}`);
          } else {
            res.end();
          }
        }
      } catch (err) {
        if (!res.headersSent) {
          res.status(502).send(`Fetch error: ${err.message}`);
        }
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
