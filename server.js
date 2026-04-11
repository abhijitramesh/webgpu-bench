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

    // Model download cache — proxy HuggingFace and cache to disk.
    // First request streams from HF to both client and disk.
    // Subsequent requests serve from disk, skipping the network entirely.
    const CACHE_DIR = path.join(__dirname, 'cache', 'models');
    app.get('/models/*', async (req, res) => {
      const modelPath = req.params[0];
      const cachePath = path.join(CACHE_DIR, modelPath);

      // Serve from cache
      if (fs.existsSync(cachePath)) {
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

      console.log(`Cache miss: downloading ${filename} from ${repo}`);

      try {
        const upstream = await fetch(hfUrl);
        if (!upstream.ok) {
          res.status(upstream.status).send(`HuggingFace: ${upstream.statusText}`);
          return;
        }

        const contentLength = upstream.headers.get('content-length');
        if (contentLength) res.setHeader('Content-Length', contentLength);
        res.setHeader('Content-Type', 'application/octet-stream');

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
