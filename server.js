import express from 'express';
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
