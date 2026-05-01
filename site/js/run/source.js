// GGUF source abstraction.
// localSource() — fetches from the local Express proxy (/models/{repo}/{file}),
//                 caches in OPFS in the browser. Used by the CLI harness +
//                 interactive Run page on localhost.
// hostedSource() — fetches directly from HF, caches in OPFS. Used by the
//                  interactive Run page on the HF Space and other hosted
//                  surfaces.
//
// Both expose the same shape:
//   isCached(repo, file)         → { cachedBytes, totalBytes? }
//   opfsHandleForModel(repo, file, onProgress, signal)
//                                → { handle, size, wasDownloaded }
//   evictModel(repo, file)       → { ok, bytesFreed, reason? }
//
// Additional helpers (OPFS-only): inventoryOpfs(), purgeOpfs().

export function localSource() {
  return {
    async isCached(repo, file) {
      try {
        const handle = await getOpfsFileHandle(repo, file, { create: false });
        const f = await handle.getFile();
        return { cachedBytes: f.size, totalBytes: f.size };
      } catch {
        return { cachedBytes: 0 };
      }
    },

    // Mirror of hostedSource.opfsHandleForModel — same OPFS write logic,
    // just fetches from the local Express proxy instead of HF directly.
    // Express still does its own disk-cache layer in commit-2 land; that
    // disappears in commit-3 and this becomes a direct HF fetch.
    async opfsHandleForModel(repo, file, onProgress, signal) {
      const cached = await getOpfsFileHandle(repo, file, { create: false }).catch(() => null);
      if (cached) {
        const f = await cached.getFile();
        if (f.size > 0) {
          onProgress?.(1, f.size, f.size);
          return { handle: cached, size: f.size, wasDownloaded: false };
        }
      }

      const url = `/models/${repo}/${file}`;
      const resp = await fetch(url, { signal });
      if (!resp.ok) {
        throw new Error(`Download failed: ${resp.status} ${resp.statusText}`);
      }
      const contentLength = parseInt(resp.headers.get('content-length') || '0', 10);

      const handle = await getOpfsFileHandle(repo, file, { create: true });
      const writable = await handle.createWritable({ keepExistingData: false });
      navigator.storage?.persist?.().catch(() => {});

      try {
        const reader = resp.body.getReader();
        let downloaded = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          await writable.write(value);
          downloaded += value.byteLength;
          if (contentLength > 0) onProgress?.(downloaded / contentLength, downloaded, contentLength);
        }
        await writable.close();
        return { handle, size: downloaded, wasDownloaded: true };
      } catch (err) {
        try { await writable.abort(err); } catch { /* ignore */ }
        throw err;
      }
    },

    async evictModel(repo, file) {
      try {
        const dir = await getOpfsDirFor(repo, { create: false });
        let bytesFreed = 0;
        try {
          const handle = await dir.getFileHandle(file, { create: false });
          const f = await handle.getFile();
          bytesFreed = f.size;
        } catch { /* not present */ }
        await dir.removeEntry(file);
        return { ok: true, bytesFreed };
      } catch (err) {
        return { ok: false, bytesFreed: 0, reason: err.message };
      }
    },
  };
}

// ──────────────── hosted / OPFS ────────────────

// Exported so bench-worker.js can re-resolve the OPFS file handle inside
// the worker. We can't transfer FileSystemFileHandle directly across
// postMessage on every browser (iOS Safari structured-clone is missing
// the implementation), so instead we send the layout key (rootDir +
// repo segments + filename) and let the worker open it itself.
export const OPFS_ROOT_NAME = 'models';

async function getOpfsRoot() {
  if (!navigator.storage?.getDirectory) {
    throw new Error('OPFS is not available in this browser.');
  }
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(OPFS_ROOT_NAME, { create: true });
}

function repoSegments(repo) {
  return String(repo).split('/').filter(Boolean);
}

async function getOpfsDirFor(repo, { create }) {
  let dir = await getOpfsRoot();
  for (const seg of repoSegments(repo)) {
    dir = await dir.getDirectoryHandle(seg, { create });
  }
  return dir;
}

async function getOpfsFileHandle(repo, file, { create }) {
  const dir = await getOpfsDirFor(repo, { create });
  return dir.getFileHandle(file, { create });
}

export function hostedSource() {
  return {
    async isCached(repo, file) {
      try {
        const handle = await getOpfsFileHandle(repo, file, { create: false });
        const f = await handle.getFile();
        return { cachedBytes: f.size, totalBytes: f.size };
      } catch {
        return { cachedBytes: 0 };
      }
    },

    // Ensure the model is fully downloaded to OPFS, then return its
    // FileSystemFileHandle. Used by the wllama-style OPFS-streaming load
    // path: the worker opens a sync access handle on this FileHandle and
    // routes MEMFS reads through it, never copying the model into the
    // WASM heap. onProgress is called during the download leg with
    // (fraction, downloaded, total). The returned `wasDownloaded` flag
    // distinguishes a fresh download from a cache hit so the caller can
    // decide whether to mark the variant for post-run eviction.
    async opfsHandleForModel(repo, file, onProgress, signal) {
      const cached = await getOpfsFileHandle(repo, file, { create: false }).catch(() => null);
      if (cached) {
        const f = await cached.getFile();
        if (f.size > 0) {
          onProgress?.(1, f.size, f.size);
          return { handle: cached, size: f.size, wasDownloaded: false };
        }
      }

      // Cache miss — download from HF straight into a writable OPFS stream.
      // signal lets the caller cancel: fetch + reader.read both reject with
      // AbortError when it fires, and the catch below propagates that up.
      const url = `https://huggingface.co/${repo}/resolve/main/${file}`;
      const resp = await fetch(url, { signal });
      if (!resp.ok) {
        throw new Error(`Download failed: ${resp.status} ${resp.statusText}`);
      }
      const contentLength = parseInt(resp.headers.get('content-length') || '0', 10);

      const handle = await getOpfsFileHandle(repo, file, { create: true });
      const writable = await handle.createWritable({ keepExistingData: false });

      // Opportunistically request persistent storage so eviction is less
      // likely once we commit to pulling large files. Best-effort — ignore
      // rejection (some browsers only grant on user gesture).
      navigator.storage?.persist?.().catch(() => {});

      try {
        const reader = resp.body.getReader();
        let downloaded = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          await writable.write(value);
          downloaded += value.byteLength;
          if (contentLength > 0) onProgress?.(downloaded / contentLength, downloaded, contentLength);
        }
        await writable.close();
        return { handle, size: downloaded, wasDownloaded: true };
      } catch (err) {
        try { await writable.abort(err); } catch { /* ignore */ }
        throw err;
      }
    },

    async evictModel(repo, file) {
      try {
        const dir = await getOpfsDirFor(repo, { create: false });
        // Read size first so we can report it, then remove.
        let bytesFreed = 0;
        try {
          const handle = await dir.getFileHandle(file, { create: false });
          const f = await handle.getFile();
          bytesFreed = f.size;
        } catch { /* not present */ }
        await dir.removeEntry(file);
        return { ok: true, bytesFreed };
      } catch (err) {
        return { ok: false, bytesFreed: 0, reason: err.message };
      }
    },
  };
}

// Walk OPFS and report every cached file as `{ 'repo/file': { cachedBytes } }`.
export async function inventoryOpfs() {
  if (!navigator.storage?.getDirectory) return {};
  const root = await navigator.storage.getDirectory();
  let modelsDir;
  try {
    modelsDir = await root.getDirectoryHandle(OPFS_ROOT_NAME, { create: false });
  } catch { return {}; }

  const out = {};
  async function walk(dir, relParts) {
    for await (const entry of dir.values()) {
      if (entry.kind === 'directory') {
        await walk(entry, [...relParts, entry.name]);
      } else if (entry.kind === 'file') {
        const f = await entry.getFile();
        const key = [...relParts, entry.name].join('/');
        out[key] = { cachedBytes: f.size };
      }
    }
  }
  await walk(modelsDir, []);
  return out;
}

// Delete every cached file under OPFS `models/`. Used by a [Purge] button.
export async function purgeOpfs() {
  if (!navigator.storage?.getDirectory) return;
  const root = await navigator.storage.getDirectory();
  try {
    await root.removeEntry(OPFS_ROOT_NAME, { recursive: true });
  } catch { /* didn't exist */ }
}
