// GGUF source. Single implementation now — every surface fetches directly
// from HF and caches in OPFS in the browser. The Express disk cache is
// gone, so localhost and HF Space share the same loader.
//
// Exposes:
//   isCached(repo, file)        → { cachedBytes, totalBytes? }
//   opfsHandleForModel(repo, file, onProgress, signal)
//                               → { handle, size, wasDownloaded }
//   evictModel(repo, file)      → { ok, bytesFreed, reason? }
//
// Helpers: inventoryOpfs(), purgeOpfs().

// Exported so bench-worker.js can re-resolve the OPFS file handle inside
// the worker. We can't transfer FileSystemFileHandle across postMessage on
// every browser (iOS Safari's structured-clone is missing the
// implementation), so instead we send the layout key (rootDir + repo
// segments + filename) and let the worker open the handle itself.
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

// WebKit (iOS Safari) returns one of these strings/names when the OPFS
// operation fails because something else (typically a stuck
// FileSystemSyncAccessHandle from a worker that was Jetsam-killed before
// it could close cleanly) is still holding the file. The handle is
// usually released within a few seconds, so retrying with backoff is the
// documented mitigation. Other "real" errors (NotFoundError, QuotaExceeded)
// are not transient and shouldn't be retried.
function isOpfsTransientError(err) {
  if (!err) return false;
  const msg = String(err.message || err);
  if (/unknown transient/i.test(msg)) return true;
  if (/no modification allowed/i.test(msg)) return true;
  if (err.name === 'InvalidStateError') return true;
  if (err.name === 'NoModificationAllowedError') return true;
  return false;
}

async function withOpfsRetry(fn) {
  const delays = [500, 2_000, 5_000];
  let lastErr;
  for (let attempt = 0; attempt <= delays.length; attempt++) {
    try {
      return await fn(attempt);
    } catch (err) {
      lastErr = err;
      if (!isOpfsTransientError(err)) throw err;
      if (attempt === delays.length) break;
      await new Promise((r) => setTimeout(r, delays[attempt]));
    }
  }
  throw lastErr;
}

export function ggufSource() {
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
    // FileSystemFileHandle. The worker (bench-worker.js) opens a sync
    // access handle on this file and routes MEMFS reads through it, so
    // model bytes never enter the WASM heap. onProgress fires during
    // download with (fraction, downloaded, total). `wasDownloaded`
    // distinguishes a fresh download from a cache hit so the caller can
    // decide whether to evict the variant after the run.
    async opfsHandleForModel(repo, file, onProgress, signal) {
      // Cache lookup — wrapped in retry because getFile() can also hit
      // the WebKit transient (a sync access handle from a previous
      // worker that was Jetsam-killed mid-run blocks this for a few
      // seconds until WebKit's GC reaps it).
      const cached = await withOpfsRetry(async () => {
        const handle = await getOpfsFileHandle(repo, file, { create: false }).catch(() => null);
        if (!handle) return null;
        const f = await handle.getFile();
        return f.size > 0 ? { handle, size: f.size } : null;
      });
      if (cached) {
        onProgress?.(1, cached.size, cached.size);
        return { handle: cached.handle, size: cached.size, wasDownloaded: false };
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

      // Opportunistically request persistent storage so eviction is less
      // likely once we commit to pulling large files. Best-effort — ignore
      // rejection (some browsers only grant on user gesture).
      navigator.storage?.persist?.().catch(() => {});

      // Retry the createWritable + drain loop on the WebKit transient.
      // Each retry restarts the download from byte 0; for streamed writes
      // we can't resume mid-file without re-issuing the fetch, and the
      // transient typically only fires on createWritable so retrying is
      // usually a no-op past attempt 0. Fresh fetch per attempt is the
      // simplest correct thing.
      return await withOpfsRetry(async (attempt) => {
        const handle = await getOpfsFileHandle(repo, file, { create: true });
        const writable = await handle.createWritable({ keepExistingData: false });

        // On retry we need a fresh response body — the original reader
        // was consumed (or aborted) by the previous attempt. Use the
        // already-fetched response on attempt 0; re-fetch on retries.
        const body = attempt === 0 ? resp.body : (await fetch(url, { signal })).body;

        try {
          const reader = body.getReader();
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
      });
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

// Delete every cached file under OPFS `models/`. Used by the [Purge] button.
export async function purgeOpfs() {
  if (!navigator.storage?.getDirectory) return;
  const root = await navigator.storage.getDirectory();
  try {
    await root.removeEntry(OPFS_ROOT_NAME, { recursive: true });
  } catch { /* didn't exist */ }
}
