// GGUF source abstraction.
// localSource() — reads through the local Express proxy (/models/{repo}/{file}),
//                 server handles disk cache at cache/models/**.
// hostedSource() — caches in OPFS, fetches directly from HF on miss.
//
// Both expose:
//   isCached(repo, file) → Promise<{ cachedBytes: number, totalBytes?: number }>
//   fetchModel(repo, file) → Promise<{ stream: ReadableStream<Uint8Array>,
//                                      contentLength: number,
//                                      source: string }>
//
// Additional hosted helpers:
//   inventoryOpfs() → Promise<{ 'repo/file': { cachedBytes } }>
//   purgeOpfs() → Promise<void>

export function localSource() {
  return {
    async isCached(repo, file) {
      try {
        const url = `/api/cache-status?path=${encodeURIComponent(`${repo}/${file}`)}`;
        const r = await fetch(url);
        if (!r.ok) return { cachedBytes: 0 };
        return await r.json();
      } catch {
        return { cachedBytes: 0 };
      }
    },

    async fetchModel(repo, file) {
      const url = `/models/${repo}/${file}`;
      const resp = await fetch(url);
      if (!resp.ok) {
        throw new Error(`Download failed: ${resp.status} ${resp.statusText}`);
      }
      const contentLength = parseInt(resp.headers.get('content-length') || '0', 10);
      return {
        stream: resp.body,
        contentLength,
        source: 'localProxy',
      };
    },

    async evictModel(repo, file) {
      try {
        const r = await fetch(`/api/cache/${repo}/${file}`, { method: 'DELETE' });
        if (r.status === 404) return { ok: false, bytesFreed: 0, reason: 'not cached' };
        if (!r.ok) return { ok: false, bytesFreed: 0, reason: `${r.status} ${r.statusText}` };
        const body = await r.json().catch(() => ({}));
        return { ok: true, bytesFreed: body.bytesFreed || 0 };
      } catch (err) {
        return { ok: false, bytesFreed: 0, reason: err.message };
      }
    },
  };
}

// ──────────────── hosted / OPFS ────────────────

const OPFS_ROOT_NAME = 'models';

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

    async fetchModel(repo, file) {
      // Cache hit → stream the OPFS file straight out.
      try {
        const handle = await getOpfsFileHandle(repo, file, { create: false });
        const f = await handle.getFile();
        if (f.size > 0) {
          return {
            stream: f.stream(),
            contentLength: f.size,
            source: 'opfs',
          };
        }
      } catch { /* miss — fall through */ }

      // Miss: fetch from HF, tee to OPFS + caller.
      const url = `https://huggingface.co/${repo}/resolve/main/${file}`;
      const resp = await fetch(url);
      if (!resp.ok) {
        throw new Error(`Download failed: ${resp.status} ${resp.statusText}`);
      }
      const contentLength = parseInt(resp.headers.get('content-length') || '0', 10);

      const handle = await getOpfsFileHandle(repo, file, { create: true });
      const writable = await handle.createWritable({ keepExistingData: false });

      const [toCache, toCaller] = resp.body.tee();

      // Opportunistically request persistent storage so eviction is less
      // likely once we commit to pulling large files. Best-effort — ignore
      // rejection (some browsers only grant on user gesture).
      navigator.storage?.persist?.().catch(() => {});

      // Pipe to OPFS in the background; log but don't block the caller.
      toCache.pipeTo(writable).catch(err => {
        console.warn(`OPFS write failed for ${repo}/${file}: ${err.message}`);
      });

      return {
        stream: toCaller,
        contentLength,
        source: 'hf-direct',
      };
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
