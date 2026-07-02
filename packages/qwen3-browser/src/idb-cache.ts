/**
 * IndexedDB chunk cache for streamed weight shards (browser/worker only).
 *
 * Layout: DB "qwen3-weights-cache"
 *   store "chunks": key `${url}#${i}` → ArrayBuffer (CHUNK_BYTES each)
 *   store "meta":   key url → { totalBytes, chunkCount }   (written LAST —
 *                   presence of meta means the entry is complete; partial
 *                   downloads are invisible and get overwritten next time)
 *
 * All failures (quota, private browsing, blocked) degrade to "no cache":
 * writes disable themselves, reads return null.
 */

const DB_NAME = "qwen3-weights-cache";
const CHUNKS = "chunks";
const META = "meta";
export const CACHE_CHUNK_BYTES = 32 * 1024 * 1024;

type ShardMeta = { totalBytes: number; chunkCount: number };

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(CHUNKS)) db.createObjectStore(CHUNKS);
      if (!db.objectStoreNames.contains(META)) db.createObjectStore(META);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function idbGet<T>(db: IDBDatabase, store: string, key: string): Promise<T | undefined> {
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, "readonly").objectStore(store).get(key);
    req.onsuccess = () => resolve(req.result as T | undefined);
    req.onerror = () => reject(req.error);
  });
}

function idbPut(db: IDBDatabase, store: string, key: string, value: unknown): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, "readwrite");
    tx.objectStore(store).put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/** Meta for a completely cached shard, or null if absent/unavailable. */
export async function getCachedShardMeta(url: string): Promise<ShardMeta | null> {
  try {
    if (typeof indexedDB === "undefined") return null;
    const db = await openDb();
    const meta = await idbGet<ShardMeta>(db, META, url);
    db.close();
    return meta ?? null;
  } catch {
    return null;
  }
}

/** Byte source over a completely cached shard (ByteCursor-compatible). */
export class CachedShardSource {
  private db: IDBDatabase | null = null;
  private chunkIdx = 0;
  constructor(
    private url: string,
    private meta: ShardMeta,
    private onBytes: (n: number) => void,
  ) {}

  async next(): Promise<Uint8Array | null> {
    if (this.chunkIdx >= this.meta.chunkCount) {
      this.db?.close();
      return null;
    }
    if (!this.db) this.db = await openDb();
    const buf = await idbGet<ArrayBuffer>(this.db, CHUNKS, `${this.url}#${this.chunkIdx}`);
    if (!buf) throw new Error(`weights cache corrupt (missing chunk ${this.chunkIdx}) — reload to re-download`);
    this.chunkIdx++;
    const bytes = new Uint8Array(buf);
    this.onBytes(bytes.length);
    return bytes;
  }
}

/**
 * Tee-writer: append network bytes as they're consumed; `finish()` writes the
 * meta record (making the entry visible). Any failure silently disables it.
 */
export class ShardCacheWriter {
  private db: IDBDatabase | null = null;
  private staging = new Uint8Array(CACHE_CHUNK_BYTES);
  private stagingLen = 0;
  private chunkIdx = 0;
  private totalBytes = 0;
  private disabled = false;

  constructor(private url: string) {}

  private async flush(): Promise<void> {
    if (this.stagingLen === 0) return;
    if (!this.db) this.db = await openDb();
    // Copy out the filled prefix (the staging buffer is reused).
    const chunk = this.staging.slice(0, this.stagingLen).buffer;
    await idbPut(this.db, CHUNKS, `${this.url}#${this.chunkIdx}`, chunk);
    this.chunkIdx++;
    this.stagingLen = 0;
  }

  async append(bytes: Uint8Array): Promise<void> {
    if (this.disabled) return;
    try {
      let off = 0;
      while (off < bytes.length) {
        const n = Math.min(bytes.length - off, CACHE_CHUNK_BYTES - this.stagingLen);
        this.staging.set(bytes.subarray(off, off + n), this.stagingLen);
        this.stagingLen += n;
        off += n;
        if (this.stagingLen === CACHE_CHUNK_BYTES) await this.flush();
      }
      this.totalBytes += bytes.length;
    } catch {
      this.disabled = true; // quota or blocked — keep streaming without cache
    }
  }

  async finish(): Promise<void> {
    if (this.disabled) return;
    try {
      await this.flush();
      if (!this.db) this.db = await openDb();
      await idbPut(this.db, META, this.url, {
        totalBytes: this.totalBytes,
        chunkCount: this.chunkIdx,
      } satisfies ShardMeta);
      this.db.close();
    } catch {
      this.disabled = true;
    }
  }
}

/** Best-effort: ask the browser not to evict the cache under pressure. */
export function requestPersistentStorage(): void {
  try {
    (navigator as unknown as { storage?: { persist?: () => Promise<boolean> } }).storage
      ?.persist?.()
      .catch(() => {});
  } catch {
    /* unavailable */
  }
}
