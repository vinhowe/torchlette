/**
 * IndexedDB chunk cache for streamed weight shards (browser/worker only).
 *
 * Layout: DB "gemma2-weights-cache"
 *   store "chunks": key `${url}#${i}` → ArrayBuffer (CHUNK_BYTES each)
 *   store "meta":   key url → { totalBytes, chunkCount }   (written LAST —
 *                   presence of meta means the entry is complete; partial
 *                   downloads are invisible and get overwritten next time)
 *
 * All failures (quota, private browsing, blocked) degrade to "no cache":
 * writes disable themselves, reads return null.
 */

const DB_NAME = "gemma2-weights-cache";
const CHUNKS = "chunks";
const META = "meta";
const PACKED = "packed";
// v2 adds the "packed" store (quantized-operand cache). One version for the
// whole DB so the shard cache and the packed cache never race on open().
const DB_VERSION = 2;
export const CACHE_CHUNK_BYTES = 32 * 1024 * 1024;

type ShardMeta = { totalBytes: number; chunkCount: number };

function upgrade(db: IDBDatabase): void {
  if (!db.objectStoreNames.contains(CHUNKS)) db.createObjectStore(CHUNKS);
  if (!db.objectStoreNames.contains(META)) db.createObjectStore(META);
  if (!db.objectStoreNames.contains(PACKED)) db.createObjectStore(PACKED);
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => upgrade(req.result);
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

// ============================================================================
// Packed-weight (quantized operand) cache
// ============================================================================
//
// The PACKED int8 form of a projection weight is cached keyed by
// `${url}#${name}#${scheme}g${G}` — conversion (host-side quantize) is
// expensive, so we cache the packed result, not re-derive it every load. The
// key versions by scheme+groupSize so a format change re-quantizes rather than
// silently reusing a stale packing. Store "packed": key → { packed, scales,
// n, k } (transferable ArrayBuffers). Failures degrade to "no cache".

type PackedRecord = {
  packed: ArrayBuffer; // Uint32Array bytes [N, K/4]
  scales: ArrayBuffer; // Uint16Array bytes [N, K/G]
  n: number;
  k: number;
};

/** Packed-cache key: url + tensor name + format (scheme+groupSize). */
export function packedKey(
  url: string,
  name: string,
  scheme: string,
  groupSize: number,
): string {
  return `${url}#${name}#${scheme}g${groupSize}`;
}

/** Read a cached packed weight, or null if absent/unavailable. */
export async function getCachedPacked(
  key: string,
): Promise<{ packed: Uint32Array; scales: Uint16Array; n: number; k: number } | null> {
  try {
    if (typeof indexedDB === "undefined") return null;
    const db = await openDb();
    const rec = await idbGet<PackedRecord>(db, PACKED, key);
    db.close();
    if (!rec) return null;
    return {
      packed: new Uint32Array(rec.packed),
      scales: new Uint16Array(rec.scales),
      n: rec.n,
      k: rec.k,
    };
  } catch {
    return null;
  }
}

/** Store a packed weight. Best-effort; failures are swallowed. */
export async function putCachedPacked(
  key: string,
  packed: Uint32Array,
  scales: Uint16Array,
  n: number,
  k: number,
): Promise<void> {
  try {
    if (typeof indexedDB === "undefined") return;
    const db = await openDb();
    // Copy to standalone ArrayBuffers (subarrays may share a larger buffer).
    const p = packed.slice().buffer;
    const s = scales.slice().buffer;
    await idbPut(db, PACKED, key, { packed: p, scales: s, n, k } satisfies PackedRecord);
    db.close();
  } catch {
    /* quota / blocked — proceed without cache */
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
