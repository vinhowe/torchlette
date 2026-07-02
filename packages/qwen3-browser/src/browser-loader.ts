/**
 * Browser streaming weight loader: fetches HF-layout safetensors (sharded or
 * single-file) over HTTP and streams tensors into an instantiated Qwen3 model
 * with CONSTANT memory overhead — pieces are converted straight into each
 * tensor's final typed array through a fixed 8MB staging buffer (no full-file
 * buffer, no assembled-bytes copy, no double conversion).
 *
 * Resilience: transparently resumes with HTTP Range on mid-stream network
 * failures (HF supports Range); progress is emitted on raw received bytes so
 * the UI moves continuously and a stall is visible as a stall.
 *
 * Works against huggingface.co resolve URLs directly (HF serves CORS) or any
 * static host with the same layout:
 *   {base}/config.json
 *   {base}/model.safetensors.index.json   (sharded)
 *   {base}/model-0000X-of-0000Y.safetensors | model.safetensors
 */

import type { Torchlette } from "torchlette";
import {
  CachedShardSource,
  getCachedShardMeta,
  requestPersistentStorage,
  ShardCacheWriter,
} from "./idb-cache";
import { configFromHF, Qwen3 } from "./model";
import {
  bf16SliceToF16Bits,
  bf16SliceToF32,
  f16SliceToF32,
  resolveDest,
} from "./weights-map";

/** Anything that yields byte chunks (network stream or IndexedDB cache). */
type ByteSource = { next(): Promise<Uint8Array | null> };

export type LoadProgress = (
  loadedBytes: number,
  totalBytes: number,
  status: string,
) => void;

type SafetensorsEntry = {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
};

// Main-thread budget: convert in small units and yield to the event loop so
// the tab never blocks (a 300M-element synchronous loop freezes the page).
const STAGING_BYTES = 8 * 1024 * 1024;
const PROGRESS_EVERY_BYTES = 8 * 1024 * 1024;
const MAX_STREAM_RETRIES = 4;
const yieldToUI = () => new Promise<void>((r) => setTimeout(r, 0));

/**
 * Byte stream over a URL that survives connection drops: on failure it
 * reopens with `Range: bytes=<delivered>-` (up to MAX_STREAM_RETRIES per
 * shard) and continues where it left off.
 */
class ResumableStream implements ByteSource {
  private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  private delivered = 0;
  private retries = 0;

  constructor(
    private url: string,
    private onBytes: (n: number) => void,
    /** Optional tee for the weights cache (awaited so writes backpressure). */
    private onChunk?: (bytes: Uint8Array) => Promise<void>,
  ) {}

  private async open(): Promise<void> {
    const headers: Record<string, string> = {};
    if (this.delivered > 0) headers.Range = `bytes=${this.delivered}-`;
    const res = await fetch(this.url, { headers });
    if (!res.ok || !res.body) {
      throw new Error(`fetch ${this.url}: ${res.status} ${res.statusText}`);
    }
    this.reader = res.body.getReader();
    // Server ignored Range (200 instead of 206): discard what we already have.
    if (this.delivered > 0 && res.status === 200) {
      let toSkip = this.delivered;
      while (toSkip > 0) {
        const { done, value } = await this.reader.read();
        if (done) throw new Error("Stream ended while re-skipping after resume");
        toSkip -= value.length;
        if (toSkip < 0) {
          throw new Error("Resume skip misaligned (chunk overshoot)"); // keep it simple: chunks are small, overshoot is pathological
        }
      }
    }
  }

  /** Next chunk, or null at end of stream. */
  async next(): Promise<Uint8Array | null> {
    for (;;) {
      try {
        if (!this.reader) await this.open();
        const { done, value } = await this.reader!.read();
        if (done) return null;
        this.delivered += value.length;
        this.onBytes(value.length);
        if (this.onChunk) await this.onChunk(value);
        return value;
      } catch (e) {
        this.reader = null;
        this.retries++;
        if (this.retries > MAX_STREAM_RETRIES) throw e;
        await new Promise((r) => setTimeout(r, 1000 * this.retries));
      }
    }
  }
}

/** Pull-based byte cursor over a ByteSource with piecewise consumption. */
class ByteCursor {
  private chunk: Uint8Array | null = null;
  private offset = 0;

  constructor(private stream: ByteSource) {}

  /** Up to `max` bytes (at least 1 unless EOF → null). Advances the cursor. */
  async piece(max: number): Promise<Uint8Array | null> {
    if (!this.chunk || this.offset >= this.chunk.length) {
      this.chunk = await this.stream.next();
      this.offset = 0;
      if (!this.chunk) return null;
    }
    const n = Math.min(max, this.chunk.length - this.offset);
    const out = this.chunk.subarray(this.offset, this.offset + n);
    this.offset += n;
    return out;
  }

  /** Exactly n bytes, assembled (only for SMALL reads: headers). */
  async takeExact(n: number): Promise<Uint8Array> {
    const out = new Uint8Array(n);
    let written = 0;
    while (written < n) {
      const p = await this.piece(n - written);
      if (!p) throw new Error(`Stream ended ${n - written} bytes early`);
      out.set(p, written);
      written += p.length;
    }
    return out;
  }

  /** Discard exactly n bytes without assembling them. */
  async discard(n: number): Promise<void> {
    let left = n;
    while (left > 0) {
      const p = await this.piece(left);
      if (!p) throw new Error(`Stream ended ${left} bytes early (discard)`);
      left -= p.length;
    }
  }
}

/**
 * Stream one tensor's payload into its destination typed array, converting
 * through a fixed staging buffer. Returns the filled array:
 * Uint16Array (raw f16 bits) for f16 dests, Float32Array for f32 dests.
 */
async function streamTensor(
  cursor: ByteCursor,
  srcDtype: string,
  byteLength: number,
  destDtype: string,
  onSlice: () => void,
): Promise<Float32Array | Uint16Array> {
  if (srcDtype !== "BF16" && srcDtype !== "F16" && srcDtype !== "F32") {
    throw new Error(`Unsupported safetensors dtype ${srcDtype}`);
  }
  if (srcDtype === "F32" && destDtype === "f16") {
    throw new Error("F32→f16 load not supported (Qwen checkpoints are BF16)");
  }
  const srcElemBytes = srcDtype === "F32" ? 4 : 2;
  const numElems = byteLength / srcElemBytes;
  const dst =
    destDtype === "f16" ? new Uint16Array(numElems) : new Float32Array(numElems);

  const staging = new Uint8Array(STAGING_BYTES);
  let stagingLen = 0;
  let dstElemOffset = 0;
  let consumed = 0;

  const flushStaging = async () => {
    const elems = Math.floor(stagingLen / srcElemBytes);
    if (elems === 0) return;
    if (srcDtype === "F32") {
      // f32 → f32 only (guarded above): aligned copy into dst bits.
      new Uint8Array(dst.buffer, dst.byteOffset + dstElemOffset * 4, elems * 4).set(
        staging.subarray(0, elems * 4),
      );
    } else {
      // Aligned u16 view of the staged bytes.
      const src16 = new Uint16Array(staging.buffer, 0, elems);
      if (destDtype === "f16") {
        const out = dst as Uint16Array;
        if (srcDtype === "F16") {
          out.set(src16.subarray(0, elems), dstElemOffset);
        } else {
          bf16SliceToF16Bits(src16, out.subarray(dstElemOffset, dstElemOffset + elems) as Uint16Array, 0, elems);
        }
      } else {
        const out = (dst as Float32Array).subarray(dstElemOffset, dstElemOffset + elems);
        (srcDtype === "BF16" ? bf16SliceToF32 : f16SliceToF32)(src16, out, 0, elems);
      }
    }
    dstElemOffset += elems;
    // Carry any trailing partial element to the front.
    const tail = stagingLen - elems * srcElemBytes;
    if (tail > 0) staging.copyWithin(0, elems * srcElemBytes, stagingLen);
    stagingLen = tail;
    onSlice();
    await yieldToUI();
  };

  while (consumed < byteLength) {
    const p = await cursor.piece(
      Math.min(byteLength - consumed, STAGING_BYTES - stagingLen),
    );
    if (!p) throw new Error(`Stream ended mid-tensor (${byteLength - consumed} bytes left)`);
    staging.set(p, stagingLen);
    stagingLen += p.length;
    consumed += p.length;
    if (stagingLen === STAGING_BYTES || consumed === byteLength) {
      await flushStaging();
    }
  }
  if (stagingLen !== 0) throw new Error("Tensor byte length not a multiple of element size");
  return dst;
}

/**
 * Fetch config.json, instantiate the model, then stream all shards' weights
 * into it. Returns the ready model.
 */
export async function loadQwen3FromUrl(
  api: Torchlette,
  baseUrl: string,
  options?: {
    maxSeqLen?: number;
    weightDtype?: "f32" | "f16";
    onProgress?: LoadProgress;
  },
): Promise<Qwen3> {
  const base = baseUrl.replace(/\/$/, "");
  const progress = options?.onProgress ?? (() => {});

  progress(0, 0, "Fetching config…");
  const hfConfig = await (await fetch(`${base}/config.json`)).json();
  const config = configFromHF(
    hfConfig,
    options?.maxSeqLen ?? 2048,
    options?.weightDtype ?? "f16",
  );
  const model = new Qwen3(api, config, { device: "webgpu" });

  // Shard list + total size.
  let shardFiles: string[];
  let totalBytes = 0;
  const idxRes = await fetch(`${base}/model.safetensors.index.json`);
  if (idxRes.ok) {
    const index = await idxRes.json();
    shardFiles = [...new Set(Object.values(index.weight_map as Record<string, string>))];
    totalBytes = index.metadata?.total_size ?? 0;
  } else {
    shardFiles = ["model.safetensors"];
  }

  let loadedBytes = 0;
  let lastProgressAt = 0;
  let loadedTensors = 0;
  let pendingBytes = 0;
  let currentTensor = "";
  const FLUSH_THRESHOLD = 256 * 1024 * 1024;
  const emitProgress = () =>
    progress(
      loadedBytes,
      totalBytes,
      `Loading weights${fromCache > 0 ? " (cached)" : ""}… ${(loadedBytes / 1e6).toFixed(0)} / ${(totalBytes / 1e6).toFixed(0)} MB` +
        (currentTensor ? ` · ${currentTensor}` : ""),
    );

  requestPersistentStorage();
  let fromCache = 0;
  for (const shard of shardFiles) {
    const url = `${base}/${shard}`;
    const onBytes = (n: number) => {
      loadedBytes += n;
      if (loadedBytes - lastProgressAt >= PROGRESS_EVERY_BYTES) {
        lastProgressAt = loadedBytes;
        emitProgress();
      }
    };
    // Complete IndexedDB entry → read locally; otherwise stream + tee into
    // the cache (meta written last, so partial downloads never poison it).
    const cachedMeta = await getCachedShardMeta(url);
    let source: ByteSource;
    let cacheWriter: ShardCacheWriter | null = null;
    if (cachedMeta) {
      fromCache++;
      source = new CachedShardSource(url, cachedMeta, onBytes);
    } else {
      cacheWriter = new ShardCacheWriter(url);
      source = new ResumableStream(url, onBytes, (bytes) => cacheWriter!.append(bytes));
    }
    const cursor = new ByteCursor(source);

    // Header
    const lenBytes = await cursor.takeExact(8);
    const headerLen = Number(new DataView(lenBytes.buffer, lenBytes.byteOffset, 8).getBigUint64(0, true));
    const headerJson = new TextDecoder().decode(await cursor.takeExact(headerLen));
    const metadata = JSON.parse(headerJson) as Record<string, SafetensorsEntry>;
    if (totalBytes === 0) {
      // Single-file repo without index: total = header + payload extent.
      const maxEnd = Math.max(
        ...Object.entries(metadata)
          .filter(([n]) => n !== "__metadata__")
          .map(([, e]) => e.data_offsets[1]),
      );
      totalBytes = 8 + headerLen + maxEnd;
    }

    // Entries in file order.
    const entries = Object.entries(metadata)
      .filter(([name]) => name !== "__metadata__")
      .sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);

    let filePos = 0; // relative to data start
    for (const [name, info] of entries) {
      const [start, end] = info.data_offsets;
      if (start > filePos) await cursor.discard(start - filePos);
      filePos = end;

      const dest = resolveDest(model, name);
      if (!dest) {
        // Skipped tensor (e.g. redundant tied lm_head): discard WITHOUT
        // allocating — progress still ticks via the stream byte counter.
        currentTensor = `${name} (skipped)`;
        await cursor.discard(end - start);
        continue;
      }
      currentTensor = name;
      const data = await streamTensor(cursor, info.dtype, end - start, dest.dtype, emitProgress);
      currentTensor = `${name} (upload)`;
      emitProgress();
      const src = api.tensorFromArray(data, info.shape, { dtype: dest.dtype });
      dest.copy_(src);
      loadedTensors++;
      pendingBytes += data.byteLength;
      await yieldToUI();
      if (pendingBytes >= FLUSH_THRESHOLD) {
        // The GPU flush is where lazy uploads + copies actually execute — if
        // the load hangs here, the status names the phase and last tensor.
        currentTensor = `${name} (GPU flush)`;
        emitProgress();
        await api.markStep();
        pendingBytes = 0;
      }
      currentTensor = name;
      emitProgress();
    }
    await cacheWriter?.finish();
  }
  currentTensor = "final GPU flush";
  emitProgress();
  await api.markStep();
  progress(totalBytes, totalBytes, `Loaded ${loadedTensors} tensors`);
  return model;
}
