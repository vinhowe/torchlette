/**
 * Browser streaming weight loader for Gemma-2-2B: fetches HF-layout safetensors
 * (sharded or single-file) over HTTP and streams tensors into an instantiated
 * Gemma2 model with CONSTANT memory overhead — pieces are converted straight
 * into each tensor's final typed array through a fixed 8MB staging buffer.
 *
 * Mirrors packages/qwen3-browser/src/browser-loader.ts's stream/cache/resume
 * machinery. Gemma-2 deltas (all in the per-tensor handler):
 *  - RMSNorm weights (zero-centered) are baked to (1 + weight) before upload.
 *  - The 2.36GB f32 embedding table exceeds the 2GB storage-buffer BINDING
 *    limit, so it is uploaded via api.tensorFromArray + registerParameter (NOT
 *    copy_, whose stridedScatterCopy would bind the whole >2GB dest), and its
 *    CPU f32 data is retained to build the tied lm_head as independent sub-2GB
 *    vocab chunks (model.lmHeadChunks) after the stream completes.
 *  - Projection linears honor weightDtype (f16 fast path); norms + embedding
 *    stay f32.
 *
 * Works against huggingface.co resolve URLs directly (HF serves CORS) or any
 * static host with the same layout.
 */

import type { Tensor, Torchlette } from "torchlette";
import {
  CachedShardSource,
  getCachedShardMeta,
  requestPersistentStorage,
  ShardCacheWriter,
} from "./idb-cache";
import { configFromHF, Gemma2 } from "./model";
import {
  bf16SliceToF16Bits,
  bf16SliceToF32,
  f16SliceToF32,
  isNormWeight,
  resolveDest,
} from "./weights-map";

/** Anything that yields byte chunks (network stream or IndexedDB cache). */
type ByteSource = { next(): Promise<Uint8Array | null> };

export type LoadProgress = (
  loadedBytes: number,
  totalBytes: number,
  status: string,
) => void;

/**
 * Structured per-tensor load events (drives network visualizations).
 * Derived entirely from the safetensors manifest — architecture agnostic.
 */
export type TensorLoadEvent =
  | {
      type: "manifest";
      tensors: {
        name: string;
        shape: number[];
        elems: number;
        dtype: string;
        skipped: boolean;
      }[];
    }
  | { type: "start"; name: string }
  | { type: "progress"; name: string; fraction: number }
  | { type: "done"; name: string };

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
          throw new Error("Resume skip misaligned (chunk overshoot)");
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
  onSlice: (fraction: number) => void,
): Promise<Float32Array | Uint16Array> {
  if (srcDtype !== "BF16" && srcDtype !== "F16" && srcDtype !== "F32") {
    throw new Error(`Unsupported safetensors dtype ${srcDtype}`);
  }
  if (srcDtype === "F32" && destDtype === "f16") {
    throw new Error("F32→f16 load not supported (Gemma checkpoints are BF16)");
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
      new Uint8Array(dst.buffer, dst.byteOffset + dstElemOffset * 4, elems * 4).set(
        staging.subarray(0, elems * 4),
      );
    } else {
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
    const tail = stagingLen - elems * srcElemBytes;
    if (tail > 0) staging.copyWithin(0, elems * srcElemBytes, stagingLen);
    stagingLen = tail;
    onSlice(consumed / byteLength);
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
 * Stream the 2.36GB f32 embedding table in bounded ROW-BLOCKS so no single CPU
 * ArrayBuffer ever exceeds ~1GB — Chrome throws `RangeError: Array buffer
 * allocation failed` for the whole-table Float32Array (256128×2304×4 = 2.36GB
 * exceeds V8's per-ArrayBuffer allocation ceiling; verified live). Each block is
 * a fresh contiguous [rows, hidden] Float32Array handed to `onBlock`, which
 * uploads it to the GPU embedding buffer (a region DMA — no >2GB binding).
 * For BF16/F16 sources each block ALSO comes as raw f16 bits (converted
 * directly from the source — bf16→f16 is exact in the f16 normal range), so
 * the tied-lm_head vocab chunks can be uploaded at HALF the f32 footprint:
 * the lm_head matmul sweeps every chunk on every decoded token, so chunk
 * bytes are both resident memory AND per-token GPU bandwidth. Converts via
 * the same fixed 8MB staging buffer as streamTensor, yielding between slices.
 */
async function streamEmbeddingBlocks(
  cursor: ByteCursor,
  srcDtype: string,
  byteLength: number,
  hidden: number,
  rowsPerBlock: number,
  onSlice: (fraction: number) => void,
  onBlock: (
    block: Float32Array,
    blockF16: Uint16Array | null,
    startRow: number,
    rows: number,
  ) => Promise<void>,
): Promise<void> {
  if (srcDtype !== "BF16" && srcDtype !== "F16" && srcDtype !== "F32") {
    throw new Error(`Unsupported safetensors dtype ${srcDtype}`);
  }
  const srcElemBytes = srcDtype === "F32" ? 4 : 2;
  const wantF16 = srcDtype !== "F32"; // f32 checkpoints keep f32 chunks
  const totalElems = byteLength / srcElemBytes;
  const totalRows = totalElems / hidden;
  if (!Number.isInteger(totalRows)) {
    throw new Error(`Embedding element count ${totalElems} not a multiple of hidden ${hidden}`);
  }

  const staging = new Uint8Array(STAGING_BYTES);
  let stagingLen = 0;
  let consumed = 0;

  // Current block being filled.
  let startRow = 0;
  const blockElems = () =>
    Math.min(rowsPerBlock, totalRows - startRow) * hidden;
  let block = new Float32Array(blockElems());
  let blockF16 = wantF16 ? new Uint16Array(blockElems()) : null;
  let blockElemOffset = 0; // elements written into `block`

  const emitFullBlocks = async () => {
    // Flush whole staging elements into the current block, rolling over to a
    // new block whenever the current one fills.
    let elems = Math.floor(stagingLen / srcElemBytes);
    if (elems === 0) return;
    let stagingElemOffset = 0;
    while (elems > 0) {
      const room = block.length - blockElemOffset;
      const take = Math.min(room, elems);
      const src16Start = stagingElemOffset;
      if (srcDtype === "F32") {
        const srcF32 = new Float32Array(
          staging.buffer,
          src16Start * 4,
          take,
        );
        block.set(srcF32, blockElemOffset);
      } else {
        const src16 = new Uint16Array(staging.buffer, src16Start * 2, take);
        const out = block.subarray(blockElemOffset, blockElemOffset + take);
        (srcDtype === "BF16" ? bf16SliceToF32 : f16SliceToF32)(src16, out, 0, take);
        if (blockF16) {
          if (srcDtype === "F16") {
            blockF16.set(src16, blockElemOffset);
          } else {
            bf16SliceToF16Bits(
              src16,
              blockF16.subarray(blockElemOffset, blockElemOffset + take),
              0,
              take,
            );
          }
        }
      }
      blockElemOffset += take;
      stagingElemOffset += take;
      elems -= take;
      if (blockElemOffset === block.length) {
        const rows = block.length / hidden;
        await onBlock(block, blockF16, startRow, rows);
        startRow += rows;
        blockElemOffset = 0;
        if (totalRows - startRow > 0) {
          block = new Float32Array(blockElems());
          blockF16 = wantF16 ? new Uint16Array(blockElems()) : null;
        }
      }
    }
    // Retain any partial-element tail (bytes that don't complete an element).
    const consumedElemBytes =
      Math.floor(stagingLen / srcElemBytes) * srcElemBytes;
    const tail = stagingLen - consumedElemBytes;
    if (tail > 0) staging.copyWithin(0, consumedElemBytes, stagingLen);
    stagingLen = tail;
    onSlice(consumed / byteLength);
    await yieldToUI();
  };

  while (consumed < byteLength) {
    const p = await cursor.piece(
      Math.min(byteLength - consumed, STAGING_BYTES - stagingLen),
    );
    if (!p) throw new Error(`Stream ended mid-embedding (${byteLength - consumed} bytes left)`);
    staging.set(p, stagingLen);
    stagingLen += p.length;
    consumed += p.length;
    if (stagingLen === STAGING_BYTES || consumed === byteLength) {
      await emitFullBlocks();
    }
  }
  if (stagingLen !== 0) throw new Error("Embedding byte length not a multiple of element size");
  if (startRow !== totalRows) {
    throw new Error(`Embedding stream incomplete: ${startRow}/${totalRows} rows`);
  }
}

/**
 * Fetch config.json, instantiate the Gemma2 model, then stream all shards'
 * weights into it. Returns the ready model.
 */
export async function loadGemma2FromUrl(
  api: Torchlette,
  baseUrl: string,
  options?: {
    maxSeqLen?: number;
    weightDtype?: "f32" | "f16";
    onProgress?: LoadProgress;
    onTensorEvent?: (ev: TensorLoadEvent) => void;
  },
): Promise<Gemma2> {
  const base = baseUrl.replace(/\/$/, "");
  const progress = options?.onProgress ?? (() => {});
  const tensorEvent = options?.onTensorEvent ?? (() => {});

  progress(0, 0, "Fetching config…");
  const hfConfig = await (await fetch(`${base}/config.json`)).json();
  const config = configFromHF(
    hfConfig,
    options?.maxSeqLen ?? 2048,
    options?.weightDtype ?? "f16",
  );
  const model = new Gemma2(api, config, { device: "webgpu" });
  const hiddenSize = config.hiddenSize;
  const vocab = config.vocabSize;

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
  let fromCache = 0;
  const emitProgress = () =>
    progress(
      loadedBytes,
      totalBytes,
      `Loading weights${fromCache > 0 ? " (cached)" : ""}… ${(loadedBytes / 1e6).toFixed(0)} / ${(totalBytes / 1e6).toFixed(0)} MB` +
        (currentTensor ? ` · ${currentTensor}` : ""),
    );

  requestPersistentStorage();
  for (const shard of shardFiles) {
    const url = `${base}/${shard}`;
    const onBytes = (n: number) => {
      loadedBytes += n;
      if (loadedBytes - lastProgressAt >= PROGRESS_EVERY_BYTES) {
        lastProgressAt = loadedBytes;
        emitProgress();
      }
    };
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
      const maxEnd = Math.max(
        ...Object.entries(metadata)
          .filter(([n]) => n !== "__metadata__")
          .map(([, e]) => e.data_offsets[1]),
      );
      totalBytes = 8 + headerLen + maxEnd;
    }

    const entries = Object.entries(metadata)
      .filter(([name]) => name !== "__metadata__")
      .sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);

    tensorEvent({
      type: "manifest",
      tensors: entries.map(([name, info]) => ({
        name,
        shape: info.shape,
        elems: info.shape.reduce((a, b) => a * b, 1),
        dtype: info.dtype,
        skipped: resolveDest(model, name) === null,
      })),
    });

    let filePos = 0; // relative to data start
    for (const [name, info] of entries) {
      const [start, end] = info.data_offsets;
      if (start > filePos) await cursor.discard(start - filePos);
      filePos = end;

      const dest = resolveDest(model, name);
      if (!dest) {
        // Skipped tensor (tied lm_head): discard without allocating.
        currentTensor = `${name} (skipped)`;
        tensorEvent({ type: "start", name });
        await cursor.discard(end - start);
        tensorEvent({ type: "done", name });
        continue;
      }
      const isNorm = isNormWeight(name);
      const isEmbed = name === "model.embed_tokens.weight";
      currentTensor = name;
      tensorEvent({ type: "start", name });

      if (isEmbed) {
        // 2.36GB f32 embedding: streaming the whole table into ONE Float32Array
        // throws `RangeError: Array buffer allocation failed` in Chrome (exceeds
        // V8's per-ArrayBuffer ceiling). Stream it in <2GB row-BLOCKS instead:
        //  - each block is a fresh [rows, hidden] GPU tensor uploaded via
        //    tensorFromArray (chunked writeBuffer, no binding);
        //  - copyInto_ DMAs it into the pre-zeroed embedding buffer at its row
        //    offset — a contiguous buffer-to-buffer region write, which never
        //    binds the >2GB dest (unlike the strided-scatter kernel path);
        //  - each block is ALSO retained as a tied-lm_head vocab chunk (matmul
        //    binds a weight operand whole, so lm_head must be sub-2GB
        //    INDEPENDENT buffers) — as f16 when the checkpoint is BF16/F16:
        //    bf16→f16 is exact in range, the f32-activation × f16-weight
        //    matmul is the same mixed-dtype class as every projection linear,
        //    and the lm_head sweeps ALL chunk bytes on EVERY decoded token, so
        //    f16 halves both resident memory and per-token bandwidth (2.36GB →
        //    1.18GB — the single biggest lever on a 16GB Mac).
        // No CPU or GPU allocation ever exceeds one block (~1GB).
        const rowsPerBlock = model.lmHeadChunkRows(); // <2GB per block
        const embedWeight = model.embedTokens.weight;
        const needLmHeadChunks = vocab * hiddenSize * 4 > (1 << 31) - 4;
        const lmHeadChunks: Tensor[] = [];
        await streamEmbeddingBlocks(
          cursor,
          info.dtype,
          end - start,
          hiddenSize,
          rowsPerBlock,
          (fraction) => {
            emitProgress();
            tensorEvent({ type: "progress", name, fraction });
          },
          async (blockArr, blockF16, startRow, rows) => {
            currentTensor = `${name} (rows ${startRow}..${startRow + rows})`;
            emitProgress();
            const blockTensor = api.tensorFromArray(blockArr, [rows, hiddenSize], {
              dtype: dest.dtype,
              device: "webgpu",
            });
            // Region DMA into the single embedding buffer (used by the gather,
            // which auto-chunks reads of the >2GB table).
            api.runtime.copyInto_(
              embedWeight._unwrap(),
              startRow * hiddenSize,
              blockTensor._unwrap(),
            );
            if (needLmHeadChunks) {
              lmHeadChunks.push(
                blockF16
                  ? api.tensorFromArray(blockF16, [rows, hiddenSize], {
                      dtype: "f16",
                      device: "webgpu",
                    })
                  : blockTensor, // f32 checkpoint: reuse the f32 block
              );
            }
            await api.markStep();
          },
        );
        if (needLmHeadChunks) model.lmHeadChunks = lmHeadChunks;
        tensorEvent({ type: "done", name });
        loadedTensors++;
        currentTensor = name;
        emitProgress();
        continue;
      }

      const data = await streamTensor(cursor, info.dtype, end - start, dest.dtype, (fraction) => {
        emitProgress();
        tensorEvent({ type: "progress", name, fraction });
      });
      tensorEvent({ type: "done", name });
      currentTensor = `${name} (upload)`;
      emitProgress();

      if (isNorm) {
        // Gemma RMSNorm: x_normed * (1 + weight). Bake +1 so the stock fused
        // kernel (x_normed * weight) is exactly correct. Norms are f32.
        const arr = data as Float32Array;
        for (let i = 0; i < arr.length; i++) arr[i] += 1.0;
        const src = api.tensorFromArray(arr, info.shape, { dtype: dest.dtype });
        dest.copy_(src);
      } else {
        const src = api.tensorFromArray(data, info.shape, { dtype: dest.dtype });
        dest.copy_(src);
      }

      loadedTensors++;
      pendingBytes += data.byteLength;
      await yieldToUI();
      if (pendingBytes >= FLUSH_THRESHOLD) {
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
