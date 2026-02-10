/**
 * Browser-compatible weight loading from HuggingFace.
 *
 * Fetches GPT-2 weights directly from HuggingFace CDN and parses
 * the safetensors format. Caches locally using IndexedDB.
 */

import {
  cacheWeights,
  getCachedWeights,
  cacheTokenizer,
  getCachedTokenizer,
  clearCache,
  getCacheInfo,
} from './cache';

// Re-export cache utilities for UI
export { clearCache, getCacheInfo };

const HF_BASE_URL = 'https://huggingface.co/openai-community/gpt2/resolve/main';

export type WeightData = {
  data: Float32Array;
  shape: number[];
};

export type ProgressCallback = (loaded: number, total: number, status: string) => void;

/**
 * Fetch GPT-2 weights from HuggingFace CDN (with local caching).
 */
export async function fetchGPT2Weights(
  onProgress?: ProgressCallback
): Promise<Map<string, WeightData>> {
  // Check cache first
  onProgress?.(0, 100, 'Checking local cache...');

  const cachedBuffer = await getCachedWeights();
  if (cachedBuffer) {
    onProgress?.(100, 100, 'Loading from cache...');
    const weights = parseSafetensors(cachedBuffer, onProgress);
    onProgress?.(100, 100, 'Loaded from cache!');
    return weights;
  }

  // Not cached, download from HuggingFace
  onProgress?.(0, 100, 'Downloading model weights (~500MB)...');

  const response = await fetch(`${HF_BASE_URL}/model.safetensors`);

  if (!response.ok) {
    throw new Error(`Failed to fetch weights: ${response.status} ${response.statusText}`);
  }

  const contentLength = response.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength, 10) : 0;

  // Stream the response for progress tracking
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Failed to get response reader');
  }

  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    chunks.push(value);
    loaded += value.length;

    if (total > 0) {
      const percent = Math.round((loaded / total) * 100);
      onProgress?.(loaded, total, `Downloading: ${percent}% (${formatBytes(loaded)} / ${formatBytes(total)})`);
    } else {
      onProgress?.(loaded, 0, `Downloading: ${formatBytes(loaded)}`);
    }
  }

  // Combine chunks
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const buffer = new ArrayBuffer(totalLength);
  const view = new Uint8Array(buffer);
  let offset = 0;
  for (const chunk of chunks) {
    view.set(chunk, offset);
    offset += chunk.length;
  }

  // Cache the buffer for next time
  onProgress?.(total, total, 'Caching locally...');
  try {
    await cacheWeights(buffer);
  } catch (e) {
    console.warn('Failed to cache weights:', e);
  }

  onProgress?.(total, total, 'Parsing weights...');

  // Parse safetensors
  return parseSafetensors(buffer, onProgress);
}

/**
 * Fetch tokenizer files from HuggingFace (with local caching).
 */
export async function fetchTokenizer(
  onProgress?: ProgressCallback
): Promise<{ vocab: Record<string, number>; merges: string[] }> {
  // Check cache first
  const cachedTokenizer = await getCachedTokenizer();
  if (cachedTokenizer) {
    onProgress?.(2, 2, 'Tokenizer loaded from cache!');
    return cachedTokenizer;
  }

  onProgress?.(0, 2, 'Fetching tokenizer...');

  const [vocabResponse, mergesResponse] = await Promise.all([
    fetch(`${HF_BASE_URL}/vocab.json`),
    fetch(`${HF_BASE_URL}/merges.txt`),
  ]);

  if (!vocabResponse.ok || !mergesResponse.ok) {
    throw new Error('Failed to fetch tokenizer files');
  }

  onProgress?.(1, 2, 'Parsing tokenizer...');

  const vocab = await vocabResponse.json();
  const mergesText = await mergesResponse.text();

  // Parse merges (skip first line which is a version comment)
  const merges = mergesText.split('\n').slice(1).filter((line) => line.trim());

  // Cache for next time
  try {
    await cacheTokenizer({ vocab, merges });
  } catch (e) {
    console.warn('Failed to cache tokenizer:', e);
  }

  onProgress?.(2, 2, 'Tokenizer ready');

  return { vocab, merges };
}

/**
 * Parse safetensors binary format.
 */
function parseSafetensors(
  buffer: ArrayBuffer,
  onProgress?: ProgressCallback
): Map<string, WeightData> {
  const view = new DataView(buffer);

  // Read header length (first 8 bytes as u64 little-endian)
  const headerLength = Number(view.getBigUint64(0, true));

  // Read header JSON
  const headerBytes = new Uint8Array(buffer, 8, headerLength);
  const headerText = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerText);

  const dataOffset = 8 + headerLength;
  const weights = new Map<string, WeightData>();

  const entries = Object.entries(header).filter(([key]) => key !== '__metadata__');
  let processed = 0;

  for (const [name, info] of entries) {
    const tensorInfo = info as {
      dtype: string;
      shape: number[];
      data_offsets: [number, number];
    };

    const [startOffset, endOffset] = tensorInfo.data_offsets;
    const tensorData = new Uint8Array(
      buffer,
      dataOffset + startOffset,
      endOffset - startOffset
    );

    // Convert to Float32Array based on dtype
    // Note: We must copy the data because the byte offset may not be aligned
    // (Float32Array requires 4-byte alignment, Uint16Array requires 2-byte alignment)
    let float32Data: Float32Array;

    switch (tensorInfo.dtype) {
      case 'F32': {
        // Copy to aligned buffer
        const alignedBuffer = new ArrayBuffer(tensorData.length);
        new Uint8Array(alignedBuffer).set(tensorData);
        float32Data = new Float32Array(alignedBuffer);
        break;
      }
      case 'F16':
        float32Data = convertFloat16ToFloat32(tensorData);
        break;
      case 'BF16':
        float32Data = convertBFloat16ToFloat32(tensorData);
        break;
      default:
        console.warn(`Unsupported dtype: ${tensorInfo.dtype} for ${name}`);
        continue;
    }

    weights.set(name, {
      data: float32Data,
      shape: tensorInfo.shape,
    });

    processed++;
    if (onProgress && processed % 10 === 0) {
      onProgress(processed, entries.length, `Loading tensors: ${processed}/${entries.length}`);
    }
  }

  return weights;
}

/**
 * Convert Float16 array to Float32.
 */
function convertFloat16ToFloat32(data: Uint8Array): Float32Array {
  // Copy to aligned buffer (Uint16Array requires 2-byte alignment)
  const alignedBuffer = new ArrayBuffer(data.length);
  new Uint8Array(alignedBuffer).set(data);
  const float16View = new Uint16Array(alignedBuffer);
  const float32 = new Float32Array(float16View.length);

  for (let i = 0; i < float16View.length; i++) {
    float32[i] = float16ToFloat32(float16View[i]);
  }

  return float32;
}

/**
 * Convert single Float16 to Float32.
 */
function float16ToFloat32(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exponent = (h & 0x7c00) >> 10;
  const fraction = h & 0x03ff;

  if (exponent === 0) {
    if (fraction === 0) {
      return sign ? -0 : 0;
    }
    // Subnormal
    return (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
  } else if (exponent === 0x1f) {
    if (fraction === 0) {
      return sign ? -Infinity : Infinity;
    }
    return NaN;
  }

  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

/**
 * Convert BFloat16 array to Float32.
 */
function convertBFloat16ToFloat32(data: Uint8Array): Float32Array {
  // Copy to aligned buffer (Uint16Array requires 2-byte alignment)
  const alignedBuffer = new ArrayBuffer(data.length);
  new Uint8Array(alignedBuffer).set(data);
  const bf16View = new Uint16Array(alignedBuffer);
  const float32 = new Float32Array(bf16View.length);

  for (let i = 0; i < bf16View.length; i++) {
    // BFloat16 is just the top 16 bits of Float32
    const asUint32 = bf16View[i] << 16;
    const float32View = new Float32Array(1);
    new Uint32Array(float32View.buffer)[0] = asUint32;
    float32[i] = float32View[0];
  }

  return float32;
}

/**
 * Format bytes as human-readable string.
 */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

/**
 * Serialize LoRA weights to safetensors format.
 */
export function serializeLoRAToSafetensors(
  tensors: Map<string, { data: Float32Array; shape: number[] }>,
  metadata: Record<string, string>
): ArrayBuffer {
  // Build header
  const header: Record<string, unknown> = {
    __metadata__: metadata,
  };

  let dataSize = 0;
  const tensorOffsets: Map<string, [number, number]> = new Map();

  for (const [name, { data, shape }] of tensors) {
    const startOffset = dataSize;
    const byteLength = data.byteLength;
    dataSize += byteLength;

    header[name] = {
      dtype: 'F32',
      shape,
      data_offsets: [startOffset, startOffset + byteLength],
    };
    tensorOffsets.set(name, [startOffset, startOffset + byteLength]);
  }

  // Serialize header
  const headerJson = JSON.stringify(header);
  const headerBytes = new TextEncoder().encode(headerJson);

  // Pad header to 8-byte alignment
  const paddedHeaderLen = Math.ceil(headerBytes.length / 8) * 8;

  // Build final buffer
  const totalSize = 8 + paddedHeaderLen + dataSize;
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const uint8View = new Uint8Array(buffer);

  // Write header length
  view.setBigUint64(0, BigInt(paddedHeaderLen), true);

  // Write header
  uint8View.set(headerBytes, 8);

  // Write tensor data
  let dataOffset = 8 + paddedHeaderLen;
  for (const [name, { data }] of tensors) {
    new Float32Array(buffer, dataOffset, data.length).set(data);
    dataOffset += data.byteLength;
  }

  return buffer;
}
