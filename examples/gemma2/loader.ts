/**
 * Gemma-2 Weight Loading (Node).
 *
 * Streams the HF safetensors (bf16) per-tensor from disk into an instantiated
 * Gemma2 model — peak JS memory is one weight at a time. HF Linear weights are
 * [out, in], same layout as torchlette nn.Linear: no transposes. lm_head is
 * tied (absent in the file). RMSNorm weights are BAKED to (1 + weight) at load
 * so the stock fused kernel (x_normed * weight) reproduces Gemma's
 * x_normed * (1 + weight) exactly.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { Tensor, Torchlette } from "torchlette";
import { configFromHF, Gemma2, type Gemma2Config } from "../../packages/gemma2-browser/src/model";
import { Gemma2Block } from "../../packages/gemma2-browser/src/model";

type SafetensorsMetadata = {
  [key: string]: {
    dtype: string;
    shape: number[];
    data_offsets: [number, number];
  };
};

function parseSafetensorsHeader(fd: number): {
  metadata: SafetensorsMetadata;
  dataStart: number;
} {
  const headerSizeBuf = Buffer.alloc(8);
  fs.readSync(fd, headerSizeBuf, 0, 8, 0);
  const headerLength = Number(
    new DataView(headerSizeBuf.buffer, headerSizeBuf.byteOffset, 8).getBigUint64(0, true),
  );
  const headerBuf = Buffer.alloc(headerLength);
  fs.readSync(fd, headerBuf, 0, headerLength, 8);
  const metadata: SafetensorsMetadata = JSON.parse(new TextDecoder().decode(headerBuf));
  return { metadata, dataStart: 8 + headerLength };
}

function float16ToFloat32(bits: number): number {
  const sign = (bits >> 15) & 0x1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;
  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * 2 ** -14 * (frac / 1024);
  }
  if (exp === 31) return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
  return (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + frac / 1024);
}

const bf16Buf = new ArrayBuffer(4);
const bf16U32 = new Uint32Array(bf16Buf);
const bf16F32 = new Float32Array(bf16Buf);
function bfloat16ToFloat32(bits: number): number {
  bf16U32[0] = bits << 16;
  return bf16F32[0];
}

/** f32 value → IEEE-754 half (f16) raw 16-bit pattern (round-to-nearest-even,
 *  with subnormal/overflow handling). Used to upload f16 weights via the fast
 *  raw-bits path (no per-element Array.from through the f32 creation route). */
const f16sBuf = new ArrayBuffer(4);
const f16sF32 = new Float32Array(f16sBuf);
const f16sU32 = new Uint32Array(f16sBuf);
function float32ToFloat16Bits(v: number): number {
  f16sF32[0] = v;
  const x = f16sU32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = x & 0x7fffff;
  if (exp <= 0) {
    // Subnormal or zero.
    if (exp < -10) return sign;
    const m = (mant | 0x800000) >>> (1 - exp);
    // Round to nearest even.
    const rounded = (m + 0x1000) >>> 13;
    return sign | rounded;
  }
  if (exp >= 0x1f) {
    // Overflow → inf (or NaN if mant set).
    return sign | 0x7c00 | (mant ? 0x200 : 0);
  }
  // Normalized: round mantissa to 10 bits (round-to-nearest-even).
  let out = sign | (exp << 10) | (mant >>> 13);
  if (mant & 0x1000) {
    // Round up; carry propagates into exp automatically via +1.
    out += 1;
  }
  return out;
}

/** Convert a raw bf16 Uint16 buffer directly to f16 raw bits (skips the
 *  f32 intermediate array + Array.from on upload — the load-time hot loop). */
function bf16BufferToF16Bits(u16: Uint16Array): Uint16Array {
  const out = new Uint16Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    bf16U32[0] = u16[i] << 16;
    out[i] = float32ToFloat16Bits(bf16F32[0]);
  }
  return out;
}

function extractWeight(
  fd: number,
  dataStart: number,
  info: { dtype: string; shape: number[]; data_offsets: [number, number] },
): Float32Array {
  const [startOffset, endOffset] = info.data_offsets;
  const byteLength = endOffset - startOffset;
  const rawBuf = Buffer.alloc(byteLength);
  fs.readSync(fd, rawBuf, 0, byteLength, dataStart + startOffset);

  if (info.dtype === "F32") {
    return new Float32Array(rawBuf.buffer, rawBuf.byteOffset, byteLength / 4);
  }
  if (info.dtype === "F16" || info.dtype === "BF16") {
    const u16 = new Uint16Array(rawBuf.buffer, rawBuf.byteOffset, byteLength / 2);
    const out = new Float32Array(u16.length);
    const conv = info.dtype === "F16" ? float16ToFloat32 : bfloat16ToFloat32;
    for (let i = 0; i < u16.length; i++) out[i] = conv(u16[i]);
    return out;
  }
  throw new Error(`Unsupported safetensors dtype ${info.dtype}`);
}

/** RMSNorm weights (all 5 kinds) are zero-centered in Gemma → bake +1. */
function isNormWeight(hfName: string): boolean {
  return hfName.endsWith("layernorm.weight") || hfName === "model.norm.weight";
}

/** Resolve an HF weight name to the destination tensor, or null to skip. */
function resolveDest(model: Gemma2, hfName: string): Tensor | null {
  if (hfName === "lm_head.weight") return null; // tied to embed_tokens
  if (hfName === "model.embed_tokens.weight") return model.embedTokens.weight;
  if (hfName === "model.norm.weight") return model.norm.weight;

  const m = hfName.match(/^model\.layers\.(\d+)\.(.+)$/);
  if (!m) return null;
  const block = model.layers.get(Number(m[1])) as Gemma2Block;
  switch (m[2]) {
    case "self_attn.q_proj.weight":
      return block.attn.qProj.weight;
    case "self_attn.k_proj.weight":
      return block.attn.kProj.weight;
    case "self_attn.v_proj.weight":
      return block.attn.vProj.weight;
    case "self_attn.o_proj.weight":
      return block.attn.oProj.weight;
    case "input_layernorm.weight":
      return block.inputNorm.weight;
    case "post_attention_layernorm.weight":
      return block.postAttnNorm.weight;
    case "pre_feedforward_layernorm.weight":
      return block.preFeedforwardNorm.weight;
    case "post_feedforward_layernorm.weight":
      return block.postFeedforwardNorm.weight;
    case "mlp.gate_proj.weight":
      return block.mlp.gateProj.weight;
    case "mlp.up_proj.weight":
      return block.mlp.upProj.weight;
    case "mlp.down_proj.weight":
      return block.mlp.downProj.weight;
    default:
      return null;
  }
}

export async function loadPretrainedGemma2(
  api: Torchlette,
  modelDir: string,
  options?: { maxSeqLen?: number; weightDtype?: "f32" | "f16" },
): Promise<Gemma2> {
  const hfConfig = JSON.parse(
    await fs.promises.readFile(path.join(modelDir, "config.json"), "utf-8"),
  );
  const config: Gemma2Config = configFromHF(
    hfConfig,
    options?.maxSeqLen ?? 4096,
    options?.weightDtype ?? "f32",
  );
  console.log("Creating Gemma2 model:", {
    ...config,
    layerTypes: `${config.layerTypes[0]}/${config.layerTypes[1]}...(${config.layerTypes.length})`,
  });
  const model = new Gemma2(api, config, { device: "webgpu" });

  const indexPath = path.join(modelDir, "model.safetensors.index.json");
  let shardFiles: string[];
  if (fs.existsSync(indexPath)) {
    const index = JSON.parse(await fs.promises.readFile(indexPath, "utf-8"));
    shardFiles = [...new Set(Object.values(index.weight_map as Record<string, string>))];
  } else {
    shardFiles = ["model.safetensors"];
  }

  let loaded = 0;
  const skipped: string[] = [];
  let pendingBytes = 0;
  const FLUSH_THRESHOLD = 128 * 1024 * 1024;
  // Keep the embedding f32 CPU data to build the tied lm_head as INDEPENDENT
  // sub-2GB weight buffers (see Gemma2.lmHeadChunks). The 256k×2304 f32 table
  // is 2.36GB — over the 2GB storage-buffer binding limit — and matmul binds a
  // weight operand whole, so a narrow view of the shared embedding buffer can't
  // be used. Building separate chunk buffers is the sub-2GB path.
  let embedData: Float32Array | null = null;
  // f16 embedding path (#59): keep the raw f16 bits to build the tied lm_head
  // chunks as f16 too (no f32 round-trip). Set only when the embed loads f16.
  let embedF16Bits: Uint16Array | null = null;
  const hiddenSize = config.hiddenSize;

  for (const shard of shardFiles) {
    const fd = fs.openSync(path.join(modelDir, shard), "r");
    try {
      const { metadata, dataStart } = parseSafetensorsHeader(fd);
      for (const [name, info] of Object.entries(metadata)) {
        if (name === "__metadata__") continue;
        const dest = resolveDest(model, name);
        if (!dest) {
          if (name !== "lm_head.weight") skipped.push(name);
          continue;
        }
        if (JSON.stringify(dest.shape) !== JSON.stringify(info.shape)) {
          throw new Error(
            `Shape mismatch for ${name}: model ${JSON.stringify(dest.shape)} vs file ${JSON.stringify(info.shape)}`,
          );
        }
        const isNorm = isNormWeight(name);
        const isEmbed = name === "model.embed_tokens.weight";
        // Fast f16 path: for f16 destinations (the projection linears — NOT
        // norms/embedding, which stay f32), convert bf16 → f16 raw bits and
        // upload directly, skipping the f32 intermediate + the slow per-element
        // Array.from in the f32 creation route (the load-time bottleneck).
        const wantF16Bits =
          dest.dtype === "f16" && !isNorm && !isEmbed && (info.dtype === "BF16" || info.dtype === "F16");
        const byteBytes = info.data_offsets[1] - info.data_offsets[0];

        if (wantF16Bits) {
          const raw = Buffer.alloc(byteBytes);
          fs.readSync(fd, raw, 0, byteBytes, dataStart + info.data_offsets[0]);
          const u16 = new Uint16Array(raw.buffer, raw.byteOffset, byteBytes / 2);
          const f16bits =
            info.dtype === "F16" ? u16.slice() : bf16BufferToF16Bits(u16);
          const src = api.tensorFromArray(f16bits, info.shape, { dtype: "f16" });
          dest.copy_(src);
          loaded++;
          pendingBytes += byteBytes;
          if (pendingBytes >= FLUSH_THRESHOLD) {
            await api.markStep();
            pendingBytes = 0;
          }
          continue;
        }

        // f16 embedding path (#59): convert bf16/f16 → f16 raw bits and upload
        // as f16 directly, skipping the 2.36GB f32 intermediate (which would
        // defeat the residency win at load time). Keep the bits to build the
        // tied lm_head chunks as f16. The 1.18GB table can't be written with
        // copy_ (stridedScatterCopy binds the whole dest > 2GB) — upload via
        // tensorFromArray (chunked writeBuffer, no binding) + re-register.
        if (isEmbed && dest.dtype === "f16") {
          const raw = Buffer.alloc(byteBytes);
          fs.readSync(fd, raw, 0, byteBytes, dataStart + info.data_offsets[0]);
          const u16 = new Uint16Array(raw.buffer, raw.byteOffset, byteBytes / 2);
          embedF16Bits =
            info.dtype === "F16" ? u16.slice() : bf16BufferToF16Bits(u16);
          const uploaded = api.tensorFromArray(embedF16Bits, info.shape, {
            dtype: "f16",
            device: "webgpu",
          });
          model.embedTokens.registerParameter("weight", uploaded);
          loaded++;
          pendingBytes += embedF16Bits.byteLength;
          if (pendingBytes >= FLUSH_THRESHOLD) {
            await api.markStep();
            pendingBytes = 0;
          }
          continue;
        }

        const data = extractWeight(fd, dataStart, info);
        if (isNorm) {
          // Gemma RMSNorm: x_normed * (1 + weight). Bake +1 into the weight so
          // the stock fused kernel (x_normed * weight) is exactly correct.
          for (let i = 0; i < data.length; i++) data[i] += 1.0;
        }
        if (isEmbed) {
          // f32 embedding: 2.36GB. Same no-binding upload path as above.
          embedData = data;
          const uploaded = api.tensorFromArray(data, info.shape, {
            dtype: dest.dtype,
            device: "webgpu",
          });
          model.embedTokens.registerParameter("weight", uploaded);
          loaded++;
          pendingBytes += data.byteLength;
          if (pendingBytes >= FLUSH_THRESHOLD) {
            await api.markStep();
            pendingBytes = 0;
          }
          continue;
        }
        const src = api.tensorFromArray(data, info.shape, { dtype: dest.dtype });
        dest.copy_(src);
        loaded++;
        pendingBytes += data.byteLength;
        if (pendingBytes >= FLUSH_THRESHOLD) {
          await api.markStep();
          pendingBytes = 0;
        }
      }
    } finally {
      fs.closeSync(fd);
    }
  }
  await api.markStep();

  // Build the tied lm_head as independent sub-2GB weight chunks when the table
  // exceeds the 2GB binding limit. The f32 table (2.36GB) always needs chunking;
  // the f16 table (#59) is 1.18GB — under the limit — so the tied lm_head reads
  // embedTokens.weight directly and no chunks are built (lmHeadChunks stays
  // null). Each chunk is [rows, hidden] from a fresh CPU subarray → own buffer.
  const vocab = config.vocabSize;
  const embedElemBytes = (config.weightDtype ?? "f32") === "f16" ? 2 : 4;
  if (embedData !== null && vocab * hiddenSize * embedElemBytes > (1 << 31) - 4) {
    const rowsPerChunk = model.lmHeadChunkRows();
    const chunks: Tensor[] = [];
    for (let start = 0; start < vocab; start += rowsPerChunk) {
      const len = Math.min(rowsPerChunk, vocab - start);
      const slice = embedData.slice(
        start * hiddenSize,
        (start + len) * hiddenSize,
      );
      chunks.push(
        api.tensorFromArray(slice, [len, hiddenSize], { device: "webgpu" }),
      );
      await api.markStep();
    }
    model.lmHeadChunks = chunks;
    console.log(`lm_head split into ${chunks.length} vocab chunks (<2GB each)`);
  } else if (embedF16Bits !== null) {
    console.log(
      "lm_head tied to f16 embedding directly (1.18GB < 2GB binding limit)",
    );
  }
  embedData = null;
  embedF16Bits = null;

  const { evictAllPoolBuffers } = await import("../../src/backend/webgpu");
  evictAllPoolBuffers();

  if (skipped.length > 0) {
    console.warn(`Unmapped weights (${skipped.length}):`, skipped.slice(0, 10));
  }
  console.log(`Loaded ${loaded} weight tensors from ${shardFiles.length} shard(s)`);
  return model;
}
