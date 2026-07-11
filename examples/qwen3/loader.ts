/**
 * Qwen3 Weight Loading
 *
 * Streams HF safetensors (sharded, bf16) per-tensor from disk directly into
 * an instantiated Qwen3 model — peak JS memory is one weight at a time.
 * HF Linear weights are [out, in], same layout as torchlette nn.Linear:
 * no transposes needed anywhere. lm_head is tied (skipped if present).
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { quantizeLinearWeight } from "../../src/backend/quantize";
import {
  resolveWeightFormat,
  type WeightFormatName,
} from "../../src/backend/types";
import type { Tensor, Torchlette } from "../../src/frontend/torchlette";
import type { Linear } from "../../src/nn/linear";
import { resolveDest } from "../../packages/qwen3-browser/src/weights-map";
import { configFromHF, Qwen3, type Qwen3Block, type Qwen3Config } from "./model";

type SafetensorsMetadata = {
  [key: string]: {
    dtype: string;
    shape: number[];
    data_offsets: [number, number];
  };
};

// ============================================================================
// Safetensors streaming (per-tensor reads)
// ============================================================================

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

// ============================================================================
// HF name → model parameter resolution
// ============================================================================

/** Resolve an HF weight name to the destination tensor in the model, or null to skip. */
function resolveDest(model: Qwen3, hfName: string): Tensor | null {
  if (hfName === "lm_head.weight") return null; // tied to embed_tokens
  if (hfName === "model.embed_tokens.weight") return model.embedTokens.weight;
  if (hfName === "model.norm.weight") return model.norm.weight;

  const m = hfName.match(/^model\.layers\.(\d+)\.(.+)$/);
  if (!m) return null;
  const block = model.layers.get(Number(m[1])) as Qwen3Block;
  switch (m[2]) {
    case "self_attn.q_proj.weight":
      return block.attn.qProj.weight;
    case "self_attn.k_proj.weight":
      return block.attn.kProj.weight;
    case "self_attn.v_proj.weight":
      return block.attn.vProj.weight;
    case "self_attn.o_proj.weight":
      return block.attn.oProj.weight;
    case "self_attn.q_norm.weight":
      return block.attn.qNorm.weight;
    case "self_attn.k_norm.weight":
      return block.attn.kNorm.weight;
    case "input_layernorm.weight":
      return block.inputNorm.weight;
    case "post_attention_layernorm.weight":
      return block.postAttnNorm.weight;
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

/**
 * The Linear projection module owning a given HF weight name, or null if the
 * weight is not a quantizable projection (norms/embedding stay unquantized —
 * see docs/quantization-design.md scope). Used by the weightFormat load path to
 * REPLACE a projection's f16 weight with a packed-int operand.
 */
function resolveProjModule(model: Qwen3, hfName: string): Linear | null {
  const m = hfName.match(/^model\.layers\.(\d+)\.(.+)$/);
  if (!m) return null;
  const block = model.layers.get(Number(m[1])) as Qwen3Block;
  switch (m[2]) {
    case "self_attn.q_proj.weight":
      return block.attn.qProj;
    case "self_attn.k_proj.weight":
      return block.attn.kProj;
    case "self_attn.v_proj.weight":
      return block.attn.vProj;
    case "self_attn.o_proj.weight":
      return block.attn.oProj;
    case "mlp.gate_proj.weight":
      return block.mlp.gateProj;
    case "mlp.up_proj.weight":
      return block.mlp.upProj;
    case "mlp.down_proj.weight":
      return block.mlp.downProj;
    default:
      return null;
  }
}

// ============================================================================
// Model loading
// ============================================================================

/**
 * Load a pretrained Qwen3 model from a HF snapshot directory
 * (config.json + model*.safetensors [+ index for sharded repos]).
 *
 * `weightFormat` (e.g. "int8-64") quantizes the per-layer PROJECTION weights
 * (q/k/v/o/gate/up/down) to a packed-int operand at load — nothing downstream
 * mentions quant (docs/quantization-design.md phase 2). Norms + embedding +
 * tied lm_head stay f16 (the gather needs the table; scope excludes them).
 */
export async function loadPretrainedQwen3(
  api: Torchlette,
  modelDir: string,
  options?: {
    maxSeqLen?: number;
    weightDtype?: "f32" | "f16";
    weightFormat?: WeightFormatName;
  },
): Promise<Qwen3> {
  const hfConfig = JSON.parse(
    await fs.promises.readFile(path.join(modelDir, "config.json"), "utf-8"),
  );
  const config: Qwen3Config = configFromHF(
    hfConfig,
    options?.maxSeqLen ?? 4096,
    options?.weightDtype ?? "f32",
  );
  console.log("Creating Qwen3 model:", config);
  const model = new Qwen3(api, config, { device: "webgpu" });

  // Enumerate shards: sharded (index.json) or single-file.
  const indexPath = path.join(modelDir, "model.safetensors.index.json");
  let shardFiles: string[];
  if (fs.existsSync(indexPath)) {
    const index = JSON.parse(await fs.promises.readFile(indexPath, "utf-8"));
    shardFiles = [...new Set(Object.values(index.weight_map as Record<string, string>))];
  } else {
    shardFiles = ["model.safetensors"];
  }

  let loaded = 0;
  let skipped: string[] = [];
  let pendingBytes = 0;
  const FLUSH_THRESHOLD = 512 * 1024 * 1024;

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
        const data = extractWeight(fd, dataStart, info);
        if (JSON.stringify(dest.shape) !== JSON.stringify(info.shape)) {
          throw new Error(
            `Shape mismatch for ${name}: model ${JSON.stringify(dest.shape)} vs file ${JSON.stringify(info.shape)}`,
          );
        }
        // weightFormat: quantize projection weights at load and REPLACE the
        // module's f16 parameter with a packed-int operand. The forward's
        // api.linear is unchanged (format-blind) — the backend matmul routes.
        const projModule = options?.weightFormat
          ? resolveProjModule(model, name)
          : null;
        if (projModule && options?.weightFormat) {
          const [N, K] = info.shape;
          if (K % 4 === 0) {
            const format = resolveWeightFormat(options.weightFormat, "f16");
            const groupSize = format.packing!.groupSize;
            if (K % groupSize === 0) {
              const q = quantizeLinearWeight(data, N, K, groupSize);
              const qWeight = await api.createQuantizedWeight(
                q.packed,
                q.scales,
                N,
                K,
                format,
              );
              projModule.registerParameter("weight", qWeight as unknown as Tensor);
              loaded++;
              await api.markStep();
              continue;
            }
          }
        }
        // Match the destination's dtype (linears/embeddings may be f16 while
        // norm weights stay f32).
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
  // Release the pool's load-time buffer cache (upload staging + init-weight
  // buffers). Loading transiently peaks at ~4x the model size on a 32GB
  // V100 (init weights + uploads + pool retention) — evicting here returns
  // the slack to the device so inference starts with full headroom. The
  // first step re-acquires what it needs (one-time cost).
  const { evictAllPoolBuffers } = await import("../../src/backend/webgpu");
  evictAllPoolBuffers();

  if (skipped.length > 0) {
    console.warn(`Unmapped weights (${skipped.length}):`, skipped.slice(0, 10));
  }
  console.log(`Loaded ${loaded} weight tensors from ${shardFiles.length} shard(s)`);
  return model;
}
