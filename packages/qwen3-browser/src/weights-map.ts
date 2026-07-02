/**
 * HF weight-name → model-parameter resolution, shared by the Node loader
 * (examples/qwen3/loader.ts) and the browser streaming loader.
 */

import type { Tensor } from "torchlette";
import type { Qwen3, Qwen3Block } from "./model";

/** Resolve an HF weight name to the destination tensor in the model, or null to skip. */
export function resolveDest(model: Qwen3, hfName: string): Tensor | null {
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

/** Convert a raw safetensors tensor payload to Float32Array (F32/F16/BF16). */
export function payloadToFloat32(
  dtype: string,
  bytes: Uint8Array,
): Float32Array {
  if (dtype === "F32") {
    // Copy to guarantee alignment.
    const out = new Float32Array(bytes.byteLength / 4);
    new Uint8Array(out.buffer).set(bytes);
    return out;
  }
  if (dtype === "F16" || dtype === "BF16") {
    const n = bytes.byteLength / 2;
    const aligned = new Uint16Array(n);
    new Uint8Array(aligned.buffer).set(bytes);
    const out = new Float32Array(n);
    if (dtype === "BF16") {
      const buf = new ArrayBuffer(4);
      const u32 = new Uint32Array(buf);
      const f32 = new Float32Array(buf);
      for (let i = 0; i < n; i++) {
        u32[0] = aligned[i] << 16;
        out[i] = f32[0];
      }
    } else {
      for (let i = 0; i < n; i++) out[i] = float16ToFloat32(aligned[i]);
    }
    return out;
  }
  throw new Error(`Unsupported safetensors dtype ${dtype}`);
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

// ============================================================================
// Fast slice converters (browser loader hot path — integer ops only, no float
// math, designed to be called in slices with event-loop yields between them so
// the main thread never blocks for more than a few ms).
// ============================================================================

/**
 * bf16 → f16 raw bits for [start, end). EXACT for the f16 normal range (f16
 * has more mantissa bits than bf16); overflow clamps to ±f16-max, underflow
 * goes through f16 denormals then flushes to ±0.
 */
export function bf16SliceToF16Bits(
  src: Uint16Array,
  dst: Uint16Array,
  start: number,
  end: number,
): void {
  for (let i = start; i < end; i++) {
    const b = src[i];
    const sign = b & 0x8000;
    const exp = (b >> 7) & 0xff; // bf16 exponent, bias 127
    const mant = b & 0x7f; // 7 mantissa bits
    let out: number;
    if (exp === 0) {
      out = sign; // bf16 zero/denormal (≤1e-38) is far below f16 denormals → ±0
    } else if (exp === 0xff) {
      out = sign | 0x7c00 | (mant ? 0x200 : 0); // ±inf / NaN
    } else {
      const e16 = exp - 127 + 15; // rebias to f16 (bias 15)
      if (e16 >= 0x1f) {
        out = sign | 0x7bff; // overflow → clamp to ±65504 (not inf: weights)
      } else if (e16 <= 0) {
        out = sign | ((0x400 | (mant << 3)) >> (1 - e16)); // f16 denormal
      } else {
        out = sign | (e16 << 10) | (mant << 3);
      }
    }
    dst[i] = out;
  }
}

/** bf16 → f32 for [start, end): pure u32 shift into the f32 buffer's bits. */
export function bf16SliceToF32(
  src: Uint16Array,
  dst: Float32Array,
  start: number,
  end: number,
): void {
  const u32 = new Uint32Array(dst.buffer, dst.byteOffset, dst.length);
  for (let i = start; i < end; i++) u32[i] = src[i] << 16;
}

/** f16 → f32 for [start, end) (norm weights are tiny; per-element is fine). */
export function f16SliceToF32(
  src: Uint16Array,
  dst: Float32Array,
  start: number,
  end: number,
): void {
  for (let i = start; i < end; i++) dst[i] = float16ToFloat32(src[i]);
}
