/**
 * GEMV NT read-wider-cast-on-load differential (#95).
 *
 * f16 M=1 decode projections under AMP read a WIDER (f32) operand and cast to
 * f16 during the tile load (inputCastA / inputCastB). Before #95 the GEMV NT
 * route bailed on ANY input cast (variants.ts `!ctx.hasInputCast`), falling
 * back to the tiled `gemv_row` config that wastes (tileM-1)/tileM of every
 * workgroup on absent rows — the ~1.8 GB/s "kernel-bound" f16 decode from #93.
 *
 * This gate proves the new cast-carrying GEMV NT kernel is NUMERICALLY
 * IDENTICAL to the tiled reference (which already handled the cast) on the same
 * f32/f16 buffers, across M=1 decode-projection shapes (Qwen3 / Gemma dims),
 * every cast axis (A, B, both), and the bias epilogue — the Corollary's
 * "differentially test the optimized path against the naive one".
 *
 * Routing: no explicit config → M=1 + transB now selects the GEMV variant
 * WITH the cast (the fix). An explicit config pins the tiled variant
 * (`hasExplicitConfig`) → the reference. Both consume the same buffers.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  getWebGPUDevice,
  initWebGPU,
  isF16Supported,
  syncWebGPU,
} from "../../src/backend/webgpu";
import type {
  GPUBuffer,
  GPUDevice,
  GPUQueue,
} from "../../src/backend/webgpu/gpu-types";
import { GPUBufferUsage, GPUMapMode } from "../../src/backend/webgpu/gpu-types";
import { dispatchTiledMatmul } from "../../src/backend/webgpu/matmul/dispatch";
import { computeGemvRoute } from "../../src/backend/webgpu/matmul/gemv";
import {
  DEFAULT_CONFIG,
  type DType,
  type EpilogueConfig,
} from "../../src/backend/webgpu/matmul/types";

import { cpuOnly } from "../helpers/webgpu";

const isWebGPUEnabled = !cpuOnly;

let device: GPUDevice;
let queue: GPUQueue;

// f32 → f16 (round-to-nearest via the same bit-twiddle the tile-ir spec uses).
function f32ToF16Bits(x: number): number {
  const tmpF32 = new Float32Array([x]);
  const bits32 = new DataView(tmpF32.buffer).getUint32(0, true);
  const sign = (bits32 >> 31) & 1;
  const exp = (bits32 >> 23) & 0xff;
  const man = bits32 & 0x7fffff;
  if (exp === 0) return sign << 15;
  if (exp === 0xff) return (sign << 15) | 0x7c00 | (man ? 1 : 0);
  const newExp = exp - 127 + 15;
  if (newExp >= 31) return (sign << 15) | 0x7c00;
  if (newExp <= 0) return sign << 15;
  return (sign << 15) | (newExp << 10) | (man >> 13);
}

function f16BitsToF32(h: number): number {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const man = h & 0x3ff;
  if (exp === 0) return (sign ? -1 : 1) * 2 ** -14 * (man / 1024);
  if (exp === 31) return man ? NaN : sign ? -Infinity : Infinity;
  return (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + man / 1024);
}

/** Round a value through f16 (so f32 buffers hold exactly f16-representable
 *  values — the cast on load is then lossless and A/B/tiled all see the same
 *  operand bits). */
function roundF16(x: number): number {
  return f16BitsToF32(f32ToF16Bits(x));
}

function makeF32Buffer(data: Float32Array): GPUBuffer {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(buf.getMappedRange()).set(data);
  buf.unmap();
  return buf;
}

function makeF16Buffer(data: Float32Array): GPUBuffer {
  const u16 = new Uint16Array(data.length);
  for (let i = 0; i < data.length; i++) u16[i] = f32ToF16Bits(data[i]);
  const buf = device.createBuffer({
    size: u16.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint16Array(buf.getMappedRange()).set(u16);
  buf.unmap();
  return buf;
}

function makeOutputBuffer(numElements: number, bytesPerElement: number) {
  return device.createBuffer({
    size: numElements * bytesPerElement,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

async function readBuffer(
  buf: GPUBuffer,
  count: number,
  dtype: DType,
): Promise<Float32Array> {
  const bpe = dtype === "f16" ? 2 : 4;
  const staging = device.createBuffer({
    size: count * bpe,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, count * bpe);
  queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  let out: Float32Array;
  if (dtype === "f16") {
    const u16 = new Uint16Array(staging.getMappedRange().slice(0));
    out = new Float32Array(count);
    for (let i = 0; i < count; i++) out[i] = f16BitsToF32(u16[i]);
  } else {
    out = new Float32Array(staging.getMappedRange().slice(0));
  }
  staging.unmap();
  staging.destroy();
  return out;
}

/** Deterministic small values, f16-exact. */
function makeVals(size: number, offset = 0): Float32Array {
  const v = new Float32Array(size);
  for (let i = 0; i < size; i++) v[i] = roundF16((((i + offset) % 11) - 5) / 2);
  return v;
}

/** NT: out[1,N] = A[1,K] · B[N,K]^T. */
function ntReference(a: Float32Array, b: Float32Array, n: number, k: number) {
  const out = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let t = 0; t < k; t++) s += a[t] * b[j * k + t];
    out[j] = s;
  }
  return out;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// Real M=1 decode-projection shapes (N=out features, K=in features), NT layout
// (weight [N,K], simple-transpose detected). Qwen3-0.6B (hidden 1024, kv heads
// share head_dim 128, MLP 3072) and Gemma-2-2B (hidden 2304, MLP 9216) —
// K % 4 === 0 so the vec4 NT path is exercised too.
const DECODE_SHAPES: Array<{ label: string; n: number; k: number }> = [
  { label: "qwen3 q_proj (2048x1024)", n: 2048, k: 1024 },
  { label: "qwen3 o_proj (1024x2048)", n: 1024, k: 2048 },
  { label: "qwen3 gate/up (3072x1024)", n: 3072, k: 1024 },
  { label: "qwen3 down (1024x3072)", n: 1024, k: 3072 },
  { label: "gemma2 q_proj (2048x2304)", n: 2048, k: 2304 },
  { label: "small odd-K NT (130x1024)", n: 130, k: 1024 },
  // N ≥ 32768 → defaultGemvNtRowsPerWg = 4 (multi-row lane-group NT path),
  // the lm_head-scale route — cast must flow through the 2D-grid multirow
  // kernel too. K small to keep the tiled reference tractable.
  { label: "lm_head-scale multirow (40000x512)", n: 40000, k: 512 },
];

// Cast axes exercised: A stored f32 (activation), B stored f32 (weight), both.
const CAST_AXES: Array<{
  label: string;
  castA?: DType;
  castB?: DType;
}> = [
  { label: "castA (f32 act × f16 weight)", castA: "f32" },
  { label: "castB (f16 act × f32 weight)", castB: "f32" },
  { label: "castA+castB (both f32)", castA: "f32", castB: "f32" },
];

describe.runIf(isWebGPUEnabled)(
  "GEMV NT input-cast differential (webgpu)",
  () => {
    beforeAll(async () => {
      const ok = await initWebGPU();
      if (!ok) throw new Error("WebGPU init failed");
      const ctx = getWebGPUDevice();
      if (!ctx) throw new Error("No WebGPU device");
      device = ctx.device;
      queue = ctx.queue;
    });

    afterAll(async () => {
      await syncWebGPU();
    });

    for (const shape of DECODE_SHAPES) {
      // The lm_head-scale shape's tiled reference is ~50s/case; one cast axis
      // (both, the strongest) covers its multirow codegen path. Medium shapes
      // run all three axes.
      const axesForShape = shape.n >= 32768 ? [CAST_AXES[2]] : CAST_AXES;
      for (const axis of axesForShape) {
        it(`${shape.label} — ${axis.label}: GEMV cast == tiled == CPU`, async () => {
          if (!isF16Supported()) return;
          const { n, k } = shape;
          // The cast route must actually be an NT GEMV route (else this test
          // would silently exercise nothing).
          expect(computeGemvRoute(n, k, /* transB */ true)).not.toBeNull();

          const aVals = makeVals(k, 0);
          const bVals = makeVals(n * k, 3);
          const expected = ntReference(aVals, bVals, n, k);

          // A/B stored in the cast (wider) dtype when castX set, else f16.
          const mkA = () =>
            axis.castA === "f32" ? makeF32Buffer(aVals) : makeF16Buffer(aVals);
          const mkB = () =>
            axis.castB === "f32" ? makeF32Buffer(bVals) : makeF16Buffer(bVals);

          // Logical matmul dtype is f16; output f16 (AMP decode).
          const common = {
            device,
            queue,
            m: 1,
            n,
            k,
            transB: true,
            dtype: "f16" as DType,
            dtypeB: "f16" as DType,
            inputCastA: axis.castA,
            inputCastB: axis.castB,
          };

          // GEMV path: no config → M=1 + transB selects the (now cast-carrying)
          // GEMV NT variant.
          const aG = mkA();
          const bG = mkB();
          const outG = makeOutputBuffer(n, 2);
          dispatchTiledMatmul({ ...common, a: aG, b: bG, out: outG });
          const gemv = await readBuffer(outG, n, "f16");

          // Tiled reference: pin an explicit config → tiled variant.
          const aT = mkA();
          const bT = mkB();
          const outT = makeOutputBuffer(n, 2);
          dispatchTiledMatmul({
            ...common,
            a: aT,
            b: bT,
            out: outT,
            config: { ...DEFAULT_CONFIG },
          });
          const tiled = await readBuffer(outT, n, "f16");

          // GEMV == tiled: both f16-accumulate-in-f32 the same lossless operands
          // → bit-identical (exact). And both track the CPU reference within
          // f16 output rounding.
          expect(maxAbsDiff(gemv, tiled)).toBe(0);
          // f16 output rounding tolerance scales with magnitude.
          let worst = 0;
          for (let i = 0; i < n; i++) {
            const bound = 1e-3 + 5e-3 * Math.abs(expected[i]);
            worst = Math.max(worst, Math.abs(gemv[i] - expected[i]) - bound);
          }
          expect(worst).toBeLessThanOrEqual(0);

          for (const buf of [aG, bG, outG, aT, bT, outT]) buf.destroy();
        }, 120_000); // wide N tiled-reference dispatch on the biggest shapes is slow
      }
    }

    it("bias epilogue: GEMV cast NT == tiled == CPU (2048x1024, castA)", async () => {
      if (!isF16Supported()) return;
      const n = 2048;
      const k = 1024;
      const aVals = makeVals(k, 1);
      const bVals = makeVals(n * k, 7);
      const biasVals = makeVals(n, 2);
      const expected = ntReference(aVals, bVals, n, k);
      for (let j = 0; j < n; j++) expected[j] += biasVals[j];

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "bias", inputIndex: 0 }],
        additionalInputCount: 1,
        outputDtype: "f16",
      };

      const common = {
        device,
        queue,
        m: 1,
        n,
        k,
        transB: true,
        dtype: "f16" as DType,
        dtypeB: "f16" as DType,
        inputCastA: "f32" as DType,
        epilogue,
      };

      const aG = makeF32Buffer(aVals);
      const bG = makeF16Buffer(bVals);
      const biasG = makeF32Buffer(biasVals); // epilogue inputs bind as f32
      const outG = makeOutputBuffer(n, 2);
      dispatchTiledMatmul({
        ...common,
        a: aG,
        b: bG,
        out: outG,
        epilogueInputs: [biasG],
      });
      const gemv = await readBuffer(outG, n, "f16");

      const aT = makeF32Buffer(aVals);
      const bT = makeF16Buffer(bVals);
      const biasT = makeF32Buffer(biasVals);
      const outT = makeOutputBuffer(n, 2);
      dispatchTiledMatmul({
        ...common,
        a: aT,
        b: bT,
        out: outT,
        epilogueInputs: [biasT],
        config: { ...DEFAULT_CONFIG },
      });
      const tiled = await readBuffer(outT, n, "f16");

      expect(maxAbsDiff(gemv, tiled)).toBe(0);
      let worst = 0;
      for (let i = 0; i < n; i++) {
        const bound = 1e-3 + 5e-3 * Math.abs(expected[i]);
        worst = Math.max(worst, Math.abs(gemv[i] - expected[i]) - bound);
      }
      expect(worst).toBeLessThanOrEqual(0);

      for (const buf of [aG, bG, biasG, outG, aT, bT, biasT, outT])
        buf.destroy();
    });
  },
);
