/**
 * Tile IR matmul tests.
 *
 * Exercises the tile-IR codegen path for coverage gaps not hit by the standard
 * matmul tests:
 *   - f16 input/output (the exact bug that caused all-zero loss in training)
 *   - Transpose modes TN, NT, TT (standard tests only run NN)
 *   - K-split matmul (small M, large K)
 *   - Binary epilogue (cast + bias + binary add — the residual connection pattern)
 *   - Mixed-dtype matmul (f16 inputs with f32 output)
 *
 * Tests use dispatchTiledMatmul directly with explicit configs so the
 * tile-IR path is exercised regardless of shape-class defaults.
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
import type { EpilogueConfig } from "../../src/backend/webgpu/matmul/types";
import {
  DEFAULT_CONFIG,
  type MatmulKernelConfig,
} from "../../src/backend/webgpu/matmul/types";

import { cpuOnly } from "../helpers/webgpu";

const isWebGPUEnabled = !cpuOnly;

// ---------------------------------------------------------------------------
// Reference matmul (CPU)
// ---------------------------------------------------------------------------

function matmulReference(
  a: number[],
  b: number[],
  m: number,
  n: number,
  k: number,
  transA = false,
  transB = false,
): number[] {
  const out = new Array<number>(m * n).fill(0);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let t = 0; t < k; t++) {
        const aIdx = transA ? t * m + i : i * k + t;
        const bIdx = transB ? j * k + t : t * n + j;
        sum += a[aIdx] * b[bIdx];
      }
      out[i * n + j] = sum;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------

let device: GPUDevice;
let queue: GPUQueue;

function makeF32Buffer(data: Float32Array, usage: number): GPUBuffer {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(buf.getMappedRange()).set(data);
  buf.unmap();
  return buf;
}

function makeF16Buffer(data: Float32Array, usage: number): GPUBuffer {
  // Convert f32 array to f16 (Uint16) using DataView
  const u16 = new Uint16Array(data.length);
  const tmpF32 = new Float32Array(1);
  const tmpU8 = new Uint8Array(tmpF32.buffer);
  for (let i = 0; i < data.length; i++) {
    tmpF32[0] = data[i];
    // Simple f32→f16 conversion: extract sign, exponent, mantissa
    const bits32 = new DataView(tmpU8.buffer).getUint32(0, true);
    const sign = (bits32 >> 31) & 1;
    const exp = (bits32 >> 23) & 0xff;
    const man = bits32 & 0x7fffff;
    let h: number;
    if (exp === 0) {
      h = sign << 15; // zero / denorm → zero
    } else if (exp === 0xff) {
      h = (sign << 15) | 0x7c00 | (man ? 1 : 0); // inf / nan
    } else {
      const newExp = exp - 127 + 15;
      if (newExp >= 31) {
        h = (sign << 15) | 0x7c00; // overflow → inf
      } else if (newExp <= 0) {
        h = sign << 15; // underflow → zero
      } else {
        h = (sign << 15) | (newExp << 10) | (man >> 13);
      }
    }
    u16[i] = h;
  }
  const buf = device.createBuffer({
    size: u16.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint16Array(buf.getMappedRange()).set(u16);
  buf.unmap();
  return buf;
}

function makeOutputBuffer(
  numElements: number,
  bytesPerElement: number,
): GPUBuffer {
  return device.createBuffer({
    size: numElements * bytesPerElement,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

async function readF32Buffer(
  buf: GPUBuffer,
  count: number,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, count * 4);
  queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

async function readF16Buffer(
  buf: GPUBuffer,
  count: number,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: count * 2,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, count * 2);
  queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const u16 = new Uint16Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();

  // Convert f16 (Uint16) back to f32
  const f32 = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const h = u16[i];
    const sign = (h >> 15) & 1;
    const exp = (h >> 10) & 0x1f;
    const man = h & 0x3ff;
    let val: number;
    if (exp === 0) {
      val = (sign ? -1 : 1) * 2 ** -14 * (man / 1024);
    } else if (exp === 31) {
      val = man ? NaN : sign ? -Infinity : Infinity;
    } else {
      val = (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + man / 1024);
    }
    f32[i] = val;
  }
  return f32;
}

function arraysClose(
  actual: number[] | Float32Array,
  expected: number[] | Float32Array,
  rtol = 1e-3,
  atol = 1e-3,
): boolean {
  if (actual.length !== expected.length) {
    console.log(
      `Length mismatch: actual=${actual.length}, expected=${expected.length}`,
    );
    return false;
  }
  let worstDiff = 0;
  let worstIdx = -1;
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    const bound = atol + rtol * Math.abs(expected[i]);
    if (diff > bound) {
      if (diff > worstDiff) {
        worstDiff = diff;
        worstIdx = i;
      }
    }
  }
  if (worstIdx >= 0) {
    console.log(
      `Worst mismatch at ${worstIdx}: actual=${actual[worstIdx]}, expected=${expected[worstIdx]}, ` +
        `diff=${worstDiff}, bound=${atol + rtol * Math.abs(expected[worstIdx])}`,
    );
    return false;
  }
  return true;
}

function makeValues(size: number, offset = 0): number[] {
  const values = new Array<number>(size);
  for (let i = 0; i < size; i++) {
    values[i] = ((i + offset) % 7) - 3; // small integers: -3..3
  }
  return values;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe.runIf(isWebGPUEnabled)("tile IR matmul (webgpu)", () => {
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

  // -------------------------------------------------------------------------
  // f16 matmul — the exact bug that caused all-zero training loss
  // -------------------------------------------------------------------------
  describe("f16 input/output", () => {
    it("computes f16 @ f16 → f16 correctly (64x64)", async () => {
      if (!isF16Supported()) return;

      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);
      const expected = matmulReference(aVals, bVals, m, n, k);

      const aBuf = makeF16Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF16Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 2); // f16 = 2 bytes

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f16",
        config: {
          ...DEFAULT_CONFIG,
          tileM: 32,
          tileN: 32,
          tileK: 16,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF16Buffer(outBuf, m * n);
      // f16 has limited precision — use wider tolerances
      expect(arraysClose(result, new Float32Array(expected), 5e-2, 1.0)).toBe(
        true,
      );

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });

    it("computes f16 @ f16 → f32 (mixed output) correctly (64x64)", async () => {
      if (!isF16Supported()) return;

      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);
      const expected = matmulReference(aVals, bVals, m, n, k);

      const aBuf = makeF16Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF16Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4); // f32 = 4 bytes

      // f16 inputs but f32 output: matches the AMP matmul pattern
      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f16",
        epilogue: {
          ops: [{ kind: "cast", toDtype: "f32" }],
          additionalInputCount: 0,
          outputDtype: "f32",
        },
        config: {
          ...DEFAULT_CONFIG,
          tileM: 32,
          tileN: 32,
          tileK: 16,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF32Buffer(outBuf, m * n);
      // f16 input quantization limits precision even with f32 output
      expect(arraysClose(result, new Float32Array(expected), 5e-2, 1.0)).toBe(
        true,
      );

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });

    it("computes non-tile-aligned f16 matmul (50x70x60)", async () => {
      if (!isF16Supported()) return;

      const m = 50,
        n = 70,
        k = 60;
      const aVals = makeValues(m * k, 5);
      const bVals = makeValues(k * n, 3);
      const expected = matmulReference(aVals, bVals, m, n, k);

      const aBuf = makeF16Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF16Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 2);

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f16",
        config: {
          ...DEFAULT_CONFIG,
          tileM: 32,
          tileN: 32,
          tileK: 16,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF16Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected), 5e-2, 1.0)).toBe(
        true,
      );

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });
  });

  // -------------------------------------------------------------------------
  // Transpose modes — existing tests only exercise NN
  // -------------------------------------------------------------------------
  describe("transpose modes", () => {
    const config: MatmulKernelConfig = {
      ...DEFAULT_CONFIG,
      tileM: 32,
      tileN: 32,
      tileK: 16,
      threadTileM: 4,
      threadTileN: 4,
    };

    it("TN: transposed A (64x64)", async () => {
      const m = 64,
        n = 64,
        k = 64;
      // A is stored as [K, M] (transposed)
      const aVals = makeValues(k * m, 0);
      const bVals = makeValues(k * n, 1);
      const expected = matmulReference(aVals, bVals, m, n, k, true, false);

      const aBuf = makeF32Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF32Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        transA: true,
        dtype: "f32",
        config,
      });

      const result = await readF32Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected))).toBe(true);

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });

    it("NT: transposed B (64x64)", async () => {
      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(m * k, 0);
      // B is stored as [N, K] (transposed)
      const bVals = makeValues(n * k, 1);
      const expected = matmulReference(aVals, bVals, m, n, k, false, true);

      const aBuf = makeF32Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF32Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        transB: true,
        dtype: "f32",
        config,
      });

      const result = await readF32Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected))).toBe(true);

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });

    it("TT: both transposed (64x64)", async () => {
      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(k * m, 0);
      const bVals = makeValues(n * k, 1);
      const expected = matmulReference(aVals, bVals, m, n, k, true, true);

      const aBuf = makeF32Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF32Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        transA: true,
        transB: true,
        dtype: "f32",
        config,
      });

      const result = await readF32Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected))).toBe(true);

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });

    it("TN with f16 inputs (64x64)", async () => {
      if (!isF16Supported()) return;

      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(k * m, 0);
      const bVals = makeValues(k * n, 1);
      const expected = matmulReference(aVals, bVals, m, n, k, true, false);

      const aBuf = makeF16Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF16Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 2);

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        transA: true,
        dtype: "f16",
        config,
      });

      const result = await readF16Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected), 5e-2, 1.0)).toBe(
        true,
      );

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });

    it("non-tile-aligned TN (50x70x60)", async () => {
      const m = 50,
        n = 70,
        k = 60;
      const aVals = makeValues(k * m, 2);
      const bVals = makeValues(k * n, 3);
      const expected = matmulReference(aVals, bVals, m, n, k, true, false);

      const aBuf = makeF32Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF32Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        transA: true,
        dtype: "f32",
        config,
      });

      const result = await readF32Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected))).toBe(true);

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });
  });

  // -------------------------------------------------------------------------
  // K-split matmul — small M with large K triggers K-split dispatch
  // -------------------------------------------------------------------------
  describe("K-split", () => {
    it("K-split f32: small M, large K (8x64x512)", async () => {
      const m = 8,
        n = 64,
        k = 512;
      const aVals = makeValues(m * k, 0).map((v) => v * 0.1);
      const bVals = makeValues(k * n, 1).map((v) => v * 0.1);
      const expected = matmulReference(aVals, bVals, m, n, k);

      const aBuf = makeF32Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF32Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      // Use a config with small tiles to trigger K-split
      // (baseWorkgroups = ceil(64/32)*ceil(8/32) = 2*1 = 2, which is < 64 and K > 512)
      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f32",
        config: {
          ...DEFAULT_CONFIG,
          tileM: 32,
          tileN: 32,
          tileK: 16,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF32Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected), 1e-3, 1e-2)).toBe(
        true,
      );

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });

    it("K-split f16: small M, large K (8x64x512)", async () => {
      if (!isF16Supported()) return;

      const m = 8,
        n = 64,
        k = 512;
      const aVals = makeValues(m * k, 0).map((v) => v * 0.1);
      const bVals = makeValues(k * n, 1).map((v) => v * 0.1);
      const expected = matmulReference(aVals, bVals, m, n, k);

      const aBuf = makeF16Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF16Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 2);

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f16",
        config: {
          ...DEFAULT_CONFIG,
          tileM: 32,
          tileN: 32,
          tileK: 16,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF16Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected), 0.1, 1.0)).toBe(
        true,
      );

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });
  });

  // -------------------------------------------------------------------------
  // Binary epilogue — the residual connection pattern from training
  // (matmul → cast(f16→f32) → bias → binary_add(residual))
  // -------------------------------------------------------------------------
  describe("binary epilogue", () => {
    it("cast + bias + binary add (f16 input, f32 output)", async () => {
      if (!isF16Supported()) return;

      const m = 32,
        n = 32,
        k = 32;
      const aVals = makeValues(m * k, 0).map((v) => v * 0.5);
      const bVals = makeValues(k * n, 1).map((v) => v * 0.5);
      const biasVals = makeValues(n, 2).map((v) => v * 0.1);
      const residualVals = makeValues(m * n, 3).map((v) => v * 0.2);
      const mmRef = matmulReference(aVals, bVals, m, n, k);
      const expected = mmRef.map(
        (v, i) => v + biasVals[i % n] + residualVals[i],
      );

      const aBuf = makeF16Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF16Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const biasBuf = makeF32Buffer(
        new Float32Array(biasVals),
        GPUBufferUsage.STORAGE,
      );
      const residualBuf = makeF32Buffer(
        new Float32Array(residualVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      const epilogue: EpilogueConfig = {
        ops: [
          { kind: "cast", toDtype: "f32" },
          { kind: "bias", inputIndex: 0 },
          { kind: "binary", op: "add", inputIndex: 1 },
        ],
        additionalInputCount: 2,
        outputDtype: "f32",
      };

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f16",
        epilogue,
        epilogueInputs: [biasBuf, residualBuf],
        config: {
          ...DEFAULT_CONFIG,
          tileM: 32,
          tileN: 32,
          tileK: 16,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF32Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected), 5e-2, 1.0)).toBe(
        true,
      );

      aBuf.destroy();
      bBuf.destroy();
      biasBuf.destroy();
      residualBuf.destroy();
      outBuf.destroy();
    });

    it("bias + binary add (f32 throughout)", async () => {
      const m = 64,
        n = 64,
        k = 32;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);
      const biasVals = makeValues(n, 2);
      const residualVals = makeValues(m * n, 3);
      const mmRef = matmulReference(aVals, bVals, m, n, k);
      const expected = mmRef.map(
        (v, i) => v + biasVals[i % n] + residualVals[i],
      );

      const aBuf = makeF32Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF32Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const biasBuf = makeF32Buffer(
        new Float32Array(biasVals),
        GPUBufferUsage.STORAGE,
      );
      const residualBuf = makeF32Buffer(
        new Float32Array(residualVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      const epilogue: EpilogueConfig = {
        ops: [
          { kind: "bias", inputIndex: 0 },
          { kind: "binary", op: "add", inputIndex: 1 },
        ],
        additionalInputCount: 2,
        outputDtype: "f32",
      };

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f32",
        epilogue,
        epilogueInputs: [biasBuf, residualBuf],
        config: {
          ...DEFAULT_CONFIG,
          tileM: 32,
          tileN: 32,
          tileK: 16,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF32Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected))).toBe(true);

      aBuf.destroy();
      bBuf.destroy();
      biasBuf.destroy();
      residualBuf.destroy();
      outBuf.destroy();
    });
  });

  // -------------------------------------------------------------------------
  // Mixed dtype — f16 @ f32 (matmul with different input types)
  // -------------------------------------------------------------------------
  describe("mixed dtype", () => {
    it("f16 @ f32 → f32 (64x64)", async () => {
      if (!isF16Supported()) return;

      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);
      const expected = matmulReference(aVals, bVals, m, n, k);

      const aBuf = makeF16Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF32Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f16",
        dtypeB: "f32",
        config: {
          ...DEFAULT_CONFIG,
          tileM: 32,
          tileN: 32,
          tileK: 16,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF32Buffer(outBuf, m * n);
      // A is quantized to f16, so limited precision
      expect(arraysClose(result, new Float32Array(expected), 5e-2, 1.0)).toBe(
        true,
      );

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });

    it("f32 @ f16 → f32 (64x64)", async () => {
      if (!isF16Supported()) return;

      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);
      const expected = matmulReference(aVals, bVals, m, n, k);

      const aBuf = makeF32Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF16Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f32",
        dtypeB: "f16",
        config: {
          ...DEFAULT_CONFIG,
          tileM: 32,
          tileN: 32,
          tileK: 16,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF32Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected), 5e-2, 1.0)).toBe(
        true,
      );

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });
  });

  // -------------------------------------------------------------------------
  // Larger tile configs — exercise different shapes/tile combos
  // -------------------------------------------------------------------------
  describe("larger tile configs", () => {
    it("64x64 tile (128x128 matmul)", async () => {
      const m = 128,
        n = 128,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);
      const expected = matmulReference(aVals, bVals, m, n, k);

      const aBuf = makeF32Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF32Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f32",
        config: {
          ...DEFAULT_CONFIG,
          tileM: 64,
          tileN: 64,
          tileK: 8,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF32Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected))).toBe(true);

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });

    it("64x64 tile with 8x4 thread tiles (128x128 matmul)", async () => {
      const m = 128,
        n = 128,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);
      const expected = matmulReference(aVals, bVals, m, n, k);

      const aBuf = makeF32Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF32Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f32",
        config: {
          ...DEFAULT_CONFIG,
          tileM: 64,
          tileN: 64,
          tileK: 8,
          threadTileM: 8,
          threadTileN: 4,
        },
      });

      const result = await readF32Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected))).toBe(true);

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });
  });

  // -------------------------------------------------------------------------
  // Epilogue with unary activations + cast (the full AMP forward pattern)
  // -------------------------------------------------------------------------
  describe("cast + bias + activation epilogues", () => {
    it("cast + bias + gelu (f16 → f32)", async () => {
      if (!isF16Supported()) return;

      const m = 32,
        n = 32,
        k = 32;
      const aVals = makeValues(m * k, 0).map((v) => v * 0.1);
      const bVals = makeValues(k * n, 1).map((v) => v * 0.1);
      const biasVals = makeValues(n, 2).map((v) => v * 0.1);
      const mmRef = matmulReference(aVals, bVals, m, n, k);
      // cast(f16→f32) + bias + gelu
      const biased = mmRef.map((v, i) => v + biasVals[i % n]);
      const expected = biased.map((x) => {
        const inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1 + Math.tanh(inner));
      });

      const aBuf = makeF16Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF16Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const biasBuf = makeF32Buffer(
        new Float32Array(biasVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 4);

      const epilogue: EpilogueConfig = {
        ops: [
          { kind: "cast", toDtype: "f32" },
          { kind: "bias", inputIndex: 0 },
          { kind: "unary", op: "gelu" },
        ],
        additionalInputCount: 1,
        outputDtype: "f32",
      };

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f16",
        epilogue,
        epilogueInputs: [biasBuf],
        config: {
          ...DEFAULT_CONFIG,
          tileM: 32,
          tileN: 32,
          tileK: 16,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF32Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected), 5e-2, 1e-1)).toBe(
        true,
      );

      aBuf.destroy();
      bBuf.destroy();
      biasBuf.destroy();
      outBuf.destroy();
    });

    it("cast + bias + gelu + cast (f16 → f32 → gelu → f16)", async () => {
      if (!isF16Supported()) return;

      const m = 32,
        n = 32,
        k = 32;
      const aVals = makeValues(m * k, 0).map((v) => v * 0.1);
      const bVals = makeValues(k * n, 1).map((v) => v * 0.1);
      const biasVals = makeValues(n, 2).map((v) => v * 0.1);
      const mmRef = matmulReference(aVals, bVals, m, n, k);
      const biased = mmRef.map((v, i) => v + biasVals[i % n]);
      const expected = biased.map((x) => {
        const inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1 + Math.tanh(inner));
      });

      const aBuf = makeF16Buffer(
        new Float32Array(aVals),
        GPUBufferUsage.STORAGE,
      );
      const bBuf = makeF16Buffer(
        new Float32Array(bVals),
        GPUBufferUsage.STORAGE,
      );
      const biasBuf = makeF32Buffer(
        new Float32Array(biasVals),
        GPUBufferUsage.STORAGE,
      );
      const outBuf = makeOutputBuffer(m * n, 2); // output is f16

      const epilogue: EpilogueConfig = {
        ops: [
          { kind: "cast", toDtype: "f32" },
          { kind: "bias", inputIndex: 0 },
          { kind: "unary", op: "gelu" },
          { kind: "cast", toDtype: "f16" },
        ],
        additionalInputCount: 1,
        outputDtype: "f16",
      };

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuf,
        b: bBuf,
        out: outBuf,
        m,
        n,
        k,
        dtype: "f16",
        epilogue,
        epilogueInputs: [biasBuf],
        config: {
          ...DEFAULT_CONFIG,
          tileM: 32,
          tileN: 32,
          tileK: 16,
          threadTileM: 4,
          threadTileN: 4,
        },
      });

      const result = await readF16Buffer(outBuf, m * n);
      expect(arraysClose(result, new Float32Array(expected), 0.1, 0.5)).toBe(
        true,
      );

      aBuf.destroy();
      bBuf.destroy();
      biasBuf.destroy();
      outBuf.destroy();
    });
  });
});
