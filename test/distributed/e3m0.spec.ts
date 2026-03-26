import { describe, expect, it } from "vitest";
import {
  e3m0CompressedSize,
  e3m0Dequantize,
  e3m0Quantize,
} from "../../src/distributed/e3m0";

describe("E3M0 4-bit quantization", () => {
  it("round-trips powers of 2 exactly", () => {
    const values = new Float32Array([1, 2, 4, 8, 0.5, 0.25, -1, -4]);
    const { codes, scales } = e3m0Quantize(values);
    const restored = e3m0Dequantize(codes, scales, values.length);
    for (let i = 0; i < values.length; i++) {
      expect(restored[i]).toBe(values[i]);
    }
  });

  it("preserves zero exactly", () => {
    const values = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]);
    const { codes, scales } = e3m0Quantize(values);
    const restored = e3m0Dequantize(codes, scales, values.length);
    for (let i = 0; i < values.length; i++) {
      expect(restored[i]).toBe(0);
    }
  });

  it("preserves sign", () => {
    const values = new Float32Array([1, -1, 2, -2, 4, -4, 8, -8]);
    const { codes, scales } = e3m0Quantize(values);
    const restored = e3m0Dequantize(codes, scales, values.length);
    for (let i = 0; i < values.length; i++) {
      expect(Math.sign(restored[i])).toBe(Math.sign(values[i]));
      expect(Math.abs(restored[i])).toBe(Math.abs(values[i]));
    }
  });

  it("handles multiple blocks", () => {
    const values = new Float32Array(24); // 3 blocks of 8
    for (let i = 0; i < 24; i++) values[i] = (i + 1) * (i % 2 === 0 ? 1 : -1);
    const { codes, scales } = e3m0Quantize(values);
    expect(codes.length).toBe(3);
    expect(scales.length).toBe(3);
    const restored = e3m0Dequantize(codes, scales, 24);
    expect(restored.length).toBe(24);
  });

  it("flushes small values to zero when block max is large", () => {
    // Block max is 128, values < 128/64 = 2 should flush to zero
    const values = new Float32Array([128, 1, 0.5, 0.1, 0.01, 0, -128, 64]);
    const { codes, scales } = e3m0Quantize(values);
    const restored = e3m0Dequantize(codes, scales, 8);
    expect(restored[0]).toBe(128); // exact
    expect(restored[1]).toBe(0); // flushed (too small relative to 128)
    expect(restored[6]).toBe(-128); // exact
    expect(restored[7]).toBe(64); // exact
  });

  it("reports correct compressed size", () => {
    // N values → N/8 u32s (codes) + N/8 u8s (scales)
    expect(e3m0CompressedSize(8)).toBe(5); // 4 + 1
    expect(e3m0CompressedSize(16)).toBe(10); // 8 + 2
    expect(e3m0CompressedSize(1024)).toBe(640); // 512 + 128
  });

  it("achieves ~6.4x compression from f32", () => {
    const n = 1024;
    const ratio = (n * 4) / e3m0CompressedSize(n);
    expect(ratio).toBeCloseTo(6.4, 1);
  });

  it("throws on non-multiple-of-8 length", () => {
    expect(() => e3m0Quantize(new Float32Array(7))).toThrow("multiple of 8");
  });

  it("has bounded relative error on uniform-magnitude inputs", () => {
    // When all values have similar magnitude, quantization error is low
    const n = 1024;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = (Math.random() - 0.5) * 2; // uniform in [-1, 1]
    }
    const { codes, scales } = e3m0Quantize(values);
    const restored = e3m0Dequantize(codes, scales, n);

    let mse = 0;
    let energy = 0;
    for (let i = 0; i < n; i++) {
      mse += (values[i] - restored[i]) ** 2;
      energy += values[i] ** 2;
    }
    // Relative RMSE should be < 50% for uniform-magnitude data
    const relRmse = Math.sqrt(mse / energy);
    expect(relRmse).toBeLessThan(0.5);
  });
});
