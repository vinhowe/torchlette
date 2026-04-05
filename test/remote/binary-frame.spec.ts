import { describe, expect, it } from "vitest";
import {
  decodeBinaryFrame,
  encodeBinaryFrame,
  valuesToTypedArray,
} from "../../src/remote/binary-frame";

describe("binary-frame codec", () => {
  it("round-trips an f32 tensor", () => {
    const values = new Float32Array([1.5, -2.25, 3.125, 0]);
    const encoded = encodeBinaryFrame({
      id: 42,
      dtype: "f32",
      shape: [2, 2],
      values,
    });
    const decoded = decodeBinaryFrame(encoded);
    expect(decoded.id).toBe(42);
    expect(decoded.dtype).toBe("f32");
    expect(decoded.shape).toEqual([2, 2]);
    expect(Array.from(decoded.values)).toEqual([1.5, -2.25, 3.125, 0]);
  });

  it("round-trips an i32 tensor", () => {
    const values = new Int32Array([-1, 0, 1, 2147483647, -2147483648]);
    const encoded = encodeBinaryFrame({
      id: 7,
      dtype: "i32",
      shape: [5],
      values,
    });
    const decoded = decodeBinaryFrame(encoded);
    expect(decoded.id).toBe(7);
    expect(decoded.dtype).toBe("i32");
    expect(decoded.shape).toEqual([5]);
    expect(Array.from(decoded.values)).toEqual([
      -1, 0, 1, 2147483647, -2147483648,
    ]);
  });

  it("round-trips a 0-rank tensor", () => {
    const values = new Float32Array([42]);
    const encoded = encodeBinaryFrame({
      id: 1,
      dtype: "f32",
      shape: [],
      values,
    });
    const decoded = decodeBinaryFrame(encoded);
    expect(decoded.shape).toEqual([]);
    expect(Array.from(decoded.values)).toEqual([42]);
  });

  it("round-trips a 4-d tensor", () => {
    const values = new Float32Array(2 * 3 * 4 * 5);
    for (let i = 0; i < values.length; i++) values[i] = i * 0.5;
    const encoded = encodeBinaryFrame({
      id: 100,
      dtype: "f32",
      shape: [2, 3, 4, 5],
      values,
    });
    const decoded = decodeBinaryFrame(encoded);
    expect(decoded.shape).toEqual([2, 3, 4, 5]);
    expect(decoded.values.length).toBe(120);
    expect(Array.from(decoded.values.slice(0, 6))).toEqual([0, 0.5, 1, 1.5, 2, 2.5]);
  });

  it("binary encoding is much smaller than JSON array", () => {
    const values = new Float32Array(1000);
    for (let i = 0; i < values.length; i++) values[i] = Math.random();
    const encoded = encodeBinaryFrame({
      id: 1,
      dtype: "f32",
      shape: [1000],
      values,
    });
    const jsonSize = JSON.stringify({ values: Array.from(values) }).length;
    expect(encoded.byteLength).toBeLessThan(jsonSize / 3);
  });

  it("rejects truncated frames", () => {
    const tiny = new ArrayBuffer(4);
    expect(() => decodeBinaryFrame(tiny)).toThrow(/too small/);
  });

  it("valuesToTypedArray wraps number arrays", () => {
    const arr = valuesToTypedArray([1, 2, 3], "f32");
    expect(arr).toBeInstanceOf(Float32Array);
    expect(Array.from(arr)).toEqual([1, 2, 3]);

    const existing = new Float32Array([4, 5, 6]);
    expect(valuesToTypedArray(existing, "f32")).toBe(existing);
  });
});
