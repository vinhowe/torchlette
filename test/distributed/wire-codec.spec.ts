import { describe, expect, it } from "vitest";
import { decodeTensors, encodeTensors } from "../../src/distributed/wire-codec";

describe("wire-codec", () => {
  // Values exact in f16 (powers of two / small integers) so the roundtrip is
  // bit-exact and the test asserts equality rather than approximate.
  const shapes = [
    [2, 3],
    [4],
  ];
  const t0 = new Float32Array([0.5, -1.25, 3.0, 0.125, -0.5, 2.0]);
  const t1 = new Float32Array([1, 2, 3, 4]);

  it("f16 roundtrips at a 2-byte-aligned offset", () => {
    const payload = encodeTensors([t0, t1], "f16");
    const dec = decodeTensors(payload, shapes, "f16");
    expect(Array.from(dec[0])).toEqual(Array.from(t0));
    expect(Array.from(dec[1])).toEqual(Array.from(t1));
  });

  it("f16 decodes from an ODD byte offset (transport frame-header misalignment)", () => {
    // Regression for the v2 relay: the frame header leaves the f16 payload at
    // an odd byteOffset, and a Uint16Array view there throws "start offset
    // should be a multiple of 2". decodeTensors must read via DataView.
    const payload = encodeTensors([t0, t1], "f16");
    const framed = new Uint8Array(payload.byteLength + 1);
    framed.set(payload, 1);
    const odd = framed.subarray(1); // byteOffset = 1
    expect(odd.byteOffset % 2).toBe(1);
    const dec = decodeTensors(odd, shapes, "f16");
    expect(Array.from(dec[0])).toEqual(Array.from(t0));
    expect(Array.from(dec[1])).toEqual(Array.from(t1));
  });

  it("f32 roundtrips at an odd offset too", () => {
    const payload = encodeTensors([t0, t1], "f32");
    const framed = new Uint8Array(payload.byteLength + 1);
    framed.set(payload, 1);
    const dec = decodeTensors(framed.subarray(1), shapes, "f32");
    expect(Array.from(dec[0])).toEqual(Array.from(t0));
    expect(Array.from(dec[1])).toEqual(Array.from(t1));
  });
});
