/**
 * Wire payload codec for tensor-carrying protocol messages.
 *
 * The v2 protocol shipped raw Float32Array bytes — ~500MB per 124M exchange
 * round (pseudo-grad out + merged average back + per-anchor F16W resyncs),
 * which makes cross-DC training I/O-bound. Pseudo-gradients tolerate
 * aggressive quantization (the DiLoCo literature uses 4-bit; v1's E3M0 path
 * validated it here), so f16 on the wire is conservative: half the bytes,
 * ~1e-3 relative error on deltas, exact for the magnitudes f16w already
 * implies. The envelope self-describes via `wireDtype`, so mixed encodings
 * interoperate; TORCHLETTE_WIRE_DTYPE=f32 opts out.
 *
 * Scalar conversion uses lookup-free bit twiddling, vectorized over arrays.
 */
import { ENV } from "../core/env";

export type WireDtype = "f32" | "f16";

export function defaultWireDtype(): WireDtype {
  return ENV.TORCHLETTE_WIRE_DTYPE === "f32" ? "f32" : "f16";
}

const f32View = new Float32Array(1);
const u32View = new Uint32Array(f32View.buffer);

function f32BitsToF16Bits(f: number): number {
  const sign = (f >>> 16) & 0x8000;
  const exp = (f >>> 23) & 0xff;
  const frac = f & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (frac ? 0x200 : 0); // Inf/NaN
  const unbiased = exp - 127;
  if (unbiased > 15) return sign | 0x7c00; // overflow → Inf
  if (unbiased >= -14) {
    // Normal, round-to-nearest-even on the dropped 13 bits
    let mant = frac >>> 13;
    const rest = frac & 0x1fff;
    if (rest > 0x1000 || (rest === 0x1000 && (mant & 1) === 1)) mant++;
    let e = unbiased + 15;
    if (mant === 0x400) {
      mant = 0;
      e++;
      if (e >= 0x1f) return sign | 0x7c00;
    }
    return sign | (e << 10) | mant;
  }
  if (unbiased >= -24) {
    // Subnormal f16
    const shift = -14 - unbiased;
    const mant = (0x400 | (frac >>> 13)) >>> shift;
    return sign | mant;
  }
  return sign; // underflow → signed zero
}

function f16BitsToF32Bits(h: number): number {
  const sign = (h & 0x8000) << 16;
  const exp = (h >>> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0x1f) return sign | 0x7f800000 | (frac << 13); // Inf/NaN
  if (exp === 0) {
    if (frac === 0) return sign; // zero
    // Subnormal: normalize
    let e = -1;
    let m = frac;
    while ((m & 0x400) === 0) {
      m <<= 1;
      e++;
    }
    return sign | ((127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
  }
  return sign | ((exp - 15 + 127) << 23) | (frac << 13);
}

/** Encode tensors into one contiguous byte buffer in the given wire dtype. */
export function encodeTensors(
  arrays: Float32Array[],
  dtype: WireDtype,
): Uint8Array {
  let elems = 0;
  for (const a of arrays) elems += a.length;
  if (dtype === "f32") {
    const out = new Uint8Array(elems * 4);
    let pos = 0;
    for (const a of arrays) {
      out.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), pos);
      pos += a.byteLength;
    }
    return out;
  }
  const out = new Uint16Array(elems);
  let pos = 0;
  for (const a of arrays) {
    for (let i = 0; i < a.length; i++) {
      f32View[0] = a[i];
      out[pos++] = f32BitsToF16Bits(u32View[0]);
    }
  }
  return new Uint8Array(out.buffer);
}

/** Decode a contiguous payload back into per-shape Float32Arrays. */
export function decodeTensors(
  payload: Uint8Array,
  shapes: number[][],
  dtype: WireDtype,
): Float32Array[] {
  const out: Float32Array[] = [];
  if (dtype === "f32") {
    let pos = 0;
    for (const shape of shapes) {
      const n = shape.reduce((a, b) => a * b, 1);
      const arr = new Float32Array(n);
      new Uint8Array(arr.buffer).set(payload.subarray(pos, pos + n * 4));
      out.push(arr);
      pos += n * 4;
    }
    return out;
  }
  // Read f16 codes via DataView: the payload's byteOffset within its frame
  // buffer is arbitrary (the transport's header is not 2-byte aligned), and a
  // Uint16Array view requires an even byteOffset — so a direct view throws
  // "start offset should be a multiple of 2" on odd-offset frames. DataView
  // imposes no alignment. Little-endian matches the encoder's native-endian
  // Uint16Array store on our (LE) hosts.
  const dv = new DataView(
    payload.buffer,
    payload.byteOffset,
    payload.byteLength,
  );
  let pos = 0;
  for (const shape of shapes) {
    const n = shape.reduce((a, b) => a * b, 1);
    const arr = new Float32Array(n);
    const u32 = new Uint32Array(arr.buffer);
    for (let i = 0; i < n; i++)
      u32[i] = f16BitsToF32Bits(dv.getUint16((pos + i) * 2, true));
    out.push(arr);
    pos += n;
  }
  return out;
}
