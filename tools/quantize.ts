/**
 * Weight-only int8 quantization utility (symmetric, per-group along K).
 *
 * See docs/quantization-design.md. Lives in tools/ per the admission-pressure
 * rule — graduates to src/ if it earns generality (a second model / browser).
 *
 * A Linear weight is [N, K] (out, in) row-major. We quantize per group of G
 * contiguous-in-K weights to a signed int8, sharing one f16 scale per group.
 * Packing: 4 int8 / u32 word, packed 4-along-K into [N, K/4] u32.
 *
 * The GPU dequant uses WGSL `unpack4x8snorm(word)[b]` = q/127, so we bake the
 * ·127 into the stored scale: scale127 = groupAbsMax / 127 · 127 = groupAbsMax.
 * i.e. the stored f16 "scale" is exactly the group's abs-max, and the kernel
 * computes `unpack4x8snorm(word)[b] * scale127` = (q/127)·absMax = q·(absMax/127)
 * = dequantized weight. Single source: this file owns the mapping; the kernel
 * only multiplies.
 */

/** f32 → f16 bit pattern (round-to-nearest-even). */
export function f32ToF16Bits(val: number): number {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = val;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = x & 0x7fffff;
  if (exp <= 0) {
    // Subnormal / underflow to zero (weights this small round to 0).
    if (exp < -10) return sign;
    const m = (mant | 0x800000) >>> (1 - exp);
    // round-to-nearest-even
    const round = (m & 0x1000) !== 0 ? 1 : 0;
    return (sign | ((m >>> 13) + round)) & 0xffff;
  }
  if (exp >= 0x1f) return sign | 0x7c00; // overflow → inf
  // round-to-nearest-even on the 13 dropped mantissa bits
  const halfMant = mant >>> 13;
  const remainder = mant & 0x1fff;
  let out = sign | (exp << 10) | halfMant;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfMant & 1) === 1)) {
    out += 1;
  }
  return out & 0xffff;
}

/** f16 bit pattern → f32 (matches examples loaders). */
export function f16BitsToF32(bits: number): number {
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

export type QuantizedWeight = {
  /** Packed signed int8, 4/word, [N, K/4] u32 row-major. */
  packed: Uint32Array;
  /** Per-group abs-max as f16 bits, [N, K/G] row-major. This IS the kernel's
   *  "scale127" (see file header): dequant = unpack4x8snorm(word)[b] * scale. */
  scales: Uint16Array;
  n: number;
  k: number;
  groupSize: number;
};

/**
 * Quantize a [N, K] f32 weight to symmetric per-group int8.
 * Requires K % 4 === 0 (packing) and K % groupSize === 0 (grouping).
 */
export function quantizeLinearWeight(
  weight: Float32Array,
  n: number,
  k: number,
  groupSize = 64,
): QuantizedWeight {
  if (weight.length !== n * k) {
    throw new Error(
      `quantize: expected ${n * k} elements, got ${weight.length}`,
    );
  }
  if (k % 4 !== 0) throw new Error(`quantize: K=${k} must be divisible by 4`);
  if (k % groupSize !== 0) {
    throw new Error(
      `quantize: K=${k} must be divisible by groupSize=${groupSize}`,
    );
  }
  const groupsPerRow = k / groupSize;
  const wordsPerRow = k / 4;
  const packed = new Uint32Array(n * wordsPerRow);
  const scales = new Uint16Array(n * groupsPerRow);

  for (let row = 0; row < n; row++) {
    const rowBase = row * k;
    // Per-group abs-max → scale.
    for (let g = 0; g < groupsPerRow; g++) {
      let absMax = 0;
      const gBase = rowBase + g * groupSize;
      for (let j = 0; j < groupSize; j++) {
        const a = Math.abs(weight[gBase + j]);
        if (a > absMax) absMax = a;
      }
      // Store abs-max as f16; a zero group → scale 0 (all q become 0).
      // Round-trip the scale through f16 so quantize error matches the kernel's
      // f16 scale read exactly (single source at the seam).
      const scaleF16 = f32ToF16Bits(absMax);
      scales[row * groupsPerRow + g] = scaleF16;
      const scale = f16BitsToF32(scaleF16); // dequant multiplier the kernel sees
      const inv = scale > 0 ? 127 / scale : 0;
      // Quantize this group's weights.
      for (let j = 0; j < groupSize; j++) {
        const q = Math.max(
          -127,
          Math.min(127, Math.round(weight[gBase + j] * inv)),
        );
        const kIdx = g * groupSize + j;
        const word = row * wordsPerRow + (kIdx >> 2);
        const byte = kIdx & 3;
        // Store signed int8 in the byte lane (two's complement, masked).
        packed[word] |= (q & 0xff) << (byte * 8);
      }
    }
  }
  return { packed, scales, n, k, groupSize };
}

/**
 * Dequantize back to f32 [N, K] — the reference the kernel-level gate compares
 * against (and the "same dequantized values" fed to the f16 control path).
 * Mirrors the GPU: value = unpack4x8snorm(word)[b] * scale = (q/127) * absMax.
 */
export function dequantizeToF32(q: QuantizedWeight): Float32Array {
  const { packed, scales, n, k, groupSize } = q;
  const groupsPerRow = k / groupSize;
  const wordsPerRow = k / 4;
  const out = new Float32Array(n * k);
  for (let row = 0; row < n; row++) {
    for (let kIdx = 0; kIdx < k; kIdx++) {
      const word = packed[row * wordsPerRow + (kIdx >> 2)];
      const byte = kIdx & 3;
      // Extract signed int8 from the byte lane.
      let qi = (word >>> (byte * 8)) & 0xff;
      if (qi >= 128) qi -= 256;
      const g = (kIdx / groupSize) | 0;
      const scale = f16BitsToF32(scales[row * groupsPerRow + g]);
      out[row * k + kIdx] = (qi / 127) * scale;
    }
  }
  return out;
}
