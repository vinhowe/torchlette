/**
 * E3M0 4-bit Quantization for Pseudo-Gradient Compression
 *
 * Format: 1 sign bit + 3 exponent bits + 0 mantissa bits = 4 bits per value.
 * Every representable value is a signed power of 2. Per-block shared max
 * exponent (8-bit) sets the dynamic range.
 *
 * Memory layout for N f32 values (block size = 8, matching pack size):
 *   - codes:  N/8 u32s (8 × 4-bit packed codes per u32)
 *   - scales: N/8 u32s (one 8-bit max_exp per block, 4 packed per u32)
 *
 * Compression: f32 (32 bits) → 4 bits + ~1 bit scale amortized ≈ 6.4x.
 * With block size 128+, scale overhead is negligible → ~8x.
 *
 * Used by Streaming DiLoCo for pseudo-gradient communication.
 */

/**
 * Quantize f32 array to E3M0 packed format (CPU reference implementation).
 * Used for testing and as spec for the GPU kernel.
 *
 * @param values - Input f32 values (length must be multiple of 8)
 * @returns { codes, scales } - Packed u32 arrays
 */
export function e3m0Quantize(values: Float32Array): {
  codes: Uint32Array;
  scales: Uint8Array;
} {
  const n = values.length;
  if (n % 8 !== 0) throw new Error("E3M0: length must be multiple of 8");

  const numBlocks = n / 8;
  const codes = new Uint32Array(numBlocks);
  const scales = new Uint8Array(numBlocks);

  for (let b = 0; b < numBlocks; b++) {
    const offset = b * 8;

    // Find max absolute value in block
    let maxAbs = 0;
    for (let i = 0; i < 8; i++) {
      const abs = Math.abs(values[offset + i]);
      if (abs > maxAbs) maxAbs = abs;
    }

    // Shared exponent: floor(log2(maxAbs))
    const maxExp = maxAbs > 0 ? Math.floor(Math.log2(maxAbs)) : 0;
    scales[b] = maxExp + 127; // bias to unsigned 8-bit (like IEEE float exponent)

    // Quantize each value to 4-bit code
    let packed = 0;
    for (let i = 0; i < 8; i++) {
      const v = values[offset + i];
      const sign = v < 0 ? 1 : 0;
      const abs = Math.abs(v);

      let expCode: number;
      if (abs === 0) {
        expCode = 0;
      } else {
        const e = Math.floor(Math.log2(abs));
        const relative = e - maxExp + 6; // 6 = 2^3 - 2 (max normal code)
        if (relative <= 0)
          expCode = 0; // flush to zero
        else if (relative >= 7)
          expCode = 7; // saturate
        else expCode = relative;
      }

      const code = (sign << 3) | expCode;
      packed |= code << (i * 4);
    }
    codes[b] = packed;
  }

  return { codes, scales };
}

/**
 * Dequantize E3M0 packed format back to f32 (CPU reference implementation).
 *
 * @param codes - Packed u32 array (8 × 4-bit codes per u32)
 * @param scales - Per-block max exponent (biased by 127)
 * @param n - Number of original f32 values
 * @returns Float32Array of dequantized values
 */
export function e3m0Dequantize(
  codes: Uint32Array,
  scales: Uint8Array,
  n: number,
): Float32Array {
  const out = new Float32Array(n);
  const numBlocks = n / 8;

  for (let b = 0; b < numBlocks; b++) {
    const packed = codes[b];
    const maxExp = scales[b] - 127; // unbias

    for (let i = 0; i < 8; i++) {
      const code = (packed >>> (i * 4)) & 0xf;
      const sign = (code >>> 3) & 1;
      const expCode = code & 0x7;

      if (expCode === 0) {
        out[b * 8 + i] = 0;
      } else {
        const value = 2 ** (expCode + maxExp - 6);
        out[b * 8 + i] = sign ? -value : value;
      }
    }
  }

  return out;
}

/**
 * Compute the compressed size in bytes for N f32 values.
 * codes: N/8 × 4 bytes + scales: N/8 × 1 byte = N × 0.625 bytes.
 * Compression ratio: 32 / 5 = 6.4x (with block size 8).
 */
export function e3m0CompressedSize(n: number): number {
  const numBlocks = Math.ceil(n / 8);
  return numBlocks * 4 + numBlocks; // codes (u32) + scales (u8)
}
