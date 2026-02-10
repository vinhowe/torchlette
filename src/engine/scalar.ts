const CANONICAL_NAN_BITS = 0x7ff8000000000000n;

export function canonicalizeF64Bits(value: number): bigint {
  if (Number.isNaN(value)) {
    return CANONICAL_NAN_BITS;
  }

  const buffer = new ArrayBuffer(8);
  const view = new DataView(buffer);
  view.setFloat64(0, value, true);
  return view.getBigUint64(0, true);
}

export function encodeF64LE(value: number): Uint8Array {
  const buffer = new ArrayBuffer(8);
  const view = new DataView(buffer);
  view.setBigUint64(0, canonicalizeF64Bits(value), true);
  return new Uint8Array(buffer);
}

export { CANONICAL_NAN_BITS };
