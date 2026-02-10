import { describe, expect, it } from "vitest";
import { CANONICAL_NAN_BITS, canonicalizeF64Bits, encodeF64LE } from "../src";

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join(
    "",
  );
}

describe("cache key stability", () => {
  it("distinguishes +0.0 and -0.0", () => {
    const plusZero = canonicalizeF64Bits(0);
    const minusZero = canonicalizeF64Bits(-0);

    expect(plusZero).not.toBe(minusZero);
  });

  it("canonicalizes NaN bit patterns", () => {
    const nanBits = canonicalizeF64Bits(Number.NaN);
    const otherNanBits = canonicalizeF64Bits(0 / 0);

    expect(nanBits).toBe(CANONICAL_NAN_BITS);
    expect(otherNanBits).toBe(CANONICAL_NAN_BITS);
  });

  it("encodes scalars as little-endian bytes deterministically", () => {
    const bytes = encodeF64LE(1);
    expect(bytesToHex(bytes)).toBe("000000000000f03f");
  });
});
