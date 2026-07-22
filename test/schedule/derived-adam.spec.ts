/**
 * R2 structural gate — the DERIVED fused Adam body (fork B), compile-only (cpu).
 *
 * The numeric derived==authored differential is the GPU tool
 * `tools/derived-adam-parity.ts` (Δparam ≤ ~3e-8, Δm=Δv bit-exact; the two named
 * reassociation lemmas). This cpu gate pins the STRUCTURE fork B commits to:
 *
 *   (a) the derived kernel binds a `bc` DATA input at the `t` slot (bias
 *       correction is DATA, computed graph-side) — the authored `t` binding is
 *       GONE from the derived body;
 *   (b) the in-kernel `expm1` bias-correction prelude DIES — the derived WGSL
 *       carries none of the Horner-series constants the authored body emits;
 *   (c) the derived body is STRICTLY SMALLER than the authored one (the deleted
 *       prelude + the un-reassociated magnitude);
 *
 * across the full 5-variant corpus (useVec4 × emitF16 × emitUnscale).
 */

import { describe, expect, it } from "vitest";
import { generateAdamShaderTileIR } from "../../src/schedule/adam-skeleton";

interface Variant {
  useVec4: boolean;
  emitF16: boolean;
  emitUnscale: boolean;
}
const variants: Variant[] = [
  { useVec4: false, emitF16: false, emitUnscale: false },
  { useVec4: false, emitF16: true, emitUnscale: false },
  { useVec4: false, emitF16: false, emitUnscale: true },
  { useVec4: true, emitF16: false, emitUnscale: true },
  { useVec4: true, emitF16: true, emitUnscale: true },
];

const bindingsOf = (wgsl: string): string[] =>
  [...wgsl.matchAll(/var<storage[^;]*> (\w+)/g)].map((m) => m[1]);

// The 1/120 = 0.0083333 leading Horner coefficient the authored expm1 emits —
// a distinctive marker of the in-kernel bias-correction prelude.
const EXPM1_MARKER = "0.00833";

describe("R2 derived fused Adam — structural contract (fork B)", () => {
  for (const v of variants) {
    const key = `${v.useVec4}:${v.emitF16}:${v.emitUnscale}`;
    it(`[${key}] derived binds bc (not t); authored binds t`, () => {
      const auth = generateAdamShaderTileIR(v.useVec4, v.emitF16, v.emitUnscale, false);
      const der = generateAdamShaderTileIR(v.useVec4, v.emitF16, v.emitUnscale, true);
      const ab = bindingsOf(auth);
      const db = bindingsOf(der);
      expect(ab).toContain("t");
      expect(ab).not.toContain("bc");
      expect(db).toContain("bc");
      expect(db).not.toContain("t");
      // grad/param/m/v/lr are unchanged across both.
      for (const shared of ["grad", "param", "m", "v", "lr"]) {
        expect(ab).toContain(shared);
        expect(db).toContain(shared);
      }
    });

    it(`[${key}] derived kills the in-kernel expm1 prelude`, () => {
      const auth = generateAdamShaderTileIR(v.useVec4, v.emitF16, v.emitUnscale, false);
      const der = generateAdamShaderTileIR(v.useVec4, v.emitF16, v.emitUnscale, true);
      expect(auth).toContain(EXPM1_MARKER);
      expect(der).not.toContain(EXPM1_MARKER);
      // The derived body is strictly smaller (prelude gone).
      expect(der.length).toBeLessThan(auth.length);
    });
  }
});
