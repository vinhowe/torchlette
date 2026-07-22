/**
 * Structural gate — the DERIVED fused Adam body (fork C, the SOLE path since R4),
 * compile-only (cpu). The numeric derived==program differential is the fold-parity
 * (`tools/optterm-fold-parity.ts`); this cpu gate pins the STRUCTURE the derived
 * kernel commits to:
 *
 *   (a) the kernel binds a `bc` DATA input (bias correction is DATA, a [2]
 *       host-computed live scalar) — never a `t` step-counter binding;
 *   (b) NO in-kernel `expm1` bias-correction prelude — the derived WGSL carries
 *       none of the Horner-series constants the deleted authored body emitted;
 *   (c) grad/param/m/v/lr are bound as usual;
 *
 * across the full 5-variant corpus (useVec4 × emitF16 × emitUnscale). R4
 * (2026-07-22): the authored body is deleted, so there is nothing to compare
 * against — this asserts the derived body's standing invariants directly.
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

// The 1/120 = 0.0083333 leading Horner coefficient the deleted authored expm1
// prelude emitted — its absence proves bias correction is DATA, not in-kernel.
const EXPM1_MARKER = "0.00833";

describe("derived fused Adam — structural contract (fork C, sole path)", () => {
  for (const v of variants) {
    const key = `${v.useVec4}:${v.emitF16}:${v.emitUnscale}`;
    it(`[${key}] binds bc (not t); grad/param/m/v/lr present`, () => {
      const der = generateAdamShaderTileIR(v.useVec4, v.emitF16, v.emitUnscale);
      const db = bindingsOf(der);
      expect(db).toContain("bc");
      expect(db).not.toContain("t");
      for (const shared of ["grad", "param", "m", "v", "lr"]) {
        expect(db).toContain(shared);
      }
    });

    it(`[${key}] carries no in-kernel expm1 prelude (bias correction is DATA)`, () => {
      const der = generateAdamShaderTileIR(v.useVec4, v.emitF16, v.emitUnscale);
      expect(der).not.toContain(EXPM1_MARKER);
    });
  }
});
