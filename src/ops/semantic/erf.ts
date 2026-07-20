/**
 * Semantic Derivation — the erf realization + GELU constants (Crystal Campaign
 * 3, Phase 2: COMPOSITES DERIVE).
 *
 * THE SINGLE SOURCE for two families of magic numbers that lived hand-copied
 * across FOUR surfaces (the "triplication" of design §1):
 *   - the Abramowitz–Stegun erf Horner polynomial (`ERF_A` / `ERF_P`), written
 *     character-for-character in `numeric.ts` `erf()`, `tile-ir.ts`
 *     `BlockExpr.erf()`, `fusion-tile-ir.ts` gelu_erf, and
 *     `custom-backward.ts` `geluErfBackward`;
 *   - the GELU-tanh constants (`GELU_SQRT_2_OVER_PI`, `GELU_TANH_C`), written in
 *     the CPU `gelu()`, the WGSL activation switch, and `geluTanhBackward`.
 *
 * `erf` is admitted as an ALGEBRA PRIMITIVE (like `exp`/`tanh`): its *realization*
 * is an approximation (the A-S poly, `erfApprox` / the WGSL emit), single-sourced
 * here; its *derivative* is the ANALYTIC gaussian `erf'(u) = 2/√π · e^(−u²)`
 * (adjoint.ts) — so a composite that uses erf differentiates to the exact
 * gaussian pdf, not the derivative of the polynomial approximation. This is the
 * split the design's §4.5 (guards/realization) and RT3 (composition-is-the-
 * reference-not-the-kernel) name: one MEANING, realized to each backend's
 * approximation within tolerance.
 */

/** Abramowitz & Stegun 7.1.26 coefficients a1..a5 (max err ~1.5e-7). */
export const ERF_A: readonly [number, number, number, number, number] = [
  0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
];

/** Abramowitz & Stegun 7.1.26 `p` (the `t = 1/(1+p|x|)` denominator coeff). */
export const ERF_P = 0.3275911;

/** 2/√π — the analytic-derivative coefficient of erf (`erf'(u)=2/√π·e^(−u²)`). */
export const TWO_OVER_SQRT_PI = 1.1283791670955126;

/** √(2/π) — the GELU-tanh inner coefficient. */
export const GELU_SQRT_2_OVER_PI = 0.7978845608;

/** The GELU-tanh cubic coefficient (`x + c·x³`). */
export const GELU_TANH_C = 0.044715;

/**
 * The A-S erf realization as an f64 scalar — the SINGLE source the CPU reference
 * interpreter reads for the `erf` primitive (design §4.3 S1). Byte-identical to
 * the deleted `numeric.ts` `erf()` (same coefficients, same Horner order).
 */
export function erfApprox(v: number): number {
  const sign = v < 0 ? -1 : 1;
  const ax = Math.abs(v);
  const t = 1.0 / (1.0 + ERF_P * ax);
  const [a1, a2, a3, a4, a5] = ERF_A;
  const poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
  return sign * (1.0 - poly * Math.exp(-ax * ax));
}
