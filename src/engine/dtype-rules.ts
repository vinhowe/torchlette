/**
 * Op Dtype Registry — single source of truth for every op's dtype behavior.
 *
 * Categories:
 * - "f16_eligible"    : compute-bound ops that benefit from f16 (matmul)
 * - "f32_required"    : numerically sensitive ops that need f32 (reductions, exp, log)
 * - "promote_inputs"  : binary ops where inputs must match dtype; promote to higher
 * - "preserve"        : output dtype = input dtype (reshape, relu, etc.)
 * - "always_f32"      : always produce f32 output (comparisons, tensorFromArray)
 * - "cast"            : explicit dtype cast
 */

import type { DType } from "../backend/types";
import type { LazyOpCode } from "./lazy";

export type OpDtypeCategory =
  | "f16_eligible"
  | "f32_required"
  | "promote_inputs"
  | "preserve"
  | "always_f32"
  | "cast";

export const OP_DTYPE_RULES: Record<LazyOpCode, { category: OpDtypeCategory }> = {
  // F16-eligible (compute-bound)
  matmul: { category: "f16_eligible" },

  // F32-required (numerically sensitive)
  sum:  { category: "f32_required" },
  mean: { category: "f32_required" },
  max:  { category: "f32_required" },
  exp:  { category: "f32_required" },
  log:  { category: "f32_required" },
  pow:  { category: "f32_required" },

  // Binary (promote mismatched dtypes)
  add: { category: "promote_inputs" },
  sub: { category: "promote_inputs" },
  mul: { category: "promote_inputs" },
  div: { category: "promote_inputs" },

  // Preserve input dtype
  sqrt:      { category: "preserve" },
  relu:      { category: "preserve" },
  neg:       { category: "preserve" },
  abs:       { category: "preserve" },
  tanh:      { category: "preserve" },
  sigmoid:   { category: "preserve" },
  gelu:      { category: "preserve" },
  silu:      { category: "preserve" },
  reshape:   { category: "preserve" },
  expand:    { category: "preserve" },
  transpose: { category: "preserve" },
  permute:   { category: "preserve" },
  contiguous:{ category: "preserve" },
  gather:    { category: "preserve" },
  scatterAdd:{ category: "preserve" },
  where:     { category: "preserve" },
  transfer:  { category: "preserve" },
  stridedScatterCopy: { category: "preserve" },
  stridedScatterAdd:  { category: "preserve" },
  tril:  { category: "preserve" },
  triu:  { category: "preserve" },

  // Always f32
  gt:    { category: "always_f32" },
  lt:    { category: "always_f32" },
  ge:    { category: "always_f32" },
  le:    { category: "always_f32" },
  eq:    { category: "always_f32" },
  ne:    { category: "always_f32" },
  argmax:{ category: "always_f32" },
  argmin:{ category: "always_f32" },
  isfinite:      { category: "always_f32" },
  tensorFromArray:{ category: "always_f32" },
  zeros: { category: "always_f32" },
  full:  { category: "always_f32" },
  arange:{ category: "always_f32" },

  // Explicit cast
  cast: { category: "cast" },
};

/**
 * Supplementary op names that exist in the frontend/AMP layer but not in LazyOpCode.
 * These are composite ops implemented as combinations of lazy ops.
 */
const SUPPLEMENTARY_F16_ELIGIBLE = new Set([
  "linear",
  "conv1d",
  "conv2d",
  "conv3d",
  "bmm",
  "addmm",
]);

const SUPPLEMENTARY_F32_REQUIRED = new Set([
  "softmax",
  "log_softmax",
  "layer_norm",
  "batch_norm",
  "group_norm",
  "loss",
  "cross_entropy",
  "mse_loss",
]);

// Derived sets from the registry
const _f16Eligible = new Set(
  Object.entries(OP_DTYPE_RULES)
    .filter(([_, r]) => r.category === "f16_eligible")
    .map(([k]) => k),
);
const _f32Required = new Set(
  Object.entries(OP_DTYPE_RULES)
    .filter(([_, r]) => r.category === "f32_required")
    .map(([k]) => k),
);

/** All f16-eligible ops (lazy + supplementary). */
export const F16_ELIGIBLE: ReadonlySet<string> = new Set([
  ..._f16Eligible,
  ...SUPPLEMENTARY_F16_ELIGIBLE,
]);

/** All f32-required ops (lazy + supplementary). */
export const F32_REQUIRED: ReadonlySet<string> = new Set([
  ..._f32Required,
  ...SUPPLEMENTARY_F32_REQUIRED,
]);

/**
 * Promote two dtypes for binary ops.
 * f16 + f32 → f32; same dtype → unchanged.
 */
export function promoteDtype(a: DType, b: DType): DType {
  if (a === b) return a;
  if ((a === "f16" && b === "f32") || (a === "f32" && b === "f16")) return "f32";
  return a;
}
