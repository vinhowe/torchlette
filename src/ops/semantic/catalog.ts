/**
 * Semantic Derivation — the elementwise definition catalog (Phase 0).
 *
 * The single source for the elementwise family: each op's formula as an `Expr`
 * term. From these the CPU reference body (interpret), the gradient (adjoint),
 * and — where the formula is shared — the WGSL emitter all derive. These terms
 * are EXACTLY the formulas that lived hand-copied in `numeric.ts` `UNARY_OPS`/
 * `BINARY_OPS` and the `registry.ts` grad lambdas; those copies delete.
 *
 * `gradPolicy`:
 *   - "derive": gradient is the adjoint of `expr` (the deletion target).
 *   - "none":   non-differentiable (floor/ceil/round/sign/isfinite) — null grad.
 *   - "hand":   grad stays hand-authored (variable-exponent pow — out of the
 *               pure-elementwise adjoint scope, design §2).
 */

import {
  abs,
  add,
  c,
  ceil,
  cos,
  div,
  type Expr,
  exp,
  floor,
  gt,
  isfinite,
  log,
  maxE,
  minE,
  mul,
  neg,
  powE,
  recip,
  round,
  sign,
  sin,
  sqrt,
  sub,
  tanh,
  where,
  x,
  y,
} from "./expr";

export type GradPolicy = "derive" | "none" | "hand";

export interface ElementwiseDef {
  name: string;
  arity: 1 | 2;
  /** The forward formula — the single source. */
  expr: Expr;
  gradPolicy: GradPolicy;
}

/**
 * Unary elementwise definitions — EXACTLY the `numeric.ts` UNARY_OPS formulas.
 * sigmoid/silu/rsqrt/relu DERIVE from the primitives (no new forward primitive).
 */
export const UNARY_DEFS: readonly ElementwiseDef[] = [
  {
    name: "sqrt",
    arity: 1,
    expr: sqrt(x),
    // grad DERIVED unguarded — matches the torch oracle (design §18 ruling).
    gradPolicy: "derive",
  },
  { name: "exp", arity: 1, expr: exp(x), gradPolicy: "derive" },
  {
    name: "log",
    arity: 1,
    expr: log(x),
    // grad DERIVED unguarded — matches the torch oracle (design §18 ruling).
    gradPolicy: "derive",
  },
  { name: "neg", arity: 1, expr: neg(x), gradPolicy: "derive" },
  { name: "abs", arity: 1, expr: abs(x), gradPolicy: "derive" },
  { name: "tanh", arity: 1, expr: tanh(x), gradPolicy: "derive" },
  // sigmoid = 1/(1+e^-x)
  {
    name: "sigmoid",
    arity: 1,
    expr: recip(add(c(1), exp(neg(x)))),
    gradPolicy: "derive",
  },
  // silu = x/(1+e^-x)
  {
    name: "silu",
    arity: 1,
    expr: div(x, add(c(1), exp(neg(x)))),
    gradPolicy: "derive",
  },
  { name: "sin", arity: 1, expr: sin(x), gradPolicy: "derive" },
  { name: "cos", arity: 1, expr: cos(x), gradPolicy: "derive" },
  // rsqrt = 1/√x
  { name: "rsqrt", arity: 1, expr: recip(sqrt(x)), gradPolicy: "derive" },
  { name: "floor", arity: 1, expr: floor(x), gradPolicy: "none" },
  { name: "ceil", arity: 1, expr: ceil(x), gradPolicy: "none" },
  { name: "round", arity: 1, expr: round(x), gradPolicy: "none" },
  { name: "sign", arity: 1, expr: sign(x), gradPolicy: "none" },
  { name: "isfinite", arity: 1, expr: isfinite(x), gradPolicy: "none" },
  // relu = x>0 ? x : 0
  {
    name: "relu",
    arity: 1,
    expr: where(gt(x, c(0)), x, c(0)),
    gradPolicy: "derive",
  },
];

/**
 * Binary elementwise definitions. add/mul/pow/minimum/maximum are the
 * `numeric.ts` BINARY_OPS; sub/div carry their own CPU bodies but their tensor
 * gradients derive here (registry ttGrad). Variable-exponent `pow` keeps its
 * hand grad.
 */
export const BINARY_DEFS: readonly ElementwiseDef[] = [
  { name: "add", arity: 2, expr: add(x, y), gradPolicy: "derive" },
  // sub's tensor grad is handled STRUCTURALLY by the frontend (no registry
  // ttGrad); its formula still derives its CPU body / documents its meaning.
  { name: "sub", arity: 2, expr: sub(x, y), gradPolicy: "hand" },
  { name: "mul", arity: 2, expr: mul(x, y), gradPolicy: "derive" },
  { name: "div", arity: 2, expr: div(x, y), gradPolicy: "derive" },
  { name: "pow", arity: 2, expr: powE(x, y), gradPolicy: "hand" },
  { name: "minimum", arity: 2, expr: minE(x, y), gradPolicy: "derive" },
  { name: "maximum", arity: 2, expr: maxE(x, y), gradPolicy: "derive" },
];
