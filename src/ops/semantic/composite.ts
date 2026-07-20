/**
 * Semantic Derivation — the COMPOSITE definitions (Crystal Campaign 3, Phase 2).
 *
 * A composite's meaning is a COMPOSITION of the primitive algebra (design §2
 * category e, §4.4). The GELU family is a *pure-elementwise* composition — a
 * unary `Expr` over `x` — so it reuses the P0 machinery wholesale: its CPU
 * reference derives via `interpret`, its gradient via `adjoint`, its runtime
 * grad-graph via `emit-rt`. No new engine; the P0 algebra is amortized (design
 * §12 "P2 DELETES against this same algebra with no new engine").
 *
 * THE PAYOFF (design §1, §13): the erf polynomial written 3–4× and the GELU-tanh
 * constants written 3× collapse to ONE source (`erf.ts` + these terms). The two
 * GELU custom backwards (`geluTanhBackward`/`geluErfBackward`) derive as the
 * adjoint of these compositions and DELETE.
 *
 * Note the erf/tanh split (design §4.5): the forward uses the `erf` primitive
 * (realized by the A-S poly) / `tanh`; the derived backward differentiates them
 * ANALYTICALLY (erf'(u)=2/√π·e^(−u²); tanh'(u)=1−tanh²u), so gelu_erf's grad is
 * the exact gaussian pdf `cdf + x·φ(x)`, matching the analytic math — NOT the
 * derivative of the polynomial approximation.
 */

import type { ElementwiseDef } from "./catalog";
import { add, c, erf, type Expr, mul, tanh, x } from "./expr";
import { GELU_SQRT_2_OVER_PI, GELU_TANH_C } from "./erf";

const HALF = c(0.5);
const ONE = c(1);

// x³ = x·x·x (mul chain — the frontend already lowers integer pow to a mul
// chain; the term stays in the primitive algebra).
const X3: Expr = mul(x, mul(x, x));

/**
 * GELU-tanh (GPT-2 "new GELU"):
 *   0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
 * Constants single-sourced from `erf.ts` (`GELU_SQRT_2_OVER_PI`, `GELU_TANH_C`).
 * Byte-identical to the deleted `numeric.ts` tanh-gelu body.
 */
export const GELU_TANH_DEF: ElementwiseDef = {
  name: "gelu",
  arity: 1,
  expr: mul(
    mul(HALF, x),
    add(
      ONE,
      tanh(mul(c(GELU_SQRT_2_OVER_PI), add(x, mul(c(GELU_TANH_C), X3)))),
    ),
  ),
  gradPolicy: "derive",
};

/**
 * GELU-erf (exact form):
 *   0.5·x·(1 + erf(x/√2))
 * `erf` is the algebra primitive (A-S realization single-sourced); its analytic
 * adjoint makes the derived backward the exact gaussian `cdf + x·φ(x)`.
 * Byte-identical to the deleted `numeric.ts` erf-gelu body.
 */
export const GELU_ERF_DEF: ElementwiseDef = {
  name: "gelu_erf",
  arity: 1,
  expr: mul(mul(HALF, x), add(ONE, erf(mul(x, c(Math.SQRT1_2))))),
  gradPolicy: "derive",
};

// ===========================================================================
// The reduction-composites (design §2 category e, §4.4): softmax / log_softmax
// / rmsnorm / layernorm as `Composition` DATA over the primitive algebra + the
// P1 reduction monoids. A composite's DEFINITION is a composition of primitives;
// its FUSED KERNEL stays schedule-state-derived, REFERENCING the composition via
// the `SemanticRegionUid` seam (§4.4 — NOT re-owned here). The composition is
// the single semantic SOURCE: the CPU/decomposed forward is checked to agree
// with it (Probe 4 shape, test/semantic-composite.spec.ts); the fused kernel is
// checked against it by the EXISTING schedule-state per-move differential (RT3).
// ===========================================================================

/** A reduce folds along the composition's `dim` (softmax's dim / a norm's last
 *  dim), supplied at interpret time — the term is dim-agnostic DATA. */
export type CompReduceOp = "sum" | "max" | "mean";
export type CompUnary = "exp" | "log" | "sqrt" | "rsqrt" | "neg";
export type CompBinary = "add" | "sub" | "mul" | "div";

/** A composition term — a small tensor dataflow (design §4.1 composite frame). */
export type CompNode =
  | { k: "in"; role: string } // a named tensor/scalar input (x, w, b, eps)
  | { k: "kc"; v: number } // a constant
  | { k: "u"; op: CompUnary; a: CompNode }
  | { k: "b"; op: CompBinary; a: CompNode; b: CompNode }
  | { k: "r"; op: CompReduceOp; a: CompNode } // reduce along dim, keepdim=true
  // gather-at-index — the P4 INDEX-SPACE bridge: select `a`'s values at the
  // integer indices carried by the `indexRole` tensor, along the composition's
  // `dim`. Its adjoint is the index algebra's gather transpose (scatter/onehot),
  // so a composition that gathers differentiates through `index-map.ts` — this is
  // what lets `cross_entropy` COMPLETE as a composition (design §2 category c/e).
  | { k: "gi"; a: CompNode; indexRole: string };

/** A named composite op: the roles it consumes + its composition term. */
export interface CompositeDef {
  name: string;
  /** Input roles (tensor operands + scalar `eps`) the composition references. */
  roles: readonly string[];
  /** The default `eps` guard (design §4.5), if the op carries one. */
  eps?: number;
  root: CompNode;
}

const cin = (role: string): CompNode => ({ k: "in", role });
const cu = (op: CompUnary, a: CompNode): CompNode => ({ k: "u", op, a });
const cb = (op: CompBinary, a: CompNode, b: CompNode): CompNode => ({
  k: "b",
  op,
  a,
  b,
});
const cr = (op: CompReduceOp, a: CompNode): CompNode => ({ k: "r", op, a });

const IN_X = cin("x");

// softmax = exp(x − max(x)) / sum(exp(x − max(x)))   (max/sum along dim)
const SM_SHIFTED = cb("sub", IN_X, cr("max", IN_X));
export const SOFTMAX_DEF: CompositeDef = {
  name: "softmax",
  roles: ["x"],
  root: cb("div", cu("exp", SM_SHIFTED), cr("sum", cu("exp", SM_SHIFTED))),
};

// log_softmax = (x − max) − log(sum(exp(x − max)))   (the log-sum-exp form)
export const LOG_SOFTMAX_DEF: CompositeDef = {
  name: "log_softmax",
  roles: ["x"],
  root: cb("sub", SM_SHIFTED, cu("log", cr("sum", cu("exp", SM_SHIFTED)))),
};

// rmsnorm = x · rsqrt(mean(x²) + eps) · w   (the design's Probe-4 composition)
export const RMSNORM_DEF: CompositeDef = {
  name: "rmsnorm",
  roles: ["x", "w", "eps"],
  eps: 1e-6,
  root: cb(
    "mul",
    cb(
      "mul",
      IN_X,
      cu("rsqrt", cb("add", cr("mean", cb("mul", IN_X, IN_X)), cin("eps"))),
    ),
    cin("w"),
  ),
};

// layernorm = (x − mean)/sqrt(mean((x−mean)²) + eps) · w + b
const LN_CENTERED = cb("sub", IN_X, cr("mean", IN_X));
export const LAYERNORM_DEF: CompositeDef = {
  name: "layernorm",
  roles: ["x", "w", "b", "eps"],
  eps: 1e-5,
  root: cb(
    "add",
    cb(
      "mul",
      cb(
        "div",
        LN_CENTERED,
        cu(
          "sqrt",
          cb(
            "add",
            cr("mean", cb("mul", LN_CENTERED, LN_CENTERED)),
            cin("eps"),
          ),
        ),
      ),
      cin("w"),
    ),
    cin("b"),
  ),
};

/**
 * cross_entropy per-sample = −log_softmax(logits)[target]  (COMPLETED, P4).
 *
 * The `log_softmax` sub-term IS a composition (`LOG_SOFTMAX_DEF.root` reused);
 * the per-row gather at `target` is now the `gi` INDEX-SPACE node (design §2
 * category c/e), whose adjoint derives through the index algebra
 * (`index-map.ts`: gather's transpose is scatter/onehot — so CE's grad is
 * `softmax − onehot(target)`, the exact fused-kernel semantics). The batch mean
 * is applied by the caller's reduction (as the fused kernel returns per-sample
 * loss), so the composition term is the per-sample loss — the single semantic
 * SOURCE for CE that P2 deferred until the index family existed.
 */
export const CROSS_ENTROPY_DEF: CompositeDef = {
  name: "cross_entropy",
  roles: ["x", "target"],
  root: cu("neg", { k: "gi", a: LOG_SOFTMAX_DEF.root, indexRole: "target" }),
};

/** The reduction-composite catalog authored as data (softmax core family). */
export const REDUCTION_COMPOSITE_DEFS: readonly CompositeDef[] = [
  SOFTMAX_DEF,
  LOG_SOFTMAX_DEF,
  RMSNORM_DEF,
  LAYERNORM_DEF,
  CROSS_ENTROPY_DEF,
];

const COMP_KINDS = new Set(["in", "kc", "u", "b", "r", "gi"]);

/**
 * Prove a composition term is DATA (the `assertNoDefinitionBody` analogue for the
 * composite frame): every leaf is an input role / const, every node a known
 * dataflow kind — no smuggled JS body, WGSL string, or buffer. A composite whose
 * "definition" hid the decomposed `softmaxImpl` behind an opaque leaf is
 * unconstructible (the covenant/R22 defense).
 */
export function assertNoCompositionBody(node: CompNode, path = "comp"): void {
  const walk = (v: unknown, p: string): void => {
    if (v === null || v === undefined)
      throw new Error(`CompositeDef schema gate: ${p} is null/undefined.`);
    const t = typeof v;
    if (t === "function")
      throw new Error(
        `CompositeDef schema gate: ${p} is a function (not DATA).`,
      );
    if (t === "string" || t === "number" || t === "boolean") return;
    if (t === "object") {
      if (ArrayBuffer.isView(v) || v instanceof ArrayBuffer)
        throw new Error(`CompositeDef schema gate: ${p} is a buffer leaf.`);
      const kind = (v as { k?: unknown }).k;
      if (typeof kind !== "string" || !COMP_KINDS.has(kind))
        throw new Error(
          `CompositeDef schema gate: ${p} has kind ${JSON.stringify(kind)}.`,
        );
      for (const [key, val] of Object.entries(v as Record<string, unknown>)) {
        if (key === "k" || key === "op" || key === "role") continue;
        walk(val, `${p}.${key}`);
      }
      return;
    }
    throw new Error(`CompositeDef schema gate: ${p} has non-data type ${t}.`);
  };
  walk(node, path);
}
