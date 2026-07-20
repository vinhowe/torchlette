/**
 * Semantic Derivation — the adjoint pass (surface S2).
 *
 * Backward is not authored per-op; it is ONE structural chain-rule pass over the
 * forward term, followed by a canonical normalizer. The op's VJP is
 * `g · d(def)/d(operand)`, normalized to reach the hand table's byte form
 * (design §4.2). Probe 2: 19/22 derived VJPs byte-match the table; the 3 named
 * divergences resolve here —
 *   - `div.dB` sign-of-zero  → the CONSERVATIVE normalizer (`0·u→0`, `u·1→u`,
 *     `0−u→neg u`) — exact IEEE folds that never perturb rounding, so the other
 *     19 stay byte-exact.
 *   - `log`/`sqrt` grad       → the epsilon is a NUMERICAL GUARD (policy, not
 *     meaning): a `denomEps` annotation the derivation injects at the gradient
 *     denominator (§4.5). PyTorch (the oracle) computes the UNGUARDED adjoint
 *     `g/x`, `g·0.5/√x`; the table's `+1e-8` biases the grad near 0 (log: 870
 *     ulp @ x≈1e-4). The guard is preserved for behavior-parity and made
 *     explicit/reviewable, not inherited silently.
 */

import {
  add,
  c,
  cos,
  div,
  type Expr,
  ge,
  gt,
  le,
  lt,
  mul,
  neg,
  powE,
  recip,
  sign,
  sin,
  sqrt,
  sub,
  tanh,
  where,
} from "./expr";

// ----------------------------------------------------------------------------
// The adjoint (chain-rule) pass — d(expr)/d(v) for v in {"x","y"}.
// ----------------------------------------------------------------------------

export function deriv(e: Expr, wrt: "x" | "y"): Expr {
  switch (e.k) {
    case "x":
      return wrt === "x" ? c(1) : c(0);
    case "y":
      return wrt === "y" ? c(1) : c(0);
    case "g":
    case "c":
      return c(0);
    case "neg":
      return neg(deriv(e.a, wrt));
    case "exp":
      return mul({ k: "exp", a: e.a }, deriv(e.a, wrt)); // d e^u = e^u u'
    case "log":
      return div(deriv(e.a, wrt), e.a); // d ln u = u'/u
    case "sqrt":
      return div(deriv(e.a, wrt), mul(c(2), sqrt(e.a))); // u'/(2√u)
    case "sin":
      return mul(cos(e.a), deriv(e.a, wrt));
    case "cos":
      return mul(neg(sin(e.a)), deriv(e.a, wrt));
    case "tanh":
      return mul(sub(c(1), mul(tanh(e.a), tanh(e.a))), deriv(e.a, wrt));
    case "abs":
      return mul(sign(e.a), deriv(e.a, wrt));
    case "recip":
      return neg(div(deriv(e.a, wrt), mul(e.a, e.a))); // -u'/u²
    // Rounding / discontinuous primitives: subgradient 0.
    case "sign":
    case "floor":
    case "ceil":
    case "round":
    case "isfinite":
      return c(0);
    case "add":
      return add(deriv(e.a, wrt), deriv(e.b, wrt));
    case "sub":
      return sub(deriv(e.a, wrt), deriv(e.b, wrt));
    case "mul":
      return add(mul(deriv(e.a, wrt), e.b), mul(e.a, deriv(e.b, wrt)));
    case "div":
      return div(
        sub(mul(deriv(e.a, wrt), e.b), mul(e.a, deriv(e.b, wrt))),
        mul(e.b, e.b),
      );
    case "pow": {
      // Constant-exponent pow only (the tsGrad path). Variable-exponent pow
      // (a^b·ln a) is OUT of the pure-elementwise adjoint scope — it stays a
      // hand-authored rule (design §2, probe scoping).
      if (e.b.k === "c" && wrt === "x") {
        return mul(mul(c(e.b.v), powE(e.a, c(e.b.v - 1))), deriv(e.a, wrt));
      }
      return c(NaN); // signal: not handled by the pure-elementwise adjoint
    }
    case "min":
      return where(le(e.a, e.b), deriv(e.a, wrt), deriv(e.b, wrt));
    case "max":
      return where(ge(e.a, e.b), deriv(e.a, wrt), deriv(e.b, wrt));
    case "mod":
      // d/da (a mod b) = 1 (a.e.); d/db is discontinuous → subgradient 0.
      return wrt === "x" ? deriv(e.a, wrt) : c(0);
    case "gt":
    case "ge":
    case "lt":
    case "le":
    case "eq":
    case "ne":
      return c(0); // comparisons: subgradient 0
    case "where":
      // Differentiate the branches, not the condition (the condition's
      // subgradient is 0 a.e.).
      return where(e.c, deriv(e.a, wrt), deriv(e.b, wrt));
  }
}

// ----------------------------------------------------------------------------
// Conservative normalizer — ONLY exact IEEE-preserving folds. These never
// change f32 rounding (×0=0, ×1=id, +0=id, −0=id, 0−u=−u, const-fold), so ops
// that already byte-match the table (Probe 2's 19) stay byte-exact while
// `div.dB`'s unsimplified `(0·y − x·1)/y²` collapses to `−x/y²`.
// ----------------------------------------------------------------------------

const isConst = (e: Expr, v: number): boolean => e.k === "c" && e.v === v;

function negateCompare(op: string): "gt" | "ge" | "lt" | "le" | "eq" | "ne" {
  switch (op) {
    case "gt":
      return "le";
    case "ge":
      return "lt";
    case "lt":
      return "ge";
    case "le":
      return "gt";
    case "eq":
      return "ne";
    default:
      return "eq";
  }
}

function normalizeStep(e: Expr): Expr {
  switch (e.k) {
    case "x":
    case "y":
    case "g":
    case "c":
      return e;
    case "neg": {
      const a = normalizeStep(e.a);
      if (a.k === "c") return c(-a.v); // const-fold
      if (a.k === "neg") return a.a; // neg(neg u) → u
      return neg(a);
    }
    case "exp":
    case "log":
    case "sqrt":
    case "sin":
    case "cos":
    case "tanh":
    case "abs":
    case "sign":
    case "recip":
    case "floor":
    case "ceil":
    case "round":
    case "isfinite": {
      return { k: e.k, a: normalizeStep(e.a) };
    }
    case "add": {
      const a = normalizeStep(e.a);
      const b = normalizeStep(e.b);
      if (isConst(a, 0)) return b; // 0 + u → u
      if (isConst(b, 0)) return a; // u + 0 → u
      if (a.k === "c" && b.k === "c") return c(a.v + b.v);
      return add(a, b);
    }
    case "sub": {
      const a = normalizeStep(e.a);
      const b = normalizeStep(e.b);
      if (isConst(b, 0)) return a; // u − 0 → u
      if (isConst(a, 0)) return normalizeStep(neg(b)); // 0 − u → neg u
      if (a.k === "c" && b.k === "c") return c(a.v - b.v);
      return sub(a, b);
    }
    case "mul": {
      const a = normalizeStep(e.a);
      const b = normalizeStep(e.b);
      if (isConst(a, 0) || isConst(b, 0)) return c(0); // 0·u → 0
      if (isConst(a, 1)) return b; // 1·u → u
      if (isConst(b, 1)) return a; // u·1 → u
      if (isConst(a, -1)) return normalizeStep(neg(b)); // -1·u → neg u (exact)
      if (isConst(b, -1)) return normalizeStep(neg(a)); // u·-1 → neg u (exact)
      if (a.k === "c" && b.k === "c") return c(a.v * b.v);
      return mul(a, b);
    }
    case "div": {
      const a = normalizeStep(e.a);
      const b = normalizeStep(e.b);
      return div(a, b);
    }
    case "pow":
      return powE(normalizeStep(e.a), normalizeStep(e.b));
    case "min":
      return { k: "min", a: normalizeStep(e.a), b: normalizeStep(e.b) };
    case "max":
      return { k: "max", a: normalizeStep(e.a), b: normalizeStep(e.b) };
    case "mod":
      return { k: "mod", a: normalizeStep(e.a), b: normalizeStep(e.b) };
    case "gt":
    case "ge":
    case "lt":
    case "le":
    case "eq":
    case "ne":
      return { k: e.k, a: normalizeStep(e.a), b: normalizeStep(e.b) };
    case "where": {
      const cnd = normalizeStep(e.c);
      const a = normalizeStep(e.a);
      const b = normalizeStep(e.b);
      // where(cond, 1, 0) → cond ; where(cmp, 0, 1) → ¬cmp. Exact: a comparison
      // yields exactly {0,1}. Collapses relu/min/max grads to the table's form.
      if (isConst(a, 1) && isConst(b, 0)) return cnd;
      if (
        isConst(a, 0) &&
        isConst(b, 1) &&
        (cnd.k === "gt" ||
          cnd.k === "ge" ||
          cnd.k === "lt" ||
          cnd.k === "le" ||
          cnd.k === "eq" ||
          cnd.k === "ne")
      ) {
        return { k: negateCompare(cnd.k), a: cnd.a, b: cnd.b };
      }
      return where(cnd, a, b);
    }
  }
}

/** Fixpoint of the conservative normalizer. */
export function normalize(e: Expr): Expr {
  let cur = e;
  for (let i = 0; i < 32; i++) {
    const next = normalizeStep(cur);
    if (JSON.stringify(next) === JSON.stringify(cur)) return next;
    cur = next;
  }
  return cur;
}

// ----------------------------------------------------------------------------
// Numerical guards (design §4.5) — a policy annotation, never the formula.
// Fixed vocabulary (Q1 recommendation: rule-of-three; only `denomEps` observed).
// ----------------------------------------------------------------------------

export interface GradGuard {
  /** Add this epsilon to the gradient's reciprocal/division denominator. */
  denomEps: number;
}

/**
 * Canonicalize a guarded grad's division form so `denomEps` lands on the exact
 * atom the hand table guards (design §4.5). Applied ONLY to guarded ops (log,
 * sqrt) — pure ops never see these rounding-changing rewrites.
 *   mul(a, div(1, b))         → div(a, b)                (log: g·(1/x) → g/x)
 *   div(p, k·u) / div(p, u·k) → div(p/k, u)              (sqrt: 1/(2√x) → 0.5/√x)
 */
function canonicalizeDivForm(e: Expr): Expr {
  const rec = (n: Expr): Expr => {
    switch (n.k) {
      case "mul": {
        const a = rec(n.a);
        const b = rec(n.b);
        if (b.k === "div" && isConst(b.a, 1)) return div(a, b.b);
        if (a.k === "div" && isConst(a.a, 1)) return div(b, a.b);
        return mul(a, b);
      }
      case "div": {
        const a = rec(n.a);
        const b = rec(n.b);
        if (a.k === "c" && b.k === "mul") {
          if (b.a.k === "c" && b.a.v !== 0) return div(c(a.v / b.a.v), b.b);
          if (b.b.k === "c" && b.b.v !== 0) return div(c(a.v / b.b.v), b.a);
        }
        return div(a, b);
      }
      case "neg":
      case "exp":
      case "log":
      case "sqrt":
      case "sin":
      case "cos":
      case "tanh":
      case "abs":
      case "sign":
      case "recip":
      case "floor":
      case "ceil":
      case "round":
      case "isfinite":
        return { k: n.k, a: rec(n.a) };
      case "add":
      case "sub":
      case "pow":
      case "min":
      case "max":
      case "mod":
      case "gt":
      case "ge":
      case "lt":
      case "le":
      case "eq":
      case "ne":
        return { k: n.k, a: rec(n.a), b: rec(n.b) } as Expr;
      case "where":
        return where(rec(n.c), rec(n.a), rec(n.b));
      default:
        return n;
    }
  };
  return rec(e);
}

/** Add `eps` to the denominator of the first div/recip in pre-order. */
function applyDenomEps(e: Expr, eps: number): Expr {
  let applied = false;
  const rec = (n: Expr): Expr => {
    if (applied) return n;
    switch (n.k) {
      case "div":
        applied = true;
        return div(n.a, add(n.b, c(eps)));
      case "recip":
        applied = true;
        return recip(add(n.a, c(eps)));
      case "neg":
      case "exp":
      case "log":
      case "sqrt":
      case "sin":
      case "cos":
      case "tanh":
      case "abs":
      case "sign":
      case "floor":
      case "ceil":
      case "round":
      case "isfinite":
        return { k: n.k, a: rec(n.a) };
      case "mul":
      case "add":
      case "sub":
      case "pow":
      case "min":
      case "max":
      case "mod":
      case "gt":
      case "ge":
      case "lt":
      case "le":
      case "eq":
      case "ne":
        return { k: n.k, a: rec(n.a), b: rec(n.b) } as Expr;
      case "where":
        return where(rec(n.c), rec(n.a), rec(n.b));
      default:
        return n;
    }
  };
  return rec(e);
}

// ----------------------------------------------------------------------------
// The public VJP builders — the derived gradient graph as a term.
// ----------------------------------------------------------------------------

/** g · d(def)/dx, normalized (+ guard applied when annotated). */
export function vjpUnary(def: Expr, guard?: GradGuard): Expr {
  let vjp = normalize(mul({ k: "g" }, deriv(def, "x")));
  if (guard) {
    vjp = normalize(applyDenomEps(canonicalizeDivForm(vjp), guard.denomEps));
  }
  return vjp;
}

/** [g · d/dx, g · d/dy] for a binary op, normalized. */
export function vjpBinary(def: Expr): [Expr, Expr] {
  return [
    normalize(mul({ k: "g" }, deriv(def, "x"))),
    normalize(mul({ k: "g" }, deriv(def, "y"))),
  ];
}
