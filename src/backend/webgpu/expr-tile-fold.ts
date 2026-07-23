/**
 * The `Expr → tile-IR` fold (COMPOSITE-CLOSURE F2) — the STRUCTURAL SIBLING of
 * `lowerOptTermToTileIR` (`schedule/optterm-fold.ts`) and of the runtime emitter
 * `emit` (`ops/semantic/emit-rt.ts`). All three are ONE recursion over the closed
 * `Expr` algebra; this one lowers to tile-IR `BlockExpr` nodes, so a fused
 * forward-activation WGSL body becomes a THEOREM of the op's `Expr` definition
 * (`catalog.ts` / `composite.ts`) rather than a hand-authored copy of its
 * arithmetic (design §4.1).
 *
 * The fold is PURELY the elementwise arithmetic: the `x` role binds to a
 * caller-supplied `BlockExpr` (a `ctx.load(...)`, a kernel value), and each
 * `Expr` kind maps to its one `BlockExpr` expression node:
 *
 *   x        → roles.x                 c(v)   → ctx.f32(v)
 *   neg exp log sqrt tanh sin cos abs sign floor ceil round → BlockExpr.{same}()
 *   recip u  → ctx.f32(1).div(u)       (the one Expr node BlockExpr lacks)
 *   add sub mul div min max pow mod    → BlockExpr.{same}(b)
 *   gt ge lt le eq ne                  → BlockExpr.{cmp}(b)
 *   where(c,a,b)                       → c.select(a, b)
 *   erf u    → the A-S Horner over ERF_A/ERF_P (the ONE source, erf.ts) — the
 *              same poly `BlockExpr.erf()`/`emitErf` build, emitted once here.
 *
 * `y`/`g` (a gradient term's operands) and `isfinite` are NOT activation-forward
 * nodes; they throw — a forward activation never references them.
 */

import { ERF_A, ERF_P, type Expr } from "../../ops/semantic";
import type { BlockExpr, KernelContext } from "./tile-ir";

/** Role name → its tile-IR value. Activations bind only `x`. */
export interface ExprFoldRoles {
  readonly x: BlockExpr;
}

/** Inline the A-S 7.1.26 erf poly over a `BlockExpr` (the ONE `ERF_A`/`ERF_P`
 *  source) — structurally identical to `BlockExpr.erf`/`emitErf`. */
function foldErf(a: BlockExpr, ctx: KernelContext): BlockExpr {
  const [a1, a2, a3, a4, a5] = ERF_A;
  const absX = a.abs();
  const t = ctx.f32(1).div(ctx.f32(1).add(absX.mul(ERF_P)));
  // Horner: ((((a5·t + a4)·t + a3)·t + a2)·t + a1)·t
  const poly = t
    .mul(a5)
    .add(a4)
    .mul(t)
    .add(a3)
    .mul(t)
    .add(a2)
    .mul(t)
    .add(a1)
    .mul(t);
  const expTerm = absX.neg().mul(absX).exp(); // exp(-x²)
  return a.sign().mul(ctx.f32(1).sub(poly.mul(expTerm)));
}

/**
 * Fold an `Expr` into a tile-IR `BlockExpr` over the given role binding.
 * Memoized by term-object identity (mirroring the sibling folds): a subterm
 * shared by object reuse folds ONCE. tile-IR's own construction-time CSE would
 * coalesce structurally-identical nodes anyway, but the memo keeps the emitted
 * DAG structurally faithful to the term.
 */
export function lowerExprToTileIR(
  e: Expr,
  ctx: KernelContext,
  roles: ExprFoldRoles,
  memo: Map<Expr, BlockExpr> = new Map(),
): BlockExpr {
  const cached = memo.get(e);
  if (cached !== undefined) return cached;
  const result = lowerExprCore(e, ctx, roles, memo);
  memo.set(e, result);
  return result;
}

function lowerExprCore(
  e: Expr,
  ctx: KernelContext,
  roles: ExprFoldRoles,
  memo: Map<Expr, BlockExpr>,
): BlockExpr {
  const rec = (sub: Expr): BlockExpr =>
    lowerExprToTileIR(sub, ctx, roles, memo);
  switch (e.k) {
    case "x":
      return roles.x;
    case "c":
      return ctx.f32(e.v);
    case "y":
    case "g":
      throw new Error(
        `lowerExprToTileIR: '${e.k}' is a gradient-term operand, not a forward activation node.`,
      );
    // -- Unary (1:1 with BlockExpr) --
    case "neg":
      return rec(e.a).neg();
    case "exp":
      return rec(e.a).exp();
    case "log":
      return rec(e.a).log();
    case "sqrt":
      return rec(e.a).sqrt();
    case "tanh":
      return rec(e.a).tanh();
    case "sin":
      return rec(e.a).sin();
    case "cos":
      return rec(e.a).cos();
    case "abs":
      return rec(e.a).abs();
    case "sign":
      return rec(e.a).sign();
    case "floor":
      return rec(e.a).floor();
    case "ceil":
      return rec(e.a).ceil();
    case "round":
      return rec(e.a).round();
    case "recip":
      return ctx.f32(1).div(rec(e.a));
    case "erf":
      return foldErf(rec(e.a), ctx);
    case "isfinite":
      throw new Error(
        "lowerExprToTileIR: 'isfinite' is not an activation node (no BlockExpr target).",
      );
    // -- Binary (1:1 with BlockExpr) --
    case "add":
      return rec(e.a).add(rec(e.b));
    case "sub":
      return rec(e.a).sub(rec(e.b));
    case "mul":
      return rec(e.a).mul(rec(e.b));
    case "div":
      return rec(e.a).div(rec(e.b));
    case "pow":
      return rec(e.a).pow(rec(e.b));
    case "min":
      return rec(e.a).min(rec(e.b));
    case "max":
      return rec(e.a).max(rec(e.b));
    case "mod":
      return rec(e.a).mod(rec(e.b));
    // -- Comparisons --
    case "gt":
      return rec(e.a).gt(rec(e.b));
    case "ge":
      return rec(e.a).ge(rec(e.b));
    case "lt":
      return rec(e.a).lt(rec(e.b));
    case "le":
      return rec(e.a).le(rec(e.b));
    case "eq":
      return rec(e.a).eq(rec(e.b));
    case "ne":
      return rec(e.a).ne(rec(e.b));
    // -- Ternary --
    case "where":
      return rec(e.c).select(rec(e.a), rec(e.b));
  }
}
