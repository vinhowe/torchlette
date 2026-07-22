/**
 * The `OptTerm → tile-IR` fold (derived-optimizer-realizer campaign, R1) — the
 * STRUCTURAL SIBLING of the graph interpreter `evalOptTerm`
 * (`ops/semantic/optimizer.ts`). Both are the SAME recursion over ONE algebra
 * (the 5-kind `OptTerm`); one lowers to runtime graph ops, this one lowers to
 * tile-IR expression nodes, so the fused optimizer kernel's per-element body
 * becomes a THEOREM of the `OptimizerProgram` rather than a hand-authored copy of
 * its arithmetic (docs/derived-optimizer-realizer-design.md, ruling O1).
 *
 * The fold is PURELY the elementwise arithmetic: each role binds to a caller-
 * supplied tile-IR expression (a `ctx.load(binding, idx)`, a `ctx.uniform(...)`,
 * or a scalar), and each `OptTerm` node maps to its one tile-IR expression node:
 *
 *     role → the bound expression        c   → ctx.f32(v)
 *     u    → BlockExpr.{neg,sqrt,sign,abs,exp}
 *     b    → BlockExpr.{add,sub,mul,div}
 *     mm   → OptimizerPackRefusal (STRUCTURAL — tile-IR carries matmul ONLY as a
 *            2-D block-op, never as an elementwise expression node, so the Muon
 *            Newton–Schulz refusal falls out of the type system, not a name-check)
 *
 * The fold NEVER branches on an optimizer name (the same discipline `evalOptTerm`
 * holds): its only inputs are an `OptTerm` and a role binding. That is what keeps
 * it a generic fold and not an Adam-shaped one (design §7.2 risk R4).
 */

import type { BlockExpr, KernelContext } from "../backend/webgpu/tile-ir";
import type { OptTerm } from "../ops/semantic/optimizer";
import { OptimizerPackRefusal } from "../optim/pack-optimizer";

/** Role name → its tile-IR value (a load, a uniform read, or a constant). */
export interface FoldRoleBindings {
  readonly [name: string]: BlockExpr;
}

/**
 * Fold an `OptTerm` into a tile-IR `BlockExpr` over the given role bindings.
 *
 * Memoized by term-object IDENTITY (mirroring `evalOptTerm`): a subterm shared by
 * object reuse folds ONCE, collapsing the interpretation to the term's DAG size.
 * tile-IR's own construction-time CSE would coalesce structurally-identical nodes
 * anyway, but the memo keeps the emitted DAG structurally faithful to the term.
 */
export function lowerOptTermToTileIR(
  t: OptTerm,
  ctx: KernelContext,
  roles: FoldRoleBindings,
  memo: Map<OptTerm, BlockExpr> = new Map(),
): BlockExpr {
  const cached = memo.get(t);
  if (cached !== undefined) return cached;
  const result = lowerOptTermCore(t, ctx, roles, memo);
  memo.set(t, result);
  return result;
}

function lowerOptTermCore(
  t: OptTerm,
  ctx: KernelContext,
  roles: FoldRoleBindings,
  memo: Map<OptTerm, BlockExpr>,
): BlockExpr {
  switch (t.k) {
    case "role": {
      const bound = roles[t.name];
      if (bound === undefined)
        throw new Error(
          `lowerOptTermToTileIR: program references role '${t.name}' but it was not bound.`,
        );
      return bound;
    }
    case "c":
      return ctx.f32(t.v);
    case "u": {
      const a = lowerOptTermToTileIR(t.a, ctx, roles, memo);
      switch (t.op) {
        case "neg":
          return a.neg();
        case "sqrt":
          return a.sqrt();
        case "sign":
          return a.sign();
        case "abs":
          return a.abs();
        case "exp":
          return a.exp();
      }
      break;
    }
    case "b": {
      const a = lowerOptTermToTileIR(t.a, ctx, roles, memo);
      const b = lowerOptTermToTileIR(t.b, ctx, roles, memo);
      switch (t.op) {
        case "add":
          return a.add(b);
        case "sub":
          return a.sub(b);
        case "mul":
          return a.mul(b);
        case "div":
          return a.div(b);
      }
      break;
    }
    case "mm":
      // STRUCTURAL refusal: there is no elementwise tile-IR node for a
      // contraction (matmul lives ONLY as a 2-D block-op). The fold CANNOT lower
      // Muon's Newton–Schulz — the typed refusal is enforced by the type system
      // at the fold seam, identical in kind to `assertFlattenable`'s hard gate.
      throw new OptimizerPackRefusal(
        `lowerOptTermToTileIR: a contraction (mm) node has no elementwise tile-IR ` +
          `target — the fold refuses structurally. Route the optimizer through the ` +
          `per-param path (this is the Muon Newton–Schulz refusal).`,
      );
  }
  throw new Error(
    `lowerOptTermToTileIR: unhandled term ${JSON.stringify(t)}`,
  );
}
