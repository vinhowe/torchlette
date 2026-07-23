/**
 * Semantic Derivation — the gradient graph emitter (surface S2, runtime seam).
 *
 * Interpret a derived VJP `Expr` over the `RuntimeEngine`: leaves bind to the
 * saved operands / upstream grad, primitives dispatch to `rt.*`. This builds the
 * exact backward tensor graph the hand `registry.ts` grad lambdas built — from
 * the ONE definition, via `adjoint` + `normalize`. The lambdas delete.
 *
 * Within-altitude (CPU) the derived graph reproduces the table byte-for-byte
 * (productized Probe 2); across the CPU↔GPU seam agreement is ULP-bounded
 * (design §4.5, the `pow`/`exp` precedent) — the gradcheck/unary-ops/oracle
 * suites are the seam gate.
 */

import type { RuntimeEngine } from "../../runtime/engine";
import type { Tensor as RuntimeTensor } from "../../runtime/tensor";
import type { BinaryTTGradFn, UnaryGradFn } from "../registry";
import { deriv, normalize, vjpBinary, vjpUnary } from "./adjoint";
import type { ElementwiseDef } from "./catalog";
import type {
  CompBinary,
  CompNode,
  CompositeDef,
  CompUnary,
} from "./composite";
import { ERF_A, ERF_P } from "./erf";
import {
  add,
  div as divE,
  type Expr,
  exp as expE,
  log as logE,
  mul as mulE,
  neg as negE,
  recip as recipE,
  sqrt as sqrtE,
  sub as subE,
  x as xE,
  y as yE,
} from "./expr";
import {
  broadcastOverDims,
  realizeIndexAdjoint,
  reduceToShape,
} from "./index-map";

type RtVal = number | RuntimeTensor;

/**
 * LAZY leaf resolvers, memoized per emit call. Lazy so a term that does not
 * reference a saved operand (e.g. add's grad `[g, g]`) never forces it — the
 * runtime's saved-tensor accessor throws when a non-saved input is read, and an
 * eager env would trip that even when the operand is unused.
 */
interface RtLeaves {
  x?: () => RtVal;
  y?: () => RtVal;
  g?: () => RtVal;
}

function need(resolver: (() => RtVal) | undefined, role: string): RtVal {
  if (resolver === undefined) {
    throw new Error(
      `emit-rt: gradient term references '${role}' but it was not provided.`,
    );
  }
  return resolver();
}

/** Memoize a resolver so a doubly-referenced operand is fetched once. */
function memo(fn: () => RtVal): () => RtVal {
  let cached: RtVal | undefined;
  let done = false;
  return () => {
    if (!done) {
      cached = fn();
      done = true;
    }
    return cached as RtVal;
  };
}

/** Interpret a term over the runtime engine, producing a tensor graph. */
function emit(e: Expr, rt: RuntimeEngine, env: RtLeaves): RtVal {
  switch (e.k) {
    case "x":
      return need(env.x, "x");
    case "y":
      return need(env.y, "y");
    case "g":
      return need(env.g, "g");
    case "c":
      return e.v;
    case "neg":
      return rt.neg(emitT(e.a, rt, env));
    case "exp":
      return rt.exp(emitT(e.a, rt, env));
    case "log":
      return rt.log(emitT(e.a, rt, env));
    case "sqrt":
      return rt.sqrt(emitT(e.a, rt, env));
    case "sin":
      return rt.sin(emitT(e.a, rt, env));
    case "cos":
      return rt.cos(emitT(e.a, rt, env));
    case "tanh":
      return rt.tanh(emitT(e.a, rt, env));
    case "abs":
      return rt.abs(emitT(e.a, rt, env));
    case "sign":
      return rt.sign(emitT(e.a, rt, env));
    case "recip": {
      const a = emit(e.a, rt, env);
      return typeof a === "number" ? 1 / a : rt.div(1, a);
    }
    case "erf":
      // Realize erf over the runtime via the A-S poly (the SAME single-sourced
      // coefficients as the CPU/WGSL realizations — no 2nd owner). Used by a
      // composite VJP that recomputes erf in backward (e.g. gelu_erf's cdf term).
      return emitErf(emitT(e.a, rt, env), rt);
    case "floor":
      return rt.floor(emitT(e.a, rt, env));
    case "ceil":
      return rt.ceil(emitT(e.a, rt, env));
    case "round":
      return rt.round(emitT(e.a, rt, env));
    case "isfinite":
      return rt.isfinite(emitT(e.a, rt, env));
    case "add":
      return rt.add(emit(e.a, rt, env), emit(e.b, rt, env));
    case "sub":
      return rt.sub(emit(e.a, rt, env), emit(e.b, rt, env));
    case "mul":
      return rt.mul(emit(e.a, rt, env), emit(e.b, rt, env));
    case "div":
      return rt.div(emit(e.a, rt, env), emit(e.b, rt, env));
    case "pow":
      return rt.pow(emit(e.a, rt, env), emit(e.b, rt, env));
    case "min":
      return rt.minimum(emit(e.a, rt, env), emit(e.b, rt, env));
    case "max":
      return rt.maximum(emit(e.a, rt, env), emit(e.b, rt, env));
    case "gt":
      return rt.gt(emit(e.a, rt, env), emit(e.b, rt, env));
    case "ge":
      return rt.ge(emit(e.a, rt, env), emit(e.b, rt, env));
    case "lt":
      return rt.lt(emit(e.a, rt, env), emit(e.b, rt, env));
    case "le":
      return rt.le(emit(e.a, rt, env), emit(e.b, rt, env));
    case "eq":
      return rt.eq(emit(e.a, rt, env), emit(e.b, rt, env));
    case "ne":
      return rt.ne(emit(e.a, rt, env), emit(e.b, rt, env));
    case "mod":
    case "where":
      throw new Error(
        `emit-rt: '${e.k}' has no elementwise-grad realization; it should not appear in a normalized derived VJP.`,
      );
  }
}

/** Emit a sub-term that MUST be a tensor (unary-op argument). */
function emitT(e: Expr, rt: RuntimeEngine, env: RtLeaves): RuntimeTensor {
  const v = emit(e, rt, env);
  if (typeof v === "number") {
    throw new Error(
      "emit-rt: expected a tensor argument but the term folded to a scalar.",
    );
  }
  return v;
}

function asTensor(v: RtVal): RuntimeTensor {
  if (typeof v === "number") {
    throw new Error(
      "emit-rt: a gradient must be a tensor, but the derived term reduced to a scalar.",
    );
  }
  return v;
}

/**
 * Realize erf(u) over the runtime engine — the A-S 7.1.26 poly, reading the ONE
 * `ERF_A`/`ERF_P` source. Byte-structure-identical to `erfApprox` (the CPU
 * reference) and `BlockExpr.erf` (WGSL); ULP-bounded across the CPU↔GPU seam.
 */
function emitErf(u: RuntimeTensor, rt: RuntimeEngine): RuntimeTensor {
  const [a1, a2, a3, a4, a5] = ERF_A;
  const au = rt.abs(u);
  const t = rt.div(1, rt.add(1, rt.mul(au, ERF_P)));
  // Horner: ((((a5·t + a4)·t + a3)·t + a2)·t + a1)·t
  const poly = rt.mul(
    rt.add(
      rt.mul(
        rt.add(rt.mul(rt.add(rt.mul(rt.add(rt.mul(t, a5), a4), t), a3), t), a2),
        t,
      ),
      a1,
    ),
    t,
  );
  const expTerm = rt.exp(rt.neg(rt.mul(au, au))); // exp(-u²)
  const erfAbs = rt.sub(1, rt.mul(poly, expTerm));
  return rt.mul(rt.sign(u), erfAbs);
}

// ----------------------------------------------------------------------------
// Registry-compatible grad factories — the derived replacements for the hand
// lambdas. VJP terms are computed ONCE at construction; each backward call only
// interprets the (constant) term over the live runtime engine.
// ----------------------------------------------------------------------------

/** Derive a `UnaryGradFn` from a definition (design S2). */
export function makeUnaryGrad(def: ElementwiseDef): UnaryGradFn {
  const vjp: Expr = vjpUnary(def.expr);
  return (rt, g, s) =>
    asTensor(
      emit(vjp, rt, {
        x: memo(() => {
          if (s === undefined) {
            throw new Error(
              `emit-rt: ${def.name} grad references its input but needsSave is false.`,
            );
          }
          return s;
        }),
        g: () => g,
      }),
    );
}

// ----------------------------------------------------------------------------
// Composite interpreter (design §4.4, Probe-4 shape) — realize a `CompositeDef`
// term over the runtime engine. This is the SEMANTIC REFERENCE: the decomposed/
// fused forward is checked to agree with it (test/semantic-composite.spec.ts).
// The fused kernel is NOT re-derived here (§4.4) — the composition is its
// reference, met at the schedule-state `SemanticRegionUid` seam.
// ----------------------------------------------------------------------------

/** Interpret a composition over `rt`, reducing along `dim`. `inputs` supplies the
 *  named roles (tensor operands + the scalar `eps`). */
export function interpretComposition(
  def: CompositeDef,
  rt: RuntimeEngine,
  dim: number,
  inputs: Readonly<Record<string, RtVal>>,
): RuntimeTensor {
  const go = (n: CompNode): RtVal => {
    switch (n.k) {
      case "in": {
        const v = inputs[n.role];
        if (v === undefined) {
          throw new Error(
            `interpretComposition: ${def.name} references role '${n.role}' but it was not provided.`,
          );
        }
        return v;
      }
      case "kc":
        return n.v;
      case "u": {
        const a = goT(n.a);
        switch (n.op) {
          case "exp":
            return rt.exp(a);
          case "log":
            return rt.log(a);
          case "sqrt":
            return rt.sqrt(a);
          case "rsqrt":
            return rt.rsqrt(a);
          case "neg":
            return rt.neg(a);
        }
        break;
      }
      case "b": {
        const a = go(n.a);
        const b = go(n.b);
        switch (n.op) {
          case "add":
            return rt.add(a, b);
          case "sub":
            return rt.sub(a, b);
          case "mul":
            return rt.mul(a, b);
          case "div":
            return rt.div(a, b);
        }
        break;
      }
      case "r": {
        const a = goT(n.a);
        const opts = { dim, keepdim: true } as const;
        const res =
          n.op === "sum"
            ? rt.sum(a, opts)
            : n.op === "max"
              ? rt.max(a, opts)
              : rt.mean(a, opts);
        if (typeof res === "number") {
          throw new Error(
            `interpretComposition: reduce '${n.op}' with keepdim returned a scalar.`,
          );
        }
        return res;
      }
      case "gi": {
        // gather-at-index — the index tensor comes from a named role (runtime
        // DATA, not part of the term), gathered along `dim`. Its adjoint is the
        // index algebra's gather transpose (scatter/onehot); this is the bridge
        // that lets cross_entropy complete as a composition (design §2 c/e).
        const a = goT(n.a);
        const index = inputs[n.indexRole];
        if (index === undefined || typeof index === "number") {
          throw new Error(
            `interpretComposition: ${def.name} gather references index role '${n.indexRole}' but no index tensor was provided.`,
          );
        }
        return rt.gather(a, index, { dim });
      }
    }
    throw new Error(
      `interpretComposition: unhandled node ${JSON.stringify(n)}`,
    );
  };
  const goT = (n: CompNode): RuntimeTensor => {
    const v = go(n);
    if (typeof v === "number") {
      throw new Error(
        "interpretComposition: expected a tensor sub-term but it folded to a scalar.",
      );
    }
    return v;
  };
  return asTensor(go(def.root));
}

// ----------------------------------------------------------------------------
// The CompNode-adjoint pass (F1, design §3) — reverse-mode VJP over the SAME
// `CompNode` composition the forward is authored from. It threads an upstream
// cotangent `ḡ` from `def.root` back to each input role, accumulating where a
// role is used more than once. It builds NO new chain rule, NO transpose, NO
// reduction fact — it COMPOSES three existing engines (design §3.1):
//   - `deriv` + `normalize` + `emit` (adjoint.ts) for the u/b elementwise local
//     factor (the same chain rule the elementwise VJPs already use);
//   - `broadcastOverDims` (index-map) for the sum/mean reduce-transpose, scaled
//     by the mean's 1/N (the reduction gradKind);
//   - `reduceToShape` (index-map) for the broadcast-transpose that reduces each
//     cotangent back to its operand's own shape (the §11.1 broadcast-adjoint
//     trap — never hand-rolled here);
//   - `realizeIndexAdjoint` (index-map) `scatterZeros` for the `gi` gather
//     transpose (CE's `softmax − onehot`).
// The `max`-reduce is the detached stability shift (§3.2 lemma) — its gradient is
// provably zero and dropped; a non-detached (load-bearing) max is out of scope
// (§3.3) and refused.
// ----------------------------------------------------------------------------

/** CompUnary → its `Expr` over `x` (so its local derivative REUSES `deriv`). */
function compUnaryExpr(op: CompUnary): Expr {
  switch (op) {
    case "exp":
      return expE(xE);
    case "log":
      return logE(xE);
    case "sqrt":
      return sqrtE(xE);
    case "rsqrt":
      return recipE(sqrtE(xE));
    case "neg":
      return negE(xE);
  }
}

/** CompBinary → its `Expr` over `x`,`y` (so its partials REUSE `deriv`). */
function compBinaryExpr(op: CompBinary): Expr {
  switch (op) {
    case "add":
      return add(xE, yE);
    case "sub":
      return subE(xE, yE);
    case "mul":
      return mulE(xE, yE);
    case "div":
      return divE(xE, yE);
  }
}

/**
 * Reverse-mode VJP over a `CompositeDef`: realize the input-role gradients over
 * `rt` given the saved operand roles (`inputs`) and the upstream cotangent
 * `gbar` (shape = the forward output). Returns `{ role → gradTensor }` for every
 * tensor-valued role reached (scalar roles like `eps` and the index role carry
 * no gradient). This is the composite analogue of `makeUnaryGrad` — the backward
 * sibling of `interpretComposition`.
 */
export function vjpComposition(
  def: CompositeDef,
  rt: RuntimeEngine,
  dim: number,
  inputs: Readonly<Record<string, RtVal>>,
  gbar: RuntimeTensor,
): Record<string, RuntimeTensor> {
  // --- Forward pass, memoized by node identity (shared subterms fold once). ---
  const fwd = new Map<CompNode, RtVal>();
  const forward = (n: CompNode): RtVal => {
    const hit = fwd.get(n);
    if (hit !== undefined) return hit;
    const v = forwardCore(n);
    fwd.set(n, v);
    return v;
  };
  const forwardT = (n: CompNode): RuntimeTensor => {
    const v = forward(n);
    if (typeof v === "number") {
      throw new Error(
        "vjpComposition: expected a tensor sub-term but it folded to a scalar.",
      );
    }
    return v;
  };
  const forwardCore = (n: CompNode): RtVal => {
    switch (n.k) {
      case "in": {
        const v = inputs[n.role];
        if (v === undefined)
          throw new Error(
            `vjpComposition: ${def.name} references role '${n.role}' but it was not provided.`,
          );
        return v;
      }
      case "kc":
        return n.v;
      case "u": {
        const a = forwardT(n.a);
        switch (n.op) {
          case "exp":
            return rt.exp(a);
          case "log":
            return rt.log(a);
          case "sqrt":
            return rt.sqrt(a);
          case "rsqrt":
            return rt.rsqrt(a);
          case "neg":
            return rt.neg(a);
        }
        break;
      }
      case "b": {
        const a = forward(n.a);
        const b = forward(n.b);
        switch (n.op) {
          case "add":
            return rt.add(a, b);
          case "sub":
            return rt.sub(a, b);
          case "mul":
            return rt.mul(a, b);
          case "div":
            return rt.div(a, b);
        }
        break;
      }
      case "r": {
        const a = forwardT(n.a);
        const opts = { dim, keepdim: true } as const;
        const res =
          n.op === "sum"
            ? rt.sum(a, opts)
            : n.op === "max"
              ? rt.max(a, opts)
              : rt.mean(a, opts);
        if (typeof res === "number")
          throw new Error("vjpComposition: keepdim reduce returned a scalar.");
        return res;
      }
      case "gi": {
        const a = forwardT(n.a);
        const index = inputs[n.indexRole];
        if (index === undefined || typeof index === "number")
          throw new Error(
            `vjpComposition: ${def.name} gather references index role '${n.indexRole}' but no index tensor was provided.`,
          );
        return rt.gather(a, index, { dim });
      }
    }
    throw new Error(`vjpComposition: unhandled node ${JSON.stringify(n)}`);
  };

  // --- Reverse-topological order (post-order DFS, reversed → parents first). ---
  const order: CompNode[] = [];
  const seen = new Set<CompNode>();
  const visit = (n: CompNode): void => {
    if (seen.has(n)) return;
    seen.add(n);
    switch (n.k) {
      case "u":
      case "r":
        visit(n.a);
        break;
      case "b":
        visit(n.a);
        visit(n.b);
        break;
      case "gi":
        visit(n.a);
        break;
    }
    order.push(n);
  };
  visit(def.root);
  order.reverse();

  // --- Cotangent sweep. ---
  const cot = new Map<CompNode, RuntimeTensor>();
  const grads: Record<string, RuntimeTensor> = {};
  cot.set(def.root, gbar);

  // Push a contribution into a child, reduced back to the child's own forward
  // shape (the broadcast-transpose) — never hand-rolled, always `reduceToShape`.
  const push = (child: CompNode, contrib: RuntimeTensor): void => {
    const target = forward(child);
    if (typeof target === "number") return; // const / eps: no gradient sink
    const reduced = reduceToShape(rt, contrib, target.shape);
    const prev = cot.get(child);
    cot.set(child, prev === undefined ? reduced : rt.add(prev, reduced));
  };

  // Local elementwise factor via the SAME `deriv`+`normalize`+`emit` engine the
  // elementwise VJPs use — `xE`/`yE` bind to the operands' forward values.
  const localFactor = (defExpr: Expr, wrt: "x" | "y", env: RtLeaves): RtVal =>
    emit(normalize(deriv(defExpr, wrt)), rt, env);

  for (const n of order) {
    const g = cot.get(n);
    if (g === undefined) continue; // no cotangent reached this node
    switch (n.k) {
      case "in": {
        if (typeof forward(n) === "number") break; // scalar role: no grad
        grads[n.role] =
          grads[n.role] === undefined ? g : rt.add(grads[n.role], g);
        break;
      }
      case "kc":
        break;
      case "u": {
        const aVal = forward(n.a);
        const factor = localFactor(compUnaryExpr(n.op), "x", {
          x: () => aVal,
        });
        push(n.a, asTensor(rt.mul(g, factor)));
        break;
      }
      case "b": {
        const aVal = forward(n.a);
        const bVal = forward(n.b);
        const env: RtLeaves = { x: () => aVal, y: () => bVal };
        const be = compBinaryExpr(n.op);
        push(n.a, asTensor(rt.mul(g, localFactor(be, "x", env))));
        push(n.b, asTensor(rt.mul(g, localFactor(be, "y", env))));
        break;
      }
      case "r": {
        if (n.op === "max") {
          // The detached stability shift (§3.2) — grad provably zero, dropped.
          if (n.detach !== true)
            throw new Error(
              "vjpComposition: a load-bearing (non-detached) `max`-reduce needs the " +
                "argmax mask-scatter VJP, which is out of scope (§3.3). Annotate the " +
                "reduce `detach:true` iff its gradient is provably zero.",
            );
          break;
        }
        const aShape = forwardT(n.a).shape;
        let back = broadcastOverDims(rt, g, aShape, [dim], true);
        if (n.op === "mean") {
          const nRed = aShape[dim < 0 ? dim + aShape.length : dim];
          back = rt.mul(back, 1 / nRed);
        }
        push(n.a, back);
        break;
      }
      case "gi": {
        const aShape = forwardT(n.a).shape;
        const index = inputs[n.indexRole] as RuntimeTensor;
        const scattered = realizeIndexAdjoint(
          rt,
          { k: "scatterZeros", dim, inShape: aShape },
          g,
          { index },
        );
        push(n.a, asTensor(scattered as RuntimeTensor));
        break;
      }
    }
  }
  return grads;
}

/** Derive a `BinaryTTGradFn` from a definition (design S2). */
export function makeBinaryTTGrad(def: ElementwiseDef): BinaryTTGradFn {
  const [dA, dB] = vjpBinary(def.expr);
  return (rt, g, gs) => {
    const leaves: RtLeaves = {
      x: memo(() => gs(0)),
      y: memo(() => gs(1)),
      g: () => g,
    };
    return [asTensor(emit(dA, rt, leaves)), asTensor(emit(dB, rt, leaves))];
  };
}
