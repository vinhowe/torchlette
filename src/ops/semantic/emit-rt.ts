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
import { type GradGuard, vjpBinary, vjpUnary } from "./adjoint";
import type { ElementwiseDef } from "./catalog";
import type { Expr } from "./expr";

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

// ----------------------------------------------------------------------------
// Registry-compatible grad factories — the derived replacements for the hand
// lambdas. VJP terms are computed ONCE at construction; each backward call only
// interprets the (constant) term over the live runtime engine.
// ----------------------------------------------------------------------------

/** Derive a `UnaryGradFn` from a definition (design S2). */
export function makeUnaryGrad(def: ElementwiseDef): UnaryGradFn {
  const vjp: Expr = vjpUnary(def.expr, def.gradGuard as GradGuard | undefined);
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
