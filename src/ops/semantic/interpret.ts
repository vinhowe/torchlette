/**
 * Semantic Derivation — the CPU reference interpreter (surface S1).
 *
 * Interpret an `Expr` as an f64 scalar function; this IS the CPU reference body
 * (design §4.3 S1 — the interpreter itself is the reference, no generation
 * step). The `numeric.ts` `UNARY_OPS`/`BINARY_OPS` scalar bodies were a
 * redundant copy of the definitions — they become `compileUnary(def)` /
 * `compileBinary(def)`, thin binds of the term to a reused env.
 *
 * The bodies compute in f64 and round once on store (the `numeric.ts`
 * discipline); Probe 3 confirms byte-for-byte reproduction of the hand bodies
 * at the f32 boundary (test/semantic-derivation.spec.ts S1).
 */

import type { Expr } from "./expr";

export interface ScalarEnv {
  x: number;
  y: number;
  g: number;
}

/** Direct f64 interpreter — the reference semantics of a term. */
export function evalScalar(e: Expr, env: ScalarEnv): number {
  switch (e.k) {
    case "x":
      return env.x;
    case "y":
      return env.y;
    case "g":
      return env.g;
    case "c":
      return e.v;
    case "neg":
      return -evalScalar(e.a, env);
    case "exp":
      return Math.exp(evalScalar(e.a, env));
    case "log":
      return Math.log(evalScalar(e.a, env));
    case "sqrt":
      return Math.sqrt(evalScalar(e.a, env));
    case "sin":
      return Math.sin(evalScalar(e.a, env));
    case "cos":
      return Math.cos(evalScalar(e.a, env));
    case "tanh":
      return Math.tanh(evalScalar(e.a, env));
    case "abs":
      return Math.abs(evalScalar(e.a, env));
    case "sign":
      return Math.sign(evalScalar(e.a, env));
    case "recip":
      return 1.0 / evalScalar(e.a, env);
    case "floor":
      return Math.floor(evalScalar(e.a, env));
    case "ceil":
      return Math.ceil(evalScalar(e.a, env));
    case "round":
      return Math.round(evalScalar(e.a, env));
    case "isfinite":
      return Number.isFinite(evalScalar(e.a, env)) ? 1.0 : 0.0;
    case "add":
      return evalScalar(e.a, env) + evalScalar(e.b, env);
    case "sub":
      return evalScalar(e.a, env) - evalScalar(e.b, env);
    case "mul":
      return evalScalar(e.a, env) * evalScalar(e.b, env);
    case "div":
      return evalScalar(e.a, env) / evalScalar(e.b, env);
    case "pow":
      return evalScalar(e.a, env) ** evalScalar(e.b, env);
    case "min":
      return Math.min(evalScalar(e.a, env), evalScalar(e.b, env));
    case "max":
      return Math.max(evalScalar(e.a, env), evalScalar(e.b, env));
    case "mod":
      return evalScalar(e.a, env) % evalScalar(e.b, env);
    case "gt":
      return evalScalar(e.a, env) > evalScalar(e.b, env) ? 1.0 : 0.0;
    case "ge":
      return evalScalar(e.a, env) >= evalScalar(e.b, env) ? 1.0 : 0.0;
    case "lt":
      return evalScalar(e.a, env) < evalScalar(e.b, env) ? 1.0 : 0.0;
    case "le":
      return evalScalar(e.a, env) <= evalScalar(e.b, env) ? 1.0 : 0.0;
    case "eq":
      return evalScalar(e.a, env) === evalScalar(e.b, env) ? 1.0 : 0.0;
    case "ne":
      return evalScalar(e.a, env) !== evalScalar(e.b, env) ? 1.0 : 0.0;
    case "where":
      return evalScalar(e.c, env) !== 0
        ? evalScalar(e.a, env)
        : evalScalar(e.b, env);
  }
}

// ----------------------------------------------------------------------------
// Derived CPU bodies (surface S1). The interpreter itself IS the reference
// (design §4.3 S1 — "the interpreter itself is the reference, no generation
// step"): a definition binds to a reused env and evaluates per element. The
// term stays the single source, guarded by `assertNoDefinitionBody`.
// ----------------------------------------------------------------------------

/** Derive a unary CPU body `(x) => number` from a term (design S1). */
export function compileUnary(def: Expr): (x: number) => number {
  const env: ScalarEnv = { x: 0, y: 0, g: 0 };
  return (xv: number) => {
    env.x = xv;
    return evalScalar(def, env);
  };
}

/** Derive a binary CPU body `(x, y) => number` from a term (design S1). */
export function compileBinary(def: Expr): (x: number, y: number) => number {
  const env: ScalarEnv = { x: 0, y: 0, g: 0 };
  return (xv: number, yv: number) => {
    env.x = xv;
    env.y = yv;
    return evalScalar(def, env);
  };
}
