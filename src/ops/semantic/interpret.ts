/**
 * Semantic Derivation — the CPU reference interpreter (surface S1).
 *
 * Interpret an `Expr` as an f64 scalar function; this IS the CPU reference body
 * (design §4.3 S1). The `numeric.ts` `UNARY_OPS`/`BINARY_OPS` scalar bodies are a
 * redundant copy of the definitions — they become `compileUnary(def)` /
 * `compileBinary(def)`, which pre-walk the term into a nested closure ONCE (no
 * per-element tree walk) so the derived body carries no runtime overhead.
 *
 * The bodies compute in f64 and round once on store (the `numeric.ts`
 * discipline); Probe 3 confirms 19/19 byte-for-byte reproduction of the hand
 * bodies at the f32 boundary.
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
// Ahead-of-time closure compilation — the derived body with no tree-walk cost.
// The closure is DERIVED from the term (it is not smuggled back in — the term
// stays the single source and `assertNoDefinitionBody` guards it).
// ----------------------------------------------------------------------------

type Thunk = (env: ScalarEnv) => number;

function compile(e: Expr): Thunk {
  switch (e.k) {
    case "x":
      return (env) => env.x;
    case "y":
      return (env) => env.y;
    case "g":
      return (env) => env.g;
    case "c": {
      const v = e.v;
      return () => v;
    }
    case "neg": {
      const a = compile(e.a);
      return (env) => -a(env);
    }
    case "exp": {
      const a = compile(e.a);
      return (env) => Math.exp(a(env));
    }
    case "log": {
      const a = compile(e.a);
      return (env) => Math.log(a(env));
    }
    case "sqrt": {
      const a = compile(e.a);
      return (env) => Math.sqrt(a(env));
    }
    case "sin": {
      const a = compile(e.a);
      return (env) => Math.sin(a(env));
    }
    case "cos": {
      const a = compile(e.a);
      return (env) => Math.cos(a(env));
    }
    case "tanh": {
      const a = compile(e.a);
      return (env) => Math.tanh(a(env));
    }
    case "abs": {
      const a = compile(e.a);
      return (env) => Math.abs(a(env));
    }
    case "sign": {
      const a = compile(e.a);
      return (env) => Math.sign(a(env));
    }
    case "recip": {
      const a = compile(e.a);
      return (env) => 1.0 / a(env);
    }
    case "floor": {
      const a = compile(e.a);
      return (env) => Math.floor(a(env));
    }
    case "ceil": {
      const a = compile(e.a);
      return (env) => Math.ceil(a(env));
    }
    case "round": {
      const a = compile(e.a);
      return (env) => Math.round(a(env));
    }
    case "isfinite": {
      const a = compile(e.a);
      return (env) => (Number.isFinite(a(env)) ? 1.0 : 0.0);
    }
    case "add": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => a(env) + b(env);
    }
    case "sub": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => a(env) - b(env);
    }
    case "mul": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => a(env) * b(env);
    }
    case "div": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => a(env) / b(env);
    }
    case "pow": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => a(env) ** b(env);
    }
    case "min": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => Math.min(a(env), b(env));
    }
    case "max": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => Math.max(a(env), b(env));
    }
    case "mod": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => a(env) % b(env);
    }
    case "gt": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => (a(env) > b(env) ? 1.0 : 0.0);
    }
    case "ge": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => (a(env) >= b(env) ? 1.0 : 0.0);
    }
    case "lt": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => (a(env) < b(env) ? 1.0 : 0.0);
    }
    case "le": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => (a(env) <= b(env) ? 1.0 : 0.0);
    }
    case "eq": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => (a(env) === b(env) ? 1.0 : 0.0);
    }
    case "ne": {
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => (a(env) !== b(env) ? 1.0 : 0.0);
    }
    case "where": {
      const cnd = compile(e.c);
      const a = compile(e.a);
      const b = compile(e.b);
      return (env) => (cnd(env) !== 0 ? a(env) : b(env));
    }
  }
}

/** Derive a unary CPU body `(x) => number` from a term (design S1). */
export function compileUnary(def: Expr): (x: number) => number {
  const thunk = compile(def);
  const env: ScalarEnv = { x: 0, y: 0, g: 0 };
  return (xv: number) => {
    env.x = xv;
    return thunk(env);
  };
}

/** Derive a binary CPU body `(x, y) => number` from a term (design S1). */
export function compileBinary(def: Expr): (x: number, y: number) => number {
  const thunk = compile(def);
  const env: ScalarEnv = { x: 0, y: 0, g: 0 };
  return (xv: number, yv: number) => {
    env.x = xv;
    env.y = yv;
    return thunk(env);
  };
}
