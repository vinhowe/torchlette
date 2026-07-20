/**
 * Semantic Derivation — the definition term (Crystal Campaign 3, Phase 0).
 *
 * An op's *meaning* (the formula it computes) as a first-class DATA object: a
 * term in a closed primitive algebra. This is the single source from which the
 * CPU reference body (interpret), the gradient (adjoint), and the WGSL emitter
 * all derive — so the four hand-written copies of one meaning stop existing.
 *
 * See docs/semantic-derivation-design.md §4.1. The schema is DATA: no leaf is a
 * JS function, a WGSL string, or a buffer — proven structurally by
 * `assertNoDefinitionBody` (the `assertNoGeneratorLeaf` analogue for meaning),
 * so a wrapped-generator adapter that hides the old hand body is UNCONSTRUCTIBLE
 * and the byte differential cannot be gamed (the R22 defense).
 */

// ----------------------------------------------------------------------------
// The Expr schema — a term over the closed primitive algebra.
// ----------------------------------------------------------------------------

/** Unary scalar primitives. Irreducible transcendental/rounding operations. */
export type Prim1Op =
  | "neg"
  | "exp"
  | "log"
  | "sqrt"
  | "sin"
  | "cos"
  | "tanh"
  | "abs"
  | "sign"
  | "recip"
  | "floor"
  | "ceil"
  | "round"
  | "isfinite";

/** Binary scalar primitives. */
export type Prim2Op =
  | "add"
  | "sub"
  | "mul"
  | "div"
  | "pow"
  | "min"
  | "max"
  | "mod";

/** Comparison primitives — evaluate to {0.0, 1.0}. */
export type CompareOp = "gt" | "ge" | "lt" | "le" | "eq" | "ne";

export type Expr =
  | { k: "x" } // primary input (unary input, or binary left operand a)
  | { k: "y" } // second input (binary right operand b)
  | { k: "g" } // upstream gradient (used only in gradient expressions)
  | { k: "c"; v: number } // constant
  | { k: Prim1Op; a: Expr }
  | { k: Prim2Op; a: Expr; b: Expr }
  | { k: CompareOp; a: Expr; b: Expr }
  | { k: "where"; c: Expr; a: Expr; b: Expr };

// ----------------------------------------------------------------------------
// Builder helpers — the term constructors (data, never closures).
// ----------------------------------------------------------------------------

export const x: Expr = { k: "x" };
export const y: Expr = { k: "y" };
export const g: Expr = { k: "g" };
export const c = (v: number): Expr => ({ k: "c", v });

export const neg = (a: Expr): Expr => ({ k: "neg", a });
export const exp = (a: Expr): Expr => ({ k: "exp", a });
export const log = (a: Expr): Expr => ({ k: "log", a });
export const sqrt = (a: Expr): Expr => ({ k: "sqrt", a });
export const sin = (a: Expr): Expr => ({ k: "sin", a });
export const cos = (a: Expr): Expr => ({ k: "cos", a });
export const tanh = (a: Expr): Expr => ({ k: "tanh", a });
export const abs = (a: Expr): Expr => ({ k: "abs", a });
export const sign = (a: Expr): Expr => ({ k: "sign", a });
export const recip = (a: Expr): Expr => ({ k: "recip", a });
export const floor = (a: Expr): Expr => ({ k: "floor", a });
export const ceil = (a: Expr): Expr => ({ k: "ceil", a });
export const round = (a: Expr): Expr => ({ k: "round", a });
export const isfinite = (a: Expr): Expr => ({ k: "isfinite", a });

export const add = (a: Expr, b: Expr): Expr => ({ k: "add", a, b });
export const sub = (a: Expr, b: Expr): Expr => ({ k: "sub", a, b });
export const mul = (a: Expr, b: Expr): Expr => ({ k: "mul", a, b });
export const div = (a: Expr, b: Expr): Expr => ({ k: "div", a, b });
export const powE = (a: Expr, b: Expr): Expr => ({ k: "pow", a, b });
export const minE = (a: Expr, b: Expr): Expr => ({ k: "min", a, b });
export const maxE = (a: Expr, b: Expr): Expr => ({ k: "max", a, b });
export const modE = (a: Expr, b: Expr): Expr => ({ k: "mod", a, b });

export const gt = (a: Expr, b: Expr): Expr => ({ k: "gt", a, b });
export const ge = (a: Expr, b: Expr): Expr => ({ k: "ge", a, b });
export const lt = (a: Expr, b: Expr): Expr => ({ k: "lt", a, b });
export const le = (a: Expr, b: Expr): Expr => ({ k: "le", a, b });
export const eq = (a: Expr, b: Expr): Expr => ({ k: "eq", a, b });
export const ne = (a: Expr, b: Expr): Expr => ({ k: "ne", a, b });
export const where = (cnd: Expr, a: Expr, b: Expr): Expr => ({
  k: "where",
  c: cnd,
  a,
  b,
});

// ----------------------------------------------------------------------------
// The structural schema gate — a definition is DATA (copied from
// execution-declaration.ts `assertNoGeneratorLeaf`, promoted to meaning).
// ----------------------------------------------------------------------------

const EXPR_KINDS = new Set<string>([
  "x",
  "y",
  "g",
  "c",
  "neg",
  "exp",
  "log",
  "sqrt",
  "sin",
  "cos",
  "tanh",
  "abs",
  "sign",
  "recip",
  "floor",
  "ceil",
  "round",
  "isfinite",
  "add",
  "sub",
  "mul",
  "div",
  "pow",
  "min",
  "max",
  "mod",
  "gt",
  "ge",
  "lt",
  "le",
  "eq",
  "ne",
  "where",
]);

/**
 * Prove a definition term is DATA: every leaf is an operand role, a constant, or
 * a primitive node — never a JS function, a WGSL/text string, or a buffer. An
 * adapter that smuggles the old `sigmoid` lambda behind an opaque leaf is
 * unconstructible, so the byte differential in the productized probe cannot be
 * gamed by a wrapped generator (the covenant/R22 defense, design §4.1).
 */
export function assertNoDefinitionBody(e: Expr, path = "def"): void {
  const walk = (value: unknown, p: string): void => {
    if (value === null || value === undefined) {
      throw new Error(
        `SemanticDefinition schema gate: ${p} is null/undefined — a term has no empty slots.`,
      );
    }
    const t = typeof value;
    if (t === "function") {
      throw new Error(
        `SemanticDefinition schema gate: ${p} is a function — the definition must be DATA (no embedded op body / callback).`,
      );
    }
    if (t === "string" || t === "number" || t === "boolean") return;
    if (t === "object") {
      if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
        throw new Error(
          `SemanticDefinition schema gate: ${p} is a buffer leaf — a term names primitives, never buffers.`,
        );
      }
      const kind = (value as { k?: unknown }).k;
      if (typeof kind !== "string" || !EXPR_KINDS.has(kind)) {
        throw new Error(
          `SemanticDefinition schema gate: ${p} has kind ${JSON.stringify(kind)} which is not a primitive of the algebra.`,
        );
      }
      for (const [key, v] of Object.entries(value as Record<string, unknown>)) {
        if (key === "k") continue;
        walk(v, `${p}.${key}`);
      }
      return;
    }
    throw new Error(
      `SemanticDefinition schema gate: ${p} has non-data type ${t}.`,
    );
  };
  walk(e, path);
}
