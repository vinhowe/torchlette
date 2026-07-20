/**
 * Semantic Derivation — the REDUCTION monoid definitions (Crystal Campaign 3,
 * Phase 1). A reduction's *meaning* is its MONOID: an identity element and an
 * associative combiner, both first-class `Expr` DATA. From this ONE source the
 * CPU reduce body derives (the identity seeds the accumulator, the combine folds
 * each element — the strided LOOP stays a kernel, only the monoid derives,
 * design §6 P1), and the `ReductionDeclaration`'s `monoid` label is a PROJECTION
 * of the combine's root (single source — the declaration stops re-authoring it).
 *
 * The derived facts the design §4.1 reduction frame names:
 *   - `mean = reduce(sum) ∘ ÷count` — a COMPOSITION: the `sum` monoid plus a
 *     post-reduction epilogue `Expr` (`div(reduced, count)`), not a 4th monoid.
 *   - arg-reduce (argmax/argmin) is an INDEXED monoid — the VALUE monoid
 *     (`max`/`min`) combines, the winning INDEX follows; the `(bestVal,bestIndex)`
 *     state machine is the realizer's (design §4.1, reduction-skeleton S2), so the
 *     definition owns only the value combiner and the index-update DERIVES from it
 *     (`compileArgBetter`).
 *   - streamability: every monoid admits a two-stage partials+combine form; the
 *     REALIZED image of the `sum` two-stage is `planChunkedFullReduction`
 *     (ops/reductions.ts) + `computeChunkGeometry` (tile-dispatch.ts) — REFERENCED
 *     here as a derived fact, never re-owned (design §4.4 compose-don't-re-own,
 *     execution-declaration.ts's two-stage note).
 *
 * The combiners are byte-exact with the CPU reduce loops they replace: `max`/`min`
 * fold via `where(cmp(elem,acc), elem, acc)` — LEFT-BIASED on ties/NaN, exactly
 * as the hand `if (val > best)` loop was (a bare `max(x,y)` primitive would
 * diverge on ±0 ordering and NaN, so the monoid term encodes the loop's precise
 * tie policy, not the abstract lattice op). The reduce combiner is thus the
 * reduction family's own term — the elementwise `maximum` binary (which uses
 * `Math.max`) is a DIFFERENT op; the two are not forced to share a spelling.
 */

import {
  add,
  assertNoDefinitionBody,
  c,
  type Expr,
  gt,
  lt,
  where,
  x,
  y,
} from "./expr";
import { evalScalar, type ScalarEnv } from "./interpret";

/** The associative-combiner label a `ReductionDeclaration`/realizer reads. */
export type ReduceMonoidName = "sum" | "max" | "min";

/** The VJP class of a reduction. The transpose itself (broadcast/scatter back to
 *  the input shape) is P4's index-algebra machinery (`_expandGrad`); a P1
 *  definition owns only the per-monoid LOCAL factor + differentiability:
 *   - `broadcast`        : sum — each input contributed with unit weight → grad
 *     is the upstream broadcast unchanged.
 *   - `broadcast-scaled` : mean — the ÷count epilogue scales the broadcast grad by
 *     1/count.
 *   - `none`             : max/min/argmax/argmin — non-differentiable in this
 *     framework (the mask-scatter VJP is P4 index-algebra; not shipped in P1). */
export type ReduceGradKind = "broadcast" | "broadcast-scaled" | "none";

/**
 * A reduction op's meaning as DATA. `identity`/`combine`/`epilogue` are `Expr`
 * terms (proven data by `assertNoReductionDefinitionBody`); everything else is a
 * derived fact or a projection of those terms.
 */
export interface ReductionDef {
  name: "sum" | "mean" | "max" | "min" | "argmax" | "argmin";
  /** The monoid identity — the accumulator's seed value (a `Const` term). */
  identity: Expr;
  /** The associative combiner over `x`=accumulator, `y`=element. */
  combine: Expr;
  /** Post-reduction epilogue over `x`=reduced, `y`=count (mean's ÷count); or null. */
  epilogue: Expr | null;
  /** INDEXED monoid (arg-reduce): the value monoid combines, the index follows. */
  indexed: boolean;
  /** The VJP class (the transpose is P4; see `ReduceGradKind`). */
  gradKind: ReduceGradKind;
}

// ----------------------------------------------------------------------------
// The catalog — the 6 reduction ops (§2 category b: sum/max/min monoids +
// the mean composition + the two indexed arg-reduce monoids).
// ----------------------------------------------------------------------------

const NEG_INF = c(Number.NEGATIVE_INFINITY);
const POS_INF = c(Number.POSITIVE_INFINITY);

/** max fold: `elem > acc ? elem : acc` — left-biased on ties/NaN (matches the
 *  hand `if (val > best)` reduce loop; NOT `Math.max`, which diverges on ±0). */
const MAX_COMBINE: Expr = where(gt(y, x), y, x);
/** min fold: `elem < acc ? elem : acc` — left-biased on ties/NaN. */
const MIN_COMBINE: Expr = where(lt(y, x), y, x);

export const SUM_DEF: ReductionDef = {
  name: "sum",
  identity: c(0),
  combine: add(x, y),
  epilogue: null,
  indexed: false,
  gradKind: "broadcast",
};

export const MEAN_DEF: ReductionDef = {
  name: "mean",
  identity: c(0),
  combine: add(x, y),
  // mean = reduce(sum) ∘ ÷count — the composition, not a 4th monoid.
  epilogue: { k: "div", a: x, b: y },
  indexed: false,
  gradKind: "broadcast-scaled",
};

export const MAX_DEF: ReductionDef = {
  name: "max",
  identity: NEG_INF,
  combine: MAX_COMBINE,
  epilogue: null,
  indexed: false,
  gradKind: "none",
};

export const MIN_DEF: ReductionDef = {
  name: "min",
  identity: POS_INF,
  combine: MIN_COMBINE,
  epilogue: null,
  indexed: false,
  gradKind: "none",
};

export const ARGMAX_DEF: ReductionDef = {
  name: "argmax",
  identity: NEG_INF,
  combine: MAX_COMBINE, // value monoid = max; the index follows (indexed).
  epilogue: null,
  indexed: true,
  gradKind: "none",
};

export const ARGMIN_DEF: ReductionDef = {
  name: "argmin",
  identity: POS_INF,
  combine: MIN_COMBINE, // value monoid = min; the index follows (indexed).
  epilogue: null,
  indexed: true,
  gradKind: "none",
};

export const REDUCTION_DEFS: readonly ReductionDef[] = [
  SUM_DEF,
  MEAN_DEF,
  MAX_DEF,
  MIN_DEF,
  ARGMAX_DEF,
  ARGMIN_DEF,
];

export const REDUCTION_DEF_BY_NAME: ReadonlyMap<string, ReductionDef> = new Map(
  REDUCTION_DEFS.map((d) => [d.name, d] as const),
);

// ----------------------------------------------------------------------------
// Derived facts — projections of the definition terms (single source).
// ----------------------------------------------------------------------------

/**
 * PROJECT the monoid label from the combine's root op — the single source the
 * `ReductionDeclaration.monoid` field and the realizer read (design §4 P1
 * unification: the declaration stops separately authoring "sum"/"max"/"min").
 *   add(...)              → "sum"
 *   where(gt(elem,acc),…) → "max"     where(lt(elem,acc),…) → "min"
 */
export function reduceMonoidOf(def: ReductionDef): ReduceMonoidName {
  const cb = def.combine;
  if (cb.k === "add") return "sum";
  if (cb.k === "where") {
    if (cb.c.k === "gt") return "max";
    if (cb.c.k === "lt") return "min";
  }
  throw new Error(
    `reduceMonoidOf: combine root ${JSON.stringify(cb.k)} is not a recognized reduction monoid.`,
  );
}

/**
 * DERIVED FACT: whether the monoid streams (a two-stage partials+combine form
 * exists). Every associative monoid streams; the REALIZED image of the `sum`
 * two-stage is `planChunkedFullReduction` + `computeChunkGeometry` — REFERENCED,
 * never re-owned (design §4.4). The only monoid whose chunked realizer exists
 * today is `sum` (ops/reductions.ts `fullReduction` gates chunking on
 * `op === "sum"`); max/min are streamable-in-principle but their chunked realizer
 * is not built — this fact is the seam that would gate adding it.
 */
export function isStreamableMonoid(def: ReductionDef): boolean {
  // Associativity holds for every monoid here — projecting the label proves it is
  // one of the recognized associative monoids (throws otherwise). The reduction is
  // streamable as a MEANING; the realized two-stage kernel is a separate (realizer)
  // concern this fact references, not a claim every op has one built.
  const monoid = reduceMonoidOf(def);
  return monoid === "sum" || monoid === "max" || monoid === "min";
}

// ----------------------------------------------------------------------------
// The derived CPU reduce bodies (surface S1) — the identity seeds the
// accumulator, the combine folds each element. The strided LOOP stays a kernel
// (numeric.ts); these are the per-element monoid facts it reads (design §6 P1).
// ----------------------------------------------------------------------------

const REDUCE_ENV: ScalarEnv = { x: 0, y: 0, g: 0 };

/** The monoid identity as an f64 scalar — the accumulator seed. */
export function reduceIdentity(def: ReductionDef): number {
  return evalScalar(def.identity, REDUCE_ENV);
}

/** Derive the combiner `(acc, elem) => number` from the monoid term (design S1). */
export function compileReduceCombine(
  def: ReductionDef,
): (acc: number, elem: number) => number {
  const env: ScalarEnv = { x: 0, y: 0, g: 0 };
  const combine = def.combine;
  return (acc, elem) => {
    env.x = acc;
    env.y = elem;
    return evalScalar(combine, env);
  };
}

/** Derive the post-reduction epilogue `(reduced, count) => number`, or null. */
export function compileReduceEpilogue(
  def: ReductionDef,
): ((reduced: number, count: number) => number) | null {
  const epi = def.epilogue;
  if (epi === null) return null;
  const env: ScalarEnv = { x: 0, y: 0, g: 0 };
  return (reduced, count) => {
    env.x = reduced;
    env.y = count;
    return evalScalar(epi, env);
  };
}

/**
 * Derive the arg-reduce "strictly better" predicate `(elem, best) => boolean`
 * from the VALUE combine — NO separately-authored comparison. `combine(best,
 * elem)` returns `elem` iff `elem` beats `best`; strict (index keeps the FIRST
 * winner on ties, matching the hand `if (val > best)` loop) requires `elem !==
 * best`. Byte-exact with the argmax/argmin `isBetter` it replaces.
 */
export function compileArgBetter(
  def: ReductionDef,
): (elem: number, best: number) => boolean {
  const combine = compileReduceCombine(def);
  return (elem, best) => combine(best, elem) === elem && elem !== best;
}

// ----------------------------------------------------------------------------
// The schema gate — a reduction definition is DATA (the `assertNoDefinitionBody`
// analogue, extended to the monoid frame: identity/combine/epilogue are Exprs).
// ----------------------------------------------------------------------------

/**
 * Prove a `ReductionDef` is DATA: its identity/combine/epilogue are terms of the
 * closed algebra (no smuggled JS body / WGSL / buffer), exactly as
 * `assertNoDefinitionBody` proves for an elementwise definition. An adapter that
 * hid the old reduce loop behind an opaque leaf is unconstructible.
 */
export function assertNoReductionDefinitionBody(def: ReductionDef): void {
  assertNoDefinitionBody(def.identity, `${def.name}.identity`);
  assertNoDefinitionBody(def.combine, `${def.name}.combine`);
  if (def.epilogue !== null) {
    assertNoDefinitionBody(def.epilogue, `${def.name}.epilogue`);
  }
}
