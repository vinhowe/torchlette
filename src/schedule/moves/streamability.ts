/**
 * The ENGINE-side streamability predicate — the `stream` move's refusal-first
 * boundary (docs/p2-moves-design.md §2.2, schedule-state-design.md §3.1 F5/F17).
 *
 * ------------------------------------------------------------------------
 * WHAT THIS IS (and what it is NOT)
 * ------------------------------------------------------------------------
 * A `stream(value, loop)` move (types.ts §13) turns a materialized intermediate
 * into a value produced/consumed inside a loop (no global store). Its invariant
 * (F17, the FA "refusal-first" boundary): the value MUST have a declared
 * head/body decomposition over the loop's axis — a way to split the axis into
 * blocks, fold each block into a carried accumulator, and RECOMPOSE the blocks
 * into the full result by a law that holds regardless of block boundaries.
 *
 * This predicate is that check, as a PURE FUNCTION over the existing
 * `SemanticBodyNode` tree (the op-catalog DAG the family skeletons already
 * produce). It SUPERSEDES the NCD spike's client-side version
 * (examples/schedule-editor/src/lib/ncd/model.ts `streamability`), which trusts
 * free-form `head: string` / `body: string` authored strings. F5 forbids
 * evidence-shaped strings: streamability is MACHINE-CHECKED over TYPED head/body
 * terms with a recomposition law. So this reads the value's ACTUAL body root and
 * classifies its accumulator form — it does not trust an authored label.
 *
 * SCOPE (design-first, P2 STOP): this file is the PREDICATE ONLY. The `stream`
 * move-algebra body that CONSUMES it (deletes the StoreEdge, adds the
 * NoMaterializationEdge, refuses on a `false` verdict) is P2 implementation
 * (docs/p2-moves-design.md §2.1) — NOT built here. Nothing live imports this;
 * it touches no dispatch path. Exercised only by
 * test/schedule/moves/streamability.spec.ts.
 *
 * The refusal on naive softmax NAMES the proof-obligation ID the online-softmax
 * lemma discharges (F28 — jam→lemma binding is by ObligationId, NEVER by
 * matching refusal text). That obligation ID
 * (`obl:online-softmax-normalizer-equals-batched-denominator`) is the wave-3
 * engine object `ONLINE_SOFTMAX_OBLIGATION` (attention-skeleton.ts).
 */

import {
  D_PRECOMPUTE_OBLIGATION,
  ONLINE_SOFTMAX_OBLIGATION,
  RECOMPUTE_P_OBLIGATION,
} from "../attention-skeleton";
import { WELFORD_OBLIGATION } from "../reduction-skeleton";
import type {
  ObligationId,
  SemanticBody,
  SemanticBodyNode,
  SemanticSchedule,
  ValueUid,
} from "../types";

// ============================================================================
// The typed head/body decomposition (F5 — NOT strings)
// ============================================================================

/** A monoid identity / accumulator initial state (the head). Typed, not a
 *  string: the recomposition law is checkable from these members. */
export type AccumulatorForm =
  | { kind: "additive"; zero: 0 } // sum: init 0, step +, merge +
  | { kind: "maxMonoid"; identity: "-inf" } // max: init -inf, step max, merge max
  | { kind: "minMonoid"; identity: "+inf" } // min: init +inf, step min, merge min
  /** mean = sum/count: two additive accumulators recomposed then divided. */
  | { kind: "meanPair"; sumZero: 0; countZero: 0 }
  /** the online-softmax carried state (m,ℓ,o) + correction exp(m_old−m_new).
   *  Supplied by the online-softmax admitted lemma, NOT a free decomposition. */
  | {
      kind: "onlineSoftmax";
      carried: "(m,l,o)";
      correction: "exp(m_old-m_new)";
    };

/** The recomposition law that makes block ORDER irrelevant (associative +
 *  commutative merge). Named, not free-form. */
export type RecompositionLaw =
  | { op: "plus"; associative: true; commutative: true }
  | { op: "max"; associative: true; commutative: true }
  | { op: "min"; associative: true; commutative: true }
  | { op: "meanCombine"; associative: true; commutative: true }
  /** the online-softmax rescale-and-add merge (associative under the correction
   *  factor; the lemma's proof obligation is exactly its correctness). */
  | { op: "onlineSoftmaxCombine"; associative: true; commutative: false };

export interface HeadBodyDecomposition {
  readonly init: AccumulatorForm; // the head
  readonly step: AccumulatorForm; // the body (same monoid family as init)
  readonly merge: RecompositionLaw; // the recomposition law
}

export interface StreamRefusal {
  readonly reason: string;
  /**
   * The proof-obligation ID whose discharge would ADMIT this stream (F28). A
   * softmax refusal names the obligation the online-softmax lemma discharges.
   * `null` when nothing can admit it (a genuinely non-decomposable body — no
   * known lemma supplies a recomposition). NEVER refusal-text matching.
   */
  readonly dischargedBy: ObligationId | null;
}

export type StreamabilityVerdict =
  | { streamable: true; decomposition: HeadBodyDecomposition }
  | { streamable: false; refusal: StreamRefusal };

// ============================================================================
// The predicate
// ============================================================================

/**
 * Is `value` streamable in `schedule` (along the value's reduction axis)?
 *
 * Reads the value's SEMANTIC BODY root and classifies its accumulator form. This
 * is the engine object docs/p2-moves-design.md §2.2 specifies. It does NOT take
 * an explicit AxisUid in this prototype: the body's reduce-op already names the
 * axis it reduces over (the reduction skeleton's single reduction loop), so the
 * classification is over the body form. When the full move algebra lands, the
 * axis is the `stream` move's `loop.axis`; the predicate additionally asserts the
 * body's reduce axis matches (a mismatch is a refusal). That axis-match is noted
 * in §2.1 and is a one-line addition once loops carry the reduce-axis binding.
 */
export function streamability(
  schedule: SemanticSchedule,
  value: ValueUid,
): StreamabilityVerdict {
  const body = findBody(schedule, value);
  if (!body) {
    return {
      streamable: false,
      refusal: {
        reason: `value ${value} has no semantic body in this schedule; nothing to decompose.`,
        dischargedBy: null,
      },
    };
  }
  return classifyBody(body.expr);
}

/** Classify a semantic body's ROOT op into a head/body decomposition or refusal. */
export function classifyBody(expr: SemanticBodyNode): StreamabilityVerdict {
  // The online-softmax post-lemma body: a marker apply carrying the lemma's
  // carried state. The lemma rewrite replaces the div-by-full-denominator root
  // with `online_softmax(...)` (the admitted-lemma BoxRewrite, P2). When present,
  // the body IS streamable (the lemma supplies the recomposition).
  if (isApply(expr, "online_softmax")) {
    return {
      streamable: true,
      decomposition: {
        init: {
          kind: "onlineSoftmax",
          carried: "(m,l,o)",
          correction: "exp(m_old-m_new)",
        },
        step: {
          kind: "onlineSoftmax",
          carried: "(m,l,o)",
          correction: "exp(m_old-m_new)",
        },
        merge: {
          op: "onlineSoftmaxCombine",
          associative: true,
          commutative: false,
        },
      },
    };
  }

  // --- P4 post-lemma ADMITTED markers (§7 local self-hosting) ---------------
  // recompute_P: the attention backward recomputes P from the saved logsumexp L
  // instead of materializing the [S,S] matrix (recompute-from-saved-statistic).
  // The saved L is the carried statistic; the box is block-locally producible.
  if (isApply(expr, "recompute_P")) {
    return {
      streamable: true,
      decomposition: {
        init: { kind: "additive", zero: 0 },
        step: { kind: "additive", zero: 0 },
        merge: { op: "plus", associative: true, commutative: true },
      },
    };
  }
  // precomputed_D: the per-row statistic D = rowsum(dO∘O) carried into the loop
  // (the inline per-(i,j) inner sum refactored out). Block-locally consumable.
  if (isApply(expr, "precomputed_D")) {
    return {
      streamable: true,
      decomposition: {
        init: { kind: "additive", zero: 0 },
        step: { kind: "additive", zero: 0 },
        merge: { op: "plus", associative: true, commutative: true },
      },
    };
  }
  // welford_variance: the (count,mean,M2) pair-merge gives variance a stable
  // head/body decomposition (the merge is associative under the δ correction).
  if (isApply(expr, "welford_variance")) {
    return {
      streamable: true,
      decomposition: {
        init: { kind: "meanPair", sumZero: 0, countZero: 0 },
        step: { kind: "meanPair", sumZero: 0, countZero: 0 },
        merge: { op: "meanCombine", associative: true, commutative: true },
      },
    };
  }

  // A pure associative/commutative reduction: streamable by its monoid.
  const reduceKind = reduceRoot(expr);
  if (reduceKind) return { streamable: true, decomposition: reduceKind };

  // --- P4 pre-lemma REFUSED markers — refuse & name the discharging obligation.
  // materialized_P: attention backward holding the full [S,S] P matrix — refused;
  // the recomputation lemma (recompute P from the saved logsumexp L) discharges it.
  if (isApply(expr, "materialized_P")) {
    return {
      streamable: false,
      refusal: {
        reason:
          "the attention backward reads a materialized [S,S] probability matrix, an " +
          "O(S²) intermediate that cannot be produced block-locally. The recomputation " +
          "identity (recompute P from the saved per-row logsumexp L) supplies the " +
          "block-local production; it is an admitted-lemma rewrite, not a free decomposition.",
        dischargedBy: RECOMPUTE_P_OBLIGATION,
      },
    };
  }
  // inline_softmax_grad_innersum: the per-(i,j) recomputed inner sum
  // Σ_k P[i,k]·(dO_i·V_k) — refused as a block-local body; the D-precompute
  // refactor (carry D = rowsum(dO∘O) once per row) discharges it.
  if (isApply(expr, "inline_softmax_grad_innersum")) {
    return {
      streamable: false,
      refusal: {
        reason:
          "the softmax-gradient inner sum Σ_k P[i,k]·(dO_i·V_k) is recomputed per (i,j); " +
          "there is no block-local (step, merge) that avoids re-walking the whole KV axis " +
          "for each output. The D-precompute refactor (carry D = rowsum(dO∘O) once per row) " +
          "supplies it — an admitted-lemma rewrite.",
        dischargedBy: D_PRECOMPUTE_OBLIGATION,
      },
    };
  }
  // Naive variance E[x²] − E[x]²: refused (cancellation-prone, no stable block
  // recomposition — a block's contribution depends on the global mean). The
  // Welford pair-merge (count,mean,M2) discharges it.
  if (isNaiveVarianceShaped(expr)) {
    return {
      streamable: false,
      refusal: {
        reason:
          "the naive variance E[x²] − E[x]² has no numerically-sound block-local (step, merge): " +
          "a block's deviation-sum depends on the GLOBAL mean, unknown until the whole axis is " +
          "seen. The Welford pair-merge (carry (count,mean,M2), merge with the δ correction) " +
          "supplies a stable recomposition — an admitted-lemma rewrite.",
        dischargedBy: WELFORD_OBLIGATION,
      },
    };
  }

  // The naive softmax shape: div( exp(sub(x, max(...))), sum(exp(sub(x, max(...)))) ).
  // The per-element output divides by the FULL-ROW denominator, unknown until all
  // of the axis is seen — NOT block-locally recomposable. Refused; the refusal
  // names the online-softmax obligation (the lemma that supplies a recomposition).
  if (isSoftmaxShaped(expr)) {
    return {
      streamable: false,
      refusal: {
        reason:
          "softmax divides each element by the full-row denominator sum_A(exp(x - max_A(x))), " +
          "which is not known until the whole axis is seen; there is no block-local (step, merge) " +
          "that recomposes it without correcting earlier blocks. That correction IS the " +
          "online-softmax lemma (an admitted-lemma rewrite), not a free head/body decomposition.",
        dischargedBy: ONLINE_SOFTMAX_OBLIGATION,
      },
    };
  }

  // Anything else: no known head/body decomposition. Refused with no discharging
  // lemma (a genuinely materialized intermediate — e.g. a plain elementwise map
  // with no reduction to fold).
  return {
    streamable: false,
    refusal: {
      reason:
        `body root op '${rootOpName(expr)}' has no head/body decomposition over a reduction axis ` +
        `(no accumulator monoid, no admitted lemma). A materialized store is required.`,
      dischargedBy: null,
    },
  };
}

// ============================================================================
// Body-shape classifiers (read the ACTUAL op-catalog tree)
// ============================================================================

function reduceRoot(expr: SemanticBodyNode): HeadBodyDecomposition | null {
  if (expr.kind !== "apply") return null;
  switch (expr.catalog.op) {
    case "reduce_sum":
      return {
        init: { kind: "additive", zero: 0 },
        step: { kind: "additive", zero: 0 },
        merge: { op: "plus", associative: true, commutative: true },
      };
    case "reduce_max":
      return {
        init: { kind: "maxMonoid", identity: "-inf" },
        step: { kind: "maxMonoid", identity: "-inf" },
        merge: { op: "max", associative: true, commutative: true },
      };
    case "reduce_min":
      return {
        init: { kind: "minMonoid", identity: "+inf" },
        step: { kind: "minMonoid", identity: "+inf" },
        merge: { op: "min", associative: true, commutative: true },
      };
    case "reduce_mean":
      // mean = sum/count: two additive accumulators, recomposed then divided.
      // (design-first open question 3: admit directly vs require Welford; the
      // prototype admits directly — the mean itself needs only sum/count.)
      return {
        init: { kind: "meanPair", sumZero: 0, countZero: 0 },
        step: { kind: "meanPair", sumZero: 0, countZero: 0 },
        merge: { op: "meanCombine", associative: true, commutative: true },
      };
    default:
      return null;
  }
}

/**
 * Recognize the numerically-stable softmax body:
 *   div( exp(sub(x, m)), <denominator that reduces exp(sub(x, m)) over the axis> ).
 * The discriminating fact is that the DIVISOR is (or contains) a `sum`/`reduce_sum`
 * over the same exp — a full-row reduction the per-element numerator divides by.
 * We match structurally on the div root whose right arg is reduction-shaped.
 */
function isSoftmaxShaped(expr: SemanticBodyNode): boolean {
  if (!isApply(expr, "div")) return false;
  if (expr.args.length !== 2) return false;
  const denom = expr.args[1];
  return containsReduction(denom);
}

/** Does this subtree contain a full-axis reduction (`sum`/`reduce_sum`)? */
function containsReduction(node: SemanticBodyNode): boolean {
  if (node.kind !== "apply") return false;
  if (node.catalog.op === "sum" || node.catalog.op === "reduce_sum")
    return true;
  return node.args.some(containsReduction);
}

/**
 * Recognize the naive variance body:  `sub(reduce_mean(mul(x,x)), <E[x]²>)` — the
 * `E[x²] − E[x]²` form. Structural match (the discharging Welford lemma owns the
 * rewrite; this classify seam re-checks, agreeing with lemma.ts by construction).
 */
function isNaiveVarianceShaped(expr: SemanticBodyNode): boolean {
  if (!isApply(expr, "sub") || expr.args.length !== 2) return false;
  const [lhs, rhs] = expr.args;
  const isMean = (n: SemanticBodyNode): boolean =>
    isApply(n, "reduce_mean") || isApply(n, "mean");
  const lhsIsMeanSq = isMean(lhs) && lhs.kind === "apply" && isApply(lhs.args[0], "mul");
  const rhsIsSquare =
    isApply(rhs, "mul") || isApply(rhs, "square") || isMean(rhs);
  return Boolean(lhsIsMeanSq && rhsIsSquare);
}

// ============================================================================
// Small tree helpers
// ============================================================================

function isApply(
  node: SemanticBodyNode,
  op: string,
): node is Extract<SemanticBodyNode, { kind: "apply" }> {
  return node.kind === "apply" && node.catalog.op === op;
}

function rootOpName(expr: SemanticBodyNode): string {
  if (expr.kind === "apply") return expr.catalog.op;
  if (expr.kind === "value") return `value(${expr.value})`;
  return `literal(${expr.value})`;
}

function findBody(
  schedule: SemanticSchedule,
  value: ValueUid,
): SemanticBody | undefined {
  return schedule.bodies.find((b) => b.result === value);
}
