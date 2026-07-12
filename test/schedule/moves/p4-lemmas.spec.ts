/**
 * P4 LEMMA gates (§7 LOCAL self-hosting — the lemma library grows by three).
 *
 * Each of the three new admitted lemmas is checked on the SAME bar the
 * online-softmax lemma set: (1) the F17 discharge round-trip — the pre-lemma body
 * REFUSES and NAMES the obligation (by ID, not text), the post-lemma body (after
 * the BoxRewrite) is ADMITTED; (2) the lemma records its LemmaApplication + carried
 * state; (3) its OWN numerical differential (the recomposition / refactor law)
 * holds to fp-exactness.
 *
 *   1. RECOMPUTATION — attention backward recomputes P from the saved logsumexp L
 *      (recompute-from-saved-statistic) instead of materializing the [S,S] matrix.
 *   2. D-PRECOMPUTE — the rowsum(dO∘O) refactor: the per-(i,j) inner sum carried
 *      once per row as the statistic D.
 *   3. WELFORD — the variance pair-merge (the teaching lemma, engine-real);
 *      layernorm's inv-std path is its consumer.
 */

import { describe, expect, it } from "vitest";
import {
  D_PRECOMPUTE_OBLIGATION,
  RECOMPUTE_P_OBLIGATION,
} from "../../../src/schedule/attention-skeleton";
import {
  applyLemma,
  dPrecomputeDifferential,
  recomputePDifferential,
  welfordDifferential,
} from "../../../src/schedule/moves/lemma";
import { classifyBody } from "../../../src/schedule/moves/streamability";
import { WELFORD_OBLIGATION } from "../../../src/schedule/reduction-skeleton";
import type {
  ObligationId,
  ScheduleState,
  SemanticBodyNode,
  SemanticSchedule,
  ValueUid,
} from "../../../src/schedule/types";

const v = (s: string): ValueUid => s as unknown as ValueUid;
const apply = (op: string, ...args: SemanticBodyNode[]): SemanticBodyNode => ({
  kind: "apply",
  catalog: { op },
  args,
});
const val = (name: string): SemanticBodyNode => ({
  kind: "value",
  value: v(name),
});

/** A one-body ScheduleState wrapper for `result = expr`. */
function stateWith(result: string, expr: SemanticBodyNode): ScheduleState {
  const semantic: SemanticSchedule = {
    blockShapes: [],
    loopNest: [],
    ordering: { kind: "flat" },
    programGridMap: { kind: "identity" },
    values: [],
    noMaterialization: [],
    stores: [],
    bodies: [{ result: v(result), expr }],
    roles: [],
    sync: [],
    atoms: [],
    lemmas: [],
  };
  return {
    semantic,
    requests: {
      warpBudget: null,
      pipeline: { kind: "none" },
      placementPreferences: [],
      cachePolicy: [],
    },
    receipts: {},
    region: "region:p4" as unknown as ScheduleState["region"],
  };
}

/** The F17 discharge round-trip on a pre-lemma body: refuse+name → apply → admit. */
function assertDischarge(
  preBody: SemanticBodyNode,
  obligation: ObligationId,
  resultName: string,
): ScheduleState {
  // Pre-lemma: refused, and NAMES the obligation (by ID — F28).
  const before = classifyBody(preBody);
  expect(before.streamable).toBe(false);
  if (before.streamable) throw new Error("unreachable");
  expect(before.refusal.dischargedBy).toBe(obligation);

  const state = stateWith(resultName, preBody);
  const outcome = applyLemma(state, v(resultName), obligation);
  expect(outcome.kind).toBe("applied");
  if (outcome.kind !== "applied") throw new Error("unreachable");

  // Post-lemma: the SAME predicate ADMITS the rewritten body.
  const after = classifyBody(outcome.state.semantic.bodies[0].expr);
  expect(after.streamable).toBe(true);
  // The LemmaApplication is recorded (F27 first-class carried state).
  expect(outcome.state.semantic.lemmas).toHaveLength(1);
  expect(outcome.state.semantic.lemmas[0].obligation).toBe(obligation);
  return outcome.state;
}

// ============================================================================
// LEMMA 1 — RECOMPUTATION identity
// ============================================================================

describe("P4 lemma — attention-backward RECOMPUTATION (P from logsumexp)", () => {
  it("discharge round-trip: materialized_P refuses+names → recompute_P admits", () => {
    const pre = apply("materialized_P", val("scores"));
    const post = assertDischarge(pre, RECOMPUTE_P_OBLIGATION, "P");
    expect(post.semantic.bodies[0].expr).toMatchObject({
      kind: "apply",
      catalog: { op: "recompute_P" },
    });
    expect(post.semantic.lemmas[0].carriedStateRef).toContain("L:logsumexp");
  });

  it("differential: exp(s − L) == forward softmax P for the saved L (fp-exact)", () => {
    // Rows with a wide dynamic range stress the m + log(ℓ) statistic.
    for (const scores of [
      [0, 1, 2, 3, 4, 5],
      [-8, -3, 0, 4, 9, 12, 30],
      [100, 100, 100, 100],
      Array.from({ length: 64 }, (_, j) => Math.sin(j * 0.7) * 8 + (j % 5)),
    ]) {
      const { maxAbsDiff } = recomputePDifferential(scores);
      expect(maxAbsDiff).toBeLessThan(1e-12);
    }
  });
});

// ============================================================================
// LEMMA 2 — D-PRECOMPUTE (rowsum(dO∘O) refactor)
// ============================================================================

describe("P4 lemma — D-PRECOMPUTE (rowsum(dO∘O) refactor)", () => {
  it("discharge round-trip: inline inner-sum refuses+names → precomputed_D admits", () => {
    const pre = apply(
      "inline_softmax_grad_innersum",
      val("P"),
      val("dO"),
      val("V"),
    );
    const post = assertDischarge(pre, D_PRECOMPUTE_OBLIGATION, "ds");
    expect(post.semantic.bodies[0].expr).toMatchObject({
      kind: "apply",
      catalog: { op: "precomputed_D" },
    });
    expect(post.semantic.lemmas[0].carriedStateRef).toContain("rowsum(dO.O)");
  });

  it("differential: rowsum(dO∘O) == Σ_k P[i,k]·(dO·V_k) (fp-exact)", () => {
    const S = 40;
    const D = 12;
    const scores = Array.from({ length: S }, (_, j) => Math.cos(j * 0.4) * 5);
    const values = Array.from({ length: S }, (_, j) =>
      Array.from({ length: D }, (_, d) => Math.sin(j * 0.2 + d) * 1.3),
    );
    const dO = Array.from({ length: D }, (_, d) => Math.cos(d * 0.9) * 0.7);
    const { absDiff } = dPrecomputeDifferential(scores, values, dO);
    // Two summation orders → tiny fp accumulation noise, not an identity break.
    expect(absDiff).toBeLessThan(1e-10);
  });
});

// ============================================================================
// LEMMA 3 — WELFORD variance pair-merge
// ============================================================================

describe("P4 lemma — WELFORD variance pair-merge (layernorm inv-std consumer)", () => {
  it("discharge round-trip: naive E[x²]−E[x]² refuses+names → welford_variance admits", () => {
    // sub(reduce_mean(mul(x,x)), mul(mean, mean))
    const x = val("x");
    const meanSq = apply("reduce_mean", apply("mul", x, x));
    const mean = apply("reduce_mean", x);
    const sqMean = apply("mul", mean, mean);
    const pre = apply("sub", meanSq, sqMean);
    const post = assertDischarge(pre, WELFORD_OBLIGATION, "var");
    expect(post.semantic.bodies[0].expr).toMatchObject({
      kind: "apply",
      catalog: { op: "welford_variance" },
    });
    expect(post.semantic.lemmas[0].carriedStateRef).toContain("M2");
  });

  it("differential: pair-merge variance == single-pass over ANY block partition", () => {
    // Values with a large offset — the naive E[x²]−E[x]² would cancel; Welford
    // (deviation-from-mean) is stable. The pair-merge must match single-pass.
    const xs = Array.from({ length: 97 }, (_, i) => 1e6 + Math.sin(i * 0.3));
    for (const bs of [1, 2, 4, 8, 16, 97]) {
      const { singlePass, welford, absDiff } = welfordDifferential(xs, bs);
      // relative agreement (var ~ O(1) atop a 1e6 offset — absolute fp scale).
      const rel = absDiff / Math.max(1e-30, Math.abs(singlePass));
      expect(rel).toBeLessThan(1e-6);
      expect(Number.isFinite(welford)).toBe(true);
    }
  });

  it("the naive E[x²]−E[x]² form catastrophically cancels (why the lemma exists)", () => {
    // Direct evidence the refusal is warranted: the naive form loses all
    // precision at a large offset, while Welford stays exact.
    const xs = Array.from({ length: 50 }, (_, i) => 1e7 + (i % 3));
    let mean = 0;
    for (const x of xs) mean += x;
    mean /= xs.length;
    let meanSq = 0;
    for (const x of xs) meanSq += x * x;
    meanSq /= xs.length;
    const naiveVar = meanSq - mean * mean; // cancellation-prone
    const { welford } = welfordDifferential(xs, 4);
    // Welford is correct; the naive form is visibly wrong (often negative).
    expect(welford).toBeGreaterThan(0);
    expect(Math.abs(naiveVar - welford)).toBeGreaterThan(1e-3);
  });
});
