/**
 * CONFORMANCE ENTRY (c) — One-pass LayerNorm via Welford.
 *
 * The ladder's exercise 3 endpoint. LayerNorm's variance is naively two passes
 * over the row: E[x²] − E[x]² (or a mean pass then a deviation pass). That naive
 * form has NO numerically-sound block-local recomposition — a block's deviation
 * sum depends on the GLOBAL mean, unknown until the whole row is seen — so the
 * `stream` move REFUSES it and NAMES the Welford proof-obligation. Importing the
 * admitted Welford lemma (carry the (count, mean, M2) pair-merge triple, merge
 * with the δ correction) discharges the obligation; the post-lemma body streams
 * in one pass. The lemma is engine-real (from P4, reduction-skeleton.ts).
 *
 * BASE: the naive-variance body E[x²] − E[x]² (the pre-fused LayerNorm variance
 *       shape isNaiveVarianceShaped recognizes).
 * SCRIPT: classifyBody REFUSES (names WELFORD_OBLIGATION) → the Welford
 *       BoxRewrite discharges → the rewritten welford_variance body is ADMITTED.
 * OUTCOME (numeric+cost): the discharge round-trip holds AND the Welford
 *       pair-merge equals the single-pass variance to <1e-10 across block sizes
 *       (welfordDifferential — the lemma's own numerical soundness gate). Cost
 *       class: two memory passes → one (a traffic halving on the reduction axis).
 *
 * Cite: Welford, "Note on a method for calculating corrected sums of squares and
 *       products" (Technometrics 1962); Chan/Golub/LeVeque parallel pair-merge
 *       (the (count,mean,M2) combine). The online/fused-LayerNorm endpoint is the
 *       repo's shipped fused row program.
 * Ladder: exercise 3 (rung 5 streaming — the first small lemma import).
 */

import {
  assertDischargeRoundTrip,
  lemmaFor,
  welfordDifferential,
} from "../../src/schedule/moves/lemma";
import { classifyBody } from "../../src/schedule/moves/streamability";
import {
  WELFORD_OBLIGATION,
  welfordLemma,
} from "../../src/schedule/reduction-skeleton";
import type { SemanticBodyNode, ValueUid } from "../../src/schedule/types";
import { type ConformanceModule, runEntry } from "./harness";

const apply = (op: string, ...args: SemanticBodyNode[]): SemanticBodyNode => ({
  kind: "apply",
  catalog: { op },
  args,
});
const val = (name: string): SemanticBodyNode => ({
  kind: "value",
  value: name as unknown as ValueUid,
});

/**
 * The naive variance body E[x²] − E[x]²:
 *   sub( reduce_mean(mul(x, x)),  mul(reduce_mean(x), reduce_mean(x)) )
 * This is the two-pass shape isNaiveVarianceShaped recognizes (the LayerNorm
 * variance the fused row program replaces).
 */
function naiveVarianceBody(): SemanticBodyNode {
  const x = val("x");
  const eSq = apply("reduce_mean", apply("mul", x, x)); // E[x²]
  const mean = apply("reduce_mean", x); // E[x]
  const meanSq = apply("mul", mean, mean); // E[x]²
  return apply("sub", eSq, meanSq);
}

export const module: ConformanceModule = {
  entry: {
    id: "layernorm-welford",
    technique:
      "One-pass LayerNorm variance via the Welford (count,mean,M2) pair-merge",
    citation:
      "Welford (Technometrics 1962); Chan/Golub/LeVeque parallel variance pair-merge — endpoint is the repo's fused LayerNorm row program",
    baseState:
      "naive variance body E[x²] − E[x]² (the pre-fused two-pass shape)",
    moveScript:
      "classifyBody REFUSES (names WELFORD_OBLIGATION) → Welford BoxRewrite discharges → welford_variance ADMITTED",
    outcomeKind: "numeric+cost",
    outcome:
      "the F17 discharge round-trip holds (naive REFUSED→ names Welford → post-lemma streamable) AND welford pair-merge == single-pass variance to <1e-10 across block sizes; cost class: two memory passes → one",
    ladderRef: "exercise 3 (rung 5 streaming — the first small lemma import)",
  },

  run(ctx): void {
    const naive = naiveVarianceBody();

    // THE REFUSAL — the naive variance body has no block-local (step, merge).
    const before = classifyBody(naive);
    ctx.oracle(
      !before.streamable,
      "stream(naive variance E[x²]−E[x]²) is REFUSED (a block's deviation sum needs the GLOBAL mean — the F17 boundary)",
    );
    if (!before.streamable) {
      ctx.oracle(
        before.refusal.dischargedBy === WELFORD_OBLIGATION,
        "the refusal NAMES the WELFORD obligation (F28 — by obligation ID, not text)",
      );
    }

    // THE LEMMA — look it up by the named obligation (the F28 seam) and discharge.
    const lemma = lemmaFor(WELFORD_OBLIGATION);
    ctx.oracle(
      lemma !== undefined && lemma.obligation === WELFORD_OBLIGATION,
      "the LEMMA_LIBRARY supplies a lemma discharging WELFORD_OBLIGATION",
    );

    // THE DISCHARGE ROUND-TRIP — pre REFUSED + names Welford, post ADMITTED.
    const rt = assertDischargeRoundTrip(naive);
    ctx.oracle(
      rt.obligationNamed === WELFORD_OBLIGATION,
      "assertDischargeRoundTrip: the pre-lemma body names WELFORD_OBLIGATION",
    );
    ctx.oracle(
      rt.postAdmitted,
      "post-lemma body (welford_variance) is ADMITTED — one-pass streaming (the carried (count,mean,M2))",
    );

    // The applied lemma records its carried state (the pair-merge triple).
    const application = welfordLemma();
    ctx.oracle(
      application.obligation === WELFORD_OBLIGATION &&
        application.carriedStateRef.includes("count") &&
        application.carriedStateRef.includes("M2"),
      "the Welford LemmaApplication carries the (count, mean, M2) pair-merge triple (the online variance state)",
    );

    // THE NUMERIC GATE — the lemma's OWN soundness differential: the pair-merge
    // over ANY block partition equals the single-pass variance. This is the
    // proof-obligation cashed numerically (the fused LayerNorm's rstd consumes it).
    let worst = 0;
    const rows: number[][] = [
      [0, 1, 2, 3, 4, 5, 6, 7],
      [-8, -3, 0, 4, 9, 12, 30, -1, 2],
      Array.from({ length: 128 }, (_, j) => Math.sin(j * 0.7) * 8 + (j % 5)),
      Array.from({ length: 96 }, (_, j) => 1e4 + Math.cos(j * 0.3) * 3), // large-mean cancellation stress
    ];
    for (const xs of rows) {
      for (const bs of [1, 2, 4, 8, 16, xs.length]) {
        worst = Math.max(worst, welfordDifferential(xs, bs).absDiff);
      }
    }
    ctx.note(
      `welford pair-merge vs single-pass variance, worst |Δ| over rows × block sizes = ${worst.toExponential(3)}`,
    );
    ctx.oracle(
      worst < 1e-10,
      `Welford differential ≤ 1e-10 (the lemma's numerical soundness; got ${worst.toExponential(3)})`,
    );

    // THE COST CLASS: the naive two-pass variance reads the row TWICE (mean pass +
    // deviation pass, or E[x²]/E[x] passes); Welford folds both into one pass —
    // a traffic halving on the reduction axis (rung-0 bytes intuition applied at
    // the streaming rung).
    ctx.note(
      "COST CLASS: memory traffic on the reduction axis — two passes → one. Welford's " +
        "single-pass (count,mean,M2) fold reads each row element once vs the naive two passes; " +
        "the fused LayerNorm row program is the shipped endpoint that consumes rstd=1/√(var+eps).",
    );
  },
};

if (
  process.argv[1] &&
  (process.argv[1].endsWith("layernorm-welford.ts") ||
    process.argv[1].endsWith("layernorm-welford.js"))
) {
  void runEntry(module);
}
