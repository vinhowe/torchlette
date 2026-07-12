/**
 * CONFORMANCE ENTRY (d) — Online-softmax streaming.
 *
 * The ladder's exercise 8 — "the lemma wall, undisguised" — formalized as its own
 * corpus entry referencing the shared online-softmax lemma. The naive softmax
 * divides each element by the FULL-ROW denominator Σ exp(x − max), unknown until
 * the whole axis is seen, so the `stream` move REFUSES it and names the online-
 * softmax obligation. The admitted online-softmax lemma (carry the running
 * (m, l, o) with the correction factor exp(m_old − m_new)) discharges it; the
 * post-lemma body streams. This is the SHARED lemma the FA forward derivation
 * (tools/fa-derivation-script.ts rung 5→7) also uses — this entry pins it as an
 * independent published-technique conformance point.
 *
 * BASE: the numerically-stable softmax body div(exp(x−m), sum(exp(x−m))).
 * SCRIPT: classifyBody REFUSES (names ONLINE_SOFTMAX_OBLIGATION) → the online-
 *       softmax BoxRewrite discharges → the rewritten body is ADMITTED.
 * OUTCOME (numeric+cost): the discharge round-trip holds AND the online recurrence
 *       equals the naive softmax·V to <1e-9 across block sizes (onlineSoftmax-
 *       Differential — the lemma's own numerical soundness gate). Cost class:
 *       O(S) materialized scores → O(1) carried state (the flash memory law).
 *
 * Cite: Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018);
 *       Rabe & Staats "Self-attention Does Not Need O(n²) Memory" (2021);
 *       Dao et al. FlashAttention (2022). The shared lemma; the repo's fused
 *       attention (attention-kernel.ts) is the composed endpoint.
 * Ladder: exercise 8 (rung 5 — the lemma wall).
 */

import { ONLINE_SOFTMAX_OBLIGATION } from "../../src/schedule/attention-skeleton";
import {
  assertDischargeRoundTrip,
  lemmaFor,
  onlineSoftmaxDifferential,
} from "../../src/schedule/moves/lemma";
import { classifyBody } from "../../src/schedule/moves/streamability";
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

/** The numerically-stable softmax: div(exp(x − max(x)), sum(exp(x − max(x)))). */
function softmaxBody(): SemanticBodyNode {
  const x = val("scores");
  const m = apply("max", x);
  const shifted = apply("exp", apply("sub", x, m));
  return apply("div", shifted, apply("sum", shifted));
}

export const module: ConformanceModule = {
  entry: {
    id: "online-softmax",
    technique:
      "Online-softmax streaming — running (m, l, o) with the exp(m_old−m_new) correction",
    citation:
      "Milakov & Gimelshein (2018); Rabe & Staats (2021); Dao et al. FlashAttention (2022) — the shared lemma; repo endpoint = fused attention",
    baseState: "stable softmax body div(exp(x−max), sum(exp(x−max)))",
    moveScript:
      "classifyBody REFUSES (names ONLINE_SOFTMAX_OBLIGATION) → online-softmax BoxRewrite discharges → ADMITTED",
    outcomeKind: "numeric+cost",
    outcome:
      "the F17 discharge round-trip holds AND the online recurrence == naive softmax·V to <1e-9 across block sizes; cost class: O(S) materialized scores → O(1) carried state (the flash memory law)",
    ladderRef: "exercise 8 (rung 5 — the lemma wall)",
  },

  run(ctx): void {
    const naive = softmaxBody();

    // THE REFUSAL — softmax divides by the full-row denominator (not block-local).
    const before = classifyBody(naive);
    ctx.oracle(
      !before.streamable,
      "stream(naive softmax) is REFUSED (division by the full-row denominator, unknown until the whole axis is seen — the F17 wall)",
    );
    if (!before.streamable) {
      ctx.oracle(
        before.refusal.dischargedBy === ONLINE_SOFTMAX_OBLIGATION,
        "the refusal NAMES the ONLINE_SOFTMAX obligation (F28 — by obligation ID)",
      );
    }

    // THE LEMMA — the shared online-softmax lemma discharges it.
    const lemma = lemmaFor(ONLINE_SOFTMAX_OBLIGATION);
    ctx.oracle(
      lemma !== undefined && lemma.obligation === ONLINE_SOFTMAX_OBLIGATION,
      "the LEMMA_LIBRARY supplies a lemma discharging ONLINE_SOFTMAX_OBLIGATION (the FA-shared lemma)",
    );

    // THE DISCHARGE ROUND-TRIP — pre REFUSED + names the obligation, post ADMITTED.
    const rt = assertDischargeRoundTrip(naive);
    ctx.oracle(
      rt.obligationNamed === ONLINE_SOFTMAX_OBLIGATION,
      "assertDischargeRoundTrip: the pre-lemma body names ONLINE_SOFTMAX_OBLIGATION",
    );
    ctx.oracle(
      rt.postAdmitted,
      "post-lemma body (online_softmax) is ADMITTED — streaming with the (m, l, o) carried state",
    );

    // THE NUMERIC GATE — the lemma's OWN soundness differential: the online
    // recurrence (with the exp(m_old−m_new) correction) equals the naive
    // softmax·V over ANY block partition. This is the recomposition law cashed.
    const S = 64;
    const D = 32;
    const scores = Array.from(
      { length: S },
      (_, j) => Math.sin(j * 0.7) * 5 + (j % 4),
    );
    const values = Array.from({ length: S }, (_, j) =>
      Array.from({ length: D }, (_, d) => Math.cos(j * 0.2 + d) * 1.2),
    );
    let worst = 0;
    for (const bs of [1, 2, 4, 8, 16, 32, S]) {
      worst = Math.max(
        worst,
        onlineSoftmaxDifferential(scores, values, bs).maxAbsDiff,
      );
    }
    ctx.note(
      `online-softmax·V vs naive softmax·V, worst |Δ| over block sizes {1,2,4,8,16,32,${S}} = ${worst.toExponential(3)}`,
    );
    ctx.oracle(
      worst < 1e-9,
      `online-softmax differential ≤ 1e-9 (the lemma's numerical soundness; got ${worst.toExponential(3)})`,
    );

    // THE COST CLASS — the flash MEMORY law: streaming softmax replaces the O(S)
    // materialized score/probability row with an O(1) carried statistic, so
    // attention memory stops scaling with the sequence length. This is the exact
    // discontinuity (rung 5) the FA forward derivation crosses via this lemma.
    ctx.note(
      "COST CLASS: memory ∝ problem size → O(1) carried state. Streaming softmax carries only " +
        "(running max m, running sum l, running output o) instead of the O(S) materialized scores — " +
        "the flash memory law; the repo's fused attention composes this lemma with tiling (rungs 3+5+6+7).",
    );
  },
};

if (
  process.argv[1] &&
  (process.argv[1].endsWith("online-softmax.ts") ||
    process.argv[1].endsWith("online-softmax.js"))
) {
  void runEntry(module);
}
