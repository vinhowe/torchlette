/**
 * The engine-side streamability predicate — unit gate (P2 design-first prototype).
 *
 * Schema-level only: builds tiny SemanticSchedules with the reduction / softmax /
 * post-lemma body forms and asserts the predicate's verdict. No GPU, no dispatch,
 * touches nothing live (imports only src/schedule/moves/streamability + types).
 *
 * The three required cases (docs/p2-moves-design.md §2.4):
 *   - softmax body                 → NOT streamable, names ONLINE_SOFTMAX_OBLIGATION
 *   - sum / mean bodies            → streamable
 *   - post-lemma online-softmax    → streamable
 * plus the max/min monoids and the no-decomposition refusal for coverage.
 */

import { describe, expect, it } from "vitest";
import { ONLINE_SOFTMAX_OBLIGATION } from "../../../src/schedule/attention-skeleton";
import {
  classifyBody,
  streamability,
} from "../../../src/schedule/moves/streamability";
import type {
  SemanticBody,
  SemanticBodyNode,
  SemanticSchedule,
  ValueUid,
} from "../../../src/schedule/types";

const v = (s: string): ValueUid => s as unknown as ValueUid;

/** A minimal SemanticSchedule carrying just the one body under test. */
function scheduleWithBody(
  result: ValueUid,
  expr: SemanticBodyNode,
): SemanticSchedule {
  const body: SemanticBody = { result, expr };
  return {
    blockShapes: [],
    loopNest: [],
    ordering: { kind: "flat" },
    programGridMap: { kind: "identity" },
    values: [],
    noMaterialization: [],
    stores: [],
    bodies: [body],
    roles: [],
    sync: [],
    atoms: [],
    lemmas: [],
  };
}

const val = (name: string): SemanticBodyNode => ({
  kind: "value",
  value: v(name),
});
const apply = (op: string, ...args: SemanticBodyNode[]): SemanticBodyNode => ({
  kind: "apply",
  catalog: { op },
  args,
});

// The naive numerically-stable softmax body:
//   div( exp(sub(x, max(x))), sum(exp(sub(x, max(x)))) )
function softmaxBody(): SemanticBodyNode {
  const x = val("scores");
  const m = apply("max", x);
  const shifted = apply("exp", apply("sub", x, m));
  const denom = apply("sum", shifted);
  return apply("div", shifted, denom);
}

describe("streamability predicate (engine object)", () => {
  it("sum body is streamable (additive monoid)", () => {
    const s = scheduleWithBody(v("r"), apply("reduce_sum", val("x")));
    const verdict = streamability(s, v("r"));
    expect(verdict.streamable).toBe(true);
    if (verdict.streamable) {
      expect(verdict.decomposition.init.kind).toBe("additive");
      expect(verdict.decomposition.merge.op).toBe("plus");
      expect(verdict.decomposition.merge.associative).toBe(true);
      expect(verdict.decomposition.merge.commutative).toBe(true);
    }
  });

  it("mean body is streamable (sum/count pair)", () => {
    const s = scheduleWithBody(v("r"), apply("reduce_mean", val("x")));
    const verdict = streamability(s, v("r"));
    expect(verdict.streamable).toBe(true);
    if (verdict.streamable) {
      expect(verdict.decomposition.init.kind).toBe("meanPair");
      expect(verdict.decomposition.merge.op).toBe("meanCombine");
    }
  });

  it("max and min bodies are streamable (max/min monoids)", () => {
    const smax = scheduleWithBody(v("r"), apply("reduce_max", val("x")));
    const vmax = streamability(smax, v("r"));
    expect(vmax.streamable).toBe(true);
    if (vmax.streamable) expect(vmax.decomposition.merge.op).toBe("max");

    const smin = scheduleWithBody(v("r"), apply("reduce_min", val("x")));
    const vmin = streamability(smin, v("r"));
    expect(vmin.streamable).toBe(true);
    if (vmin.streamable) expect(vmin.decomposition.merge.op).toBe("min");
  });

  it("softmax body is NOT streamable and names the online-softmax obligation (F28)", () => {
    const s = scheduleWithBody(v("P"), softmaxBody());
    const verdict = streamability(s, v("P"));
    expect(verdict.streamable).toBe(false);
    if (!verdict.streamable) {
      // The refusal binds to the proof-obligation ID, NEVER refusal text (F28).
      expect(verdict.refusal.dischargedBy).toBe(ONLINE_SOFTMAX_OBLIGATION);
      // And it is the wave-3 engine object's obligation, not an ad-hoc string.
      expect(verdict.refusal.dischargedBy).toBe(
        "obl:online-softmax-normalizer-equals-batched-denominator",
      );
    }
  });

  it("post-lemma online-softmax body IS streamable (the lemma supplies the recomposition)", () => {
    // The online-softmax lemma's BoxRewrite (P2) replaces the div-by-full-denominator
    // root with `online_softmax(...)` carrying the (m,l,o) recurrence. Once present,
    // the SAME predicate admits it — refusal-first → discharge → re-admit (F17).
    const s = scheduleWithBody(v("P"), apply("online_softmax", val("scores")));
    const verdict = streamability(s, v("P"));
    expect(verdict.streamable).toBe(true);
    if (verdict.streamable) {
      expect(verdict.decomposition.init.kind).toBe("onlineSoftmax");
      expect(verdict.decomposition.merge.op).toBe("onlineSoftmaxCombine");
    }
  });

  it("the discharge round-trip: softmax refuses → obligation → post-lemma admits", () => {
    // This is the FA F17 sequence read as a predicate round-trip: the refusal
    // names the obligation, and the body the matching lemma produces is admitted.
    const naive = classifyBody(softmaxBody());
    expect(naive.streamable).toBe(false);
    const obligation = naive.streamable ? null : naive.refusal.dischargedBy;
    expect(obligation).toBe(ONLINE_SOFTMAX_OBLIGATION);

    const postLemma = classifyBody(apply("online_softmax", val("scores")));
    expect(postLemma.streamable).toBe(true);
  });

  it("a plain elementwise map (no reduction) is refused with no discharging lemma", () => {
    // add(x, y) has no reduction axis to fold — a genuine materialized store.
    const s = scheduleWithBody(v("r"), apply("add", val("x"), val("y")));
    const verdict = streamability(s, v("r"));
    expect(verdict.streamable).toBe(false);
    if (!verdict.streamable) expect(verdict.refusal.dischargedBy).toBeNull();
  });

  it("a missing body is refused (nothing to decompose)", () => {
    const s = scheduleWithBody(v("other"), apply("reduce_sum", val("x")));
    const verdict = streamability(s, v("absent"));
    expect(verdict.streamable).toBe(false);
    if (!verdict.streamable) expect(verdict.refusal.dischargedBy).toBeNull();
  });
});
