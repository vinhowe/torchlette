/**
 * LEMMA APPLICATION unit gate (P2 wave B deliverable 2).
 *
 * The online-softmax admitted lemma rewrites the softmax body into the
 * streamable online form (discharging the obligation the stream refusal names),
 * AND its own differential (rewritten region == original numerically) holds.
 */

import { describe, expect, it } from "vitest";
import { ONLINE_SOFTMAX_OBLIGATION } from "../../../src/schedule/attention-skeleton";
import {
  applyLemma,
  assertDischargeRoundTrip,
  onlineSoftmaxDifferential,
} from "../../../src/schedule/moves/lemma";
import { classifyBody } from "../../../src/schedule/moves/streamability";
import type {
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

/** The naive numerically-stable softmax body. */
function softmaxBody(): SemanticBodyNode {
  const x = val("scores");
  const m = apply("max", x);
  const shifted = apply("exp", apply("sub", x, m));
  return apply("div", shifted, apply("sum", shifted));
}

function softmaxState(): ScheduleState {
  const semantic: SemanticSchedule = {
    blockShapes: [],
    loopNest: [],
    ordering: { kind: "flat" },
    programGridMap: { kind: "identity" },
    values: [],
    noMaterialization: [],
    stores: [],
    bodies: [{ result: v("P"), expr: softmaxBody() }],
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
    region: "region:softmax" as unknown as ScheduleState["region"],
  };
}

describe("lemma application — online softmax", () => {
  it("rewrites the softmax body to the streamable online form + records the lemma", () => {
    const state = softmaxState();
    // pre-lemma: the body refuses streaming (F17 boundary).
    expect(classifyBody(state.semantic.bodies[0].expr).streamable).toBe(false);

    const outcome = applyLemma(state, v("P"), ONLINE_SOFTMAX_OBLIGATION);
    expect(outcome.kind).toBe("applied");
    if (outcome.kind !== "applied") throw new Error("unreachable");

    // post-lemma: the SAME predicate admits the rewritten body.
    expect(classifyBody(outcome.state.semantic.bodies[0].expr).streamable).toBe(
      true,
    );
    expect(outcome.state.semantic.bodies[0].expr).toMatchObject({
      kind: "apply",
      catalog: { op: "online_softmax" },
    });
    // The LemmaApplication is recorded (first-class carried state, F27).
    expect(outcome.state.semantic.lemmas).toHaveLength(1);
    expect(outcome.state.semantic.lemmas[0].obligation).toBe(
      ONLINE_SOFTMAX_OBLIGATION,
    );
    expect(outcome.application.carriedStateRef).toContain("m:running-max");
  });

  it("refuses where the pattern does not match (obligation does not arise)", () => {
    const s = softmaxState();
    const plain: ScheduleState = {
      ...s,
      semantic: {
        ...s.semantic,
        bodies: [{ result: v("P"), expr: apply("add", val("x"), val("y")) }],
      },
    };
    const outcome = applyLemma(plain, v("P"), ONLINE_SOFTMAX_OBLIGATION);
    expect(outcome.kind).toBe("refused");
  });

  it("the discharge round-trip: refuses → names obligation → post-lemma admits (F17)", () => {
    const { obligationNamed, postAdmitted } = assertDischargeRoundTrip(
      softmaxBody(),
    );
    expect(obligationNamed).toBe(ONLINE_SOFTMAX_OBLIGATION);
    expect(postAdmitted).toBe(true);
  });
});

describe("lemma differential — online softmax == naive (the lemma's own gate)", () => {
  it("the online recurrence equals the batched softmax·V to the tolerance envelope", () => {
    // A row of scores with a wide dynamic range (stresses the exp rescale) + V.
    const S = 37;
    const D = 8;
    const scores: number[] = [];
    for (let j = 0; j < S; j++)
      scores.push(Math.sin(j * 0.9) * 6 + (j % 5) * 2);
    const values: number[][] = [];
    for (let j = 0; j < S; j++) {
      const row: number[] = [];
      for (let d = 0; d < D; d++) row.push(Math.cos(j * 0.3 + d) * 1.5);
      values.push(row);
    }
    // Multiple block sizes — the recomposition law must hold for ANY partition.
    for (const bs of [1, 2, 4, 8, 16, S]) {
      const { maxAbsDiff } = onlineSoftmaxDifferential(scores, values, bs);
      expect(maxAbsDiff).toBeLessThan(1e-12);
    }
  });

  it("holds under an extreme-max block ordering (the rescale is exercised)", () => {
    // The last block contains the global max — every earlier block's accumulator
    // must be rescaled by exp(m_old − m_new). If the rescale were dropped, the
    // diff would be O(1).
    const scores = [0, 0, 0, 0, 0, 0, 0, 100];
    const values = scores.map((_, j) => [j, j * 2]);
    const { maxAbsDiff } = onlineSoftmaxDifferential(scores, values, 2);
    expect(maxAbsDiff).toBeLessThan(1e-9);
  });
});
