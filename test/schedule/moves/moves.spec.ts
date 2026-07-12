/**
 * THE MOVE-ALGEBRA unit gate (P2 wave B deliverable 1).
 *
 * Per move: legal application transforms as specified; illegal application
 * refuses with the typed reason (a stable code); apply ∘ inverse = identity
 * (digest). Schema-level only — no GPU, no dispatch. Uses the naive-attention
 * composition regions (real derived states) plus tiny hand-built states.
 */

import { describe, expect, it } from "vitest";
import { naiveAttentionComposition } from "../../../src/schedule/attention-skeleton";
import { scheduleDigest } from "../../../src/schedule/canonical";
import { applyInverse, applyMove } from "../../../src/schedule/moves/moves";
import type {
  ProgramGridMap,
  ScheduleMove,
  ScheduleState,
  SemanticSchedule,
} from "../../../src/schedule/types";

const b = <T>(s: string): T => s as unknown as T;

/** A tiny reduction-shaped state (one parallel loop over `out`, a sum body). */
function sumState(): ScheduleState {
  const semantic: SemanticSchedule = {
    blockShapes: [[]],
    loopNest: [
      {
        uid: b("loop:out"),
        entity: b("ent:loop:out"),
        axis: b("axis:out"),
        kind: "parallel",
        bound: {
          kind: "affineLeaf",
          leaf: { kind: "uniformRef", name: "size" },
        },
        children: [
          {
            uid: b("loop:reduce"),
            entity: b("ent:loop:reduce"),
            axis: b("axis:reduce"),
            kind: "sequential",
            bound: {
              kind: "affineLeaf",
              leaf: { kind: "uniformRef", name: "size" },
            },
            children: [],
          },
        ],
      },
    ],
    ordering: { kind: "flat" },
    programGridMap: { kind: "identity" },
    values: [
      {
        uid: b("input"),
        entity: b("ent:input"),
        allocation: "global",
        dtype: "f32",
        aliasOf: null,
      },
      {
        uid: b("result"),
        entity: b("ent:result"),
        allocation: "register",
        dtype: "f32",
        aliasOf: null,
      },
      {
        uid: b("out"),
        entity: b("ent:out"),
        allocation: "global",
        dtype: "f32",
        aliasOf: null,
      },
    ],
    noMaterialization: [],
    stores: [{ source: b("result"), target: b("out"), atLoop: b("loop:out") }],
    bodies: [
      {
        result: b("result"),
        expr: {
          kind: "apply",
          catalog: { op: "reduce_sum" },
          args: [{ kind: "value", value: b("input") }],
        },
      },
    ],
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
    region: b("region:test"),
  };
}

/** apply a move, expecting success; return the after-state + provenance. */
function expectApplied(state: ScheduleState, move: ScheduleMove) {
  const outcome = applyMove(state, move);
  expect(outcome.kind).toBe("applied");
  if (outcome.kind !== "applied") throw new Error("unreachable");
  return outcome;
}

describe("move algebra — tile", () => {
  const state = sumState();
  const move: ScheduleMove = {
    move: "tile",
    loop: b("loop:reduce"),
    axis: b("axis:reduce"),
    factor: 32,
  };

  it("legal tile splits the loop into outer × inner (as specified)", () => {
    const { state: after } = expectApplied(state, move);
    // The outer loop replaces the reduce loop; it encloses an inner factor loop.
    const outer = after.semantic.loopNest[0].children[0];
    expect(outer.uid).toBe("loop:reduce:outer");
    expect(outer.children[0].uid).toBe("loop:reduce:inner");
    expect(outer.bound.kind).toBe("affineCeilDiv");
    // Block shape gains the tiled sub-extent.
    expect(after.semantic.blockShapes).toContainEqual([32]);
    // The after-state is a DIFFERENT point (digest differs).
    expect(scheduleDigest(after)).not.toBe(scheduleDigest(state));
  });

  it("apply ∘ inverse = identity (digest)", () => {
    const { state: after, provenance } = expectApplied(state, move);
    const back = applyInverse(after, provenance);
    expect(scheduleDigest(back)).toBe(scheduleDigest(state));
  });

  it("refuses a missing loop with the typed reason", () => {
    const outcome = applyMove(state, {
      move: "tile",
      loop: b("loop:absent"),
      axis: b("axis:reduce"),
      factor: 32,
    });
    expect(outcome.kind).toBe("refused");
    if (outcome.kind === "refused")
      expect(outcome.refusal.code).toBe("TILE_LOOP_NOT_FOUND");
  });

  it("refuses an axis mismatch", () => {
    const outcome = applyMove(state, {
      move: "tile",
      loop: b("loop:reduce"),
      axis: b("axis:WRONG"),
      factor: 32,
    });
    expect(outcome.kind).toBe("refused");
    if (outcome.kind === "refused")
      expect(outcome.refusal.code).toBe("TILE_AXIS_MISMATCH");
  });

  it("refuses an invalid factor (< 2 / non-integer)", () => {
    for (const factor of [1, 0, 1.5]) {
      const outcome = applyMove(state, {
        move: "tile",
        loop: b("loop:reduce"),
        axis: b("axis:reduce"),
        factor,
      });
      expect(outcome.kind).toBe("refused");
      if (outcome.kind === "refused")
        expect(outcome.refusal.code).toBe("TILE_FACTOR_INVALID");
    }
  });
});

describe("move algebra — stream (the refusal-first F17 boundary)", () => {
  const state = sumState();

  it("streams a value with a head/body decomposition (sum monoid)", () => {
    const move: ScheduleMove = {
      move: "stream",
      value: b("result"),
      loop: b("loop:reduce"),
    };
    const { state: after } = expectApplied(state, move);
    // The store of `result` is deleted; a no-materialization edge is added.
    expect(
      after.semantic.stores.find((e) => e.target === "result"),
    ).toBeUndefined();
    expect(
      after.semantic.noMaterialization.some(
        (e) => e.producer === "result" && e.acrossLoop === "loop:reduce",
      ),
    ).toBe(true);
  });

  it("apply ∘ inverse = identity (the deleted store is restored)", () => {
    const move: ScheduleMove = {
      move: "stream",
      value: b("result"),
      loop: b("loop:reduce"),
    };
    const { state: after, provenance } = expectApplied(state, move);
    const back = applyInverse(after, provenance);
    expect(scheduleDigest(back)).toBe(scheduleDigest(state));
  });

  it("REFUSES streaming a value with no head/body decomposition (a plain map)", () => {
    // Build a state whose body is add(x,y) — no reduction to fold.
    const s = sumState();
    const mapped: ScheduleState = {
      ...s,
      semantic: {
        ...s.semantic,
        bodies: [
          {
            result: b("result"),
            expr: {
              kind: "apply",
              catalog: { op: "add" },
              args: [
                { kind: "value", value: b("input") },
                { kind: "value", value: b("input") },
              ],
            },
          },
        ],
      },
    };
    const outcome = applyMove(mapped, {
      move: "stream",
      value: b("result"),
      loop: b("loop:reduce"),
    });
    expect(outcome.kind).toBe("refused");
    if (outcome.kind === "refused") {
      expect(outcome.refusal.code).toBe("STREAM_NO_HEAD_BODY");
      // A plain map names no discharging lemma.
      expect(outcome.refusal.refusal?.dischargedBy).toBeNull();
    }
  });
});

describe("move algebra — recolor", () => {
  const state = sumState();
  const move: ScheduleMove = {
    move: "recolor",
    value: b("result"),
    column: 0,
    tier: "shared",
    transitionRole: "materialization-boundary",
  };

  it("changes the value's residency-intent tier", () => {
    const { state: after } = expectApplied(state, move);
    expect(
      after.semantic.values.find((v) => v.uid === "result")?.allocation,
    ).toBe("shared");
  });

  it("apply ∘ inverse = identity (self-inverse: restores the prior tier)", () => {
    const { state: after, provenance } = expectApplied(state, move);
    const back = applyInverse(after, provenance);
    expect(scheduleDigest(back)).toBe(scheduleDigest(state));
  });

  it("refuses an absent value", () => {
    const outcome = applyMove(state, { ...move, value: b("absent") });
    expect(outcome.kind).toBe("refused");
    if (outcome.kind === "refused")
      expect(outcome.refusal.code).toBe("RECOLOR_VALUE_NOT_FOUND");
  });
});

describe("move algebra — program-map (R4 bijection legality)", () => {
  // Use the matmul QK^T region which has real m/n axes.
  const state = naiveAttentionComposition(64).qkT.state;

  it("applies a grouped map over a real axis (bijective by construction)", () => {
    const move: ScheduleMove = {
      move: "program-map",
      map: { kind: "grouped", groupAxis: b("axis:m"), groupSize: 8 },
    };
    const { state: after } = expectApplied(state, move);
    expect(after.semantic.programGridMap.kind).toBe("grouped");
  });

  it("apply ∘ inverse = identity (restores the previous map)", () => {
    const move: ScheduleMove = {
      move: "program-map",
      map: { kind: "swap", axes: [b("axis:m"), b("axis:n")] },
    };
    const { state: after, provenance } = expectApplied(state, move);
    const back = applyInverse(after, provenance);
    expect(scheduleDigest(back)).toBe(scheduleDigest(state));
  });

  it("REFUSES a map naming an axis not in the loop nest (bijection can't hold)", () => {
    const move: ScheduleMove = {
      move: "program-map",
      map: { kind: "grouped", groupAxis: b("axis:PHANTOM"), groupSize: 8 },
    };
    const outcome = applyMove(state, move);
    expect(outcome.kind).toBe("refused");
    if (outcome.kind === "refused")
      expect(outcome.refusal.code).toBe("PROGRAM_MAP_UNKNOWN_AXIS");
  });

  it("REFUSES a grouped map with a non-positive groupSize", () => {
    const move: ScheduleMove = {
      move: "program-map",
      map: {
        kind: "grouped",
        groupAxis: b("axis:m"),
        groupSize: 0,
      } as ProgramGridMap,
    };
    const outcome = applyMove(state, move);
    expect(outcome.kind).toBe("refused");
    if (outcome.kind === "refused")
      expect(outcome.refusal.code).toBe("PROGRAM_MAP_NOT_BIJECTIVE");
  });
});

describe("move algebra — pack and role-partition", () => {
  const state = sumState();

  it("pack wraps the nest in a pack axis; apply ∘ inverse = identity", () => {
    const move: ScheduleMove = {
      move: "pack",
      loops: [b("loop:out")],
      kind: "map",
    };
    const { state: after, provenance } = expectApplied(state, move);
    expect(after.semantic.loopNest[0].uid).toContain("loop:pack");
    const back = applyInverse(after, provenance);
    expect(scheduleDigest(back)).toBe(scheduleDigest(state));
  });

  it("pack refuses an empty loop list", () => {
    const outcome = applyMove(state, { move: "pack", loops: [], kind: "map" });
    expect(outcome.kind).toBe("refused");
    if (outcome.kind === "refused")
      expect(outcome.refusal.code).toBe("PACK_EMPTY");
  });

  it("role-partition adds named roles; apply ∘ inverse = identity", () => {
    const move: ScheduleMove = {
      move: "role-partition",
      loop: b("loop:out"),
      roles: [b("producer"), b("consumer")],
    };
    const { state: after, provenance } = expectApplied(state, move);
    expect(after.semantic.roles.map((r) => r.role)).toEqual([
      "producer",
      "consumer",
    ]);
    const back = applyInverse(after, provenance);
    expect(scheduleDigest(back)).toBe(scheduleDigest(state));
  });

  it("role-partition refuses empty roles", () => {
    const outcome = applyMove(state, {
      move: "role-partition",
      loop: b("loop:out"),
      roles: [],
    });
    expect(outcome.kind).toBe("refused");
    if (outcome.kind === "refused")
      expect(outcome.refusal.code).toBe("ROLE_PARTITION_EMPTY_ROLES");
  });
});

describe("move algebra — determinism", () => {
  it("applying the same move twice yields digest-identical after-states", () => {
    const state = sumState();
    const move: ScheduleMove = {
      move: "tile",
      loop: b("loop:reduce"),
      axis: b("axis:reduce"),
      factor: 16,
    };
    const a = expectApplied(state, move).state;
    const b2 = expectApplied(state, move).state;
    expect(scheduleDigest(a)).toBe(scheduleDigest(b2));
  });
});
