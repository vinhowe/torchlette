import { beforeEach, describe, expect, it } from "vitest";

import {
  aliasGroupsKey,
  aliasPatternKey,
  analyzeExternalizeNeeds,
  analyzeRegionExit,
  buildStateIfaceSig,
  computeAliasGroups,
  computeStateSlotAliasPattern,
  generateExtendedCacheKey,
  getNullStateSentinel,
  isInPlaceMutation,
  resetNullStateSentinels,
  serializeExtendedCacheKey,
  stateIfaceSigKey,
  toOutOfPlaceOp,
} from "../src/engine/compiled-region";

describe("§8.3 Canonical Arg Alias Groups", () => {
  it("groups arguments with same baseId together", () => {
    // Args 0 and 2 share baseId 100, arg 1 has baseId 200
    const groups = computeAliasGroups([100, 200, 100]);

    expect(groups.length).toBe(2);
    // First group: args 0 and 2 (aliased)
    expect(groups[0].argIndices).toEqual([0, 2]);
    expect(groups[0].baseId).toBe(100);
    // Second group: arg 1 (single)
    expect(groups[1].argIndices).toEqual([1]);
    expect(groups[1].baseId).toBe(200);
  });

  it("handles non-aliased arguments", () => {
    const groups = computeAliasGroups([undefined, undefined, undefined]);

    expect(groups.length).toBe(3);
    expect(groups[0].argIndices).toEqual([0]);
    expect(groups[1].argIndices).toEqual([1]);
    expect(groups[2].argIndices).toEqual([2]);
  });

  it("handles mixed aliased and non-aliased", () => {
    // Args 1 and 3 share baseId 50, others are independent
    const groups = computeAliasGroups([100, 50, undefined, 50, 200]);

    // Should have: [1,3] aliased, then [0], [2], [4] singles
    const aliasedGroup = groups.find((g) => g.argIndices.length > 1);
    expect(aliasedGroup).toBeDefined();
    expect(aliasedGroup?.argIndices).toEqual([1, 3]);
  });

  it("produces deterministic keys", () => {
    const groups1 = computeAliasGroups([100, 200, 100]);
    const groups2 = computeAliasGroups([100, 200, 100]);

    expect(aliasGroupsKey(groups1)).toBe(aliasGroupsKey(groups2));
  });

  it("produces different keys for different patterns", () => {
    const groups1 = computeAliasGroups([100, 100]); // Args alias
    const groups2 = computeAliasGroups([100, 200]); // Args don't alias

    expect(aliasGroupsKey(groups1)).not.toBe(aliasGroupsKey(groups2));
  });
});

describe("§8.4 State Interface Signature", () => {
  it("builds signature from state accesses", () => {
    const stateAccesses = new Map<number, "load" | "store" | "load_store">([
      [0, "load"],
      [1, "store"],
      [2, "load_store"],
    ]);
    const argAccesses = new Map<number, "load" | "store" | "load_store">([
      [0, "load"],
    ]);

    const sig = buildStateIfaceSig(
      { epoch: 1, nodes: [], fusionGroups: [] },
      stateAccesses,
      argAccesses,
      true,
    );

    expect(sig.accesses.length).toBe(5); // 3 state + 1 arg + 1 global
    expect(sig.usesGlobalToken).toBe(true);
    expect(sig.touchedTargets.length).toBe(5);
  });

  it("orders accesses deterministically", () => {
    const stateAccesses = new Map<number, "load" | "store" | "load_store">([
      [2, "load"],
      [0, "store"],
      [1, "load"],
    ]);

    const sig = buildStateIfaceSig(
      { epoch: 1, nodes: [], fusionGroups: [] },
      stateAccesses,
      new Map(),
      false,
    );

    // Should be ordered by slot number: 0, 1, 2
    expect(sig.accesses[0].target).toEqual({ kind: "state", slot: 0 });
    expect(sig.accesses[1].target).toEqual({ kind: "state", slot: 1 });
    expect(sig.accesses[2].target).toEqual({ kind: "state", slot: 2 });
  });

  it("produces deterministic signature keys", () => {
    const stateAccesses = new Map<number, "load" | "store" | "load_store">([
      [0, "load"],
      [1, "store"],
    ]);

    const sig1 = buildStateIfaceSig(
      { epoch: 1, nodes: [], fusionGroups: [] },
      stateAccesses,
      new Map(),
      true,
    );
    const sig2 = buildStateIfaceSig(
      { epoch: 2, nodes: [], fusionGroups: [] },
      stateAccesses,
      new Map(),
      true,
    );

    expect(stateIfaceSigKey(sig1)).toBe(stateIfaceSigKey(sig2));
  });
});

describe("§8.5 State-Slot Alias Pattern", () => {
  it("identifies aliased state slots", () => {
    // Slots 0 and 2 share baseId 100
    const pattern = computeStateSlotAliasPattern([100, 200, 100, 300]);

    expect(pattern.universalMayAlias).toBe(false);
    expect(pattern.aliasGroups.length).toBe(1);
    expect(pattern.aliasGroups[0]).toEqual([0, 2]);
  });

  it("returns empty groups when no aliasing", () => {
    const pattern = computeStateSlotAliasPattern([100, 200, 300]);

    expect(pattern.universalMayAlias).toBe(false);
    expect(pattern.aliasGroups.length).toBe(0);
  });

  it("supports universal may-alias fallback", () => {
    const pattern = computeStateSlotAliasPattern([100, 200], true);

    expect(pattern.universalMayAlias).toBe(true);
    expect(aliasPatternKey(pattern)).toBe("universal");
  });

  it("produces deterministic keys", () => {
    const pattern1 = computeStateSlotAliasPattern([100, 200, 100]);
    const pattern2 = computeStateSlotAliasPattern([100, 200, 100]);

    expect(aliasPatternKey(pattern1)).toBe(aliasPatternKey(pattern2));
  });

  it("produces different keys for different patterns", () => {
    const pattern1 = computeStateSlotAliasPattern([100, 100]); // Alias
    const pattern2 = computeStateSlotAliasPattern([100, 200]); // No alias

    expect(aliasPatternKey(pattern1)).not.toBe(aliasPatternKey(pattern2));
  });
});

describe("§8.7 Auto-Externalize", () => {
  it("identifies SSA bases needing externalization", () => {
    const ssaBases = [1, 2, 3];
    const bindings = new Map([
      [1, { kind: "ssa" }],
      [2, { kind: "loc", locId: 10 }],
      [3, { kind: "pending_loc", locId: 20 }],
    ]);

    const requests = analyzeExternalizeNeeds(ssaBases, bindings);

    // Base 1 (SSA) and 3 (pending_loc) need externalization
    expect(requests.length).toBe(2);
    expect(requests.find((r) => r.baseId === 1)).toBeDefined();
    expect(requests.find((r) => r.baseId === 3)).toBeDefined();
    expect(requests.find((r) => r.baseId === 2)).toBeUndefined(); // loc-backed
  });

  it("handles bases with no existing binding", () => {
    const ssaBases = [1, 2];
    const bindings = new Map<number, { kind: string; locId?: number }>();

    const requests = analyzeExternalizeNeeds(ssaBases, bindings);

    expect(requests.length).toBe(2);
    expect(requests[0].needsMaterializer).toBe(true);
    expect(requests[1].needsMaterializer).toBe(true);
  });
});

describe("§8.8 Null-State Sentinels", () => {
  beforeEach(() => {
    resetNullStateSentinels();
  });

  it("creates stable sentinels for same (fn, slot) pair", () => {
    const sentinel1 = getNullStateSentinel(1, 0);
    const sentinel2 = getNullStateSentinel(1, 0);

    expect(sentinel1.sentinelId).toBe(sentinel2.sentinelId);
    expect(sentinel1.compiledFnId).toBe(1);
    expect(sentinel1.stateSlot).toBe(0);
  });

  it("creates different sentinels for different (fn, slot) pairs", () => {
    const sentinel1 = getNullStateSentinel(1, 0);
    const sentinel2 = getNullStateSentinel(1, 1);
    const sentinel3 = getNullStateSentinel(2, 0);

    expect(sentinel1.sentinelId).not.toBe(sentinel2.sentinelId);
    expect(sentinel1.sentinelId).not.toBe(sentinel3.sentinelId);
    expect(sentinel2.sentinelId).not.toBe(sentinel3.sentinelId);
  });
});

describe("§8.10 Functionalization", () => {
  it("identifies in-place mutation ops", () => {
    expect(isInPlaceMutation("add_")).toBe(true);
    expect(isInPlaceMutation("mul_")).toBe(true);
    expect(isInPlaceMutation("relu_")).toBe(true);
    expect(isInPlaceMutation("add")).toBe(false);
    expect(isInPlaceMutation("matmul")).toBe(false);
  });

  it("converts in-place ops to out-of-place", () => {
    expect(toOutOfPlaceOp("add_")).toBe("add");
    expect(toOutOfPlaceOp("mul_")).toBe("mul");
    expect(toOutOfPlaceOp("add")).toBe("add"); // No change
  });
});

describe("§8.11 Region-Exit Persistence", () => {
  it("analyzes region exit with mutations", () => {
    const mutations = [
      { originalNodeId: 1, mutatedBaseId: 100, newValueNodeId: 2 },
      { originalNodeId: 3, mutatedBaseId: 200, newValueNodeId: 4 },
    ];
    const outputBases = [100, 200, 300];
    const bindings = new Map([
      [100, { kind: "loc", locId: 10 }],
      [200, { kind: "pending_loc", locId: 20 }],
      [300, { kind: "ssa" }],
    ]);

    const plan = analyzeRegionExit(mutations, outputBases, bindings);

    expect(plan.commits.length).toBe(2);
    expect(plan.usesGlobalToken).toBe(true);
    // Pending_loc outputs need writeback
    expect(plan.writebacks.length).toBe(1);
    expect(plan.writebacks[0].baseId).toBe(200);
  });

  it("handles region with no mutations", () => {
    const plan = analyzeRegionExit(
      [],
      [100],
      new Map([[100, { kind: "ssa" }]]),
    );

    expect(plan.commits.length).toBe(0);
    expect(plan.usesGlobalToken).toBe(false);
  });
});

describe("Extended Cache Key", () => {
  it("generates deterministic extended cache keys", () => {
    const aliasGroups = computeAliasGroups([100, 200, 100]);
    const sig = buildStateIfaceSig(
      { epoch: 1, nodes: [], fusionGroups: [] },
      new Map([[0, "load"]]),
      new Map(),
      false,
    );
    const pattern = computeStateSlotAliasPattern([100, 200]);

    const key1 = generateExtendedCacheKey(
      "abc123",
      [
        [2, 3],
        [3, 4],
      ],
      ["f32", "f32"],
      aliasGroups,
      sig,
      pattern,
    );
    const key2 = generateExtendedCacheKey(
      "abc123",
      [
        [2, 3],
        [3, 4],
      ],
      ["f32", "f32"],
      aliasGroups,
      sig,
      pattern,
    );

    expect(serializeExtendedCacheKey(key1)).toBe(
      serializeExtendedCacheKey(key2),
    );
  });

  it("produces different keys for different configurations", () => {
    const aliasGroups = computeAliasGroups([100, 200]);
    const sig = buildStateIfaceSig(
      { epoch: 1, nodes: [], fusionGroups: [] },
      new Map(),
      new Map(),
      false,
    );
    const pattern = computeStateSlotAliasPattern([100, 200]);

    const key1 = generateExtendedCacheKey(
      "abc123",
      [[2, 3]],
      ["f32"],
      aliasGroups,
      sig,
      pattern,
    );
    const key2 = generateExtendedCacheKey(
      "abc123",
      [[2, 4]], // Different shape
      ["f32"],
      aliasGroups,
      sig,
      pattern,
    );
    const key3 = generateExtendedCacheKey(
      "abc123",
      [[2, 3]],
      ["i32"], // Different dtype
      aliasGroups,
      sig,
      pattern,
    );

    expect(serializeExtendedCacheKey(key1)).not.toBe(
      serializeExtendedCacheKey(key2),
    );
    expect(serializeExtendedCacheKey(key1)).not.toBe(
      serializeExtendedCacheKey(key3),
    );
  });
});
