import { describe, expect, it } from "vitest";

import type { IRGraph, IRNode } from "../src/engine/ir";
import {
  analyzeTokAfterOpportunities,
  generateCSEKey,
  isCSEable,
  isEffectful,
  isPureOp,
  isRandomOp,
  optimizeIR,
  performCSE,
  performDCE,
} from "../src/engine/ir-optimize";

// Helper to create test IR nodes
function makeNode(
  id: number,
  op: string,
  inputs: number[] = [],
  shape: number[] = [2, 3],
  dtype: "f32" | "i32" = "f32",
): IRNode {
  return { id, op, epoch: 1, kind: "lazy_op", inputs, shape, dtype };
}

function makeGraph(nodes: IRNode[]): IRGraph {
  return { epoch: 1, nodes, fusionGroups: [] };
}

describe("§15 Op Classification", () => {
  it("identifies pure ops", () => {
    expect(isPureOp("add")).toBe(true);
    expect(isPureOp("mul")).toBe(true);
    expect(isPureOp("relu")).toBe(true);
    expect(isPureOp("matmul")).toBe(true);
    expect(isPureOp("reshape")).toBe(true);
    expect(isPureOp("sum")).toBe(true);
  });

  it("identifies random ops", () => {
    expect(isRandomOp("rand")).toBe(true);
    expect(isRandomOp("randn")).toBe(true);
    expect(isRandomOp("dropout")).toBe(true);
    expect(isRandomOp("add")).toBe(false);
    expect(isRandomOp("matmul")).toBe(false);
  });

  it("random ops are not CSEable", () => {
    expect(isCSEable("add")).toBe(true);
    expect(isCSEable("rand")).toBe(false);
    expect(isCSEable("randn")).toBe(false);
    expect(isCSEable("dropout")).toBe(false);
  });

  it("identifies effectful ops", () => {
    expect(isEffectful("add_")).toBe(true); // In-place
    expect(isEffectful("mul_")).toBe(true);
    expect(isEffectful("loc_store")).toBe(true);
    expect(isEffectful("add")).toBe(false);
    expect(isEffectful("relu")).toBe(false);
  });
});

describe("§15 CSE Key Generation", () => {
  it("generates same key for identical pure ops", () => {
    const node1 = makeNode(1, "add", [10, 20]);
    const node2 = makeNode(2, "add", [10, 20]);

    const key1 = generateCSEKey(
      node1,
      new Map([
        [10, 10],
        [20, 20],
      ]),
    );
    const key2 = generateCSEKey(
      node2,
      new Map([
        [10, 10],
        [20, 20],
      ]),
    );

    expect(key1).toBe(key2);
  });

  it("generates different keys for different ops", () => {
    const node1 = makeNode(1, "add", [10, 20]);
    const node2 = makeNode(2, "mul", [10, 20]);

    const cseIds = new Map([
      [10, 10],
      [20, 20],
    ]);
    expect(generateCSEKey(node1, cseIds)).not.toBe(
      generateCSEKey(node2, cseIds),
    );
  });

  it("generates different keys for different inputs", () => {
    const node1 = makeNode(1, "add", [10, 20]);
    const node2 = makeNode(2, "add", [10, 30]);

    const cseIds = new Map([
      [10, 10],
      [20, 20],
      [30, 30],
    ]);
    expect(generateCSEKey(node1, cseIds)).not.toBe(
      generateCSEKey(node2, cseIds),
    );
  });

  it("generates different keys for different shapes", () => {
    const node1 = makeNode(1, "add", [10, 20], [2, 3]);
    const node2 = makeNode(2, "add", [10, 20], [3, 4]);

    const cseIds = new Map([
      [10, 10],
      [20, 20],
    ]);
    expect(generateCSEKey(node1, cseIds)).not.toBe(
      generateCSEKey(node2, cseIds),
    );
  });

  it("generates unique keys for random ops", () => {
    const node1 = makeNode(1, "rand", []);
    const node2 = makeNode(2, "rand", []);

    expect(generateCSEKey(node1, new Map())).not.toBe(
      generateCSEKey(node2, new Map()),
    );
  });
});

describe("§15 Common Subexpression Elimination", () => {
  it("eliminates duplicate pure ops", () => {
    const graph = makeGraph([
      makeNode(1, "input", []),
      makeNode(2, "input", []),
      makeNode(3, "add", [1, 2]), // First add
      makeNode(4, "add", [1, 2]), // Duplicate add
      makeNode(5, "mul", [3, 4]),
    ]);

    const result = performCSE(graph);

    expect(result.eliminatedNodes).toContain(4);
    expect(result.stats.eliminatedCount).toBe(1);
    expect(result.optimizedGraph.nodes.length).toBe(4);

    // The mul should now reference node 3 twice
    const mulNode = result.optimizedGraph.nodes.find((n) => n.op === "mul");
    expect(mulNode?.inputs).toEqual([3, 3]);
  });

  it("does NOT eliminate random ops", () => {
    const graph = makeGraph([
      makeNode(1, "rand", [], [2, 3]),
      makeNode(2, "rand", [], [2, 3]), // Same shape, but must not CSE
      makeNode(3, "add", [1, 2]),
    ]);

    const result = performCSE(graph);

    expect(result.eliminatedNodes).toEqual([]);
    expect(result.optimizedGraph.nodes.length).toBe(3);
  });

  it("handles chains of CSE opportunities", () => {
    const graph = makeGraph([
      makeNode(1, "input", []),
      makeNode(2, "relu", [1]), // First relu(input)
      makeNode(3, "relu", [1]), // Duplicate relu(input)
      makeNode(4, "add", [2, 3]), // add(relu, relu) -> add(relu, relu)
      makeNode(5, "add", [2, 3]), // Duplicate add
    ]);

    const result = performCSE(graph);

    // Should eliminate node 3 (duplicate relu) and node 5 (duplicate add)
    expect(result.eliminatedNodes.length).toBe(2);
    expect(result.eliminatedNodes).toContain(3);
    expect(result.eliminatedNodes).toContain(5);
  });

  it("preserves input order for CSE key", () => {
    const graph = makeGraph([
      makeNode(1, "input", []),
      makeNode(2, "input", []),
      makeNode(3, "sub", [1, 2]), // sub(a, b)
      makeNode(4, "sub", [2, 1]), // sub(b, a) - different!
    ]);

    const result = performCSE(graph);

    // sub(a,b) !== sub(b,a), so no elimination
    expect(result.eliminatedNodes).toEqual([]);
  });
});

describe("§15 Dead Code Elimination", () => {
  it("eliminates unreachable nodes", () => {
    const graph = makeGraph([
      makeNode(1, "input", []),
      makeNode(2, "relu", [1]), // Used in output
      makeNode(3, "add", [1, 1]), // Not used - dead
      makeNode(4, "mul", [2, 2]), // Output
    ]);

    const result = performDCE(graph, [4]);

    expect(result.eliminatedNodes).toContain(3);
    expect(result.optimizedGraph.nodes.length).toBe(3);
    expect(result.optimizedGraph.nodes.map((n) => n.id)).toEqual([1, 2, 4]);
  });

  it("keeps all nodes transitively reachable from outputs", () => {
    const graph = makeGraph([
      makeNode(1, "input", []),
      makeNode(2, "relu", [1]),
      makeNode(3, "sigmoid", [2]),
      makeNode(4, "mul", [2, 3]), // Output - needs 2 and 3
    ]);

    const result = performDCE(graph, [4]);

    expect(result.eliminatedNodes).toEqual([]);
    expect(result.optimizedGraph.nodes.length).toBe(4);
  });

  it("keeps effectful nodes even if unreachable", () => {
    const graph = makeGraph([
      makeNode(1, "input", []),
      makeNode(2, "relu", [1]), // Output
      makeNode(3, "add_", [1, 1]), // In-place - effectful, kept
    ]);

    const result = performDCE(graph, [2]);

    // add_ is effectful, so it's kept
    expect(result.eliminatedNodes).toEqual([]);
    expect(result.optimizedGraph.nodes.length).toBe(3);
  });

  it("handles multiple outputs", () => {
    const graph = makeGraph([
      makeNode(1, "input", []),
      makeNode(2, "relu", [1]), // Output 1
      makeNode(3, "sigmoid", [1]), // Output 2
      makeNode(4, "tanh", [1]), // Dead
    ]);

    const result = performDCE(graph, [2, 3]);

    expect(result.eliminatedNodes).toContain(4);
    expect(result.optimizedGraph.nodes.length).toBe(3);
  });
});

describe("§15 tok_after Optimization", () => {
  it("identifies redundant loads from same loc", () => {
    const graph = makeGraph([
      makeNode(1, "loc_load", []), // First load from loc 0
      makeNode(2, "relu", [1]),
      makeNode(3, "loc_load", []), // Second load from loc 0 (redundant)
      makeNode(4, "add", [2, 3]),
    ]);

    // Both nodes 1 and 3 load from loc 0
    const locLoads = new Map([[0, [1, 3]]]);
    const locStores = new Map<number, number[]>();

    const result = analyzeTokAfterOpportunities(graph, locLoads, locStores);

    expect(result.redundantLoads).toContain(3);
    expect(result.firstLoadMap.get(3)).toBe(1);
  });

  it("does not mark load as redundant if store intervenes", () => {
    const graph = makeGraph([
      makeNode(1, "loc_load", []), // Load
      makeNode(2, "loc_store", [1]), // Store between loads
      makeNode(3, "loc_load", []), // Load (not redundant due to store)
    ]);

    const locLoads = new Map([[0, [1, 3]]]);
    const locStores = new Map([[0, [2]]]);

    const result = analyzeTokAfterOpportunities(graph, locLoads, locStores);

    expect(result.redundantLoads).toEqual([]);
  });

  it("handles multiple locs independently", () => {
    const graph = makeGraph([
      makeNode(1, "loc_load", []), // Load from loc 0
      makeNode(2, "loc_load", []), // Load from loc 1
      makeNode(3, "loc_load", []), // Second load from loc 0 (redundant)
      makeNode(4, "loc_load", []), // Second load from loc 1 (redundant)
    ]);

    const locLoads = new Map([
      [0, [1, 3]],
      [1, [2, 4]],
    ]);
    const locStores = new Map<number, number[]>();

    const result = analyzeTokAfterOpportunities(graph, locLoads, locStores);

    expect(result.redundantLoads.length).toBe(2);
    expect(result.redundantLoads).toContain(3);
    expect(result.redundantLoads).toContain(4);
  });
});

describe("§15 Full Optimization Pipeline", () => {
  it("runs CSE and DCE together", () => {
    const graph = makeGraph([
      makeNode(1, "input", []),
      makeNode(2, "relu", [1]),
      makeNode(3, "relu", [1]), // Duplicate - CSE'd
      makeNode(4, "add", [2, 3]), // After CSE: add(2, 2)
      makeNode(5, "sigmoid", [1]), // Dead - not in output
    ]);

    const result = optimizeIR(graph, {
      enableCSE: true,
      enableDCE: true,
      outputNodeIds: [4],
    });

    // CSE eliminates node 3, DCE eliminates node 5
    expect(result.stats.cseEliminated).toBe(1);
    expect(result.stats.dceEliminated).toBe(1);
    expect(result.stats.finalNodeCount).toBe(3); // input, relu, add
  });

  it("respects option flags", () => {
    const graph = makeGraph([
      makeNode(1, "input", []),
      makeNode(2, "relu", [1]),
      makeNode(3, "relu", [1]), // Would be CSE'd
    ]);

    const withCSE = optimizeIR(graph, { enableCSE: true, enableDCE: false });
    const withoutCSE = optimizeIR(graph, {
      enableCSE: false,
      enableDCE: false,
    });

    expect(withCSE.stats.cseEliminated).toBe(1);
    expect(withoutCSE.stats.cseEliminated).toBe(0);
  });

  it("handles empty graph", () => {
    const graph = makeGraph([]);

    const result = optimizeIR(graph, {
      enableCSE: true,
      enableDCE: true,
      outputNodeIds: [],
    });

    expect(result.optimizedGraph.nodes.length).toBe(0);
    expect(result.stats.finalNodeCount).toBe(0);
  });

  it("preserves random ops through full pipeline", () => {
    const graph = makeGraph([
      makeNode(1, "rand", [], [2, 3]),
      makeNode(2, "rand", [], [2, 3]), // Same but must stay separate
      makeNode(3, "add", [1, 2]), // Output
    ]);

    const result = optimizeIR(graph, {
      enableCSE: true,
      enableDCE: true,
      outputNodeIds: [3],
    });

    // Both rand nodes should survive
    expect(result.optimizedGraph.nodes.length).toBe(3);
    expect(result.stats.cseEliminated).toBe(0);
  });
});
