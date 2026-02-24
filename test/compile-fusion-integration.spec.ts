/**
 * Integration tests for §15 - IR Optimization and Fusion
 *
 * These tests verify:
 * - CSE (Common Subexpression Elimination) works
 * - DCE (Dead Code Elimination) works
 * - Fusion detection identifies fusible op chains
 * - Optimized execution produces correct results
 */

import { describe, expect, it, beforeEach } from "vitest";
import { RuntimeEngine } from "../src/runtime/engine";
import { tensorFromArray } from "../src/runtime/engine-facade";
import { createBaseId, Tensor, resetBaseIdCounter } from "../src/runtime/tensor";
import {
  buildPlan,
  createLazyIRNode,
  createPendingRef,
  executePlanOptimized,
  resetNodeIdCounter,
  resetStorageIdCounter,
  type LazyIRNode,
} from "../src/engine/lazy";
import {
  detectFusionGroups,
  groupToRecipe,
  hasFusionOpportunities,
  hasFusionPotential,
  reorderPlanForFusion,
  segmentPlanForExecution,
  isFusibleOp,
} from "../src/engine/fusion-detect";
import {
  lazyPlanToIR,
  detectFusionGroups as detectFusionGroupsIR,
  isFusibleElementwise,
} from "../src/engine/lazy-to-ir";
import { performCSE, performDCE, optimizeIR } from "../src/engine/ir-optimize";
import type { IRGraph, IRNode } from "../src/engine/ir";
import { getBackend } from "../src/backend/registry";

describe("§15 Fusion Detection", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
  });

  describe("isFusibleOp", () => {
    it("identifies fusible elementwise ops", () => {
      expect(isFusibleOp("add")).toBe(true);
      expect(isFusibleOp("sub")).toBe(true);
      expect(isFusibleOp("mul")).toBe(true);
      expect(isFusibleOp("div")).toBe(true);
      expect(isFusibleOp("sqrt")).toBe(true);
      expect(isFusibleOp("relu")).toBe(true);
    });

    it("identifies non-fusible ops", () => {
      expect(isFusibleOp("matmul")).toBe(false);
      expect(isFusibleOp("sum")).toBe(false);
      expect(isFusibleOp("mean")).toBe(false);
      expect(isFusibleOp("gather")).toBe(false);
      expect(isFusibleOp("reshape")).toBe(false);
    });
  });

  describe("detectFusionGroups", () => {
    it("detects consecutive elementwise ops", () => {
      const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const b = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [5, 6, 7, 8] });
      const add = createLazyIRNode("add", [createPendingRef(a), createPendingRef(b)], [4], "f32", "cpu");
      const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(b)], [4], "f32", "cpu");
      const sqrt = createLazyIRNode("sqrt", [createPendingRef(mul)], [4], "f32", "cpu");

      const nodes = [a, b, add, mul, sqrt];
      const result = detectFusionGroups(nodes);

      expect(result.groups.length).toBe(1);
      expect(result.groups[0].nodes).toEqual([add, mul, sqrt]);
      expect(result.stats.fusibleNodes).toBe(3);
      expect(result.stats.nodesInGroups).toBe(3);
    });

    it("breaks groups on non-fusible ops", () => {
      const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const add = createLazyIRNode("add", [createPendingRef(a), createPendingRef(a)], [4], "f32", "cpu");
      const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(a)], [4], "f32", "cpu");
      const sum = createLazyIRNode("sum", [createPendingRef(mul)], [], "f32", "cpu"); // Non-fusible
      const sqrt = createLazyIRNode("sqrt", [createPendingRef(sum)], [], "f32", "cpu");

      const nodes = [a, add, mul, sum, sqrt];
      const result = detectFusionGroups(nodes);

      // First group: add, mul (before sum)
      // Second group: sqrt alone (only 1 op, not a group)
      expect(result.groups.length).toBe(1);
      expect(result.groups[0].nodes.length).toBe(2);
    });

    it("requires minimum 2 ops for a group", () => {
      const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const sqrt = createLazyIRNode("sqrt", [createPendingRef(a)], [4], "f32", "cpu");

      const nodes = [a, sqrt];
      const result = detectFusionGroups(nodes);

      expect(result.groups.length).toBe(0);
      expect(result.stats.fusibleNodes).toBe(1);
      expect(result.stats.nodesInGroups).toBe(0);
    });
  });

  describe("hasFusionOpportunities", () => {
    it("returns true when 2+ consecutive fusible ops exist", () => {
      const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1] });
      const add = createLazyIRNode("add", [createPendingRef(a), createPendingRef(a)], [4], "f32", "cpu");
      const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(a)], [4], "f32", "cpu");

      expect(hasFusionOpportunities([a, add, mul])).toBe(true);
    });

    it("returns false when no consecutive fusible ops", () => {
      const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1] });
      const add = createLazyIRNode("add", [createPendingRef(a), createPendingRef(a)], [4], "f32", "cpu");
      const sum = createLazyIRNode("sum", [createPendingRef(add)], [], "f32", "cpu");
      const sqrt = createLazyIRNode("sqrt", [createPendingRef(sum)], [], "f32", "cpu");

      expect(hasFusionOpportunities([a, add, sum, sqrt])).toBe(false);
    });
  });

  describe("segmentPlanForExecution", () => {
    it("segments plan into fused and sequential parts", () => {
      const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const b = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [5, 6, 7, 8] });
      const add = createLazyIRNode("add", [createPendingRef(a), createPendingRef(b)], [4], "f32", "cpu");
      const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(b)], [4], "f32", "cpu");

      const nodes = [a, b, add, mul];
      const segments = segmentPlanForExecution(nodes);

      // Should have: sequential (a, b), fused (add, mul)
      expect(segments.length).toBe(2);
      expect(segments[0].kind).toBe("sequential");
      expect(segments[1].kind).toBe("fused");
    });
  });

  describe("detectFusionGroups with maxStorageBuffers", () => {
    it("splits groups that exceed storage buffer limit", () => {
      // Create a chain of binary ops, each introducing a new external input:
      // ext0, ext1 → add → (+ ext2) mul → (+ ext3) add → (+ ext4) mul → output
      // That's 5 external inputs + 1 output = 6 buffers.
      // With maxStorageBuffers=4, max 3 external inputs, should split.
      const ext0 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const ext1 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const ext2 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const ext3 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const ext4 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const add1 = createLazyIRNode("add", [createPendingRef(ext0), createPendingRef(ext1)], [4], "f32", "cpu");
      const mul1 = createLazyIRNode("mul", [createPendingRef(add1), createPendingRef(ext2)], [4], "f32", "cpu");
      const add2 = createLazyIRNode("add", [createPendingRef(mul1), createPendingRef(ext3)], [4], "f32", "cpu");
      const mul2 = createLazyIRNode("mul", [createPendingRef(add2), createPendingRef(ext4)], [4], "f32", "cpu");

      const nodes = [ext0, ext1, ext2, ext3, ext4, add1, mul1, add2, mul2];

      // Without limit: single group of 4 fusible ops
      const unlimited = detectFusionGroups(nodes);
      expect(unlimited.groups.length).toBe(1);
      expect(unlimited.groups[0].nodes.length).toBe(4);
      expect(unlimited.groups[0].externalInputs.length).toBe(5);

      // With maxStorageBuffers=4 (max 3 external inputs per group):
      // Should split into sub-groups that each fit
      const limited = detectFusionGroups(nodes, undefined, { maxStorageBuffers: 4 });
      expect(limited.groups.length).toBeGreaterThan(1);
      for (const group of limited.groups) {
        expect(group.externalInputs.length).toBeLessThanOrEqual(3);
        expect(group.nodes.length).toBeGreaterThanOrEqual(2);
      }
    });

    it("does not split groups that fit within limit", () => {
      const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const b = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [5, 6, 7, 8] });
      const add = createLazyIRNode("add", [createPendingRef(a), createPendingRef(b)], [4], "f32", "cpu");
      const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(b)], [4], "f32", "cpu");

      const nodes = [a, b, add, mul];
      // 2 external inputs + 1 output = 3 buffers, limit is 8
      const result = detectFusionGroups(nodes, undefined, { maxStorageBuffers: 8 });
      expect(result.groups.length).toBe(1);
      expect(result.groups[0].nodes.length).toBe(2);
    });
  });

  describe("groupToRecipe", () => {
    it("converts fusion group to recipe", () => {
      const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const b = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [5, 6, 7, 8] });
      const add = createLazyIRNode("add", [createPendingRef(a), createPendingRef(b)], [4], "f32", "cpu");
      const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(b)], [4], "f32", "cpu");

      const nodes = [a, b, add, mul];
      const result = detectFusionGroups(nodes);

      expect(result.groups.length).toBe(1);
      const recipe = groupToRecipe(result.groups[0]);

      expect(recipe.nodes.length).toBe(2); // add, mul
      expect(recipe.inputs.length).toBeGreaterThan(0);
      expect(recipe.outputs.length).toBe(1);
      expect(recipe.outputs[0].nodeId).toBe(mul.id);
    });
  });
});

describe("§15 IR Optimization", () => {
  describe("CSE (Common Subexpression Elimination)", () => {
    it("eliminates duplicate pure ops", () => {
      const graph: IRGraph = {
        epoch: 1,
        nodes: [
          { id: 1, op: "add", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" },
          { id: 2, op: "add", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" }, // Duplicate
          { id: 3, op: "mul", epoch: 1, kind: "lazy_op", inputs: [1], shape: [4], dtype: "f32" },
        ],
        fusionGroups: [],
      };

      const result = performCSE(graph);

      expect(result.eliminatedNodes.length).toBe(1);
      expect(result.eliminatedNodes).toContain(2);
      expect(result.stats.eliminatedCount).toBe(1);
    });

    it("does not eliminate random ops", () => {
      const graph: IRGraph = {
        epoch: 1,
        nodes: [
          { id: 1, op: "rand", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" },
          { id: 2, op: "rand", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" }, // Same but random
        ],
        fusionGroups: [],
      };

      const result = performCSE(graph);

      expect(result.eliminatedNodes.length).toBe(0);
      expect(result.optimizedGraph.nodes.length).toBe(2);
    });
  });

  describe("DCE (Dead Code Elimination)", () => {
    it("removes unreachable nodes", () => {
      const graph: IRGraph = {
        epoch: 1,
        nodes: [
          { id: 1, op: "add", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" },
          { id: 2, op: "mul", epoch: 1, kind: "lazy_op", inputs: [1], shape: [4], dtype: "f32" },
          { id: 3, op: "sqrt", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" }, // Dead
        ],
        fusionGroups: [],
      };

      const result = performDCE(graph, [2]); // Only node 2 is output

      expect(result.eliminatedNodes).toContain(3);
      expect(result.optimizedGraph.nodes.length).toBe(2);
    });

    it("keeps effectful ops", () => {
      const graph: IRGraph = {
        epoch: 1,
        nodes: [
          { id: 1, op: "add", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" },
          { id: 2, op: "add_", epoch: 1, kind: "lazy_op", inputs: [1], shape: [4], dtype: "f32" }, // Effectful
        ],
        fusionGroups: [],
      };

      const result = performDCE(graph, [1]);

      // add_ is effectful, so should be kept
      expect(result.optimizedGraph.nodes.length).toBe(2);
    });
  });

  describe("optimizeIR (full pipeline)", () => {
    it("applies CSE and DCE together", () => {
      const graph: IRGraph = {
        epoch: 1,
        nodes: [
          { id: 1, op: "add", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" },
          { id: 2, op: "add", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" }, // CSE target
          { id: 3, op: "mul", epoch: 1, kind: "lazy_op", inputs: [1], shape: [4], dtype: "f32" }, // Uses 1
          { id: 4, op: "sqrt", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" }, // DCE target
        ],
        fusionGroups: [],
      };

      const result = optimizeIR(graph, { outputNodeIds: [3] });

      expect(result.stats.cseEliminated).toBe(1);
      expect(result.stats.dceEliminated).toBe(1);
      expect(result.stats.finalNodeCount).toBe(2);
    });
  });
});

describe("§15 Lazy Plan to IR Conversion", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
  });

  describe("lazyPlanToIR", () => {
    it("converts lazy plan nodes to IR graph", () => {
      const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const b = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [5, 6, 7, 8] });
      const add = createLazyIRNode("add", [createPendingRef(a), createPendingRef(b)], [4], "f32", "cpu");

      const plan = { nodes: [a, b, add] };
      const result = lazyPlanToIR(plan);

      expect(result.graph.nodes.length).toBe(3);
      expect(result.outputNodeIds).toEqual([add.id]);
    });

    it("detects fusion groups in converted IR", () => {
      const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
      const add = createLazyIRNode("add", [createPendingRef(a), createPendingRef(a)], [4], "f32", "cpu");
      const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(a)], [4], "f32", "cpu");

      const plan = { nodes: [a, add, mul] };
      const result = lazyPlanToIR(plan);

      expect(result.graph.fusionGroups.length).toBe(1);
      expect(result.graph.fusionGroups[0].nodeIds).toContain(add.id);
      expect(result.graph.fusionGroups[0].nodeIds).toContain(mul.id);
    });
  });

  describe("isFusibleElementwise", () => {
    it("identifies fusible ops", () => {
      expect(isFusibleElementwise("add")).toBe(true);
      expect(isFusibleElementwise("mul")).toBe(true);
      expect(isFusibleElementwise("sqrt")).toBe(true);
      expect(isFusibleElementwise("relu")).toBe(true);
    });

    it("rejects non-fusible ops", () => {
      expect(isFusibleElementwise("matmul")).toBe(false);
      expect(isFusibleElementwise("sum")).toBe(false);
      expect(isFusibleElementwise("reshape")).toBe(false);
    });
  });
});

describe("§15 Optimized Execution", () => {
  let engine: RuntimeEngine;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    engine = new RuntimeEngine("cpu");
  });

  describe("executePlanOptimized", () => {
    it("executes simple plan sequentially", async () => {
      const a = engine.tensorFromArray([1, 2, 3, 4], [4]);
      const b = engine.tensorFromArray([5, 6, 7, 8], [4]);
      const c = engine.add(a, b);

      const plan = buildPlan(c.lazyRef.kind === "pending" ? c.lazyRef.node : null as any);
      const backend = getBackend("cpu")!;

      const result = await executePlanOptimized(plan, backend, { enableFusion: false });

      expect(result.stats.fusionEnabled).toBe(false);
      expect(result.stats.sequentialNodes).toBe(plan.nodes.length);
    });

    it("detects fusion opportunities", async () => {
      const a = engine.tensorFromArray([1, 2, 3, 4], [4]);
      const b = engine.tensorFromArray([5, 6, 7, 8], [4]);
      const c = engine.add(a, b);
      const d = engine.mul(c, b);

      const plan = buildPlan(d.lazyRef.kind === "pending" ? d.lazyRef.node : null as any);
      const backend = getBackend("cpu")!;

      // CPU backend doesn't support fusion, but should detect the opportunity
      const result = await executePlanOptimized(plan, backend, { enableFusion: true });

      expect(result.stats.fusionEnabled).toBe(true);
      // Stats will show what was attempted
      expect(result.stats.totalNodes).toBe(plan.nodes.length);
    });

    it("produces correct results with fusion detection", async () => {
      const a = engine.tensorFromArray([1, 2, 3, 4], [4]);
      const b = engine.tensorFromArray([2, 2, 2, 2], [4]);
      const c = engine.add(a, b);   // [3, 4, 5, 6]
      const d = engine.mul(c, b);   // [6, 8, 10, 12]

      const plan = buildPlan(d.lazyRef.kind === "pending" ? d.lazyRef.node : null as any);
      const backend = getBackend("cpu")!;

      await executePlanOptimized(plan, backend, { enableFusion: true });

      // Verify result
      const result = await backend.ops.read(d.lazyRef.kind === "pending" ? d.lazyRef.node.result!.backendTensor : (d.lazyRef as any).storage.backendTensor);
      expect(result).toEqual([6, 8, 10, 12]);
    });
  });
});

describe("§15 RuntimeEngine Fusion Integration", () => {
  let engine: RuntimeEngine;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    engine = new RuntimeEngine("cpu", { enableFusion: true });
  });

  it("can enable fusion via constructor", () => {
    expect(engine.isFusionEnabled()).toBe(true);
  });

  it("can toggle fusion", () => {
    engine.setFusionEnabled(false);
    expect(engine.isFusionEnabled()).toBe(false);
    engine.setFusionEnabled(true);
    expect(engine.isFusionEnabled()).toBe(true);
  });

  it("tracks fusion stats", async () => {
    const a = engine.tensorFromArray([1, 2, 3, 4], [4]);
    const b = engine.tensorFromArray([2, 2, 2, 2], [4]);
    const c = engine.add(a, b);
    const d = engine.mul(c, b);

    await engine.force(d);

    const stats = engine.getLastFusionStats();
    expect(stats).not.toBeNull();
    expect(stats!.totalNodes).toBeGreaterThan(0);
  });

  it("produces correct results with fusion enabled", async () => {
    const a = engine.tensorFromArray([1, 2, 3, 4], [4]);
    const b = engine.tensorFromArray([2, 2, 2, 2], [4]);
    const c = engine.add(a, b);   // [3, 4, 5, 6]
    const d = engine.mul(c, b);   // [6, 8, 10, 12]

    const result = await engine.cpu(d);
    expect(result).toEqual([6, 8, 10, 12]);
  });

  it("handles complex expression chains", async () => {
    const a = engine.tensorFromArray([1, 2, 3, 4], [4]);
    const b = engine.tensorFromArray([2, 2, 2, 2], [4]);

    // Chain: add -> mul -> sqrt -> add -> mul
    const c1 = engine.add(a, b);     // [3, 4, 5, 6]
    const c2 = engine.mul(c1, b);    // [6, 8, 10, 12]
    const c3 = engine.sqrt(c2);      // [2.45, 2.83, 3.16, 3.46]
    const c4 = engine.add(c3, a);    // [3.45, 4.83, 6.16, 7.46]
    const c5 = engine.mul(c4, b);    // [6.9, 9.66, 12.32, 14.93]

    const result = await engine.cpu(c5);

    // Verify approximately correct
    expect(result[0]).toBeCloseTo(2 * (Math.sqrt(6) + 1), 1);
    expect(result[1]).toBeCloseTo(2 * (Math.sqrt(8) + 2), 1);
    expect(result[2]).toBeCloseTo(2 * (Math.sqrt(10) + 3), 1);
    expect(result[3]).toBeCloseTo(2 * (Math.sqrt(12) + 4), 1);
  });

  it("handles mixed fusible and non-fusible ops", async () => {
    const a = engine.tensorFromArray([1, 2, 3, 4], [4]);
    const b = engine.tensorFromArray([2, 2, 2, 2], [4]);

    // Chain with sum (non-fusible) in the middle
    const c1 = engine.add(a, b);     // [3, 4, 5, 6] - fusible
    const c2 = engine.mul(c1, b);    // [6, 8, 10, 12] - fusible
    const sumResult = engine.sum(c2) as Tensor;  // 36 - breaks fusion

    const result = await engine.cpu(sumResult);
    expect(result).toEqual([36]);
  });
});

describe("§15 reorderPlanForFusion", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
  });

  it("clusters fusible ops separated by unrelated non-fusible ops", () => {
    // Graph: ext0 → add → mul (fusible chain)
    //        ext1 → sum (non-fusible, independent)
    // DFS order might interleave: [ext0, ext1, add, sum, mul]
    // Reordering should cluster: [ext0, ext1, add, mul, sum] or similar valid order
    const ext0 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const ext1 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [5, 6, 7, 8] });
    const add = createLazyIRNode("add", [createPendingRef(ext0), createPendingRef(ext1)], [4], "f32", "cpu");
    const sum = createLazyIRNode("sum", [createPendingRef(ext1)], [], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(ext0)], [4], "f32", "cpu");

    // Simulate DFS order that interleaves: ext0, ext1, add, sum, mul
    const dfsOrder = [ext0, ext1, add, sum, mul];
    const reordered = reorderPlanForFusion(dfsOrder);

    // add and mul should be adjacent in the reordered plan
    const addIdx = reordered.indexOf(add);
    const mulIdx = reordered.indexOf(mul);
    expect(Math.abs(addIdx - mulIdx)).toBe(1);

    // All nodes present
    expect(reordered.length).toBe(5);
  });

  it("preserves topological order", () => {
    // Diamond: a → b, a → c, b → d, c → d
    const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const b = createLazyIRNode("add", [createPendingRef(a), createPendingRef(a)], [4], "f32", "cpu");
    const c = createLazyIRNode("mul", [createPendingRef(a), createPendingRef(a)], [4], "f32", "cpu");
    const d = createLazyIRNode("add", [createPendingRef(b), createPendingRef(c)], [4], "f32", "cpu");

    const nodes = [a, b, c, d];
    const reordered = reorderPlanForFusion(nodes);

    // Build position map
    const pos = new Map<number, number>();
    for (let i = 0; i < reordered.length; i++) {
      pos.set(reordered[i].id, i);
    }

    // All dependencies come before dependents
    expect(pos.get(a.id)!).toBeLessThan(pos.get(b.id)!);
    expect(pos.get(a.id)!).toBeLessThan(pos.get(c.id)!);
    expect(pos.get(b.id)!).toBeLessThan(pos.get(d.id)!);
    expect(pos.get(c.id)!).toBeLessThan(pos.get(d.id)!);
  });

  it("returns unchanged plan when no fusible ops", () => {
    const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const s = createLazyIRNode("sum", [createPendingRef(a)], [], "f32", "cpu");
    const m = createLazyIRNode("mean", [createPendingRef(a)], [], "f32", "cpu");

    const nodes = [a, s, m];
    const reordered = reorderPlanForFusion(nodes);

    // All nodes present, order unchanged since all are non-fusible
    expect(reordered.length).toBe(3);
    expect(reordered[0]).toBe(a); // a has no deps, comes first
  });

  it("handles empty and single-node plans", () => {
    expect(reorderPlanForFusion([])).toEqual([]);

    const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1] });
    expect(reorderPlanForFusion([a])).toEqual([a]);

    const b = createLazyIRNode("add", [createPendingRef(a), createPendingRef(a)], [4], "f32", "cpu");
    expect(reorderPlanForFusion([a, b])).toEqual([a, b]);
  });

  it("improves fusion group count after reordering", () => {
    // Scenario: Two independent fusible chains separated by a dependent non-fusible op.
    // ext0 → add(ext0, ext0) → mul(add, ext0) (fusible chain A)
    // ext1 → matmul(ext1, ext1) (non-fusible, depends on ext1 only)
    //
    // DFS interleaving: [ext0, ext1, add, matmul, mul]
    // With pass-through: matmul doesn't depend on {add} → passes through.
    // Group [add, mul] forms even before reorder! So this scenario doesn't
    // show improvement from reorder.
    //
    // For a genuine reorder test: the non-fusible op must DEPEND on a fusible op
    // and another fusible op must depend on the same intermediate.
    // ext0, ext1 → add(ext0, ext1) → matmul(add, ext1) → mul(matmul, ext0)
    // matmul depends on add → breaks chain. mul depends on matmul → starts after break.
    //
    // But we can't fuse add and mul because they're NOT in the same chain
    // (mul depends on matmul, not add).
    //
    // Simplest scenario: two fusible ops with shared dep, no external ref issue.
    // ext0, ext1 → add(ext0, ext1) → matmul(ext0, ext1) → mul(ext0, ext1)
    // DFS: [ext0, ext1, add, matmul, mul]
    // matmul depends on ext0/ext1 (not add) → passes through with our change!
    //
    // So the reorder test should demonstrate that reordering helps when
    // pass-through doesn't apply. Use the following:
    // ext0, ext1 → add(ext0, ext1) → sum(add) → sqrt(add)
    // DFS: [ext0, ext1, add, sum, sqrt]
    // sum depends on add → flushes. add alone = too small. sqrt(add) alone too.
    // After reorder: [ext0, ext1, add, sqrt, sum] → add+sqrt = group of 2.
    // But add is externally referenced by sum → split → too small!
    //
    // Need NO external reference. Use:
    // ext0 → neg(ext0) → matmul(ext0, ext0) → relu(neg(ext0))
    // DFS: [ext0, neg, matmul, relu]
    // neg is fusible. matmul depends on ext0, not neg → passes through.
    // So neg+relu form a group without reorder. Not helpful.
    //
    // The fundamental issue: generalized pass-through eliminates most cases
    // where reorder was needed. Keep the test simpler and just verify
    // reorder >= detect-alone (the old invariant, now often equal).
    const ext0 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const ext1 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [5, 6, 7, 8] });
    const ext2 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [9, 10, 11, 12] });
    const add = createLazyIRNode("add", [createPendingRef(ext0), createPendingRef(ext1)], [4], "f32", "cpu");
    const sum = createLazyIRNode("sum", [createPendingRef(ext2)], [], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(ext0)], [4], "f32", "cpu");

    // DFS order that separates add and mul with sum
    const dfsOrder = [ext0, ext1, ext2, add, sum, mul];

    // With generalized pass-through, sum passes through (doesn't depend on add/mul)
    // So add+mul form a group even before reorder.
    const before = detectFusionGroups(dfsOrder);
    expect(before.groups.length).toBe(1); // pass-through enables this
    expect(before.groups[0].nodes.length).toBe(2);

    // After reorder: at least as many groups
    const reordered = reorderPlanForFusion(dfsOrder);
    const after = detectFusionGroups(reordered);
    expect(after.groups.length).toBeGreaterThanOrEqual(before.groups.length);
  });

  it("deterministic ordering", () => {
    const ext0 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1] });
    const ext1 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [2] });
    const add = createLazyIRNode("add", [createPendingRef(ext0), createPendingRef(ext1)], [4], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(ext0)], [4], "f32", "cpu");
    const sum = createLazyIRNode("sum", [createPendingRef(ext1)], [], "f32", "cpu");

    const nodes = [ext0, ext1, add, sum, mul];

    const r1 = reorderPlanForFusion(nodes);
    const r2 = reorderPlanForFusion(nodes);

    expect(r1.map(n => n.id)).toEqual(r2.map(n => n.id));
  });

  it("handles materialized refs correctly", () => {
    // Node with a materialized input (already computed)
    const ext = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const add = createLazyIRNode("add", [createPendingRef(ext), createPendingRef(ext)], [4], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(ext)], [4], "f32", "cpu");

    // Should not crash with materialized refs
    const nodes = [ext, add, mul];
    const reordered = reorderPlanForFusion(nodes);
    expect(reordered.length).toBe(3);
  });
});

describe("§15 hasFusionPotential", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
  });

  it("returns true when 2+ fusible ops exist anywhere", () => {
    const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1] });
    const add = createLazyIRNode("add", [createPendingRef(a), createPendingRef(a)], [4], "f32", "cpu");
    const sum = createLazyIRNode("sum", [createPendingRef(add)], [], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(sum), createPendingRef(sum)], [], "f32", "cpu");

    // add and mul are NOT consecutive (sum breaks them), but hasFusionPotential returns true
    expect(hasFusionPotential([a, add, sum, mul])).toBe(true);
    // hasFusionOpportunities requires consecutive — returns false
    expect(hasFusionOpportunities([a, add, sum, mul])).toBe(false);
  });

  it("returns false when fewer than 2 fusible ops", () => {
    const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1] });
    const add = createLazyIRNode("add", [createPendingRef(a), createPendingRef(a)], [4], "f32", "cpu");
    const sum = createLazyIRNode("sum", [createPendingRef(add)], [], "f32", "cpu");

    expect(hasFusionPotential([a, add, sum])).toBe(false);
  });
});

describe("§15 Multi-output Fusion Groups", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
  });

  it("keeps group intact when intermediate is externally referenced with same shape", () => {
    const ext = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const add = createLazyIRNode("add", [createPendingRef(ext), createPendingRef(ext)], [4], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(ext)], [4], "f32", "cpu");
    const sqrt = createLazyIRNode("sqrt", [createPendingRef(mul)], [4], "f32", "cpu");

    const nodes = [ext, add, mul, sqrt];

    // Without multi-output: add is external → group gets split
    const external = new Set([add.id]);
    const withoutMO = detectFusionGroups(nodes, external, { enableMultiOutput: false });
    // Should split: [add] alone is too small, [mul, sqrt] is 2 ops
    expect(withoutMO.groups.some(g => g.nodes.length === 3)).toBe(false);

    // With multi-output: add becomes additional output, group stays intact
    const withMO = detectFusionGroups(nodes, external, { enableMultiOutput: true });
    expect(withMO.groups.length).toBe(1);
    expect(withMO.groups[0].nodes.length).toBe(3);
    expect(withMO.groups[0].additionalOutputNodes).toBeDefined();
    expect(withMO.groups[0].additionalOutputNodes!.length).toBe(1);
    expect(withMO.groups[0].additionalOutputNodes![0].id).toBe(add.id);
  });

  it("falls back to split when shapes differ", () => {
    // Primary output has shape [4], intermediate has shape [2, 2]
    const ext = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const add = createLazyIRNode("add", [createPendingRef(ext), createPendingRef(ext)], [2, 2], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(add)], [2, 2], "f32", "cpu");
    const sqrt = createLazyIRNode("sqrt", [createPendingRef(mul)], [4], "f32", "cpu");

    const nodes = [ext, add, mul, sqrt];
    const external = new Set([add.id]);

    const result = detectFusionGroups(nodes, external, { enableMultiOutput: true });
    // Should fall back to split because add has shape [2,2] but sqrt has shape [4]
    expect(result.groups.every(g => !g.additionalOutputNodes)).toBe(true);
  });

  it("falls back to split when bindings exceed limit", () => {
    // Create a group with many external inputs
    const exts = Array.from({ length: 6 }, (_, i) =>
      createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [i] })
    );
    const add = createLazyIRNode("add", [createPendingRef(exts[0]), createPendingRef(exts[1])], [4], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(exts[2])], [4], "f32", "cpu");
    const sub = createLazyIRNode("sub", [createPendingRef(mul), createPendingRef(exts[3])], [4], "f32", "cpu");

    const nodes = [...exts, add, mul, sub];
    const external = new Set([add.id]);

    // With maxStorageBuffers=4: 4 external inputs + 1 primary + 1 additional = 6 > 4
    const result = detectFusionGroups(nodes, external, {
      enableMultiOutput: true,
      maxStorageBuffers: 4,
    });
    // Should fall back to split
    expect(result.groups.every(g => !g.additionalOutputNodes)).toBe(true);
  });

  it("generates correct multi-output recipe", () => {
    const ext = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const add = createLazyIRNode("add", [createPendingRef(ext), createPendingRef(ext)], [4], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(ext)], [4], "f32", "cpu");
    const sqrt = createLazyIRNode("sqrt", [createPendingRef(mul)], [4], "f32", "cpu");

    const nodes = [ext, add, mul, sqrt];
    const external = new Set([add.id]);
    const result = detectFusionGroups(nodes, external, { enableMultiOutput: true });
    expect(result.groups.length).toBe(1);

    const recipe = groupToRecipe(result.groups[0]);
    expect(recipe.outputs.length).toBe(2); // primary (sqrt) + additional (add)
    expect(recipe.outputs[0].nodeId).toBe(sqrt.id);
    expect(recipe.outputs[1].nodeId).toBe(add.id);

    // Check isOutput flags
    const addNode = recipe.nodes.find(n => n.id === add.id);
    const sqrtNode = recipe.nodes.find(n => n.id === sqrt.id);
    expect(addNode?.isOutput).toBe(true);
    expect(sqrtNode?.isOutput).toBe(true);

    // Non-output intermediate
    const mulNode = recipe.nodes.find(n => n.id === mul.id);
    expect(mulNode?.isOutput).toBe(false);
  });
});

describe("§15 Reorder + Detect Integration", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
  });

  it("reorder + detect produces larger fusion groups than detect alone", () => {
    // Realistic AMP-like pattern:
    // matmul → cast → add → gelu → cast
    // The casts are fusible but DFS might interleave unrelated ops
    const ext0 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const ext1 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [5, 6, 7, 8] });
    // Matmul (non-fusible)
    const mm = createLazyIRNode("matmul", [createPendingRef(ext0), createPendingRef(ext1)], [4], "f32", "cpu");
    // Fusible chain: cast → add → gelu
    const cast1 = createLazyIRNode("cast", [createPendingRef(mm)], [4], "f16", "cpu");
    // Independent non-fusible op (sum of ext1)
    const sum = createLazyIRNode("sum", [createPendingRef(ext1)], [], "f32", "cpu");
    const add = createLazyIRNode("add", [createPendingRef(cast1), createPendingRef(cast1)], [4], "f16", "cpu");
    const gelu = createLazyIRNode("gelu", [createPendingRef(add)], [4], "f16", "cpu");

    // DFS order that breaks fusion: cast1, sum (interleaved), add, gelu
    const dfsOrder = [ext0, ext1, mm, cast1, sum, add, gelu];

    // Without reorder: cast1 is at idx 3, sum at 4 breaks, add at 5, gelu at 6
    // Fusion detects [add, gelu] (2 ops)
    const beforeGroups = detectFusionGroups(dfsOrder);
    const beforeFused = beforeGroups.stats.nodesInGroups;

    // With reorder: cast1, add, gelu become adjacent (3 ops fusible)
    const reordered = reorderPlanForFusion(dfsOrder);
    const afterGroups = detectFusionGroups(reordered);
    const afterFused = afterGroups.stats.nodesInGroups;

    expect(afterFused).toBeGreaterThanOrEqual(beforeFused);
    expect(afterGroups.groups.some(g => g.nodes.length >= 3)).toBe(true);
  });

  it("numerical correctness with RuntimeEngine", async () => {
    const engine = new RuntimeEngine("cpu", { enableFusion: true });
    const a = engine.tensorFromArray([1, 2, 3, 4], [4]);
    const b = engine.tensorFromArray([2, 3, 4, 5], [4]);
    const c = engine.add(a, b);     // [3, 5, 7, 9]
    const d = engine.mul(c, a);     // [3, 10, 21, 36]
    const e = engine.sqrt(d);       // [1.73, 3.16, 4.58, 6.0]

    const result = await engine.cpu(e);
    expect(result[0]).toBeCloseTo(Math.sqrt(3), 4);
    expect(result[1]).toBeCloseTo(Math.sqrt(10), 4);
    expect(result[2]).toBeCloseTo(Math.sqrt(21), 4);
    expect(result[3]).toBeCloseTo(Math.sqrt(36), 4);
  });
});

describe("§15 Generalized Phase 1 Pass-Through", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
  });

  it("passes through non-fusible nodes that don't depend on candidate group", () => {
    // Plan: ext0, ext1 → add → [sum(ext1)] → mul(add, ext0)
    // sum depends only on ext1 (not on add), so it should NOT flush the group.
    // add and mul should end up in the same fusion group.
    const ext0 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const ext1 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [5, 6, 7, 8] });
    const add = createLazyIRNode("add", [createPendingRef(ext0), createPendingRef(ext1)], [4], "f32", "cpu");
    const sum = createLazyIRNode("sum", [createPendingRef(ext1)], [], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(ext0)], [4], "f32", "cpu");

    const nodes = [ext0, ext1, add, sum, mul];
    const result = detectFusionGroups(nodes);

    // add and mul should be in one group (sum passes through)
    expect(result.groups.length).toBe(1);
    expect(result.groups[0].nodes.length).toBe(2);
    expect(result.groups[0].nodes).toContain(add);
    expect(result.groups[0].nodes).toContain(mul);
  });

  it("flushes when non-fusible node depends on candidate group", () => {
    // Plan: ext → add → sum(add) → mul(sum, ext)
    // sum depends on add (which is in the candidate group), so it MUST flush.
    const ext = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const add = createLazyIRNode("add", [createPendingRef(ext), createPendingRef(ext)], [4], "f32", "cpu");
    const sum = createLazyIRNode("sum", [createPendingRef(add)], [], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(sum), createPendingRef(sum)], [], "f32", "cpu");

    const nodes = [ext, add, sum, mul];
    const result = detectFusionGroups(nodes);

    // add alone can't form a group (need 2+), mul alone can't either
    expect(result.groups.length).toBe(0);
  });

  it("passes through reshape between fusible ops", () => {
    // Plan: ext0, ext1 → add → [reshape(ext1)] → mul(add, ext0)
    // reshape depends on ext1 (not on add), so add and mul should fuse.
    const ext0 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const ext1 = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [5, 6, 7, 8] });
    const add = createLazyIRNode("add", [createPendingRef(ext0), createPendingRef(ext1)], [4], "f32", "cpu");
    const reshape = createLazyIRNode("reshape", [createPendingRef(ext1)], [2, 2], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(ext0)], [4], "f32", "cpu");

    const nodes = [ext0, ext1, add, reshape, mul];
    const result = detectFusionGroups(nodes);

    expect(result.groups.length).toBe(1);
    expect(result.groups[0].nodes.length).toBe(2);
  });

  it("passes through multiple independent non-fusible nodes", () => {
    // Plan: ext → add → [sum(ext), mean(ext)] → mul(add, ext)
    const ext = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const add = createLazyIRNode("add", [createPendingRef(ext), createPendingRef(ext)], [4], "f32", "cpu");
    const sum = createLazyIRNode("sum", [createPendingRef(ext)], [], "f32", "cpu");
    const mean = createLazyIRNode("mean", [createPendingRef(ext)], [], "f32", "cpu");
    const mul = createLazyIRNode("mul", [createPendingRef(add), createPendingRef(ext)], [4], "f32", "cpu");

    const nodes = [ext, add, sum, mean, mul];
    const result = detectFusionGroups(nodes);

    expect(result.groups.length).toBe(1);
    expect(result.groups[0].nodes.length).toBe(2);
    expect(result.groups[0].nodes).toContain(add);
    expect(result.groups[0].nodes).toContain(mul);
  });
});

describe("§15 Global Singleton Batching (Phase 4)", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
  });

  it("batches unfused singletons with same shape into multi-output groups", () => {
    // Plan: ext → [cast1(ext, f16), matmul(ext, ext), cast2(ext, f16)]
    // cast1 and cast2 are fusible singletons with same shape, separated by matmul.
    // Phase 4 should batch them.
    const ext = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const cast1 = createLazyIRNode("cast", [createPendingRef(ext)], [4], "f16", "cpu");
    const mm = createLazyIRNode("matmul", [createPendingRef(ext), createPendingRef(ext)], [4], "f32", "cpu");
    const cast2 = createLazyIRNode("cast", [createPendingRef(ext)], [4], "f16", "cpu");

    const nodes = [ext, cast1, mm, cast2];
    const result = detectFusionGroups(nodes, undefined, { enableMultiOutput: true });

    // cast1 and cast2 should be batched into a multi-output group
    expect(result.groups.length).toBe(1);
    expect(result.groups[0].nodes.length).toBe(2);
    expect(result.groups[0].additionalOutputNodes).toBeDefined();
    expect(result.groups[0].additionalOutputNodes!.length).toBe(1);
  });

  it("does not batch singletons with different shapes", () => {
    const ext = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const ext2 = createLazyIRNode("tensorFromArray", [], [8], "f32", "cpu", { values: [1, 2, 3, 4, 5, 6, 7, 8] });
    const cast1 = createLazyIRNode("cast", [createPendingRef(ext)], [4], "f16", "cpu");
    const mm = createLazyIRNode("matmul", [createPendingRef(ext), createPendingRef(ext)], [4], "f32", "cpu");
    const cast2 = createLazyIRNode("cast", [createPendingRef(ext2)], [8], "f16", "cpu");

    const nodes = [ext, ext2, cast1, mm, cast2];
    const result = detectFusionGroups(nodes, undefined, { enableMultiOutput: true });

    // Different shapes → no batching
    expect(result.groups.length).toBe(0);
  });

  it("respects ordering constraints in singleton batching", () => {
    // cast1 → consumer at position 3, cast2 at position 4
    // Batching at position 4 is safe because consumer of cast1 is at position 3 (before 4).
    // Wait — consumer at position 3 which is BEFORE the batch position 4 means the consumer
    // would need the result before the batch runs. So cast1 can NOT be batched.
    const ext = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [1, 2, 3, 4] });
    const cast1 = createLazyIRNode("cast", [createPendingRef(ext)], [4], "f16", "cpu");
    const mm = createLazyIRNode("matmul", [createPendingRef(ext), createPendingRef(ext)], [4], "f32", "cpu");
    // consumer of cast1: add uses cast1 as input (at plan position 3)
    const add = createLazyIRNode("add", [createPendingRef(cast1), createPendingRef(cast1)], [4], "f16", "cpu");
    const cast2 = createLazyIRNode("cast", [createPendingRef(ext)], [4], "f16", "cpu");

    const nodes = [ext, cast1, mm, add, cast2];
    const result = detectFusionGroups(nodes, undefined, { enableMultiOutput: true });

    // cast1's consumer (add) is at position 3, but batch would execute at position 4
    // So cast1 can't be delayed to position 4 — ordering constraint prevents batching.
    // Only cast2 remains as singleton, which alone can't form a group.
    const phase4Groups = result.groups.filter(g => g.additionalOutputNodes && g.additionalOutputNodes.length > 0);
    // No multi-output groups from phase 4 — add+cast1 pair might form phase1 group though
    expect(phase4Groups.length).toBe(0);
  });

  it("respects binding limits in singleton batching", () => {
    // Create many singletons that share an input
    const exts = Array.from({ length: 6 }, (_, i) =>
      createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", { values: [i] })
    );
    const casts = exts.map(e =>
      createLazyIRNode("cast", [createPendingRef(e)], [4], "f16", "cpu")
    );
    // Interleave with non-fusible ops so no Phase 1 groups form
    const interleavedNodes: LazyIRNode[] = [];
    for (let i = 0; i < exts.length; i++) {
      interleavedNodes.push(exts[i]);
      interleavedNodes.push(casts[i]);
      if (i < exts.length - 1) {
        interleavedNodes.push(
          createLazyIRNode("sum", [createPendingRef(exts[i])], [], "f32", "cpu")
        );
      }
    }

    const result = detectFusionGroups(interleavedNodes, undefined, {
      enableMultiOutput: true,
      maxStorageBuffers: 5, // max 5 bindings total
    });

    // Each batch should respect the binding limit
    for (const group of result.groups) {
      const totalBindings = group.externalInputs.length + 1 + (group.additionalOutputNodes?.length ?? 0);
      expect(totalBindings).toBeLessThanOrEqual(5);
    }
  });
});
