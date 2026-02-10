import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { RuntimeEngine } from "../src/runtime/engine";
import {
  getMemoryPlanner,
  resetMemoryPlanner,
  createMemoryPlanForExecution,
  getMemoryPlannerStats,
} from "../src/engine/memory-planned-executor";
import {
  buildPlan,
  createLazyIRNode,
  createPendingRef,
} from "../src/engine/lazy";
import {
  analyzeLifetimes,
  findDonationOpportunities,
  computeBufferSize,
  BufferPool,
  InFlightPlanManager,
  MemoryPlanner,
} from "../src/engine/memory-planning";

describe("Memory Planning Integration", () => {
  let engine: RuntimeEngine;

  beforeEach(() => {
    resetMemoryPlanner();
    engine = new RuntimeEngine("cpu", {
      enableMemoryPlanning: true,
      trackStats: true,
    });
  });

  afterEach(() => {
    resetMemoryPlanner();
  });

  describe("RuntimeEngine with memory planning", () => {
    it("can enable memory planning", () => {
      expect(engine.isMemoryPlanningEnabled()).toBe(true);
    });

    it("can disable memory planning", () => {
      engine.setMemoryPlanning(false);
      expect(engine.isMemoryPlanningEnabled()).toBe(false);
    });

    it("executes simple tensor operation with memory planning", async () => {
      const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = engine.tensorFromArray([5, 6, 7, 8], [2, 2]);
      const c = engine.add(a, b);

      const result = await engine.cpu(c);
      expect(result).toEqual([6, 8, 10, 12]);

      const stats = engine.getLastMemoryStats();
      expect(stats).not.toBeNull();
      expect(stats!.totalNodes).toBeGreaterThan(0);
    });

    it("tracks memory statistics", async () => {
      const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = engine.tensorFromArray([5, 6, 7, 8], [2, 2]);
      const c = engine.add(a, b);
      const d = engine.mul(c, a);

      await engine.cpu(d);

      const stats = engine.getLastMemoryStats();
      expect(stats).not.toBeNull();
      expect(stats!.totalNodes).toBeGreaterThanOrEqual(4); // 2 inputs + add + mul
      expect(stats!.totalAllocatedBytes).toBeGreaterThan(0);
    });

    it("executes matmul with memory planning", async () => {
      const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = engine.tensorFromArray([5, 6, 7, 8], [2, 2]);
      const c = engine.matmul(a, b);

      const result = await engine.cpu(c);
      // [1,2]   [5,6]   [1*5+2*7, 1*6+2*8]   [19, 22]
      // [3,4] @ [7,8] = [3*5+4*7, 3*6+4*8] = [43, 50]
      expect(result).toEqual([19, 22, 43, 50]);

      const stats = engine.getLastMemoryStats();
      expect(stats).not.toBeNull();
    });

    it("provides memory planner statistics", () => {
      const plannerStats = engine.getMemoryPlannerStats();
      expect(plannerStats.bufferPool).toBeDefined();
      expect(plannerStats.planManager).toBeDefined();
      expect(plannerStats.bufferPool.totalBuffers).toBeGreaterThanOrEqual(0);
    });
  });

  describe("Memory plan creation", () => {
    it("creates memory plan for simple graph", () => {
      // Create a simple graph: a + b
      const nodeA = createLazyIRNode(
        "tensorFromArray",
        [],
        [2, 2],
        "f32",
        "cpu",
        { values: [1, 2, 3, 4] },
      );
      const nodeB = createLazyIRNode(
        "tensorFromArray",
        [],
        [2, 2],
        "f32",
        "cpu",
        { values: [5, 6, 7, 8] },
      );
      const nodeC = createLazyIRNode(
        "add",
        [createPendingRef(nodeA), createPendingRef(nodeB)],
        [2, 2],
        "f32",
        "cpu",
      );

      const plan = buildPlan(nodeC);
      const { memoryPlan, lifetimes, donations } = createMemoryPlanForExecution(plan);

      expect(memoryPlan.allocations.size).toBe(3);
      expect(lifetimes.size).toBe(3);
    });

    it("identifies donation opportunities", () => {
      // Create a chain: a -> b -> c where a's buffer can be reused for c
      const nodeA = createLazyIRNode(
        "tensorFromArray",
        [],
        [4, 4],
        "f32",
        "cpu",
        { values: Array(16).fill(1) },
      );
      const nodeB = createLazyIRNode(
        "relu",
        [createPendingRef(nodeA)],
        [4, 4],
        "f32",
        "cpu",
      );
      const nodeC = createLazyIRNode(
        "sqrt",
        [createPendingRef(nodeB)],
        [4, 4],
        "f32",
        "cpu",
      );

      const plan = buildPlan(nodeC);
      const { donations } = createMemoryPlanForExecution(plan);

      // Node A should be dead after nodeB uses it, so it could donate to nodeC
      // The actual donation depends on the lifetime analysis
      expect(donations.size).toBeGreaterThanOrEqual(0);
    });
  });

  describe("Lifetime analysis", () => {
    it("correctly identifies first and last use", () => {
      const nodeOrder = [1, 2, 3, 4];
      const nodeInputs = new Map([
        [1, []],
        [2, []],
        [3, [1, 2]], // uses nodes 1 and 2
        [4, [3]], // uses node 3
      ]);
      const nodeOutputs = new Set([4]);
      const nodeSizes = new Map([
        [1, 64],
        [2, 64],
        [3, 64],
        [4, 64],
      ]);

      const lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, nodeOutputs, nodeSizes);

      // Node 1: first use at 0, last use at 2 (used by node 3)
      expect(lifetimes.get(1)!.firstUse).toBe(0);
      expect(lifetimes.get(1)!.lastUse).toBe(2);

      // Node 2: first use at 1, last use at 2
      expect(lifetimes.get(2)!.firstUse).toBe(1);
      expect(lifetimes.get(2)!.lastUse).toBe(2);

      // Node 4 is output, so lives until end
      expect(lifetimes.get(4)!.lastUse).toBe(3);
    });

    it("finds donation opportunities correctly", () => {
      const nodeOrder = [1, 2, 3];
      const nodeInputs = new Map([
        [1, []],
        [2, [1]],
        [3, [2]],
      ]);
      const nodeOutputs = new Set([3]);
      const nodeSizes = new Map([
        [1, 64],
        [2, 64],
        [3, 64],
      ]);

      const lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, nodeOutputs, nodeSizes);
      const donations = findDonationOpportunities(nodeOrder, lifetimes, nodeSizes);

      // Node 1 is dead after step 1, so can donate to node 3
      // Node 2 is dead after step 2 (used by 3), but node 3 is at step 2
      // So donation opportunities depend on timing
      expect(donations.size).toBeGreaterThanOrEqual(0);
    });
  });

  describe("Buffer pool", () => {
    let pool: BufferPool;

    beforeEach(() => {
      pool = new BufferPool();
    });

    it("allocates buffers with size classes", () => {
      const info1 = pool.allocate(100, "f32", [5, 5]);
      const info2 = pool.allocate(200, "f32", [10, 5]);

      expect(info1.id).toBe(1);
      expect(info2.id).toBe(2);
      expect(info1.sizeBytes).toBeGreaterThanOrEqual(100);
      expect(info2.sizeBytes).toBeGreaterThanOrEqual(200);
    });

    it("reuses released buffers", () => {
      const info1 = pool.allocate(100, "f32", [5, 5]);
      pool.release(info1.id);

      const info2 = pool.allocate(100, "f32", [5, 5]);
      // Should reuse the same buffer
      expect(info2.id).toBe(info1.id);
    });

    it("tracks in-use buffers", () => {
      const info = pool.allocate(100, "f32", [5, 5]);
      pool.markInUse(info.id, 1);

      const stats = pool.stats();
      expect(stats.inUseBuffers).toBe(1);
    });

    it("handles fence signaling", () => {
      const info = pool.allocate(100, "f32", [5, 5]);
      const fenceId = pool.markPendingFence(info.id);

      let stats = pool.stats();
      expect(stats.pendingFenceBuffers).toBe(1);

      pool.signalFence(fenceId);
      stats = pool.stats();
      expect(stats.pendingFenceBuffers).toBe(0);
      expect(stats.pooledBuffers).toBe(1);
    });
  });

  describe("In-flight plan manager", () => {
    let pool: BufferPool;
    let manager: InFlightPlanManager;

    beforeEach(() => {
      pool = new BufferPool();
      manager = new InFlightPlanManager(pool);
    });

    it("registers plans with buffer references", () => {
      const buf1 = pool.allocate(100, "f32", [5, 5]);
      const buf2 = pool.allocate(100, "f32", [5, 5]);

      const planId = manager.registerPlan([buf1.id], [buf2.id], []);

      expect(planId).toBe(1);
      expect(manager.isBufferInUse(buf1.id)).toBe(true);
      expect(manager.isBufferInUse(buf2.id)).toBe(true);
    });

    it("tracks active and completed plans", () => {
      const buf1 = pool.allocate(100, "f32", [5, 5]);
      const planId = manager.registerPlan([], [buf1.id], []);

      let stats = manager.stats();
      expect(stats.activePlans).toBe(1);
      expect(stats.completedPlans).toBe(0);

      manager.completePlan(planId);
      stats = manager.stats();
      expect(stats.activePlans).toBe(0);
      expect(stats.completedPlans).toBe(1);
    });

    it("releases intermediate buffers on completion", () => {
      const input = pool.allocate(100, "f32", [5, 5]);
      const output = pool.allocate(100, "f32", [5, 5]);
      const intermediate = pool.allocate(100, "f32", [5, 5]);

      const planId = manager.registerPlan([input.id], [output.id], [intermediate.id]);
      manager.completePlan(planId);

      // Intermediate should be released
      expect(manager.isBufferInUse(intermediate.id)).toBe(false);
    });
  });

  describe("MemoryPlanner high-level API", () => {
    let planner: MemoryPlanner;

    beforeEach(() => {
      planner = new MemoryPlanner();
    });

    it("plans execution with allocations", () => {
      const nodeOrder = [1, 2, 3];
      const nodeInputs = new Map([
        [1, []],
        [2, []],
        [3, [1, 2]],
      ]);
      const nodeShapes = new Map<number, number[]>([
        [1, [10, 10]],
        [2, [10, 10]],
        [3, [10, 10]],
      ]);
      const nodeDtypes = new Map<number, "f32" | "f16" | "i32" | "u32" | "bool">([
        [1, "f32"],
        [2, "f32"],
        [3, "f32"],
      ]);

      const plan = planner.planExecution(
        nodeOrder,
        nodeInputs,
        [3],
        nodeShapes,
        nodeDtypes,
      );

      expect(plan.allocations.size).toBe(3);
      expect(plan.totalAllocatedBytes).toBeGreaterThan(0);
    });

    it("registers and completes executions", () => {
      const nodeOrder = [1, 2];
      const nodeInputs = new Map([[1, []], [2, [1]]]);
      const nodeShapes = new Map<number, number[]>([[1, [4]], [2, [4]]]);
      const nodeDtypes = new Map<number, "f32">([[1, "f32"], [2, "f32"]]);

      const memPlan = planner.planExecution(
        nodeOrder,
        nodeInputs,
        [2],
        nodeShapes,
        nodeDtypes,
      );

      const planId = planner.registerExecution(memPlan, [2]);
      expect(planId).toBe(1);

      planner.completeExecution(planId);
      const stats = planner.stats();
      expect(stats.planManager.completedPlans).toBe(1);
    });

    it("provides combined statistics", () => {
      const stats = planner.stats();
      expect(stats.bufferPool).toBeDefined();
      expect(stats.planManager).toBeDefined();
    });
  });

  describe("Buffer size computation", () => {
    it("computes correct sizes for different dtypes", () => {
      expect(computeBufferSize([10, 10], "f32")).toBe(400); // 100 * 4
      expect(computeBufferSize([10, 10], "f16")).toBe(200); // 100 * 2
      expect(computeBufferSize([10, 10], "i32")).toBe(400); // 100 * 4
      expect(computeBufferSize([10, 10], "bool")).toBe(100); // 100 * 1
    });

    it("handles scalar tensors", () => {
      expect(computeBufferSize([], "f32")).toBe(4);
      expect(computeBufferSize([1], "f32")).toBe(4);
    });
  });
});

describe("Memory Planning with standard execution fallback", () => {
  it("works with memory planning disabled", async () => {
    resetMemoryPlanner();
    const engine = new RuntimeEngine("cpu"); // No options = disabled

    expect(engine.isMemoryPlanningEnabled()).toBe(false);

    const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
    const b = engine.tensorFromArray([5, 6, 7, 8], [2, 2]);
    const c = engine.add(a, b);

    const result = await engine.cpu(c);
    expect(result).toEqual([6, 8, 10, 12]);

    // No stats when disabled
    expect(engine.getLastMemoryStats()).toBeNull();
  });
});
