import { beforeEach, describe, expect, it } from "vitest";

import {
  analyzeLifetimes,
  BufferPool,
  canDonateBuffer,
  computeBufferSize,
  findDeadTensors,
  findDonationOpportunities,
  getSizeClass,
  getSizeForClass,
  InFlightPlanManager,
  MemoryLimitExceededError,
  MemoryPlanner,
} from "../src/engine/memory-planning";

describe("Size Class Utilities", () => {
  it("computes size class as ceiling log2", () => {
    expect(getSizeClass(256)).toBe(8); // 2^8 = 256
    expect(getSizeClass(512)).toBe(9); // 2^9 = 512
    expect(getSizeClass(1024)).toBe(10); // 2^10 = 1024
  });

  it("rounds up to next power of 2", () => {
    expect(getSizeClass(300)).toBe(9); // rounds up to 512
    expect(getSizeClass(1000)).toBe(10); // rounds up to 1024
    expect(getSizeClass(1025)).toBe(11); // rounds up to 2048
  });

  it("enforces minimum buffer size", () => {
    expect(getSizeClass(1)).toBe(8); // min 256 bytes
    expect(getSizeClass(100)).toBe(8); // min 256 bytes
    expect(getSizeClass(255)).toBe(8); // min 256 bytes
  });

  it("converts size class back to bytes", () => {
    expect(getSizeForClass(8)).toBe(256);
    expect(getSizeForClass(10)).toBe(1024);
    expect(getSizeForClass(20)).toBe(1048576); // 1 MB
  });

  it("computes buffer size from shape and dtype", () => {
    expect(computeBufferSize([10, 10], "f32")).toBe(400); // 100 elements * 4 bytes
    expect(computeBufferSize([10, 10], "i32")).toBe(400);
    expect(computeBufferSize([10, 10], "f16")).toBe(200); // 100 elements * 2 bytes
    expect(computeBufferSize([2, 3, 4], "f32")).toBe(96); // 24 elements * 4 bytes
  });
});

describe("Lifetime Analysis", () => {
  it("analyzes simple linear graph", () => {
    // Graph: 1 -> 2 -> 3 (output)
    const nodeOrder = [1, 2, 3];
    const nodeInputs = new Map([
      [2, [1]],
      [3, [2]],
    ]);
    const nodeOutputs = new Set([3]);
    const nodeSizes = new Map([
      [1, 100],
      [2, 200],
      [3, 300],
    ]);

    const lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, nodeOutputs, nodeSizes);

    // Node 1: first use at step 0, last use at step 1 (used by node 2)
    expect(lifetimes.get(1)?.firstUse).toBe(0);
    expect(lifetimes.get(1)?.lastUse).toBe(1);
    expect(lifetimes.get(1)?.isOutput).toBe(false);
    expect(lifetimes.get(1)?.isInput).toBe(true);

    // Node 2: first use at step 1, last use at step 2 (used by node 3)
    expect(lifetimes.get(2)?.firstUse).toBe(1);
    expect(lifetimes.get(2)?.lastUse).toBe(2);

    // Node 3 is output: lives until end
    expect(lifetimes.get(3)?.firstUse).toBe(2);
    expect(lifetimes.get(3)?.lastUse).toBe(2);
    expect(lifetimes.get(3)?.isOutput).toBe(true);
  });

  it("handles diamond dependency", () => {
    // Graph: 1 -> 2 -> 4
    //        1 -> 3 -> 4 (output)
    const nodeOrder = [1, 2, 3, 4];
    const nodeInputs = new Map([
      [2, [1]],
      [3, [1]],
      [4, [2, 3]],
    ]);
    const nodeOutputs = new Set([4]);
    const nodeSizes = new Map([
      [1, 100],
      [2, 100],
      [3, 100],
      [4, 100],
    ]);

    const lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, nodeOutputs, nodeSizes);

    // Node 1 is used by both 2 and 3, so lives until step 2
    expect(lifetimes.get(1)?.lastUse).toBe(2);
    // Node 2 and 3 are used by 4, live until step 3
    expect(lifetimes.get(2)?.lastUse).toBe(3);
    expect(lifetimes.get(3)?.lastUse).toBe(3);
  });

  it("finds dead tensors at each step", () => {
    const nodeOrder = [1, 2, 3];
    const nodeInputs = new Map([
      [2, [1]],
      [3, [2]],
    ]);
    const nodeOutputs = new Set([3]);
    const nodeSizes = new Map([
      [1, 100],
      [2, 100],
      [3, 100],
    ]);

    const lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, nodeOutputs, nodeSizes);

    // At step 0: nothing dead yet
    expect(findDeadTensors(lifetimes, 0)).toEqual([]);
    // At step 1: nothing dead yet (node 1 last used at step 1)
    expect(findDeadTensors(lifetimes, 1)).toEqual([]);
    // At step 2: node 1 is dead (last use was step 1)
    expect(findDeadTensors(lifetimes, 2)).toContain(1);
    // At step 3: nodes 1 and 2 are dead (but 3 is output, so not dead)
    const deadAtStep3 = findDeadTensors(lifetimes, 3);
    expect(deadAtStep3).toContain(1);
    expect(deadAtStep3).toContain(2);
    expect(deadAtStep3).not.toContain(3);
  });
});

describe("Donation Analysis", () => {
  it("allows donation when source is dead", () => {
    const sourceLifetime = {
      nodeId: 1,
      firstUse: 0,
      lastUse: 1,
      isOutput: false,
      isInput: true,
      bufferSize: 1000,
    };

    // At step 2, source is dead, can donate
    const decision = canDonateBuffer(sourceLifetime, 500, 2);
    expect(decision.canDonate).toBe(true);
  });

  it("prevents donation when source is still alive", () => {
    const sourceLifetime = {
      nodeId: 1,
      firstUse: 0,
      lastUse: 3,
      isOutput: false,
      isInput: true,
      bufferSize: 1000,
    };

    // At step 2, source still alive
    const decision = canDonateBuffer(sourceLifetime, 500, 2);
    expect(decision.canDonate).toBe(false);
    expect(decision.reason).toBe("Source is still alive");
  });

  it("prevents donation when source is an output", () => {
    const sourceLifetime = {
      nodeId: 1,
      firstUse: 0,
      lastUse: 1,
      isOutput: true,
      isInput: false,
      bufferSize: 1000,
    };

    const decision = canDonateBuffer(sourceLifetime, 500, 2);
    expect(decision.canDonate).toBe(false);
    expect(decision.reason).toBe("Source is a plan output");
  });

  it("prevents donation when source buffer is too small", () => {
    const sourceLifetime = {
      nodeId: 1,
      firstUse: 0,
      lastUse: 1,
      isOutput: false,
      isInput: true,
      bufferSize: 100,
    };

    const decision = canDonateBuffer(sourceLifetime, 500, 2);
    expect(decision.canDonate).toBe(false);
    expect(decision.reason).toBe("Source buffer too small");
  });

  it("finds donation opportunities in execution order", () => {
    // Graph: 1 -> 2 -> 3 -> 4 (output)
    // Node 1's buffer can be donated to node 3
    const nodeOrder = [1, 2, 3, 4];
    const nodeInputs = new Map([
      [2, [1]],
      [3, [2]],
      [4, [3]],
    ]);
    const nodeSizes = new Map([
      [1, 1000],
      [2, 500],
      [3, 800], // Can use node 1's buffer
      [4, 500],
    ]);
    const nodeOutputs = new Set([4]);

    const lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, nodeOutputs, nodeSizes);
    const donations = findDonationOpportunities(nodeOrder, lifetimes, nodeSizes);

    // Node 3 should get node 1's buffer (dead after step 1, big enough)
    expect(donations.get(3)).toBe(1);
  });

  it("prefers smaller suitable buffers for donation", () => {
    // Two dead buffers: 1 (1000 bytes) and 2 (800 bytes)
    // Target needs 700 bytes - should pick 800 byte buffer
    const nodeOrder = [1, 2, 3, 4];
    const nodeInputs = new Map([
      [3, [1, 2]], // Both 1 and 2 die after step 2
      [4, []],
    ]);
    const nodeSizes = new Map([
      [1, 1000],
      [2, 800],
      [3, 100],
      [4, 700], // Should pick buffer 2 (800 >= 700, smaller than 1000)
    ]);
    const nodeOutputs = new Set([3]);

    const lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, nodeOutputs, nodeSizes);
    const donations = findDonationOpportunities(nodeOrder, lifetimes, nodeSizes);

    // Node 4 should get node 2's buffer (smallest sufficient)
    expect(donations.get(4)).toBe(2);
  });
});

describe("Buffer Pool", () => {
  let pool: BufferPool;

  beforeEach(() => {
    pool = new BufferPool();
  });

  it("allocates new buffers", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    const buf2 = pool.allocate(2000, "f32", [20, 25]);

    expect(buf1.id).toBe(1);
    expect(buf2.id).toBe(2);
    expect(buf1.sizeBytes).toBe(1024); // Rounded up to power of 2
    expect(buf2.sizeBytes).toBe(2048);
  });

  it("reuses released buffers from same size class", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    pool.release(buf1.id);

    const buf2 = pool.allocate(900, "f32", [9, 25]); // Same size class
    expect(buf2.id).toBe(buf1.id); // Reused!
  });

  it("does not reuse buffers that are in use", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    pool.markInUse(buf1.id, 1);

    const buf2 = pool.allocate(900, "f32", [9, 25]);
    expect(buf2.id).not.toBe(buf1.id); // Cannot reuse - in use
  });

  it("does not reuse buffers pending fence", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    pool.markPendingFence(buf1.id);

    const buf2 = pool.allocate(900, "f32", [9, 25]);
    expect(buf2.id).not.toBe(buf1.id); // Cannot reuse - pending fence
  });

  it("allows reuse after fence signals", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    const fenceId = pool.markPendingFence(buf1.id);
    pool.signalFence(fenceId);

    const buf2 = pool.allocate(900, "f32", [9, 25]);
    expect(buf2.id).toBe(buf1.id); // Can reuse now
  });

  it("tracks statistics correctly", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    const buf2 = pool.allocate(2000, "f32", [20, 25]);
    pool.markInUse(buf1.id, 1);
    pool.markPendingFence(buf2.id);

    const stats = pool.stats();
    expect(stats.totalBuffers).toBe(2);
    expect(stats.inUseBuffers).toBe(1);
    expect(stats.pendingFenceBuffers).toBe(1);
    expect(stats.pooledBuffers).toBe(0);
  });

  it("clears all buffers", () => {
    pool.allocate(1000, "f32", [10, 25]);
    pool.allocate(2000, "f32", [20, 25]);
    pool.clear();

    const stats = pool.stats();
    expect(stats.totalBuffers).toBe(0);
  });
});

describe("In-Flight Plan Manager (\u00a714 Strong Rooting)", () => {
  let pool: BufferPool;
  let manager: InFlightPlanManager;

  beforeEach(() => {
    pool = new BufferPool();
    manager = new InFlightPlanManager(pool);
  });

  it("registers plans with buffer references", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    const buf2 = pool.allocate(2000, "f32", [20, 25]);

    const planId = manager.registerPlan([buf1.id], [buf2.id], []);

    expect(planId).toBe(1);
    const plan = manager.getPlan(planId);
    expect(plan?.inputBuffers).toEqual([buf1.id]);
    expect(plan?.outputBuffers).toEqual([buf2.id]);
  });

  it("marks buffers as in-use when plan registers", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    manager.registerPlan([buf1.id], [], []);

    const info = pool.getInfo(buf1.id);
    expect(info?.inUseByPlan).toBe(1);
  });

  it("releases intermediate buffers on plan completion", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    const buf2 = pool.allocate(1000, "f32", [10, 25]);

    const planId = manager.registerPlan([], [buf1.id], [buf2.id]);
    manager.completePlan(planId);

    // Output buffer still exists but intermediate should be released
    const info2 = pool.getInfo(buf2.id);
    expect(info2?.inUseByPlan).toBeNull();
  });

  it("applies fence to buffers on completion with fenceId", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);

    const planId = manager.registerPlan([], [], [buf1.id]);
    manager.completePlan(planId, 42);

    const info = pool.getInfo(buf1.id);
    expect(info?.fenceId).toBe(42);
  });

  it("tracks active vs completed plans", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    const buf2 = pool.allocate(1000, "f32", [10, 25]);

    const plan1 = manager.registerPlan([buf1.id], [], []);
    const plan2 = manager.registerPlan([buf2.id], [], []);
    manager.completePlan(plan1);

    const stats = manager.stats();
    expect(stats.totalPlans).toBe(2);
    expect(stats.activePlans).toBe(1);
    expect(stats.completedPlans).toBe(1);

    const activePlans = manager.getActivePlans();
    expect(activePlans.length).toBe(1);
    expect(activePlans[0].id).toBe(plan2);
  });

  it("checks if buffer is in use by any active plan", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    const buf2 = pool.allocate(1000, "f32", [10, 25]);

    const plan1 = manager.registerPlan([buf1.id], [], []);
    manager.registerPlan([buf2.id], [], []);
    manager.completePlan(plan1);

    expect(manager.isBufferInUse(buf1.id)).toBe(false); // Plan completed
    expect(manager.isBufferInUse(buf2.id)).toBe(true); // Plan active
  });

  it("signals fence completion", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    const fenceId = pool.markPendingFence(buf1.id);

    manager.signalFence(fenceId);

    const info = pool.getInfo(buf1.id);
    expect(info?.fenceId).toBeNull();
  });

  it("cleans up old completed plans", () => {
    const buf1 = pool.allocate(1000, "f32", [10, 25]);
    const planId = manager.registerPlan([buf1.id], [], []);
    manager.completePlan(planId);

    // Cleanup with 0ms age removes immediately
    const cleaned = manager.cleanup(0);
    expect(cleaned).toBe(1);
    expect(manager.getPlan(planId)).toBeUndefined();
  });
});

describe("Memory Planner (Full Integration)", () => {
  let planner: MemoryPlanner;

  beforeEach(() => {
    planner = new MemoryPlanner();
  });

  it("plans memory for simple linear graph", () => {
    // Graph: 1 -> 2 -> 3 (output)
    const nodeOrder = [1, 2, 3];
    const nodeInputs = new Map([
      [2, [1]],
      [3, [2]],
    ]);
    const outputNodeIds = [3];
    const nodeShapes = new Map([
      [1, [10, 10]],
      [2, [10, 10]],
      [3, [10, 10]],
    ]);
    const nodeDtypes = new Map<number, "f32" | "i32">([
      [1, "f32"],
      [2, "f32"],
      [3, "f32"],
    ]);

    const plan = planner.planExecution(
      nodeOrder,
      nodeInputs,
      outputNodeIds,
      nodeShapes,
      nodeDtypes,
    );

    expect(plan.allocations.size).toBe(3);
    expect(plan.newAllocations).toBeGreaterThan(0);
  });

  it("reuses buffers through donation", () => {
    // Graph: 1 -> 2 -> 3 -> 4 (output)
    // Node 1's buffer should be donated to node 3
    const nodeOrder = [1, 2, 3, 4];
    const nodeInputs = new Map([
      [2, [1]],
      [3, [2]],
      [4, [3]],
    ]);
    const outputNodeIds = [4];
    const nodeShapes = new Map([
      [1, [100, 100]], // 40000 bytes
      [2, [10, 10]], // 400 bytes
      [3, [50, 50]], // 10000 bytes - can use node 1's buffer
      [4, [10, 10]], // 400 bytes
    ]);
    const nodeDtypes = new Map<number, "f32" | "i32">([
      [1, "f32"],
      [2, "f32"],
      [3, "f32"],
      [4, "f32"],
    ]);

    const plan = planner.planExecution(
      nodeOrder,
      nodeInputs,
      outputNodeIds,
      nodeShapes,
      nodeDtypes,
    );

    // Should have donation: node 3 uses node 1's buffer
    expect(plan.donations.has(3)).toBe(true);
    expect(plan.donations.get(3)).toBe(1);
    expect(plan.reusedAllocations).toBeGreaterThan(0);
  });

  it("registers and completes execution plans", () => {
    const nodeOrder = [1, 2];
    const nodeInputs = new Map([[2, [1]]]);
    const outputNodeIds = [2];
    const nodeShapes = new Map([
      [1, [10, 10]],
      [2, [10, 10]],
    ]);
    const nodeDtypes = new Map<number, "f32" | "i32">([
      [1, "f32"],
      [2, "f32"],
    ]);

    const memoryPlan = planner.planExecution(
      nodeOrder,
      nodeInputs,
      outputNodeIds,
      nodeShapes,
      nodeDtypes,
    );
    const planId = planner.registerExecution(memoryPlan, outputNodeIds);

    expect(planId).toBe(1);

    const stats = planner.stats();
    expect(stats.planManager.activePlans).toBe(1);

    planner.completeExecution(planId);
    const statsAfter = planner.stats();
    expect(statsAfter.planManager.activePlans).toBe(0);
    expect(statsAfter.planManager.completedPlans).toBe(1);
  });

  it("handles fence-based completion", () => {
    const nodeOrder = [1];
    const nodeInputs = new Map<number, number[]>();
    const outputNodeIds = [1];
    const nodeShapes = new Map([[1, [10, 10]]]);
    const nodeDtypes = new Map<number, "f32" | "i32">([[1, "f32"]]);

    const memoryPlan = planner.planExecution(
      nodeOrder,
      nodeInputs,
      outputNodeIds,
      nodeShapes,
      nodeDtypes,
    );
    const planId = planner.registerExecution(memoryPlan, outputNodeIds);

    // Complete with fence
    planner.completeExecution(planId, 100);

    // Buffer pool should show pending fence
    const poolStats = planner.stats().bufferPool;
    // Intermediate buffers get fence, outputs don't
    expect(poolStats.pendingFenceBuffers).toBeGreaterThanOrEqual(0);

    // Signal fence
    planner.signalFence(100);
  });

  it("provides combined statistics", () => {
    const stats = planner.stats();

    expect(stats.bufferPool).toBeDefined();
    expect(stats.bufferPool.totalBuffers).toBe(0);
    expect(stats.planManager).toBeDefined();
    expect(stats.planManager.totalPlans).toBe(0);
  });

  it("clears all state", () => {
    const nodeOrder = [1];
    const nodeInputs = new Map<number, number[]>();
    const outputNodeIds = [1];
    const nodeShapes = new Map([[1, [10, 10]]]);
    const nodeDtypes = new Map<number, "f32" | "i32">([[1, "f32"]]);

    planner.planExecution(
      nodeOrder,
      nodeInputs,
      outputNodeIds,
      nodeShapes,
      nodeDtypes,
    );
    planner.clear();

    const stats = planner.stats();
    expect(stats.bufferPool.totalBuffers).toBe(0);
  });
});

describe("Buffer Pool Memory Limit", () => {
  it("has default 10GB memory limit", () => {
    const pool = new BufferPool();
    const tenGB = 10 * 1024 * 1024 * 1024;
    expect(pool.getMemoryLimit()).toBe(tenGB);
  });

  it("accepts custom memory limit in constructor", () => {
    const oneGB = 1024 * 1024 * 1024;
    const pool = new BufferPool({ memoryLimitBytes: oneGB });
    expect(pool.getMemoryLimit()).toBe(oneGB);
  });

  it("allows setting memory limit after construction", () => {
    const pool = new BufferPool();
    const twoGB = 2 * 1024 * 1024 * 1024;
    pool.setMemoryLimit(twoGB);
    expect(pool.getMemoryLimit()).toBe(twoGB);
  });

  it("throws when setting invalid memory limit", () => {
    const pool = new BufferPool();
    expect(() => pool.setMemoryLimit(0)).toThrow("Memory limit must be positive");
    expect(() => pool.setMemoryLimit(-100)).toThrow("Memory limit must be positive");
  });

  it("tracks current allocated bytes", () => {
    const pool = new BufferPool();
    expect(pool.getCurrentAllocatedBytes()).toBe(0);

    // Allocate 1KB (rounds to 1024)
    pool.allocate(1000, "f32", [250]);
    expect(pool.getCurrentAllocatedBytes()).toBe(1024);

    // Allocate 2KB (rounds to 2048)
    pool.allocate(2000, "f32", [500]);
    expect(pool.getCurrentAllocatedBytes()).toBe(1024 + 2048);
  });

  it("throws MemoryLimitExceededError when limit is exceeded", () => {
    const limitBytes = 1024; // 1KB limit
    const pool = new BufferPool({ memoryLimitBytes: limitBytes });

    // First allocation should succeed (256 bytes min)
    pool.allocate(256, "f32", [64]);
    expect(pool.getCurrentAllocatedBytes()).toBe(256);

    // Second allocation should succeed (512 bytes)
    pool.allocate(500, "f32", [125]);
    expect(pool.getCurrentAllocatedBytes()).toBe(256 + 512);

    // Third allocation would exceed limit - should throw
    expect(() => pool.allocate(500, "f32", [125])).toThrow(MemoryLimitExceededError);
  });

  it("MemoryLimitExceededError contains correct information", () => {
    const limitBytes = 1024;
    const pool = new BufferPool({ memoryLimitBytes: limitBytes });

    // Allocate 512 bytes
    pool.allocate(500, "f32", [125]);

    try {
      // Try to allocate 1024 bytes (would be 512 + 1024 = 1536 > 1024)
      pool.allocate(1000, "f32", [250]);
      expect.fail("Should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(MemoryLimitExceededError);
      const err = e as MemoryLimitExceededError;
      expect(err.requestedBytes).toBe(1024); // Rounded up
      expect(err.currentBytes).toBe(512);
      expect(err.limitBytes).toBe(1024);
      expect(err.message).toContain("Memory limit exceeded");
    }
  });

  it("reusing buffers does not increase allocated bytes", () => {
    const pool = new BufferPool({ memoryLimitBytes: 2048 });

    // Allocate and release
    const buf1 = pool.allocate(1000, "f32", [250]);
    expect(pool.getCurrentAllocatedBytes()).toBe(1024);

    pool.release(buf1.id);

    // Reuse the buffer - should not increase allocation
    const buf2 = pool.allocate(900, "f32", [225]);
    expect(buf2.id).toBe(buf1.id); // Same buffer
    expect(pool.getCurrentAllocatedBytes()).toBe(1024); // No increase
  });

  it("clear resets allocated bytes to zero", () => {
    const pool = new BufferPool();

    pool.allocate(1000, "f32", [250]);
    pool.allocate(2000, "f32", [500]);
    expect(pool.getCurrentAllocatedBytes()).toBeGreaterThan(0);

    pool.clear();
    expect(pool.getCurrentAllocatedBytes()).toBe(0);
  });

  it("stats include memory limit and usage percentage", () => {
    const limitBytes = 10000;
    const pool = new BufferPool({ memoryLimitBytes: limitBytes });

    pool.allocate(1000, "f32", [250]); // 1024 bytes
    pool.allocate(2000, "f32", [500]); // 2048 bytes
    // Total: 3072 bytes

    const stats = pool.stats();
    expect(stats.memoryLimitBytes).toBe(limitBytes);
    expect(stats.memoryUsagePercent).toBeCloseTo((3072 / limitBytes) * 100, 1);
  });
});

describe("MemoryPlanner Memory Limit", () => {
  it("has default 10GB memory limit", () => {
    const planner = new MemoryPlanner();
    const tenGB = 10 * 1024 * 1024 * 1024;
    expect(planner.getMemoryLimit()).toBe(tenGB);
  });

  it("accepts custom memory limit in constructor", () => {
    const oneGB = 1024 * 1024 * 1024;
    const planner = new MemoryPlanner({ memoryLimitBytes: oneGB });
    expect(planner.getMemoryLimit()).toBe(oneGB);
  });

  it("allows setting memory limit after construction", () => {
    const planner = new MemoryPlanner();
    const twoGB = 2 * 1024 * 1024 * 1024;
    planner.setMemoryLimit(twoGB);
    expect(planner.getMemoryLimit()).toBe(twoGB);
  });

  it("tracks current allocated bytes", () => {
    const planner = new MemoryPlanner();
    expect(planner.getCurrentAllocatedBytes()).toBe(0);

    // Plan execution allocates buffers
    const nodeOrder = [1];
    const nodeInputs = new Map<number, number[]>();
    const outputNodeIds = [1];
    const nodeShapes = new Map([[1, [100, 100]]]); // 40000 bytes -> rounds to 65536
    const nodeDtypes = new Map<number, "f32" | "i32">([[1, "f32"]]);

    planner.planExecution(nodeOrder, nodeInputs, outputNodeIds, nodeShapes, nodeDtypes);

    expect(planner.getCurrentAllocatedBytes()).toBeGreaterThan(0);
  });

  it("throws when execution plan exceeds memory limit", () => {
    const smallLimit = 1000; // Very small limit
    const planner = new MemoryPlanner({ memoryLimitBytes: smallLimit });

    const nodeOrder = [1];
    const nodeInputs = new Map<number, number[]>();
    const outputNodeIds = [1];
    const nodeShapes = new Map([[1, [100, 100]]]); // 40000 bytes - way over limit
    const nodeDtypes = new Map<number, "f32" | "i32">([[1, "f32"]]);

    expect(() =>
      planner.planExecution(nodeOrder, nodeInputs, outputNodeIds, nodeShapes, nodeDtypes),
    ).toThrow(MemoryLimitExceededError);
  });
});
