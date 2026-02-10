/**
 * Engine Visibility Tools Tests
 *
 * Tests for the engine state visibility tools that help debug
 * memory usage, saved tensors, and base state.
 */

import { describe, expect, it, beforeEach } from "vitest";
import {
  Engine,
  type EngineMemoryStats,
  type SavedTensorInfo,
  type BaseStateInfo,
  type MemoryStatsProvider,
} from "../src/engine/engine";
import { TraceRecorder } from "../src/engine/trace";

describe("Engine Visibility Tools", () => {
  let engine: Engine;
  let trace: TraceRecorder;

  beforeEach(() => {
    trace = new TraceRecorder();
    engine = new Engine(trace);
  });

  describe("Memory Stats", () => {
    it("returns default stats when no provider is set", () => {
      const stats = engine._debug_getMemoryStats();

      expect(stats.gpuCurrentBytes).toBe(0);
      expect(stats.gpuPeakBytes).toBe(0);
      expect(stats.gpuLimitBytes).toBe(0);
      expect(stats.pooledBuffers).toBe(0);
      expect(stats.inUseBuffers).toBe(0);
      expect(stats.pendingFenceBuffers).toBe(0);
      expect(stats.activeBases).toBe(0);
      expect(stats.totalPinCount).toBe(0);
      expect(stats.savedTensorCount).toBe(0);
      expect(stats.pendingTensorCount).toBe(0);
      expect(stats.activePlans).toBe(0);
      expect(stats.completedPlans).toBe(0);
    });

    it("uses stats from provider when set", () => {
      const mockProvider: MemoryStatsProvider = {
        getGPUStats: () => ({
          currentBytes: 1000,
          peakBytes: 2000,
          limitBytes: 10000,
        }),
        getBufferPoolStats: () => ({
          pooledBuffers: 5,
          inUseBuffers: 3,
          pendingFenceBuffers: 2,
        }),
        getPlanStats: () => ({
          activePlans: 1,
          completedPlans: 10,
        }),
        getPendingTensorCount: () => 7,
      };

      engine.setMemoryStatsProvider(mockProvider);
      const stats = engine._debug_getMemoryStats();

      expect(stats.gpuCurrentBytes).toBe(1000);
      expect(stats.gpuPeakBytes).toBe(2000);
      expect(stats.gpuLimitBytes).toBe(10000);
      expect(stats.pooledBuffers).toBe(5);
      expect(stats.inUseBuffers).toBe(3);
      expect(stats.pendingFenceBuffers).toBe(2);
      expect(stats.activePlans).toBe(1);
      expect(stats.completedPlans).toBe(10);
      expect(stats.pendingTensorCount).toBe(7);
    });

    it("handles partial provider (some methods missing)", () => {
      const partialProvider: MemoryStatsProvider = {
        getGPUStats: () => ({
          currentBytes: 500,
          peakBytes: 600,
          limitBytes: 5000,
        }),
        // No buffer pool or plan stats
      };

      engine.setMemoryStatsProvider(partialProvider);
      const stats = engine._debug_getMemoryStats();

      expect(stats.gpuCurrentBytes).toBe(500);
      expect(stats.pooledBuffers).toBe(0); // Falls back to 0
      expect(stats.activePlans).toBe(0); // Falls back to 0
    });

    it("tracks active bases and pin counts", () => {
      // Create some tensors
      const tensor1 = engine.createTensor(1);
      const tensor2 = engine.createTensor(2);
      const tensor3 = engine.createTensor(1); // Same base as tensor1

      const stats = engine._debug_getMemoryStats();

      expect(stats.activeBases).toBe(2); // Two unique bases
      expect(stats.totalPinCount).toBe(3); // Three tensors total
    });

    it("updates pin count when tensors are disposed", () => {
      const tensor1 = engine.createTensor(1);
      const tensor2 = engine.createTensor(1);

      let stats = engine._debug_getMemoryStats();
      expect(stats.totalPinCount).toBe(2);

      engine.dispose(tensor1);

      stats = engine._debug_getMemoryStats();
      expect(stats.totalPinCount).toBe(1);
    });
  });

  describe("Saved Tensor Tracking", () => {
    it("tracks saved tensors", () => {
      engine.createTensor(1);

      // Save a tensor
      const record1 = engine._debug_saveForBackward(1);
      const record2 = engine._debug_saveForBackward(2);

      const savedTensors = engine._debug_getSavedTensorsInfo();
      expect(savedTensors).toHaveLength(2);

      const info1 = savedTensors.find((s) => s.id === record1.id);
      expect(info1).toBeDefined();
      expect(info1?.baseId).toBe(1);
      expect(info1?.commitVersionAtSave).toBe(0);
      expect(typeof info1?.savedAt).toBe("number");
    });

    it("includes saved tensor count in memory stats", () => {
      engine.createTensor(1);

      let stats = engine._debug_getMemoryStats();
      expect(stats.savedTensorCount).toBe(0);

      engine._debug_saveForBackward(1);
      engine._debug_saveForBackward(2);

      stats = engine._debug_getMemoryStats();
      expect(stats.savedTensorCount).toBe(2);
    });

    it("can release saved tensors", () => {
      engine.createTensor(1);
      const record = engine._debug_saveForBackward(1);

      expect(engine._debug_getSavedTensorsInfo()).toHaveLength(1);

      engine._debug_releaseSavedTensor(record.id);

      expect(engine._debug_getSavedTensorsInfo()).toHaveLength(0);
    });

    it("can clear all saved tensors", () => {
      engine.createTensor(1);
      engine._debug_saveForBackward(1);
      engine._debug_saveForBackward(2);
      engine._debug_saveForBackward(3);

      expect(engine._debug_getSavedTensorsInfo()).toHaveLength(3);

      engine._debug_clearSavedTensors();

      expect(engine._debug_getSavedTensorsInfo()).toHaveLength(0);
    });
  });

  describe("Base State Info", () => {
    it("returns base state information", () => {
      const tensor1 = engine.createTensor(1);
      const tensor2 = engine.createTensor(2);

      const baseStates = engine._debug_getBaseStatesInfo();

      expect(baseStates).toHaveLength(2);

      const base1 = baseStates.find((b) => b.baseId === 1);
      expect(base1).toBeDefined();
      expect(base1?.pinCount).toBe(1);
      expect(base1?.binding).toBe("ssa");
      expect(base1?.commitVersion).toBe(0);
    });

    it("tracks pin count correctly", () => {
      engine.createTensor(1);
      engine.createTensor(1);
      engine.createTensor(1);

      const baseStates = engine._debug_getBaseStatesInfo();
      const base1 = baseStates.find((b) => b.baseId === 1);

      expect(base1?.pinCount).toBe(3);
    });
  });

  describe("Memory Snapshots", () => {
    it("takes memory snapshots with labels", () => {
      engine._debug_takeMemorySnapshot("initial");

      const snapshots = engine._debug_getMemorySnapshots();
      expect(snapshots).toHaveLength(1);
      expect(snapshots[0].label).toBe("initial");
      expect(typeof snapshots[0].timestamp).toBe("number");
      expect(snapshots[0].stats).toBeDefined();
    });

    it("captures current stats in snapshot", () => {
      const tensor = engine.createTensor(1);
      engine._debug_saveForBackward(1);

      engine._debug_takeMemorySnapshot("with_tensor");

      const snapshots = engine._debug_getMemorySnapshots();
      expect(snapshots[0].stats.activeBases).toBe(1);
      expect(snapshots[0].stats.savedTensorCount).toBe(1);
    });

    it("accumulates multiple snapshots", () => {
      engine._debug_takeMemorySnapshot("step1");
      engine._debug_takeMemorySnapshot("step2");
      engine._debug_takeMemorySnapshot("step3");

      const snapshots = engine._debug_getMemorySnapshots();
      expect(snapshots).toHaveLength(3);
      expect(snapshots[0].label).toBe("step1");
      expect(snapshots[1].label).toBe("step2");
      expect(snapshots[2].label).toBe("step3");
    });

    it("clears snapshots", () => {
      engine._debug_takeMemorySnapshot("a");
      engine._debug_takeMemorySnapshot("b");

      expect(engine._debug_getMemorySnapshots()).toHaveLength(2);

      engine._debug_clearMemorySnapshots();

      expect(engine._debug_getMemorySnapshots()).toHaveLength(0);
    });

    it("returns a copy of snapshots (not internal array)", () => {
      engine._debug_takeMemorySnapshot("test");

      const snapshots1 = engine._debug_getMemorySnapshots();
      const snapshots2 = engine._debug_getMemorySnapshots();

      expect(snapshots1).not.toBe(snapshots2);
      expect(snapshots1).toEqual(snapshots2);
    });
  });

  describe("Integration with Tidy Scopes", () => {
    it("tracks tensors created in tidy scope", () => {
      engine.tidy(() => {
        const t1 = engine.createTensor(1);
        const t2 = engine.createTensor(2);

        const stats = engine._debug_getMemoryStats();
        expect(stats.activeBases).toBe(2);
        expect(stats.totalPinCount).toBe(2);

        return null;
      });

      // After tidy, tensors should be disposed
      const stats = engine._debug_getMemoryStats();
      expect(stats.activeBases).toBe(0);
      expect(stats.totalPinCount).toBe(0);
    });

    it("tracks kept tensors that escape tidy", () => {
      const escaped = engine.tidy(() => {
        const t1 = engine.createTensor(1);
        const t2 = engine.createTensor(2);
        engine.keep(t1);
        return t1;
      });

      // Only kept tensor should survive
      const stats = engine._debug_getMemoryStats();
      expect(stats.activeBases).toBe(1);
      expect(stats.totalPinCount).toBe(1);
    });
  });
});
