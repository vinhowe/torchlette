/**
 * Tests for tensor lifetime analysis (src/engine/lifetime-analysis.ts)
 *
 * Covers: lifetime computation, dead tensor detection, size class utilities.
 */

import { describe, expect, it } from "vitest";
import {
  analyzeLifetimes,
  findDeadTensorsAtStep,
  getSizeClass,
  getSizeForClass,
} from "../src/engine/lifetime-analysis";

describe("Lifetime Analysis", () => {
  // ========================================================================
  // Size class utilities
  // ========================================================================
  describe("getSizeClass / getSizeForClass", () => {
    it("maps small sizes to minimum buffer size class", () => {
      // MIN_BUFFER_SIZE = 256, so sizes < 256 all map to ceil(log2(256)) = 8
      expect(getSizeClass(1)).toBe(8);
      expect(getSizeClass(100)).toBe(8);
      expect(getSizeClass(256)).toBe(8);
    });

    it("maps powers of 2 exactly", () => {
      expect(getSizeClass(512)).toBe(9);
      expect(getSizeClass(1024)).toBe(10);
      expect(getSizeClass(4096)).toBe(12);
    });

    it("rounds up non-powers of 2", () => {
      expect(getSizeClass(300)).toBe(9); // ceil(log2(300)) = 9
      expect(getSizeClass(1025)).toBe(11); // ceil(log2(1025)) = 11
    });

    it("getSizeForClass is inverse for exact powers", () => {
      expect(getSizeForClass(8)).toBe(256);
      expect(getSizeForClass(10)).toBe(1024);
      expect(getSizeForClass(20)).toBe(1048576);
    });

    it("round-trip: getSizeForClass(getSizeClass(n)) >= n", () => {
      for (const n of [1, 100, 256, 500, 1000, 4096, 65536]) {
        expect(getSizeForClass(getSizeClass(n))).toBeGreaterThanOrEqual(n);
      }
    });
  });

  // ========================================================================
  // analyzeLifetimes
  // ========================================================================
  describe("analyzeLifetimes", () => {
    it("computes lifetime for a simple linear chain", () => {
      // a → b → c
      const nodeOrder = [10, 20, 30];
      const nodeInputs = new Map<number, number[]>([
        [10, []],
        [20, [10]],
        [30, [20]],
      ]);
      const nodeOutputs = new Set([30]);
      const nodeSizes = new Map<number, number>([
        [10, 1024],
        [20, 1024],
        [30, 1024],
      ]);

      const lifetimes = analyzeLifetimes(
        nodeOrder,
        nodeInputs,
        nodeOutputs,
        nodeSizes,
      );

      // Node 10: created at step 0, last used at step 1 (input to 20)
      expect(lifetimes.get(10)!.firstUse).toBe(0);
      expect(lifetimes.get(10)!.lastUse).toBe(1);
      expect(lifetimes.get(10)!.isInput).toBe(true);
      expect(lifetimes.get(10)!.isOutput).toBe(false);

      // Node 20: created at step 1, last used at step 2 (input to 30)
      expect(lifetimes.get(20)!.firstUse).toBe(1);
      expect(lifetimes.get(20)!.lastUse).toBe(2);

      // Node 30: output — lastUse extended to last step
      expect(lifetimes.get(30)!.firstUse).toBe(2);
      expect(lifetimes.get(30)!.lastUse).toBe(2);
      expect(lifetimes.get(30)!.isOutput).toBe(true);
    });

    it("extends lastUse for nodes used by multiple consumers", () => {
      // a used by b (step 1) and c (step 2)
      const nodeOrder = [1, 2, 3];
      const nodeInputs = new Map<number, number[]>([
        [1, []],
        [2, [1]],
        [3, [1]],
      ]);
      const nodeOutputs = new Set([3]);
      const nodeSizes = new Map<number, number>([
        [1, 512],
        [2, 512],
        [3, 512],
      ]);

      const lifetimes = analyzeLifetimes(
        nodeOrder,
        nodeInputs,
        nodeOutputs,
        nodeSizes,
      );

      // Node 1 is used at step 1 (by 2) and step 2 (by 3), so lastUse = 2
      expect(lifetimes.get(1)!.lastUse).toBe(2);
    });

    it("records buffer sizes", () => {
      const nodeOrder = [1, 2];
      const nodeInputs = new Map<number, number[]>([
        [1, []],
        [2, [1]],
      ]);
      const nodeOutputs = new Set([2]);
      const nodeSizes = new Map<number, number>([
        [1, 4096],
        [2, 8192],
      ]);

      const lifetimes = analyzeLifetimes(
        nodeOrder,
        nodeInputs,
        nodeOutputs,
        nodeSizes,
      );

      expect(lifetimes.get(1)!.bufferSize).toBe(4096);
      expect(lifetimes.get(2)!.bufferSize).toBe(8192);
    });

    it("defaults to 0 buffer size for unknown nodes", () => {
      const nodeOrder = [1];
      const nodeInputs = new Map<number, number[]>([[1, []]]);
      const nodeOutputs = new Set([1]);
      const nodeSizes = new Map<number, number>();

      const lifetimes = analyzeLifetimes(
        nodeOrder,
        nodeInputs,
        nodeOutputs,
        nodeSizes,
      );

      expect(lifetimes.get(1)!.bufferSize).toBe(0);
    });

    it("handles diamond pattern", () => {
      // a → b, a → c, b → d, c → d
      const nodeOrder = [1, 2, 3, 4];
      const nodeInputs = new Map<number, number[]>([
        [1, []],
        [2, [1]],
        [3, [1]],
        [4, [2, 3]],
      ]);
      const nodeOutputs = new Set([4]);
      const nodeSizes = new Map<number, number>([
        [1, 100],
        [2, 100],
        [3, 100],
        [4, 100],
      ]);

      const lifetimes = analyzeLifetimes(
        nodeOrder,
        nodeInputs,
        nodeOutputs,
        nodeSizes,
      );

      // a: used at step 1 (b) and step 2 (c), lastUse = 2
      expect(lifetimes.get(1)!.lastUse).toBe(2);
      // b: used at step 3 (d), lastUse = 3
      expect(lifetimes.get(2)!.lastUse).toBe(3);
      // c: used at step 3 (d), lastUse = 3
      expect(lifetimes.get(3)!.lastUse).toBe(3);
    });
  });

  // ========================================================================
  // findDeadTensorsAtStep
  // ========================================================================
  describe("findDeadTensorsAtStep", () => {
    it("finds tensors whose lastUse is before current step", () => {
      const nodeOrder = [1, 2, 3];
      const nodeInputs = new Map<number, number[]>([
        [1, []],
        [2, [1]],
        [3, [2]],
      ]);
      const nodeOutputs = new Set([3]);
      const nodeSizes = new Map<number, number>([
        [1, 100],
        [2, 100],
        [3, 100],
      ]);

      const lifetimes = analyzeLifetimes(
        nodeOrder,
        nodeInputs,
        nodeOutputs,
        nodeSizes,
      );

      // At step 2: node 1's lastUse is 1, so it's dead
      const dead = findDeadTensorsAtStep(lifetimes, 2, nodeOutputs, new Set());
      expect(dead).toContain(1);
      expect(dead).not.toContain(2); // lastUse is 2, not < 2
      expect(dead).not.toContain(3); // output
    });

    it("excludes output nodes", () => {
      const nodeOrder = [1, 2];
      const nodeInputs = new Map<number, number[]>([
        [1, []],
        [2, [1]],
      ]);
      const nodeOutputs = new Set([2]);
      const nodeSizes = new Map<number, number>([
        [1, 100],
        [2, 100],
      ]);

      const lifetimes = analyzeLifetimes(
        nodeOrder,
        nodeInputs,
        nodeOutputs,
        nodeSizes,
      );

      // Even at a late step, outputs are never dead
      const dead = findDeadTensorsAtStep(lifetimes, 10, nodeOutputs, new Set());
      expect(dead).not.toContain(2);
    });

    it("excludes already-released tensors", () => {
      const nodeOrder = [1, 2, 3];
      const nodeInputs = new Map<number, number[]>([
        [1, []],
        [2, [1]],
        [3, [2]],
      ]);
      const nodeOutputs = new Set([3]);
      const nodeSizes = new Map<number, number>([
        [1, 100],
        [2, 100],
        [3, 100],
      ]);

      const lifetimes = analyzeLifetimes(
        nodeOrder,
        nodeInputs,
        nodeOutputs,
        nodeSizes,
      );

      const alreadyReleased = new Set([1]);
      const dead = findDeadTensorsAtStep(
        lifetimes,
        2,
        nodeOutputs,
        alreadyReleased,
      );
      expect(dead).not.toContain(1); // already released
    });

    it("returns empty array when nothing is dead", () => {
      const nodeOrder = [1, 2];
      const nodeInputs = new Map<number, number[]>([
        [1, []],
        [2, [1]],
      ]);
      const nodeOutputs = new Set([2]);
      const nodeSizes = new Map<number, number>([
        [1, 100],
        [2, 100],
      ]);

      const lifetimes = analyzeLifetimes(
        nodeOrder,
        nodeInputs,
        nodeOutputs,
        nodeSizes,
      );

      // At step 0, nothing can be dead yet
      const dead = findDeadTensorsAtStep(lifetimes, 0, nodeOutputs, new Set());
      expect(dead).toEqual([]);
    });
  });
});
