/**
 * Tests for memory donation (§14).
 *
 * Memory donation allows reusing input buffers as output buffers
 * when the input tensor dies at the operation.
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  findDonationOpportunities,
  analyzeLifetimes,
  computeBufferSize,
} from "../src/engine/memory-planning";

describe("memory donation", () => {
  describe("findDonationOpportunities", () => {
    it("identifies donation when input dies at consumer", () => {
      // Graph: a -> relu -> b
      // 'a' dies after relu, so its buffer can be donated to 'b'
      const nodeOrder = [1, 2]; // a=1, relu output=2
      const lifetimes = new Map([
        [1, { nodeId: 1, firstUse: 0, lastUse: 0, isOutput: false, isInput: true, bufferSize: 1024 }],
        [2, { nodeId: 2, firstUse: 1, lastUse: 1, isOutput: true, isInput: false, bufferSize: 1024 }],
      ]);
      const nodeSizes = new Map([
        [1, 1024],
        [2, 1024],
      ]);

      const donations = findDonationOpportunities(nodeOrder, lifetimes, nodeSizes);

      // Node 2 can receive donation from node 1
      expect(donations.get(2)).toBe(1);
    });

    it("does not donate when input is still used", () => {
      // Graph: a -> relu -> b, a -> add(b) -> c
      // 'a' is used by both relu and add, so it can't be donated to relu
      const nodeOrder = [1, 2, 3]; // a=1, relu=2, add=3
      const lifetimes = new Map([
        [1, { nodeId: 1, firstUse: 0, lastUse: 2, isOutput: false, isInput: true, bufferSize: 1024 }], // lives until add
        [2, { nodeId: 2, firstUse: 1, lastUse: 2, isOutput: false, isInput: false, bufferSize: 1024 }],
        [3, { nodeId: 3, firstUse: 2, lastUse: 2, isOutput: true, isInput: false, bufferSize: 1024 }],
      ]);
      const nodeSizes = new Map([
        [1, 1024],
        [2, 1024],
        [3, 1024],
      ]);

      const donations = findDonationOpportunities(nodeOrder, lifetimes, nodeSizes);

      // Node 2 cannot receive donation from node 1 (still alive)
      expect(donations.has(2)).toBe(false);
    });

    it("does not donate when buffer is too small", () => {
      // Graph: a -> op -> b (where b needs more space than a)
      const nodeOrder = [1, 2];
      const lifetimes = new Map([
        [1, { nodeId: 1, firstUse: 0, lastUse: 0, isOutput: false, isInput: true, bufferSize: 512 }],
        [2, { nodeId: 2, firstUse: 1, lastUse: 1, isOutput: true, isInput: false, bufferSize: 1024 }],
      ]);
      const nodeSizes = new Map([
        [1, 512],
        [2, 1024], // needs 1024 but donor only has 512
      ]);

      const donations = findDonationOpportunities(nodeOrder, lifetimes, nodeSizes);

      // Cannot donate because buffer is too small
      expect(donations.has(2)).toBe(false);
    });

    it("prefers smallest sufficient donor", () => {
      // Graph: a, b both die at step 2, c needs 512 bytes
      // a has 1024 bytes, b has 512 bytes -> prefer b
      const nodeOrder = [1, 2, 3];
      const lifetimes = new Map([
        [1, { nodeId: 1, firstUse: 0, lastUse: 1, isOutput: false, isInput: true, bufferSize: 1024 }],
        [2, { nodeId: 2, firstUse: 0, lastUse: 1, isOutput: false, isInput: true, bufferSize: 512 }],
        [3, { nodeId: 3, firstUse: 2, lastUse: 2, isOutput: true, isInput: false, bufferSize: 512 }],
      ]);
      const nodeSizes = new Map([
        [1, 1024],
        [2, 512],
        [3, 512],
      ]);

      const donations = findDonationOpportunities(nodeOrder, lifetimes, nodeSizes);

      // Should prefer node 2 (512 bytes) over node 1 (1024 bytes)
      expect(donations.get(3)).toBe(2);
    });

    it("does not double-donate same buffer", () => {
      // Graph: a dies at step 1, b and c both want donation
      const nodeOrder = [1, 2, 3];
      const lifetimes = new Map([
        [1, { nodeId: 1, firstUse: 0, lastUse: 0, isOutput: false, isInput: true, bufferSize: 1024 }],
        [2, { nodeId: 2, firstUse: 1, lastUse: 1, isOutput: false, isInput: false, bufferSize: 1024 }],
        [3, { nodeId: 3, firstUse: 1, lastUse: 1, isOutput: true, isInput: false, bufferSize: 1024 }],
      ]);
      const nodeSizes = new Map([
        [1, 1024],
        [2, 1024],
        [3, 1024],
      ]);

      const donations = findDonationOpportunities(nodeOrder, lifetimes, nodeSizes);

      // Node 1 can only be donated once
      const donatedTo = [donations.get(2), donations.get(3)].filter(x => x === 1);
      expect(donatedTo.length).toBe(1);
    });
  });

  describe("analyzeLifetimes", () => {
    it("computes correct first and last use", () => {
      // Graph: a -> relu -> b -> add(a) -> c
      // a is used at step 0 (creation) and step 2 (add input)
      const nodeOrder = [1, 2, 3]; // a=1, relu=2 (uses a), add=3 (uses a and 2)
      const nodeInputs = new Map([
        [1, []],           // a has no inputs
        [2, [1]],          // relu uses a
        [3, [1, 2]],       // add uses a and relu output
      ]);
      const outputSet = new Set([3]);
      const nodeSizes = new Map([
        [1, 1024],
        [2, 1024],
        [3, 1024],
      ]);

      const lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, outputSet, nodeSizes);

      // Node 1 (a): created at step 0, last used at step 2
      expect(lifetimes.get(1)?.firstUse).toBe(0);
      expect(lifetimes.get(1)?.lastUse).toBe(2);

      // Node 2 (relu output): created at step 1, last used at step 2
      expect(lifetimes.get(2)?.firstUse).toBe(1);
      expect(lifetimes.get(2)?.lastUse).toBe(2);

      // Node 3 (output): created at step 2, last used at step 2 (output)
      expect(lifetimes.get(3)?.firstUse).toBe(2);
      expect(lifetimes.get(3)?.lastUse).toBe(2);
      expect(lifetimes.get(3)?.isOutput).toBe(true);
    });
  });

  describe("computeBufferSize", () => {
    it("computes correct size for f32", () => {
      expect(computeBufferSize([10, 10], "f32")).toBe(10 * 10 * 4);
    });

    it("computes correct size for f16", () => {
      expect(computeBufferSize([10, 10], "f16")).toBe(10 * 10 * 2);
    });

    it("computes correct size for i32", () => {
      expect(computeBufferSize([10, 10], "i32")).toBe(10 * 10 * 4);
    });

    it("handles scalar (empty shape)", () => {
      expect(computeBufferSize([], "f32")).toBe(4);
    });
  });
});
