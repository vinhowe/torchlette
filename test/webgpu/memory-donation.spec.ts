/**
 * WebGPU-specific tests for memory donation.
 *
 * Tests that buffer donation actually works at the GPU level.
 */

import { describe, it, expect, beforeAll } from "vitest";
import {
  initWebGPU,
  webgpuBackend,
  donateBuffer,
  getBufferSize,
  getBufferPoolStats,
  clearBufferPool,
} from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

describe("WebGPU memory donation", { skip: cpuOnly }, () => {
  beforeAll(async () => {
    const success = await initWebGPU();
    if (!success) {
      throw new Error("WebGPU not available");
    }
    clearBufferPool();
  });

  describe("donateBuffer", () => {
    it("returns buffer for owned tensor with compatible usage", () => {
      // Create a tensor via compute op (will have pool-compatible usage)
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = webgpuBackend.ops.tensorFromArray([5, 6, 7, 8], [2, 2]);
      const c = webgpuBackend.ops.add(a, b); // Compute output has STORAGE|COPY_SRC|COPY_DST

      // Donate the buffer
      const buffer = donateBuffer(c);

      // Should succeed for compute output
      expect(buffer).not.toBeNull();

      // Cleanup (a and b still own their buffers)
      a.destroy?.();
      b.destroy?.();
      // c's buffer was donated, destroying c shouldn't destroy the buffer
      c.destroy?.();
    });

    it("returns null for tensor that doesn't own buffer", () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = webgpuBackend.ops.tensorFromArray([5, 6, 7, 8], [2, 2]);
      const c = webgpuBackend.ops.add(a, b);

      // Donate once
      const buffer1 = donateBuffer(c);
      expect(buffer1).not.toBeNull();

      // Try to donate again - should fail (no longer owns buffer)
      const buffer2 = donateBuffer(c);
      expect(buffer2).toBeNull();

      a.destroy?.();
      b.destroy?.();
    });

    it("tensorFromArray buffers are donatable (same usage flags)", () => {
      // tensorFromArray uses mappedAtCreation but still has compatible usage flags
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);

      // Should be donatable - usage flags are STORAGE|COPY_SRC|COPY_DST
      const buffer = donateBuffer(a);
      expect(buffer).not.toBeNull();

      // After donation, a no longer owns buffer
      const buffer2 = donateBuffer(a);
      expect(buffer2).toBeNull();
    });
  });

  describe("getBufferSize", () => {
    it("returns correct size for tensor", () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const size = getBufferSize(a);

      // 4 elements * 4 bytes/element = 16 bytes, but aligned/size-classed
      expect(size).toBeGreaterThanOrEqual(16);

      a.destroy?.();
    });
  });

  describe("donation with outBuffer", () => {
    it("uses donated buffer for add output", async () => {
      clearBufferPool();
      const statsBefore = getBufferPoolStats();

      // Create tensors
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = webgpuBackend.ops.tensorFromArray([5, 6, 7, 8], [2, 2]);

      // First add - creates new buffer
      const c = webgpuBackend.ops.add(a, b);

      // Donate c's buffer
      const donatedBuf = donateBuffer(c);
      expect(donatedBuf).not.toBeNull();

      // Second add using donated buffer
      const d = (webgpuBackend.ops.add as any)(a, b, { outBuffer: donatedBuf });

      // Verify result is correct
      const result = await webgpuBackend.ops.read(d);
      expect(result).toEqual([6, 8, 10, 12]);

      // Cleanup
      a.destroy?.();
      b.destroy?.();
      // c doesn't own buffer anymore
      // d owns the buffer now
      d.destroy?.();
    });

    it("uses donated buffer for relu output", async () => {
      const a = webgpuBackend.ops.tensorFromArray([-1, 2, -3, 4], [2, 2]);
      const b = webgpuBackend.ops.tensorFromArray([0, 0, 0, 0], [2, 2]);

      // Create intermediate that will be donated
      const c = webgpuBackend.ops.add(a, b); // [-1, 2, -3, 4]

      // Donate c's buffer
      const donatedBuf = donateBuffer(c);
      expect(donatedBuf).not.toBeNull();

      // Use donated buffer for relu
      const d = (webgpuBackend.ops.relu as any)(a, { outBuffer: donatedBuf });

      // Verify result
      const result = await webgpuBackend.ops.read(d);
      expect(result).toEqual([0, 2, 0, 4]);

      a.destroy?.();
      b.destroy?.();
      d.destroy?.();
    });
  });
});
