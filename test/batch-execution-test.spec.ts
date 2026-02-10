/**
 * Simple test to verify batch execution works correctly.
 */

import { describe, expect, it, beforeAll } from "vitest";
import {
  initWebGPU,
  beginBatchExecution,
  endBatchExecution,
  isBatchActive,
  abortBatch,
  webgpuBackend,
  tensorFromArrayWithDtype,
} from "../src/backend/webgpu";
import { canUseWebGPU } from "./helpers/webgpu";

describe("Batch Execution", { timeout: 60000 }, () => {
  let webgpuAvailable = false;

  beforeAll(async () => {
    webgpuAvailable = await canUseWebGPU();
  });

  it("simple add works in batch mode", async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    // Create input tensors
    const a = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "f32");
    const b = tensorFromArrayWithDtype([5, 6, 7, 8], [2, 2], "f32");

    // Run in batch mode
    beginBatchExecution();
    expect(isBatchActive()).toBe(true);

    const result = webgpuBackend.ops.add(a, b);

    await endBatchExecution();
    expect(isBatchActive()).toBe(false);

    // Verify result
    const data = await webgpuBackend.ops.read(result);
    expect(data).toEqual([6, 8, 10, 12]);
  });

  it("chained ops work in batch mode", async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    // Create input tensors
    const a = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "f32");
    const b = tensorFromArrayWithDtype([1, 1, 1, 1], [2, 2], "f32");

    // Run chained ops in batch mode: (a + b) * 2 = (a + b) + (a + b)
    beginBatchExecution();

    try {
      const sum1 = webgpuBackend.ops.add(a, b);  // [2, 3, 4, 5]
      const sum2 = webgpuBackend.ops.add(sum1, sum1);  // [4, 6, 8, 10] - uses sum1 twice

      await endBatchExecution();

      // Verify result
      const data = await webgpuBackend.ops.read(sum2);
      expect(data).toEqual([4, 6, 8, 10]);
    } catch (error) {
      if (isBatchActive()) {
        abortBatch();
      }
      throw error;
    }
  });

  it("immediate mode still works", async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    // Create input tensors
    const a = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "f32");
    const b = tensorFromArrayWithDtype([5, 6, 7, 8], [2, 2], "f32");

    // Run without batch mode (immediate)
    expect(isBatchActive()).toBe(false);
    const result = webgpuBackend.ops.add(a, b);

    // Verify result
    const data = await webgpuBackend.ops.read(result);
    expect(data).toEqual([6, 8, 10, 12]);
  });
});
