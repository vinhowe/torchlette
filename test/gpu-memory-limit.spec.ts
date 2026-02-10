/**
 * Tests for GPU memory limit enforcement.
 * These tests verify that ALL buffer allocations respect the memory limit.
 */
import { beforeEach, describe, expect, it } from "vitest";
import {
  gpuMemoryTracker,
  GPUMemoryLimitExceededError,
  setGPUMemoryLimit,
  getGPUMemoryLimit,
  getGPUMemoryStats,
} from "../src/backend/webgpu/memory-tracker";

describe("GPU Memory Tracker", () => {
  beforeEach(() => {
    gpuMemoryTracker.reset();
    // Reset to default limit (10GB) after each test
    setGPUMemoryLimit(10 * 1024 * 1024 * 1024);
  });

  it("has default 10GB memory limit", () => {
    const tenGB = 10 * 1024 * 1024 * 1024;
    expect(getGPUMemoryLimit()).toBe(tenGB);
  });

  it("allows setting custom memory limit", () => {
    const oneGB = 1024 * 1024 * 1024;
    setGPUMemoryLimit(oneGB);
    expect(getGPUMemoryLimit()).toBe(oneGB);
  });

  it("throws when setting invalid memory limit", () => {
    expect(() => setGPUMemoryLimit(0)).toThrow("Memory limit must be positive");
    expect(() => setGPUMemoryLimit(-100)).toThrow("Memory limit must be positive");
  });

  it("tracks allocations correctly", () => {
    const buffer1 = { id: 1 };
    const buffer2 = { id: 2 };

    gpuMemoryTracker.trackAllocation(buffer1, 1024);
    expect(gpuMemoryTracker.getCurrentAllocatedBytes()).toBe(1024);

    gpuMemoryTracker.trackAllocation(buffer2, 2048);
    expect(gpuMemoryTracker.getCurrentAllocatedBytes()).toBe(1024 + 2048);
  });

  it("tracks deallocations correctly", () => {
    const buffer1 = { id: 1 };
    const buffer2 = { id: 2 };

    gpuMemoryTracker.trackAllocation(buffer1, 1024);
    gpuMemoryTracker.trackAllocation(buffer2, 2048);
    expect(gpuMemoryTracker.getCurrentAllocatedBytes()).toBe(3072);

    gpuMemoryTracker.trackDeallocation(buffer1);
    expect(gpuMemoryTracker.getCurrentAllocatedBytes()).toBe(2048);

    gpuMemoryTracker.trackDeallocation(buffer2);
    expect(gpuMemoryTracker.getCurrentAllocatedBytes()).toBe(0);
  });

  it("throws GPUMemoryLimitExceededError when limit exceeded", () => {
    setGPUMemoryLimit(1024); // 1KB limit

    const buffer1 = { id: 1 };
    gpuMemoryTracker.trackAllocation(buffer1, 512);

    const buffer2 = { id: 2 };
    expect(() => gpuMemoryTracker.trackAllocation(buffer2, 1024)).toThrow(
      GPUMemoryLimitExceededError,
    );
  });

  it("GPUMemoryLimitExceededError contains correct information", () => {
    setGPUMemoryLimit(1024);

    const buffer1 = { id: 1 };
    gpuMemoryTracker.trackAllocation(buffer1, 512);

    try {
      const buffer2 = { id: 2 };
      gpuMemoryTracker.trackAllocation(buffer2, 768);
      expect.fail("Should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(GPUMemoryLimitExceededError);
      const err = e as GPUMemoryLimitExceededError;
      expect(err.requestedBytes).toBe(768);
      expect(err.currentBytes).toBe(512);
      expect(err.limitBytes).toBe(1024);
      expect(err.message).toContain("GPU memory limit exceeded");
    }
  });

  it("tracks peak usage", () => {
    const buffer1 = { id: 1 };
    const buffer2 = { id: 2 };

    gpuMemoryTracker.trackAllocation(buffer1, 1024);
    gpuMemoryTracker.trackAllocation(buffer2, 2048);
    expect(gpuMemoryTracker.getPeakUsageBytes()).toBe(3072);

    gpuMemoryTracker.trackDeallocation(buffer1);
    // Peak should still be 3072 even after deallocation
    expect(gpuMemoryTracker.getPeakUsageBytes()).toBe(3072);
    expect(gpuMemoryTracker.getCurrentAllocatedBytes()).toBe(2048);
  });

  it("provides correct statistics", () => {
    setGPUMemoryLimit(10000);

    const buffer1 = { id: 1 };
    const buffer2 = { id: 2 };

    gpuMemoryTracker.trackAllocation(buffer1, 1024);
    gpuMemoryTracker.trackAllocation(buffer2, 2048);

    const stats = getGPUMemoryStats();
    expect(stats.currentBytes).toBe(3072);
    expect(stats.peakBytes).toBe(3072);
    expect(stats.limitBytes).toBe(10000);
    expect(stats.usagePercent).toBeCloseTo(30.72, 1);
    expect(stats.allocationCount).toBe(2);
    expect(stats.availableBytes).toBe(10000 - 3072);
  });

  it("reset clears all tracking state", () => {
    const buffer1 = { id: 1 };
    gpuMemoryTracker.trackAllocation(buffer1, 1024);
    expect(gpuMemoryTracker.getCurrentAllocatedBytes()).toBe(1024);

    gpuMemoryTracker.reset();

    expect(gpuMemoryTracker.getCurrentAllocatedBytes()).toBe(0);
    expect(gpuMemoryTracker.getPeakUsageBytes()).toBe(0);
    expect(gpuMemoryTracker.getAllocationCount()).toBe(0);
  });

  it("wouldExceedLimit correctly predicts overflows", () => {
    setGPUMemoryLimit(1024);

    const buffer1 = { id: 1 };
    gpuMemoryTracker.trackAllocation(buffer1, 512);

    expect(gpuMemoryTracker.wouldExceedLimit(256)).toBe(false);
    expect(gpuMemoryTracker.wouldExceedLimit(512)).toBe(false);
    expect(gpuMemoryTracker.wouldExceedLimit(513)).toBe(true);
    expect(gpuMemoryTracker.wouldExceedLimit(1024)).toBe(true);
  });
});
