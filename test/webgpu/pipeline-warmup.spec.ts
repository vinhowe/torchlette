import { describe, expect, it, beforeAll, afterAll } from "vitest";
import {
  initWebGPU,
  destroyWebGPU,
  getWebGPUDevice,
  startPipelineRecording,
  stopPipelineRecording,
  warmupPipelines,
  warmupFromStep,
  warmupFromRegistry,
  clearWarmupCache,
  serializeRegistry,
  deserializeRegistry,
  getPipeline,
} from "../../src/backend/webgpu/index";
import { requireContext } from "../../src/backend/webgpu/gpu-context";
import { getWarmupPipeline } from "../../src/backend/webgpu/pipeline-warmup";

describe("Pipeline Warmup", () => {
  beforeAll(async () => {
    await initWebGPU();
  });

  afterAll(() => {
    destroyWebGPU();
  });

  // A trivial WGSL shader for testing
  const trivialShader = (id: number) => `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  out[gid.x] = f32(${id});
}
`;

  describe("Recording", () => {
    it("captures pipeline creations during recording", () => {
      const ctx = requireContext();

      startPipelineRecording();
      // Create 3 different pipelines
      getPipeline(ctx, "warmup_test_1", trivialShader(1));
      getPipeline(ctx, "warmup_test_2", trivialShader(2));
      getPipeline(ctx, "warmup_test_3", trivialShader(3));
      const entries = stopPipelineRecording();

      expect(entries.length).toBe(3);
      expect(entries.map((e) => e.key)).toEqual([
        "warmup_test_1",
        "warmup_test_2",
        "warmup_test_3",
      ]);
      // Each entry has WGSL source
      for (const entry of entries) {
        expect(entry.wgsl).toContain("@compute");
      }
    });

    it("deduplicates repeated keys during recording", () => {
      const ctx = requireContext();

      startPipelineRecording();
      getPipeline(ctx, "warmup_dedup_1", trivialShader(1));
      // Second call with same key hits pipeline cache, no record
      getPipeline(ctx, "warmup_dedup_1", trivialShader(1));
      const entries = stopPipelineRecording();

      expect(entries.length).toBe(1);
    });

    it("does not record when not in recording mode", () => {
      const ctx = requireContext();

      // Not recording — entries should be empty
      getPipeline(ctx, "warmup_norecord", trivialShader(99));
      startPipelineRecording();
      const entries = stopPipelineRecording();

      expect(entries.length).toBe(0);
    });
  });

  describe("Warmup", () => {
    it("compiles pipelines and populates warmup cache", async () => {
      const devInfo = getWebGPUDevice()!;
      clearWarmupCache();

      const entries = [
        { key: "warmup_async_1", wgsl: trivialShader(10) },
        { key: "warmup_async_2", wgsl: trivialShader(20) },
      ];

      const result = await warmupPipelines(devInfo.device, entries);

      expect(result.compiled).toBe(2);
      expect(result.skipped).toBe(0);
      expect(result.timeMs).toBeGreaterThanOrEqual(0);

      // Verify warmup cache has the pipelines
      expect(getWarmupPipeline("warmup_async_1")).toBeDefined();
      expect(getWarmupPipeline("warmup_async_2")).toBeDefined();
    });

    it("skips already-cached entries", async () => {
      const devInfo = getWebGPUDevice()!;
      clearWarmupCache();

      const entries = [
        { key: "warmup_skip_1", wgsl: trivialShader(30) },
      ];

      // First warmup compiles
      const r1 = await warmupPipelines(devInfo.device, entries);
      expect(r1.compiled).toBe(1);

      // Second warmup skips
      const r2 = await warmupPipelines(devInfo.device, entries);
      expect(r2.compiled).toBe(0);
      expect(r2.skipped).toBe(1);
    });

    it("dispatch paths find warmed-up pipelines", async () => {
      const devInfo = getWebGPUDevice()!;
      const ctx = requireContext();
      clearWarmupCache();

      const key = "warmup_dispatch_find";
      const wgsl = trivialShader(40);

      // Pre-compile via warmup
      await warmupPipelines(devInfo.device, [{ key, wgsl }]);

      // Now getPipeline should find it in warmup cache (no sync compilation)
      startPipelineRecording();
      const pipeline = getPipeline(ctx, key, wgsl);
      const entries = stopPipelineRecording();

      expect(pipeline).toBeDefined();
      // Should NOT have recorded (found in warmup cache, no compilation)
      expect(entries.length).toBe(0);
    });
  });

  describe("Serialization", () => {
    it("round-trips registry entries", () => {
      const entries = [
        { key: "serial_1", wgsl: "shader code 1" },
        { key: "serial_2", wgsl: "shader code 2" },
      ];

      const json = serializeRegistry(entries);
      const restored = deserializeRegistry(json);

      expect(restored).toEqual(entries);
    });
  });

  describe("warmupFromStep", () => {
    it("records pipelines during step execution", async () => {
      const ctx = requireContext();
      clearWarmupCache();

      const registry = await warmupFromStep(() => {
        // Simulate creating some pipelines during a step
        getPipeline(ctx, "warmup_step_1", trivialShader(50));
        getPipeline(ctx, "warmup_step_2", trivialShader(51));
      });

      expect(registry.length).toBe(2);
      expect(registry[0].key).toBe("warmup_step_1");
      expect(registry[1].key).toBe("warmup_step_2");
    });
  });

  describe("warmupFromRegistry", () => {
    it("compiles from registry using current device", async () => {
      clearWarmupCache();

      const entries = [
        { key: "warmup_reg_1", wgsl: trivialShader(60) },
        { key: "warmup_reg_2", wgsl: trivialShader(61) },
      ];

      const result = await warmupFromRegistry(entries);

      expect(result.compiled).toBe(2);
      expect(getWarmupPipeline("warmup_reg_1")).toBeDefined();
      expect(getWarmupPipeline("warmup_reg_2")).toBeDefined();
    });
  });

  describe("clearWarmupCache", () => {
    it("clears all warmed-up pipelines", async () => {
      const devInfo = getWebGPUDevice()!;
      clearWarmupCache();

      await warmupPipelines(devInfo.device, [
        { key: "warmup_clear_1", wgsl: trivialShader(70) },
      ]);
      expect(getWarmupPipeline("warmup_clear_1")).toBeDefined();

      clearWarmupCache();
      expect(getWarmupPipeline("warmup_clear_1")).toBeUndefined();
    });
  });
});
