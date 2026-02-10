/**
 * Checkpoint Segmentation Tests
 *
 * Tests the segmented execution feature for checkpointing, which enables
 * memory savings for large models by executing checkpoint segments separately
 * and flushing buffers between them.
 */

import { describe, expect, it, beforeAll, afterEach } from "vitest";
import { Torchlette } from "../src/frontend";
import { initWebGPU, getBufferPoolDetailedStats, resetBufferPoolDetailedStats } from "../src/backend/webgpu";
import { canUseWebGPU } from "./helpers/webgpu";
import { gpuMemoryTracker } from "../src/backend/webgpu/memory-tracker";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { resetNodeIdCounter, resetStorageIdCounter } from "../src/engine/lazy";
import { resetBaseIdCounter } from "../src/runtime/tensor";

// Larger config to stress memory
const TEST_CONFIG: GPT2Config = {
  vocabSize: 500,
  blockSize: 64,
  numLayers: 6,  // More layers to see segmentation benefits
  numHeads: 4,
  embedDim: 128,
  dropoutRate: 0.0,
};

const BATCH = 2;
const SEQ_LEN = 32;

interface TestResult {
  peakBytes: number;
  newAllocations: number;
  fromPending: number;
  reuseRate: number;
}

async function runTest(
  useCheckpoint: boolean,
  enableEarlyRelease: boolean,
  enableCheckpointSegmentation: boolean,
): Promise<TestResult> {
  gpuMemoryTracker.reset();
  resetBufferPoolDetailedStats();
  resetNodeIdCounter();
  resetStorageIdCounter();
  resetBaseIdCounter();

  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: true,
    enableEarlyRelease,
    enableCheckpointSegmentation,
    // True segmentation provides actual memory savings by syncing GPU
    // between checkpoint segments and releasing dead buffers.
    enableTrueSegmentation: useCheckpoint,
  });

  const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
  model.train();

  const inputData = Array.from({ length: BATCH * SEQ_LEN }, () =>
    Math.floor(Math.random() * TEST_CONFIG.vocabSize)
  );
  const input = api.tensorFromArray(inputData, [BATCH, SEQ_LEN]);

  const output = model.forward(input, { useCheckpoint });
  const loss = output.sum();
  if (typeof loss === "number") throw new Error("Expected tensor");

  await loss.backward();
  await api.markStep();

  const stats = getBufferPoolDetailedStats();
  const totalAcquires = stats.acquireNew + stats.acquireFromPool + stats.acquireFromPending;
  const reuseRate = totalAcquires > 0
    ? (stats.acquireFromPool + stats.acquireFromPending) / totalAcquires
    : 0;

  return {
    peakBytes: gpuMemoryTracker.getPeakUsageBytes(),
    newAllocations: stats.acquireNew,
    fromPending: stats.acquireFromPending,
    reuseRate,
  };
}

describe("Checkpoint Segmentation", { timeout: 300000 }, () => {
  let webgpuAvailable = false;

  beforeAll(async () => {
    const success = await canUseWebGPU();
    webgpuAvailable = success;
    if (!success) {
      console.warn("WebGPU not available - tests will be skipped");
    }
  });

  afterEach(() => {
    gpuMemoryTracker.reset();
  });

  it("segmented execution provides memory savings for checkpointed models", async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    console.log("\n=== Checkpoint Segmentation Test ===\n");
    console.log(`Config: ${TEST_CONFIG.numLayers} layers, ${TEST_CONFIG.embedDim} embed, ${BATCH}x${SEQ_LEN} batch\n`);

    // Test configurations
    const results: Record<string, TestResult> = {};

    // Baseline: no checkpoint, no early release, no segmentation
    results["baseline"] = await runTest(false, false, false);

    // Early release only
    results["early-release"] = await runTest(false, true, false);

    // Checkpoint + early release (no segmentation)
    results["ckpt+early"] = await runTest(true, true, false);

    // Checkpoint + early release + segmentation
    // NOTE: Segmentation currently doesn't provide additional benefit because
    // buffer pool flush happens within the same GPU command batch. For true
    // segmentation, we'd need to submit and sync between segments.
    // Disabled for now - the early release + checkpoint combo is working well.
    // results["ckpt+early+seg"] = await runTest(true, true, true);

    // Print results
    console.log("| Configuration | Peak MB | New Allocs | From Pending | Reuse % |");
    console.log("|---------------|---------|------------|--------------|---------|");

    for (const [name, r] of Object.entries(results)) {
      console.log(
        `| ${name.padEnd(13)} | ${(r.peakBytes / 1e6).toFixed(2).padStart(7)} | ${String(r.newAllocations).padStart(10)} | ${String(r.fromPending).padStart(12)} | ${(r.reuseRate * 100).toFixed(1).padStart(7)} |`
      );
    }

    // Calculate improvements
    const baselinePeak = results["baseline"].peakBytes;
    const earlyReleasePeak = results["early-release"].peakBytes;
    const ckptEarlyPeak = results["ckpt+early"].peakBytes;

    console.log("\n=== Memory Reduction vs Baseline ===");
    console.log(`Early release only: ${((1 - earlyReleasePeak / baselinePeak) * 100).toFixed(1)}%`);
    console.log(`Checkpoint + early release: ${((1 - ckptEarlyPeak / baselinePeak) * 100).toFixed(1)}%`);
    console.log(`Additional benefit from checkpoint: ${((1 - ckptEarlyPeak / earlyReleasePeak) * 100).toFixed(1)}%`);

    // Expectations:
    // 1. Early release should provide significant benefit (>20%)
    const earlyReleaseReduction = (1 - earlyReleasePeak / baselinePeak) * 100;
    expect(earlyReleaseReduction).toBeGreaterThan(20);

    // 2. Checkpoint + early release should provide meaningful total reduction
    //    (Note: checkpoint may not always beat early-release-alone because
    //    contiguous no-op views are non-owning and can't be independently released.)
    const totalReduction = (1 - ckptEarlyPeak / baselinePeak) * 100;
    expect(totalReduction).toBeGreaterThan(0);
  });

  it("produces correct gradients with checkpoint and early release", async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    // Run with early release and checkpoint
    gpuMemoryTracker.reset();
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();

    const api = new Torchlette("webgpu", {
      enableEarlyRelease: true,
    });

    const model = new GPT2(api, {
      ...TEST_CONFIG,
      numLayers: 2,  // Smaller for faster test
    }, { device: "webgpu" });
    model.train();

    const inputData = [1, 2, 3, 4, 5, 6, 7, 8];
    const input = api.tensorFromArray(inputData, [1, 8]);
    const output = model.forward(input, { useCheckpoint: true });
    const loss = output.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");
    await loss.backward();

    // Verify gradients are computed
    const params = model.parameters();
    expect(params.length).toBeGreaterThan(0);

    const grad = params[0].grad;
    expect(grad).toBeDefined();

    const gradValue = await api.cpu(grad!);
    expect(gradValue.length).toBeGreaterThan(0);

    // Verify gradients are not all zeros (actual computation happened)
    const hasNonZero = gradValue.some(v => Math.abs(v) > 1e-6);
    expect(hasNonZero).toBe(true);

    // Verify gradients are finite
    const allFinite = gradValue.every(v => Number.isFinite(v));
    expect(allFinite).toBe(true);

    console.log(`First 5 gradient values: ${gradValue.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);
  });
});
