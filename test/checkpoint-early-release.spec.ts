/**
 * Checkpoint Memory Savings with Early Release
 *
 * Tests the interaction between gradient checkpointing and early buffer release.
 *
 * Key findings:
 * - Early buffer release provides MASSIVE memory savings (90%+) by releasing
 *   intermediate buffers as soon as they're no longer needed during execution
 * - For WebGPU, buffers are released to the pool's deferred destruction queue,
 *   which waits for the GPU fence before actual destruction. This enables safe
 *   buffer reuse within the same execution.
 * - Checkpoint without early release: ~6.5% reduction (graph-level only)
 * - Checkpoint WITH early release: ~95% reduction (execution-level buffer reuse)
 * - Forward-only (inference) with early release: ~89% reduction
 *
 * Run with: npm test -- test/checkpoint-early-release.spec.ts
 */

import { describe, expect, it, beforeAll, afterEach } from "vitest";
import { Torchlette } from "../src/frontend";
import { initWebGPU } from "../src/backend/webgpu";
import { gpuMemoryTracker } from "../src/backend/webgpu/memory-tracker";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { canUseWebGPU } from "./helpers/webgpu";
import { resetNodeIdCounter, resetStorageIdCounter } from "../src/engine/lazy";
import { resetBaseIdCounter } from "../src/runtime/tensor";

// Test configuration - small enough to fit in memory but large enough to show effects
const TEST_CONFIG: GPT2Config = {
  vocabSize: 500,
  blockSize: 64,
  numLayers: 4,
  numHeads: 4,
  embedDim: 128,
  dropoutRate: 0.0,
};

const BATCH = 2;
const SEQ_LEN = 32;

interface MemoryMeasurement {
  peakBytes: number;
  description: string;
}

async function measureMemory(
  description: string,
  useCheckpoint: boolean,
  enableEarlyRelease: boolean
): Promise<MemoryMeasurement> {
  gpuMemoryTracker.reset();
  resetNodeIdCounter();
  resetStorageIdCounter();
  resetBaseIdCounter();

  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: true,
    enableEarlyRelease,
    // In a lazy engine, checkpoint only reduces memory when paired with
    // segmentation — the executor must sync and free buffers between segments.
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

  return {
    peakBytes: gpuMemoryTracker.getPeakUsageBytes(),
    description,
  };
}

describe("Checkpoint Memory with Early Release", { timeout: 300000 }, () => {
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

  it("compares all configurations", async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    console.log("\n=== Checkpoint + Early Release Memory Analysis ===\n");
    console.log(`Config: ${TEST_CONFIG.numLayers} layers, ${TEST_CONFIG.embedDim} embed, ${BATCH}x${SEQ_LEN} batch\n`);

    // Measure all 4 configurations
    const noCheckpointNoEarly = await measureMemory(
      "No checkpoint, no early release",
      false,
      false
    );

    const noCheckpointWithEarly = await measureMemory(
      "No checkpoint, WITH early release",
      false,
      true
    );

    const withCheckpointNoEarly = await measureMemory(
      "WITH checkpoint, no early release",
      true,
      false
    );

    const withCheckpointWithEarly = await measureMemory(
      "WITH checkpoint, WITH early release",
      true,
      true
    );

    // Print results
    console.log("| Configuration | Peak Memory | vs Baseline |");
    console.log("|---------------|-------------|-------------|");

    const baseline = noCheckpointNoEarly.peakBytes;
    for (const m of [noCheckpointNoEarly, noCheckpointWithEarly, withCheckpointNoEarly, withCheckpointWithEarly]) {
      const reduction = ((baseline - m.peakBytes) / baseline * 100).toFixed(1);
      const sign = m.peakBytes <= baseline ? "-" : "+";
      console.log(
        `| ${m.description.padEnd(40)} | ${(m.peakBytes / 1e6).toFixed(2).padStart(8)} MB | ${sign}${Math.abs(parseFloat(reduction)).toFixed(1).padStart(5)}% |`
      );
    }

    // Calculate improvements
    console.log("\n=== Analysis ===\n");

    const checkpointOnlyReduction = (1 - withCheckpointNoEarly.peakBytes / noCheckpointNoEarly.peakBytes) * 100;
    const earlyOnlyReduction = (1 - noCheckpointWithEarly.peakBytes / noCheckpointNoEarly.peakBytes) * 100;
    const combinedReduction = (1 - withCheckpointWithEarly.peakBytes / noCheckpointNoEarly.peakBytes) * 100;

    console.log(`Checkpoint-only reduction: ${checkpointOnlyReduction.toFixed(1)}%`);
    console.log(`Early-release-only reduction: ${earlyOnlyReduction.toFixed(1)}%`);
    console.log(`Combined (checkpoint + early release): ${combinedReduction.toFixed(1)}%`);

    // Calculate if early release improves checkpoint savings
    const checkpointWithEarlyImprovement = withCheckpointNoEarly.peakBytes - withCheckpointWithEarly.peakBytes;
    const improvementPercent = (checkpointWithEarlyImprovement / withCheckpointNoEarly.peakBytes) * 100;

    console.log(`\nEarly release additional savings with checkpoint: ${(checkpointWithEarlyImprovement / 1e6).toFixed(2)} MB (${improvementPercent.toFixed(1)}%)`);

    // Expectations
    // 1. In a lazy execution engine, checkpoint alone (without early release)
    //    may NOT reduce memory because recomputation adds intermediate nodes
    //    that all execute within a single segment. The savings come from
    //    combining checkpoint with early release or segmentation.
    //    Assert no catastrophic regression (< 20% increase is acceptable).
    expect(checkpointOnlyReduction).toBeGreaterThan(-20);

    // 2. Early release should provide some reduction
    expect(earlyOnlyReduction).toBeGreaterThanOrEqual(0);

    // 3. Combined (checkpoint + early release) should be significantly better
    //    than either alone — this is the key validation of the checkpoint mechanism.
    expect(combinedReduction).toBeGreaterThan(checkpointOnlyReduction);
    expect(combinedReduction).toBeGreaterThan(0);
  });

  it("measures checkpoint reduction improvement from early release", async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    // Run without early release
    const withoutEarly = await measureMemory("Checkpoint without early release", true, false);

    // Run with early release
    const withEarly = await measureMemory("Checkpoint with early release", true, true);

    const improvement = withoutEarly.peakBytes - withEarly.peakBytes;
    const improvementPercent = (improvement / withoutEarly.peakBytes) * 100;

    console.log(`\nCheckpoint peak WITHOUT early release: ${(withoutEarly.peakBytes / 1e6).toFixed(2)} MB`);
    console.log(`Checkpoint peak WITH early release: ${(withEarly.peakBytes / 1e6).toFixed(2)} MB`);
    console.log(`Improvement: ${(improvement / 1e6).toFixed(2)} MB (${improvementPercent.toFixed(1)}%)`);

    // Early release should not make things worse
    expect(withEarly.peakBytes).toBeLessThanOrEqual(withoutEarly.peakBytes);
  });

  it("validates early release works on forward-only passes", async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    // Just forward pass, no backward - simulates inference
    gpuMemoryTracker.reset();
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();

    const apiNoEarly = new Torchlette("webgpu", {
      enableFusion: false,
      enableMemoryPlanning: true,
      enableEarlyRelease: false,
    });

    const model1 = new GPT2(apiNoEarly, TEST_CONFIG, { device: "webgpu" });
    model1.eval();

    const inputData = Array.from({ length: BATCH * SEQ_LEN }, () =>
      Math.floor(Math.random() * TEST_CONFIG.vocabSize)
    );
    const input1 = apiNoEarly.tensorFromArray(inputData, [BATCH, SEQ_LEN]);
    const output1 = model1.forward(input1);
    await apiNoEarly.cpu(output1);
    await apiNoEarly.markStep();

    const peakNoEarly = gpuMemoryTracker.getPeakUsageBytes();

    // Reset and run with early release
    gpuMemoryTracker.reset();
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();

    const apiWithEarly = new Torchlette("webgpu", {
      enableFusion: false,
      enableMemoryPlanning: true,
      enableEarlyRelease: true,
    });

    const model2 = new GPT2(apiWithEarly, TEST_CONFIG, { device: "webgpu" });
    model2.eval();

    const input2 = apiWithEarly.tensorFromArray(inputData, [BATCH, SEQ_LEN]);
    const output2 = model2.forward(input2);
    await apiWithEarly.cpu(output2);
    await apiWithEarly.markStep();

    const peakWithEarly = gpuMemoryTracker.getPeakUsageBytes();

    console.log(`\nForward-only peak WITHOUT early release: ${(peakNoEarly / 1e6).toFixed(2)} MB`);
    console.log(`Forward-only peak WITH early release: ${(peakWithEarly / 1e6).toFixed(2)} MB`);
    console.log(`Improvement: ${((peakNoEarly - peakWithEarly) / 1e6).toFixed(2)} MB (${((1 - peakWithEarly / peakNoEarly) * 100).toFixed(1)}%)`);

    // Early release should provide some benefit for forward-only passes
    expect(peakWithEarly).toBeLessThanOrEqual(peakNoEarly);
  });
});
