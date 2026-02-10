/**
 * Checkpoint Memory Profile Tests
 *
 * Detailed memory profiling to understand where memory goes during
 * checkpoint forward/backward passes.
 *
 * Run with: npm test -- test/checkpoint-memory-profile.spec.ts
 */

import { describe, expect, it, beforeAll, afterEach } from "vitest";
import { Torchlette } from "../src/frontend";
import { initWebGPU } from "../src/backend/webgpu";
import { canUseWebGPU } from "./helpers/webgpu";
import { gpuMemoryTracker } from "../src/backend/webgpu/memory-tracker";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { resetNodeIdCounter, resetStorageIdCounter } from "../src/engine/lazy";
import { resetBaseIdCounter } from "../src/runtime/tensor";

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

describe("Checkpoint Memory Profile", () => {
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

  it("profiles memory at each phase WITHOUT checkpoint", { timeout: 120000 }, async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    console.log("\n=== Memory Profile WITHOUT Checkpoint ===\n");

    gpuMemoryTracker.reset();
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();

    const m0 = gpuMemoryTracker.getCurrentAllocatedBytes();
    console.log(`[1] Initial: ${(m0 / 1e6).toFixed(2)} MB`);

    const api = new Torchlette("webgpu", {
      enableFusion: false,
      enableMemoryPlanning: true,
    });

    const m1 = gpuMemoryTracker.getCurrentAllocatedBytes();
    console.log(`[2] After Torchlette init: ${(m1 / 1e6).toFixed(2)} MB (+${((m1 - m0) / 1e6).toFixed(2)})`);

    const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
    model.train();

    const m2 = gpuMemoryTracker.getCurrentAllocatedBytes();
    console.log(`[3] After model init: ${(m2 / 1e6).toFixed(2)} MB (+${((m2 - m1) / 1e6).toFixed(2)})`);

    const inputData = Array.from({ length: BATCH * SEQ_LEN }, () =>
      Math.floor(Math.random() * TEST_CONFIG.vocabSize)
    );
    const input = api.tensorFromArray(inputData, [BATCH, SEQ_LEN]);

    const m3 = gpuMemoryTracker.getCurrentAllocatedBytes();
    console.log(`[4] After input creation: ${(m3 / 1e6).toFixed(2)} MB (+${((m3 - m2) / 1e6).toFixed(2)})`);

    const output = model.forward(input, { useCheckpoint: false });

    const m4 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const peak4 = gpuMemoryTracker.getPeakUsageBytes();
    console.log(`[5] After forward: ${(m4 / 1e6).toFixed(2)} MB (+${((m4 - m3) / 1e6).toFixed(2)}), Peak: ${(peak4 / 1e6).toFixed(2)} MB`);

    const loss = output.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    const m5 = gpuMemoryTracker.getCurrentAllocatedBytes();
    console.log(`[6] After loss.sum(): ${(m5 / 1e6).toFixed(2)} MB (+${((m5 - m4) / 1e6).toFixed(2)})`);

    await loss.backward();

    const m6 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const peak6 = gpuMemoryTracker.getPeakUsageBytes();
    console.log(`[7] After backward: ${(m6 / 1e6).toFixed(2)} MB (+${((m6 - m5) / 1e6).toFixed(2)}), Peak: ${(peak6 / 1e6).toFixed(2)} MB`);

    await api.markStep();

    const m7 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const peakFinal = gpuMemoryTracker.getPeakUsageBytes();
    console.log(`[8] After markStep: ${(m7 / 1e6).toFixed(2)} MB (+${((m7 - m6) / 1e6).toFixed(2)}), Final Peak: ${(peakFinal / 1e6).toFixed(2)} MB`);

    console.log(`\nSummary (No Checkpoint):`);
    console.log(`  Model params: ~${((m2 - m1) / 1e6).toFixed(2)} MB`);
    console.log(`  Forward allocations: ~${((m4 - m3) / 1e6).toFixed(2)} MB`);
    console.log(`  Backward allocations: ~${((m6 - m5) / 1e6).toFixed(2)} MB`);
    console.log(`  Peak memory: ${(peakFinal / 1e6).toFixed(2)} MB`);
  });

  it("profiles memory at each phase WITH checkpoint", { timeout: 120000 }, async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    console.log("\n=== Memory Profile WITH Checkpoint ===\n");

    gpuMemoryTracker.reset();
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();

    const m0 = gpuMemoryTracker.getCurrentAllocatedBytes();
    console.log(`[1] Initial: ${(m0 / 1e6).toFixed(2)} MB`);

    const api = new Torchlette("webgpu", {
      enableFusion: false,
      enableMemoryPlanning: true,
    });

    const m1 = gpuMemoryTracker.getCurrentAllocatedBytes();
    console.log(`[2] After Torchlette init: ${(m1 / 1e6).toFixed(2)} MB (+${((m1 - m0) / 1e6).toFixed(2)})`);

    const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
    model.train();

    const m2 = gpuMemoryTracker.getCurrentAllocatedBytes();
    console.log(`[3] After model init: ${(m2 / 1e6).toFixed(2)} MB (+${((m2 - m1) / 1e6).toFixed(2)})`);

    const inputData = Array.from({ length: BATCH * SEQ_LEN }, () =>
      Math.floor(Math.random() * TEST_CONFIG.vocabSize)
    );
    const input = api.tensorFromArray(inputData, [BATCH, SEQ_LEN]);

    const m3 = gpuMemoryTracker.getCurrentAllocatedBytes();
    console.log(`[4] After input creation: ${(m3 / 1e6).toFixed(2)} MB (+${((m3 - m2) / 1e6).toFixed(2)})`);

    const output = model.forward(input, { useCheckpoint: true });

    const m4 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const peak4 = gpuMemoryTracker.getPeakUsageBytes();
    console.log(`[5] After forward (checkpointed): ${(m4 / 1e6).toFixed(2)} MB (+${((m4 - m3) / 1e6).toFixed(2)}), Peak: ${(peak4 / 1e6).toFixed(2)} MB`);

    const loss = output.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    const m5 = gpuMemoryTracker.getCurrentAllocatedBytes();
    console.log(`[6] After loss.sum(): ${(m5 / 1e6).toFixed(2)} MB (+${((m5 - m4) / 1e6).toFixed(2)})`);

    await loss.backward();

    const m6 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const peak6 = gpuMemoryTracker.getPeakUsageBytes();
    console.log(`[7] After backward (with recomputation): ${(m6 / 1e6).toFixed(2)} MB (+${((m6 - m5) / 1e6).toFixed(2)}), Peak: ${(peak6 / 1e6).toFixed(2)} MB`);

    await api.markStep();

    const m7 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const peakFinal = gpuMemoryTracker.getPeakUsageBytes();
    console.log(`[8] After markStep: ${(m7 / 1e6).toFixed(2)} MB (+${((m7 - m6) / 1e6).toFixed(2)}), Final Peak: ${(peakFinal / 1e6).toFixed(2)} MB`);

    console.log(`\nSummary (With Checkpoint):`);
    console.log(`  Model params: ~${((m2 - m1) / 1e6).toFixed(2)} MB`);
    console.log(`  Forward allocations: ~${((m4 - m3) / 1e6).toFixed(2)} MB`);
    console.log(`  Backward allocations: ~${((m6 - m5) / 1e6).toFixed(2)} MB`);
    console.log(`  Peak memory: ${(peakFinal / 1e6).toFixed(2)} MB`);
  });

  it("compares phase-by-phase memory usage", { timeout: 240000 }, async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    console.log("\n=== Phase-by-Phase Comparison ===\n");

    // Run without checkpoint
    gpuMemoryTracker.reset();
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();

    const api1 = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: true });
    const model1 = new GPT2(api1, TEST_CONFIG, { device: "webgpu" });
    model1.train();
    const inputData = Array.from({ length: BATCH * SEQ_LEN }, () =>
      Math.floor(Math.random() * TEST_CONFIG.vocabSize)
    );
    const input1 = api1.tensorFromArray(inputData, [BATCH, SEQ_LEN]);

    const preForward1 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const output1 = model1.forward(input1, { useCheckpoint: false });
    const postForward1 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const peakForward1 = gpuMemoryTracker.getPeakUsageBytes();

    const loss1 = output1.sum();
    if (typeof loss1 === "number") throw new Error("Expected tensor");
    const preBackward1 = gpuMemoryTracker.getCurrentAllocatedBytes();
    await loss1.backward();
    const postBackward1 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const peakTotal1 = gpuMemoryTracker.getPeakUsageBytes();

    // Run with checkpoint
    gpuMemoryTracker.reset();
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();

    const api2 = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: true });
    const model2 = new GPT2(api2, TEST_CONFIG, { device: "webgpu" });
    model2.train();
    const input2 = api2.tensorFromArray(inputData, [BATCH, SEQ_LEN]);

    const preForward2 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const output2 = model2.forward(input2, { useCheckpoint: true });
    const postForward2 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const peakForward2 = gpuMemoryTracker.getPeakUsageBytes();

    const loss2 = output2.sum();
    if (typeof loss2 === "number") throw new Error("Expected tensor");
    const preBackward2 = gpuMemoryTracker.getCurrentAllocatedBytes();
    await loss2.backward();
    const postBackward2 = gpuMemoryTracker.getCurrentAllocatedBytes();
    const peakTotal2 = gpuMemoryTracker.getPeakUsageBytes();

    console.log("| Phase | No Checkpoint | With Checkpoint | Diff |");
    console.log("|-------|---------------|-----------------|------|");
    console.log(`| Forward alloc | ${((postForward1 - preForward1) / 1e6).toFixed(2)} MB | ${((postForward2 - preForward2) / 1e6).toFixed(2)} MB | ${(((postForward2 - preForward2) - (postForward1 - preForward1)) / 1e6).toFixed(2)} MB |`);
    console.log(`| Forward peak | ${(peakForward1 / 1e6).toFixed(2)} MB | ${(peakForward2 / 1e6).toFixed(2)} MB | ${((peakForward2 - peakForward1) / 1e6).toFixed(2)} MB |`);
    console.log(`| Backward alloc | ${((postBackward1 - preBackward1) / 1e6).toFixed(2)} MB | ${((postBackward2 - preBackward2) / 1e6).toFixed(2)} MB | ${(((postBackward2 - preBackward2) - (postBackward1 - preBackward1)) / 1e6).toFixed(2)} MB |`);
    console.log(`| **Total peak** | **${(peakTotal1 / 1e6).toFixed(2)} MB** | **${(peakTotal2 / 1e6).toFixed(2)} MB** | **${((peakTotal2 - peakTotal1) / 1e6).toFixed(2)} MB** |`);

    const reduction = 1 - peakTotal2 / peakTotal1;
    console.log(`\nOverall reduction: ${(reduction * 100).toFixed(1)}%`);

    // Analyze where savings come from
    const forwardSavings = peakForward1 - peakForward2;
    console.log(`\nBreakdown:`);
    console.log(`  Forward peak savings: ${(forwardSavings / 1e6).toFixed(2)} MB`);
    console.log(`  (This should be main source of checkpoint savings)`);

    if (peakTotal2 > peakForward2) {
      const backwardSpike = peakTotal2 - peakForward2;
      console.log(`  Backward spike above forward: ${(backwardSpike / 1e6).toFixed(2)} MB`);
      console.log(`  (Recomputation may be causing this)`);
    }
  });
});
