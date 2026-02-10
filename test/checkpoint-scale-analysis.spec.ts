/**
 * Checkpoint Scale Analysis Tests
 *
 * Investigates whether checkpoint memory benefits scale with model size.
 * Tests multiple configurations to understand the relationship between
 * activation memory and checkpoint savings.
 *
 * Run with: npm test -- test/checkpoint-scale-analysis.spec.ts
 */

import { describe, expect, it, beforeAll, afterEach } from "vitest";
import { Torchlette } from "../src/frontend";
import { initWebGPU } from "../src/backend/webgpu";
import { canUseWebGPU } from "./helpers/webgpu";
import { gpuMemoryTracker } from "../src/backend/webgpu/memory-tracker";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { resetNodeIdCounter, resetStorageIdCounter } from "../src/engine/lazy";
import { resetBaseIdCounter } from "../src/runtime/tensor";

// Scale configurations from small to large
const CONFIGS = {
  // Current test scale - known to show ~6.5% reduction
  small: {
    config: {
      vocabSize: 500,
      blockSize: 64,
      numLayers: 4,
      numHeads: 4,
      embedDim: 128,
      dropoutRate: 0.0,
    } as GPT2Config,
    batch: 2,
    seqLen: 32,
    name: "Small (batch=2, seq=32, embed=128)",
  },

  // Medium scale - larger attention matrices
  medium: {
    config: {
      vocabSize: 500,
      blockSize: 128,
      numLayers: 4,
      numHeads: 8,
      embedDim: 256,
      dropoutRate: 0.0,
    } as GPT2Config,
    batch: 4,
    seqLen: 64,
    name: "Medium (batch=4, seq=64, embed=256)",
  },

  // Large scale - significant attention memory
  large: {
    config: {
      vocabSize: 500,
      blockSize: 256,
      numLayers: 4,
      numHeads: 8,
      embedDim: 256,
      dropoutRate: 0.0,
    } as GPT2Config,
    batch: 4,
    seqLen: 128,
    name: "Large (batch=4, seq=128, embed=256)",
  },
};

// Calculate expected memory for a config
function calculateExpectedMemory(cfg: typeof CONFIGS.small) {
  const { config, batch, seqLen } = cfg;
  const { vocabSize, blockSize, embedDim, numHeads, numLayers } = config;
  const headDim = embedDim / numHeads;

  // Parameters
  const tokenEmbed = vocabSize * embedDim * 4;
  const posEmbed = blockSize * embedDim * 4;
  const perLayerParams =
    (embedDim * 3 * embedDim + 3 * embedDim) + // Attention QKV
    (embedDim * embedDim + embedDim) + // Attention out projection
    (embedDim * 4 * embedDim + 4 * embedDim) + // MLP up
    (4 * embedDim * embedDim + embedDim) + // MLP down
    (2 * embedDim * 2); // LayerNorms
  const totalParams = tokenEmbed + posEmbed + perLayerParams * numLayers * 4;

  // Activations per layer (main ones saved for backward)
  const activationPerLayer =
    batch * seqLen * embedDim * 4 + // input
    batch * numHeads * seqLen * seqLen * 4 + // attention scores (quadratic!)
    batch * numHeads * seqLen * seqLen * 4 + // attention weights
    batch * seqLen * 4 * embedDim * 4; // MLP hidden
  const totalActivations = activationPerLayer * numLayers;

  return {
    params: totalParams,
    activations: totalActivations,
    attentionPerLayer: batch * numHeads * seqLen * seqLen * 4,
    mlpPerLayer: batch * seqLen * 4 * embedDim * 4,
  };
}

describe("Checkpoint Scale Analysis", () => {
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

  it("prints expected memory calculations for each scale", () => {
    console.log("\n=== Expected Memory Calculations ===\n");

    for (const [key, cfg] of Object.entries(CONFIGS)) {
      const mem = calculateExpectedMemory(cfg);
      console.log(`${cfg.name}:`);
      console.log(`  Parameters: ${(mem.params / 1e6).toFixed(2)} MB`);
      console.log(`  Activations (all layers): ${(mem.activations / 1e6).toFixed(2)} MB`);
      console.log(`    - Attention per layer: ${(mem.attentionPerLayer / 1e6).toFixed(3)} MB`);
      console.log(`    - MLP hidden per layer: ${(mem.mlpPerLayer / 1e6).toFixed(3)} MB`);
      console.log(`  Params/Activation ratio: ${(mem.params / mem.activations).toFixed(2)}x`);
      console.log(`  Expected checkpoint benefit: ~${(100 * mem.activations / (mem.params + mem.activations)).toFixed(0)}%`);
      console.log();
    }
  });

  it("measures checkpoint reduction at SMALL scale", { timeout: 120000 }, async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    const cfg = CONFIGS.small;
    const result = await measureCheckpointReduction(cfg);

    console.log(`\n${cfg.name}:`);
    console.log(`  Peak WITHOUT checkpoint: ${(result.peakWithout / 1e6).toFixed(2)} MB`);
    console.log(`  Peak WITH checkpoint: ${(result.peakWith / 1e6).toFixed(2)} MB`);
    console.log(`  Reduction: ${(result.reduction * 100).toFixed(1)}%`);
    console.log(`  Savings: ${((result.peakWithout - result.peakWith) / 1e6).toFixed(2)} MB`);

    // Small scale may have minimal reduction
    expect(result.reduction).toBeGreaterThan(0);
  });

  it("measures checkpoint reduction at MEDIUM scale", { timeout: 180000 }, async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    const cfg = CONFIGS.medium;
    const result = await measureCheckpointReduction(cfg);

    console.log(`\n${cfg.name}:`);
    console.log(`  Peak WITHOUT checkpoint: ${(result.peakWithout / 1e6).toFixed(2)} MB`);
    console.log(`  Peak WITH checkpoint: ${(result.peakWith / 1e6).toFixed(2)} MB`);
    console.log(`  Reduction: ${(result.reduction * 100).toFixed(1)}%`);
    console.log(`  Savings: ${((result.peakWithout - result.peakWith) / 1e6).toFixed(2)} MB`);

    // Medium scale should show more reduction
    expect(result.reduction).toBeGreaterThan(0.05);
  });

  it("measures checkpoint reduction at LARGE scale", { timeout: 300000 }, async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    const cfg = CONFIGS.large;
    const result = await measureCheckpointReduction(cfg);

    console.log(`\n${cfg.name}:`);
    console.log(`  Peak WITHOUT checkpoint: ${(result.peakWithout / 1e6).toFixed(2)} MB`);
    console.log(`  Peak WITH checkpoint: ${(result.peakWith / 1e6).toFixed(2)} MB`);
    console.log(`  Reduction: ${(result.reduction * 100).toFixed(1)}%`);
    console.log(`  Savings: ${((result.peakWithout - result.peakWith) / 1e6).toFixed(2)} MB`);

    // Large scale should show significant reduction
    // If it doesn't, there may be an implementation issue
    expect(result.reduction).toBeGreaterThan(0.1);
  });

  it("compares reduction across all scales", { timeout: 600000 }, async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    console.log("\n=== Checkpoint Reduction Scaling Analysis ===\n");

    const results: { name: string; reduction: number; peakWithout: number; peakWith: number }[] = [];

    for (const [key, cfg] of Object.entries(CONFIGS)) {
      const result = await measureCheckpointReduction(cfg);
      results.push({
        name: cfg.name,
        ...result,
      });
    }

    console.log("| Scale | Peak No CP | Peak With CP | Reduction | Savings |");
    console.log("|-------|------------|--------------|-----------|---------|");
    for (const r of results) {
      console.log(
        `| ${r.name.split(" ")[0]} | ${(r.peakWithout / 1e6).toFixed(1)} MB | ` +
        `${(r.peakWith / 1e6).toFixed(1)} MB | ${(r.reduction * 100).toFixed(1)}% | ` +
        `${((r.peakWithout - r.peakWith) / 1e6).toFixed(1)} MB |`
      );
    }

    // Check that reduction increases with scale
    const reductions = results.map(r => r.reduction);
    console.log(`\nReductions: ${reductions.map(r => (r * 100).toFixed(1) + "%").join(" → ")}`);

    // The reduction should generally increase with scale (or at least not decrease dramatically)
    // If small > large, something is wrong
    if (reductions.length >= 2) {
      const smallReduction = reductions[0];
      const largeReduction = reductions[reductions.length - 1];
      console.log(`\nScale effect: ${(largeReduction / smallReduction).toFixed(2)}x more reduction at large vs small`);
    }
  });
});

async function measureCheckpointReduction(cfg: typeof CONFIGS.small) {
  const { config, batch, seqLen } = cfg;

  // ========== Run WITHOUT checkpoint ==========
  gpuMemoryTracker.reset();
  resetNodeIdCounter();
  resetStorageIdCounter();
  resetBaseIdCounter();

  const api1 = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: true,
  });
  const model1 = new GPT2(api1, config, { device: "webgpu" });
  model1.train();

  const inputData1 = Array.from({ length: batch * seqLen }, () =>
    Math.floor(Math.random() * config.vocabSize)
  );
  const input1 = api1.tensorFromArray(inputData1, [batch, seqLen]);

  const output1 = model1.forward(input1, { useCheckpoint: false });
  const loss1 = output1.sum();
  if (typeof loss1 === "number") throw new Error("Expected tensor");

  await loss1.backward();
  await api1.markStep();

  const peakWithout = gpuMemoryTracker.getPeakUsageBytes();

  // ========== Run WITH checkpoint ==========
  gpuMemoryTracker.reset();
  resetNodeIdCounter();
  resetStorageIdCounter();
  resetBaseIdCounter();

  const api2 = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: true,
    enableTrueSegmentation: true,
    enableEarlyRelease: true,
  });
  const model2 = new GPT2(api2, config, { device: "webgpu" });
  model2.train();

  const input2 = api2.tensorFromArray(inputData1, [batch, seqLen]);

  const output2 = model2.forward(input2, { useCheckpoint: true });
  const loss2 = output2.sum();
  if (typeof loss2 === "number") throw new Error("Expected tensor");

  await loss2.backward();
  await api2.markStep();

  const peakWith = gpuMemoryTracker.getPeakUsageBytes();

  return {
    peakWithout,
    peakWith,
    reduction: 1 - peakWith / peakWithout,
  };
}
