/**
 * True Segmentation Benchmark
 *
 * Measures the speed difference between:
 * - Standard execution (immediate submit per op)
 * - Checkpoint segmentation (buffer pool flush between segments)
 * - True segmentation (GPU sync between segments)
 */

import { describe, expect, it, beforeAll, afterEach } from "vitest";
import { Torchlette } from "../src/frontend";
import {
  initWebGPU,
  getBufferPoolDetailedStats,
  resetBufferPoolDetailedStats,
  syncWebGPU,
} from "../src/backend/webgpu";
import { gpuMemoryTracker } from "../src/backend/webgpu/memory-tracker";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { canUseWebGPU } from "./helpers/webgpu";
import { resetNodeIdCounter, resetStorageIdCounter } from "../src/engine/lazy";
import { resetBaseIdCounter } from "../src/runtime/tensor";

// Config for benchmarking - smaller for faster tests
const TEST_CONFIG: GPT2Config = {
  vocabSize: 500,
  blockSize: 64,
  numLayers: 3, // Small for quick memory measurement
  numHeads: 4,
  embedDim: 128,
  dropoutRate: 0.0,
};

const BATCH = 2;
const SEQ_LEN = 16;
const WARMUP_RUNS = 0;
const TIMED_RUNS = 1;

interface BenchmarkResult {
  avgTimeMs: number;
  minTimeMs: number;
  maxTimeMs: number;
  peakBytes: number;
  newAllocations: number;
}

async function runBenchmark(
  useCheckpoint: boolean,
  enableEarlyRelease: boolean,
  enableCheckpointSegmentation: boolean,
  enableTrueSegmentation: boolean,
): Promise<BenchmarkResult> {
  const times: number[] = [];
  let lastPeakBytes = 0;
  let lastNewAllocations = 0;

  for (let run = 0; run < WARMUP_RUNS + TIMED_RUNS; run++) {
    gpuMemoryTracker.reset();
    resetBufferPoolDetailedStats();
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();

    const api = new Torchlette("webgpu", {
      enableFusion: false,
      enableMemoryPlanning: false,
      enableEarlyRelease,
      enableCheckpointSegmentation,
      enableTrueSegmentation,
    });

    const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
    model.train();

    const inputData = Array.from({ length: BATCH * SEQ_LEN }, () =>
      Math.floor(Math.random() * TEST_CONFIG.vocabSize)
    );
    const input = api.tensorFromArray(inputData, [BATCH, SEQ_LEN]);

    // Ensure GPU is idle before timing
    await syncWebGPU();

    const startTime = performance.now();

    const output = model.forward(input, { useCheckpoint });
    const loss = output.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();
    await api.markStep();

    // Ensure all GPU work is done before measuring time
    await syncWebGPU();

    const endTime = performance.now();

    if (run >= WARMUP_RUNS) {
      times.push(endTime - startTime);
    }

    lastPeakBytes = gpuMemoryTracker.getPeakUsageBytes();
    const stats = getBufferPoolDetailedStats();
    lastNewAllocations = stats.acquireNew;
  }

  return {
    avgTimeMs: times.reduce((a, b) => a + b, 0) / times.length,
    minTimeMs: Math.min(...times),
    maxTimeMs: Math.max(...times),
    peakBytes: lastPeakBytes,
    newAllocations: lastNewAllocations,
  };
}

describe("True Segmentation Benchmark", { timeout: 600000 }, () => {
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

  it("measures speed difference between execution modes", async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    console.log("\n=== True Segmentation Benchmark ===\n");
    console.log(`Config: ${TEST_CONFIG.numLayers} layers, ${TEST_CONFIG.embedDim} embed, ${BATCH}x${SEQ_LEN} batch`);
    console.log(`Warmup: ${WARMUP_RUNS}, Timed runs: ${TIMED_RUNS}\n`);

    const results: Record<string, BenchmarkResult> = {};

    // 1. Baseline: no checkpoint, no early release
    console.log("Running baseline...");
    results["baseline"] = await runBenchmark(false, false, false, false);

    // 2. Early release only (no checkpoint)
    console.log("Running early release...");
    results["early-release"] = await runBenchmark(false, true, false, false);

    // 3. Checkpoint + early release (no segmentation)
    console.log("Running checkpoint + early release...");
    results["ckpt+early"] = await runBenchmark(true, true, false, false);

    // 4. Checkpoint + early release + old segmentation (buffer pool flush)
    console.log("Running checkpoint + segmentation (old)...");
    results["ckpt+seg-old"] = await runBenchmark(true, true, true, false);

    // 5. Checkpoint + early release + true segmentation (GPU sync)
    console.log("Running checkpoint + true segmentation...");
    results["ckpt+seg-true"] = await runBenchmark(true, true, false, true);

    // Print timing results
    console.log("\n| Configuration      | Avg (ms) | Min (ms) | Max (ms) | Peak MB | Allocs |");
    console.log("|--------------------|----------|----------|----------|---------|--------|");

    for (const [name, r] of Object.entries(results)) {
      console.log(
        `| ${name.padEnd(18)} | ${r.avgTimeMs.toFixed(1).padStart(8)} | ${r.minTimeMs.toFixed(1).padStart(8)} | ${r.maxTimeMs.toFixed(1).padStart(8)} | ${(r.peakBytes / 1e6).toFixed(2).padStart(7)} | ${String(r.newAllocations).padStart(6)} |`
      );
    }

    // Calculate slowdowns
    const baselineTime = results["baseline"].avgTimeMs;
    const earlyReleaseTime = results["early-release"].avgTimeMs;
    const ckptEarlyTime = results["ckpt+early"].avgTimeMs;
    const oldSegTime = results["ckpt+seg-old"].avgTimeMs;
    const trueSegTime = results["ckpt+seg-true"].avgTimeMs;

    console.log("\n=== Speed Analysis (vs baseline) ===");
    console.log(`Early release: ${((earlyReleaseTime / baselineTime - 1) * 100).toFixed(1)}% overhead`);
    console.log(`Checkpoint + early: ${((ckptEarlyTime / baselineTime - 1) * 100).toFixed(1)}% overhead`);
    console.log(`Old segmentation: ${((oldSegTime / baselineTime - 1) * 100).toFixed(1)}% overhead`);
    console.log(`True segmentation: ${((trueSegTime / baselineTime - 1) * 100).toFixed(1)}% overhead`);

    // Memory analysis
    const baselineMem = results["baseline"].peakBytes;
    const trueSegMem = results["ckpt+seg-true"].peakBytes;

    console.log("\n=== Memory Analysis ===");
    console.log(`Baseline peak: ${(baselineMem / 1e6).toFixed(2)} MB`);
    console.log(`True segmentation peak: ${(trueSegMem / 1e6).toFixed(2)} MB`);
    console.log(`Memory reduction: ${((1 - trueSegMem / baselineMem) * 100).toFixed(1)}%`);

    // Trade-off analysis
    const speedOverhead = (trueSegTime / ckptEarlyTime - 1) * 100;
    const memSavings = (1 - trueSegMem / results["ckpt+early"].peakBytes) * 100;

    console.log("\n=== True Segmentation Trade-off ===");
    console.log(`Speed overhead vs ckpt+early: ${speedOverhead.toFixed(1)}%`);
    console.log(`Memory savings vs ckpt+early: ${memSavings.toFixed(1)}%`);

    // Verify tests complete without error
    expect(results["baseline"].avgTimeMs).toBeGreaterThan(0);
    expect(results["ckpt+seg-true"].avgTimeMs).toBeGreaterThan(0);
  });
});
