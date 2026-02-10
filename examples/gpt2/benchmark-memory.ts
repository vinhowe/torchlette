/**
 * Memory Comparison Benchmark for GPT-2 Finetuning
 *
 * Compares peak memory usage and training speed across four configurations:
 * 1. Baseline (no optimizations)
 * 2. AMP only (autocast for f16 computation)
 * 3. Checkpointing only (recompute activations during backward)
 * 4. AMP + Checkpointing (both optimizations)
 *
 * Usage:
 *   TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/benchmark-memory.ts
 *   TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/benchmark-memory.ts --steps 10 --batch 4
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { gpuMemoryTracker } from "../../src/backend/webgpu/memory-tracker";
import { Adam, GradScaler } from "../../src/optim";
import { GPT2, type GPT2Config } from "./model";

// ============================================================================
// Configuration
// ============================================================================

type BenchmarkConfig = {
  numSteps: number;
  batchSize: number;
  seqLength: number;
  modelConfig: GPT2Config;
};

// Smaller config for benchmarking (faster iteration)
// Note: Reduced vocab size to avoid exceeding WebGPU workgroup limits (65535 max)
const BENCHMARK_CONFIG: GPT2Config = {
  vocabSize: 8192,   // Reduced from 50257 for WebGPU workgroup limits
  blockSize: 512,
  numLayers: 6,      // Reduced from 12 for faster benchmarking
  numHeads: 6,       // Reduced from 12
  embedDim: 384,     // Reduced from 768
  dropoutRate: 0.0,  // Disable dropout for deterministic comparison
};

function parseArgs(): BenchmarkConfig {
  const args = process.argv.slice(2);
  let numSteps = 5;
  let batchSize = 2;
  let seqLength = 64;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--steps" && args[i + 1]) {
      numSteps = parseInt(args[i + 1], 10);
      i++;
    } else if (args[i] === "--batch" && args[i + 1]) {
      batchSize = parseInt(args[i + 1], 10);
      i++;
    } else if (args[i] === "--seq" && args[i + 1]) {
      seqLength = parseInt(args[i + 1], 10);
      i++;
    } else if (args[i] === "--help") {
      console.log(`
Memory Comparison Benchmark for GPT-2 Finetuning

Usage:
  TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/benchmark-memory.ts [options]

Options:
  --steps N     Number of training steps (default: 5)
  --batch N     Batch size (default: 2)
  --seq N       Sequence length (default: 64)
  --help        Show this help message
`);
      process.exit(0);
    }
  }

  return {
    numSteps,
    batchSize,
    seqLength,
    modelConfig: BENCHMARK_CONFIG,
  };
}

// ============================================================================
// Benchmark Runner
// ============================================================================

type BenchmarkResult = {
  name: string;
  peakMemoryMB: number;
  avgTimePerStepMs: number;
  finalLoss: number;
  stepsCompleted: number;
};

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }
  return `${(bytes / 1024).toFixed(2)} KB`;
}

async function runBenchmark(
  name: string,
  config: BenchmarkConfig,
  options: {
    useAMP: boolean;
    useCheckpoint: boolean;
  },
): Promise<BenchmarkResult> {
  console.log(`\n  Running: ${name}...`);

  // Reset memory tracker and increase limit for benchmark
  gpuMemoryTracker.reset();

  // Force garbage collection if available (V8)
  if (typeof global !== "undefined" && (global as any).gc) {
    (global as any).gc();
  }

  // Create fresh API instance
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
  });

  // Create model
  const model = new GPT2(api, config.modelConfig, { device: "webgpu" });
  model.train();

  // Create optimizer
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);

  // Create GradScaler for AMP
  const scaler = options.useAMP
    ? new GradScaler(api, { initScale: 65536, enabled: true })
    : new GradScaler(api, { enabled: false });

  const times: number[] = [];
  let lastLoss = 0;
  let stepsCompleted = 0;

  try {
    for (let step = 0; step < config.numSteps; step++) {
      const stepStart = performance.now();

      // Generate random input batch
      const inputData = Array.from(
        { length: config.batchSize * config.seqLength },
        () => Math.floor(Math.random() * config.modelConfig.vocabSize),
      );
      const input = api.tensorFromArray(
        inputData,
        [config.batchSize, config.seqLength],
        { device: "webgpu" },
      );

      // Targets are input shifted by 1 (language modeling)
      const targetData = inputData.slice(1).concat([0]);
      const target = api.tensorFromArray(
        targetData,
        [config.batchSize, config.seqLength],
        { device: "webgpu" },
      );

      // Forward pass (with optional AMP and checkpointing)
      let loss: Tensor;
      if (options.useAMP) {
        loss = api.autocast(() => {
          const result = model.forwardWithLoss(input, target, {
            useCheckpoint: options.useCheckpoint,
          });
          return result.loss!;
        });
      } else {
        const result = model.forwardWithLoss(input, target, {
          useCheckpoint: options.useCheckpoint,
        });
        loss = result.loss!;
      }

      // Scale loss for AMP
      const scaledLoss = scaler.scale(loss);
      lastLoss = await loss.item();

      // Backward pass
      await scaledLoss.backward();

      // Unscale gradients and check for NaN
      scaler.unscale_(optimizer);

      // Optimizer step (skipped if NaN detected)
      scaler.step(optimizer);

      // Update scaler
      scaler.update();

      // Zero gradients
      optimizer.zeroGrad();

      // Cleanup
      input.dispose();
      target.dispose();
      loss.dispose();
      scaledLoss.dispose();

      const stepTime = performance.now() - stepStart;
      times.push(stepTime);
      stepsCompleted++;

      // Progress indicator
      process.stdout.write(".");
    }
    console.log(" done");
  } catch (error) {
    console.log(` error after ${stepsCompleted} steps`);
    console.error(`    ${error}`);
  }

  // Get peak memory
  const stats = gpuMemoryTracker.stats();
  const peakMemoryMB = stats.peakBytes / (1024 * 1024);

  // Calculate average time
  const avgTimeMs = times.length > 0
    ? times.reduce((a, b) => a + b, 0) / times.length
    : 0;

  return {
    name,
    peakMemoryMB,
    avgTimePerStepMs: avgTimeMs,
    finalLoss: lastLoss,
    stepsCompleted,
  };
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log("=".repeat(70));
  console.log("  GPT-2 Finetuning Memory Comparison Benchmark");
  console.log("=".repeat(70));

  // Initialize WebGPU
  console.log("\nInitializing WebGPU...");
  const success = await initWebGPU();
  if (!success) {
    console.error("Failed to initialize WebGPU");
    process.exit(1);
  }

  const config = parseArgs();

  console.log("\nBenchmark Configuration:");
  console.log(`  Steps:        ${config.numSteps}`);
  console.log(`  Batch size:   ${config.batchSize}`);
  console.log(`  Seq length:   ${config.seqLength}`);
  console.log(`  Model layers: ${config.modelConfig.numLayers}`);
  console.log(`  Embed dim:    ${config.modelConfig.embedDim}`);
  console.log(`  Num heads:    ${config.modelConfig.numHeads}`);

  console.log("\nRunning benchmarks...");

  // Run all four configurations
  const results: BenchmarkResult[] = [];

  // 1. Baseline (no optimizations)
  results.push(
    await runBenchmark("Baseline", config, {
      useAMP: false,
      useCheckpoint: false,
    }),
  );

  // 2. AMP only
  results.push(
    await runBenchmark("AMP Only", config, {
      useAMP: true,
      useCheckpoint: false,
    }),
  );

  // 3. Checkpointing only
  results.push(
    await runBenchmark("Checkpoint Only", config, {
      useAMP: false,
      useCheckpoint: true,
    }),
  );

  // 4. AMP + Checkpointing
  results.push(
    await runBenchmark("AMP + Checkpoint", config, {
      useAMP: true,
      useCheckpoint: true,
    }),
  );

  // Print results table
  console.log("\n" + "=".repeat(70));
  console.log("  Results");
  console.log("=".repeat(70));

  console.log("\n┌─────────────────────┬──────────────┬──────────────┬──────────────┬────────┐");
  console.log("│ Configuration       │ Peak Memory  │ Time/Step    │ Final Loss   │ Steps  │");
  console.log("├─────────────────────┼──────────────┼──────────────┼──────────────┼────────┤");

  const baseline = results[0];

  for (const r of results) {
    const memSavings = baseline.peakMemoryMB > 0
      ? ((1 - r.peakMemoryMB / baseline.peakMemoryMB) * 100).toFixed(1)
      : "0.0";
    const speedup = baseline.avgTimePerStepMs > 0
      ? (baseline.avgTimePerStepMs / r.avgTimePerStepMs).toFixed(2)
      : "1.00";

    const memStr = `${r.peakMemoryMB.toFixed(1)} MB`;
    const timeStr = `${r.avgTimePerStepMs.toFixed(1)} ms`;
    const lossStr = r.finalLoss.toFixed(4);

    console.log(
      `│ ${r.name.padEnd(19)} │ ${memStr.padStart(12)} │ ${timeStr.padStart(12)} │ ${lossStr.padStart(12)} │ ${String(r.stepsCompleted).padStart(6)} │`,
    );
  }

  console.log("└─────────────────────┴──────────────┴──────────────┴──────────────┴────────┘");

  // Print savings summary
  console.log("\n" + "-".repeat(70));
  console.log("  Observed Results vs Baseline");
  console.log("-".repeat(70));

  for (let i = 1; i < results.length; i++) {
    const r = results[i];
    const memSavings = baseline.peakMemoryMB > 0
      ? ((1 - r.peakMemoryMB / baseline.peakMemoryMB) * 100)
      : 0;
    const speedChange = baseline.avgTimePerStepMs > 0
      ? ((r.avgTimePerStepMs / baseline.avgTimePerStepMs - 1) * 100)
      : 0;

    const memSign = memSavings >= 0 ? "-" : "+";
    const speedSign = speedChange >= 0 ? "+" : "";

    console.log(
      `  ${r.name.padEnd(20)}: ${memSign}${Math.abs(memSavings).toFixed(1)}% memory, ` +
      `${speedSign}${speedChange.toFixed(1)}% time`,
    );
  }

  // Print theoretical checkpoint savings
  console.log("\n" + "-".repeat(70));
  console.log("  Theoretical Checkpoint Savings (when fully implemented)");
  console.log("-".repeat(70));

  // Import estimator
  const { estimateCheckpointSavings } = await import("../../src/nn/checkpoint");
  const savings = estimateCheckpointSavings(
    config.modelConfig.numLayers,
    config.modelConfig.embedDim,
    config.seqLength,
    config.batchSize,
    "all",
  );

  console.log(`  Activation memory without checkpoint: ${formatBytes(savings.withoutCheckpoint)}`);
  console.log(`  Activation memory with checkpoint:    ${formatBytes(savings.withCheckpoint)}`);
  console.log(`  Estimated savings:                    ${savings.savingsPercent.toFixed(1)}%`);
  console.log("\n  Note: Pack/unpack hooks are implemented and gradient correctness works.");
  console.log("  However, memory savings require the autograd to not keep tensor refs in");
  console.log("  the inputs array. This needs opaque checkpoint nodes - a deeper change.")

  console.log("\n" + "=".repeat(70));
  console.log("  Benchmark Complete");
  console.log("=".repeat(70));

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
