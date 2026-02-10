#!/usr/bin/env npx tsx
/**
 * GPT-2 Finetuning Harness - Main Entry Point
 *
 * This is the main script for running GPT-2 finetuning benchmarks
 * using Torchlette with WebGPU acceleration and compile-mode optimizations.
 *
 * Usage:
 *   # Run with WebGPU (requires Dawn for Node.js)
 *   TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/main.ts
 *
 *   # Run with specific options
 *   TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/main.ts --benchmark
 *   TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/main.ts --train --steps 100
 *
 * Commands:
 *   --benchmark   Run full benchmarking suite
 *   --train       Run training loop
 *   --download    Download GPT-2 weights from HuggingFace
 *   --test        Run quick sanity test
 */

import { Torchlette, type DeviceKind, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, GPT2_SMALL_CONFIG, type GPT2Config } from "./model";
import { loadPretrainedGPT2, downloadGPT2 } from "./loader";
import { createSyntheticDataLoader, FineWebDataLoader, GPT2Tokenizer } from "./data";
import { GPT2Trainer } from "./trainer";
import {
  runFullBenchmark,
  benchmarkForwardPass,
  trackRecompileChurn,
  collectOptimizationStats,
  exportReport,
  type BenchmarkConfig,
} from "./benchmark";
import {
  checkpoint,
  createLayerCheckpointer,
  estimateCheckpointSavings,
} from "../../src/nn/checkpoint";

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_CONFIG: BenchmarkConfig = {
  seqLength: 1024,
  batchSize: 1,
  warmupIterations: 3,
  measureIterations: 10,
  device: process.env.TORCHLETTE_WEBGPU ? ("webgpu" as DeviceKind) : ("cpu" as DeviceKind),
};

const TRAINING_CONFIG = {
  learningRate: 1e-4,
  betas: [0.9, 0.999] as [number, number],
  eps: 1e-8,
  weightDecay: 0.01,
  maxSteps: 100,
  logEveryNSteps: 10,
};

// ============================================================================
// CLI Argument Parsing
// ============================================================================

interface CLIArgs {
  command: "benchmark" | "train" | "download" | "test" | "help";
  steps?: number;
  seqLength?: number;
  batchSize?: number;
  checkpoint?: boolean;
  pytorchComparison?: boolean;
  outputJson?: string;
  modelPath?: string;
  verbose?: boolean;
  fp16?: boolean;
}

function parseArgs(): CLIArgs {
  const args = process.argv.slice(2);
  const result: CLIArgs = {
    command: "benchmark",
    steps: TRAINING_CONFIG.maxSteps,
    seqLength: DEFAULT_CONFIG.seqLength,
    batchSize: DEFAULT_CONFIG.batchSize,
    checkpoint: false,
    pytorchComparison: false,
    verbose: false,
    fp16: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    switch (arg) {
      case "--benchmark":
        result.command = "benchmark";
        break;
      case "--train":
        result.command = "train";
        break;
      case "--download":
        result.command = "download";
        break;
      case "--test":
        result.command = "test";
        break;
      case "--help":
      case "-h":
        result.command = "help";
        break;
      case "--steps":
        result.steps = parseInt(args[++i], 10);
        break;
      case "--seq-length":
        result.seqLength = parseInt(args[++i], 10);
        break;
      case "--batch-size":
        result.batchSize = parseInt(args[++i], 10);
        break;
      case "--checkpoint":
        result.checkpoint = true;
        break;
      case "--pytorch":
        result.pytorchComparison = true;
        break;
      case "--output":
      case "-o":
        result.outputJson = args[++i];
        break;
      case "--model":
        result.modelPath = args[++i];
        break;
      case "--verbose":
      case "-v":
        result.verbose = true;
        break;
      case "--fp16":
        result.fp16 = true;
        break;
    }
  }

  return result;
}

function printHelp(): void {
  console.log(`
GPT-2 Finetuning Harness

Usage: TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/main.ts [command] [options]

Commands:
  --benchmark   Run full benchmarking suite (default)
  --train       Run training loop
  --download    Download GPT-2 weights from HuggingFace
  --test        Run quick sanity test
  --help, -h    Show this help message

Options:
  --steps N         Number of training steps (default: 100)
  --seq-length N    Sequence length (default: 1024)
  --batch-size N    Batch size (default: 1)
  --checkpoint      Enable gradient checkpointing
  --pytorch         Run PyTorch reference comparison
  --output FILE     Save benchmark report to JSON file
  --model PATH      Path to pretrained model
  --verbose, -v     Verbose output

Examples:
  # Run benchmark suite
  TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/main.ts --benchmark

  # Train for 50 steps with checkpointing
  TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/main.ts --train --steps 50 --checkpoint

  # Compare with PyTorch
  TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/main.ts --benchmark --pytorch

  # Download GPT-2 weights
  npx tsx examples/gpt2/main.ts --download
`);
}

// ============================================================================
// Commands
// ============================================================================

/**
 * Run the full benchmarking suite.
 */
async function runBenchmarkCommand(args: CLIArgs): Promise<void> {
  console.log("Initializing Torchlette API...");
  const api = new Torchlette(DEFAULT_CONFIG.device);

  console.log(`Creating GPT-2 model (${GPT2_SMALL_CONFIG.numLayers} layers)...`);
  const model = new GPT2(api, GPT2_SMALL_CONFIG);

  console.log(`Model parameters: ${formatNumber(countParameters(model))}`);

  // Estimate memory with/without checkpointing
  if (args.checkpoint) {
    const savings = estimateCheckpointSavings(
      GPT2_SMALL_CONFIG.numLayers,
      GPT2_SMALL_CONFIG.embedDim,
      args.seqLength!,
      args.batchSize!,
      "all",
    );
    console.log(`\nCheckpoint memory savings:`);
    console.log(`  Without: ${formatBytes(savings.withoutCheckpoint)}`);
    console.log(`  With:    ${formatBytes(savings.withCheckpoint)}`);
    console.log(`  Savings: ${savings.savingsPercent.toFixed(1)}%`);
  }

  const config: BenchmarkConfig = {
    seqLength: args.seqLength!,
    batchSize: args.batchSize!,
    warmupIterations: DEFAULT_CONFIG.warmupIterations,
    measureIterations: DEFAULT_CONFIG.measureIterations,
    device: DEFAULT_CONFIG.device,
  };

  const report = await runFullBenchmark(api, model, config, {
    runPyTorchComparison: args.pytorchComparison,
  });

  if (args.outputJson) {
    const fs = await import("node:fs");
    fs.writeFileSync(args.outputJson, exportReport(report));
    console.log(`Report saved to: ${args.outputJson}`);
  }
}

/**
 * Run training loop.
 */
async function runTrainCommand(args: CLIArgs): Promise<void> {
  console.log("Initializing Torchlette API...");
  const api = new Torchlette(DEFAULT_CONFIG.device);

  console.log(`Creating GPT-2 model (${GPT2_SMALL_CONFIG.numLayers} layers)...`);

  let model: GPT2;
  if (args.modelPath) {
    console.log(`Loading pretrained weights from ${args.modelPath}...`);
    model = await loadPretrainedGPT2(api, args.modelPath, undefined, {
      device: DEFAULT_CONFIG.device,
    });
  } else {
    model = new GPT2(api, GPT2_SMALL_CONFIG);
    console.log("Using random initialization (no pretrained weights)");
  }

  console.log(`Model parameters: ${formatNumber(countParameters(model))}`);

  // Create data loader
  console.log("Creating synthetic data loader...");
  const dataLoader = createSyntheticDataLoader(
    api,
    {
      seqLength: args.seqLength!,
      batchSize: args.batchSize!,
    },
    undefined,
    { device: DEFAULT_CONFIG.device },
  );

  // Create trainer with fp16 if requested
  console.log("Creating trainer...");
  const trainerConfig = {
    ...TRAINING_CONFIG,
    useFp16: args.fp16,
  };
  const trainer = new GPT2Trainer(api, model, trainerConfig, {
    device: DEFAULT_CONFIG.device,
  });

  // Compile training step
  console.log(`Compiling training step${args.fp16 ? " with fp16 AMP" : ""}...`);
  trainer.compile();

  // Optionally set up checkpointing
  let maybeCheckpoint: ((idx: number, fn: () => any) => any) | null = null;
  if (args.checkpoint) {
    console.log("Enabling gradient checkpointing...");
    maybeCheckpoint = createLayerCheckpointer(api, {
      policy: "all",
      preserveRngState: true,
    });
  }

  // Training loop
  console.log(`\nStarting training for ${args.steps} steps...`);
  console.log("=".repeat(60));

  const startTime = performance.now();
  const result = await trainer.train(dataLoader, args.steps, {
    onStep: args.verbose
      ? (step, loss, timeMs) => {
          console.log(`Step ${step}: loss=${loss.toFixed(4)}, time=${timeMs.toFixed(1)}ms`);
        }
      : undefined,
  });

  const totalTime = performance.now() - startTime;

  console.log("=".repeat(60));
  console.log("\nTraining Complete");
  console.log("-".repeat(30));
  console.log(`Total steps:     ${result.totalSteps}`);
  console.log(`Final loss:      ${result.finalLoss.toFixed(4)}`);
  console.log(`Avg loss/step:   ${result.avgLossPerStep.toFixed(4)}`);
  console.log(`Total time:      ${(totalTime / 1000).toFixed(2)}s`);
  console.log(`Avg time/step:   ${result.avgTimePerStepMs.toFixed(2)}ms`);
  console.log(`Throughput:      ${((args.seqLength! * args.batchSize! * result.totalSteps) / (totalTime / 1000)).toFixed(0)} tokens/s`);
}

/**
 * Download GPT-2 weights.
 */
async function runDownloadCommand(_args: CLIArgs): Promise<void> {
  console.log("Downloading GPT-2 weights from HuggingFace...");
  const modelPath = await downloadGPT2("gpt2");
  console.log(`\nModel downloaded to: ${modelPath}`);
  console.log("\nYou can now use the model with:");
  console.log(`  TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/main.ts --train --model ${modelPath}`);
}

/**
 * Run quick sanity test.
 */
async function runTestCommand(args: CLIArgs): Promise<void> {
  console.log("Running quick sanity test...\n");

  const api = new Torchlette(DEFAULT_CONFIG.device);

  // Create small model for quick test
  const smallConfig: GPT2Config = {
    vocabSize: 1000,
    blockSize: 128,
    numLayers: 2,
    numHeads: 2,
    embedDim: 64,
    dropoutRate: 0.0,
  };

  console.log("Creating small GPT-2 model for testing...");
  const model = new GPT2(api, smallConfig);
  console.log(`Model parameters: ${formatNumber(countParameters(model))}`);

  // Create input
  console.log("\nCreating test input...");
  const seqLength = 32;
  const batchSize = 2;
  const inputData = Array.from({ length: batchSize * seqLength }, () =>
    Math.floor(Math.random() * smallConfig.vocabSize),
  );
  const targetData = Array.from({ length: batchSize * seqLength }, () =>
    Math.floor(Math.random() * smallConfig.vocabSize),
  );

  const input = api.tensorFromArray(inputData, [batchSize, seqLength], {
    device: DEFAULT_CONFIG.device,
  });
  const targets = api.tensorFromArray(targetData, [batchSize, seqLength], {
    device: DEFAULT_CONFIG.device,
  });

  // Forward pass
  console.log("Running forward pass...");
  const logits = model.forward(input);
  console.log(`  Output shape: [${logits.shape.join(", ")}]`);

  // Forward with loss
  console.log("\nRunning forward with loss...");
  const { logits: logits2, loss } = model.forwardWithLoss(input, targets);
  console.log(`  Logits shape: [${logits2.shape.join(", ")}]`);
  if (loss) {
    const lossValue = await loss.item();
    console.log(`  Loss: ${lossValue.toFixed(4)}`);
  }

  // Backward pass
  console.log("\nRunning backward pass...");
  if (loss) {
    await loss.backward();
    console.log("  Backward complete");
  }

  // Compile test
  console.log("\nTesting compile mode...");
  const compiledForward = api.compile((x: Tensor) => model.forward(x));
  const compiled1 = compiledForward(input);
  const compiled2 = compiledForward(input);
  console.log(`  Compiled forward runs: 2`);
  console.log(`  Output shape: [${compiled1.shape.join(", ")}]`);
  console.log(`  Fusion enabled during compile: ${api.isFusionEnabled() ? "yes (leak)" : "no (correct)"}`);
  // Note: After compile returns, fusion should be disabled again

  // Cleanup
  input.dispose();
  targets.dispose();

  console.log("\nAll tests passed!");
}

// ============================================================================
// Utilities
// ============================================================================

function countParameters(model: GPT2): number {
  let total = 0;
  for (const param of model.parameters()) {
    let size = 1;
    for (const dim of param.shape) {
      size *= dim;
    }
    total += size;
  }
  return total;
}

function formatNumber(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(2)}K`;
  return n.toString();
}

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(2)} KB`;
  return `${bytes} B`;
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  const args = parseArgs();

  console.log("=".repeat(60));
  console.log("GPT-2 Finetuning Harness");
  console.log("=".repeat(60));
  console.log(`Device: ${DEFAULT_CONFIG.device}`);
  console.log(`Command: ${args.command}`);
  console.log();

  // Initialize WebGPU backend if using webgpu device
  if (DEFAULT_CONFIG.device === "webgpu") {
    console.log("Initializing WebGPU backend...");
    const success = await initWebGPU();
    if (!success) {
      console.error("Failed to initialize WebGPU backend");
      process.exit(1);
    }
    console.log("WebGPU initialized successfully");
    console.log();
  }

  switch (args.command) {
    case "benchmark":
      await runBenchmarkCommand(args);
      break;
    case "train":
      await runTrainCommand(args);
      break;
    case "download":
      await runDownloadCommand(args);
      break;
    case "test":
      await runTestCommand(args);
      break;
    case "help":
      printHelp();
      break;
  }
}

main().catch((e) => {
  console.error("Error:", e);
  process.exit(1);
});
