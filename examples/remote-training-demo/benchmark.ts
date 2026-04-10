/**
 * Benchmark: remote vs local training at various model scales.
 *
 * Runs the same transformer training loop in two modes:
 *   1. LOCAL:  Torchlette("webgpu") — native execution, no serialization
 *   2. REMOTE: createRemoteEngine(ws) → server on same machine
 *
 * Reports: steps/sec, ms/step, and remote overhead %.
 *
 * Usage:
 *   npx tsx examples/remote-training-demo/benchmark.ts [--size s|m|l] [--steps 10]
 */

import { spawn } from "node:child_process";
import { performance } from "node:perf_hooks";
import { initWebGPU } from "../../src/backend/webgpu/index.js";
import { crossEntropy } from "../../src/nn/functional.js";
import { Torchlette } from "../../src/frontend/torchlette.js";
import type { Tensor } from "../../src/frontend/tensor.js";
import { createRemoteEngine } from "../../src/remote/client-engine.js";
import { RpcClient } from "./client/transport.js";
import {
  buildCharDataset,
  createModel,
  forward,
  parameters,
  sampleBatch,
  type ModelConfig,
  type Dataset,
  type TransformerModel,
} from "./client/model.js";

// ============================================================================
// Model configs at different scales
// ============================================================================

const CONFIGS: Record<string, ModelConfig & { label: string }> = {
  s: {
    label: "Small (D=32, 2L, 4H)",
    vocabSize: 0, // filled from dataset
    blockSize: 16,
    embedDim: 32,
    numHeads: 4,
    numLayers: 2,
    mlpRatio: 2,
  },
  m: {
    label: "Medium (D=128, 4L, 4H)",
    vocabSize: 0,
    blockSize: 64,
    embedDim: 128,
    numHeads: 4,
    numLayers: 4,
    mlpRatio: 2,
  },
  l: {
    label: "Large (D=256, 6L, 8H)",
    vocabSize: 0,
    blockSize: 128,
    embedDim: 256,
    numHeads: 8,
    numLayers: 6,
    mlpRatio: 2,
  },
};

const TRAIN_TEXT =
  "the quick brown fox jumps over the lazy dog. how vexingly quick daft zebras jump! pack my box with five dozen liquor jugs. the five boxing wizards jump quickly. sphinx of black quartz, judge my vow. ".repeat(
    10,
  );

function makePrng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return ((s >>> 0) / 0x100000000) * 2 - 1;
  };
}

// ============================================================================
// Shared training step
// ============================================================================

async function trainStep(
  api: Torchlette,
  model: TransformerModel,
  ds: Dataset,
  params: Tensor[],
  batchSize: number,
  lr: number,
  rng: () => number,
): Promise<number> {
  const { inputs, targets } = sampleBatch(ds, batchSize, model.config.blockSize, rng);

  const device = model.config.device ?? "cpu";
  const inputT = api.tensorFromArray(inputs, [batchSize, model.config.blockSize], {
    device,
    dtype: "i32",
  });
  const targetT = api.tensorFromArray(targets, [batchSize * model.config.blockSize], {
    device,
    dtype: "i32",
  });

  const logits = forward(api, model, inputT);
  const flatLogits = api.reshape(logits, [
    batchSize * model.config.blockSize,
    model.config.vocabSize,
  ]);
  const loss = crossEntropy(api, flatLogits, targetT, { reduction: "mean" });
  const lossVal = await loss.item();
  await loss.backward();

  for (const p of params) {
    if (!p.grad) continue;
    api.noGrad(() => {
      const updated = api.sub(p, api.mul(p.grad!, lr));
      p.copy_(updated);
    });
    p.zeroGrad();
  }

  return lossVal;
}

// ============================================================================
// Local benchmark
// ============================================================================

async function benchLocal(
  cfg: ModelConfig,
  warmup: number,
  steps: number,
): Promise<{ msPerStep: number[]; losses: number[] }> {
  const ok = await initWebGPU();
  if (!ok) throw new Error("WebGPU init failed");

  const api = new Torchlette("webgpu", { enableFusion: true });
  const ds = buildCharDataset(TRAIN_TEXT);
  cfg = { ...cfg, vocabSize: ds.vocabSize, device: "webgpu" };
  const model = createModel(api, cfg, 42);
  const params = parameters(model);
  const rng = makePrng(1);
  const batchSize = 4;
  const lr = 0.01;

  // Warmup
  for (let i = 0; i < warmup; i++) {
    await trainStep(api, model, ds, params, batchSize, lr, rng);
    await api.markStep();
  }

  const msPerStep: number[] = [];
  const losses: number[] = [];
  for (let i = 0; i < steps; i++) {
    const t0 = performance.now();
    const loss = await trainStep(api, model, ds, params, batchSize, lr, rng);
    await api.markStep();
    msPerStep.push(performance.now() - t0);
    losses.push(loss);
  }

  return { msPerStep, losses };
}

// ============================================================================
// Remote benchmark
// ============================================================================

async function waitForServer(url: string): Promise<void> {
  for (let i = 0; i < 40; i++) {
    try {
      const ws = new WebSocket(url);
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("timeout")), 300);
        ws.addEventListener("open", () => {
          clearTimeout(timer);
          ws.close();
          resolve();
        });
        ws.addEventListener("error", () => {
          clearTimeout(timer);
          reject(new Error("ws error"));
        });
      });
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 200));
    }
  }
  throw new Error(`Server did not start at ${url}`);
}

async function benchRemote(
  cfg: ModelConfig,
  warmup: number,
  steps: number,
  port: number,
): Promise<{ msPerStep: number[]; losses: number[]; stats: unknown }> {
  const url = `ws://localhost:${port}/ws`;

  // Start server.
  const server = spawn(
    "npx",
    [
      "tsx",
      "examples/remote-training-demo/server.ts",
      "--port",
      String(port),
    ],
    { stdio: ["ignore", "inherit", "inherit"] },
  );

  try {
    await waitForServer(url);

    // Init WebGPU client-side: createRemoteEngine builds a webgpu-flavored
    // client Torchlette so the lazy-graph dtype/op decisions match the
    // server's webgpu executor. Idempotent if benchLocal already called it.
    const ok = await initWebGPU();
    if (!ok) throw new Error("WebGPU init failed for remote client");

    const rpc = new RpcClient({ url, onLog: () => {} });
    await rpc.connect();
    const engine = createRemoteEngine(rpc);
    const { torch: api } = engine;

    const ds = buildCharDataset(TRAIN_TEXT);
    cfg = { ...cfg, vocabSize: ds.vocabSize };
    const model = createModel(api, cfg, 42);
    const params = parameters(model);
    const rng = makePrng(1);
    const batchSize = 4;
    const lr = 0.01;

    // Pre-upload model weights via binary (eliminates cold-start JSON spike).
    const uploaded = await engine.preUpload(params);
    console.log(`  [remote] pre-uploaded ${uploaded} param tensors`);

    // Warmup
    for (let i = 0; i < warmup; i++) {
      await trainStep(api, model, ds, params, batchSize, lr, rng);
      await engine.markStep(params);
    }

    const msPerStep: number[] = [];
    const losses: number[] = [];
    for (let i = 0; i < steps; i++) {
      const t0 = performance.now();
      const loss = await trainStep(api, model, ds, params, batchSize, lr, rng);
      await engine.markStep(params);
      msPerStep.push(performance.now() - t0);
      losses.push(loss);
    }

    rpc.close();
    return { msPerStep, losses, stats: engine.stats };
  } finally {
    server.kill("SIGTERM");
    await new Promise((r) => setTimeout(r, 500));
  }
}

// ============================================================================
// Report
// ============================================================================

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function report(
  label: string,
  localMs: number[],
  remoteMs: number[],
  localLosses: number[],
  remoteLosses: number[],
  remoteStats: unknown,
): void {
  const localMedian = median(localMs);
  const remoteMedian = median(remoteMs);
  const overhead = ((remoteMedian - localMedian) / localMedian) * 100;

  console.log(`\n=== ${label} ===`);
  console.log(
    `  LOCAL:  median ${localMedian.toFixed(1)}ms/step  ` +
      `(${(1000 / localMedian).toFixed(1)} steps/sec)`,
  );
  console.log(
    `  REMOTE: median ${remoteMedian.toFixed(1)}ms/step  ` +
      `(${(1000 / remoteMedian).toFixed(1)} steps/sec)`,
  );
  console.log(
    `  OVERHEAD: ${overhead > 0 ? "+" : ""}${overhead.toFixed(1)}%`,
  );
  console.log(
    `  Loss match: first local=${localLosses[0]?.toFixed(4)} remote=${remoteLosses[0]?.toFixed(4)}`,
  );

  const stats = remoteStats as Record<string, number>;
  if (stats) {
    const totalSteps = localMs.length;
    console.log(
      `  Remote stats: executes=${stats.executes} nodes=${stats.nodesShipped} ` +
        `reads=${stats.scalarReads} released=${stats.handlesReleased} ` +
        `up=${((stats.bytesUp || 0) / 1024).toFixed(0)}KB`,
    );
    if (stats.serializeMs !== undefined) {
      console.log(
        `  Per-step phase breakdown (total / ${totalSteps} steps):` +
          `  serialize=${(stats.serializeMs / totalSteps).toFixed(1)}ms` +
          `  transport=${(stats.transportMs / totalSteps).toFixed(1)}ms` +
          `  bookkeep=${(stats.bookkeepingMs / totalSteps).toFixed(1)}ms`,
      );
    }
  }

  console.log(`  Per-step detail:`);
  for (let i = 0; i < Math.min(localMs.length, 5); i++) {
    console.log(
      `    step ${i}: local=${localMs[i].toFixed(1)}ms  remote=${remoteMs[i].toFixed(1)}ms`,
    );
  }
  if (localMs.length > 5)
    console.log(`    ... (${localMs.length - 5} more)`);
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let size = "m";
  let steps = 10;
  let warmup = 3;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--size") size = args[++i];
    if (args[i] === "--steps") steps = Number(args[++i]);
    if (args[i] === "--warmup") warmup = Number(args[++i]);
  }

  const cfg = CONFIGS[size];
  if (!cfg) {
    console.error(`Unknown size "${size}". Use s, m, or l.`);
    process.exit(1);
  }

  const port = 9890;
  console.log(`Benchmark: ${cfg.label}`);
  console.log(`  warmup=${warmup} steps=${steps} batchSize=4`);

  console.log(`\nRunning LOCAL (WebGPU, same GPU)...`);
  const local = await benchLocal(cfg, warmup, steps);

  console.log(`\nRunning REMOTE (WebSocket → server on same machine)...`);
  const remote = await benchRemote(cfg, warmup, steps, port);

  report(
    cfg.label,
    local.msPerStep,
    remote.msPerStep,
    local.losses,
    remote.losses,
    remote.stats,
  );

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
