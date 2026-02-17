/**
 * Reproducer: Dawn V100 timestamp-query + onSubmittedWorkDone deadlock
 *
 * On Tesla V100 (driver 545.23.08), requesting the "timestamp-query" device
 * feature previously caused queue.onSubmittedWorkDone() to deadlock after the
 * backward pass of a DistilGPT-2 training step.
 *
 * The bug does NOT reproduce with simple synthetic workloads — it requires the
 * specific dispatch pattern of a real training backward pass.
 *
 * Environment:
 *   - Node.js v22.22.0, webgpu (Dawn/node-webgpu) v0.3.8
 *   - Tesla V100-SXM3-32GB, NVIDIA driver 545.23.08, Vulkan 1.3.204
 *   - Linux 5.4.0-172-generic x86_64
 *
 * Usage:
 *   TORCHLETTE_PROFILE=1 npx tsx tools/repro-timestamp-deadlock.ts
 *     → Requests timestamp-query feature, tests fences at each phase boundary
 *
 *   npx tsx tools/repro-timestamp-deadlock.ts
 *     → No timestamp-query (control case)
 *
 * Diagnostic findings (from 15 test runs during initial investigation):
 *   1. Forward pass (~172 dispatches) → onSubmittedWorkDone: OK
 *   2. Backward pass (~400+ dispatches) → onSubmittedWorkDone: TIMEOUT
 *   3. Bug triggers even with NO timestampWrites (just having the feature enabled)
 *   4. Bug triggers even after explicit encoder flush before fence
 *   5. Without timestamp-query in requiredFeatures → everything works
 *   6. copyBufferToBuffer + mapAsync also deadlocks (not just onSubmittedWorkDone)
 *   7. writeBuffer + mapAsync does NOT deadlock (CPU-only path, no real GPU fence)
 *
 * Current status (2026-02-17):
 *   Bug no longer reproduces — likely fixed by dispatch pattern changes (reduced
 *   submits: 91→12/step, packed Adam: 76→8 dispatches, buffer arena, etc.).
 *   Workarounds kept as defensive measures:
 *   - issueDeferredFence uses writeBuffer+mapAsync during profiling
 *   - resolveGpuTimestamps deferred to isolated submission
 *   - setTimestampsEnabled limits timestamps to single profiled step
 *   - flushAndReadGpuTimestamps reads after forward, disables for backward
 */

import * as path from "node:path";
import { Torchlette, type Tensor } from "../src/frontend";
import {
  initWebGPU,
  destroyWebGPU,
  getWebGPUDevice,
  flushSharedEncoder,
  isF16Supported,
} from "../src/backend/webgpu";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { Adam, GradScaler } from "../src/optim";
import { crossEntropy } from "../src/nn";

const FENCE_TIMEOUT_MS = 10_000;
const PROFILING = !!process.env.TORCHLETTE_PROFILE;
const SEQ_LEN = 512;

async function testFence(label: string): Promise<string> {
  const dev = getWebGPUDevice();
  if (!dev) return "NO_CTX";

  flushSharedEncoder();

  const start = performance.now();
  const result = await Promise.race([
    dev.queue.onSubmittedWorkDone().then(() => "OK" as const),
    new Promise<"TIMEOUT">((r) => setTimeout(() => r("TIMEOUT"), FENCE_TIMEOUT_MS)),
  ]);
  const elapsed = (performance.now() - start).toFixed(1);
  console.log(`  [fence] ${label}: ${result} (${elapsed}ms)`);
  return result;
}

async function main() {
  console.log("=== Dawn V100 timestamp-query deadlock reproducer ===\n");
  console.log(`TORCHLETTE_PROFILE=${PROFILING ? "1" : "(unset)"}`);
  console.log(`timestamp-query will be ${PROFILING ? "REQUESTED" : "SKIPPED"}`);
  console.log();

  // Initialize WebGPU (requests timestamp-query when TORCHLETTE_PROFILE=1)
  const gpuOk = await initWebGPU();
  if (!gpuOk) { console.error("WebGPU init failed"); process.exit(1); }

  const dev = getWebGPUDevice()!;
  console.log(`Device features: ${[...dev.device.features].sort().join(", ")}\n`);

  // Set up model + optimizer (same as profile-training.ts)
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  console.log("Loading DistilGPT-2...");
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0 }, { device: "webgpu" });
  model.train();
  console.log("Model loaded.\n");

  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
  // AMP required to reproduce — the bug needs the larger dispatch count from f16 casts
  const useAMP = isF16Supported();
  const scaler = useAMP ? new GradScaler(api, { initScale: 2 ** 16 }) : null;
  console.log(`AMP: ${useAMP ? "enabled" : "disabled"}, seq_len: ${SEQ_LEN}\n`);

  // Compiled forward
  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    const fwd = useAMP
      ? api.autocast(() => model.forward(input))
      : model.forward(input);
    const config = model.config;
    const [batch, seqLen] = target.shape;
    const flatLogits = fwd.reshape([batch * seqLen, config.vocabSize]);
    const flatTargets = target.reshape([batch * seqLen]);
    return crossEntropy(api, flatLogits, flatTargets);
  });

  // Create input data arrays (tensors created fresh each step)
  const inputData = Array.from({ length: SEQ_LEN }, (_, i) => i % 50257);
  const targetData = Array.from({ length: SEQ_LEN }, (_, i) => (i + 1) % 50257);

  // Warmup steps (pipeline compilation + steady-state convergence)
  for (let step = 0; step < 3; step++) {
    console.log(`Step ${step} (warmup)...`);
    if (scaler) await scaler.resolveDeferred();
    const input = api.tensorFromArray(inputData, [1, SEQ_LEN], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, SEQ_LEN], { device: "webgpu" });
    const loss = compiledForward(input, target);
    const v = await loss.item();
    console.log(`  loss = ${v.toFixed(4)}`);
    if (scaler) {
      const scaledLoss = scaler.scale(loss);
      await scaledLoss.backward();
      scaler.unscale_(optimizer);
      scaler.step(optimizer);
      scaler.update();
      scaledLoss.dispose();
    } else {
      await loss.backward();
      optimizer.step();
    }
    optimizer.zeroGrad();
    loss.dispose();
    input.dispose();
    target.dispose();
    await api.markStep();
  }

  // Diagnostic step: test fences at each phase boundary
  console.log("\nDiagnostic step...");
  if (scaler) await scaler.resolveDeferred();
  const input = api.tensorFromArray(inputData, [1, SEQ_LEN], { device: "webgpu" });
  const target = api.tensorFromArray(targetData, [1, SEQ_LEN], { device: "webgpu" });

  // Forward
  console.log("  Forward pass...");
  const loss = compiledForward(input, target);
  const lossVal = await loss.item();
  console.log(`  loss = ${lossVal.toFixed(4)}`);

  const fwdResult = await testFence("after forward");

  // Backward
  console.log("  Backward pass...");
  let scaledLoss: Tensor | null = null;
  if (scaler) {
    scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();
  } else {
    await loss.backward();
  }

  const bwdResult = await testFence("after backward");

  // Optimizer
  console.log("  Optimizer step...");
  if (scaler) {
    scaler.unscale_(optimizer);
    scaler.step(optimizer);
    scaler.update();
  } else {
    optimizer.step();
  }

  const optResult = await testFence("after optimizer");

  // Cleanup
  optimizer.zeroGrad();
  if (scaledLoss) scaledLoss.dispose();
  loss.dispose();
  input.dispose();
  target.dispose();
  await api.markStep();

  // Report
  console.log("\n" + "=".repeat(60));
  console.log("RESULTS");
  console.log("=".repeat(60));
  console.log(`timestamp-query: ${PROFILING ? "ENABLED" : "DISABLED"}`);
  console.log(`Forward fence:   ${fwdResult}`);
  console.log(`Backward fence:  ${bwdResult}`);
  console.log(`Optimizer fence: ${optResult}`);

  if (bwdResult === "TIMEOUT" && fwdResult === "OK") {
    console.log("\nBUG CONFIRMED: onSubmittedWorkDone deadlocks after backward pass");
    console.log("when timestamp-query device feature is enabled on V100/Dawn.");
    console.log("\nRun without TORCHLETTE_PROFILE=1 to verify the control case works.");
  } else if (fwdResult === "OK" && bwdResult === "OK" && optResult === "OK") {
    if (PROFILING) {
      console.log("\nBug did NOT reproduce (all fences OK with timestamp-query).");
    } else {
      console.log("\nControl case: all fences work without timestamp-query.");
    }
  }

  destroyWebGPU();
  process.exit(bwdResult === "OK" ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(1); });
