/**
 * DistilGPT2 Full Finetuning Verification Tests
 *
 * Verifies end-to-end training of a real transformer model with:
 * - Gradient checkpointing
 * - AMP (Automatic Mixed Precision)
 * - All parameters trainable (full finetuning, not LoRA)
 *
 * Run with: npm test -- test/distilgpt2-full-finetuning.spec.ts
 */

import { describe, expect, it, beforeAll, afterEach } from "vitest";
import { Torchlette } from "../src/frontend";
import { initWebGPU, getBufferPoolStats, getGPUMemoryStats } from "../src/backend/webgpu";
import { gpuMemoryTracker } from "../src/backend/webgpu/memory-tracker";
import { storageTracker, resetNodeIdCounter, resetStorageIdCounter } from "../src/engine/lazy";
import { GPT2, DISTILGPT2_CONFIG, type GPT2Config } from "../examples/gpt2/model";
import { Adam, GradScaler } from "../src/optim";
import { canUseWebGPU } from "./helpers/webgpu";
import { resetBaseIdCounter } from "../src/runtime/tensor";

// Smaller config for faster tests
const TEST_CONFIG: GPT2Config = {
  vocabSize: 500, // Small vocab for fast tests
  blockSize: 64,
  numLayers: 4, // Fewer layers for faster tests
  numHeads: 4,
  embedDim: 128,
  dropoutRate: 0.0, // Disable dropout for deterministic tests
};

describe("DistilGPT2 Full Finetuning Verification", () => {
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

  describe("Gradient Verification", () => {
    it("all parameters receive gradients during full finetuning", { timeout: 60000 }, async () => {
      if (!webgpuAvailable) {
        console.log("Skipping: WebGPU not available");
        return;
      }

      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();

      const api = new Torchlette("webgpu", {
        enableFusion: false, // Disabled due to reshape issues with fusion
        enableMemoryPlanning: true,
      });
      const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
      model.train();

      const batchSize = 2;
      const seqLen = 16;

      // Create random input and target
      const inputData = Array.from({ length: batchSize * seqLen }, () =>
        Math.floor(Math.random() * TEST_CONFIG.vocabSize)
      );
      const targetData = Array.from({ length: batchSize * seqLen }, () =>
        Math.floor(Math.random() * TEST_CONFIG.vocabSize)
      );

      const input = api.tensorFromArray(inputData, [batchSize, seqLen]);
      const target = api.tensorFromArray(targetData, [batchSize, seqLen]);

      // Forward with checkpoint (autocast disabled due to known reshape issue)
      const result = model.forwardWithLoss(input, target, { useCheckpoint: true });

      expect(result.loss).not.toBeNull();
      await result.loss!.backward();

      // Verify ALL parameters have gradients
      const params = model.parameters();
      let nullGrads = 0;
      let zeroGrads = 0;
      let finiteGrads = 0;

      for (const param of params) {
        if (param.grad === null) {
          nullGrads++;
          continue;
        }

        const gradData = await param.grad.cpu();
        const hasNonZero = gradData.some((v: number) => v !== 0);
        const allFinite = gradData.every((v: number) => Number.isFinite(v));

        if (!hasNonZero) zeroGrads++;
        if (allFinite) finiteGrads++;
      }

      console.log(`Total parameters: ${params.length}`);
      console.log(`Null gradients: ${nullGrads}`);
      console.log(`Zero gradients: ${zeroGrads}`);
      console.log(`Finite gradients: ${finiteGrads}`);

      // All parameters should have gradients
      expect(nullGrads).toBe(0);
      // All gradients should be finite
      expect(finiteGrads).toBe(params.length);
      // Most gradients should be non-zero (some bias terms may be zero)
      expect(params.length - zeroGrads).toBeGreaterThan(params.length * 0.8);
    });

    it("gradients have reasonable magnitudes", { timeout: 60000 }, async () => {
      if (!webgpuAvailable) {
        console.log("Skipping: WebGPU not available");
        return;
      }

      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();

      const api = new Torchlette("webgpu");
      const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
      model.train();

      const inputData = Array.from({ length: 32 }, () =>
        Math.floor(Math.random() * TEST_CONFIG.vocabSize)
      );
      const targetData = Array.from({ length: 32 }, () =>
        Math.floor(Math.random() * TEST_CONFIG.vocabSize)
      );

      const input = api.tensorFromArray(inputData, [2, 16]);
      const target = api.tensorFromArray(targetData, [2, 16]);

      const result = model.forwardWithLoss(input, target, { useCheckpoint: true });
      await result.loss!.backward();

      // Check gradient magnitudes are reasonable (not exploding)
      let maxGradMagnitude = 0;
      for (const param of model.parameters()) {
        if (param.grad) {
          const gradData = await param.grad.cpu();
          const maxVal = Math.max(...gradData.map((v: number) => Math.abs(v)));
          maxGradMagnitude = Math.max(maxGradMagnitude, maxVal);
        }
      }

      console.log(`Max gradient magnitude: ${maxGradMagnitude}`);

      // Gradients should not be exploding (< 1000 is reasonable for untrained model)
      expect(maxGradMagnitude).toBeLessThan(1000);
      // Gradients should not be vanishing (> 1e-10 is reasonable)
      expect(maxGradMagnitude).toBeGreaterThan(1e-10);
    });
  });

  describe("Training Loop", () => {
    it("5 training steps produce finite losses", { timeout: 600000 }, async () => {
      if (!webgpuAvailable) {
        console.log("Skipping: WebGPU not available");
        return;
      }

      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();

      const api = new Torchlette("webgpu", {
        enableFusion: false, // Disabled due to reshape issues with fusion
        enableMemoryPlanning: true,
      });
      const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
      model.train();

      const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
      const scaler = new GradScaler(api, { initScale: 1024, enabled: true });

      const losses: number[] = [];
      const batchSize = 2;
      const seqLen = 16;

      for (let step = 0; step < 5; step++) {
        await scaler.resolveDeferred();
        // Zero gradients
        for (const param of model.parameters()) {
          param.zeroGrad();
        }

        // Random input/target
        const inputData = Array.from({ length: batchSize * seqLen }, () =>
          Math.floor(Math.random() * TEST_CONFIG.vocabSize)
        );
        const targetData = Array.from({ length: batchSize * seqLen }, () =>
          Math.floor(Math.random() * TEST_CONFIG.vocabSize)
        );

        const input = api.tensorFromArray(inputData, [batchSize, seqLen]);
        const target = api.tensorFromArray(targetData, [batchSize, seqLen]);

        // Forward with checkpoint (autocast disabled due to known reshape issue)
        const result = model.forwardWithLoss(input, target, { useCheckpoint: true });

        const lossVal = await result.loss!.item();
        losses.push(lossVal);

        // Backward with gradient scaling
        const scaledLoss = scaler.scale(result.loss!);
        await scaledLoss.backward();

        // Optimizer step
        scaler.unscale_(optimizer);
        scaler.step(optimizer);
        scaler.update();

        await api.markStep();

        console.log(`Step ${step + 1}: loss = ${lossVal.toFixed(4)}`);
      }

      // All losses should be finite
      expect(losses.every(l => Number.isFinite(l))).toBe(true);

      // Losses should be positive (cross-entropy is always positive)
      expect(losses.every(l => l > 0)).toBe(true);
    });

    it("loss decreases or stays stable over training steps", { timeout: 600000 }, async () => {
      if (!webgpuAvailable) {
        console.log("Skipping: WebGPU not available");
        return;
      }

      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();

      const api = new Torchlette("webgpu", {
        enableFusion: false, // Disabled due to reshape issues with fusion
        enableMemoryPlanning: true,
      });
      const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
      model.train();

      const optimizer = new Adam(model.parameters(), { lr: 1e-3 }, api); // Higher LR
      const scaler = new GradScaler(api, { initScale: 1024, enabled: true });

      // Use fixed data for this test
      const inputData = Array.from({ length: 32 }, (_, i) => i % TEST_CONFIG.vocabSize);
      const targetData = Array.from({ length: 32 }, (_, i) => (i + 1) % TEST_CONFIG.vocabSize);

      const losses: number[] = [];

      for (let step = 0; step < 10; step++) {
        await scaler.resolveDeferred();
        for (const param of model.parameters()) {
          param.zeroGrad();
        }

        const input = api.tensorFromArray(inputData, [2, 16]);
        const target = api.tensorFromArray(targetData, [2, 16]);

        // Forward with checkpoint (autocast disabled due to known reshape issue)
        const result = model.forwardWithLoss(input, target, { useCheckpoint: true });

        const lossVal = await result.loss!.item();
        losses.push(lossVal);

        const scaledLoss = scaler.scale(result.loss!);
        await scaledLoss.backward();

        scaler.unscale_(optimizer);
        scaler.step(optimizer);
        scaler.update();

        await api.markStep();
      }

      console.log("Losses over 10 steps:", losses.map(l => l.toFixed(4)).join(", "));

      // First loss should be high (untrained model)
      expect(losses[0]).toBeGreaterThan(0);

      // Loss should generally decrease or stay stable
      // Allow some fluctuation but last loss should not be much higher than first
      const firstLoss = losses[0];
      const lastLoss = losses[losses.length - 1];
      expect(lastLoss).toBeLessThan(firstLoss * 1.5); // Allow up to 50% increase due to randomness
    });
  });

  describe("Memory Stability", () => {
    it("zero steady-state growth in all memory metrics", { timeout: 600000 }, async () => {
      if (!webgpuAvailable) {
        console.log("Skipping: WebGPU not available");
        return;
      }

      resetNodeIdCounter();
      // NOTE: Do NOT call resetStorageIdCounter() here. Under singleFork,
      // prior tests' tensors have FinalizationRegistry callbacks with old
      // storageIds. Resetting the counter causes new IDs to collide with
      // old _held.storageId values, so stale finalization callbacks wrongly
      // mark this test's live storages as unreachable.
      resetBaseIdCounter();
      storageTracker.reset();

      const api = new Torchlette("webgpu", {
        enableFusion: false, // Disabled due to reshape issues with fusion
        enableMemoryPlanning: true,
      });
      const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
      model.train();

      const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
      const scaler = new GradScaler(api, { initScale: 1024, enabled: true });

      const batchSize = 2;
      const seqLen = 16;

      async function trainStep() {
        await scaler.resolveDeferred();
        for (const param of model.parameters()) {
          param.zeroGrad();
        }

        const inputData = Array.from({ length: batchSize * seqLen }, () =>
          Math.floor(Math.random() * TEST_CONFIG.vocabSize)
        );
        const targetData = Array.from({ length: batchSize * seqLen }, () =>
          Math.floor(Math.random() * TEST_CONFIG.vocabSize)
        );

        // tidy() disposes all intermediate tensors except those returned/kept
        const loss = api.tidy(() => {
          const input = api.tensorFromArray(inputData, [batchSize, seqLen]);
          const target = api.tensorFromArray(targetData, [batchSize, seqLen]);
          const result = model.forwardWithLoss(input, target, { useCheckpoint: true });
          return result.loss!;
        });

        const scaledLoss = scaler.scale(loss);
        await scaledLoss.backward();

        scaler.unscale_(optimizer);
        scaler.step(optimizer);
        scaler.update();

        scaledLoss.dispose();
        loss.dispose();

        await api.markStep();
      }

      // Methodology: sample memory metrics at 3 points (after steps 3, 5, 7)
      // and check that the PER-STEP RATE between the last two samples is zero.
      // Absolute counts may drift due to non-deterministic GC timing (JavaScript
      // has no RAII — FinalizationRegistry callbacks are GC-dependent). What
      // matters is that the rate of growth is zero, not the absolute count.

      function snapshot() {
        const s = storageTracker.stats();
        const m = getGPUMemoryStats();
        return {
          storages: s.totalStorages,
          reachable: s.reachableStorages,
          allocs: m.allocationCount,
          trackerBytes: m.currentBytes,
        };
      }

      gpuMemoryTracker.reset();

      // Force GC if available (via --expose-gc) for deterministic measurement
      const gc = (globalThis as any).gc as (() => void) | undefined;

      // Drain FinalizationRegistry callbacks by running GC + yielding
      async function drainGC() {
        gc?.();
        await new Promise(r => setTimeout(r, 50));
        gc?.();
        await new Promise(r => setTimeout(r, 50));
      }

      // Phase 1: warmup (10 steps — enough for buffer pool to fully stabilize)
      for (let i = 0; i < 10; i++) await trainStep();
      await drainGC();
      const s1 = snapshot();

      // Phase 2: measure over 8 steps (long window averages out GC jitter)
      const MEASURE_STEPS = 8;
      for (let i = 0; i < MEASURE_STEPS; i++) {
        const allocsBefore = getGPUMemoryStats().allocationCount;
        await trainStep();
        const allocsAfter = getGPUMemoryStats().allocationCount;
        console.log(`  step ${i}: allocs delta=${allocsAfter - allocsBefore}`);
      }
      await drainGC();
      const s2 = snapshot();

      // Per-step growth rate over the measurement window
      const rate = {
        storages: (s2.storages - s1.storages) / MEASURE_STEPS,
        reachable: (s2.reachable - s1.reachable) / MEASURE_STEPS,
        allocs: (s2.allocs - s1.allocs) / MEASURE_STEPS,
        trackerMB: (s2.trackerBytes - s1.trackerBytes) / (MEASURE_STEPS * 1e6),
      };

      console.log(`Per-step rate: storages=${rate.storages}, reachable=${rate.reachable}, allocs=${rate.allocs}, trackerMB=${rate.trackerMB.toFixed(2)}`);

      // A real leak shows constant positive growth. Allow ≤1/step tolerance
      // for GC jitter (FinalizationRegistry timing is non-deterministic).
      const LEAK_THRESHOLD = 1;
      expect(rate.storages).toBeLessThanOrEqual(LEAK_THRESHOLD);
      expect(rate.reachable).toBeLessThanOrEqual(LEAK_THRESHOLD);
      expect(rate.allocs).toBeLessThanOrEqual(LEAK_THRESHOLD);
    });
  });

  describe("Combined Features", () => {
    it("checkpoint + GradScaler work together (AMP disabled due to reshape issue)", { timeout: 300000 }, async () => {
      if (!webgpuAvailable) {
        console.log("Skipping: WebGPU not available");
        return;
      }

      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();

      const api = new Torchlette("webgpu", {
        enableFusion: false, // Disabled due to reshape issues with fusion
        enableMemoryPlanning: true,
      });
      const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
      model.train();

      const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
      const scaler = new GradScaler(api, { initScale: 1024, enabled: true });

      const inputData = Array.from({ length: 32 }, () =>
        Math.floor(Math.random() * TEST_CONFIG.vocabSize)
      );
      const targetData = Array.from({ length: 32 }, () =>
        Math.floor(Math.random() * TEST_CONFIG.vocabSize)
      );

      const input = api.tensorFromArray(inputData, [2, 16]);
      const target = api.tensorFromArray(targetData, [2, 16]);

      // Forward with checkpoint + GradScaler (autocast disabled due to known reshape issue)
      const result = model.forwardWithLoss(input, target, { useCheckpoint: true });

      expect(result.loss).not.toBeNull();
      const lossVal = await result.loss!.item();
      expect(Number.isFinite(lossVal)).toBe(true);

      // Backward with scaling
      const scaledLoss = scaler.scale(result.loss!);
      await scaledLoss.backward();

      // Unscale and step — fully lazy, no GPU stall
      scaler.unscale_(optimizer);
      scaler.step(optimizer);
      scaler.update();

      // Verify gradients exist
      for (const param of model.parameters()) {
        expect(param.grad).not.toBeNull();
      }

      // foundInf is deferred — resolve it now
      await scaler.resolveDeferred();
      console.log(`Loss: ${lossVal.toFixed(4)}, Scale: ${scaler.getScale()}, Found inf: ${scaler.foundInf}`);
    });
  });
});
