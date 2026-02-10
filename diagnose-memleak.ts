/**
 * Memory leak diagnostic script — v2.
 * Tracks NET allocation growth per step to find true leaks.
 */
import * as path from "node:path";
import { Torchlette, type Tensor } from "./src/frontend";
import {
  initWebGPU,
  getGPUMemoryStats,
  getBufferPoolStats,
} from "./src/backend/webgpu";
import {
  enableAllAllocDebug,
  setAllocStep,
  getAndResetFlowCounters,
  getGPUAllocationHistogram,
} from "./src/backend/webgpu/memory-tracker";
import { gpuMemoryTracker } from "./src/backend/webgpu/memory-tracker";
import { storageTracker } from "./src/engine/lazy";
import { loadPretrainedGPT2 } from "./examples/gpt2/loader";
import { Adam, GradScaler } from "./src/optim";

const TRAIN_TOKENS = [
  2484, 439, 314, 8996, 25587, 284, 257, 3931, 338, 1110, 30, 198, 1858,
  5765, 448, 517, 8574, 290, 517, 44539, 13, 198, 49, 619, 9985, 466,
  15614, 262, 288, 6138, 22303, 286,
];

async function main() {
  console.log("=== Memory Leak Diagnostic v2 ===\n");

  await initWebGPU();

  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.train();
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    return api.autocast(() => {
      const result = model.forwardWithLoss(input, target, { useCheckpoint: true });
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });

  const inputData = TRAIN_TOKENS.slice(0, -1);
  const targetData = TRAIN_TOKENS.slice(1);

  // Warm up
  for (let step = 0; step < 3; step++) {
    const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });
    const loss = compiledForward(input, target);
    await loss.item();
    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();
    scaler.unscale_(optimizer);
    scaler.step(optimizer);
    scaler.update();
    optimizer.zeroGrad();
    scaledLoss.dispose();
    loss.dispose();
    input.dispose();
    target.dispose();
    await api.markStep();
  }

  console.log("Warmup done (3 steps).\n");

  // Enable tracking, snapshot before each step
  enableAllAllocDebug();

  for (let step = 3; step < 8; step++) {
    setAllocStep(step);

    // Snapshot state BEFORE step
    const memBefore = getGPUMemoryStats();
    const poolBefore = getBufferPoolStats();
    const storageBefore = storageTracker.stats();

    // Take a snapshot of all currently tracked buffers
    const trackedBefore = gpuMemoryTracker.getTrackedBuffers();

    getAndResetFlowCounters();

    const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });
    const loss = compiledForward(input, target);
    const lossVal = await loss.item();
    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();
    scaler.unscale_(optimizer);
    scaler.step(optimizer);
    scaler.update();
    optimizer.zeroGrad();
    scaledLoss.dispose();
    loss.dispose();
    input.dispose();
    target.dispose();
    await api.markStep();

    // Snapshot AFTER step
    const memAfter = getGPUMemoryStats();
    const poolAfter = getBufferPoolStats();
    const storageAfter = storageTracker.stats();
    const flow = getAndResetFlowCounters();

    // Find NEW buffers that appeared this step
    const trackedAfter = gpuMemoryTracker.getTrackedBuffers();
    let newBuffers = 0;
    let newBytes = 0;
    const newBufferSizes: number[] = [];
    for (const buf of trackedAfter) {
      if (!trackedBefore.has(buf)) {
        newBuffers++;
        // Get size via the alloc stacks
        const info = (gpuMemoryTracker as any)._allocStacks.get(buf);
        if (info) {
          newBytes += info.size;
          newBufferSizes.push(info.size);
        }
      }
    }

    // Find buffers that DISAPPEARED (freed)
    let freedBuffers = 0;
    for (const buf of trackedBefore) {
      if (!trackedAfter.has(buf)) {
        freedBuffers++;
      }
    }

    console.log(`Step ${step}: loss=${lossVal.toFixed(4)}`);
    console.log(`  Memory: ${(memBefore.currentBytes / 1e6).toFixed(1)}MB → ${(memAfter.currentBytes / 1e6).toFixed(1)}MB (delta: +${((memAfter.currentBytes - memBefore.currentBytes) / 1e6).toFixed(1)}MB)`);
    console.log(`  Allocs: ${memBefore.allocationCount} → ${memAfter.allocationCount} (net: +${memAfter.allocationCount - memBefore.allocationCount})`);
    console.log(`  Flow: ${flow.allocs} allocs, ${flow.deallocs} deallocs, ${flow.deallocMisses} misses`);
    console.log(`  Storages: ${storageBefore.totalStorages}/${storageBefore.reachableStorages} → ${storageAfter.totalStorages}/${storageAfter.reachableStorages}`);
    console.log(`  Pool: ${poolBefore.pooledBuffers}→${poolAfter.pooledBuffers} buffers, ${poolBefore.pendingBuffers}→${poolAfter.pendingBuffers} pending`);
    console.log(`  Buffer diff: +${newBuffers} new, -${freedBuffers} freed = net ${newBuffers - freedBuffers}`);

    if (newBufferSizes.length > 0) {
      // Group by size
      const sizeGroups = new Map<number, number>();
      for (const s of newBufferSizes) {
        sizeGroups.set(s, (sizeGroups.get(s) || 0) + 1);
      }
      const sorted = [...sizeGroups.entries()].sort((a, b) => b[0] * b[1] - a[0] * a[1]);
      console.log(`  New buffer sizes:`);
      for (const [size, count] of sorted.slice(0, 8)) {
        console.log(`    ${(size / 1024).toFixed(1)}KB × ${count} = ${(size * count / 1e6).toFixed(2)}MB`);
      }
    }

    // For step 4: save the new buffer set to check which survive to step 5
    if (step === 4) {
      const newBufferSet = new Set<any>();
      for (const buf of trackedAfter) {
        if (!trackedBefore.has(buf)) newBufferSet.add(buf);
      }
      (globalThis as any).__step4NewBuffers = newBufferSet;
    }

    // For step 5: check which of step 4's new buffers survived
    if (step === 5 && (globalThis as any).__step4NewBuffers) {
      const step4New = (globalThis as any).__step4NewBuffers as Set<any>;
      const survivedFromStep4: any[] = [];
      for (const buf of step4New) {
        if (trackedBefore.has(buf)) {
          survivedFromStep4.push(buf);
        }
      }
      console.log(`\n  === Survived from step 4 → step 5: ${survivedFromStep4.length} buffers ===`);
      const stackGroups = new Map<string, { count: number; totalBytes: number; exampleStack: string }>();
      for (const buf of survivedFromStep4) {
        const info = (gpuMemoryTracker as any)._allocStacks.get(buf);
        if (!info) continue;
        const key = info.stack.split('\n').slice(0, 3).join('\n');
        const existing = stackGroups.get(key) || { count: 0, totalBytes: 0, exampleStack: info.stack };
        existing.count++;
        existing.totalBytes += info.size;
        stackGroups.set(key, existing);
      }
      const sortedStacks = [...stackGroups.entries()].sort((a, b) => b[1].totalBytes - a[1].totalBytes);
      for (const [, { count, totalBytes, exampleStack }] of sortedStacks) {
        console.log(`\n    ${count}× (${(totalBytes / 1e6).toFixed(2)}MB):`);
        const lines = exampleStack.split('\n');
        for (const line of lines.slice(0, 6)) {
          console.log(`      ${line.trim()}`);
        }
      }

      // Also check: which of step 4's new buffers were freed within step 4?
      const freedInStep4 = step4New.size - survivedFromStep4.length;
      // But the diagnostic says 252 freed total (including old ones), and 277 new.
      // The survived set should be exactly the "net 25" that leaked.
    }

    console.log("");
  }

  console.log("Done.");
}

main().catch(e => { console.error(e); process.exit(1); });
