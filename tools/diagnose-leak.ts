/**
 * Memory leak diagnostic script for the training loop.
 * Runs training steps and dumps detailed info about leaking storages,
 * cross-references with GPU buffer tracker to find orphaned buffers.
 *
 * Usage:
 *   npx tsx tools/diagnose-leak.ts [--steps N]
 */

import * as path from "node:path";
import { Torchlette, type Tensor } from "../src/frontend";
import {
  initWebGPU,
  destroyWebGPU,
  getGPUMemoryStats,
  getBufferPoolStats,
  setGPUMemoryLimit,
} from "../src/backend/webgpu";
import {
  getAndResetFlowCounters,
  enableAllAllocDebug,
  setAllocStep,
  snapshotLeakedAllocsForStep,
  getTrackedBuffers,
  getLeakedSizeHistogramForStep,
} from "../src/backend/webgpu/memory-tracker";
import { storageTracker } from "../src/engine/lazy";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { Adam, GradScaler } from "../src/optim";

// Parse --steps N from CLI args
function parseArgs(): { steps: number } {
  const args = process.argv.slice(2);
  let steps = 5;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--steps" && i + 1 < args.length) {
      steps = parseInt(args[i + 1], 10);
      if (isNaN(steps) || steps < 1) steps = 5;
    }
  }
  return { steps };
}

async function main() {
  const { steps } = parseArgs();
  console.log(`=== Memory Leak Diagnostic (${steps} steps) ===\n`);

  // Set higher memory limit to allow more steps
  setGPUMemoryLimit(30 * 1024 * 1024 * 1024); // 30GB

  // Initialize
  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  // Load model
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.train();

  // Create optimizer and scaler
  const optimizer = new Adam(model.parameters(), {
    lr: 5e-4,
    betas: [0.9, 0.999],
    eps: 1e-8,
    weightDecay: 0.01,
    adamW: true,
  }, api);

  const scaler = new GradScaler(api);

  // Prepare a single small training sequence
  const inputData = [50256, 1820, 4238, 338, 257, 5765, 1110]; // "The sun is a beautiful day"
  const targetData = [1820, 4238, 338, 257, 5765, 1110, 764];  // shifted by 1

  // Compiled forward
  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    return api.autocast(() => {
      const result = model.forwardWithLoss(input, target, { useCheckpoint: true });
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });

  console.log("Model loaded, starting training loop...\n");

  // Enable allocation stack tracking for leak detection
  enableAllAllocDebug();

  // Reset debug counters and flow counters
  storageTracker.debugCounters();
  getAndResetFlowCounters();

  // Snapshot reachable storage IDs before training
  let prevReachableIds = storageTracker.getReachableIds();

  // Per-step tracking for trend analysis
  const stepData: Array<{
    storages: number;
    reachable: number;
    trackerBytes: number;
    allocCount: number;
    pendingDestroy: number;
    pendingRelease: number;
  }> = [];

  for (let step = 0; step < steps; step++) {
    setAllocStep(step);
    const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });

    // Forward
    const loss = compiledForward(input, target);
    const lossValue = await loss.item();

    // Backward
    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();

    // Optimizer step
    scaler.unscale_(optimizer);
    scaler.step(optimizer);
    scaler.update();
    optimizer.zeroGrad();

    // Dispose
    scaledLoss.dispose();
    loss.dispose();
    input.dispose();
    target.dispose();

    // markStep
    await api.markStep();

    // Collect stats
    const counters = storageTracker.debugCounters();
    const stats = storageTracker.stats();
    const memStats = getGPUMemoryStats();
    const poolStats = getBufferPoolStats();
    const flowCounters = getAndResetFlowCounters();

    stepData.push({
      storages: stats.totalStorages,
      reachable: stats.reachableStorages,
      trackerBytes: memStats.currentBytes,
      allocCount: memStats.allocationCount,
      pendingDestroy: (poolStats as any).pendingDestroy ?? 0,
      pendingRelease: (poolStats as any).pendingRelease ?? 0,
    });

    console.log(`Step ${step}: loss=${lossValue.toFixed(4)}`);
    console.log(`  Storages: total=${stats.totalStorages}, reachable=${stats.reachableStorages}, unreachable=${stats.unreachableStorages}`);
    console.log(`  Counters: registered=${counters.registered}, reachable=${counters.reachable}, unreachable=${counters.unreachable}, destroyed=${counters.destroyed}`);
    console.log(`  Delta: registered-destroyed=${counters.registered - counters.destroyed}, reachable-unreachable=${counters.reachable - counters.unreachable}`);
    console.log(`  Memory: ${(memStats.currentBytes / 1e9).toFixed(3)}GB, tracked buffers=${memStats.bufferSizesCount}`);
    console.log(`  Pool: pooled=${poolStats.pooledBuffers} (${(poolStats.pooledBytes / 1e6).toFixed(1)}MB), pending-release=${(poolStats as any).pendingRelease}, pending-destroy=${(poolStats as any).pendingDestroy}`);
    console.log(`  Flow: +${flowCounters.allocs} allocs, -${flowCounters.deallocs} deallocs, net=${flowCounters.allocs - flowCounters.deallocs}`);

    // Find NEW reachable storage IDs using getNewReachableSince
    const newReachable = storageTracker.getNewReachableSince(prevReachableIds);
    const currentReachableIds = storageTracker.getReachableIds();
    const lostReachableIds: number[] = [];
    for (const id of prevReachableIds) {
      if (!currentReachableIds.has(id)) {
        lostReachableIds.push(id);
      }
    }
    console.log(`  New reachable: ${newReachable.length}, Lost reachable: ${lostReachableIds.length}`);

    // Count live vs orphaned
    let orphanedCount = 0;
    let liveRefCount = 0;
    const byType = new Map<string, number>();
    for (const entry of newReachable) {
      if (entry.hasLiveTensorRef) {
        liveRefCount++;
      } else {
        orphanedCount++;
      }
      const type = entry.debugInfo?.type ?? "unknown";
      byType.set(type, (byType.get(type) ?? 0) + 1);
    }
    console.log(`  Live refs: ${liveRefCount}, Orphaned: ${orphanedCount}`);
    if (byType.size > 0) {
      const typeStr = [...byType.entries()].map(([k, v]) => `${k}=${v}`).join(", ");
      console.log(`  By type: ${typeStr}`);
    }

    // For step >= 1, print detailed shape distribution
    if (step >= 1) {
      const shapeGroups = new Map<string, { count: number; sizeBytes: number; disposedCount: number }>();
      let totalDisposed = 0;
      let totalNotDisposed = 0;
      for (const entry of newReachable) {
        const info = entry.debugInfo;
        const storage = storageTracker.getStorage(entry.id);
        let sizeBytes = 0;
        if (storage) {
          const bt = storage.backendTensor as any;
          sizeBytes = bt.size ?? bt.buffer?.size ?? 0;
        }
        const shapeKey = info?.shape ? `[${info.shape.join(",")}]` : "unknown";
        const typeKey = info?.type ?? "unknown";
        const key = `${typeKey}:${shapeKey}:${info?.dtype ?? "?"}`;
        const existing = shapeGroups.get(key) || { count: 0, sizeBytes: 0, disposedCount: 0 };
        existing.count++;
        existing.sizeBytes += sizeBytes;
        if (info?.disposed) {
          existing.disposedCount++;
          totalDisposed++;
        } else {
          totalNotDisposed++;
        }
        shapeGroups.set(key, existing);
      }
      const sorted = [...shapeGroups.entries()].sort((a, b) => b[1].count - a[1].count);
      console.log(`  Disposed: ${totalDisposed}, Not disposed: ${totalNotDisposed}`);
      console.log(`  Shape distribution of ${newReachable.length} new reachable storages:`);
      for (const [key, val] of sorted.slice(0, 25)) {
        const sizeLabel = val.sizeBytes > 1024 * 1024 ? `${(val.sizeBytes / 1e6).toFixed(1)}MB` :
                          val.sizeBytes > 1024 ? `${(val.sizeBytes / 1024).toFixed(1)}KB` :
                          `${val.sizeBytes}B`;
        const dispInfo = val.disposedCount > 0 ? ` (${val.disposedCount} disposed)` : "";
        console.log(`    ${key}: x${val.count}${dispInfo} (${sizeLabel} total)`);
      }
    }

    // For step >= 2, find the 3 "extra" new reachable that have no matching lost shape
    if (step >= 2) {
      // Shape distribution of lost reachable
      const lostShapes = new Map<string, number>();
      for (const id of lostReachableIds) {
        const info = storageTracker.getTensorRefDebugInfo(id);
        const shapeKey = info?.shape ? `[${info.shape.join(",")}]` : "unknown";
        const key = `${info?.type ?? "unknown"}:${shapeKey}:${info?.dtype ?? "?"}`;
        lostShapes.set(key, (lostShapes.get(key) ?? 0) + 1);
      }
      // Count shape distribution of new reachable
      const newShapes = new Map<string, number>();
      for (const entry of newReachable) {
        const info = entry.debugInfo;
        const shapeKey = info?.shape ? `[${info.shape.join(",")}]` : "unknown";
        const key = `${info?.type ?? "unknown"}:${shapeKey}:${info?.dtype ?? "?"}`;
        newShapes.set(key, (newShapes.get(key) ?? 0) + 1);
      }
      // Find unmatched: shapes in new but not enough in lost
      const extraShapes: string[] = [];
      for (const [key, newCount] of newShapes) {
        const lostCount = lostShapes.get(key) ?? 0;
        if (newCount > lostCount) {
          extraShapes.push(`${key}: new=${newCount}, lost=${lostCount} (delta=+${newCount - lostCount})`);
        }
      }
      if (extraShapes.length > 0) {
        console.log(`  EXTRA new reachable shapes (no matching lost):`);
        for (const s of extraShapes) {
          console.log(`    ${s}`);
        }
      }
      // Show the actual storage IDs and tensor status of the extra ones
      // Find the individual entries matching the extra shapes
      for (const entry of newReachable) {
        const info = entry.debugInfo;
        const shapeKey = info?.shape ? `[${info.shape.join(",")}]` : "unknown";
        const key = `${info?.type ?? "unknown"}:${shapeKey}:${info?.dtype ?? "?"}`;
        const lostCount = lostShapes.get(key) ?? 0;
        const newCount = newShapes.get(key) ?? 0;
        if (newCount > lostCount) {
          console.log(`    storageId=${entry.id} shape=${shapeKey} dtype=${info?.dtype} disposed=${info?.disposed} liveRef=${entry.hasLiveTensorRef}`);
        }
      }
    }

    // Show ALL reachable f16 tensor storages and logits to see accumulation
    if (step >= 1) {
      const suspectStorages: string[] = [];
      for (const id of currentReachableIds) {
        const info = storageTracker.getTensorRefDebugInfo(id);
        if (info?.type === 'tensor' && (info.dtype === 'f16' || (info.shape && info.shape.join(',') === '7,50257'))) {
          const shapeKey = info?.shape ? `[${info.shape.join(",")}]` : "unknown";
          suspectStorages.push(`storageId=${id} shape=${shapeKey} dtype=${info?.dtype} disposed=${info?.disposed}`);
        }
      }
      if (suspectStorages.length > 0) {
        console.log(`  ALL reachable f16/logits tensor storages: ${suspectStorages.length}`);
        for (const s of suspectStorages) {
          console.log(`    ${s}`);
        }
      }
    }

    // Cross-reference: find GPU buffers tracked by memory tracker but not owned by any storage
    const liveOwnedBuffers = storageTracker.getLiveOwnedBuffers();
    console.log(`  Storage-owned buffers: ${liveOwnedBuffers.size}, Tracker tracked: ${memStats.bufferSizesCount}`);
    if (memStats.bufferSizesCount > liveOwnedBuffers.size) {
      console.log(`  => ${memStats.bufferSizesCount - liveOwnedBuffers.size} tracked buffers have no owning storage (orphaned GPU buffers)`);
    }

    prevReachableIds = currentReachableIds;
    console.log("");
  }

  // === Orphaned Buffer Analysis ===
  console.log("=== Orphaned Buffer Analysis ===\n");
  {
    const trackedBuffers = getTrackedBuffers();
    const ownedBuffers = storageTracker.getLiveOwnedBuffers();
    const orphanedBuffers = [...trackedBuffers].filter(b => !ownedBuffers.has(b));
    console.log(`Tracked buffers: ${trackedBuffers.size}`);
    console.log(`Storage-owned buffers: ${ownedBuffers.size}`);
    console.log(`Orphaned (tracked but no owning storage): ${orphanedBuffers.length}`);

    // Dump alloc stacks for orphaned buffers from the _allocStacks debug map
    // We need to cross-reference orphaned buffer objects with their alloc stacks
    const { gpuMemoryTracker } = await import("../src/backend/webgpu/memory-tracker");
    const allocStacks = (gpuMemoryTracker as any)._allocStacks as Map<unknown, { size: number; stack: string; step: number }>;
    if (allocStacks && allocStacks.size > 0) {
      // Group orphaned buffer alloc stacks by call site
      const orphanedByCallSite = new Map<string, { count: number; totalBytes: number; steps: Set<number>; exampleStack: string }>();
      for (const buf of orphanedBuffers) {
        const info = allocStacks.get(buf);
        if (!info) continue;
        const key = info.stack.split('\n').slice(0, 3).join('\n');
        const existing = orphanedByCallSite.get(key) || { count: 0, totalBytes: 0, steps: new Set<number>(), exampleStack: info.stack };
        existing.count++;
        existing.totalBytes += info.size;
        existing.steps.add(info.step);
        orphanedByCallSite.set(key, existing);
      }
      if (orphanedByCallSite.size > 0) {
        console.log(`\n  Orphaned buffer allocation sites:`);
        // Use deeper stack grouping (up to 8 frames) to differentiate callers
        const orphanedByDeepSite = new Map<string, { count: number; totalBytes: number; steps: Set<number>; exampleStack: string; sizes: number[] }>();
        for (const buf of orphanedBuffers) {
          const info = allocStacks.get(buf);
          if (!info) continue;
          const key = info.stack.split('\n').slice(0, 6).join('\n');
          const existing = orphanedByDeepSite.get(key) || { count: 0, totalBytes: 0, steps: new Set<number>(), exampleStack: info.stack, sizes: [] };
          existing.count++;
          existing.totalBytes += info.size;
          existing.steps.add(info.step);
          existing.sizes.push(info.size);
          orphanedByDeepSite.set(key, existing);
        }
        const sorted = [...orphanedByDeepSite.entries()].sort((a, b) => b[1].count - a[1].count);
        for (const [, info] of sorted) {
          const stepList = [...info.steps].sort((a, b) => a - b);
          // Show unique sizes
          const uniqueSizes = [...new Set(info.sizes)].sort((a, b) => a - b);
          const sizeStr = uniqueSizes.map(s => s >= 1024*1024 ? `${(s/1e6).toFixed(1)}MB` : s >= 1024 ? `${(s/1024).toFixed(1)}KB` : `${s}B`).join(", ");
          console.log(`  ${info.count}x (${(info.totalBytes / 1e6).toFixed(1)}MB, steps: ${stepList.join(",")}, sizes: [${sizeStr}]):`);
          console.log(`    ${info.exampleStack.split('\n').slice(0, 12).join('\n    ')}`);
        }
      }
    }
  }

  // === Leaked Alloc Stacks (per steady-state step) ===
  const analyzeStep = Math.min(3, steps - 1);
  console.log(`\n=== Leaked Alloc Stacks (step ${analyzeStep}) ===`);
  {
    const leaked = snapshotLeakedAllocsForStep(analyzeStep);
    if (leaked.size === 0) {
      console.log("  No leaked allocations for this step.");
    } else {
      const sorted = [...leaked.entries()].sort((a, b) => b[1].count - a[1].count);
      for (const [, info] of sorted) {
        console.log(`  ${info.count}x (${(info.totalBytes / 1e6).toFixed(1)}MB):`);
        console.log(`    ${info.exampleStack.split('\n').slice(0, 5).join('\n    ')}`);
      }
    }

    // Size histogram
    const histogram = getLeakedSizeHistogramForStep(analyzeStep);
    if (histogram.size > 0) {
      console.log(`\n  Size histogram (step ${analyzeStep}):`);
      const sortedHist = [...histogram.entries()].sort((a, b) => b[1] - a[1]);
      for (const [size, count] of sortedHist) {
        const label = size >= 1024 * 1024 ? `${(size / 1e6).toFixed(1)}MB` :
                      size >= 1024 ? `${(size / 1024).toFixed(1)}KB` :
                      `${size}B`;
        console.log(`    ${label}: x${count}`);
      }
    }
  }

  // === Trend Summary ===
  console.log("\n=== Trend Summary ===\n");
  console.log("| Step | Storages | Reachable | PendingDestroy | TrackerMB | Allocs |");
  console.log("|------|----------|-----------|----------------|-----------|--------|");
  for (let i = 0; i < stepData.length; i++) {
    const d = stepData[i];
    console.log(
      `| ${String(i).padStart(4)} | ${String(d.storages).padStart(8)} | ${String(d.reachable).padStart(9)} | ${String(d.pendingDestroy).padStart(14)} | ${(d.trackerBytes / 1e6).toFixed(1).padStart(9)} | ${String(d.allocCount).padStart(6)} |`,
    );
  }

  // Steady-state deltas (skip step 0 warmup)
  if (stepData.length >= 3) {
    const steadyStart = Math.max(1, stepData.length - 3);
    const steadyEnd = stepData.length - 1;
    const n = steadyEnd - steadyStart;
    if (n > 0) {
      const storagesDelta = (stepData[steadyEnd].storages - stepData[steadyStart].storages) / n;
      const reachableDelta = (stepData[steadyEnd].reachable - stepData[steadyStart].reachable) / n;
      const trackerDelta = (stepData[steadyEnd].trackerBytes - stepData[steadyStart].trackerBytes) / n / 1e6;
      const allocDelta = (stepData[steadyEnd].allocCount - stepData[steadyStart].allocCount) / n;
      const pdStart = stepData[steadyStart].pendingDestroy;
      const pdEnd = stepData[steadyEnd].pendingDestroy;
      const pdGrowing = pdEnd > pdStart + 10;

      console.log(`\nSteady-state delta (steps ${steadyStart}-${steadyEnd}):`);
      console.log(`  Storages/step: ${storagesDelta >= 0 ? "+" : ""}${storagesDelta.toFixed(1)}`);
      console.log(`  Reachable/step: ${reachableDelta >= 0 ? "+" : ""}${reachableDelta.toFixed(1)}`);
      console.log(`  TrackerMB/step: ${trackerDelta >= 0 ? "+" : ""}${trackerDelta.toFixed(1)}`);
      console.log(`  Allocs/step: ${allocDelta >= 0 ? "+" : ""}${allocDelta.toFixed(1)}`);
      console.log(`  PendingDestroy: ${pdStart} -> ${pdEnd} (${pdGrowing ? "GROWING" : "stable"})`);

      // Verdict
      if (pdGrowing) {
        console.log("\nLEAK STATUS: FAIL - pendingDestroy is growing (physical GPU buffer leak)");
      } else if (allocDelta > 5 || reachableDelta > 5) {
        console.log(`\nLEAK STATUS: WARNING - +${allocDelta.toFixed(0)} allocs/step, +${reachableDelta.toFixed(0)} reachable/step`);
      } else {
        console.log("\nLEAK STATUS: OK");
      }
    }
  }

  console.log("\nDone!");
  destroyWebGPU();
}

main().catch(console.error);
