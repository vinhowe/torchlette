/**
 * GPU buffer-level memory diagnostic.
 * Uses the memory tracker's debug features to find alloc/dealloc imbalances.
 */

import * as path from "node:path";
import { Torchlette, type Tensor } from "../src/frontend";
import {
  initWebGPU,
  getGPUMemoryStats,
  getBufferPoolStats,
  getBufferPoolDetailedStats,
  setGPUMemoryLimit,
  enableAllAllocDebug,
  getAndResetFlowCounters,
  getLeakedAllocCount,
  getLeakedAllocCountForStep,
  snapshotLeakedAllocsForStep,
  setAllocStep,
  getTrackedBuffers,
  getGPUAllocationHistogram,
} from "../src/backend/webgpu";
import { storageTracker } from "../src/engine/lazy";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { Adam, GradScaler } from "../src/optim";

async function main() {
  console.log("=== GPU Buffer Memory Diagnostic ===\n");

  setGPUMemoryLimit(30 * 1024 * 1024 * 1024);

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.train();

  const optimizer = new Adam(model.parameters(), {
    lr: 5e-4,
    betas: [0.9, 0.999],
    eps: 1e-8,
    weightDecay: 0.01,
    adamW: true,
  }, api);

  const scaler = new GradScaler(api);

  const inputData = [50256, 1820, 4238, 338, 257, 5765, 1110];
  const targetData = [1820, 4238, 338, 257, 5765, 1110, 764];

  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    return api.autocast(() => {
      const result = model.forwardWithLoss(input, target, { useCheckpoint: true });
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });

  console.log("Model loaded, starting training loop...\n");

  // Enable debug tracking
  enableAllAllocDebug();

  // Get initial state
  const memBefore = getGPUMemoryStats();
  const poolBefore = getBufferPoolStats();
  console.log(`Initial: ${(memBefore.currentBytes / 1e9).toFixed(3)}GB tracked, ${memBefore.allocationCount} allocs, pool: ${poolBefore.pooledBuffers} buffers (${(poolBefore.pooledBytes / 1e6).toFixed(1)}MB)`);

  // Reset flow counters
  getAndResetFlowCounters();

  for (let step = 0; step < 5; step++) {
    setAllocStep(step);

    const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });

    const loss = compiledForward(input, target);
    const lossValue = await loss.item();

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

    // Collect stats
    const mem = getGPUMemoryStats();
    const pool = getBufferPoolStats() as any;
    const flow = getAndResetFlowCounters();
    const storageStats = storageTracker.stats();

    console.log(`\nStep ${step}: loss=${lossValue.toFixed(4)}`);
    console.log(`  Tracker: ${(mem.currentBytes / 1e9).toFixed(3)}GB, ${mem.allocationCount} live allocs, ${mem.bufferSizesCount} bufferSizes entries`);
    console.log(`  Flow: +${flow.allocs} allocs, -${flow.deallocs} deallocs, ${flow.deallocMisses} misses, ${flow.doubleTracked} doubles`);
    console.log(`  Net flow: ${flow.allocs - flow.deallocs} (should be ~0 at steady state)`);
    console.log(`  Pool: ${pool.pooledBuffers} pooled (${(pool.pooledBytes / 1e6).toFixed(1)}MB), ${pool.pendingRelease ?? pool.pendingBuffers} pending-release, ${pool.pendingDestroy ?? '?'} pending-destroy`);
    console.log(`  Pool reuse: ${pool.reuseCount} reuses / ${pool.allocCount} new allocs (${(pool.reuseRate * 100).toFixed(1)}%)`);
    console.log(`  Storages: total=${storageStats.totalStorages}, reachable=${storageStats.reachableStorages}`);

    // Per-step leaked allocs
    const leakedThisStep = getLeakedAllocCountForStep(step);
    console.log(`  Leaked allocs this step: ${leakedThisStep}`);
    if (step >= 1 && leakedThisStep > 0) {
      const leaked = snapshotLeakedAllocsForStep(step);
      const sorted = [...leaked.entries()].sort((a, b) => b[1].totalBytes - a[1].totalBytes);
      console.log(`  Top leaked allocation sites (step ${step}):`);
      for (const [site, info] of sorted.slice(0, 5)) {
        console.log(`    ${info.count}x (${(info.totalBytes / 1e6).toFixed(2)}MB total):`);
        const lines = site.split('\n');
        for (const line of lines) {
          console.log(`      ${line.trim()}`);
        }
      }
    }

    // Allocation size histogram
    if (step >= 1) {
      const histogram = getGPUAllocationHistogram();
      console.log(`  Allocation histogram:`);
      for (const [label, info] of histogram) {
        console.log(`    ${label}: ${info.count} allocs (${(info.totalBytes / 1e6).toFixed(1)}MB)`);
      }
    }
  }

  // Final summary
  const memFinal = getGPUMemoryStats();
  const totalLeaked = getLeakedAllocCount();
  console.log(`\n=== Final Summary ===`);
  console.log(`Tracked memory: ${(memFinal.currentBytes / 1e9).toFixed(3)}GB`);
  console.log(`Total unmatched allocs across all steps: ${totalLeaked}`);

  console.log("\nDone!");
}

main().catch(console.error);
