/**
 * Comprehensive GPU Training Profiler for DistilGPT-2
 *
 * Runs training steps with full profiling instrumentation:
 * - Per-phase GPU timing (forward/backward/optimizer/cleanup)
 * - Per-module GPU timing (embedding/attention/mlp/layernorm/etc.)
 * - Per-op GPU kernel timing
 * - Fusion statistics
 * - Memory statistics
 *
 * Usage:
 *   TORCHLETTE_PROFILE=1 npx tsx tools/profile-training.ts
 */

import * as path from "node:path";
import { Torchlette, type Tensor } from "../src/frontend";
import {
  initWebGPU,
  destroyWebGPU,
  getSubmitCount,
  resetSubmitCount,
  setProfilePhase,
  setProfileModule,
  readGpuTimestamps,
  printProfileSummary,
  resetProfileStats,
  isProfilingEnabled,
  setTimestampsEnabled,
  getProfileJSON,
  writeProfileJSON,
  getGPUMemoryStats,
  getBufferPoolStats,
  getBufferPoolDetailedStats,
  resetBufferPoolDetailedStats,
  isF16Supported,
  getBindGroupCacheStats,
  resetBindGroupCacheStats,
  getBindGroupCacheMissLog,
  getArenaResolveStats,
} from "../src/backend/webgpu";
import { getAndResetFlowCounters, getGPUAllocationHistogram } from "../src/backend/webgpu/memory-tracker";
import { storageTracker } from "../src/engine/lazy";
import { GPT2, type GPT2Config, DISTILGPT2_CONFIG } from "../examples/gpt2/model";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { Adam, GradScaler } from "../src/optim";
import { crossEntropy } from "../src/nn";

// ============================================================================
// Configuration
// ============================================================================

const NUM_STEPS = 5;
const PROFILE_STEP = 4; // Which step to analyze in detail (steady-state)
const BATCH_SIZE = parseInt(process.env.TORCHLETTE_BATCH_SIZE ?? "1", 10);

// Shakespeare sonnets tokens — tiled to seq_len+1 for seq_len=512 profiling
const BASE_TOKENS: number[] = [
  2484, 439, 314, 8996, 20218, 284, 257, 3931, 338, 1110, 30,
  198, 1986, 280, 1242, 517, 8855, 290, 517, 29815, 13,
  198, 49, 619, 9985, 466, 13508, 262, 38482, 31007, 286, 1737,
];
const SEQ_LEN = parseInt(process.env.TORCHLETTE_SEQ_LEN ?? "512", 10);
const FIXED_TOKENS: number[] = [];
for (let i = 0; i < SEQ_LEN + 1; i++) {
  FIXED_TOKENS.push(BASE_TOKENS[i % BASE_TOKENS.length]);
}

// ============================================================================
// Module-instrumented forward pass
// ============================================================================

/**
 * Set both the GPU profile module label and the autograd node label,
 * so backward ops inherit the forward module name via dispatch hooks.
 */
function profileModule(api: Torchlette, name: string) {
  setProfileModule(name);
  api.setNodeLabel(name);
}

/**
 * Wraps the model's forward pass with module-level profiling labels.
 * Each sub-module call is surrounded by profileModule() to tag
 * the lazy IR nodes with their originating module AND capture the
 * label on autograd nodes for backward attribution.
 */
function instrumentedForwardWithLoss(
  model: GPT2,
  api: Torchlette,
  idx: Tensor,
  targets: Tensor,
): Tensor {
  const config = model.config;
  const [_batch, seqLen] = idx.shape;

  // --- Embedding ---
  profileModule(api, "embed.wte");
  const tokEmb = model.wte.forward(idx);

  profileModule(api, "embed.wpe");
  const pos = api.arange(seqLen).reshape([1, seqLen]);
  const posEmb = model.wpe.forward(pos);

  profileModule(api, "embed.combine");
  let x = api.add(tokEmb, posEmb);
  // skip dropout in profiling (model.drop)

  // --- Transformer Blocks ---
  for (let i = 0; i < model.h.length; i++) {
    const block = model.h[i];

    // LayerNorm 1
    profileModule(api, `block${i}.ln1`);
    const ln1Out = block.ln1.forward(x);

    // Attention
    profileModule(api, `block${i}.attn.qkv`);
    const qkv = block.attn.cAttn.forward(ln1Out);

    profileModule(api, `block${i}.attn.split`);
    const [batch, sl, _ed] = ln1Out.shape;
    const numHeads = block.attn["numHeads"] as number;
    const headDim = block.attn["headDim"] as number;
    const embedDim = block.attn["embedDim"] as number;

    const qkvFor3 = qkv.reshape([batch, sl, 3, embedDim]);
    const qSlice = qkvFor3.narrow(2, 0, 1);
    const kSlice = qkvFor3.narrow(2, 1, 1);
    const vSlice = qkvFor3.narrow(2, 2, 1);

    const q = qSlice.reshape([batch, sl, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const k = kSlice.reshape([batch, sl, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const v = vSlice.reshape([batch, sl, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();

    profileModule(api, `block${i}.attn.sdpa`);
    const scale = 1.0 / Math.sqrt(headDim);
    const attnOutput = api.scaledDotProductAttention(q, k, v, scale, true);

    profileModule(api, `block${i}.attn.proj`);
    const attnConcat = attnOutput
      .permute([0, 2, 1, 3])
      .contiguous()
      .reshape([batch, sl, embedDim]);
    const attnProjOut = block.attn.cProj.forward(attnConcat);

    profileModule(api, `block${i}.residual1`);
    let h = api.add(x, attnProjOut);

    // LayerNorm 2
    profileModule(api, `block${i}.ln2`);
    const ln2Out = block.ln2.forward(h);

    // MLP
    profileModule(api, `block${i}.mlp.fc`);
    let mlpH = block.mlp.cFc.forward(ln2Out);

    profileModule(api, `block${i}.mlp.gelu`);
    mlpH = mlpH.gelu();

    profileModule(api, `block${i}.mlp.proj`);
    mlpH = block.mlp.cProj.forward(mlpH);

    profileModule(api, `block${i}.residual2`);
    h = api.add(h, mlpH);

    x = h;
  }

  // --- Final LayerNorm ---
  profileModule(api, "final.ln");
  x = model.lnF.forward(x);

  // --- LM Head ---
  profileModule(api, "final.lm_head");
  const logits = api.linear(x, model.wte.weight, null);

  // --- Loss ---
  profileModule(api, "loss.cross_entropy");
  const [batch2, seqLenT] = targets.shape;
  const flatLogits = logits.reshape([batch2 * seqLenT, model.paddedVocabSize]);
  const realLogits = model.paddedVocabSize > model.config.vocabSize
    ? flatLogits.narrow(1, 0, model.config.vocabSize)
    : flatLogits;
  const flatTargets = targets.reshape([batch2 * seqLenT]);

  const loss = crossEntropy(api, realLogits, flatTargets);

  profileModule(api, "unknown");
  return loss;
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  if (!isProfilingEnabled()) {
    console.error("ERROR: Set TORCHLETTE_PROFILE=1 to enable profiling");
    console.error("Usage: TORCHLETTE_PROFILE=1 npx tsx tools/profile-training.ts");
    process.exit(1);
  }

  console.log("=== DistilGPT-2 Training Profiler ===\n");

  // Initialize
  const gpuOk = await initWebGPU();
  if (!gpuOk) {
    console.error("ERROR: WebGPU initialization failed (is the GPU available?)");
    process.exit(1);
  }

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  // Load model
  console.log("Loading model...");
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.train();
  console.log("Model loaded.\n");

  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
  const useAMP = isF16Supported();
  const scaler = useAMP ? new GradScaler(api, { initScale: 1024.0 }) : null;
  console.log(`AMP (f16): ${useAMP ? "enabled" : "disabled (shader-f16 not available)"}`);
  console.log(`Batch size: ${BATCH_SIZE}, Sequence length: ${FIXED_TOKENS.length - 1}\n`);

  // Compiled forward with module instrumentation
  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    if (useAMP) {
      return api.autocast(() => {
        return instrumentedForwardWithLoss(model, api, input, target);
      });
    }
    return instrumentedForwardWithLoss(model, api, input, target);
  });

  // Register backward dispatch hook: propagate forward module labels into backward phase
  api.onBackwardDispatch(({ label }) => {
    if (label) setProfileModule(label);
  });

  const inputTokens = FIXED_TOKENS.slice(0, -1); // 31 tokens
  const targetTokens = FIXED_TOKENS.slice(1); // 31 tokens
  // Tile tokens across batch dimension
  const inputData: number[] = [];
  const targetData: number[] = [];
  for (let b = 0; b < BATCH_SIZE; b++) {
    inputData.push(...inputTokens);
    targetData.push(...targetTokens);
  }
  const seqLen = inputTokens.length;

  const allTimings: { step: number; loss: number; fwd: number; bwd: number; opt: number; cleanup: number; submits: number }[] = [];

  // Per-step memory tracking for leak detection
  const stepMemory: Array<{
    storages: number;
    reachable: number;
    pendingDestroy: number;
    currentMB: number;
    peakMB: number;
    allocCount: number;
    poolReuseRate: number;
  }> = [];

  // Reset flow counters for clean tracking
  getAndResetFlowCounters();
  storageTracker.debugCounters();

  for (let step = 0; step < NUM_STEPS; step++) {
    if (scaler) await scaler.resolveDeferred();
    await api.beginStep();
    const input = api.tensorFromArray(inputData, [BATCH_SIZE, seqLen], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [BATCH_SIZE, seqLen], { device: "webgpu" });

    resetSubmitCount();
    resetProfileStats();
    resetBufferPoolDetailedStats();
    resetBindGroupCacheStats();

    // V100/Dawn workaround: accumulated resolveQuerySet operations corrupt Vulkan
    // fence state, causing mapAsync deadlocks. Only enable GPU timestamps for the
    // step we actually want to profile.
    setTimestampsEnabled(step === PROFILE_STEP);

    // --- Forward ---
    setProfilePhase("forward");
    const t0 = performance.now();
    const loss = compiledForward(input, target);
    const lossValue = await loss.item();
    const t1 = performance.now();

    // --- Backward ---
    setProfilePhase("backward");
    let scaledLoss: Tensor | null = null;
    if (scaler) {
      scaledLoss = scaler.scale(loss);
      await scaledLoss.backward();
    } else {
      await loss.backward();
    }
    setProfileModule("unknown");
    const t2 = performance.now();


    // --- Optimizer ---
    setProfilePhase("optimizer");
    if (scaler) {
      setProfileModule("optimizer.unscale");
      scaler.unscale_(optimizer);
      setProfileModule("optimizer.step");
      scaler.step(optimizer);
      setProfileModule("optimizer.update");
      scaler.update();
    } else {
      setProfileModule("optimizer.step");
      optimizer.step();
    }
    setProfileModule("optimizer.zero_grad");
    optimizer.zeroGrad();
    setProfileModule("unknown");
    const t3 = performance.now();


    // Capture fusion stats
    const cumulativeStats = api._runtime().getCumulativeFusionStats();
    const lastStats = api._runtime().getLastFusionStats();

    // --- Cleanup ---
    setProfilePhase("cleanup");
    if (scaledLoss) scaledLoss.dispose();
    loss.dispose();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
    const t4 = performance.now();

    // Read GPU timestamps (only the profile step has timestamp data)
    await readGpuTimestamps();

    // Collect memory stats for leak detection
    const storageStats = storageTracker.stats();
    const memStatsStep = getGPUMemoryStats();
    const poolStatsStep = getBufferPoolStats();
    stepMemory.push({
      storages: storageStats.totalStorages,
      reachable: storageStats.reachableStorages,
      pendingDestroy: poolStatsStep.pendingDestroy ?? 0,
      currentMB: memStatsStep.currentBytes / 1024 / 1024,
      peakMB: memStatsStep.peakBytes / 1024 / 1024,
      allocCount: memStatsStep.allocationCount,
      poolReuseRate: poolStatsStep.reuseRate * 100,
    });

    const stepSubmits = getSubmitCount();
    allTimings.push({
      step,
      loss: lossValue,
      fwd: t1 - t0,
      bwd: t2 - t1,
      opt: t3 - t2,
      cleanup: t4 - t3,
      submits: stepSubmits,
    });

    // Print summary for the target step
    if (step === PROFILE_STEP) {
      console.log(`\n${"=".repeat(80)}`);
      console.log(`DETAILED PROFILE — Step ${step} (steady-state)`);
      console.log(`${"=".repeat(80)}\n`);

      printProfileSummary(`step ${step}`);
      await writeProfileJSON(`/tmp/torchlette-profile-step${step}.json`);

      // Memory stats
      const memStats = getGPUMemoryStats();
      const poolStats = getBufferPoolStats();
      const detailedStats = getBufferPoolDetailedStats();
      const histogram = getGPUAllocationHistogram();
      console.log("=== Memory Stats ===\n");
      if (memStats) {
        console.log(`GPU Memory: ${(memStats.currentBytes / 1024 / 1024).toFixed(1)}MB current / ${(memStats.limitBytes / 1024 / 1024).toFixed(0)}MB limit (${memStats.allocationCount} live buffers)`);
        console.log(`Peak Memory: ${(memStats.peakBytes / 1024 / 1024).toFixed(1)}MB (${memStats.usagePercent.toFixed(1)}% current utilization)`);
      }
      if (poolStats) {
        console.log(`\nBuffer Pool:`);
        console.log(`  Pooled: ${poolStats.pooledBuffers} buffers (${(poolStats.pooledBytes / 1024 / 1024).toFixed(1)}MB)`);
        console.log(`  Pending release: ${poolStats.pendingBuffers} buffers`);
        console.log(`  Reuse rate: ${(poolStats.reuseRate * 100).toFixed(1)}% (reuse: ${poolStats.reuseCount}, new alloc: ${poolStats.allocCount})`);
      }
      if (detailedStats) {
        console.log(`\nPool Acquire/Release (this step):`);
        console.log(`  Acquire from pool: ${detailedStats.acquireFromPool} | from pending: ${detailedStats.acquireFromPending} | new alloc: ${detailedStats.acquireNew}`);
        console.log(`  Release to pool: ${detailedStats.releaseToPool} | to destroy: ${detailedStats.releaseToDestroy}`);
      }
      if (histogram && histogram.size > 0) {
        console.log(`\nAllocation Size Histogram:`);
        for (const [bucket, stats] of histogram) {
          const bytesStr = stats.totalBytes >= 1024 * 1024 * 1024
            ? `${(stats.totalBytes / 1024 / 1024 / 1024).toFixed(1)}GB`
            : stats.totalBytes >= 1024 * 1024
              ? `${(stats.totalBytes / 1024 / 1024).toFixed(1)}MB`
              : stats.totalBytes >= 1024
                ? `${(stats.totalBytes / 1024).toFixed(1)}KB`
                : `${stats.totalBytes}B`;
          console.log(`  ${bucket.padEnd(10)} ${String(stats.count).padStart(4)} buffers (${bytesStr.padStart(8)})`);
        }
      }
      console.log();

      // Bind group cache stats
      const bgStats = getBindGroupCacheStats();
      console.log(`Bind Group Cache: ${bgStats.hits} hits / ${bgStats.misses} misses (${(bgStats.hitRate * 100).toFixed(1)}%), ${bgStats.size} cached entries`);
      const arenaStats = getArenaResolveStats();
      console.log(`Arena resolve: ${arenaStats.hits} hits, ${arenaStats.aliased} aliased (fallthrough), ${arenaStats.noArena} no-arena`);
      const missLog = getBindGroupCacheMissLog();
      if (missLog.length > 0) {
        console.log(`\nBind Group Cache Misses (step 4):`);
        console.log(`${"Idx".padStart(5)}  ${"Reason".padEnd(20)}  ${"Label".padEnd(28)}  Details`);
        console.log(`${"─".repeat(5)}  ${"─".repeat(20)}  ${"─".repeat(28)}  ${"─".repeat(40)}`);
        for (const m of missLog) {
          console.log(`${String(m.idx).padStart(5)}  ${m.reason.padEnd(20)}  ${(m.label ?? "(null)").padEnd(28)}  ${m.details}`);
        }
      }
      console.log();

      // Fusion stats
      if (cumulativeStats) {
        console.log("=== Fusion Stats ===\n");
        console.log(`Total nodes: ${cumulativeStats.totalNodes}`);
        console.log(`Fused nodes: ${cumulativeStats.fusedNodes} (${(cumulativeStats.fusedNodes / cumulativeStats.totalNodes * 100).toFixed(1)}%)`);
        console.log(`Fusion groups: ${cumulativeStats.fusionGroups}`);
        console.log(`Sequential nodes: ${cumulativeStats.sequentialNodes}`);
        console.log();
      }
    } else {
      // Brief summary for non-target steps with memory info
      const stepMem = stepMemory[stepMemory.length - 1];
      const stepPoolStats = getBufferPoolDetailedStats();
      console.log(`Step ${step}: loss=${lossValue.toFixed(4)}, fwd=${(t1 - t0).toFixed(0)}ms, bwd=${(t2 - t1).toFixed(0)}ms, opt=${(t3 - t2).toFixed(0)}ms, cleanup=${(t4 - t3).toFixed(0)}ms | mem=${stepMem.currentMB.toFixed(1)}MB, pool=${stepMem.poolReuseRate.toFixed(1)}%, new_alloc=${stepPoolStats.acquireNew}`);
    }
  }

  // Final summary table
  console.log("\n" + "=".repeat(80));
  console.log("WALL CLOCK SUMMARY");
  console.log("=".repeat(80));
  console.log("\n| Step | Loss     | Fwd(ms) | Bwd(ms) | Opt(ms) | Cleanup(ms) | Total(ms) | Submits |");
  console.log("|------|----------|---------|---------|---------|-------------|-----------|---------|");
  for (const t of allTimings) {
    const total = t.fwd + t.bwd + t.opt + t.cleanup;
    console.log(
      `| ${t.step}    | ${t.loss.toFixed(4)} | ${t.fwd.toFixed(0).padStart(7)} | ${t.bwd.toFixed(0).padStart(7)} | ${t.opt.toFixed(0).padStart(7)} | ${t.cleanup.toFixed(0).padStart(11)} | ${total.toFixed(0).padStart(9)} | ${String(t.submits).padStart(7)} |`,
    );
  }

  // Steady-state averages (steps 2-4)
  const steadyState = allTimings.filter(t => t.step >= 2);
  if (steadyState.length > 0) {
    const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
    console.log(`\nSteady-state avg (steps 2-${NUM_STEPS - 1}):`);
    console.log(`  Forward:  ${avg(steadyState.map(t => t.fwd)).toFixed(0)}ms`);
    console.log(`  Backward: ${avg(steadyState.map(t => t.bwd)).toFixed(0)}ms`);
    console.log(`  Optimizer: ${avg(steadyState.map(t => t.opt)).toFixed(0)}ms`);
    console.log(`  Cleanup:  ${avg(steadyState.map(t => t.cleanup)).toFixed(0)}ms`);
    const totalAvg = avg(steadyState.map(t => t.fwd + t.bwd + t.opt + t.cleanup));
    console.log(`  Total:    ${totalAvg.toFixed(0)}ms`);
  }

  // === Memory Leak Report ===
  console.log("\n" + "=".repeat(80));
  console.log("MEMORY LEAK REPORT");
  console.log("=".repeat(80));
  console.log("\n| Step | Storages | Reachable | PeakMB   | CurrentMB | PoolReuse% | PendingDestroy | Allocs |");
  console.log("|------|----------|-----------|----------|-----------|------------|----------------|--------|");
  for (let i = 0; i < stepMemory.length; i++) {
    const m = stepMemory[i];
    console.log(
      `| ${String(i).padStart(4)} | ${String(m.storages).padStart(8)} | ${String(m.reachable).padStart(9)} | ${m.peakMB.toFixed(1).padStart(8)} | ${m.currentMB.toFixed(1).padStart(9)} | ${m.poolReuseRate.toFixed(1).padStart(10)} | ${String(m.pendingDestroy).padStart(14)} | ${String(m.allocCount).padStart(6)} |`,
    );
  }

  // Compute steady-state deltas (use last 3 steps, skipping warmup)
  let leakDetected = false;
  if (stepMemory.length >= 3) {
    const steadyStart = Math.max(1, stepMemory.length - 3);
    const steadyEnd = stepMemory.length - 1;
    const n = steadyEnd - steadyStart;
    if (n > 0) {
      const storagesDelta = (stepMemory[steadyEnd].storages - stepMemory[steadyStart].storages) / n;
      const reachableDelta = (stepMemory[steadyEnd].reachable - stepMemory[steadyStart].reachable) / n;
      const trackerDelta = (stepMemory[steadyEnd].currentMB - stepMemory[steadyStart].currentMB) / n;
      const allocDelta = (stepMemory[steadyEnd].allocCount - stepMemory[steadyStart].allocCount) / n;
      const pdStart = stepMemory[steadyStart].pendingDestroy;
      const pdEnd = stepMemory[steadyEnd].pendingDestroy;
      const pdGrowing = pdEnd > pdStart + 10;

      console.log(`\nSteady-state delta (avg steps ${steadyStart}-${steadyEnd}):`);
      console.log(`  Storages/step: ${storagesDelta >= 0 ? "+" : ""}${storagesDelta.toFixed(1)}`);
      console.log(`  Reachable/step: ${reachableDelta >= 0 ? "+" : ""}${reachableDelta.toFixed(1)}`);
      console.log(`  CurrentMB/step: ${trackerDelta >= 0 ? "+" : ""}${trackerDelta.toFixed(1)}`);
      console.log(`  Allocs/step: ${allocDelta >= 0 ? "+" : ""}${allocDelta.toFixed(1)}`);
      console.log(`  PendingDestroy: ${pdStart} -> ${pdEnd} (${pdGrowing ? "GROWING" : "stable"})`);

      if (pdGrowing) {
        console.log("\nLEAK STATUS: FAIL - pendingDestroy is growing (physical GPU buffer leak)");
        leakDetected = true;
      } else if (allocDelta > 5 || reachableDelta > 5) {
        console.log(`\nLEAK STATUS: WARNING - +${allocDelta.toFixed(0)} allocs/step, +${reachableDelta.toFixed(0)} reachable/step`);
      } else {
        console.log("\nLEAK STATUS: OK");
      }
    }
  }

  console.log("\nProfile JSON written to /tmp/torchlette-profile-step4.json");
  console.log("Done.");
  destroyWebGPU();

  // Exit with code 1 if physical leak detected (for CI)
  if (leakDetected) {
    process.exit(1);
  }
}

main().then(() => process.exit(0)).catch((e) => {
  console.error(e);
  process.exit(1);
});
