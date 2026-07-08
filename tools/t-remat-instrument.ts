/**
 * Stage-3 S3.0 instrumentation probe (docs/stage4-compile-from-ir.md §Stage 3).
 *
 * Runs the PRODUCTION WebGPUGPT2Trainer on the compiled path (checkpointing
 * configurable; TORCHLETTE_CHECKPOINT_ARENA=1 keeps the arena so build-from-IR
 * engages under checkpointing) and reports what the step-global release seam
 * (S3.1) WOULD reclaim: per-producer-template releasable result bytes grouped
 * by the observed stable last reader, alongside the registry/memory totals.
 * Pure observation — no behavior change; this is the G0-discipline check that
 * the lifetime model predicts the measured compiled-vs-arena-free gap before
 * any buffer moves.
 *
 *   VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *     REG_EMBED=768 REG_LAYERS=12 REG_HEADS=12 REG_BATCH=1 \
 *     TORCHLETTE_CHECKPOINT_ARENA=1 npx tsx tools/t-remat-instrument.ts
 */

import * as fs from "node:fs";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { getGPUMemoryStats } from "../src/backend/webgpu/memory-tracker";
import { bufferPool } from "../src/backend/webgpu/buffer-pool";
import { Torchlette } from "../src/frontend/torchlette";
import {
  type TokenSource,
  WebGPUGPT2Trainer,
} from "../src/distributed/protocol/webgpu-gpt2-trainer";
import { debugPlannerRegistryStats } from "../src/executor/compiled-plan";
import {
  debugReleasableSummary,
  debugTopHeldPairs,
  getObservedLivenessStats,
} from "../src/executor/observed-liveness";

const ROUNDS = parseInt(process.env.REG_ROUNDS ?? "5", 10);
const TOKENS_PATH =
  process.env.LOCAL_TOKENS ??
  "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin";

const log = (m: string) => console.error(`[s3.0] ${m}`);

class FullCacheTokenSource implements TokenSource {
  private cached: Uint16Array | null = null;
  constructor(private readonly path: string) {}
  load(): Uint16Array {
    if (this.cached) return this.cached;
    const buf = fs.readFileSync(this.path);
    this.cached = new Uint16Array(
      buf.buffer,
      buf.byteOffset,
      buf.byteLength / 2,
    );
    return this.cached;
  }
  async fetch(_minTokens: number): Promise<ArrayLike<number>> {
    return this.load();
  }
}

async function main() {
  if (!(await initWebGPU())) {
    log("FATAL: WebGPU init failed");
    process.exit(1);
  }
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "12000";

  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(42);
  const tokenSource = new FullCacheTokenSource(TOKENS_PATH);
  tokenSource.load();

  const trainer = new WebGPUGPT2Trainer({
    api,
    modelConfig: {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: parseInt(process.env.REG_LAYERS ?? "8", 10),
      numHeads: parseInt(process.env.REG_HEADS ?? "4", 10),
      embedDim: parseInt(process.env.REG_EMBED ?? "128", 10),
      dropoutRate: 0,
    },
    tokenSource,
    innerLr: 5e-4,
    outerLr: 1.0,
    outerMu: 0.0,
    innerSteps: 20,
    batchSize: parseInt(process.env.REG_BATCH ?? "8", 10),
    seqLen: parseInt(process.env.REG_SEQ ?? "256", 10),
    accumSteps: 1,
    weightDecay: 0.01,
    checkpointing: process.env.CHECKPOINTING !== "0",
    useAutocast: true,
    gradClipNorm: 1.0,
    log: (m: string) => log(`trainer: ${m}`),
  });
  await trainer.initialize();
  await trainer.setAnchor();

  for (let r = 0; r < ROUNDS; r++) {
    const t0 = Date.now();
    const loss = await trainer.innerSteps(r);
    const mem = getGPUMemoryStats();
    // Unified PHYSICAL meter (coordinator metric ruling): tracker-current +
    // pool-held (pooled + pendingRelease) — pool-idle bytes are physically
    // resident but trackDeallocation'd, so `cur` alone under-reports the
    // lowered arm (which cycles everything through the pool) vs the compiled
    // arm (whose registry entries stay tracker-counted). Report BOTH.
    const poolHeld = bufferPool.getTotalHeldBytes();
    log(
      `round ${r}: loss=${loss.toFixed(4)} peak_mb=${(mem.peakBytes / 1e6).toFixed(1)} cur_mb=${(mem.currentBytes / 1e6).toFixed(1)} phys_mb=${((mem.currentBytes + poolHeld) / 1e6).toFixed(1)} (pool_held=${(poolHeld / 1e6).toFixed(1)}) dt=${((Date.now() - t0) / 1000).toFixed(2)}s`,
    );
  }

  log("");
  log(`registry: ${JSON.stringify(debugPlannerRegistryStats())}`);
  const { debugTemplatePlanMemory, debugCrossPlanPersistentBindings } =
    await import("../src/executor/executor");
  log(`templates: ${JSON.stringify(debugTemplatePlanMemory())}`);
  log(
    `crossPlanPersistentBindings: ${JSON.stringify(debugCrossPlanPersistentBindings())}`,
  );
  log(`stats: ${JSON.stringify(getObservedLivenessStats())}`);
  log("=== step-global releasable summary (per producer template) ===");
  const summary = debugReleasableSummary();
  for (const [fp, s] of Object.entries(summary)) {
    log(
      `${fp}: resultMB=${s.resultMB} releasableMB=${s.releasableMB} (${s.releasablePairs} pairs) byLastReader=${JSON.stringify(s.byLastReader)} heldMB=${JSON.stringify(s.heldMB)}`,
    );
  }
  log("=== top held pairs per template ===");
  for (const [fp, rows] of Object.entries(debugTopHeldPairs(12))) {
    log(`${fp}:`);
    for (const r of rows) {
      log(
        `  ${r.pair} ${r.op ?? "?"} ${r.MB}MB cls=${r.cls}${r.lastReader ? ` lastReader=${r.lastReader}` : ""}`,
      );
    }
  }
  let totalRel = 0;
  for (const s of Object.values(summary)) totalRel += s.releasableMB;
  log(`TOTAL releasable: ${totalRel.toFixed(1)} MB`);

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
