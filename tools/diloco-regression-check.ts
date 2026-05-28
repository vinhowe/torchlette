/**
 * Regression-protection harness for the DiLoCo trainer.
 *
 * Runs the **production** WebGPUGPT2Trainer for 10 rounds × 20 inner steps
 * against the local cached TinyStories token blob, then asserts the loss
 * trajectory hits recorded checkpoints below a tolerance. Also asserts
 * GPU peak memory doesn't grow round-over-round (catches buffer leaks).
 *
 * Update BASELINE when an intentional change improves convergence. Exits
 * 0 on pass, 1 on regression. Designed to be wired into CI.
 *
 *   VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *     npx tsx tools/diloco-regression-check.ts
 */

import * as fs from "node:fs";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import {
  type TokenSource,
  WebGPUGPT2Trainer,
} from "../src/distributed/protocol/webgpu-gpt2-trainer";

const SEED = 42;
const ROUNDS = 10;
const TOKENS_PATH =
  process.env.LOCAL_TOKENS ??
  "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin";

// Baseline trajectory recorded 2026-05-28 on dw-2-1 V100, AC ON + scaler +
// AdamW + plain GPT2 (no LoRA wrapper) + no-accumGrads path + selective
// checkpointing + nanoGPT init (scaled residual projections, zeroed biases).
// Loss should descend monotonically past these checkpoints. Tolerance
// accounts for small variance from scaler scale adjustments + nondeterministic
// kernel reductions; a regression > 0.4 nats here means training broke.
const BASELINE: Record<number, number> = {
  0: 9.56,
  3: 5.79,
  6: 5.24,
  9: 4.91,
};
const LOSS_TOLERANCE = 0.4;
// Steady-state peak GPU memory tolerance (after warmup): catches leaks
// without flagging the pool's small ramp-up over the first few rounds.
const MEM_GROWTH_MB_MAX = 50;

const log = (m: string) => console.error(`[regression] ${m}`);

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
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "4000";

  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(SEED);

  const tokenSource = new FullCacheTokenSource(TOKENS_PATH);
  tokenSource.load();
  log(`Tokens loaded: ${tokenSource.load().length.toLocaleString()}`);

  const trainer = new WebGPUGPT2Trainer({
    api,
    modelConfig: {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: 8,
      numHeads: 4,
      embedDim: 128,
      dropoutRate: 0,
    },
    tokenSource,
    innerLr: 5e-4,
    outerLr: 1.0,
    outerMu: 0.0,
    innerSteps: 20,
    batchSize: 8,
    seqLen: 256,
    accumSteps: 1,
    weightDecay: 0.01,
    checkpointing: true,
    useAutocast: true,
    gradClipNorm: 1.0,
    log: (m: string) => log(`trainer: ${m}`),
  });
  await trainer.initialize();
  // Setting anchor here matches what the SM does before the run loop —
  // the harness exercises trainer.innerSteps directly so we have to do it.
  await trainer.setAnchor();

  const { getGPUMemoryStats } = await import(
    "../src/backend/webgpu/memory-tracker"
  );
  const losses: number[] = [];
  const peakMems: number[] = [];
  for (let r = 0; r < ROUNDS; r++) {
    const t0 = Date.now();
    const loss = await trainer.innerSteps(r);
    losses.push(loss);
    const mem = getGPUMemoryStats();
    peakMems.push(mem.peakBytes / 1e6);
    log(
      `round ${r}: loss=${loss.toFixed(4)} peak_mb=${(mem.peakBytes / 1e6).toFixed(1)} dt=${((Date.now() - t0) / 1000).toFixed(2)}s`,
    );
  }

  // ---- Check loss thresholds ----
  log("");
  log("=== Loss regression check ===");
  let pass = true;
  for (const [roundStr, expected] of Object.entries(BASELINE)) {
    const r = parseInt(roundStr, 10);
    const actual = losses[r]!;
    const ok = actual <= expected + LOSS_TOLERANCE;
    const status = ok ? "OK  " : "FAIL";
    log(
      `${status} round=${r} expected<=${(expected + LOSS_TOLERANCE).toFixed(2)} actual=${actual.toFixed(4)} (baseline=${expected.toFixed(2)})`,
    );
    if (!ok) pass = false;
  }

  // ---- Check memory doesn't grow round-over-round ----
  log("");
  log("=== Memory growth check ===");
  // Skip the first 2 rounds (warmup / pool ramp). Then peak should be flat.
  const tail = peakMems.slice(2);
  const minTail = Math.min(...tail);
  const maxTail = Math.max(...tail);
  const memGrowthMb = maxTail - minTail;
  const memOk = memGrowthMb < MEM_GROWTH_MB_MAX;
  log(
    `${memOk ? "OK  " : "FAIL"} peak growth rounds 2-${ROUNDS - 1}: ${memGrowthMb.toFixed(1)} MB (max ${maxTail.toFixed(1)}, min ${minTail.toFixed(1)}, tolerance ${MEM_GROWTH_MB_MAX} MB)`,
  );
  if (!memOk) pass = false;

  log("");
  log(pass ? "PASS" : "FAIL");

  await destroyWebGPU();
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
