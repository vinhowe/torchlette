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
const ROUNDS = parseInt(process.env.REG_ROUNDS ?? "10", 10);
const TOKENS_PATH =
  process.env.LOCAL_TOKENS ??
  "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin";
// The baseline trajectory is a function of the input blob: the canonical
// tinystories-tokens.bin has exactly this many tokens. Any other blob (or a
// REG_* override) silently shifts round-0 loss and every baseline after it —
// a non-canonical input must be LOUD, not a mysterious "regression".
const CANONICAL_TOKEN_COUNT = 473_992_236;

// Baseline trajectory re-verified 2026-05-29 on V100 (sivri), AC ON + scaler +
// AdamW + plain GPT2 (no LoRA wrapper) + no-accumGrads path + selective
// checkpointing + nanoGPT init (scaled residual projections, zeroed biases).
// Re-recorded after the CSE-outputIndex, integer-pow, and row-program
// scalar-output correctness fixes — the self-training trajectory is unchanged
// from the original 2026-05-28 numbers within run-to-run noise (those fixes
// matter for matched-weight gradient parity vs PyTorch, not this self-play loss
// curve, since wrong-but-correlated grads still descend). Loss should descend
// monotonically past these checkpoints. Tolerance absorbs scaler-scale jitter +
// nondeterministic kernel reductions; a regression > 0.4 nats means training broke.
// Re-recorded 2026-05-31 after the compiled-plan intra-plan-copy fix
// (recordedCopyBufferToBuffer): the default/compiled path's embedding grads
// were inflating +1x/replay (scatterAdd's a→out copy was unreplayed), which
// made this trajectory converge to ~4.92; with correct grads it reached ~4.78.
// Re-recorded 2026-06-10 after the volatile-uniform fix (TAG_UNIFORM): the
// compiled optimizer plan was replaying Adam's bias-corrected step_size frozen
// at record time (t of the recording step, forever), i.e. a wrong LR schedule.
// That was the REAL cause of the 4.78-vs-lowered-4.64 gap previously written
// off as "benign clip-amplified fp32 noise". With per-replay config re-derive,
// the compiled path now matches the lowered trajectory: faster early descent
// is gone (round 0: 9.54→9.81, the frozen early-t step size was inflated) and
// final convergence improves (round 9: 4.78→4.64).
const BASELINE: Record<number, number> = {
  0: 9.81,
  3: 5.92,
  6: 5.15,
  9: 4.64,
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
  const tokenCount = tokenSource.load().length;
  log(`Tokens loaded: ${tokenCount.toLocaleString()} (${TOKENS_PATH})`);
  if (tokenCount !== CANONICAL_TOKEN_COUNT) {
    log(
      `FATAL: non-canonical token blob (${tokenCount.toLocaleString()} tokens, ` +
        `expected ${CANONICAL_TOKEN_COUNT.toLocaleString()}). The baselines are ` +
        `only valid against the canonical blob — fix LOCAL_TOKENS before ` +
        `interpreting any loss delta as a regression.`,
    );
    process.exit(2);
  }

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
    // Feature toggles (default on) — used to localize compiled-vs-lowered diffs.
    checkpointing: process.env.CHECKPOINTING !== "0",
    useAutocast: process.env.USE_AUTOCAST !== "0",
    gradClipNorm: process.env.GRAD_CLIP === "0" ? undefined : 1.0,
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
