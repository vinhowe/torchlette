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

import { execSync } from "node:child_process";
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
// scalar-output correctness fixes; then 2026-05-31 (recordedCopyBufferToBuffer
// embedding-grad fix, →~4.78) and 2026-06-10 (TAG_UNIFORM volatile step_size
// fix, →4.64). See git history for the full provenance of each re-record.
//
// ---------------------------------------------------------------------------
// HARDWARE-QUALIFIED TOLERANCES (measured 2026-07-23, tolerance study).
// ---------------------------------------------------------------------------
// The run is deterministic in its INPUTS (seed 42 → fixed init; the window
// sampler is a fixed LCG), so all run-to-run variance is nondeterministic GPU
// kernel reductions (parallel-reduction ordering + scatterAdd atomics). That
// variance is HARDWARE-dependent, so a single tolerance mis-fires: the old
// informal "rounds 3/6/9 within 3e-4" band was measured on A100 and false-alarms
// on V100, whose band is ~6× wider.
//
// Measured run-to-run bands (max spread of the round loss across repeats):
//   • A100 (dw-2-1): 8 runs × 2 physical GPUs → r0 1.2e-6, r3 1.7e-4,
//     r6 2.6e-4, r9 3.3e-4. Band(r3/6/9) = 3.3e-4.
//   • V100 (sivri):  12 runs × 3 physical GPUs → r0 1.1e-6, r3 6.7e-4,
//     r6 2.0e-3, r9 1.8e-3. Band(r3/6/9) = 2.0e-3.
// The variance is RUN-TO-RUN, not device-to-device: a single V100 already spans
// the full 2.0e-3 band across repeats; per-device means are indistinguishable.
// Round 0 is NOT bit-exact — it drifts ~1.2e-6 run-to-run (and the baseline
// VALUE itself differs ~1e-3 between A100 9.80789 and V100 9.80891, so the
// baselines are recorded per-hardware). Cross-hardware trajectory means differ
// by up to 3.4e-3 (r6), which is why one shared baseline+tolerance cannot serve
// both — REG_HW / nvidia-smi selects the profile below.
//
// Baselines are the per-hardware MEANS. Tolerance = measured band × 1.5 (small-
// sample safety headroom; the realized worst-case excursion from the mean is
// only ~band/2, so this is ~3× the observed excursion yet still orders of
// magnitude tighter than a real regression, which the fix history above shows
// lands at 0.1–1.0 nats). The check is a two-sided band |actual − baseline| ≤
// tol: this replaces the old one-sided coarse 0.4 guard, and because the band
// is tight, an intentional convergence change (better OR worse) trips it and
// must be re-baselined — matching the "Update BASELINE when …" policy above.
type HwName = "a100" | "v100";
interface HwProfile {
  label: string;
  baseline: Record<number, number>;
  tolR0: number; // round 0 is near-deterministic; its own tiny tolerance
  tol: number; // rounds 3/6/9: measured run-to-run band × 1.5
}
const HW_PROFILES: Record<HwName, HwProfile> = {
  a100: {
    label: "A100",
    baseline: { 0: 9.807889, 3: 5.922661, 6: 5.150417, 9: 4.637923 },
    tolR0: 1e-5, // measured spread 1.2e-6
    tol: 5e-4, // band 3.3e-4 × 1.5 ≈ 4.9e-4
  },
  v100: {
    label: "V100",
    baseline: { 0: 9.808891, 3: 5.921947, 6: 5.153782, 9: 4.640370 },
    tolR0: 1e-5, // measured spread 1.1e-6
    tol: 3e-3, // band 2.0e-3 × 1.5 = 3.0e-3
  },
};

// Detect the GPU class from nvidia-smi (env override REG_HW=a100|v100). Unknown
// hardware defaults to the WIDER (V100) band so an unrecognized box cannot
// false-alarm — better to miss a subtle regression on unknown hw than to fail
// spuriously; the round-0 guard (bit-near-exact everywhere) still catches gross
// breakage.
function detectHardware(): HwName {
  const override = process.env.REG_HW?.toLowerCase();
  if (override === "a100" || override === "v100") return override;
  try {
    const name =
      execSync("nvidia-smi --query-gpu=name --format=csv,noheader", {
        encoding: "utf8",
      })
        .split("\n")[0]
        ?.trim() ?? "";
    if (/A100/i.test(name)) return "a100";
    if (/V100/i.test(name)) return "v100";
    log(`WARN: unrecognized GPU '${name}' — defaulting to V100 (wider) band`);
  } catch (e) {
    log(`WARN: nvidia-smi unavailable (${e}) — defaulting to V100 (wider) band`);
  }
  return "v100";
}
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

  // ---- Check loss thresholds (hardware-qualified two-sided band) ----
  const profile = HW_PROFILES[detectHardware()];
  log("");
  log(
    `=== Loss regression check (hw=${profile.label}, ` +
      `tol r0=${profile.tolR0}, r3/6/9=${profile.tol}) ===`,
  );
  let pass = true;
  for (const [roundStr, expected] of Object.entries(profile.baseline)) {
    const r = parseInt(roundStr, 10);
    const actual = losses[r]!;
    const tol = r === 0 ? profile.tolR0 : profile.tol;
    const dev = Math.abs(actual - expected);
    const ok = dev <= tol;
    const status = ok ? "OK  " : "FAIL";
    log(
      `${status} round=${r} baseline=${expected.toFixed(6)} actual=${actual.toFixed(6)} |dev|=${dev.toExponential(2)} tol=${tol.toExponential(2)}`,
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
