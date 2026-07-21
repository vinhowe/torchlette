/**
 * Coverage-campaign C-EXIT: the packed-vs-fused peak-memory premium in the
 * WHOLE-STEP REMAT regime (the real-training substrate), measured per-arm in an
 * ISOLATED child process (flags are read once at module load). Reports late-step
 * FLAT peak + per-step storage slope (must be ~0 = flat, per agent-ops).
 *
 * Arms (ARM env): fused | packed | packed-donation | packed-split | packed-split-donation
 *   fused                : default (WGSL adamStep) — the baseline.
 *   packed               : TORCHLETTE_FUSED_ADAM=0 (foreach → packOptimizerClass), donation OFF.
 *   packed-donation      : packed + TORCHLETTE_PLANNER_DONATION=1.
 *   packed-split         : packed + TORCHLETTE_PACK_MAX_BYTES (class-split), donation OFF
 *                          (measures the §5.2 transient-shrink dividend ALONE).
 *   packed-split-donation: packed + split + donation.
 *
 * Run the driver (spawns all arms): npx tsx tools/t-donation-premium.ts
 * Env: MODEL(distilgpt2) SEQ_LEN(512) STEPS(16) PACK_MAX_BYTES(for split arms).
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { getGPUMemoryStats, resetGPUMemoryPeak, setGPUMemoryLimit } from "../src/backend/webgpu/memory-tracker";
import { getSubmitCount, resetSubmitCount } from "../src/backend/webgpu/webgpu-state";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler } from "../src/optim/index.ts";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import type { Tensor } from "../src/frontend/tensor";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "512", 10);
const STEPS = parseInt(process.env.STEPS ?? "16", 10);
const LR = parseFloat(process.env.LR ?? "1e-4");
const log = (m: string) => console.error(`[premium] ${m}`);

type Arm =
  | "fused"
  | "packed"
  | "packed-donation"
  | "packed-split"
  | "packed-split-donation";

interface ArmResult {
  arm: Arm;
  losses: number[];
  peakMB: number;
  currentMB: number;
  lateSubmits: number;
  storageSlope: number;
  lateStorages: number;
}

async function run(arm: Arm): Promise<ArmResult> {
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  const model = await loadPretrainedGPT2(
    api,
    path.join(ROOT, "models", MODEL),
    { dropoutRate: 0 },
    { device: "webgpu" },
  );
  model.train(true);
  const opt = new Adam(model.parameters(), { lr: LR, weightDecay: 0.0 }, api);
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;

  const TOKENS =
    process.env.LOCAL_TOKENS ?? path.join(ROOT, "ckpts", "tinystories-tokens.bin");
  const tb = fs.readFileSync(TOKENS);
  const toks = new Uint16Array(tb.buffer, tb.byteOffset, tb.byteLength / 2);
  const inp = new Int32Array(SEQ);
  const tgt = new Int32Array(SEQ);

  const losses: number[] = [];
  const submits: number[] = [];
  const storages: number[] = [];
  const WARMUP = Math.floor(STEPS / 2); // reset peak after pool reuse settles
  for (let step = 0; step < STEPS; step++) {
    if (step === WARMUP) resetGPUMemoryPeak(); // steady-state flat peak only
    resetSubmitCount();
    await scaler.resolveDeferred();
    const base = (step * SEQ) % (toks.length - SEQ - 1);
    for (let i = 0; i < SEQ; i++) {
      inp[i] = toks[base + i]! % V;
      tgt[i] = toks[base + i + 1]! % V;
    }
    const input = api.tensorFromArray(Array.from(inp), [1, SEQ], { device: "webgpu" });
    const target = api.tensorFromArray(Array.from(tgt), [1, SEQ], { device: "webgpu" });

    await api.beginStep();
    const readLoss: Tensor = await api.wholeStep(async () => {
      const loss = api.tidy(() => {
        const l = model.forwardWithLoss(input, target).loss!;
        api.keep(l);
        return l;
      });
      const lossOut = api.noGrad(() => api.mul(loss, 1));
      api.registerState(lossOut);
      await scaler.scale(loss).backward();
      scaler.unscale_(opt);
      scaler.step(opt);
      scaler.update();
      opt.zeroGrad();
      return lossOut;
    });
    api.endStep();
    await api.markStep();
    losses.push(await readLoss.item());
    readLoss.dispose();
    input.dispose();
    target.dispose();
    submits.push(getSubmitCount());
    const mem = getGPUMemoryStats();
    storages.push(mem?.allocationCount ?? mem?.currentBytes ?? 0);
  }
  const mem = getGPUMemoryStats();
  // storage slope over the late window (flatness).
  const half = Math.floor(STEPS / 2);
  const win = storages.slice(half);
  const n = win.length;
  const xs = win.map((_, i) => i);
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = win.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (xs[i] - mx) * (win[i] - my); den += (xs[i] - mx) ** 2; }
  const slope = den ? num / den : 0;
  const lateSub = submits.slice(half);
  return {
    arm,
    losses,
    peakMB: (mem?.peakBytes ?? 0) / 1024 / 1024,
    currentMB: (mem?.currentBytes ?? 0) / 1024 / 1024,
    lateSubmits: lateSub.reduce((a, b) => a + b, 0) / lateSub.length,
    storageSlope: slope,
    lateStorages: win[n - 1] ?? 0,
  };
}

async function runArmInChild(arm: Arm): Promise<ArmResult> {
  const { execFileSync } = await import("node:child_process");
  const env: Record<string, string> = { ...process.env, PREM_ARM: arm, TORCHLETTE_WHOLE_STEP: "1" };
  if (arm !== "fused") env.TORCHLETTE_FUSED_ADAM = "0";
  if (arm === "packed-donation" || arm === "packed-split-donation")
    env.TORCHLETTE_PLANNER_DONATION = "1";
  if (arm === "packed-split" || arm === "packed-split-donation")
    env.TORCHLETTE_PACK_MAX_BYTES = process.env.PACK_MAX_BYTES ?? "134217728"; // 128 MB
  const out = execFileSync(process.execPath, ["--import", "tsx", __filename], {
    env, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"], maxBuffer: 64 * 1024 * 1024,
  });
  const line = out.split("\n").find((l) => l.startsWith("=== ARM-RESULT === "));
  if (!line) throw new Error(`arm ${arm}: no result line`);
  return JSON.parse(line.slice("=== ARM-RESULT === ".length)) as ArmResult;
}

async function main() {
  const armEnv = process.env.PREM_ARM as Arm | undefined;
  if (armEnv) {
    if (!(await initWebGPU())) { log("WebGPU init failed"); process.exit(1); }
    // Raise the tracker's safety cap (default 32 GB) so gpt2-medium's packed
    // whole-step peak can be measured on the 80 GB A100 rather than throwing.
    setGPUMemoryLimit(Number(process.env.MEM_LIMIT_BYTES ?? 78 * 1024 * 1024 * 1024));
    const r = await run(armEnv);
    console.log(`=== ARM-RESULT === ${JSON.stringify(r)}`);
    destroyWebGPU();
    process.exit(0);
  }

  log(`config: MODEL=${MODEL} SEQ=${SEQ} STEPS=${STEPS}`);
  const arms = (process.env.ARMS ?? "fused,packed,packed-donation,packed-split,packed-split-donation").split(",") as Arm[];
  const results: ArmResult[] = [];
  for (const a of arms) {
    const r = await runArmInChild(a);
    results.push(r);
    log(`${a.padEnd(22)} peak=${r.peakMB.toFixed(1)}MB cur=${r.currentMB.toFixed(1)}MB submits=${r.lateSubmits.toFixed(0)} slope=${r.storageSlope.toFixed(3)} loss[last]=${r.losses[r.losses.length-1]?.toFixed(4)}`);
  }
  const fused = results.find((r) => r.arm === "fused");
  if (fused) {
    log(`--- premium vs fused (peak=${fused.peakMB.toFixed(1)}MB) ---`);
    for (const r of results) {
      if (r.arm === "fused") continue;
      const prem = ((r.peakMB - fused.peakMB) / fused.peakMB) * 100;
      log(`${r.arm.padEnd(22)} +${prem.toFixed(1)}% (+${(r.peakMB - fused.peakMB).toFixed(1)}MB)`);
    }
  }
  // Cross-arm loss agreement (all arms are the same math → should agree).
  const ref = results[0].losses;
  let maxDiff = 0;
  for (const r of results) for (let i = 0; i < ref.length; i++) maxDiff = Math.max(maxDiff, Math.abs(r.losses[i] - ref[i]));
  log(`cross-arm max loss diff = ${maxDiff.toExponential(2)} (all arms same math)`);
  process.exit(0);
}

void main();
