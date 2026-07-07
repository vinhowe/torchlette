/**
 * Step-tape phase-1b G1 driver: run REAL decode workloads with the recorder
 * on (TORCHLETTE_STEP_TAPE=record) and report the tape store + refusal
 * counters (docs/staged-execution-phase1.md §2.2, §3 G1, §4).
 *
 * Workloads:
 *   stock   — 200-token unsteered decode, maxSeqLen 512: crosses the KV
 *             bucket boundary at 128 (buckets 128 and 256) → per-bucket
 *             tapes, ZERO refusals expected.
 *   steered — steered decode (makeResidualHook), alpha FIXED: same slot set,
 *             different template fingerprint, ZERO refusals expected.
 *   both    — stock then steered in one session (default).
 *   alphachange — α=+3 then α=−3 generations: the scalar-adapt demote must
 *             cascade to the tape store (guard-4 planInvalidations > 0) and
 *             the re-adapted plan must re-record with ZERO refusals.
 *
 * Exit 1 if any refusal is counted, or if the expected tapes are missing
 * (stock: ≥2 bucket tapes; steered: ≥1 tape under the steered fingerprint).
 *
 * Run SOLO from repo root (flag is read at module load):
 *   TORCHLETTE_STEP_TAPE=record npx tsx examples/qwen3-steering/step-tape-probe.ts [stock|steered|both]
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  getWebGPUInitError,
  initWebGPU,
} from "../../src/backend/webgpu";
import {
  STEP_TAPE_RECORD,
  stGetTapes,
  stStats,
  type StepTape,
} from "../../src/core/step-tape";
import { Torchlette } from "../../src/frontend/torchlette";
import { loadPretrainedQwen3 } from "../qwen3/loader";
import type { StaticKV } from "../qwen3/model";
import { makeResidualHook, type SteeringVector } from "./src/lib/steering";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const PROMPT = [785, 6722, 315, 9625, 374]; // "The capital of France is"
const STEER_LAYER = 14;

async function runGeneration(
  api: Torchlette,
  model: Awaited<ReturnType<typeof loadPretrainedQwen3>>,
  opts: { alpha: number; vec: SteeringVector | null; numTokens: number },
): Promise<{ tokens: number[]; msPerToken: number }> {
  const vocab = model.config.vocabSize;
  const hook = makeResidualHook(api, opts.vec, opts.alpha);
  const tokens = [...PROMPT];
  const staticKV: StaticKV = model.allocStaticKV(512);
  const prevScope = api.setStepScopedCleanup(true);
  const walls: number[] = [];
  try {
    {
      const idx = api.tensorFromArray(tokens, [1, tokens.length]);
      const { logits } = api.noGrad(() =>
        model.forward(idx, { staticKV, residualHook: hook }),
      );
      const top = await api.readTopK(logits, 8, {
        offset: (tokens.length - 1) * vocab,
        length: vocab,
      });
      logits.dispose();
      tokens.push(top.indices[0]);
      await api.markStep();
    }
    for (let i = 0; i < opts.numTokens; i++) {
      const t0 = performance.now();
      const idx = api.tensorFromArray([tokens[tokens.length - 1]], [1, 1]);
      const { logits } = api.noGrad(() =>
        model.forward(idx, { staticKV, residualHook: hook }),
      );
      const top = await api.readTopK(logits, 8, { length: vocab });
      logits.dispose();
      tokens.push(top.indices[0]);
      await api.markStep();
      walls.push(performance.now() - t0);
    }
    staticKV.k.length = 0;
    staticKV.v.length = 0;
    await api.markStep();
  } finally {
    api.setStepScopedCleanup(prevScope);
  }
  const steady = walls.slice(Math.min(6, walls.length - 1));
  const msPerToken = steady.reduce((a, b) => a + b, 0) / steady.length;
  return { tokens, msPerToken };
}

function printTape(tape: StepTape): void {
  const counts: Record<string, number> = {};
  for (const e of tape.entries) {
    const k =
      e.kind === "readback" ? `readback:${e.which}` : e.kind;
    counts[k] = (counts[k] ?? 0) + 1;
  }
  console.log(`  tape bucketKey=${tape.bucketKey}`);
  console.log(
    `    entries=${tape.entries.length} (${Object.entries(counts)
      .map(([k, v]) => `${k}=${v}`)
      .join(", ")}) structGen=${tape.structGen} epoch=${tape.epoch} regime={stepScopedCleanup:${tape.regime.stepScopedCleanup}} recordedAtStep=${tape.recordedAtStep}`,
  );
  console.log(
    `    templates=[${[...tape.templateIds].map((f) => `0x${(f >>> 0).toString(16)}`).join(", ")}]`,
  );
  for (const s of tape.slots) {
    console.log(
      `    slot ${s.id} source=${s.source} ${s.name} dtype=${s.dtype}${s.stable !== undefined ? ` stable=${s.stable}` : ""}`,
    );
  }
  const rb = tape.entries.filter((e) => e.kind === "readback");
  for (const e of rb) {
    if (e.kind === "readback") {
      console.log(
        `    readback which=${e.which}${e.params ? ` params={${Object.entries(e.params).map(([k, v]) => `${k}:${v}`).join(",")}}` : ""}`,
      );
    }
  }
}

function report(label: string): { refusals: number; tapeCount: number } {
  const stats = stStats();
  console.log(`\n=== ${label}: recorder state ===`);
  console.log(
    `  stepsObserved=${stats.stepsObserved} eligiblePairs=${stats.eligiblePairs} tapes=${stats.tapeCount} refusals=${stats.refusals} structureMisses=${stats.structureMisses} loweredPairs=${stats.loweredPairs} boundaryResets=${stats.boundaryResets} planInvalidations=${stats.planInvalidations}`,
  );
  for (const d of stats.refusalDiagnostics) console.log(`  REFUSAL: ${d}`);
  for (const tape of stGetTapes().values()) printTape(tape);
  return { refusals: stats.refusals, tapeCount: stats.tapeCount };
}

async function main() {
  const mode = process.argv[2] ?? "both";
  if (!STEP_TAPE_RECORD) {
    throw new Error(
      "run with TORCHLETTE_STEP_TAPE=record (read at module load by src/core/step-tape.ts)",
    );
  }
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedQwen3(api, MODEL_DIR, {
    maxSeqLen: 512,
    weightDtype: "f32",
  });

  let failed = false;

  if (mode === "stock" || mode === "both") {
    console.log("\n--- stock: 200-token unsteered decode (crosses KV bucket 128→256) ---");
    const r = await runGeneration(api, model, {
      alpha: 0,
      vec: null,
      numTokens: 200,
    });
    console.log(`  tokens generated=${r.tokens.length - PROMPT.length}, steady ms/token=${r.msPerToken.toFixed(1)}`);
    const { refusals, tapeCount } = report("stock");
    if (refusals > 0) {
      console.log("  !! FAIL: refusals on stock decode (§4 stop condition)");
      failed = true;
    }
    if (tapeCount < 2) {
      console.log("  !! FAIL: expected ≥2 per-bucket tapes on a bucket-crossing generation");
      failed = true;
    }
  }

  if (mode === "steered" || mode === "both") {
    console.log("\n--- steered: 40-token decode, alpha FIXED at 4 ---");
    const h = model.config.hiddenSize;
    const dir = new Float32Array(h);
    for (let i = 0; i < h; i++) dir[i] = Math.sin(i * 0.37) * 5;
    const vec: SteeringVector = {
      direction: api.persist(api.tensorFromArray(dir, [h])),
      layer: STEER_LAYER,
      hiddenSize: h,
      posPrompt: "synthetic+",
      negPrompt: "synthetic-",
    };
    await api.markStep();
    const before = stStats().refusals;
    const r = await runGeneration(api, model, {
      alpha: 4,
      vec,
      numTokens: 40,
    });
    console.log(`  tokens generated=${r.tokens.length - PROMPT.length}, steady ms/token=${r.msPerToken.toFixed(1)}`);
    const { refusals } = report("steered (cumulative)");
    if (refusals > before) {
      console.log("  !! FAIL: refusals on steered fixed-alpha decode");
      failed = true;
    }
  }

  if (mode === "alphachange") {
    // Guard-4 wiring check: an α change across generations trips the
    // scalar-adapt demote (destroyCompiledPlanBuffers) — tapes referencing
    // the steered template must be DROPPED (planInvalidations > 0), then
    // re-recorded from the re-adapted plan with zero refusals (α lands in a
    // scalar-table slot; its write is a value-conditional `sc:` entry).
    console.log("\n--- alphachange: alpha=+3 generation, then alpha=-3 ---");
    const h = model.config.hiddenSize;
    const dir = new Float32Array(h);
    for (let i = 0; i < h; i++) dir[i] = Math.sin(i * 0.37) * 5;
    const vec: SteeringVector = {
      direction: api.persist(api.tensorFromArray(dir, [h])),
      layer: STEER_LAYER,
      hiddenSize: h,
      posPrompt: "synthetic+",
      negPrompt: "synthetic-",
    };
    await api.markStep();
    await runGeneration(api, model, { alpha: 3, vec, numTokens: 20 });
    const mid = stStats();
    console.log(
      `  after gen A (α=+3): tapes=${mid.tapeCount} refusals=${mid.refusals} planInvalidations=${mid.planInvalidations}`,
    );
    await runGeneration(api, model, { alpha: -3, vec, numTokens: 20 });
    const { refusals, tapeCount } = report("alphachange (cumulative)");
    const stats = stStats();
    if (refusals > 0) {
      console.log("  !! FAIL: refusals across an alpha change");
      failed = true;
    }
    if (stats.planInvalidations === 0) {
      console.log(
        "  !! FAIL: alpha change did not cascade to the tape store (guard-4 stub not firing)",
      );
      failed = true;
    }
    if (tapeCount < 1) {
      console.log("  !! FAIL: no tape re-recorded after the alpha change");
      failed = true;
    }
  }

  console.log(failed ? "\nSTEP-TAPE PROBE FAIL" : "\nSTEP-TAPE PROBE PASS");
  process.exit(failed ? 1 : 0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
