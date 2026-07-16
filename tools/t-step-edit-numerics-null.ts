/**
 * S3 FUSE LIVE WIRING — the NUMERICS NULL gate for structural edits.
 *
 * A partition merge changes the step's ISLAND BOUNDARIES, never its math: the
 * merged island's interior is the SAME dispatch sequence (islands-design §2.3 —
 * an island may lower to N dispatches; the merge changes only the grouping/
 * identity, not the lowered plan). So an EDITED trajectory must match the
 * UNEDITED one to ≤1e-5 over 20+ steps — the null differential for a structural
 * edit (the CLAUDE.md "differentially test optimized paths" discipline applied to
 * the editor's move layer).
 *
 * Two trajectories, identical pretrained init + identical data:
 *   A — no edit (control).
 *   B — after WARMUP steps, an accepted merge is applied to a live plan; the
 *       remaining steps run under the merged partition.
 * Per-step losses must agree; a divergence is a merge that changed math (a bug),
 * never benign.
 *
 * Run (device 3):
 *   VULKAN_DEVICE_INDEX=3 LD_LIBRARY_PATH=tools/vk-shim \
 *   TORCHLETTE_STEP_TAPE=record npx tsx tools/t-step-edit-numerics-null.ts
 */
import * as path from "node:path";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { stDeriveStepObjects, stResetAll } from "../src/core/step-tape";
import {
  applyPartitionMerge,
  getCachedPlanPartition,
  getEditedPartitionCount,
  rollbackPartitionMerge,
} from "../src/executor/executor";
import { Torchlette } from "../src/frontend/torchlette";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { Adam, GradScaler } from "../src/optim/index.ts";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "128", 10);
const BATCH = parseInt(process.env.BATCH ?? "1", 10);
const WARMUP = parseInt(process.env.WARMUP ?? "6", 10);
const NUM = parseInt(process.env.NUM ?? "24", 10);
const TOL = parseFloat(process.env.TOL ?? "1e-5");
const log = (m: string) => console.error(`[t-step-edit-null-numerics] ${m}`);

async function runTrajectory(edit: boolean): Promise<number[]> {
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
  const params = model.parameters();
  const opt = new Adam(
    params,
    { lr: 1e-4, weightDecay: 0.01, adamW: true },
    api,
  );
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;

  const inp = new Int32Array(BATCH * SEQ);
  const tgt = new Int32Array(BATCH * SEQ);
  // Deterministic data stream — IDENTICAL across both trajectories.
  let s = 12345;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s;
  };

  const losses: number[] = [];
  let editedFp = 0;
  for (let step = 0; step < WARMUP + NUM; step++) {
    await scaler.resolveDeferred();
    for (let i = 0; i < BATCH * SEQ; i++) {
      inp[i] = rnd() % V;
      tgt[i] = rnd() % V;
    }
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
      device: "webgpu",
    });
    const loss = api.tidy(() => {
      const l = api.autocast(
        () =>
          model.forwardWithLoss(input, target, { useCheckpoint: true }).loss,
      );
      api.keep(l);
      return l;
    });
    const lv = await loss.item();
    if (step >= WARMUP) losses.push(lv);
    const scaled = scaler.scale(loss);
    await scaled.backward();
    scaler.unscale_(opt);
    clipGradNorm_(api, params, 1.0);
    scaler.step(opt);
    scaler.update();
    opt.zeroGrad();
    scaled.dispose();
    loss.dispose();
    input.dispose();
    target.dispose();

    // Apply the merge exactly at the WARMUP boundary (the edit takes effect on
    // the next step — "the next executed steps re-witness"). Chosen against the
    // first plan with ≥2 islands, the same target the binding probe uses.
    if (edit && step === WARMUP - 1) {
      const objs = stDeriveStepObjects();
      for (const o of objs) {
        for (const pl of o.declaration.partition.plans) {
          const pp = getCachedPlanPartition(pl.fp);
          if (pp && pp.islands.length >= 2) {
            applyPartitionMerge(pl.fp, 0, 1);
            editedFp = pl.fp;
            break;
          }
        }
        if (editedFp) break;
      }
      log(
        `applied merge(0,1) on plan 0x${editedFp.toString(16)}; edits=${getEditedPartitionCount()}`,
      );
    }
  }
  if (edit) rollbackPartitionMerge(editedFp);
  return losses;
}

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  const NOEDIT = process.env.NOEDIT === "1"; // determinism control: run B also no-edit
  log(`control trajectory (no edit), ${NUM} post-warmup steps…`);
  const control = await runTrajectory(false);
  stResetAll();
  log(
    NOEDIT
      ? "SECOND control (determinism check)…"
      : `edited trajectory (merge at step ${WARMUP})…`,
  );
  const edited = await runTrajectory(!NOEDIT);

  let maxAbs = 0;
  let maxRel = 0;
  const rows: string[] = [];
  for (let i = 0; i < Math.min(control.length, edited.length); i++) {
    const a = control[i];
    const b = edited[i];
    const abs = Math.abs(a - b);
    const rel = abs / (Math.abs(a) + 1e-9);
    maxAbs = Math.max(maxAbs, abs);
    maxRel = Math.max(maxRel, rel);
    if (i < 6 || abs > TOL)
      rows.push(
        `  step ${i}: control=${a.toFixed(7)} edited=${b.toFixed(7)} |Δ|=${abs.toExponential(2)}`,
      );
  }
  console.log("=== STEP-EDIT-NUMERICS-NULL ===");
  for (const r of rows) console.log(r);
  console.log(
    JSON.stringify(
      { steps: control.length, maxAbs, maxRel, tol: TOL },
      null,
      2,
    ),
  );
  await destroyWebGPU();
  const pass = control.length === edited.length && maxAbs <= TOL;
  console.log(
    pass
      ? `PASS: edited trajectory == control ≤${TOL} over ${control.length} steps (structural edit is math-null)`
      : `FAIL: edited trajectory diverged from control (maxAbs=${maxAbs.toExponential(3)} > ${TOL})`,
  );
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  log(`FATAL ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
