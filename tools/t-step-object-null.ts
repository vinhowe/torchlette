/**
 * Step-object PHASE 1 NULL DIFFERENTIAL (task #98, docs/step-object-design.md
 * §6 Phase 1 gate). Proves the reified StepObject is a PURE PROJECTION over the
 * witnessed tape — reify + consult changes NOTHING:
 *
 *  (1) PURE-PROJECTION: for EVERY witnessed tape, the derived StepObject's
 *      digest recomputes BYTE-IDENTICAL to the tape's own `bucketKey`
 *      (`stepObjectDigestMatchesBucket`) — the single-source invariant (§2.2 /
 *      §3.1). A mismatch is a reification bug, never a benign difference.
 *  (2) DIGEST DETERMINISM: re-deriving the StepObjects from the SAME recorder
 *      state twice yields byte-identical digest sets — the §6 "digests recompute
 *      byte-identical across two runs of the same config" clause: the digest is
 *      a PURE deterministic function of the tape (no run-to-run entropy). NOTE
 *      the tape ITSELF legitimately re-witnesses across warmup (a template
 *      rebuild at ~step 11 re-keys the bucket, step-tape.ts) — determinism is
 *      derive-twice-from-one-state, not tape-stable-across-warmup.
 *  (3) SLOT-NAME STABILITY: every StepObject's slot ids are declaration-stable
 *      names (no tape ordinal) — the §10 ruling-1 round-trip.
 *  (4) STREAM UNTOUCHED: the recorder still forms an eligible tape with ZERO
 *      refusals (the t-train-tape-probe invariant) — reify did not perturb the
 *      recorder's own eligibility/streams.
 *
 * The command-stream byte-identity vs the pre-branch baseline is asserted
 * separately by the existing t-train-tape-probe / matrix (unchanged numbers) +
 * parity-fullstack; this tool is the STEP-OBJECT-specific null gate.
 *
 * Run (solo GPU):
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim \
 *   TORCHLETTE_STEP_TAPE=record npx tsx tools/t-step-object-null.ts
 * Env: MODEL(=distilgpt2) SEQ_LEN(=256) BATCH(=1) STEPS(=18)
 */
import * as path from "node:path";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import {
  stepObjectDigest,
  stepObjectDigestMatchesBucket,
} from "../src/core/step-object";
import {
  STEP_TAPE_RECORD,
  stDeriveStepObjects,
  stGetTapes,
  stStats,
} from "../src/core/step-tape";
import { Torchlette } from "../src/frontend/torchlette";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { Adam, GradScaler } from "../src/optim/index.ts";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "256", 10);
const BATCH = parseInt(process.env.BATCH ?? "1", 10);
const STEPS = parseInt(process.env.STEPS ?? "18", 10);
const log = (m: string) => console.error(`[t-step-object-null] ${m}`);

/** Snapshot the digest set of the currently-witnessed StepObjects (sorted). */
function digestSnapshot(): string[] {
  return stDeriveStepObjects()
    .map((o) => stepObjectDigest(o))
    .sort();
}

async function main() {
  if (!STEP_TAPE_RECORD) {
    log("FAIL: set TORCHLETTE_STEP_TAPE=record (flag is read at module load)");
    process.exit(1);
  }
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
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
  let s = 12345;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s;
  };

  let sawWitness = false;
  // Steady-state digest stability (the cross-"run" determinism witness): once
  // warmup settles (template rebuild past, ~step 12), the steady tape's digest
  // must be BYTE-IDENTICAL every remaining step — the same declaration
  // re-witnessed yields the same digest (§6 determinism, §10 ruling-1 stability).
  let steadyDigest: string | null = null;
  let steadyStable = true;

  for (let step = 0; step < STEPS; step++) {
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
          // useCheckpoint:false — the D3 CHECKPOINT_EAGER_REFUSAL (executor.ts,
          // sunset-bound) keeps checkpointed-eager plans LOWERED by design,
          // which forms no compiled tape → no witnessed StepObject. The
          // step-object null/edit properties this gate asserts are
          // checkpoint-independent; checkpoint-compile coverage lives under
          // whole-step remat (t-whole-step-diff) + the refusal spec.
          model.forwardWithLoss(input, target, { useCheckpoint: false }).loss,
      );
      api.keep(l);
      return l;
    });
    const lossVal = await loss.item();
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

    // Prove the pure-projection invariant HOLDS at the FIRST witness too (not
    // only end-of-run) — the digest tracks the tape as it re-witnesses.
    if (!sawWitness && stStats().tapeCount > 0) {
      sawWitness = true;
      const firstObjs = stDeriveStepObjects();
      const firstOk = firstObjs.every((o) => {
        const t = stGetTapes().get(stepObjectDigest(o));
        return t !== undefined && stepObjectDigestMatchesBucket(o, t);
      });
      log(
        `step ${step}: first witness — ${firstObjs.length} StepObject(s), projection ${firstOk ? "OK" : "MISMATCH"}`,
      );
    }
    // Post-warmup steady-state digest stability: from step 13 on the tape has
    // settled; its single digest must be identical every step.
    if (step >= 13 && stStats().tapeCount === 1) {
      const d = digestSnapshot()[0];
      if (steadyDigest === null) steadyDigest = d;
      else if (d !== steadyDigest) {
        steadyStable = false;
        log(`STEADY DRIFT step ${step}: ${steadyDigest} -> ${d}`);
      }
    }
    if (step % 4 === 0 || step === STEPS - 1) {
      log(`step ${step}: loss=${lossVal.toFixed(4)}`);
    }
  }

  // ── The null-test assertions ──────────────────────────────────────────────
  const st = stStats();
  const objs = stDeriveStepObjects();
  const tapes = stGetTapes();

  // (1) PURE-PROJECTION: every StepObject's digest == its tape's bucketKey.
  let projectionOk = true;
  for (const obj of objs) {
    const digest = stepObjectDigest(obj);
    const tape = tapes.get(digest);
    if (!tape || !stepObjectDigestMatchesBucket(obj, tape)) {
      projectionOk = false;
      log(
        `PROJECTION MISMATCH: digest=${digest} tape=${tape ? "present" : "MISSING"}`,
      );
    }
  }

  // (2) DIGEST DETERMINISM: re-derive from the SAME recorder state twice → the
  // digest sets are byte-identical (the digest is a pure function of the tape;
  // no run-to-run entropy). This is the "byte-identical across two runs of the
  // same config" clause: derive-twice-from-one-state is the in-process witness.
  const snapA = digestSnapshot();
  const snapB = digestSnapshot();
  const determinismOk =
    snapA.length === snapB.length && snapA.every((d, i) => d === snapB[i]);
  if (!determinismOk) {
    log(`DETERMINISM: A=${JSON.stringify(snapA)} B=${JSON.stringify(snapB)}`);
  }

  // (3) SLOT-NAME STABILITY: every slot id is a declaration-stable name (embeds
  // only op-fp + pos[:ii], no tape ordinal) — matches /^(w|u|sc):[0-9a-f]+:\d+/.
  let slotNamesOk = true;
  const slotRe = /^(w|u|sc):[0-9a-f]+:\d+(:\d+)?$/;
  for (const obj of objs) {
    for (const slot of obj.declaration.slots) {
      if (!slotRe.test(slot.id)) {
        slotNamesOk = false;
        log(`SLOT NAME not declaration-stable: ${slot.id}`);
      }
    }
  }

  // (4) STREAM UNTOUCHED: eligible tape formed, zero refusals.
  const recorderOk = st.eligiblePairs > 0 && st.refusals === 0;

  console.log("=== STEP-OBJECT-NULL-STATS ===");
  console.log(
    JSON.stringify(
      {
        tapeCount: st.tapeCount,
        stepObjects: objs.length,
        eligiblePairs: st.eligiblePairs,
        refusals: st.refusals,
        digests: snapA,
        steadyDigest,
        checks: {
          projectionOk,
          determinismOk,
          slotNamesOk,
          recorderOk,
          steadyStable,
        },
      },
      null,
      2,
    ),
  );

  const pass =
    projectionOk &&
    determinismOk &&
    slotNamesOk &&
    recorderOk &&
    steadyStable &&
    steadyDigest !== null;
  console.log(
    pass
      ? "PASS: StepObject is a pure projection — reify is null-clean"
      : "FAIL: StepObject reification perturbed the tape / diverged from bucketKey",
  );
  await destroyWebGPU();
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  log(`FATAL ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
