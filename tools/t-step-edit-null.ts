/**
 * Step-object PHASE 6 NULL-EDIT GATE (task #98, docs/step-object-design.md §6
 * Phase 6 / §5). Proves the partition FACET + the StepEditChannel are null-clean:
 * reifying the per-step partition onto the StepObject and routing an IDENTITY
 * edit (a refused merge + a rollback) through the channel changes NOTHING.
 *
 *  (1) PARTITION PROJECTION (the I1 agreement seam at step altitude, §3.3): for
 *      EVERY witnessed tape, the reified `StepPartition.plans` reproduces the
 *      per-plan islands boundaryHashes the recorder observed byte-identically
 *      (`stepPartitionReproducesPerPlan`) — the detector OWNS membership; the
 *      facet is a pure read-only projection (no second owner).
 *  (2) PARTITION-DIGEST DETERMINISM: the step-level `boundaryDigest` re-derives
 *      byte-identical from the SAME recorder state twice, and is STEADY-STABLE
 *      post-warmup (a static graph's partition hashes identically every step).
 *  (3) IDENTITY EDIT IS NULL: an illegal merge REQUEST (no island-flow → refused)
 *      + a rollback of a legal-but-then-rolled-back edit leaves the channel
 *      pending list EMPTY — nothing mutated, the declaration untouched (§5.2).
 *  (4) STREAM / DIGEST UNTOUCHED: the recorder still forms an eligible tape with
 *      ZERO refusals AND the step-object DIGEST (bucketKey) is byte-stable — the
 *      partition facet + channel did not perturb identity (they hash into the
 *      SAME bucketKey; the partition token is already in each plan's fp via I1,
 *      so nothing new enters the digest).
 *
 * The command-stream byte-identity vs the pre-branch baseline rides the existing
 * t-train-tape-probe / t-step-object-null (unchanged numbers) + parity-fullstack.
 *
 * Run (device 11 ONLY — another agent owns device 10):
 *   VULKAN_DEVICE_INDEX=11 LD_LIBRARY_PATH=tools/vk-shim \
 *   TORCHLETTE_STEP_TAPE=record npx tsx tools/t-step-edit-null.ts
 * Env: MODEL(=distilgpt2) SEQ_LEN(=256) BATCH(=1) STEPS(=18)
 */
import * as path from "node:path";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import {
  stepObjectDigest,
  stepPartitionDigest,
  stepPartitionReproducesPerPlan,
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
import type { IslandId } from "../src/schedule/moves/fuse";
import { makeStepEditChannel } from "../src/schedule/moves/step-edit-channel";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "256", 10);
const BATCH = parseInt(process.env.BATCH ?? "1", 10);
const STEPS = parseInt(process.env.STEPS ?? "18", 10);
const log = (m: string) => console.error(`[t-step-edit-null] ${m}`);

/** Snapshot the (digest, partitionDigest) pairs of the witnessed StepObjects. */
function snapshot(): Array<[string, number]> {
  return stDeriveStepObjects()
    .map(
      (o) =>
        [stepObjectDigest(o), o.declaration.partition.boundaryDigest] as [
          string,
          number,
        ],
    )
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] - b[1]));
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

  let steadyPartDigest: number | null = null;
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

    if (step >= 13 && stStats().tapeCount === 1) {
      const objs = stDeriveStepObjects();
      const d = objs[0]?.declaration.partition.boundaryDigest ?? 0;
      if (steadyPartDigest === null) steadyPartDigest = d;
      else if (d !== steadyPartDigest) {
        steadyStable = false;
        log(`STEADY PARTITION DRIFT step ${step}: ${steadyPartDigest} -> ${d}`);
      }
    }
    if (step % 4 === 0 || step === STEPS - 1) {
      log(`step ${step}: loss=${lossVal.toFixed(4)}`);
    }
  }

  // ── The null-edit assertions ───────────────────────────────────────────────
  const st = stStats();
  const objs = stDeriveStepObjects();
  const tapes = stGetTapes();

  // (1) PARTITION PROJECTION reproduces the per-plan boundaryHashes byte-identical.
  let projectionOk = true;
  for (const obj of objs) {
    const tape = tapes.get(stepObjectDigest(obj));
    const perPlan = tape?.partitionHashes ?? [];
    if (!stepPartitionReproducesPerPlan(obj.declaration.partition, perPlan)) {
      projectionOk = false;
      log(
        `PARTITION PROJECTION MISMATCH: digest=${stepObjectDigest(obj)} plans=${obj.declaration.partition.plans.length} perPlan=${perPlan.length}`,
      );
    }
    // And the step-level digest is a pure function of the pairs.
    if (
      obj.declaration.partition.boundaryDigest !==
      stepPartitionDigest(obj.declaration.partition.plans)
    ) {
      projectionOk = false;
      log(`PARTITION DIGEST not pure for ${stepObjectDigest(obj)}`);
    }
  }

  // (2) DETERMINISM: derive-twice-from-one-state → identical (digest, partDigest).
  const a = snapshot();
  const b = snapshot();
  const determinismOk =
    a.length === b.length &&
    a.every((p, i) => p[0] === b[i][0] && p[1] === b[i][1]);

  // (3) IDENTITY EDIT IS NULL: an illegal merge (no flow) is refused; a legal
  // merge rolled back returns the channel to empty. Nothing mutates.
  const ch = makeStepEditChannel();
  const refused = ch.requestMerge(
    "0" as unknown as IslandId,
    "9" as unknown as IslandId,
  );
  const legal = ch.requestMerge(
    "3" as unknown as IslandId,
    "4" as unknown as IslandId,
  ); // adjacent → convex default
  if (legal.kind === "accepted") ch.rollback(legal.handle);
  const pause = ch.pauseAtBoundary(0x1);
  const identityEditNull =
    refused.kind === "refused" &&
    ch.pending.length === 0 && // legal merge rolled back, refused never queued
    pause.kind === "not-implemented";

  // The digests are UNCHANGED after routing edits (the channel is the write side
  // of a read-only projection — a refused/rolled-back edit touches no facet).
  const afterEdit = snapshot();
  const digestUntouched =
    afterEdit.length === a.length &&
    afterEdit.every((p, i) => p[0] === a[i][0] && p[1] === a[i][1]);

  // (4) STREAM UNTOUCHED: eligible tape formed, zero refusals.
  const recorderOk = st.eligiblePairs > 0 && st.refusals === 0;

  console.log("=== STEP-EDIT-NULL-STATS ===");
  console.log(
    JSON.stringify(
      {
        tapeCount: st.tapeCount,
        stepObjects: objs.length,
        eligiblePairs: st.eligiblePairs,
        refusals: st.refusals,
        partitionDigests: a.map((p) => p[1]),
        steadyPartDigest,
        checks: {
          projectionOk,
          determinismOk,
          identityEditNull,
          digestUntouched,
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
    identityEditNull &&
    digestUntouched &&
    recorderOk &&
    steadyStable &&
    steadyPartDigest !== null;
  console.log(
    pass
      ? "PASS: partition facet + StepEditChannel are null-clean — identity edit changes nothing"
      : "FAIL: the partition facet / edit channel perturbed the step object",
  );
  await destroyWebGPU();
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  log(`FATAL ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
