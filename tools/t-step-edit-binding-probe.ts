/**
 * Step-object PHASE 6 EDITOR-BINDING PROBE (task #98, docs/step-object-design.md
 * §5.3 / §6 Phase 6). The admission-pressure proof that the P3 EDITOR SOCKET is
 * REAL: it opens the `StepEditChannel` against a LIVE recording engine (a real
 * GPT-2 training step that witnesses tapes), binds the channel to the reified
 * `StepPartition` facet of a witnessed StepObject, issues a REFUSED merge + a
 * ROLLBACK, and reads the TYPED refusals — the exact interaction the P3 editor
 * performs, proving the seam the editor binds to exists and behaves.
 *
 * This is NOT the editor UI (§5.3: "the editor itself is out of scope"). It is
 * the server-side binding proof — code proves itself against a live engine before
 * the UI binds it (the house admission-pressure rule).
 *
 * What it proves:
 *  (1) LIVE BIND: a witnessed StepObject's partition facet is a real per-plan
 *      partition (fps + boundaryHashes) — the channel binds to it, not a mock.
 *  (2) REFUSED MERGE: an illegal merge (islands with no convex island-flow edge)
 *      returns a typed MERGE_REFUSED — loud, not a throw, not a silent no-op.
 *  (3) ROLLBACK IS IDENTITY: a legal merge request rolled back leaves the channel
 *      pending list EMPTY (§5.2 — a rolled-back edit is the identity).
 *  (4) RESERVED PAUSE: pauseAtBoundary returns the typed NOT_IMPLEMENTED shape.
 *  (5) S3 FUSE LIVE WIRING: an ACCEPTED merge, bound to the executor via
 *      `applyMerge → applyPartitionMerge`, drives the detector's own partition
 *      merge — the plan re-fingerprints, the tape re-witnesses under the merged
 *      partition (island count −1), and a rollback restores the original. This is
 *      the first STRUCTURAL edit executed end-to-end against the live engine.
 *
 * Run (device 3 — another agent may own others; SIGTERM only):
 *   VULKAN_DEVICE_INDEX=3 LD_LIBRARY_PATH=tools/vk-shim \
 *   TORCHLETTE_STEP_TAPE=record npx tsx tools/t-step-edit-binding-probe.ts
 * Env: MODEL(=distilgpt2) SEQ_LEN(=256) BATCH(=1) STEPS(=16)
 */
import * as path from "node:path";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { mergeIslands } from "../src/compiler/fusion-detect";
import { stepObjectDigest } from "../src/core/step-object";
import {
  STEP_TAPE_RECORD,
  stDeriveStepObjects,
  stResetAll,
  stStats,
} from "../src/core/step-tape";
import {
  applyPartitionMerge,
  getCachedPlanPartition,
  getEditedPartitionCount,
  getEditedPlanPartition,
  rollbackPartitionMerge,
} from "../src/executor/executor";
import { Torchlette } from "../src/frontend/torchlette";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { Adam, GradScaler } from "../src/optim/index.ts";
import type { IslandId } from "../src/schedule/moves/fuse";
import { makeStepEditChannel } from "../src/schedule/moves/step-edit-channel";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "256", 10);
const BATCH = parseInt(process.env.BATCH ?? "1", 10);
const STEPS = parseInt(process.env.STEPS ?? "16", 10);
const log = (m: string) => console.error(`[t-step-edit-binding] ${m}`);

async function main() {
  if (!STEP_TAPE_RECORD) {
    log("FAIL: set TORCHLETTE_STEP_TAPE=record");
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
  let s = 999;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s;
  };

  // ── One training step (a live recording engine) ────────────────────────────
  const trainSteps = async (n: number) => {
    for (let step = 0; step < n; step++) {
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
      await loss.item();
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
    }
  };

  // ── Run until a tape is witnessed ──────────────────────────────────────────
  await trainSteps(STEPS);

  // ── Bind the channel to a live witnessed partition facet ───────────────────
  const objs = stDeriveStepObjects();
  if (objs.length === 0) {
    log("FAIL: no witnessed StepObject to bind against");
    await destroyWebGPU();
    process.exit(1);
  }
  const obj = objs[0];
  const partition = obj.declaration.partition;
  log(
    `bound to StepObject ${stepObjectDigest(obj)}: ${partition.plans.length} plans, ` +
      `partitionDigest=${partition.boundaryDigest}, device=${partition.device}`,
  );

  // The P3 editor's interaction: open the channel bound to the live partition.
  // With NO island-flow provided (the conservative default), any non-adjacent
  // merge is refused — the loud refusal the editor surfaces as "illegal move".
  const ch = makeStepEditChannel({ partition });

  // (2) REFUSED MERGE — two non-adjacent islands (islands 0 and 5), no convex
  // island-flow edge → typed MERGE_REFUSED.
  const refused = ch.requestMerge(
    "0" as unknown as IslandId,
    "5" as unknown as IslandId,
  );
  const refusedOk =
    refused.kind === "refused" && refused.code === "MERGE_REFUSED";
  log(
    `refused merge(0,5): ${refused.kind}` +
      (refused.kind === "refused" ? ` code=${refused.code}` : ""),
  );

  // (3) ROLLBACK IS IDENTITY — a legal (adjacent) merge, rolled back → empty.
  const legal = ch.requestMerge(
    "2" as unknown as IslandId,
    "3" as unknown as IslandId,
  );
  const acceptedOk = legal.kind === "accepted" && ch.pending.length === 1;
  if (legal.kind === "accepted") ch.rollback(legal.handle);
  const rollbackOk = ch.pending.length === 0;
  log(
    `legal merge(2,3): ${legal.kind}; after rollback pending=${ch.pending.length}`,
  );

  // (4) RESERVED PAUSE — typed NOT_IMPLEMENTED, zero behavioral surface.
  const pause = ch.pauseAtBoundary(partition.plans[0]?.fp ?? 1);
  const pauseOk = pause.kind === "not-implemented" && ch.pending.length === 0;

  // ════════════════════════════════════════════════════════════════════════════
  // (5) S3 FUSE LIVE WIRING — the first STRUCTURAL edit executed end-to-end:
  //     an accepted merge drives a REAL partition merge on the live executor,
  //     re-fingerprints, re-witnesses, and runs under the merged partition;
  //     rollback restores the original partition + digest.
  // ════════════════════════════════════════════════════════════════════════════
  // Find a target plan with ≥2 islands (a mergeable partition). The plan's fp is
  // its DEFAULT (token-free) execution fingerprint — the executor's registry key.
  let targetFp = 0;
  let targetDefault: ReturnType<typeof getCachedPlanPartition> | undefined;
  for (const pl of partition.plans) {
    const pp = getCachedPlanPartition(pl.fp);
    if (pp && pp.islands.length >= 2) {
      targetFp = pl.fp;
      targetDefault = pp;
      break;
    }
  }
  let liveMergeOk = false;
  let reWitnessOk = false;
  let islandCountDropOk = false;
  let rollbackDigestOk = false;
  let numericsNote = "";
  if (!targetDefault) {
    log("S3: no plan with ≥2 islands to merge — cannot exercise live wiring");
  } else {
    const defaultIslands = targetDefault.islands.length;
    const defaultToken = targetDefault.boundaryHash >>> 0;
    // The detector's own merge of islands (0,1) — the I1 agreement token the
    // executor will mix into the fingerprint. Computed here ONLY to assert the
    // witnessed partition matches (single source: the executor calls the same fn).
    const expected = mergeIslands(targetDefault, 0, 1);
    const expectedToken = expected.ok
      ? expected.partition.boundaryHash >>> 0
      : 0;

    // Bind the channel LIVE: applyMerge drives the executor's partition merge.
    const chLive = makeStepEditChannel({
      partition,
      applyMerge: (a, b) => applyPartitionMerge(targetFp, Number(a), Number(b)),
    });
    const accept = chLive.requestMerge(
      "0" as unknown as IslandId,
      "1" as unknown as IslandId,
    );
    liveMergeOk = accept.kind === "accepted" && getEditedPartitionCount() === 1;
    log(
      `S3 live merge(0,1) on plan 0x${targetFp.toString(16)} ` +
        `(${defaultIslands} islands): ${accept.kind}; edits=${getEditedPartitionCount()}`,
    );

    // Re-witness under the merged partition: clear tapes, run enough steps for
    // the bootstrap (1) + the re-witness (K_w=2 consecutive) to fire.
    stResetAll();
    await trainSteps(8);
    const editedPart = getEditedPlanPartition(targetFp);
    islandCountDropOk =
      !!editedPart && editedPart.islands.length === defaultIslands - 1;
    const objsEdited = stDeriveStepObjects();
    // The merged partition WITNESSED: some step's partition projection carries
    // the merged token for the target plan (proves the merged boundary ran).
    const witnessedMerged = objsEdited.some((o) =>
      o.declaration.partition.plans.some(
        (p) => p.boundaryHash >>> 0 === expectedToken,
      ),
    );
    const witnessedDefaultGone = !objsEdited.some((o) =>
      o.declaration.partition.plans.some(
        (p) =>
          p.fp >>> 0 === targetFp >>> 0 &&
          p.boundaryHash >>> 0 === defaultToken,
      ),
    );
    reWitnessOk = witnessedMerged && witnessedDefaultGone;
    log(
      `S3 re-witness: editedIslands=${editedPart?.islands.length ?? "-"} ` +
        `(default ${defaultIslands}), merged-token witnessed=${witnessedMerged}, ` +
        `default-token gone=${witnessedDefaultGone}`,
    );

    // ROLLBACK — restore the default partition; re-witness back to identity.
    rollbackPartitionMerge(targetFp);
    const rolledBackToZero = getEditedPartitionCount() === 0;
    stResetAll();
    await trainSteps(8);
    const objsRolled = stDeriveStepObjects();
    const witnessedDefaultBack = objsRolled.some((o) =>
      o.declaration.partition.plans.some(
        (p) =>
          p.fp >>> 0 === targetFp >>> 0 &&
          p.boundaryHash >>> 0 === defaultToken,
      ),
    );
    // The default template is never mutated: its reified partition is byte-
    // identical before edit and after rollback (digest identity at this altitude).
    const defaultUnchanged =
      (getCachedPlanPartition(targetFp)?.boundaryHash ?? -1) >>> 0 ===
      defaultToken;
    rollbackDigestOk =
      rolledBackToZero && witnessedDefaultBack && defaultUnchanged;
    log(
      `S3 rollback: edits=${getEditedPartitionCount()}, default-token re-witnessed=` +
        `${witnessedDefaultBack}, default partition unchanged=${defaultUnchanged}`,
    );
    numericsNote =
      "merged interior = same dispatches (islands-design §2.3) ⇒ trajectory null; " +
      "see t-step-edit-numerics-null probe for the ≤1e-5/20-step gate";
  }

  const st = stStats();
  const recorderOk = st.eligiblePairs > 0 && st.refusals === 0;

  console.log("=== STEP-EDIT-BINDING-STATS ===");
  console.log(
    JSON.stringify(
      {
        boundDigest: stepObjectDigest(obj),
        plans: partition.plans.length,
        partitionDigest: partition.boundaryDigest,
        eligiblePairs: st.eligiblePairs,
        refusals: st.refusals,
        s3: {
          targetFp: `0x${targetFp.toString(16)}`,
          liveMergeOk,
          reWitnessOk,
          islandCountDropOk,
          rollbackDigestOk,
          note: numericsNote,
        },
        checks: { refusedOk, acceptedOk, rollbackOk, pauseOk, recorderOk },
      },
      null,
      2,
    ),
  );

  const pass =
    refusedOk &&
    acceptedOk &&
    rollbackOk &&
    pauseOk &&
    recorderOk &&
    liveMergeOk &&
    reWitnessOk &&
    islandCountDropOk &&
    rollbackDigestOk;
  console.log(
    pass
      ? "PASS: S3 FUSE LIVE — an accepted merge drove a REAL partition merge, re-witnessed under the merged partition, and rolled back to identity"
      : "FAIL: the S3 live wiring did not behave end-to-end",
  );
  await destroyWebGPU();
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  log(`FATAL ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
