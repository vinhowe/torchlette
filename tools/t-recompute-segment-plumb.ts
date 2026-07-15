/**
 * RECOMPUTE-SEGMENT PLUMBING PROBE — task #99 phase R1
 * (docs/arena-recompute-design.md §5 Phase R1).
 *
 * R1 plumbs the declared RecomputeSegment data (isCheckpointBoundary node
 * positions → checkpoint-recompute result stamps) to the memory-planner INPUT
 * SURFACE as DATA — visible, logged, asserted present — while changing ZERO
 * planning decisions (the stamped entries stay whole-step RESULTs; R2 splits
 * their liveness).
 *
 * This probe validates BOTH sides of the R1 finding:
 *
 *   WIRING (synthetic boundary): flag a REAL forward-activation pending node with
 *   isCheckpointBoundary BEFORE it is forced, then confirm the compiled plan that
 *   produces it stamps `_recomputeSegments` mapping (nodeIndex,oi,slot)→entryIdx.
 *   Proves the plumbing is LIVE end-to-end on a real plan + real planner entry.
 *
 *   REAL-PATH FINDING (the inert declared source): with the model's own selective
 *   checkpointing, markAsCheckpointBoundary NEVER fires — the recomputed tensor is
 *   consumed by backward in the SAME force, so its lazyRef.kind is "materialized"
 *   (the pending node is gone) by the time the boundary block runs. So the
 *   declared `isCheckpointBoundary`/`hasRecompute`/`recomputeFps` substrate is
 *   INERT on the real path: 0 stamps, and correctly so (not a plumbing bug). This
 *   is the design finding R1 surfaces (see docs/arena-recompute-design.md R1
 *   report). R2 must source the recompute boundary from the LIVE observed-liveness
 *   witness signal, not this flag.
 *
 * === MEASURED THIS COMMIT (A100 dw-2-1 device 10) ===
 *   WIRING (synthetic boundary flagged on the real loss node):
 *     template 0xbd0dd584 hasRecompute=true segments=1 (ni=222,oi=0,slot=254,entry=103)
 *     recompute-bearing-templates=1, stamped=1, mappedToPlannerEntry=1 → PASS.
 *   REAL-PATH (model selective checkpointing, no synthetic flag):
 *     recompute-bearing-templates=0, stampedSegments=0, flaggedAny=false — the
 *     inert declared source (markAsCheckpointBoundary never fires;
 *     lastCapturedTensor.lazyRef.kind=materialized). PASS (FINDING).
 *
 * Run (solo GPU):
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim TORCHLETTE_STEP_TAPE=record \
 *     npx tsx tools/t-recompute-segment-plumb.ts            # MODE=wiring (default)
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim TORCHLETTE_STEP_TAPE=record \
 *     MODE=realpath npx tsx tools/t-recompute-segment-plumb.ts   # the inert-source finding
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { debugRecomputeSegments } from "../src/executor/executor";
import type { Tensor } from "../src/frontend/tensor";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim/index.ts";

const L = 6;
const H = 12;
const E = 768;
const SEQ = 512;
const STEPS = 6;
const VOCAB = 50257;
const BATCH = 1;
const MODE = process.env.MODE ?? "wiring"; // wiring | realpath
const log = (m: string) => console.error(`[recompute-plumb:${MODE}] ${m}`);

function randInput(seq: number) {
  const inp = new Int32Array(BATCH * seq);
  const tgt = new Int32Array(BATCH * seq);
  for (let i = 0; i < BATCH * seq; i++) {
    inp[i] = Math.floor(Math.random() * VOCAB);
    tgt[i] = Math.floor(Math.random() * VOCAB);
  }
  return { inp, tgt };
}

/** [WIRING mode] Flag a REAL pending activation node as a checkpoint boundary so
 *  the plan that produces it carries the declared boundary at build. Returns
 *  whether a pending node was found+flagged. */
function flagAsBoundary(t: Tensor): boolean {
  const ref = (
    t as unknown as {
      _runtimeTensor(): {
        lazyRef?: { kind?: string; node?: { isCheckpointBoundary?: boolean } };
      };
    }
  )._runtimeTensor().lazyRef;
  if (ref?.kind === "pending" && ref.node) {
    ref.node.isCheckpointBoundary = true;
    return true;
  }
  return false;
}

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "10000";

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  const { GPT2 } = await import("../examples/gpt2/model.ts");
  const model = new GPT2(
    api,
    {
      vocabSize: VOCAB,
      blockSize: 1024,
      numLayers: L,
      numHeads: H,
      embedDim: E,
      dropoutRate: 0,
    },
    { device: "webgpu" },
  );

  await api.beginStep();
  api.endStep();
  await api.markStep();

  model.train(true);
  const params = model.parameters();
  const opt = new Adam(
    params,
    { lr: 5e-4, weightDecay: 0.01, adamW: true },
    api,
  );

  log(`START L=${L} H=${H} E=${E} seq=${SEQ} steps=${STEPS} mode=${MODE}`);

  const losses: number[] = [];
  let flaggedAny = false;
  for (let step = 0; step < STEPS; step++) {
    const { inp, tgt } = randInput(SEQ);
    await api.beginStep();
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
      device: "webgpu",
    });
    // realpath: use the model's own selective checkpointing (declares recompute
    // through the real markAsCheckpointBoundary path — which is inert, the
    // finding). wiring: no model checkpointing; instead flag a real activation
    // node synthetically so the declared boundary reaches the compiled build.
    const useCkpt = MODE === "realpath";
    const loss = api.tidy(() => {
      const l = model.forwardWithLoss(input, target, {
        useCheckpoint: useCkpt,
        selectiveCheckpoint: useCkpt,
      }).loss;
      // WIRING: flag the loss's producing pending node as a checkpoint boundary
      // before it is forced — a real plan-result node, so the compiled plan for
      // that plan stamps it as a recompute segment.
      if (MODE === "wiring") {
        if (flagAsBoundary(l)) flaggedAny = true;
      }
      api.keep(l);
      return l;
    });
    const lossVal = await loss.item();
    losses.push(lossVal);
    await loss.backward();
    opt.step();
    opt.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
    if (step < 2 || step === STEPS - 1)
      log(`step ${step}: loss=${lossVal.toFixed(4)}`);
  }

  const segs = debugRecomputeSegments();
  const templatesWithRecompute = Object.keys(segs).length;
  let totalStamped = 0;
  let mappedToEntry = 0;
  for (const [fp, s] of Object.entries(segs)) {
    totalStamped += s.segments.length;
    for (const seg of s.segments)
      if (seg.entryIdx !== undefined) mappedToEntry++;
    log(
      `template ${fp}: hasRecompute=${s.hasRecompute} segments=${s.segments.length} ` +
        s.segments
          .map(
            (x) =>
              `(ni=${x.nodeIndex},oi=${x.outputIndex},slot=${x.slot},entry=${x.entryIdx})`,
          )
          .join(" "),
    );
  }

  log("=== VERDICT ===");
  log(
    `mode=${MODE} flaggedAny=${flaggedAny} recompute-bearing-templates=${templatesWithRecompute} ` +
      `stampedSegments=${totalStamped} mappedToPlannerEntry=${mappedToEntry} allFinite=${losses.every((l) => Number.isFinite(l))}`,
  );

  let fail = false;
  if (!losses.every((l) => Number.isFinite(l))) {
    log("FAIL: non-finite loss");
    fail = true;
  }
  if (MODE === "wiring") {
    // The plumbing must carry a synthetically-flagged real boundary to a stamped,
    // planner-entry-mapped segment. (flaggedAny may be false on some steps if the
    // loss node is already materialized; the compiled plan builds on 2nd+ exec, so
    // over STEPS the stamp lands on at least one recurring template.)
    if (!flaggedAny) {
      log(
        "WARN: no pending loss node found to flag — the wiring proof needs a pending plan-result node",
      );
    }
    if (templatesWithRecompute === 0 || totalStamped === 0) {
      log(
        "FAIL: WIRING — flagged a real boundary but NO recompute segment stamped (plumbing dead)",
      );
      fail = true;
    }
    if (totalStamped !== mappedToEntry) {
      log(
        `FAIL: WIRING — ${totalStamped - mappedToEntry} stamped segment(s) not mapped to a planner entry (identity seam violation)`,
      );
      fail = true;
    }
    if (!fail)
      log(
        `RESULT: PASS — WIRING live: ${totalStamped} segment(s) across ${templatesWithRecompute} template(s), all planner-entry-mapped.`,
      );
  } else {
    // realpath: the declared source is INERT — 0 stamps is the CORRECT, expected
    // result (the finding). Any stamp here would be a surprise (would mean the
    // real markAsCheckpointBoundary path started firing).
    if (totalStamped === 0) {
      log(
        "RESULT: PASS (FINDING) — real selective-checkpointing declared source is INERT " +
          "(markAsCheckpointBoundary never fires; lazyRef.kind=materialized). 0 stamps, correctly. " +
          "R2 must use the observed-liveness witness signal, not this flag.",
      );
    } else {
      log(
        `RESULT: PASS (UNEXPECTED) — the real path stamped ${totalStamped} segment(s); the inert-source finding no longer holds — re-verify R2 assumptions.`,
      );
    }
  }

  process.stdout.write(
    `${JSON.stringify({
      recomputePlumb: true,
      mode: MODE,
      flaggedAny,
      recomputeBearingTemplates: templatesWithRecompute,
      stampedSegments: totalStamped,
      mappedToPlannerEntry: mappedToEntry,
      pass: !fail,
    })}\n`,
  );
  destroyWebGPU();
  process.exit(fail ? 1 : 0);
}

main().catch((e) => {
  console.error(`[recompute-plumb] FATAL: ${e?.stack ?? e}`);
  process.exit(1);
});
