/**
 * WITNESS-TIME HARVEST GATE (task #98 phase 4, docs/step-object-design.md §4).
 *
 * The failing-first gate for the #97 STOP: the checkpoint-recompute
 * `contiguous[512,768]` forward activation is PRUNED from the generated harvest
 * (its backward/recompute reader resolves LOWERED, invisible to `observeConsumed`
 * — `graphHeld=false`, and at the forward-plan build seam the backward consumer
 * does not exist yet, `stage4 §Task #97`). The recorded build masks the prune by
 * re-harvesting it (build-WITH-execution); with the recorded build's harvest
 * role removed the prune becomes a deterministic `Input not ready` throw.
 *
 * WITNESS-TIME HARVEST resolves it: the step-tape recorder observes the LOWERED
 * cross-plan read at end-of-step time (AFTER backward + recompute ran), and once
 * a producer template is witnessed with a stable read set across two consecutive
 * steps (K_w=2 per producer) publishes it to observed-liveness, which keeps it
 * in the pruned harvest. §4.1.
 *
 * THIS GATE (the §4.4 differential + failing-first oracle) runs the exact #97
 * config — distil@512 dims + selective checkpointing + step-tape recording — and
 * asserts:
 *
 *   1. WITNESS COVERAGE: the witnessed harvest set is non-empty for the producer
 *      templates whose cross-plan activations the generated harvest would prune
 *      (the shadow-parity signal). PRE-MECHANISM this set is empty (no witness
 *      recorder) → the assertion FAILS. POST-MECHANISM it is populated → PASS.
 *
 *   2. NO `Input not ready`: zero occurrences across the run (the #97 stage-3
 *      negative assertion). A throw here is the exact STOP symptom.
 *
 * PASS ⇒ exit 0. Any witness-empty / `Input not ready` ⇒ exit 1.
 *
 * Random-init (no model files needed). This is the CHECKPOINT cell of the phase-4
 * config matrix; medium@512 and 124M chunked-sum are separate invocations of
 * t-witness-harvest-matrix.ts.
 *
 * Run (solo GPU):
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim \
 *     TORCHLETTE_STEP_TAPE=record npx tsx tools/t-witness-harvest-probe.ts
 * Env: STEPS(=12) NUM_LAYERS(=6) NUM_HEADS(=12) EMBED_DIM(=768) SEQ_LEN(=512)
 *      BATCH_SIZE(=1)
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import {
  getWitnessProducerKeepSets,
  STEP_TAPE_RECORD,
  stStats,
} from "../src/core/step-tape";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim/index.ts";

const STEPS = parseInt(process.env.STEPS ?? "12", 10);
const L = parseInt(process.env.NUM_LAYERS ?? "6", 10);
const H = parseInt(process.env.NUM_HEADS ?? "12", 10);
const E = parseInt(process.env.EMBED_DIM ?? "768", 10);
const SEQ = parseInt(process.env.SEQ_LEN ?? "512", 10);
const BATCH = parseInt(process.env.BATCH_SIZE ?? "1", 10);
const VOCAB = 50257;
const log = (m: string) => console.error(`[witness-harvest] ${m}`);

function randInput() {
  const inp = new Int32Array(BATCH * SEQ);
  const tgt = new Int32Array(BATCH * SEQ);
  for (let i = 0; i < BATCH * SEQ; i++) {
    inp[i] = Math.floor(Math.random() * VOCAB);
    tgt[i] = Math.floor(Math.random() * VOCAB);
  }
  return { inp, tgt };
}

async function main() {
  if (!STEP_TAPE_RECORD) {
    log("FAIL: set TORCHLETTE_STEP_TAPE=record (flag read at module load)");
    process.exit(1);
  }
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "8000";

  // Intercept the `Input not ready` throw as a loud FAIL signal (it surfaces as
  // a thrown Error caught by the loop below, but also as a warn on some paths).
  let inputNotReady = 0;
  const origErr = console.error.bind(console);
  console.error = (...a: unknown[]) => {
    if (a.map(String).join(" ").includes("Input not ready")) inputNotReady++;
    origErr(...a);
  };

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

  log(
    `START steps=${STEPS} L=${L} H=${H} E=${E} seq=${SEQ} batch=${BATCH} ` +
      `checkpoint=selective TAPE=${process.env.TORCHLETTE_STEP_TAPE}`,
  );

  let threw = false;
  for (let step = 0; step < STEPS; step++) {
    const { inp, tgt } = randInput();
    try {
      await api.beginStep();
      const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
        device: "webgpu",
      });
      const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
        device: "webgpu",
      });
      const loss = api.tidy(() => {
        const l = model.forwardWithLoss(input, target, {
          useCheckpoint: true,
          selectiveCheckpoint: true,
        }).loss;
        api.keep(l);
        return l;
      });
      const lossVal = await loss.item();
      await loss.backward();
      opt.step();
      opt.zeroGrad();
      input.dispose();
      target.dispose();
      api.endStep();
      await api.markStep();
      if (step < 4 || step === STEPS - 1)
        log(`step ${step}: loss=${lossVal.toFixed(4)}`);
      if (!Number.isFinite(lossVal)) {
        log(`FATAL non-finite loss at step ${step}`);
        threw = true;
        break;
      }
    } catch (e) {
      threw = true;
      log(
        `THREW at step ${step}: ${e instanceof Error ? e.message : String(e)}`,
      );
      if (e instanceof Error && e.message.includes("Input not ready"))
        inputNotReady++;
      break;
    }
  }

  const st = stStats();
  const witnessed = getWitnessProducerKeepSets();
  const witnessedTemplates = witnessed.size;
  let witnessedPairs = 0;
  for (const s of witnessed.values()) witnessedPairs += s.length;

  log("=== VERDICT ===");
  log(
    `eligiblePairs=${st.eligiblePairs} structureMisses=${st.structureMisses} loweredPairs=${st.loweredPairs}`,
  );
  log(
    `witnessedTemplates=${witnessedTemplates} witnessedPairs=${witnessedPairs} witnessVariances=${st.witnessVariances}`,
  );
  log(`inputNotReady=${inputNotReady} threw=${threw}`);

  // The two acceptance oracles.
  let fail = false;
  if (witnessedTemplates === 0 || witnessedPairs === 0) {
    log(
      "RESULT: FAIL — witnessed harvest set is EMPTY. The witness mechanism did " +
        "not publish any producer's cross-plan reads (the pre-mechanism state). " +
        "The checkpoint-recompute activation is unprotected → the generated prune " +
        "would drop it (#97 STOP).",
    );
    fail = true;
  }
  if (inputNotReady > 0 || threw) {
    log(
      `RESULT: FAIL — ${inputNotReady} 'Input not ready' / threw=${threw} (the #97 stage-3 twin fired)`,
    );
    fail = true;
  }
  if (!fail) {
    log(
      `RESULT: PASS — witnessed ${witnessedPairs} pairs across ${witnessedTemplates} ` +
        `producer templates; zero 'Input not ready'; ${STEPS} checkpoint steps clean.`,
    );
  }

  console.error = origErr;
  destroyWebGPU();
  process.exit(fail ? 1 : 0);
}

main().catch((e) => {
  console.error(`[witness-harvest] FATAL: ${e?.stack ?? e}`);
  process.exit(1);
});
