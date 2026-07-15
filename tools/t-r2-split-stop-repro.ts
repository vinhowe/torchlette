/**
 * R2 SPLIT — DETERMINISTIC STOP REPRO (task #99 phase R2).
 *
 * The mandated R2 mechanism (docs/arena-recompute-design.md §3 Candidate B):
 * SPLIT a checkpointed activation's RESULT interval at the recompute boundary,
 * sourcing the boundary from the LIVE witness signal (getWitnessedHarvest — the
 * checkpoint-recompute cross-plan reads observeConsumed is blind to). This tool
 * runs that EXACT mechanism (TORCHLETTE_R2_SPLIT_PROBE=1 demotes every witnessed
 * RESULT slot to a packable temp — the interval split) on the distil@512 +
 * SELECTIVE-checkpointing config and demonstrates it is UNSOUND, with hard GPU
 * evidence.
 *
 * WHY IT STOPS (the read-wiring finding, R2 report):
 *   Selective checkpointing SAVES attention activations (read DIRECTLY by
 *   backward — 47 cross-plan reads 0x19e72088←0xbd0dd584, shapes [1,512,768] /
 *   [1,12,512,64] / [512,50257], 39 graph-held) and recomputes ONLY the MLP
 *   (cheaply, as backward-INTERNAL temps that never become a stamped cross-plan
 *   producer). So the witnessed producers name GENUINE saved-for-backward
 *   activations, NOT recompute-fed dead values. "Witnessed lowered read" does
 *   NOT imply "recompute-fed / dead-after-producer". Demoting those RESULT slots
 *   strands the backward read on a packed/overlaid buffer → the demoted buffer
 *   is bound read-write-and-read in one sync scope → Dawn DROPS the submit →
 *   downstream reads see stale data (silent training corruption).
 *
 * The genuinely dead-between-spans value (the MLP recompute) is ALREADY a
 * backward-internal temp and is NOT among the pinned RESULT entries, so the
 * witness-sourced interval split has nothing sound to pack: the 1919 MB pin is
 * the saved-for-backward working set + gradients, which the arena-free lowered
 * path frees via WITHIN-plan action-indexed liveness (executor.ts) + pool
 * return between steps, not via a cross-plan interval split.
 *
 * ASSERTION: with the split probe ON, the run raises WebGPU uncaptured errors
 * (submit-drop / read-write aliasing) — the deterministic UAF the [lifetime] /
 * GPU-error guard catches. Exit 0 = STOP reproduced (corruption observed as
 * predicted). Exit 1 = the corruption did NOT reproduce (the finding no longer
 * holds — re-verify before attempting the split).
 *
 * Run (solo GPU):
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim TORCHLETTE_STEP_TAPE=record \
 *     TORCHLETTE_R2_SPLIT_PROBE=1 TORCHLETTE_STRICT_GPU=0 \
 *     npx tsx tools/t-r2-split-stop-repro.ts
 */
import {
  destroyWebGPU,
  getGpuUncapturedErrorCount,
  initWebGPU,
} from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim/index.ts";

const L = 6,
  H = 12,
  E = 768,
  SEQ = 512,
  STEPS = 12,
  VOCAB = 50257,
  BATCH = 1;
const log = (m: string) => console.error(`[r2-stop-repro] ${m}`);

function randInput(seq: number) {
  const inp = new Int32Array(BATCH * seq);
  const tgt = new Int32Array(BATCH * seq);
  for (let i = 0; i < BATCH * seq; i++) {
    inp[i] = Math.floor(Math.random() * VOCAB);
    tgt[i] = Math.floor(Math.random() * VOCAB);
  }
  return { inp, tgt };
}

async function main() {
  if (process.env.TORCHLETTE_R2_SPLIT_PROBE !== "1") {
    log(
      "NOTE: set TORCHLETTE_R2_SPLIT_PROBE=1 to exercise the mandated split (this tool asserts it corrupts).",
    );
  }
  if (!(await initWebGPU())) process.exit(1);
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "10000";
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  const { GPT2 } = await import("../examples/gpt2/model.ts");
  const model = new GPT2(
    api,
    { vocabSize: VOCAB, blockSize: 1024, numLayers: L, numHeads: H, embedDim: E, dropoutRate: 0 },
    { device: "webgpu" },
  );
  await api.beginStep();
  api.endStep();
  await api.markStep();
  model.train(true);
  const opt = new Adam(model.parameters(), { lr: 5e-4, weightDecay: 0.01, adamW: true }, api);
  const losses: number[] = [];
  for (let step = 0; step < STEPS; step++) {
    const { inp, tgt } = randInput(SEQ);
    await api.beginStep();
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], { device: "webgpu" });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], { device: "webgpu" });
    const loss = api.tidy(() => {
      const l = model.forwardWithLoss(input, target, { useCheckpoint: true, selectiveCheckpoint: true }).loss;
      api.keep(l);
      return l;
    });
    losses.push(await loss.item());
    await loss.backward();
    opt.step();
    opt.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
  }

  const gpuErrors = getGpuUncapturedErrorCount();
  const probeOn = process.env.TORCHLETTE_R2_SPLIT_PROBE === "1";
  log(`losses: ${losses.map((l) => l.toFixed(2)).join(" ")}`);
  log(`gpuUncapturedErrors=${gpuErrors} splitProbe=${probeOn}`);

  const corruptionObserved = gpuErrors > 0;
  process.stdout.write(
    `${JSON.stringify({
      r2StopRepro: true,
      splitProbe: probeOn,
      gpuUncapturedErrors: gpuErrors,
      corruptionObserved,
      firstLoss: +losses[0].toFixed(2),
      lastLoss: +losses[losses.length - 1].toFixed(2),
    })}\n`,
  );

  if (probeOn) {
    if (corruptionObserved) {
      log(
        `STOP REPRODUCED: the witness-sourced RESULT split raised ${gpuErrors} GPU uncaptured error(s) ` +
          `(submit-drop / read-write aliasing) — the witnessed pairs are GENUINE saved-for-backward ` +
          `activations, not recompute-fed dead values. The mandated R2 mechanism is UNSOUND on selective ` +
          `checkpointing. See docs/arena-recompute-design.md §R2 STOP.`,
      );
      destroyWebGPU();
      process.exit(0);
    }
    log(
      `UNEXPECTED: split probe ON but NO GPU errors — the STOP finding no longer reproduces. Re-verify the ` +
        `read-wiring (a witnessed producer may now be genuinely recompute-fed) BEFORE attempting the split.`,
    );
    destroyWebGPU();
    process.exit(1);
  }
  log("split probe OFF — baseline run (no corruption expected). Set the probe to reproduce the STOP.");
  destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error(`[r2-stop-repro] FATAL: ${e?.stack ?? e}`);
  process.exit(1);
});
