/**
 * [#84] Second-in-process-run determinism (the plain/uncaptured ORACLE path).
 *
 * Two `new Torchlette` engines trained in ONE process share module-global
 * singletons. Before the fix, the FIRST engine's residual storages (params /
 * optimizer m/v / lr tensors) AND its leftover pending Tensors (the strongly-
 * held `pendingTensorsByNodeId` registry) lingered until GC collected them — at
 * an unpredictable time DURING the NEXT engine's run — and their buffers were
 * then released into the SHARED buffer pool MID-STEP of that run (the
 * "released-to-pool mid-step" corruption class). Result: the second-and-later
 * run sporadically (GC-timing dependent) diverged onto a WRONG trajectory —
 * silent wrongness in the exact path every differential gate diffs against.
 *
 * The fix (RuntimeEngine-constructor instance-boundary reset): (1)
 * `storageTracker.disposeAllForNewEngine()` ORPHANS the previous engine's
 * residual storages so their buffers can never be re-pooled mid-step; (2)
 * `clearPendingTensorsForNewEngine()` drops the previous engine's carried-over
 * pending Tensors so the next engine's forceAllPending() never executes them.
 *
 * This gate pins BOTH the SEAM (the tracker is empty at each new engine's
 * construction) and the OBSERVABLE (compounding second/third runs stay on the
 * first run's trajectory, not a bimodal wrong one). It uses the SAME stack the
 * bug requires — autocast f16 + gradient checkpointing + GradScaler + grad-clip
 * + AdamW + a per-step CosineAnnealingLR — because a stripped-down tiny model
 * exercises different residue. The amplifier is COMPOUNDING: each additional run
 * leaves more residue; pre-fix, run 2 already diverges by ~0.5..1.8 nats.
 */
import * as path from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { isF16Supported } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { storageTracker } from "../src/graph/storage-tracker";
import { clipGradNorm_ } from "../src/nn";
import { Adam, CosineAnnealingLR, GradScaler } from "../src/optim";
import { canUseWebGPU } from "./helpers/webgpu";

const TIMEOUT = 180_000;
const ROOT = path.resolve(__dirname, "..");
const STEPS = 10;
const SEQ = 128;

/** One full training run on a FRESH engine (the repro stack); returns per-step
 *  losses. Also asserts the SEAM: the module-global storage tracker is empty of
 *  any PRIOR engine's residue at this engine's construction. */
async function trainOnce(tag: string): Promise<number[]> {
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  // Right after construction, before this run creates any tensor, the tracker
  // must hold NO storages — a prior run's residue must have been cleared. This
  // is the instance-boundary seam the fix pins.
  expect(
    storageTracker.stats().totalStorages,
    `[${tag}] storageTracker not cleared at new-engine construction — prior run's residue leaked in (#84)`,
  ).toBe(0);

  const model = await loadPretrainedGPT2(
    api,
    path.join(ROOT, "models", "distilgpt2"),
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
  const sched = new CosineAnnealingLR(opt, STEPS, 1e-5);
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;
  let seed = 1234;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed;
  };
  const inp = new Int32Array(SEQ);
  const tgt = new Int32Array(SEQ);
  const losses: number[] = [];
  for (let stp = 0; stp < STEPS; stp++) {
    await scaler.resolveDeferred();
    for (let i = 0; i < SEQ; i++) {
      inp[i] = rnd() % V;
      tgt[i] = rnd() % V;
    }
    const input = api.tensorFromArray(Array.from(inp), [1, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(Array.from(tgt), [1, SEQ], {
      device: "webgpu",
    });
    const l = api.tidy(() => {
      const ll = api.autocast(
        () =>
          model.forwardWithLoss(input, target, { useCheckpoint: true }).loss,
      );
      api.keep(ll);
      return ll;
    });
    const lossOut = api.noGrad(() => api.mul(l, 1));
    const scaled = scaler.scale(l);
    await scaled.backward();
    scaler.unscale_(opt);
    clipGradNorm_(api, params, 1.0);
    scaler.step(opt);
    scaler.update();
    opt.zeroGrad();
    scaled.dispose();
    losses.push(await lossOut.item());
    lossOut.dispose();
    input.dispose();
    target.dispose();
    sched.step();
  }
  return losses;
}

describe("[#84] second-in-process-run determinism (plain oracle path)", () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = await canUseWebGPU();
  });

  it(
    "compounding runs in one process stay on the first run's trajectory",
    async () => {
      if (!webgpu) return;
      // The stack uses autocast f16; skip on hardware without shader-f16.
      if (!isF16Supported()) return;
      const ref = await trainOnce("run0");
      const arms: number[][] = [];
      for (let r = 1; r <= 2; r++) arms.push(await trainOnce(`run${r}`));

      // Band: pre-fix run 2 diverged 0.5..1.8 nats from an early step; GPU fp
      // reduction-order nondeterminism is ~1e-3. 0.05 cleanly separates them.
      const BAND = 0.05;
      for (let a = 0; a < arms.length; a++) {
        let maxD = 0;
        let firstBad = -1;
        for (let i = 0; i < ref.length; i++) {
          const d = Math.abs(ref[i] - arms[a][i]);
          if (d > maxD) maxD = d;
          if (d > BAND && firstBad < 0) firstBad = i;
        }
        expect(
          firstBad,
          `run${a + 1} diverged from run0 at step ${firstBad} (maxΔ=${maxD.toFixed(4)} nats > ${BAND}); ` +
            `the second-in-process bimodality is back (#84)`,
        ).toBe(-1);
      }
    },
    TIMEOUT,
  );
});
