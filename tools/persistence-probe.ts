/**
 * Persistence-contract probe: tensors CREATED MID-STEP and held across steps.
 *
 * The engine infers persistence at beginStep (snapshotForStep): tensors alive
 * at the snapshot are persistent; storage created during the step is
 * step-scoped and demoted at markStep. The CONTRACT users rely on (and the
 * per-param optimizer pattern assumes) is: storage referenced by a live
 * RuntimeTensor is never destroyed or recycled, regardless of when it was
 * created. This probe tests exactly that, with no optimizer code involved:
 *
 *   per step, for each of NSTATES state tensors:
 *     mNew = m * 0.9 + g * 0.1     (g = per-state constant)
 *     m.dispose(); m = mNew         (replace, hold across markStep)
 *
 * Expected per-state value after step t: sum over k<=t of 0.9^(t-k) * 0.1 * g
 * (closed form: g * 0.1 * (0.9^t-ish geometric)). Computed in JS below.
 * Reports the first step+state where GPU diverges from JS.
 *
 * Env: NSTATES (default 2), STEPS (default 4), SIZE (default 8),
 *      FUSION=0, COMPILED=0 — bisect which engine layer breaks the contract.
 */

import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

const NSTATES = parseInt(process.env.NSTATES ?? "2", 10);
const STEPS = parseInt(process.env.STEPS ?? "4", 10);
const SIZE = parseInt(process.env.SIZE ?? "8", 10);
if (process.env.COMPILED === "0") process.env.TORCHLETTE_COMPILED_PLAN = "0";

async function main() {
  if (!(await initWebGPU())) {
    console.error("WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", {
    enableFusion: process.env.FUSION !== "0",
  });
  const runtime = api._runtime();

  // BACKWARD=1: drive the state updates from real autograd grads of
  // requiresGrad params (loss = sum((p - t)^2), dL/dp = 2(p - t)).
  // PARAMUPDATE=1: also update the params via copy_ (the optimizer shape).
  const useBackward = process.env.BACKWARD === "1";
  const useParamUpdate = process.env.PARAMUPDATE === "1";

  const params = Array.from({ length: NSTATES }, (_, i) =>
    api.tensorFromArray(
      Array.from({ length: SIZE }, (_, j) => ((i + 1) * 10 + j) / 100),
      [SIZE],
      { device: "webgpu", requiresGrad: true },
    ),
  );
  const targets = Array.from({ length: NSTATES }, () =>
    api.tensorFromArray(new Array(SIZE).fill(0), [SIZE], {
      device: "webgpu",
    }),
  );
  // Per-state driver constants for the no-backward mode.
  const gs = Array.from({ length: NSTATES }, (_, i) =>
    api.tensorFromArray(
      Array.from({ length: SIZE }, (_, j) => (i + 1) * 10 + j),
      [SIZE],
      { device: "webgpu" },
    ),
  );

  // States start at zero, created OUTSIDE any step (persistent by snapshot).
  let states = gs.map(() => runtime.zeros([SIZE], "webgpu"));

  // JS mirrors
  const jsP = params.map((_, i) =>
    Array.from({ length: SIZE }, (_, j) => ((i + 1) * 10 + j) / 100),
  );
  const js = gs.map(() => new Array(SIZE).fill(0));

  let firstBad: string | null = null;
  for (let step = 0; step < STEPS; step++) {
    await api.beginStep();
    if (useBackward) {
      let loss = null;
      for (let i = 0; i < NSTATES; i++) {
        const d = api.sub(params[i], targets[i]);
        const partial = api.sum(api.mul(d, d));
        loss = loss ? api.add(loss, partial) : partial;
      }
      await loss!.backward();
    }
    for (let i = 0; i < NSTATES; i++) {
      const gRt = useBackward
        ? params[i].grad!._unwrap()
        : gs[i]._unwrap();
      const mNew = runtime.add(
        runtime.mul(states[i], 0.9),
        runtime.mul(gRt, 0.1),
      );
      // Two modes, demonstrating the persistence contract:
      //   default      — REPLACEMENT anti-pattern: the new tensor is demoted
      //                  as a step temp at markStep (buffer pooled while
      //                  live). Trips the [lifetime] read guard; survives
      //                  only by allocation-order luck.
      //   PERSIST=1    — runtime.persist() adopts the mid-step tensor into
      //                  the step snapshot: the supported way to create
      //                  long-lived state inside a step. No warning.
      states[i].dispose();
      states[i] =
        process.env.PERSIST === "1" ? runtime.registerState(mNew) : mNew;
      if (useParamUpdate) {
        runtime.copy_(
          params[i]._unwrap(),
          runtime.sub(params[i]._unwrap(), runtime.mul(mNew, 0.01)),
        );
      }
      for (let j = 0; j < SIZE; j++) {
        const g = useBackward ? 2 * jsP[i][j] : (i + 1) * 10 + j;
        js[i][j] = js[i][j] * 0.9 + g * 0.1;
        if (useParamUpdate) jsP[i][j] -= 0.01 * js[i][j];
      }
    }
    if (useBackward) {
      for (const p of params) p.zeroGrad();
    }
    api.endStep();
    await api.markStep();

    // READBACK=1: read back every step (forces materialization — may itself
    // protect storage). Default: never read mid-training, like real
    // optimizer state; only verify at the end.
    if (process.env.READBACK === "1" || step === STEPS - 1) {
      for (let i = 0; i < NSTATES; i++) {
        const vals = Array.from(await runtime.cpu(states[i]));
        for (let j = 0; j < SIZE; j++) {
          if (Math.abs(vals[j] - js[i][j]) > 1e-4 && !firstBad) {
            firstBad = `step=${step} state=${i} elem=${j}: gpu=${vals[j]} js=${js[i][j].toFixed(5)} (gpu row: ${vals.slice(0, 4).map((v) => v.toFixed(4))})`;
          }
        }
      }
      if (firstBad) break;
    }
  }

  console.log(firstBad ? `CONTRACT VIOLATED: ${firstBad}` : "CONTRACT HELD");
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
