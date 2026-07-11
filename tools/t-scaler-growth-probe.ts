/**
 * Scale-GROWTH-under-capture probe (inc-3 item-2 retirement proof).
 *
 * A GradScaler with a SMALL growthInterval grows its scale mid-run. The question
 * this probe answers: does a scale CHANGE cause a step-tape MISS (a re-record),
 * or does it flow through as DATA (the tape keeps HITTING)?
 *
 *  - If scale is baked as a graph SCALAR (mul(loss, jsNumber)), a growth event
 *    changes that scalar. It is covered by the scalar-table (re-dressed per
 *    replay) IFF the mul(loss, scale) node's scalar ref lands in the plan's
 *    scalar slots. If NOT covered, growth thrashes the template → miss.
 *  - After item-2 (scale as a persistent TENSOR read live), growth is pure DATA:
 *    the scale tensor's buffer is read by the replay, no scalar changes, NO miss.
 *
 * PASS: hits > 0 AND the growth crossing produces NO new refusal/miss (the tape
 * stays hitting across the growth event) AND the captured trajectory matches an
 * uncaptured control to the noise floor.
 *
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim \
 *   TORCHLETTE_STEP_TAPE=1 npx tsx tools/t-scaler-growth-probe.ts
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler } from "../src/optim/index.ts";
import { STEP_TAPE_REPLAY } from "../src/core/step-tape";
import type { Tensor } from "../src/frontend/tensor";
import { assertReferenceLossNonzero } from "./parity-sanity";

const STEPS = parseInt(process.env.STEPS ?? "24", 10);
const GROWTH_INTERVAL = parseInt(process.env.GROWTH_INTERVAL ?? "4", 10);
const log = (m: string) => console.error(`[scaler-growth] ${m}`);

async function run(useCapture: boolean) {
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
  });
  const w = api.registerState(
    api.tensorFromArray([0.5, -0.5, 0.25, 0.75], [2, 2], { requiresGrad: true }),
  );
  const target = api.registerState(api.tensorFromArray([1, 0], [1, 2]));
  const opt = new Adam([w], { lr: 1e-2 }, api);
  // Small growthInterval so scale grows several times during the run (no inf →
  // every GROWTH_INTERVAL steps the scale doubles).
  const scaler = new GradScaler(api, {
    initScale: 4.0,
    growthInterval: GROWTH_INTERVAL,
    growthFactor: 2.0,
  });
  let bodyRuns = 0;

  const stepFn = async (x: Tensor): Promise<Tensor> => {
    bodyRuns++;
    const pred = api.matmul(x, w);
    const diff = api.sub(pred, target);
    const loss = api.mean(api.mul(diff, diff));
    const lossOut = api.noGrad(() => api.mul(loss, 1));
    const scaled = scaler.scale(loss);
    await scaled.backward();
    scaler.unscale_(opt);
    scaler.step(opt);
    await scaler.update();
    opt.zeroGrad();
    scaled.dispose();
    return lossOut;
  };
  const step = useCapture ? api.capture(stepFn, { training: true }) : null;

  const losses: number[] = [];
  const scales: number[] = [];
  const prev = api.setStepScopedCleanup(true);
  try {
    for (let i = 0; i < STEPS; i++) {
      await scaler.resolveDeferred();
      const x = api.tensorFromArray([1 + i * 0.01, 2 - i * 0.01], [1, 2]);
      const loss = step ? ((await step(x)) as Tensor) : await stepFn(x);
      losses.push((await api.cpu(loss))[0]);
      scales.push(scaler.getScale());
      loss.dispose();
      x.dispose();
      await api.markStep();
    }
  } finally {
    api.setStepScopedCleanup(prev);
  }
  const replay = api.getStepTapeStats().replay;
  return {
    losses,
    scales,
    bodyRuns,
    hits: step?.stats().hits ?? 0,
    calls: step?.stats().calls ?? 0,
    refusals:
      replay.missScalar + replay.missShape + replay.missValidity + replay.missEpoch,
    missShape: replay.missShape,
    missScalar: replay.missScalar,
  };
}

async function main() {
  if (!STEP_TAPE_REPLAY) {
    log("FAIL: set TORCHLETTE_STEP_TAPE=1");
    process.exit(1);
  }
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  const ctl = await run(false);
  assertReferenceLossNonzero(ctl.losses[0], "scaler-growth/control");
  log(`control scales: ${ctl.scales.join(", ")}`);
  const cap = await run(true);
  log(`captured scales: ${cap.scales.join(", ")}`);
  log(
    `captured: calls=${cap.calls} hits=${cap.hits} bodyRuns=${cap.bodyRuns} refusals=${cap.refusals} missShape=${cap.missShape} missScalar=${cap.missScalar}`,
  );

  let maxD = 0;
  for (let i = 0; i < STEPS; i++)
    maxD = Math.max(maxD, Math.abs(ctl.losses[i] - cap.losses[i]));

  // DESIRED (post item-2): the captured scale grows in lockstep with the
  // control (scale is a persistent tensor advanced in-plan; growth is DATA).
  const capGrew = cap.scales[cap.scales.length - 1] > cap.scales[0];
  const tracksControl =
    Math.abs(cap.scales[cap.scales.length - 1] - ctl.scales[ctl.scales.length - 1]) < 1e-9;
  const pass =
    cap.hits > 0 && cap.refusals === 0 && capGrew && tracksControl && maxD < 2.5e-3;
  log(
    `captured scale ${cap.scales[0]}→${cap.scales[cap.scales.length - 1]} ` +
      `vs control ${ctl.scales[0]}→${ctl.scales[ctl.scales.length - 1]}; ` +
      `hits=${cap.hits} refusals=${cap.refusals} maxLossΔ=${maxD.toExponential(2)}`,
  );
  if (pass) {
    log("PASS: scale growth flows through hits as DATA (item-2 landed).");
  } else if (!capGrew || !tracksControl) {
    log(
      "FAIL (EXPECTED ON MAIN — item-2 not landed): the CPU scale mirror FREEZES " +
        "on tape hits (the body — hence scaler.update()'s JS growth — never runs), " +
        "and mul(loss, jsScale) bakes the FROZEN scale into the replayed graph → " +
        "the captured trajectory diverges from control. This is the frozen-scalar " +
        "class item-2 retires (scale/invScale as persistent tensors, growth in-plan). " +
        "This tool is the RETIREMENT GATE — it goes green when item-2 lands.",
    );
  } else {
    log(`FAIL: hits=${cap.hits} refusals=${cap.refusals} maxLossΔ=${maxD.toExponential(2)}`);
  }
  destroyWebGPU();
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
