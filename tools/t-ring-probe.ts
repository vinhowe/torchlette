/**
 * inc-3 falsification probe: does the RUNAHEAD ring (deferred gen-scoped settle)
 * form a tape AND track the serial control at K=1 and K=2?
 *
 *   VULKAN_DEVICE_INDEX=2 LD_LIBRARY_PATH=tools/vk-shim \
 *   TORCHLETTE_STEP_TAPE=1 npx tsx tools/t-ring-probe.ts
 *
 * Small MSE-on-matmul step (no model load — fast, GPU-only). Prints, per arm,
 * the loss trajectory + capture hits + whether the body froze on hits.
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim/index.ts";
import { STEP_TAPE_REPLAY } from "../src/core/step-tape";
import type { Tensor } from "../src/frontend/tensor";
import { assertReferenceLossNonzero } from "./parity-sanity";

const STEPS = 20;
const log = (m: string) => console.error(`[t-ring] ${m}`);

async function run(mode: "serial" | "ringNow" | "ringK1" | "ringK2"): Promise<{
  losses: number[];
  hits: number;
  calls: number;
  bodyRuns: number;
}> {
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
  });
  const w = api.persist(
    api.tensorFromArray([0.5, -0.5, 0.25, 0.75], [2, 2], { requiresGrad: true }),
  );
  const target = api.persist(api.tensorFromArray([1, 0], [1, 2]));
  const opt = new Adam([w], { lr: 1e-2 }, api);
  let bodyRuns = 0;

  const stepFn = async (x: Tensor): Promise<Tensor> => {
    bodyRuns++;
    const pred = api.matmul(x, w);
    const diff = api.sub(pred, target);
    const loss = api.mean(api.mul(diff, diff));
    const lossOut = api.noGrad(() => api.mul(loss, 1));
    await loss.backward();
    opt.step();
    opt.zeroGrad();
    return lossOut;
  };

  const losses: number[] = [];
  const prev = api.setStepScopedCleanup(true);
  try {
    if (mode === "serial") {
      // Serial 2b: driver OWNS markStep (no runahead), reads loss each step.
      const step = api.capture(stepFn, { training: true });
      for (let i = 0; i < STEPS; i++) {
        const x = api.tensorFromArray([1 + i * 0.01, 2 - i * 0.01], [1, 2]);
        const loss = (await step(x)) as Tensor;
        losses.push((await api.cpu(loss))[0]);
        await api.markStep();
      }
      return { losses, hits: step.stats().hits, calls: step.stats().calls, bodyRuns };
    }
    // Runahead: read each loss handle on a K-BEHIND cadence so runahead
    // proceeds. After step i's push the ring holds steps [i-K+1 .. i]; the
    // oldest still-valid handle is i-K+1. Read that one (a deferred readback
    // that never fences the freshest step). Tail read at drain.
    // ringNow: runahead ring (fence deferred) BUT read the loss IMMEDIATELY each
    // step (K-behind=0). Isolates compute correctness from the deferred-readback
    // buffer-lifetime issue.
    const immediate = mode === "ringNow";
    const K = mode === "ringK1" || immediate ? 1 : 2;
    const step = api.capture(stepFn, { training: true, runahead: true, ringDepth: K });
    const handles: Tensor[] = [];
    const readVal = new Array<number | undefined>(STEPS).fill(undefined);
    for (let i = 0; i < STEPS; i++) {
      const x = api.tensorFromArray([1 + i * 0.01, 2 - i * 0.01], [1, 2]);
      handles.push((await step(x)) as Tensor);
      if (immediate) {
        readVal[i] = (await api.cpu(handles[i]))[0];
        continue;
      }
      // Read the OLDEST still-in-window handle (a K-behind logging cadence that
      // never fences the freshest step). After push i the ring holds
      // [i-K+1 .. i]; its oldest is i-K+1.
      const oldest = i - K + 1;
      if (oldest >= 0) readVal[oldest] = (await api.cpu(handles[oldest]))[0];
    }
    await step.drain();
    // Tail: any handle not yet read (the last K-1) survives drain (drain does
    // not expire — it only fences).
    for (let i = 0; i < STEPS; i++) {
      if (readVal[i] === undefined) readVal[i] = (await api.cpu(handles[i]))[0];
    }
    for (let i = 0; i < STEPS; i++) losses.push(readVal[i] as number);
    return { losses, hits: step.stats().hits, calls: step.stats().calls, bodyRuns };
  } finally {
    api.setStepScopedCleanup(prev);
  }
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
  const serial = await run("serial");
  log(`serial   losses: ${serial.losses.map((l) => l.toFixed(5)).join(", ")}`);
  log(`serial   hits=${serial.hits} calls=${serial.calls} bodyRuns=${serial.bodyRuns}`);
  // ABSOLUTE sanity (device-2 lesson): the K-parity deltas below are 0.0 even
  // when BOTH arms read ~0 from a silent submit-drop. The reference toy MSE
  // starts at O(1) — assert it, or a tainted device passes a false gate.
  assertReferenceLossNonzero(serial.losses[0], "t-ring-probe/serial");
  const now = await run("ringNow");
  log(`ringNow  losses: ${now.losses.map((l) => l.toFixed(5)).join(", ")}`);
  log(`ringNow  hits=${now.hits} calls=${now.calls} bodyRuns=${now.bodyRuns}`);
  let dNow = 0;
  for (let i = 0; i < Math.min(serial.losses.length, now.losses.length); i++)
    dNow = Math.max(dNow, Math.abs(serial.losses[i] - now.losses[i]));
  log(`maxΔ serial-vs-ringNow(immediate-read, fence-deferred)=${dNow.toExponential(2)}`);
  const k1 = await run("ringK1");
  log(`ringK1   losses: ${k1.losses.map((l) => l.toFixed(5)).join(", ")}`);
  log(`ringK1   hits=${k1.hits} calls=${k1.calls} bodyRuns=${k1.bodyRuns}`);
  const k2 = await run("ringK2");
  log(`ringK2   losses: ${k2.losses.map((l) => l.toFixed(5)).join(", ")}`);
  log(`ringK2   hits=${k2.hits} calls=${k2.calls} bodyRuns=${k2.bodyRuns}`);

  const n = Math.min(serial.losses.length, k1.losses.length, k2.losses.length);
  let dK1 = 0;
  let dK2 = 0;
  let dK1K2 = 0;
  for (let i = 0; i < n; i++) {
    dK1 = Math.max(dK1, Math.abs(serial.losses[i] - k1.losses[i]));
    dK2 = Math.max(dK2, Math.abs(serial.losses[i] - k2.losses[i]));
    dK1K2 = Math.max(dK1K2, Math.abs(k1.losses[i] - k2.losses[i]));
  }
  log(`maxΔ serial-vs-K1=${dK1.toExponential(2)} serial-vs-K2=${dK2.toExponential(2)} K1-vs-K2=${dK1K2.toExponential(2)}`);

  // VALIDATED (inc-3): the ring — deferred gen-scoped boundary (recorder-
  // finalize synchronous + sweep-after-fence), per-settle ISOLATED fences, and
  // POOL-EXCLUDED staged scalar readbacks — is BIT-IDENTICAL to the serial 2b
  // path at K=1 AND K=2 (K is a pure knob), with the body frozen on hits
  // (run-exactly-once) and ZERO GPU "used in submit while destroyed".
  const k1Pass =
    now.hits > 0 &&
    now.bodyRuns < now.calls &&
    dNow < 1e-9 && // ringNow bit-identical to serial
    dK1 < 1e-9 && // ringK1 bit-identical to serial
    k1.hits > 0 &&
    k1.bodyRuns < k1.calls;
  const k2Runahead = dK2 < 1e-9 && dK1K2 < 1e-9 && k2.hits > 0;
  log(`K1 mechanism ${k1Pass ? "PASS (bit-identical to serial, body frozen)" : "FAIL"}`);
  log(`K2 runahead ${k2Runahead ? "PASS (bit-identical: K is a pure knob)" : "FAIL"}`);
  const pass = k1Pass && k2Runahead;
  log(pass ? "PASS" : "FAIL");
  destroyWebGPU();
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  log(`FATAL ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
