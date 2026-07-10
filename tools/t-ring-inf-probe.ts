/**
 * inc-3 gate 3 — THE INJECTED-INF GATE (charter §2b design (c)).
 *
 * A runahead training loop WITH GradScaler; a batch poisoned to produce inf
 * gradients at step INF_STEP (a tape-HIT-era step). Asserts the contract:
 *  - K=1 and K=2 trajectories BIT-IDENTICAL (the in-graph per-element
 *    zero-mask in the unscaleGrad kernel is the skip — exact for any K, and
 *    replay-faithful: INF_STEP is a hit, so the mask ran inside a REPLAYED
 *    plan, not a body).
 *  - The CPU scale-mirror bookkeeping lags by EXACTLY ≤K steps (asserted as a
 *    bound, not absence): the backoff (1024→512) lands at call INF_STEP+K via
 *    `scaler.resolveOldestDeferred()` at the K-behind cadence — found-inf
 *    NEVER rides a hot-loop readback (that would cap K=1).
 *  - Scale timing is trajectory-invisible: scales are powers of two, so
 *    scale/unscale cancel exactly in fp32 — a lagged backoff cannot perturb
 *    the trajectory (why the arms stay bit-identical despite different lags).
 *
 *   VULKAN_DEVICE_INDEX=2 LD_LIBRARY_PATH=tools/vk-shim \
 *   TORCHLETTE_STEP_TAPE=1 npx tsx tools/t-ring-inf-probe.ts
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler } from "../src/optim/index.ts";
import { STEP_TAPE_REPLAY } from "../src/core/step-tape";
import type { Tensor } from "../src/frontend/tensor";

const STEPS = 20;
const INF_STEP = 12;
const INIT_SCALE = 1024.0;
const log = (m: string) => console.error(`[t-ring-inf] ${m}`);

async function run(K: number): Promise<{
  losses: number[];
  scaleAfterCall: number[];
  hits: number;
  bodyRuns: number;
  finalScale: number;
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
  const scaler = new GradScaler(api, {
    initScale: INIT_SCALE,
    growthInterval: 100000, // no growth in-window: isolate the backoff event
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

  const step = api.capture(stepFn, { training: true, runahead: true, ringDepth: K });
  const handles: Tensor[] = [];
  const readVal = new Array<number | undefined>(STEPS).fill(undefined);
  const scaleAfterCall: number[] = [];
  const prev = api.setStepScopedCleanup(true);
  try {
    for (let i = 0; i < STEPS; i++) {
      // Poisoned batch at INF_STEP: 1e30 input → squared-error ~1e60 → f32 inf
      // loss → inf/NaN grads → the unscaleGrad kernel's per-element zero-mask.
      const base = i === INF_STEP ? 1e30 : 1 + i * 0.01;
      const x = api.tensorFromArray([base, 2 - i * 0.01], [1, 2]);
      handles.push((await step(x)) as Tensor);
      // Per-step report snapshot (queue-ordered isolation; hit-safe).
      scaler.snapshotDeferred();
      // K-behind logging cadence: read the oldest in-window loss + resolve the
      // oldest found-inf report (CPU mirror lag = exactly K steps).
      const oldest = i - K + 1;
      if (oldest >= 0) readVal[oldest] = (await api.cpu(handles[oldest]))[0];
      if (i >= K) await scaler.resolveOldestDeferred();
      scaleAfterCall.push(scaler.getScale());
    }
    await step.drain();
    // Tail: remaining reports + losses.
    while (await scaler.resolveOldestDeferred()) {
      /* drain in order */
    }
    for (let i = 0; i < STEPS; i++) {
      if (readVal[i] === undefined) readVal[i] = (await api.cpu(handles[i]))[0];
    }
  } finally {
    api.setStepScopedCleanup(prev);
  }
  return {
    losses: readVal as number[],
    scaleAfterCall,
    hits: step.stats().hits,
    bodyRuns,
    finalScale: scaler.getScale(),
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

  const k1 = await run(1);
  log(`K1 losses: ${k1.losses.map((l) => l.toPrecision(6)).join(", ")}`);
  log(`K1 scaleAfterCall: ${k1.scaleAfterCall.join(",")}`);
  log(`K1 hits=${k1.hits} bodyRuns=${k1.bodyRuns} finalScale=${k1.finalScale}`);
  const k2 = await run(2);
  log(`K2 losses: ${k2.losses.map((l) => l.toPrecision(6)).join(", ")}`);
  log(`K2 scaleAfterCall: ${k2.scaleAfterCall.join(",")}`);
  log(`K2 hits=${k2.hits} bodyRuns=${k2.bodyRuns} finalScale=${k2.finalScale}`);

  // 1. K=1 == K=2 trajectories BIT-IDENTICAL (Infinity === Infinity at INF_STEP).
  let identical = true;
  for (let i = 0; i < STEPS; i++) {
    const a = k1.losses[i];
    const b = k2.losses[i];
    if (!(a === b || (Number.isNaN(a) && Number.isNaN(b)))) identical = false;
  }
  // 2. The inf actually happened at INF_STEP and the mask recovered the run
  //    (post-inf losses finite and the trajectory keeps descending).
  const infSeen =
    !Number.isFinite(k1.losses[INF_STEP]) && !Number.isFinite(k2.losses[INF_STEP]);
  const recovered =
    Number.isFinite(k1.losses[STEPS - 1]) &&
    k1.losses[STEPS - 1] < k1.losses[INF_STEP - 1];
  // 3. INF_STEP was a HIT-era step (the mask ran inside a REPLAYED plan).
  const hitEra = k1.bodyRuns <= INF_STEP && k2.bodyRuns <= INF_STEP;
  // 4. The CPU scale mirror backed off with lag EXACTLY ≤ K per arm:
  //    still INIT at call INF_STEP+K-1, halved at call INF_STEP+K.
  const lagBound = (r: { scaleAfterCall: number[] }, K: number) =>
    r.scaleAfterCall[INF_STEP + K - 1] === INIT_SCALE &&
    r.scaleAfterCall[INF_STEP + K] === INIT_SCALE / 2;
  const k1Lag = lagBound(k1, 1);
  const k2Lag = lagBound(k2, 2);
  const finalScales = k1.finalScale === INIT_SCALE / 2 && k2.finalScale === INIT_SCALE / 2;

  log(
    `identical=${identical} infSeen=${infSeen} recovered=${recovered} hitEra=${hitEra} k1Lag<=1=${k1Lag} k2Lag<=2=${k2Lag} finalScales=${finalScales}`,
  );
  const pass =
    identical && infSeen && recovered && hitEra && k1Lag && k2Lag && finalScales &&
    k1.hits > 0 && k2.hits > 0;
  log(pass ? "PASS" : "FAIL");
  destroyWebGPU();
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  log(`FATAL ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
