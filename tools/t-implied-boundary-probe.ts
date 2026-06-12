/** Probe: train the implied-boundary test model under MODE, print losses.
 *  One engine per process (the established differential methodology). */
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";
import { Adam } from "../src/optim/adam";

const MODE = process.env.MODE ?? "explicit";
const STEPS = parseInt(process.env.STEPS ?? "8", 10);

const D_IN = 8,
  D_H = 16,
  D_OUT = 4,
  BATCH = 2;

function initData(scale: number, n: number, offset = 0): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = Math.sin(i * 12.9898 + offset) * scale;
  return out;
}

async function trainOnce(api: Torchlette, mode: string, steps: number): Promise<number[]> {
  const device = "webgpu" as const;
  const w1 = api
    .tensorFromArray(initData(0.3, D_IN * D_H, 1), [D_IN, D_H], { device })
    .requires_grad_();
  const w2 = api
    .tensorFromArray(initData(0.3, D_H * D_OUT, 2), [D_H, D_OUT], { device })
    .requires_grad_();

  const fwd = (x: Tensor, t: Tensor) =>
    api.tidy(() => {
      const h = api.relu(api.matmul(x, w1));
      const pred = api.matmul(h, w2);
      const diff = api.sub(pred, t);
      const loss = api.mean(api.mul(diff, diff));
      api.keep(loss);
      return loss;
    });

  const batch = (s: number) => ({
    x: api.tensorFromArray(initData(1.0, BATCH * D_IN, 100 + s), [BATCH, D_IN], { device }),
    t: api.tensorFromArray(initData(1.0, BATCH * D_OUT, 200 + s), [BATCH, D_OUT], { device }),
  });

  const opts =
    mode === "two-opt"
      ? [new Adam([w1], { lr: 1e-2 }, api), new Adam([w2], { lr: 1e-2 }, api)]
      : [new Adam([w1, w2], { lr: 1e-2 }, api)];

  const losses: number[] = [];
  for (let s = 0; s < steps; s++) {
    if (mode === "explicit") await api.beginStep();
    const { x, t } = batch(s);
    const loss = fwd(x, t);
    losses.push(await loss.item());
    await loss.backward();
    for (const o of opts) o.step();
    for (const o of opts) o.zeroGrad();
    if (mode === "explicit") {
      x.dispose();
      t.dispose();
      api.endStep();
      await api.markStep();
    } else if (mode === "mixed") {
      await api.markStep();
    }
    // minimal / two-opt: no ceremony
  }
  return losses;
}

async function trainMetric(api: Torchlette, steps: number): Promise<void> {
  const device = "webgpu" as const;
  const w1 = api
    .tensorFromArray(initData(0.3, D_IN * D_H, 1), [D_IN, D_H], { device })
    .requires_grad_();
  const w2 = api
    .tensorFromArray(initData(0.3, D_H * D_OUT, 2), [D_H, D_OUT], { device })
    .requires_grad_();
  const opt = new Adam([w1, w2], { lr: 1e-2 }, api);
  for (let s = 0; s < steps; s++) {
    const x = api.tensorFromArray(initData(1.0, BATCH * D_IN, 100 + s), [BATCH, D_IN], { device });
    const t = api.tensorFromArray(initData(1.0, BATCH * D_OUT, 200 + s), [BATCH, D_OUT], { device });
    const { loss, metric } = api.tidy(() => {
      const h = api.relu(api.matmul(x, w1));
      const pred = api.matmul(h, w2);
      const diff = api.sub(pred, t);
      const l = api.mean(api.mul(diff, diff));
      // Step-temp probe OUTSIDE the autograd graph (x has no grad): reading
      // forward-graph-derived tensors after backward() is a separate,
      // pre-existing limitation (backward disposes the graph).
      const metricT = api.mean(x);
      api.keep(l);
      api.keep(metricT);
      return { loss: l, metric: metricT };
    });
    const before = await metric.item();
    await loss.backward();
    if (process.env.NO_STEP !== "1") opt.step();
    const after = await metric.item();
    opt.zeroGrad();
    if (Math.abs(after - before) > 1e-6) {
      console.error(`step ${s}: metric mismatch before=${before} after=${after}`);
      process.exit(1);
    }
    console.error(`step ${s}: metric ok (${before.toFixed(6)})`);
  }
}

async function main() {
  if (!(await initWebGPU())) {
    console.error("no webgpu");
    process.exit(1);
  }
  const api = new Torchlette("webgpu");
  if (MODE === "multi") {
    // Simulate suite context: several throwaway instances, then a clean
    // explicit train on a fresh instance. Canonical first loss: 0.544801...
    const predMode = process.env.PRED_MODE ?? "explicit";
    const predN = parseInt(process.env.PRED_N ?? "4", 10);
    for (let i = 0; i < predN; i++) {
      const tmp = new Torchlette("webgpu");
      await trainOnce(tmp, predMode, 3);
      if (process.env.PRED_FINALIZE === "1") await tmp.markStep();
    }
    const api5 = new Torchlette("webgpu");
    const r = await trainOnce(api5, "explicit", STEPS);
    console.log(JSON.stringify(r));
    process.exit(0);
  }
  if (MODE === "metric") {
    await trainMetric(api, STEPS);
    console.log("METRIC PASS");
    process.exit(0);
  }
  if (MODE.includes("+")) {
    // Same-ENGINE sequential differential: "a+b" trains a then b on one api.
    const [m1, m2] = MODE.split("+");
    const r1 = await trainOnce(api, m1, STEPS);
    const r2 = await trainOnce(api, m2, STEPS);
    console.log(JSON.stringify({ first: r1, second: r2 }));
  } else {
    console.log(JSON.stringify(await trainOnce(api, MODE, STEPS)));
  }
  process.exit(0);
}
main();
