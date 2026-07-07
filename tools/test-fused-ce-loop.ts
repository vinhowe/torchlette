/**
 * Simulate bio-like training loop and check if fused CE produces NaN over many steps.
 * Two models are trained in parallel: one uses fused CE (2D), one uses decomposed (3D).
 */

import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: true });

  const N = 2016;
  const V = 427;
  const D = 64;

  // Reproducible random
  let seed = 42;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  // Two identical models (same init)
  function makeLinearWeights(): Float32Array {
    const w = new Float32Array(D * V);
    for (let i = 0; i < w.length; i++) w[i] = (rng() - 0.5) * 0.1;
    return w;
  }
  const w0 = makeLinearWeights();

  // Fused model
  const wF = api.tensorFromArray(w0.slice(), [D, V], { requiresGrad: true });
  const optF = new Adam([wF], { lr: 3e-4 });

  // Decomposed model
  seed = 42; const _resetRng = rng();
  const wD = api.tensorFromArray(w0.slice(), [D, V], { requiresGrad: true });
  const optD = new Adam([wD], { lr: 3e-4 });

  // Generate fixed batches of (x, y) — same for both
  function makeBatch() {
    const x = new Float32Array(N * D);
    const y = new Float32Array(N);
    for (let i = 0; i < x.length; i++) x[i] = (rng() - 0.5) * 2.0;
    for (let i = 0; i < N; i++) y[i] = Math.floor(rng() * V);
    return { x, y };
  }

  console.log("step | loss_fused | loss_decomp | diff");
  console.log("-----|------------|-------------|------");
  for (let step = 0; step < 1000; step++) {
    const batch = makeBatch();
    const xt = api.tensorFromArray(batch.x, [N, D]);
    const yt = api.tensorFromArray(batch.y, [N]);

    // Fused: logits = x @ w (2D path)
    await api.beginStep();
    const logitsF = api.matmul(xt, wF);
    const lossF = crossEntropy(api, logitsF, yt);
    const lfVal = await lossF.item();
    await lossF.backward();
    optF.step();
    optF.zeroGrad();
    api.endStep();

    // Decomposed: logits reshaped to 3D
    await api.beginStep();
    const logitsD = api.matmul(xt, wD).reshape([N, 1, V]);
    const ytD = yt.reshape([N, 1]);
    const lossD = crossEntropy(api, logitsD, ytD);
    const ldVal = await lossD.item();
    await lossD.backward();
    optD.step();
    optD.zeroGrad();
    api.endStep();

    xt.dispose(); yt.dispose();

    if (step % 25 === 0 || !isFinite(lfVal) || !isFinite(ldVal)) {
      const diff = Math.abs(lfVal - ldVal);
      console.log(`${step.toString().padStart(4)} | ${lfVal.toFixed(6).padStart(10)} | ${ldVal.toFixed(6).padStart(11)} | ${diff.toExponential(2)}`);
      if (!isFinite(lfVal) || !isFinite(ldVal)) {
        console.log("DIVERGED");
        break;
      }
    }
  }
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
