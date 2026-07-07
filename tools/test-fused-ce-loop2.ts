/**
 * Simulate bio-like training loop with LEARNABLE signal.
 * Compares fused CE vs decomposed over many steps.
 */

import { Torchlette, initWebGPU, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: true });

  const N = 2016;
  const V = 427;
  const D = 64;

  let seed = 42;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  // Identical init
  const w0 = new Float32Array(D * V);
  for (let i = 0; i < w0.length; i++) w0[i] = (rng() - 0.5) * 0.1;

  // Fused model
  const wF = api.tensorFromArray(w0.slice(), [D, V], { requiresGrad: true });
  const optF = new Adam([wF], { lr: 3e-4 });

  // Decomposed model (same init)
  const wD = api.tensorFromArray(w0.slice(), [D, V], { requiresGrad: true });
  const optD = new Adam([wD], { lr: 3e-4 });

  // Learnable data: each sample has a "concept" which dictates target class.
  // Input = noisy one-hot over concept dimension.
  const nConcepts = 20;
  const classPerConcept = Math.floor(V / nConcepts);

  function makeBatch() {
    const x = new Float32Array(N * D);
    const y = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const concept = Math.floor(rng() * nConcepts);
      // Input: activate dim = concept, plus noise
      for (let d = 0; d < D; d++) {
        x[i * D + d] = (rng() - 0.5) * 0.3;
      }
      x[i * D + concept] = 1.0;
      y[i] = concept * classPerConcept;
    }
    return { x, y };
  }

  console.log("step | loss_fused | loss_decomp | diff");
  console.log("-----|------------|-------------|------");
  for (let step = 0; step < 2000; step++) {
    const batch = makeBatch();
    const xt = api.tensorFromArray(batch.x, [N, D]);
    const yt = api.tensorFromArray(batch.y, [N]);

    // Fused
    await api.beginStep();
    const logitsF = api.matmul(xt, wF);
    const lossF = crossEntropy(api, logitsF, yt);
    const lfVal = await lossF.item();
    await lossF.backward();
    optF.step();
    optF.zeroGrad();
    api.endStep();

    // Decomposed (3D)
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

    const shouldLog = step % 50 === 0 || !isFinite(lfVal) || !isFinite(ldVal);
    if (shouldLog) {
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
