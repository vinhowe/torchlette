/**
 * Reproducer for suspected bug in fused cross-entropy kernel.
 *
 * Compares fused CE (webgpu 2D path) vs decomposed CE (3D path, via reshape)
 * for forward loss and backward gradients. With bio-like shapes (N=2016, V=427).
 */

import { Torchlette, initWebGPU } from "../src/index";
import { crossEntropy } from "../src/nn/functional";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: true });

  const N = 2016;
  const V = 427;

  // Reproducible random logits
  let seed = 12345;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  const logitsData = new Float32Array(N * V);
  for (let i = 0; i < logitsData.length; i++) {
    logitsData[i] = (rng() - 0.5) * 4.0; // range [-2, 2]
  }
  const targetsData = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    targetsData[i] = Math.floor(rng() * V);
  }

  console.log(`N=${N}, V=${V}`);
  console.log(`logits sample:`, Array.from(logitsData.slice(0, 5)));
  console.log(`targets sample:`, Array.from(targetsData.slice(0, 5)));

  // ============ Fused path (2D) ============
  await api.beginStep();
  const logitsF = api.tensorFromArray(logitsData, [N, V], { requiresGrad: true });
  const targetsF = api.tensorFromArray(targetsData, [N]);
  const lossFused = crossEntropy(api, logitsF, targetsF);
  const lossF = await lossFused.item();
  await lossFused.backward();
  const gradFused = await logitsF.grad!.cpu();
  api.endStep();

  // ============ Decomposed path (3D) ============
  await api.beginStep();
  const logitsD = api.tensorFromArray(logitsData, [N, 1, V], { requiresGrad: true });
  const targetsD = api.tensorFromArray(targetsData, [N, 1]);
  const lossDecomp = crossEntropy(api, logitsD, targetsD);
  const lossD = await lossDecomp.item();
  await lossDecomp.backward();
  const gradDecomp = await logitsD.grad!.cpu();
  api.endStep();

  console.log(`\nForward loss:`);
  console.log(`  fused:      ${lossF}`);
  console.log(`  decomposed: ${lossD}`);
  console.log(`  diff:       ${Math.abs(lossF - lossD).toExponential(3)}`);

  // Compare gradients
  let maxDiff = 0;
  let totalDiff = 0;
  let nanCount = 0;
  for (let i = 0; i < N * V; i++) {
    const a = gradFused[i], b = gradDecomp[i];
    if (!isFinite(a) || !isFinite(b)) {
      nanCount++;
      if (nanCount < 5) {
        console.log(`  idx ${i} (row=${Math.floor(i/V)}): fused=${a}, decomp=${b}`);
      }
      continue;
    }
    const diff = Math.abs(a - b);
    if (diff > maxDiff) maxDiff = diff;
    totalDiff += diff;
  }
  console.log(`\nBackward grad comparison:`);
  console.log(`  max diff:     ${maxDiff.toExponential(3)}`);
  console.log(`  avg diff:     ${(totalDiff / (N * V)).toExponential(3)}`);
  console.log(`  nan/inf count: ${nanCount}`);
  console.log(`  fused sample: ${Array.from(gradFused.slice(0, 5)).map(x => x.toFixed(6))}`);
  console.log(`  decomp sample:${Array.from(gradDecomp.slice(0, 5)).map(x => x.toFixed(6))}`);

  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
