/**
 * Reproduce browser's CE issue in Node. If Node shows same bug, it's the kernel.
 * If Node is fine, it's browser-specific caching/timing.
 */
import { Torchlette, initWebGPU } from "../src/index";
import { crossEntropy } from "../src/nn/functional";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  const N = 2016, V = 427;
  const logitsData = new Float32Array(N * V);
  for (let i = 0; i < logitsData.length; i++) logitsData[i] = Math.sin(i * 0.01);
  const tgtData = new Float32Array(N);
  let validCount = 0;
  for (let i = 0; i < N; i++) {
    tgtData[i] = i % 3 === 0 ? -1 : (i * 7) % V;
    if (tgtData[i] !== -1) validCount++;
  }

  const logits = api.tensorFromArray(logitsData, [N, V]);
  const tgt = api.tensorFromArray(tgtData, [N]);
  const lPer = crossEntropy(api, logits, tgt, { ignoreIndex: -1, reduction: "none" });
  const perSample = await lPer.cpu();
  let count999 = 0, count0 = 0, sumIgnored = 0, sumValid = 0, nonZero = 0;
  for (let i = 0; i < N; i++) {
    const v = perSample[i];
    if (v === 999 || (v > 998 && v < 1000)) count999++;
    if (v === 0) count0++;
    if (v !== 0) nonZero++;
    if (tgtData[i] < 0) sumIgnored += v;
    else sumValid += v;
  }
  console.log(`count999=${count999}, count0=${count0}, nonZero=${nonZero}, sumIgnored=${sumIgnored.toFixed(2)}, sumValid=${sumValid.toFixed(2)}, validCount=${validCount}`);
  console.log(`first 10 values:`, Array.from(perSample.slice(0, 10)).map(x => x.toFixed(3)));
  console.log(`first 10 targets:`, Array.from(tgtData.slice(0, 10)));
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
