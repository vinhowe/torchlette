/**
 * Test fused CE with large N (similar to bio) and padded targets.
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

  // per-sample loss
  const lPer = crossEntropy(api, logits, tgt, { ignoreIndex: -1, reduction: 'none' });
  const perSample = await lPer.cpu();
  let sumPer = 0, nonZeroCount = 0;
  for (let i = 0; i < N; i++) {
    sumPer += perSample[i];
    if (perSample[i] !== 0) nonZeroCount++;
  }
  console.log(`perSample: sum=${sumPer.toFixed(2)}, nonZero=${nonZeroCount}, expected_nonzero=${validCount}`);
  console.log(`first 10 values:`, Array.from(perSample.slice(0, 10)).map(x => x.toFixed(3)));
  console.log(`targets first 10:`, Array.from(tgtData.slice(0, 10)));

  const lossAgg = crossEntropy(api, logits, tgt, { ignoreIndex: -1 });
  console.log(`aggregate loss: ${(await lossAgg.item()).toFixed(4)}, expected ~${(sumPer / validCount).toFixed(4)}`);

  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
