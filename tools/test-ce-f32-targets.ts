/**
 * Compare fused CE with f32 targets vs CPU-computed expected value.
 * And with i32 targets.
 */
import { Torchlette, initWebGPU } from "../src/index";
import { crossEntropy } from "../src/nn/functional";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  const N = 6, V = 10;
  const logitsData = new Float32Array(N * V);
  for (let i = 0; i < logitsData.length; i++) {
    logitsData[i] = (i * 31 + 7) % 13 - 6;
  }
  // Mix valid and ignored (-1) targets
  const targetsData = [3, -1, 0, -1, 7, 2]; // 4 valid, 2 ignored

  const logits = api.tensorFromArray(logitsData, [N, V]);
  const targetsF32 = api.tensorFromArray(targetsData, [N]);
  const targetsI32 = api.tensorFromArray(targetsData, [N], { dtype: "i32" });

  const lossF32 = crossEntropy(api, logits, targetsF32, { ignoreIndex: -1 });
  const lF32 = await lossF32.item();

  const lossI32 = crossEntropy(api, logits, targetsI32, { ignoreIndex: -1 });
  const lI32 = await lossI32.item();

  // CPU expected
  let totalLoss = 0, validCount = 0;
  for (let b = 0; b < N; b++) {
    if (targetsData[b] < 0) continue;
    validCount++;
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (logitsData[b * V + v] > mx) mx = logitsData[b * V + v];
    let sumExp = 0;
    for (let v = 0; v < V; v++) sumExp += Math.exp(logitsData[b * V + v] - mx);
    const lse = mx + Math.log(sumExp);
    totalLoss += lse - logitsData[b * V + targetsData[b]];
  }
  const cpuLoss = totalLoss / validCount;

  console.log(`GPU f32 targets: ${lF32.toFixed(4)}`);
  console.log(`GPU i32 targets: ${lI32.toFixed(4)}`);
  console.log(`CPU expected:    ${cpuLoss.toFixed(4)}`);
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
