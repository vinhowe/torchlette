/**
 * Reproduce bio's loss computation: narrow+contiguous+reshape, then CE with ignoreIndex.
 */
import { Torchlette, initWebGPU } from "../src/index";
import { crossEntropy } from "../src/nn/functional";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  const B = 2, SL = 8, V = 10;
  const logits3d = new Float32Array(B * SL * V);
  for (let i = 0; i < logits3d.length; i++) {
    logits3d[i] = Math.sin(i * 0.37) * 3.0;
  }
  // Targets per (b, t) for t in 0..SL-2, plus SL-1 unused.
  // batch size B, each row has (SL-1) targets, some valid some -1.
  const N = B * (SL - 1);
  const tgtShifted = new Float32Array(N);
  tgtShifted[0] = 3; tgtShifted[1] = -1; tgtShifted[2] = 5;
  tgtShifted[3] = -1; tgtShifted[4] = 7; tgtShifted[5] = 2;
  tgtShifted[6] = -1; tgtShifted[7] = 4; tgtShifted[8] = -1;
  tgtShifted[9] = 0; tgtShifted[10] = 1; tgtShifted[11] = -1;
  tgtShifted[12] = 8; tgtShifted[13] = 9;

  const logits = api.tensorFromArray(logits3d, [B, SL, V]);
  const tgt = api.tensorFromArray(tgtShifted, [N]);

  // Like bio: narrow+contiguous+reshape
  const logits2d = logits.narrow(1, 0, SL - 1).contiguous().reshape([N, V]);
  const loss = crossEntropy(api, logits2d, tgt, { ignoreIndex: -1 });
  const gpuLoss = await loss.item();

  // CPU: iterate over [B, t] for t in 0..SL-2, using logits3d.
  let totalLoss = 0, validCount = 0;
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < SL - 1; t++) {
      const flat = b * (SL - 1) + t;
      const tgtVal = tgtShifted[flat];
      if (tgtVal < 0) continue;
      validCount++;
      const off = (b * SL + t) * V;
      let mx = -Infinity;
      for (let v = 0; v < V; v++) if (logits3d[off + v] > mx) mx = logits3d[off + v];
      let sumExp = 0;
      for (let v = 0; v < V; v++) sumExp += Math.exp(logits3d[off + v] - mx);
      const lse = mx + Math.log(sumExp);
      totalLoss += lse - logits3d[off + tgtVal];
    }
  }
  const cpuLoss = totalLoss / validCount;

  console.log(`GPU: ${gpuLoss.toFixed(4)}, CPU: ${cpuLoss.toFixed(4)}, valid: ${validCount}`);

  // Also read back what 2d logits look like
  const logits2dData = await logits2d.cpu();
  console.log("logits2d[0, 0:5]:", Array.from(logits2dData.slice(0, 5)).map(x => x.toFixed(3)));
  console.log("logits3d[0, 0, 0:5]:", Array.from(logits3d.slice(0, 5)).map(x => x.toFixed(3)));

  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
