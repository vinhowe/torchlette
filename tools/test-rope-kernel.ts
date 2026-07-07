/**
 * Correctness test for fused RoPE tile-IR kernel.
 *
 * Compares api.applyRoPE output against a reference implementation
 * built from narrow+cat+mul+add. Half-split (GPT-NeoX/Llama) convention.
 */
import { Torchlette, initWebGPU } from "../src/index";

function referenceRoPE(
  api: Torchlette,
  qk: any,
  cos: any, // [S, D/2]
  sin: any, // [S, D/2]
): any {
  const shape = qk.shape;
  const D = shape[shape.length - 1];
  const half = D / 2;
  // Reshape cos/sin from [S, D/2] → [1, 1, S, D/2] to broadcast with qk [B,H,S,D/2].
  const rank = shape.length;
  const broadcastShape: number[] = Array(rank - 2).fill(1).concat([shape[rank - 2], half]);
  const cosB = cos.reshape(broadcastShape);
  const sinB = sin.reshape(broadcastShape);
  const x1 = qk.narrow(rank - 1, 0, half);     // first half
  const x2 = qk.narrow(rank - 1, half, half);  // second half
  // rotated = [x1*cos - x2*sin, x1*sin + x2*cos]
  const r1 = api.sub(api.mul(x1, cosB), api.mul(x2, sinB));
  const r2 = api.add(api.mul(x1, sinB), api.mul(x2, cosB));
  return api.cat([r1, r2], rank - 1);
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  const B = 2, H = 2, S = 10, D = 8;
  const half = D / 2;
  const base = 10000;

  // Precompute cos/sin tables [S, D/2]
  const cosData = new Float32Array(S * half);
  const sinData = new Float32Array(S * half);
  for (let m = 0; m < S; m++) {
    for (let i = 0; i < half; i++) {
      const theta = m * Math.pow(base, -(2 * i) / D);
      cosData[m * half + i] = Math.cos(theta);
      sinData[m * half + i] = Math.sin(theta);
    }
  }

  // Random qk input [B,H,S,D]
  const total = B * H * S * D;
  const qkData = new Float32Array(total);
  for (let i = 0; i < total; i++) qkData[i] = Math.random() * 2 - 1;

  await api.beginStep();
  const qk = api.tensorFromArray(qkData, [B, H, S, D]);
  const cos = api.tensorFromArray(cosData, [S, half]);
  const sin = api.tensorFromArray(sinData, [S, half]);

  // Fused (tile-IR) kernel
  const fused = api.applyRoPE(qk, cos, sin);
  const fusedArr = await fused.cpu();

  // Reference using narrow+cat+mul+add+sub
  const ref = referenceRoPE(api, qk, cos, sin);
  const refArr = await ref.cpu();

  api.endStep();

  const mad = maxAbsDiff(fusedArr, refArr);
  console.log(`[forward] max |fused - ref| = ${mad.toExponential(3)}`);
  if (mad > 1e-5) {
    console.error("FAILED: forward mismatch");
    console.error("fused[0..8]:", Array.from(fusedArr.slice(0, 8)));
    console.error("ref  [0..8]:", Array.from(refArr.slice(0, 8)));
    process.exit(1);
  }

  // --- Backward test: apply forward then backward with sin negated.
  // Should recover original input.
  await api.beginStep();
  const qk2 = api.tensorFromArray(qkData, [B, H, S, D]);
  const cos2 = api.tensorFromArray(cosData, [S, half]);
  const sin2 = api.tensorFromArray(sinData, [S, half]);
  const rotated = api.applyRoPE(qk2, cos2, sin2);

  // Negate sin table: inverse rotation.
  const sinNegData = new Float32Array(sinData.length);
  for (let i = 0; i < sinData.length; i++) sinNegData[i] = -sinData[i];
  const sinNeg = api.tensorFromArray(sinNegData, [S, half]);
  const roundTrip = api.applyRoPE(rotated, cos2, sinNeg);
  const rtArr = await roundTrip.cpu();
  api.endStep();

  const rtDiff = maxAbsDiff(qkData, rtArr);
  console.log(`[round-trip] max |qk - R^-1(R(qk))| = ${rtDiff.toExponential(3)}`);
  if (rtDiff > 1e-5) {
    console.error("FAILED: round-trip mismatch");
    process.exit(1);
  }

  console.log("PASS");
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
