/**
 * Debug: is sliceColumns producing wrong values?
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: true });

  const K = 768;
  const N = 50257;

  // Create B with column 0 = [1,1,...,1] (768 ones)
  // Test: after sliceColumns, does the column survive correctly?

  // Instead of sliceColumns, let's test the full chunked matmul
  // but with different B values to distinguish chunk 0 vs chunk 1

  // Test 1: B with only first 43690 columns non-zero (= 1.0)
  // All other columns = 0. So result should be 768 for cols < 43690, 0 for cols >= 43690
  const bData1 = new Array(K * N);
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < N; j++) {
      bData1[i * N + j] = j < 43690 ? 1.0 : 0.0;
    }
  }
  const a = api.tensorFromArray(Array(K).fill(1), [1, K], { device: "webgpu" });
  const b1 = api.tensorFromArray(bData1, [K, N], { device: "webgpu" });
  const result1 = a.matmul(b1);
  const idx0 = api.tensorFromArray([0], [1, 1], { device: "webgpu" });
  const val1_0 = Array.from(await result1.gather(idx0, { dim: 1 }).cpu())[0];
  console.log(`Test 1 (cols 0-43689 = 1, rest = 0): result[0]=${val1_0?.toFixed(2)} (expected 768)`);

  // Test 2: B with only column 0 = [1, 1, ..., 1], rest = 0
  const bData2 = new Array(K * N).fill(0);
  for (let i = 0; i < K; i++) {
    bData2[i * N + 0] = 1.0;
  }
  const b2 = api.tensorFromArray(bData2, [K, N], { device: "webgpu" });
  const result2 = a.matmul(b2);
  const val2_0 = Array.from(await result2.gather(idx0, { dim: 1 }).cpu())[0];
  console.log(`Test 2 (only col 0 = 1): result[0]=${val2_0?.toFixed(2)} (expected 768)`);

  // Test 3: B with column 0 = [row index], rest = 0
  // This way result[0] = sum of 0+1+2+...+767 = 768*767/2 = 294528
  const bData3 = new Array(K * N).fill(0);
  for (let i = 0; i < K; i++) {
    bData3[i * N + 0] = i;
  }
  const b3 = api.tensorFromArray(bData3, [K, N], { device: "webgpu" });
  const result3 = a.matmul(b3);
  const val3_0 = Array.from(await result3.gather(idx0, { dim: 1 }).cpu())[0];
  console.log(`Test 3 (col 0 = row index): result[0]=${val3_0?.toFixed(2)} (expected ${K*(K-1)/2})`);

  process.exit(0);
}

main().catch(e => { console.error("ERROR:", e.message, e.stack); process.exit(1); });
