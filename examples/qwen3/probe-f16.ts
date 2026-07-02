/** Fast isolation: the mixed-dtype ops the f16-weights model relies on. */
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { Embedding, Linear } from "../../src/nn";

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  // 1. f16 zeros creation
  const z = api.zeros([2, 4], { dtype: "f16" });
  console.log("f16 zeros:", Array.from(new Float32Array(await z.cpu())), "dtype", z.dtype);

  // 2. tensorFromArray f16 + copy_ into f16 param
  const lin = new Linear(api, 4, 3, { bias: false, dtype: "f16" });
  const w = api.tensorFromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 4], { dtype: "f16" });
  lin.weight.copy_(w);

  // 3. mixed matmul via api.linear: f32 x @ f16 W^T
  const x = api.tensorFromArray([1, 0, 0, 0, 0, 1, 0, 0], [2, 4]);
  const y = lin.forward(x);
  console.log("linear f32x@f16W dtype:", y.dtype, "vals:", Array.from(new Float32Array(await y.cpu())), "(expect [1,5,9, 2,6,10])");

  // 4. f16 embedding gather + .float()
  const emb = new Embedding(api, 8, 4, { dtype: "f16" });
  const ew = api.tensorFromArray(Array.from({ length: 32 }, (_, i) => i), [8, 4], { dtype: "f16" });
  emb.weight.copy_(ew);
  const idx = api.tensorFromArray([2, 7], [1, 2]);
  const e = emb.forward(idx).float();
  console.log("embed f16→f32 dtype:", e.dtype, "vals:", Array.from(new Float32Array(await e.cpu())), "(expect [8..11, 28..31])");

  console.log("F16 PROBE PASS");
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
