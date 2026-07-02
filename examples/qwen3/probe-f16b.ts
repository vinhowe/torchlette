/** Bisect the f16 corruption: tensorFromArray vs copy_ vs matmul vs gather. */
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { Linear } from "../../src/nn";

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  const show = async (label: string, t: import("../../src/frontend/torchlette").Tensor) =>
    console.log(label, t.dtype, Array.from(new Float32Array(await t.cpu())));

  // A. tensorFromArray f16 round-trip
  const w = api.tensorFromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 4], { dtype: "f16" });
  await show("A tensorFromArray f16:", w); // expect 1..12

  // B. copy_ f16→f16 round-trip
  const dst = api.zeros([3, 4], { dtype: "f16" });
  dst.copy_(w);
  await show("B copy_ f16→f16:    ", dst); // expect 1..12

  // C. mixed matmul with the SOURCE tensor (no copy_ involved)
  const x = api.tensorFromArray([1, 0, 0, 0, 0, 1, 0, 0], [2, 4]);
  const y = api.linear(x, w, null);
  await show("C linear(x_f32, w_f16):", y); // expect [1,5,9, 2,6,10]

  // D. gather from f16 table (no copy_)
  const table = api.tensorFromArray(Array.from({ length: 32 }, (_, i) => i), [8, 4], { dtype: "f16" });
  const idx = api.tensorFromArray([2, 7], [1, 2]);
  const g = api.embedding(table, idx);
  await show("D embedding(f16 table):", g); // expect [8,9,10,11, 28,29,30,31]

  // E. plain f16 elementwise (add) — sanity that f16 compute works at all
  const s = api.add(w, w);
  await show("E add f16+f16:      ", s); // expect 2..24

  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
