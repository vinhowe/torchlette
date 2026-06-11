// Does compiled replay FREEZE RNG? rand() in a markStep loop with a
// compiled-eligible plan: distinct values per step on the lowered path;
// if replays bake the recorded seed, steps 2+ repeat step-recorded values.
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
async function main() {
  if (!(await initWebGPU())) process.exit(1);
  const api = new Torchlette("webgpu", { enableFusion: true });
  const rt = api._runtime();
  const w = api.tensorFromArray([1, 1, 1, 1], [4], { device: "webgpu", requiresGrad: false });
  const sums: number[] = [];
  for (let step = 0; step < 6; step++) {
    await api.beginStep();
    // plan: rand -> mul by persistent weight -> sum (compiled-eligible shape)
    const r = rt.rand([1024], "webgpu");
    const x = rt.mul(r, rt.sum(rt.mul(w._unwrap(), 1.0)));
    const s = rt.sum(x);
    const v = (await rt.cpu(s))[0];
    sums.push(v);
    api.endStep();
    await api.markStep();
  }
  console.log("sums:", sums.map((v) => v.toFixed(3)).join(" "));
  const uniq = new Set(sums.map((v) => v.toFixed(6))).size;
  console.log(uniq === sums.length ? "ALL DISTINCT (fresh RNG per step)" : `ONLY ${uniq}/6 distinct — RNG FROZEN under replay`);
  process.exit(0);
}
main().catch((e) => { console.error(e); process.exit(1); });
