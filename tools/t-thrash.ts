import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
async function main() {
  if (!(await initWebGPU())) process.exit(1);
  const api = new Torchlette("webgpu", { enableFusion: true });
  const rt = api._runtime();
  const w = rt.tensorFromArray([1, 2, 3, 4], [4], "webgpu");
  for (let step = 0; step < 8; step++) {
    await api.beginStep();
    // triu's k payload varies per step → structurally identical plans,
    // differing payload → should trip the detector.
    const m = rt.tensorFromArray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], [4,4], "webgpu");
    const t = rt.triu(m, step % 4);
    const s = rt.sum(rt.mul(rt.sum(t), rt.sum(w)));
    await rt.cpu(s);
    api.endStep();
    await api.markStep();
  }
  const { getPayloadThrashStats } = await import("../src/executor/executor");
  console.log("thrash stats:", JSON.stringify(getPayloadThrashStats()));
  process.exit(0);
}
main().catch((e) => { console.error(e); process.exit(1); });
