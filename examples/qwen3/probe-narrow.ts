/** Isolate: does narrow(dim0, offset>0) round-trip correctly through cpu() and contiguous()? */
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  const t = api.tensorFromArray([0, 1, 2, 3, 4, 5, 6, 7], [4, 2]);
  const view = t.narrow(0, 2, 2); // expect [[4,5],[6,7]]

  console.log("narrow.cpu():            ", Array.from(new Float32Array(await view.cpu())));
  console.log("narrow.contiguous().cpu():", Array.from(new Float32Array(await view.contiguous().cpu())));
  // Through an op (forces kernel path): add 0
  const through = api.add(view.contiguous(), api.tensorFromArray([0, 0, 0, 0], [2, 2]));
  console.log("via add kernel:          ", Array.from(new Float32Array(await through.cpu())));
  // applyRoPE with sliced tables: q=[1,1,1,2] ones, cos slice row2 of [0,1,2,3] halves
  const cosT = api.tensorFromArray([10, 20, 30, 40], [4, 1]); // half=1
  const sinT = api.tensorFromArray([0, 0, 0, 0], [4, 1]);
  const q = api.tensorFromArray([1, 1], [1, 1, 1, 2]);
  const r = api.applyRoPE(q, cosT.narrow(0, 2, 1).contiguous(), sinT.narrow(0, 2, 1).contiguous());
  // expected: q*cos(row2)=30 → [30, 30] (sin=0)
  console.log("applyRoPE sliced cos:    ", Array.from(new Float32Array(await r.cpu())), "(expect [30,30])");
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
