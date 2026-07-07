/**
 * Minimal probe for the compiled-harvest view-base retain accounting.
 * One randn param + clipGradNorm_ + Adam (the 867f6c1 leak repro shape),
 * PLUS a registered-buffer-style persistent tensor narrowed every step
 * (the posIndices false-positive shape).
 *
 * Prints per-step: totalStorages / reachableStorages, and rc of the
 * persistent tensor's storage. Run under different code variants to see
 * which release paths fire.
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu/index.ts";
import { Torchlette } from "../src/frontend/torchlette.ts";
import { Adam } from "../src/optim/index.ts";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { storageTracker } from "../src/graph/storage-tracker.ts";
import { rcGet } from "../src/graph/refcount.ts";

async function main() {
  if (!(await initWebGPU())) {
    console.error("no webgpu");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  const param = api.randn([64], { device: "webgpu", requiresGrad: true });
  // posIndices-like persistent buffer: lazy arange+reshape, narrowed each step
  const posIdx = api.arange(64).reshape([1, 64]);
  const opt = new Adam([param], { lr: 1e-3, adamW: true }, api);

  for (let step = 0; step < 12; step++) {
    await api.beginStep();
    const pos = posIdx.narrow(1, 0, 32); // view of persistent buffer, used in fwd
    const posF = api.toDtype(pos, "f32");
    const x = api.mul(param, param);
    const loss = api.add(api.sum(x), api.sum(posF));
    await loss.backward();
    clipGradNorm_(api, [param], 1.0);
    await opt.step();
    opt.zeroGrad();
    api.endStep();
    await api.markStep();
    const s = storageTracker.stats();
    const posStorage = (posIdx as any)._runtimeTensor?.ref ?? null;
    const ref = (posIdx as any).runtimeTensor?._lazyRef ?? (posIdx as any)._lazyRef;
    let posInfo = "?";
    try {
      const rt: any = (posIdx as any).runtime ?? null;
      posInfo = "";
    } catch {}
    console.log(
      `step ${step}: total=${s.totalStorages} reachable=${s.reachableStorages}`,
    );
  }
  // report the persistent tensor's storage rc/destroyed state via a read
  try {
    const arr = await posIdx.cpu();
    console.log("posIdx read-back ok, first vals:", Array.from(arr.slice(0, 4)));
  } catch (e) {
    console.log("posIdx read-back FAILED:", (e as Error).message);
  }
  await destroyWebGPU();
  process.exit(0);
}
main();
