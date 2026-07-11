/**
 * Regression probe for the static-KV harvest-view reclaimed-read FALSE POSITIVE
 * (task #90). Reproduces the Gemma-2 static-KV decode lifetime seam WITHOUT the
 * 5GB model:
 *
 *   persistent KV buffer, updated in place every captured step via
 *     copy_(kv, kv.scatterAdd(idx, x))    // the model's kSlot update
 *   then read through a view
 *     narrow(kv, dim, 0, len)             // the attention-score read
 *
 * Under TORCHLETTE_STEP_TAPE=1 the compiled-plan harvest re-creates a VIEW
 * handle over the persistent `kv` base each replay (`planOwnedBaseRetain`). The
 * per-replay view handle is reaped at markStep (rc 0) so `isDestroyed(viewId)`
 * is true even though its GPU buffer aliases the live base the PLAN keeps
 * alive — the scatterAdd `dst` external-leaf read then trips the [lifetime]
 * reclaimed-read guard. Pre-fix, the strict guard (throw-by-default since task
 * #73) THROWS here; the read is provably safe (view buffer === live base
 * buffer), so the fix exonerates it and this probe runs clean.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *      TORCHLETTE_STEP_TAPE=1 \
 *      npx tsx tools/t-static-kv-harvest-lifetime.ts
 */
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { type Tensor, Torchlette } from "../src/frontend/torchlette";

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  const H = 4;
  const S = 8;
  const D = 4;
  const NSTEPS = 6;

  const idxFor = (pos: number): Tensor => {
    const arr = new Float32Array(H * 1 * D).fill(pos);
    return api.tensorFromArray(arr, [1, H, 1, D]);
  };
  // The decode body: in-place KV update (the model's copy_(kSlot,
  // kSlot.scatterAdd(...))), then read the accumulated KV back through a view
  // (the model's narrow read). Always scatter into slot 0 and read the FULL kv
  // so the captured plan is structurally IDENTICAL every step → it goes hot and
  // the harvest activates.
  const body = (kv: Tensor, x: Tensor): Tensor => {
    kv.copy_(kv.scatterAdd(idxFor(0), x, { dim: 2 }));
    return api.narrow(kv, 2, 0, S).sum([2]); // [1,H,D]
  };

  const run = async (captured: boolean): Promise<number[][]> => {
    const kv = api.registerState(api.zeros([1, H, S, D]));
    // captured=false: call body directly (golden, no replay). captured=true:
    // wrap in capture() so the harvest/replay path (the bug surface) runs.
    const decode = captured ? api.capture((x: Tensor) => body(kv, x)) : null;
    const prev = api.setStepScopedCleanup(true);
    const sums: number[][] = [];
    try {
      await api.markStep();
      for (let t = 0; t < NSTEPS; t++) {
        const x = api.tensorFromArray(new Array(H * D).fill(t + 1), [
          1,
          H,
          1,
          D,
        ]);
        const out = decode
          ? ((await decode(x)) as Tensor)
          : api.noGrad(() => body(kv, x));
        sums.push(Array.from(await api.cpu(out)));
        await api.markStep();
      }
      if (decode && decode.stats().hits === 0) {
        throw new Error(
          "probe never went hot (traces only) — mechanism not exercised",
        );
      }
    } finally {
      api.setStepScopedCleanup(prev);
    }
    if (decode) console.log(`stats: ${JSON.stringify(decode.stats())}`);
    return sums;
  };

  // Golden trajectory (no capture) vs captured trajectory (harvest/replay). Any
  // stale/reclaimed harvest read shows as a DIVERGENCE here; under STRICT the
  // pre-fix code throws before reaching the diff.
  const golden = await run(false);
  const captured = await run(true);

  let allOk = true;
  for (let p = 0; p < NSTEPS; p++) {
    for (let i = 0; i < golden[p].length; i++) {
      if (Math.abs(golden[p][i] - captured[p][i]) > 1e-3) {
        allOk = false;
        console.error(
          `step ${p}[${i}]: captured ${captured[p][i]} != golden ${golden[p][i]}`,
        );
      }
    }
  }
  console.log(
    allOk
      ? "PASS: captured trajectory == golden, no reclaimed-read throw"
      : "FAIL",
  );
  process.exit(allOk ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
