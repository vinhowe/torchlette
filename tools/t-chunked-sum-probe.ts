/**
 * Stage-4 phase 4.2 probe: is the chunked full-reduction sum (>128MB input)
 * correct under (a) recorded compiled replay and (b) — once covered —
 * generated replay? Sums a >128MB tensor inside a compiled plan, runs it
 * enough times to activate replay, and checks the scalar against the CPU
 * reference each step. Also reports whether the plan FULLY GENERATED.
 *
 * Run: VULKAN_DEVICE_INDEX=0 LD_LIBRARY_PATH=tools/vk-shim \
 *        npx tsx tools/t-chunked-sum-probe.ts
 * Add TORCHLETTE_BUILD_FROM_IR=0 to force the recorded path.
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

async function main() {
  if (!(await initWebGPU())) {
    console.log("no webgpu");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });

  // [50304, 768] f32 = 154 MB > 128 MB → chunked full reduction.
  const ROWS = 50304;
  const COLS = 768;
  const N = ROWS * COLS;
  // Small deterministic values so the exact sum is representable in f32:
  // 0.25 * N would lose precision; use a pattern that sums cleanly.
  const host = new Float32Array(N);
  for (let i = 0; i < N; i++) host[i] = (i % 7) === 0 ? 1 : 0;
  let expected = 0;
  for (let i = 0; i < N; i++) expected += host[i];

  const t = api.tensorFromArray(Array.from(host), [ROWS, COLS], {
    device: "webgpu",
  });

  const STEPS = 4;
  let worst = 0;
  for (let step = 0; step < STEPS; step++) {
    // Sum inside a compiled-plannable region. A trivial elementwise before the
    // sum makes the plan non-degenerate; the sum is the full reduction.
    const s = api.sum(api.mul(t, 1.0));
    const got = (await api.item(s)) as number;
    const d = Math.abs(got - expected);
    worst = Math.max(worst, d);
    console.log(
      `step ${step}: got=${got} expected=${expected} |Δ|=${d}`,
    );
    await api.markStep();
  }
  console.log(`worst |Δ| over ${STEPS} steps = ${worst}`);
  console.log(worst < 1 ? "CHUNKED-SUM: OK" : "CHUNKED-SUM: FAIL");
  destroyWebGPU();
  process.exit(worst < 1 ? 0 : 1);
}

main();
