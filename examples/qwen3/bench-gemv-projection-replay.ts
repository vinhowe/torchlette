/**
 * Decode-projection replay timing (relative, honest): a stack of M=1 mixed
 * f32×f16 api.linear projections (gemma2/qwen3 decode shapes) run REPEATEDLY so
 * the compiled/generated plan replays, timed GEMV-on (default) vs GEMV-off
 * (TORCHLETTE_GEMV=0). Measures the per-token matmul-family delta the #95 route
 * actually buys IN THE COMPILED PLAN (where the browser decode runs) — not a
 * per-dispatch microbench. Run both:
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim npx tsx examples/qwen3/bench-gemv-projection-replay.ts
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim TORCHLETTE_GEMV=0 npx tsx examples/qwen3/bench-gemv-projection-replay.ts
 */
import {
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
} from "../../src/backend/webgpu";
import { getGeneratedGemvDispatchCount } from "../../src/executor/stream-generate";
import { Torchlette } from "../../src/frontend/torchlette";

function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 31 - 1;
  };
}

// A few real gemma2-2b/qwen3 decode projection [N,K] shapes (per layer).
const PROJ: Array<[number, number]> = [
  [2048, 2304], // q_proj
  [1024, 2304], // k/v_proj
  [2304, 2048], // o_proj
  [9216, 2304], // gate/up
  [2304, 9216], // down
];

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const rng = makeRng(0xbee);

  const weights = PROJ.map(([N, K]) => {
    const w = new Float32Array(N * K);
    for (let i = 0; i < w.length; i++) w[i] = rng() * 0.02;
    return { W: api.tensorFromArray(w, [N, K], { dtype: "f16" }), N, K };
  });

  const WARMUP = 8;
  const ITERS = 60;
  const walls: number[] = [];
  const genBefore = getGeneratedGemvDispatchCount();
  for (let it = 0; it < WARMUP + ITERS; it++) {
    const t0 = performance.now();
    api.noGrad(() => {
      for (const { W, K } of weights) {
        const xArr = new Float32Array(K);
        for (let i = 0; i < xArr.length; i++) xArr[i] = rng();
        const x = api.tensorFromArray(xArr, [1, K], { dtype: "f32" });
        const y = api.linear(x, W, null);
        y.detach();
      }
    });
    // Fence so the wall reflects real GPU work (decode is latency-bound).
    // eslint-disable-next-line no-await-in-loop
    await syncWebGPU();
    const dt = performance.now() - t0;
    if (it >= WARMUP) walls.push(dt);
    // eslint-disable-next-line no-await-in-loop
    await api.markStep();
  }
  const avg = walls.reduce((a, b) => a + b, 0) / walls.length;
  const sorted = walls.slice().sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const gemvOn = process.env.TORCHLETTE_GEMV !== "0";
  console.log(
    `GEMV=${gemvOn ? "on" : "off"} projections/iter=${weights.length} ` +
      `avg=${avg.toFixed(3)}ms median=${median.toFixed(3)}ms ` +
      `generatedGemv=${getGeneratedGemvDispatchCount() - genBefore}`,
  );
  process.exit(0);
}

main().catch((e) => {
  console.log("BENCH FAIL:", e?.stack || String(e));
  process.exit(1);
});
