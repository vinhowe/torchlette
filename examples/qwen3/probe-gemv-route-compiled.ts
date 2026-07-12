/**
 * task #95 follow-up: does the M=1 decode projection GEMV route ENGAGE end to
 * end under the DEFAULT compiled+generated plan (the path the browser decode
 * runs), or does a compiled-plan DECISION POINT (build-from-IR capture /
 * directive detection / stream generator) route it to the tiled kernel before
 * variant selection's #95 applicability ever applies?
 *
 * Shape: the real browser decode projection — mixed f32 activation × f16 weight
 * (weightDtype:"f16", NO autocast), [1,K] @ [N,K]^T, M=1. The SAME-shape
 * api.linear runs REPEATEDLY so the compiled plan builds + replays (crosses the
 * 2nd-exec activation threshold — CLAUDE.md Corollary 2).
 *
 * TWO route-engagement signals:
 *  - loweredGemvHits (getGemvDispatchCount): ticks in dispatchTiledMatmul — the
 *    variant seam. Only the FIRST (lowered) execution ticks it; it is
 *    REPLAY-BLIND once the template cuts over to the generated stream.
 *  - generatedGemv (getGeneratedGemvDispatchCount): ticks when the STREAM
 *    GENERATOR bakes a `_gemv` dispatch into the compiled plan — the signal
 *    that SURVIVES the cutover. generatedGemv == 0 = the route was bypassed at a
 *    compiled-plan decision point (the #93/#95 class). Negative control:
 *    TORCHLETTE_GEMV=0 ⇒ both counts 0.
 *
 * Run:
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim \
 *     TORCHLETTE_STRICT_GPU=1 npx tsx examples/qwen3/probe-gemv-route-compiled.ts
 */

import {
  getGpuUncapturedErrorCount,
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
} from "../../src/backend/webgpu";
import { getGemvDispatchCount } from "../../src/backend/webgpu/matmul/dispatch";
import { getGeneratedGemvDispatchCount } from "../../src/executor/stream-generate";
import { Torchlette } from "../../src/frontend/torchlette";

function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 31 - 1;
  };
}

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  // Gemma-2-2b-ish decode projection shape: [1,K] @ [N,K]^T.
  const N = 2048;
  const K = 2304;
  const rng = makeRng(0x95);

  // f16 weight (persistent) — the REAL browser decode dtype config: mixed
  // f32-activation x f16-weight matmul (weightDtype:"f16", NO autocast).
  const wArr = new Float32Array(N * K);
  for (let i = 0; i < wArr.length; i++) wArr[i] = rng() * 0.02;
  const weight = api.tensorFromArray(wArr, [N, K], { dtype: "f16" });

  // f32 control weight for a numeric cross-check.
  const ctrlWeight = api.tensorFromArray(wArr, [N, K], { dtype: "f32" });

  let maxDrift = 0;
  let failures = 0;
  let loweredHits = 0;
  const genBaseline = getGeneratedGemvDispatchCount();
  const ITERS = 12; // cross the 2nd-exec compiled-capture threshold
  for (let it = 0; it < ITERS; it++) {
    const xArr = new Float32Array(1 * K);
    for (let i = 0; i < xArr.length; i++) xArr[i] = rng();
    const x = api.tensorFromArray(xArr, [1, K], { dtype: "f32" });

    const gemvBefore = getGemvDispatchCount();
    // Mixed f32 x @ f16 W^T — the browser decode projection (no autocast).
    const y = api.noGrad(() => api.linear(x, weight, null));
    const yArr = new Float32Array(await y.cpu());
    loweredHits += getGemvDispatchCount() - gemvBefore;

    // f32 control (tiled reference).
    const yC = api.noGrad(() => api.linear(x, ctrlWeight, null));
    const yCArr = new Float32Array(await yC.cpu());

    let maxAbs = 0,
      sumSq = 0;
    for (let i = 0; i < yArr.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(yArr[i] - yCArr[i]));
      sumSq += yCArr[i] * yCArr[i];
    }
    const rms = Math.sqrt(sumSq / Math.max(1, yArr.length));
    const drift = maxAbs / Math.max(1e-6, rms);
    maxDrift = Math.max(maxDrift, drift);

    console.log(`iter=${it} drift=${drift.toExponential(2)}`);
    await api.markStep();
  }

  await syncWebGPU();
  const gpuErrs = getGpuUncapturedErrorCount();
  if (gpuErrs > 0) {
    console.log(`GPU_ERRS=${gpuErrs}`);
    failures++;
  }
  // Route-engagement, TWO independent signals:
  //  - loweredHits>0: the first (lowered) execution routed GEMV (variant seam).
  //  - generatedGemv>0: the compiled/generated plan BAKED a GEMV dispatch —
  //    the signal that survives the compiled-plan activation threshold (the
  //    lowered counter reads 0 once a template cuts over, even when GEMV runs).
  const generatedGemv = getGeneratedGemvDispatchCount() - genBaseline;
  console.log(`loweredGemvHits=${loweredHits} generatedGemv=${generatedGemv}`);
  if (loweredHits === 0) {
    console.log("PROBE FAIL: lowered GEMV route never engaged (variant seam)");
    failures++;
  }
  if (generatedGemv === 0) {
    console.log(
      "PROBE FAIL: compiled/generated plan never baked a GEMV dispatch — route bypassed at cutover (#93/#95 class)",
    );
    failures++;
  }
  if (maxDrift > 1e-2) {
    console.log(`PROBE FAIL: drift ${maxDrift.toExponential(2)} too high`);
    failures++;
  }
  console.log(
    failures === 0
      ? `GEMV COMPILED PROBE PASS maxDrift=${maxDrift.toExponential(2)} lowered=${loweredHits} generated=${generatedGemv}`
      : `PROBE FAIL: ${failures} maxDrift=${maxDrift.toExponential(2)}`,
  );
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.log("PROBE FAIL:", e?.stack || String(e));
  process.exit(1);
});
