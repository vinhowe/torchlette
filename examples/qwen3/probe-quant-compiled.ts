/**
 * task #93 follow-up: quantized M=1 decode matmul through the DEFAULT
 * compiled/generated-plan path (GENERATED_PLAN on) — the path the operand
 * gate explicitly disabled (TORCHLETTE_COMPILED_PLAN/GENERATED_PLAN=0), which
 * is why the "int8 decode not engaging" bug shipped.
 *
 * The same-shape M=1 quantized `api.linear` is executed REPEATEDLY so the
 * compiled plan builds + replays (and, by default, stage-4 build-from-IR
 * covers it). The pre-fix bug: the build-without-execution matmul capture
 * derived IR-stub metadata WITHOUT the packed StorageFormat, so planBareMatmul
 * planned a plain f16 tiled matmul over the PACKED buffer → the plan claimed
 * "fully covered", ran build-from-IR, and NEVER touched the seam → NaN and
 * qHits=0 (the seam bypassed). We assert the compiled trajectory equals the
 * f32 control across many iterations, with the quant GEMV route engaged.
 *
 * Run: TORCHLETTE_STRICT_GPU=1 npx tsx examples/qwen3/probe-quant-compiled.ts
 */

import {
  dequantizeToF32,
  quantizeLinearWeight,
} from "../../src/backend/quantize";
import { resolveWeightFormat } from "../../src/backend/types";
import {
  getGpuUncapturedErrorCount,
  getQuantGemvDispatchCount,
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
} from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";

function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 31 - 1;
  };
}

async function main() {
  // DEFAULT paths ON (the whole point): compiled + generated plans engage.
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  const N = 2048;
  const K = 2304; // Gemma-2-2b hidden; K % 4 == 0, K % 64 == 0
  const G = 64;
  const rng = makeRng(0x93);
  const wf32 = new Float32Array(N * K);
  for (let i = 0; i < wf32.length; i++) wf32[i] = rng() * 0.05;
  const q = quantizeLinearWeight(wf32, N, K, G);
  const deq = dequantizeToF32(q);
  const format = resolveWeightFormat("int8-64", "f16");
  const qWeight = await api.createQuantizedWeight(
    q.packed,
    q.scales,
    N,
    K,
    format,
  );
  const ctrlWeight = api.tensorFromArray(deq, [N, K], { dtype: "f32" });

  let maxDrift = 0;
  let failures = 0;
  let totalQHits = 0;
  const ITERS = 12; // enough to cross the 2nd-exec compiled-capture threshold
  for (let it = 0; it < ITERS; it++) {
    const xArr = new Float32Array(1 * K);
    for (let i = 0; i < xArr.length; i++) xArr[i] = rng();
    const x = api.tensorFromArray(xArr, [1, K], { dtype: "f32" });

    const qBefore = getQuantGemvDispatchCount();
    const yQ = api.noGrad(() => api.linear(x, qWeight, null));
    const yQArr = new Float32Array(await yQ.cpu());
    const qHits = getQuantGemvDispatchCount() - qBefore;
    totalQHits += qHits;

    const yC = api.noGrad(() => api.linear(x, ctrlWeight, null));
    const yCArr = new Float32Array(await yC.cpu());

    let maxAbs = 0,
      sumSq = 0;
    for (let i = 0; i < yQArr.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(yQArr[i] - yCArr[i]));
      sumSq += yCArr[i] * yCArr[i];
    }
    const rms = Math.sqrt(sumSq / Math.max(1, yQArr.length));
    const drift = maxAbs / Math.max(1e-6, rms);
    maxDrift = Math.max(maxDrift, drift);
    const ok2 = drift <= 5e-3;
    if (!ok2) failures++;
    console.log(
      `iter=${it} drift=${drift.toExponential(2)} qHits=${qHits} ${ok2 ? "" : "<<FAIL"}`,
    );
    await api.markStep();
  }

  await syncWebGPU();
  const gpuErrs = getGpuUncapturedErrorCount();
  if (gpuErrs > 0) {
    console.log(`GPU_ERRS=${gpuErrs}`);
    failures++;
  }
  // The quant GEMV route MUST engage at least during the lowered recording
  // passes (0 across the whole run = the seam was bypassed — the shipped bug).
  if (totalQHits === 0) {
    console.log("PROBE FAIL: quant GEMV never dispatched (seam bypassed)");
    failures++;
  }
  console.log(
    failures === 0
      ? `QUANT COMPILED PROBE PASS maxDrift=${maxDrift.toExponential(2)} qHits=${totalQHits}`
      : `PROBE FAIL: ${failures} maxDrift=${maxDrift.toExponential(2)}`,
  );
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.log("PROBE FAIL:", e?.stack || String(e));
  process.exit(1);
});
