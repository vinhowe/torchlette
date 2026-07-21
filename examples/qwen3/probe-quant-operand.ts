/**
 * Operand-format (StorageFormat) parity probe — phase-2 gate for task #93.
 *
 * The phase-1 probe (probe-quant-gemv.ts) exercised the raw eager
 * dispatchQuantizedGemvNT kernel. THIS probe exercises the full OPERAND PATH:
 * a quantized weight is created via api.runtime.createQuantizedWeight and fed
 * to the ordinary `api.linear(x, w)` — the frontend is format-BLIND. It must
 *
 *  (1) produce the same result as an f32 control weight (the SAME dequantized
 *      values) through the SAME api.linear — kernel exactness, invisible route;
 *  (2) route M=1 to the fused-dequant GEMV (getQuantGemvDispatchCount++),
 *      route M>1 to the explicit dequant + stock matmul
 *      (getDequantI8DispatchCount++) — the capability seam, keyed on format;
 *  (3) NEVER let the packed format leak above the backend — the control path
 *      and the quant path build IDENTICAL lazy graphs (same api.linear call).
 *
 * Run: TORCHLETTE_STRICT_GPU=1 npx tsx examples/qwen3/probe-quant-operand.ts
 */

import {
  getDequantI8DispatchCount,
  getGpuUncapturedErrorCount,
  getQuantGemvDispatchCount,
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
} from "../../src/backend/webgpu";
import { resolveWeightFormat } from "../../src/backend/types";
import { Torchlette } from "../../src/frontend/torchlette";
import { dequantizeToF32, quantizeLinearWeight } from "../../src/backend/quantize";

const SHAPES: Array<[number, number]> = [
  // [N, K]
  [128, 128],
  [512, 1024],
  [2048, 6144],
];
const GROUP_SIZES = [64, 128];
// Parity is measured as max-abs drift NORMALIZED by the output's own scale
// (RMS), NOT per-element relative error — the latter explodes near zeros where
// random-data cancellation makes the denominator tiny (a numerics artifact, not
// a quant error). Tight kernel-exactness (per-element, f16-grid control) is the
// job of probe-quant-gemv.ts; this probe proves the OPERAND ROUTE + agreement.
const DRIFT_TOL = 5e-3; // max-abs drift / output RMS

function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 31 - 1;
  };
}

async function main() {
  process.env.TORCHLETTE_COMPILED_PLAN = "0";

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  let failures = 0;
  let cases = 0;

  for (const [N, K] of SHAPES) {
    for (const G of GROUP_SIZES) {
      if (K % G !== 0) continue;
      const rng = makeRng(0x51 ^ N ^ (K << 3) ^ (G << 7));
      const wf32 = new Float32Array(N * K);
      for (let i = 0; i < wf32.length; i++) wf32[i] = rng() * 0.1;
      const q = quantizeLinearWeight(wf32, N, K, G);
      // The f32 control weight is the EXACT dequantized values the kernel sees.
      const deq = dequantizeToF32(q);

      const format = resolveWeightFormat(G === 64 ? "int8-64" : "int8-128", "f16");
      const qWeight = await api.createQuantizedWeight(
        q.packed,
        q.scales,
        N,
        K,
        format,
      );
      // Control: an ordinary f32 weight [N, K] holding the dequantized values.
      const ctrlWeight = api.tensorFromArray(deq, [N, K], { dtype: "f32" });

      for (const M of [1, 4]) {
        const xArr = new Float32Array(M * K);
        for (let i = 0; i < xArr.length; i++) xArr[i] = rng();
        const x = api.tensorFromArray(xArr, [M, K], { dtype: "f32" });

        const qBefore = getQuantGemvDispatchCount();
        const dBefore = getDequantI8DispatchCount();
        // FORMAT-BLIND call: identical to the control's api.linear.
        const yQ = api.noGrad(() => api.linear(x, qWeight, null));
        const yQArr = new Float32Array(await yQ.cpu());
        const qHits = getQuantGemvDispatchCount() - qBefore;
        const dHits = getDequantI8DispatchCount() - dBefore;

        const yC = api.noGrad(() => api.linear(x, ctrlWeight, null));
        const yCArr = new Float32Array(await yC.cpu());

        // Numeric parity: max-abs drift normalized by output RMS.
        let maxAbs = 0;
        let sumSq = 0;
        for (let i = 0; i < yQArr.length; i++) {
          maxAbs = Math.max(maxAbs, Math.abs(yQArr[i] - yCArr[i]));
          sumSq += yCArr[i] * yCArr[i];
        }
        const rms = Math.sqrt(sumSq / Math.max(1, yQArr.length));
        const drift = maxAbs / Math.max(1e-6, rms);
        // Route: M=1 → GEMV quant; M>1 → explicit dequant.
        const routeOk = M === 1 ? qHits === 1 && dHits === 0 : dHits === 1;
        const parityOk = drift <= DRIFT_TOL;
        cases++;
        if (!parityOk || !routeOk) failures++;
        console.log(
          `${parityOk && routeOk ? "PASS" : "FAIL"} N=${N} K=${K} G=${G} M=${M} ` +
            `drift=${drift.toExponential(2)} qHits=${qHits} dHits=${dHits}`,
        );
        await api.markStep();
      }
    }
  }

  await syncWebGPU();
  const gpuErrs = getGpuUncapturedErrorCount();
  if (gpuErrs > 0) {
    console.log(`PROBE FAIL: ${gpuErrs} GPU uncaptured errors`);
    failures++;
  }
  if (failures === 0) {
    console.log(`QUANT OPERAND PROBE PASS (${cases} cases)`);
  } else {
    console.log(`PROBE FAIL: ${failures}/${cases} cases`);
  }
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("PROBE FAIL:", e);
  process.exit(1);
});
