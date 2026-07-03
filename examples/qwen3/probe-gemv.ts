/**
 * GEMV kernel differential probe (gate 1 for the M=1 decode matmul route).
 *
 * Battery: [1,K] × [K,N] for K,N in {33, 128, 1024, 2048, 6144} (includes
 * non-multiples of tile sizes), in BOTH storage layouts the dispatcher can
 * see — NN (B contiguous [K,N]) and NT (B = transpose view of a contiguous
 * [N,K] weight, the api.linear layout) — and both dtype combos f32×f32 and
 * f32×f16(B). Each case runs the matmul twice: TORCHLETTE_GEMV=0 (tiled
 * path) and default (GEMV route), and compares BOTH against a float64 JS
 * reference built from the read-back (dequantized) operands, plus the
 * GEMV-vs-tiled diff directly. A dispatch counter asserts the GEMV kernel
 * was actually hit (and NOT hit when disabled).
 *
 * Compiled/generated plan replay is disabled so both executions re-dispatch
 * through the live routing seam (otherwise the second run could replay the
 * first run's recording and silently compare a path against itself).
 *
 * Run (repo root, GPU otherwise quiet):
 *   TORCHLETTE_STRICT_GPU=1 npx tsx examples/qwen3/probe-gemv.ts
 */

import {
  getGpuUncapturedErrorCount,
  getWebGPUInitError,
  initWebGPU,
} from "../../src/backend/webgpu";
import { getGemvDispatchCount } from "../../src/backend/webgpu/matmul/dispatch";
import { computeGemvRoute } from "../../src/backend/webgpu/matmul/gemv";
import { Torchlette } from "../../src/frontend/torchlette";

const SIZES = [33, 128, 1024, 2048, 6144];
const REL_TOL = 5e-4;

/** Deterministic LCG in [-1, 1). */
function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s / 2 ** 31 - 1);
  };
}

function maxRelErr(test: Float32Array, ref: Float64Array): number {
  let worst = 0;
  for (let i = 0; i < ref.length; i++) {
    const e = Math.abs(test[i] - ref[i]) / Math.max(1, Math.abs(ref[i]));
    if (e > worst) worst = e;
  }
  return worst;
}

function maxAbsDiff(x: Float32Array, y: Float32Array): number {
  let worst = 0;
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(x[i] - y[i]);
    if (d > worst) worst = d;
  }
  return worst;
}

async function main() {
  // Force live dispatch on every execution (see header).
  process.env.TORCHLETTE_COMPILED_PLAN = "0";
  process.env.TORCHLETTE_GENERATED_PLAN = "0";

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  let failures = 0;
  let cases = 0;
  const rng = makeRng(0xc0ffee);

  for (const layout of ["nn", "nt"] as const) {
    for (const bDtype of ["f32", "f16"] as const) {
      for (const K of SIZES) {
        for (const N of SIZES) {
          cases++;
          const aData = new Float32Array(K);
          for (let i = 0; i < K; i++) aData[i] = rng();
          const bData = new Float32Array(K * N);
          for (let i = 0; i < K * N; i++) bData[i] = rng();

          const a = api.tensorFromArray(aData, [1, K]);
          // NN: B is [K,N] contiguous. NT: weight [N,K] contiguous,
          // transposed view [K,N] → simple-transpose detection → NT dispatch.
          const bStore = api.tensorFromArray(
            bData,
            layout === "nn" ? [K, N] : [N, K],
            { dtype: bDtype },
          );
          const b =
            layout === "nn" ? bStore : bStore.transpose({ dim0: 0, dim1: 1 });

          // Host reference from the READ-BACK operand (dequantized f16 →
          // exact f16 grid values), accumulated in f64.
          const bHost = new Float32Array(await bStore.cpu());
          const ref = new Float64Array(N);
          if (layout === "nn") {
            for (let k = 0; k < K; k++) {
              const av = aData[k];
              const rowOff = k * N;
              for (let n = 0; n < N; n++) ref[n] += av * bHost[rowOff + n];
            }
          } else {
            for (let n = 0; n < N; n++) {
              let acc = 0;
              const rowOff = n * K;
              for (let k = 0; k < K; k++) acc += aData[k] * bHost[rowOff + k];
              ref[n] = acc;
            }
          }

          // Tiled reference path (route disabled).
          process.env.TORCHLETTE_GEMV = "0";
          const beforeTiled = getGemvDispatchCount();
          const outTiledT = a.matmul(b);
          const outTiled = new Float32Array(await outTiledT.cpu());
          const tiledHits = getGemvDispatchCount() - beforeTiled;

          // GEMV path (default).
          delete process.env.TORCHLETTE_GEMV;
          const beforeGemv = getGemvDispatchCount();
          const outGemvT = a.matmul(b);
          const outGemv = new Float32Array(await outGemvT.cpu());
          const gemvHits = getGemvDispatchCount() - beforeGemv;

          const errTiled = maxRelErr(outTiled, ref);
          const errGemv = maxRelErr(outGemv, ref);
          const diff = maxAbsDiff(outGemv, outTiled);

          // Expected routing from the same geometry seam the dispatcher uses
          // (some NN shapes threshold back to tiled — see computeGemvRoute).
          const expectRoute = computeGemvRoute(N, K, layout === "nt") !== null;
          const routeOk =
            tiledHits === 0 && (expectRoute ? gemvHits >= 1 : gemvHits === 0);
          const numOk = errGemv <= REL_TOL && errTiled <= REL_TOL;
          const pass = routeOk && numOk;
          if (!pass) failures++;
          console.log(
            `${pass ? "PASS" : "FAIL"} ${layout} f32x${bDtype} K=${K} N=${N}` +
              ` relErr gemv=${errGemv.toExponential(2)} tiled=${errTiled.toExponential(2)}` +
              ` |gemv-tiled|=${diff.toExponential(2)} hits=${gemvHits}/${tiledHits}${expectRoute ? "" : " (thresholded→tiled)"}`,
          );

          outTiledT.dispose();
          outGemvT.dispose();
          if (b !== bStore) b.dispose();
          bStore.dispose();
          a.dispose();
          await api.markStep();
        }
      }
    }
  }

  const gpuErrs = getGpuUncapturedErrorCount();
  if (gpuErrs > 0) {
    console.log(`FAIL: ${gpuErrs} uncaptured GPU errors`);
    failures++;
  }
  console.log(
    failures === 0
      ? `GEMV PROBE PASS (${cases} cases)`
      : `GEMV PROBE FAIL (${failures} failures / ${cases} cases)`,
  );
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
