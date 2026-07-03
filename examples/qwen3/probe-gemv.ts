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
  dispatchMatmul,
  getGpuUncapturedErrorCount,
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
  webgpuBackend,
} from "../../src/backend/webgpu";
import {
  getGemvDispatchCount,
  getGemvEpilogueDispatchCount,
} from "../../src/backend/webgpu/matmul/dispatch";
import { computeGemvRoute } from "../../src/backend/webgpu/matmul/gemv";
import type { EpilogueConfig } from "../../src/backend/webgpu/matmul/types";
import { Torchlette } from "../../src/frontend/torchlette";

const SIZES = [33, 128, 1024, 2048, 6144];
const REL_TOL = 5e-4;

/** Deterministic LCG in [-1, 1). */
function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 31 - 1;
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

  // --- Epilogue seam battery (stage 3): m=1 matmuls WITH a simple epilogue
  // (bias and/or unary activation) now route to the GEMV kernels, with the
  // epilogue expression injected at the kernel's "epilogue" seam. Exercised
  // at the dispatchMatmul level with an explicit EpilogueConfig (the same
  // level the canonical epilogue tests use — both paths get the identical
  // config): tiled epilogue path (TORCHLETTE_GEMV=0) vs GEMV seam path,
  // compared at ≤1e-4 rel tolerance, plus a f64 host reference.
  const EPI_REL_TOL = 1e-4;
  const EPI_SHAPES: Array<[number, number]> = [
    [128, 512],
    [1024, 1024],
    [2048, 33],
  ];
  const silu = (x: number) => x / (1 + Math.exp(-x));
  const geluTanh = (x: number) => {
    const inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    const c = Math.max(-10, Math.min(10, inner));
    return x * 0.5 * (1 + Math.tanh(c));
  };
  type EpiCase = {
    name: string;
    config: EpilogueConfig;
    usesBias: boolean;
    ref: (acc: number, bias: number) => number;
  };
  const epiCases: EpiCase[] = [
    {
      name: "bias",
      config: {
        ops: [{ kind: "bias", inputIndex: 0 }],
        additionalInputCount: 1,
        outputDtype: "f32",
      },
      usesBias: true,
      ref: (acc, b) => acc + b,
    },
    {
      name: "bias+gelu",
      config: {
        ops: [
          { kind: "bias", inputIndex: 0 },
          { kind: "unary", op: "gelu" },
        ],
        additionalInputCount: 1,
        outputDtype: "f32",
      },
      usesBias: true,
      ref: (acc, b) => geluTanh(acc + b),
    },
    {
      name: "bias+silu",
      config: {
        ops: [
          { kind: "bias", inputIndex: 0 },
          { kind: "unary", op: "silu" },
        ],
        additionalInputCount: 1,
        outputDtype: "f32",
      },
      usesBias: true,
      ref: (acc, b) => silu(acc + b),
    },
    {
      name: "silu",
      config: {
        ops: [{ kind: "unary", op: "silu" }],
        additionalInputCount: 0,
        outputDtype: "f32",
      },
      usesBias: false,
      ref: (acc) => silu(acc),
    },
  ];

  for (const layout of ["nn", "nt"] as const) {
    for (const epi of epiCases) {
      for (const [K, N] of EPI_SHAPES) {
        cases++;
        const aData = new Float32Array(K);
        for (let i = 0; i < K; i++) aData[i] = rng();
        const bData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) bData[i] = rng();
        const biasData = new Float32Array(N);
        for (let i = 0; i < N; i++) biasData[i] = rng();

        const a = webgpuBackend.ops.tensorFromArray(Array.from(aData), [1, K]);
        // NN: b is [K,N], transB=false. NT: b is [N,K], transB=true (the
        // api.linear weight layout after simple-transpose detection).
        const b = webgpuBackend.ops.tensorFromArray(
          Array.from(bData),
          layout === "nn" ? [K, N] : [N, K],
        );
        const bias = webgpuBackend.ops.tensorFromArray(Array.from(biasData), [
          N,
        ]);
        const transB = layout === "nt";
        const epilogueOpts = {
          epilogue: epi.config,
          epilogueInputs: epi.usesBias ? [bias] : [],
        };

        // f64 reference
        const ref = new Float64Array(N);
        for (let n = 0; n < N; n++) {
          let acc = 0;
          for (let k = 0; k < K; k++) {
            acc +=
              aData[k] *
              (layout === "nn" ? bData[k * N + n] : bData[n * K + k]);
          }
          ref[n] = epi.ref(acc, epi.usesBias ? biasData[n] : 0);
        }

        // Tiled epilogue reference path (GEMV route disabled).
        process.env.TORCHLETTE_GEMV = "0";
        const cTiled = dispatchMatmul(
          a,
          b,
          false,
          transB,
          undefined,
          epilogueOpts,
        );
        await syncWebGPU();
        const outTiled = new Float32Array(await webgpuBackend.ops.read(cTiled));

        // GEMV epilogue-seam path (default).
        delete process.env.TORCHLETTE_GEMV;
        const beforeGemv = getGemvDispatchCount();
        const beforeEpi = getGemvEpilogueDispatchCount();
        const cGemv = dispatchMatmul(
          a,
          b,
          false,
          transB,
          undefined,
          epilogueOpts,
        );
        await syncWebGPU();
        const outGemv = new Float32Array(await webgpuBackend.ops.read(cGemv));
        const gemvHits = getGemvDispatchCount() - beforeGemv;
        const epiHits = getGemvEpilogueDispatchCount() - beforeEpi;

        const errTiled = maxRelErr(outTiled, ref);
        const errGemv = maxRelErr(outGemv, ref);
        let relDiff = 0;
        for (let i = 0; i < N; i++) {
          const d =
            Math.abs(outGemv[i] - outTiled[i]) /
            Math.max(1, Math.abs(outTiled[i]));
          if (d > relDiff) relDiff = d;
        }

        // Routing: the m=1+epilogue matmul must hit the GEMV kernel with the
        // epilogue FUSED at its seam (epiHits) when the route is valid; the
        // NN K-split shapes stay tiled (epilogues can't apply to partials).
        const route = computeGemvRoute(N, K, transB);
        const expectRoute = route !== null && route.splitK === 1;
        const routeOk = expectRoute
          ? gemvHits >= 1 && epiHits >= 1
          : gemvHits === 0;
        const numOk =
          errGemv <= EPI_REL_TOL &&
          errTiled <= EPI_REL_TOL &&
          relDiff <= EPI_REL_TOL;
        const pass = routeOk && numOk;
        if (!pass) failures++;
        console.log(
          `${pass ? "PASS" : "FAIL"} ${layout} epi=${epi.name} K=${K} N=${N}` +
            ` relErr gemv=${errGemv.toExponential(2)} tiled=${errTiled.toExponential(2)}` +
            ` |gemv-tiled|rel=${relDiff.toExponential(2)} hits=${gemvHits}/${epiHits}${expectRoute ? "" : " (thresholded→tiled)"}`,
        );

        for (const t of [cTiled, cGemv, a, b, bias]) {
          (t as { destroy?: () => void }).destroy?.();
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
