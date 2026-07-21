/**
 * Quantized (int8-grouped) GEMV kernel differential probe — gate 1 for
 * weight-only quantization (docs/quantization-design.md).
 *
 * KERNEL-LEVEL EXACTNESS (isolates kernel arithmetic from quant error): we
 * quantize a random weight, then dequantize it back to the EXACT f16 grid
 * values the kernel will see (dequantizeToF32 → f16-rounded). Those f16 values
 * feed the existing f16 GEMV control path; the packed (q, scale) feed the new
 * quant kernel. Both compute sum(a · w_f16) in f32 — so they must agree to f32
 * rounding noise. This proves the unpack/dequant/accumulate is correct
 * independent of how lossy int8 is.
 *
 * A f64 host reference over the same dequantized values bounds both paths.
 * Dispatch counters assert the quant kernel was actually hit.
 *
 * Run (repo root, GPU otherwise quiet):
 *   CUDA_VISIBLE_DEVICES=0 TORCHLETTE_STRICT_GPU=1 npx tsx examples/qwen3/probe-quant-gemv.ts
 */

import {
  dispatchMatmul,
  dispatchQuantizedGemvNT,
  getGpuUncapturedErrorCount,
  getQuantGemvDispatchCount,
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
  webgpuBackend,
} from "../../src/backend/webgpu";
import { requireContext } from "../../src/backend/webgpu/gpu-context";
import { GPUBufferUsage } from "../../src/backend/webgpu/gpu-types";
import { dequantizeToF32, quantizeLinearWeight } from "../../tools/quantize";

const SHAPES: Array<[number, number]> = [
  // [N, K] — decode projections + lm_head-scale. K multiple of 4 and G.
  [128, 128],
  [512, 1024],
  [1024, 2048],
  [2048, 6144],
  [151936, 2048], // lm_head-scale N (2D row grid)
];
const GROUP_SIZES = [64, 128];

function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 31 - 1;
  };
}

/** Upload a Uint32Array as a storage buffer. */
function uploadU32(data: Uint32Array): GPUBuffer {
  const ctx = requireContext();
  const buf = ctx.device.createBuffer({
    size: data.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  ctx.device.queue.writeBuffer(
    buf,
    0,
    data.buffer,
    data.byteOffset,
    data.byteLength,
  );
  return buf;
}

async function main() {
  process.env.TORCHLETTE_COMPILED_PLAN = "0";

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");

  let failures = 0;
  let cases = 0;
  const rng = makeRng(0x51a7ee);

  for (const G of GROUP_SIZES) {
    for (const [N, K] of SHAPES) {
      if (K % G !== 0) continue;
      cases++;
      // Random weight [N, K] and activation [1, K].
      const wData = new Float32Array(N * K);
      for (let i = 0; i < N * K; i++) wData[i] = rng();
      const aData = new Float32Array(K);
      for (let i = 0; i < K; i++) aData[i] = rng();

      // Quantize, then dequantize on the host to EXACTLY the values the GPU
      // computes: (qi/127) · f16(scale). The quant kernel does the same in f32
      // (unpack4x8snorm × unpacked-f16 scale), so a f32 control matmul over
      // `deq` must agree to f32 noise — isolating kernel arithmetic.
      const q = quantizeLinearWeight(wData, N, K, G);
      const deq = dequantizeToF32(q);

      // f64 host reference over the dequantized values (NT: y[n] = sum_k a·w[n,k]).
      const ref = new Float64Array(N);
      for (let n = 0; n < N; n++) {
        let acc = 0;
        const base = n * K;
        for (let k = 0; k < K; k++) acc += aData[k] * deq[base + k];
        ref[n] = acc;
      }

      // --- Control: f32 dequantized weight through the standard NT GEMV. ---
      const a = webgpuBackend.ops.tensorFromArray(aData, [1, K]);
      // Weight stored [N,K]; transB=true gives the NT (api.linear) route —
      // same layout the quant kernel packs.
      const wStore = webgpuBackend.ops.tensorFromArray(deq, [N, K]);
      const cControl = dispatchMatmul(a, wStore, false, true);
      await syncWebGPU();
      const outControl = new Float32Array(
        await webgpuBackend.ops.read(cControl),
      );

      // --- Quant kernel: packed int8 + scales. ---
      const packedBuf = uploadU32(q.packed);
      const scalesBuf = uploadU32(
        new Uint32Array(
          q.scales.buffer,
          q.scales.byteOffset,
          q.scales.length / 2,
        ),
      );
      const ctx = requireContext();
      const outBuf = ctx.device.createBuffer({
        size: N * 4,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
      const beforeQ = getQuantGemvDispatchCount();
      dispatchQuantizedGemvNT(
        ctx.device,
        (a as { buffer: GPUBuffer }).buffer,
        packedBuf,
        scalesBuf,
        outBuf,
        N,
        K,
        G,
        "f32",
      );
      await syncWebGPU();
      const qHits = getQuantGemvDispatchCount() - beforeQ;
      // Read back outBuf.
      const readBuf = ctx.device.createBuffer({
        size: N * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = ctx.device.createCommandEncoder();
      enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, N * 4);
      ctx.device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(1 /* READ */);
      const outQuant = new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();

      // Compare quant-vs-control (kernel exactness) and both vs f64 ref.
      let maxKernelDiff = 0;
      let maxRefErrQ = 0;
      let maxRefErrC = 0;
      for (let n = 0; n < N; n++) {
        const kd =
          Math.abs(outQuant[n] - outControl[n]) /
          Math.max(1, Math.abs(outControl[n]));
        if (kd > maxKernelDiff) maxKernelDiff = kd;
        const rq =
          Math.abs(outQuant[n] - ref[n]) / Math.max(1, Math.abs(ref[n]));
        if (rq > maxRefErrQ) maxRefErrQ = rq;
        const rc =
          Math.abs(outControl[n] - ref[n]) / Math.max(1, Math.abs(ref[n]));
        if (rc > maxRefErrC) maxRefErrC = rc;
      }
      // Kernel exactness: the control matmuls the SAME f32 dequant values the
      // quant kernel reconstructs on-the-fly (unpack4x8snorm × unpacked-f16
      // scale). The only difference is f32 accumulation order → f32 rounding
      // noise. Measured ~1e-5 across all shapes; 5e-4 is a comfortable ceiling.
      const KERNEL_TOL = 5e-4;
      const REF_TOL = 5e-4;
      const numOk =
        maxKernelDiff <= KERNEL_TOL &&
        maxRefErrQ <= REF_TOL &&
        maxRefErrC <= REF_TOL;
      const routeOk = qHits === 1;
      const pass = numOk && routeOk;
      if (!pass) failures++;
      console.log(
        `${pass ? "PASS" : "FAIL"} N=${N} K=${K} G=${G}` +
          ` |quant-ctrl|rel=${maxKernelDiff.toExponential(2)}` +
          ` refErr q=${maxRefErrQ.toExponential(2)} ctrl=${maxRefErrC.toExponential(2)}` +
          ` hits=${qHits}`,
      );

      (cControl as { destroy?: () => void }).destroy?.();
      (a as { destroy?: () => void }).destroy?.();
      (wStore as { destroy?: () => void }).destroy?.();
      packedBuf.destroy();
      scalesBuf.destroy();
      outBuf.destroy();
      readBuf.destroy();
    }
  }

  const gpuErrs = getGpuUncapturedErrorCount();
  if (gpuErrs > 0) {
    console.log(`FAIL: ${gpuErrs} uncaptured GPU errors`);
    failures++;
  }
  console.log(
    failures === 0
      ? `QUANT GEMV PROBE PASS (${cases} cases)`
      : `QUANT GEMV PROBE FAIL (${failures} failures / ${cases} cases)`,
  );
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
