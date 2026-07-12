/**
 * #95 decode-projection A100 spot-check: f16 M=1 NT projections under AMP
 * (inputCastA f32→f16). Before #95 these bailed to the tiled gemv_row config
 * (the "kernel-bound" f16 decode from #93); now they route to the GEMV NT
 * cast kernel. Measures both by toggling TORCHLETTE_GEMV.
 *
 *   TORCHLETTE_GEMV=0 npx tsx tools/t-gemv-cast-decode-bench.ts   # pre-fix (tiled)
 *   npx tsx tools/t-gemv-cast-decode-bench.ts                     # post-fix (GEMV)
 *
 * Reports wall ms/dispatch over a steady-state loop (submit+fence each iter).
 */

import { getWebGPUDevice, initWebGPU, syncWebGPU } from "../src/backend/webgpu";
import type { GPUBuffer } from "../src/backend/webgpu/gpu-types";
import { dispatchTiledMatmul } from "../src/backend/webgpu/matmul/dispatch";
import type { DType } from "../src/backend/webgpu/matmul/types";

const GPUBufferUsage = { STORAGE: 0x0080, COPY_SRC: 0x0004, COPY_DST: 0x0008 };

const WARMUP = Number(process.env.BENCH_WARMUP ?? 10);
const ITERS = Number(process.env.BENCH_ITERS ?? 200);

const SHAPES: Array<{ name: string; n: number; k: number }> = [
  { name: "qwen3 q_proj   N=2048 K=1024", n: 2048, k: 1024 },
  { name: "qwen3 o_proj   N=1024 K=2048", n: 1024, k: 2048 },
  { name: "qwen3 gate/up  N=3072 K=1024", n: 3072, k: 1024 },
  { name: "qwen3 down     N=1024 K=3072", n: 1024, k: 3072 },
  { name: "gemma2 q_proj  N=2048 K=2304", n: 2048, k: 2304 },
  { name: "gemma2 gate/up N=9216 K=2304", n: 9216, k: 2304 },
];

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error("WebGPU init failed");
  const ctx = getWebGPUDevice();
  if (!ctx) throw new Error("No WebGPU device");
  const { device, queue } = ctx;

  const gemvMode =
    process.env.TORCHLETTE_GEMV === "0" ? "TILED (pre-fix)" : "GEMV (post-fix)";
  console.log(`route: ${gemvMode}  warmup=${WARMUP} iters=${ITERS}\n`);

  const mkF32 = (len: number): GPUBuffer => {
    const buf = device.createBuffer({
      size: len * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    const f = new Float32Array(buf.getMappedRange());
    for (let i = 0; i < len; i++) f[i] = ((i % 7) - 3) / 4;
    buf.unmap();
    return buf;
  };
  const mkF16 = (len: number): GPUBuffer =>
    device.createBuffer({
      size: len * 2,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

  for (const s of SHAPES) {
    // A stored f32 (AMP activation), B f16 weight, cast-on-load A → f16 logical.
    const a = mkF32(s.k);
    const b = mkF16(s.n * s.k);
    const out = mkF16(s.n);
    const opts = {
      device,
      queue,
      a,
      b,
      out,
      m: 1,
      n: s.n,
      k: s.k,
      transB: true,
      dtype: "f16" as DType,
      dtypeB: "f16" as DType,
      inputCastA: "f32" as DType,
    };

    for (let i = 0; i < WARMUP; i++) dispatchTiledMatmul(opts);
    await syncWebGPU();

    const t0 = performance.now();
    for (let i = 0; i < ITERS; i++) dispatchTiledMatmul(opts);
    await syncWebGPU();
    const ms = (performance.now() - t0) / ITERS;
    const bytes = s.n * s.k * 2 + s.k * 4; // B (f16) + A (f32)
    const gbps = bytes / (ms / 1000) / 1e9;
    console.log(
      `${s.name}  ${ms.toFixed(3)} ms/dispatch  (${gbps.toFixed(0)} GB/s B-read)`,
    );

    a.destroy();
    b.destroy();
    out.destroy();
  }

  await syncWebGPU();
  process.exit(0);
}

main();
