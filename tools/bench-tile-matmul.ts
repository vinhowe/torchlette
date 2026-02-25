/**
 * Benchmark: tile-IR matmul vs production codegen on DistilGPT-2 shapes.
 *
 * Compares GPU kernel time for both codegen paths on each shape,
 * using the same tile config and dispatch parameters.
 *
 * Usage: npx tsx tools/bench-tile-matmul.ts
 * Env: BENCH_WARMUP=N BENCH_ITERS=N
 */

import {
  getWebGPUDevice,
  initWebGPU,
  syncWebGPU,
} from "../src/backend/webgpu";
import {
  classifyShape,
  type DType,
  dispatchTiledMatmul,
  getDefaultConfigForShape,
  getWorkgroupSize,
  type MatmulKernelConfig,
  type ShapeClass,
} from "../src/backend/webgpu/matmul";
import {
  generateTiledMatmulShader,
  getShaderCacheKey,
  generateKSplitReductionShader,
  type CodegenOptions,
  type EpilogueConfig,
} from "../src/backend/webgpu/matmul/codegen";
import {
  generateTiledMatmulShaderTileIR,
  generateKSplitReductionShaderTileIR,
} from "../src/backend/webgpu/matmul/tile-matmul";
import { getTransposeMode } from "../src/backend/webgpu/matmul/types";

const GPUBufferUsage = {
  STORAGE: 0x0080,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
};

// ---------------------------------------------------------------------------
// DistilGPT-2 shapes at 512 tokens
// ---------------------------------------------------------------------------

interface MatmulShape {
  name: string;
  m: number;
  n: number;
  k: number;
  transA: boolean;
  transB: boolean;
  epilogue?: EpilogueConfig;
  batchSize?: number;
  count: number; // how many times this shape appears per step
}

const SHAPES: MatmulShape[] = [
  // Forward — epilogue matmuls (cast+bias+activation)
  { name: "QKV proj (fwd)",       m: 512, n: 2304, k: 768,  transA: false, transB: false, count: 6,
    epilogue: { ops: [{ kind: "cast", toDtype: "f32" }, { kind: "bias", inputIndex: 0 }], additionalInputCount: 1, outputDtype: "f32" } },
  { name: "attn out (fwd)",       m: 512, n: 768,  k: 768,  transA: false, transB: false, count: 6,
    epilogue: { ops: [{ kind: "cast", toDtype: "f32" }, { kind: "bias", inputIndex: 0 }], additionalInputCount: 1, outputDtype: "f32" } },
  { name: "MLP fc1 gelu (fwd)",   m: 512, n: 3072, k: 768,  transA: false, transB: false, count: 6,
    epilogue: { ops: [{ kind: "cast", toDtype: "f32" }, { kind: "bias", inputIndex: 0 }, { kind: "gelu" }, { kind: "cast", toDtype: "f16" }], additionalInputCount: 1, outputDtype: "f16" } },
  { name: "MLP fc2 (fwd)",        m: 512, n: 768,  k: 3072, transA: false, transB: false, count: 6,
    epilogue: { ops: [{ kind: "cast", toDtype: "f32" }, { kind: "bias", inputIndex: 0 }], additionalInputCount: 1, outputDtype: "f32" } },
  { name: "lm_head (fwd)",        m: 512, n: 50304,k: 768,  transA: false, transB: false, count: 1,
    epilogue: { ops: [{ kind: "cast", toDtype: "f32" }, { kind: "bias", inputIndex: 0 }, { kind: "add", inputIndex: 1 }], additionalInputCount: 2, outputDtype: "f32" } },

  // Backward — bare matmuls (no epilogue)
  { name: "attn out dX (bwd)",    m: 512, n: 768,  k: 768,  transA: false, transB: true,  count: 6 },
  { name: "attn out dW (bwd)",    m: 768, n: 768,  k: 512,  transA: true,  transB: false, count: 6 },
  { name: "MLP fc1 dX (bwd)",     m: 512, n: 768,  k: 3072, transA: false, transB: true,  count: 6 },
  { name: "MLP fc1 dW (bwd)",     m: 3072,n: 768,  k: 512,  transA: true,  transB: false, count: 6 },
  { name: "MLP fc2 dX (bwd)",     m: 512, n: 3072, k: 768,  transA: false, transB: true,  count: 6 },
  { name: "MLP fc2 dW (bwd)",     m: 768, n: 3072, k: 512,  transA: true,  transB: false, count: 6 },
  { name: "lm_head dX (bwd)",     m: 512, n: 768,  k: 50304,transA: false, transB: true,  count: 1 },
  { name: "lm_head dW (bwd)",     m: 50304,n: 768, k: 512,  transA: true,  transB: false, count: 1 },
];

// ---------------------------------------------------------------------------
// Benchmark infrastructure
// ---------------------------------------------------------------------------

async function benchmarkKernel(
  device: any,
  queue: any,
  shape: MatmulShape,
  useTileIR: boolean,
  warmup: number,
  iters: number,
): Promise<number> {
  const { m, n, k, transA, transB, epilogue, batchSize = 1 } = shape;
  const dtype: DType = "f16";
  const hasEpilogue = !!epilogue;
  const shapeClass = classifyShape(m, n, k, batchSize);
  const config = getDefaultConfigForShape(shapeClass, hasEpilogue);
  const transposeMode = getTransposeMode(transA, transB);

  // Check K-split eligibility (same logic as dispatch.ts)
  const wgSize = getWorkgroupSize(config);
  const workgroupsX = Math.ceil(n / config.tileN);
  const workgroupsY = Math.ceil(m / config.tileM);
  const baseWorkgroups = workgroupsX * workgroupsY;
  let kSplitFactor = 0;
  if (batchSize === 1 && !hasEpilogue && baseWorkgroups < 64 && k >= 512) {
    kSplitFactor = Math.min(
      Math.ceil(128 / baseWorkgroups),
      Math.floor(k / config.tileK),
      32,
    );
  }

  // Build codegen options
  const codegenOptions: CodegenOptions = {
    config,
    transposeMode,
    dtype,
    dtypeB: dtype,
    batched: batchSize > 1,
    ...(epilogue ? { epilogue } : {}),
    ...(kSplitFactor >= 2 ? { kSplit: kSplitFactor } : {}),
  };

  // Generate WGSL
  const shaderCode = useTileIR
    ? generateTiledMatmulShaderTileIR(codegenOptions)
    : generateTiledMatmulShader(codegenOptions);

  // Create pipeline
  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  // K-split reduction pipeline (if needed)
  let reductionPipeline: any = null;
  if (kSplitFactor >= 2) {
    const outputDtype = epilogue?.outputDtype ?? (dtype === "f32" ? "f32" : "f32");
    const reduceCode = useTileIR
      ? generateKSplitReductionShaderTileIR(kSplitFactor, outputDtype as DType)
      : generateKSplitReductionShader(kSplitFactor, outputDtype as DType);
    const reduceModule = device.createShaderModule({ code: reduceCode });
    reductionPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: reduceModule, entryPoint: "main" },
    });
  }

  // Allocate buffers
  const bytesPerElement = dtype === "f16" ? 2 : 4;
  const aBuffer = device.createBuffer({
    size: Math.max(16, m * k * bytesPerElement),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bBuffer = device.createBuffer({
    size: Math.max(16, k * n * bytesPerElement),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outBytesPerElement = epilogue?.outputDtype === "f16" ? 2 : 4;
  const outBuffer = device.createBuffer({
    size: Math.max(16, m * n * outBytesPerElement),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Epilogue extra input buffers
  const epilogueBuffers: any[] = [];
  if (epilogue) {
    for (let i = 0; i < epilogue.additionalInputCount; i++) {
      // Bias is 1D (column count), binary epilogue is full M×N
      const isRowwise = epilogue.ops.some(o => (o.kind === "bias" || o.kind === "add" || o.kind === "mul") && o.inputIndex === i);
      const size = isRowwise
        ? (epilogue.ops.find(o => o.kind === "bias" && o.inputIndex === i) ? n : m * n)
        : m * n;
      epilogueBuffers.push(device.createBuffer({
        size: Math.max(16, size * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }));
    }
  }

  // K-split temp buffer
  let kSplitBuffer: any = null;
  if (kSplitFactor >= 2) {
    kSplitBuffer = device.createBuffer({
      size: Math.max(16, kSplitFactor * m * n * 4),
      usage: GPUBufferUsage.STORAGE,
    });
  }

  // Uniform params
  const lda = transA ? m : k;
  const ldb = transB ? k : n;
  const ldc = n;
  const paramsData = new ArrayBuffer(48);
  const u32View = new Uint32Array(paramsData);
  const f32View = new Float32Array(paramsData);
  u32View[0] = m; u32View[1] = n; u32View[2] = k;
  u32View[3] = lda; u32View[4] = ldb; u32View[5] = ldc;
  f32View[6] = 1.0; // alpha
  u32View[7] = batchSize;
  u32View[8] = m * k; // batchStrideA
  u32View[9] = k * n; // batchStrideB
  u32View[10] = m * n; // batchStrideC

  const paramsBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  queue.writeBuffer(paramsBuffer, 0, new Uint8Array(paramsData));

  // Build bind group entries
  const entries: any[] = [
    { binding: 0, resource: { buffer: aBuffer } },
    { binding: 1, resource: { buffer: bBuffer } },
    { binding: 2, resource: { buffer: kSplitFactor >= 2 ? kSplitBuffer : outBuffer } },
    { binding: 3, resource: { buffer: paramsBuffer } },
  ];
  for (let i = 0; i < epilogueBuffers.length; i++) {
    entries.push({ binding: 4 + i, resource: { buffer: epilogueBuffers[i] } });
  }

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  // K-split reduction bind group
  let reductionBindGroup: any = null;
  if (kSplitFactor >= 2 && reductionPipeline) {
    const reduceParamsData = new ArrayBuffer(8);
    const ru32 = new Uint32Array(reduceParamsData);
    const rf32 = new Float32Array(reduceParamsData);
    ru32[0] = m * n; // totalElements
    rf32[1] = 1.0;   // alpha
    const reduceParamsBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(reduceParamsBuffer, 0, new Uint8Array(reduceParamsData));

    reductionBindGroup = device.createBindGroup({
      layout: reductionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: kSplitBuffer } },
        { binding: 1, resource: { buffer: outBuffer } },
        { binding: 2, resource: { buffer: reduceParamsBuffer } },
      ],
    });
  }

  // Dispatch function
  const wgX = kSplitFactor >= 2 ? workgroupsX : Math.ceil(n / config.tileN);
  const wgY = kSplitFactor >= 2 ? workgroupsY : Math.ceil(m / config.tileM);
  const wgZ = kSplitFactor >= 2 ? kSplitFactor : batchSize;

  function dispatch() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY, wgZ);

    if (kSplitFactor >= 2 && reductionPipeline && reductionBindGroup) {
      pass.setPipeline(reductionPipeline);
      pass.setBindGroup(0, reductionBindGroup);
      pass.dispatchWorkgroups(Math.ceil(m * n / 256), 1, 1);
    }

    pass.end();
    queue.submit([encoder.finish()]);
  }

  // Warmup
  for (let i = 0; i < warmup; i++) dispatch();
  await syncWebGPU();

  // Timed runs — submit one at a time and sync to measure GPU time
  const times: number[] = [];
  for (let i = 0; i < iters; i++) {
    const start = performance.now();
    dispatch();
    await syncWebGPU();
    const elapsed = performance.now() - start;
    times.push(elapsed);
  }

  // Cleanup
  aBuffer.destroy();
  bBuffer.destroy();
  outBuffer.destroy();
  paramsBuffer.destroy();
  for (const b of epilogueBuffers) b.destroy();
  if (kSplitBuffer) kSplitBuffer.destroy();

  // Return median
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const warmup = parseInt(process.env.BENCH_WARMUP ?? "5");
  const iters = parseInt(process.env.BENCH_ITERS ?? "15");

  const ok = await initWebGPU();
  if (!ok) { console.error("WebGPU init failed"); process.exit(1); }
  const ctx = getWebGPUDevice();
  if (!ctx) { console.error("No WebGPU device"); process.exit(1); }
  const { device, queue } = ctx;

  console.log(`Tile-IR Matmul Benchmark (warmup=${warmup}, iters=${iters})\n`);
  console.log("Shape".padEnd(28) +
    "Config".padEnd(22) +
    "K-split".padEnd(8) +
    "Prod(ms)".padStart(10) +
    "TileIR(ms)".padStart(12) +
    "Ratio".padStart(8) +
    "  GFLOPS(P)".padStart(12) +
    "GFLOPS(T)".padStart(12));
  console.log("-".repeat(112));

  let totalProdWeighted = 0;
  let totalTileWeighted = 0;

  for (const shape of SHAPES) {
    const { m, n, k, epilogue, batchSize = 1 } = shape;
    const hasEpilogue = !!epilogue;
    const shapeClass = classifyShape(m, n, k, batchSize);
    const config = getDefaultConfigForShape(shapeClass, hasEpilogue);
    const configStr = `${config.tileM}x${config.tileN}x${config.tileK} t${config.threadTileM}x${config.threadTileN}`;

    // K-split check
    const workgroupsX = Math.ceil(n / config.tileN);
    const workgroupsY = Math.ceil(m / config.tileM);
    const baseWG = workgroupsX * workgroupsY;
    let kSplit = 0;
    if (batchSize === 1 && !hasEpilogue && baseWG < 64 && k >= 512) {
      kSplit = Math.min(Math.ceil(128 / baseWG), Math.floor(k / config.tileK), 32);
    }

    try {
      const prodMs = await benchmarkKernel(device, queue, shape, false, warmup, iters);
      const tileMs = await benchmarkKernel(device, queue, shape, true, warmup, iters);

      const flops = 2 * m * n * k * (batchSize || 1);
      const prodGflops = flops / (prodMs * 1e6);
      const tileGflops = flops / (tileMs * 1e6);
      const ratio = tileMs / prodMs;

      totalProdWeighted += prodMs * shape.count;
      totalTileWeighted += tileMs * shape.count;

      const ratioStr = ratio <= 1.05 ? `${ratio.toFixed(3)}` :
                       ratio <= 1.10 ? `${ratio.toFixed(3)}*` :
                       `${ratio.toFixed(3)}**`;

      console.log(
        shape.name.padEnd(28) +
        configStr.padEnd(22) +
        (kSplit >= 2 ? `×${kSplit}` : "-").padEnd(8) +
        prodMs.toFixed(3).padStart(10) +
        tileMs.toFixed(3).padStart(12) +
        ratioStr.padStart(8) +
        prodGflops.toFixed(0).padStart(12) +
        tileGflops.toFixed(0).padStart(12)
      );
    } catch (e: any) {
      console.log(
        shape.name.padEnd(28) +
        configStr.padEnd(22) +
        (kSplit >= 2 ? `×${kSplit}` : "-").padEnd(8) +
        `ERROR: ${e.message}`
      );
    }
  }

  console.log("-".repeat(112));
  const overallRatio = totalTileWeighted / totalProdWeighted;
  console.log(
    `WEIGHTED TOTAL (per step)`.padEnd(28) +
    "".padEnd(22) +
    "".padEnd(8) +
    totalProdWeighted.toFixed(3).padStart(10) +
    totalTileWeighted.toFixed(3).padStart(12) +
    overallRatio.toFixed(3).padStart(8)
  );
  console.log(`\n* = 5-10% slower, ** = >10% slower`);

  process.exit(0);
}

main();
