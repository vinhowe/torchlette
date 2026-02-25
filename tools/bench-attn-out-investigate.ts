/**
 * Investigate attn-out 32x32x16 regression: bare vs epilogue
 */
import { initWebGPU, getWebGPUDevice, syncWebGPU } from "../src/backend/webgpu";
import { generateTiledMatmulShader, type CodegenOptions } from "../src/backend/webgpu/matmul/codegen";
import { generateTiledMatmulShaderTileIR } from "../src/backend/webgpu/matmul/tile-matmul";

const GPUBufferUsage = { STORAGE: 0x0080, COPY_SRC: 0x0004, COPY_DST: 0x0008, UNIFORM: 0x0040 };

async function benchShape(
  device: any, queue: any,
  label: string, opts: CodegenOptions,
  m: number, n: number, k: number,
  useTileIR: boolean,
  epilogueCount: number,
): Promise<number> {
  const code = useTileIR ? generateTiledMatmulShaderTileIR(opts) : generateTiledMatmulShader(opts);
  const mod = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });

  const a = device.createBuffer({ size: Math.max(16, m * k * 2), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const b = device.createBuffer({ size: Math.max(16, k * n * 2), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const out = device.createBuffer({ size: Math.max(16, m * n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const params = new ArrayBuffer(48);
  const u = new Uint32Array(params);
  const f = new Float32Array(params);
  u[0] = m; u[1] = n; u[2] = k; u[3] = k; u[4] = n; u[5] = n; f[6] = 1.0; u[7] = 1; u[8] = m*k; u[9] = k*n; u[10] = m*n;
  const pb = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  queue.writeBuffer(pb, 0, new Uint8Array(params));

  const entries: any[] = [
    { binding: 0, resource: { buffer: a } },
    { binding: 1, resource: { buffer: b } },
    { binding: 2, resource: { buffer: out } },
    { binding: 3, resource: { buffer: pb } },
  ];

  const epilogueBufs: any[] = [];
  for (let i = 0; i < epilogueCount; i++) {
    const eb = device.createBuffer({ size: Math.max(16, n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    epilogueBufs.push(eb);
    entries.push({ binding: 4 + i, resource: { buffer: eb } });
  }

  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  const { tileM, tileN } = opts.config;
  const wgX = Math.ceil(n / tileN), wgY = Math.ceil(m / tileM);

  const dispatch = () => {
    const e = device.createCommandEncoder();
    const p = e.beginComputePass();
    p.setPipeline(pipeline);
    p.setBindGroup(0, bg);
    p.dispatchWorkgroups(wgX, wgY, 1);
    p.end();
    queue.submit([e.finish()]);
  };

  for (let i = 0; i < 5; i++) dispatch();
  await syncWebGPU();

  const times: number[] = [];
  for (let i = 0; i < 30; i++) {
    const s = performance.now();
    dispatch();
    await syncWebGPU();
    times.push(performance.now() - s);
  }

  a.destroy(); b.destroy(); out.destroy(); pb.destroy();
  for (const eb of epilogueBufs) eb.destroy();

  times.sort((a, b) => a - b);
  return times[15]; // median
}

async function main() {
  await initWebGPU();
  const ctx = getWebGPUDevice();
  if (!ctx) { console.error("No WebGPU device"); process.exit(1); }
  const { device, queue } = ctx;

  const m = 512, n = 768, k = 768;
  const config32 = { tileM: 32, tileN: 32, tileK: 16, threadTileM: 4, threadTileN: 4, useSubgroups: false, vectorWidth: 1 };

  console.log("=== attn out (fwd) investigation: 512x768x768 at 32x32x16 t4x4 ===\n");

  // Test 1: bare (no epilogue)
  const bareProd = await benchShape(device, queue, "prod-bare", {
    config: config32, transposeMode: "NN", dtype: "f16", dtypeB: "f16",
  }, m, n, k, false, 0);
  const bareTile = await benchShape(device, queue, "tile-bare", {
    config: config32, transposeMode: "NN", dtype: "f16", dtypeB: "f16",
  }, m, n, k, true, 0);
  console.log(`Bare:     prod=${bareProd.toFixed(3)}ms  tile=${bareTile.toFixed(3)}ms  ratio=${(bareTile/bareProd).toFixed(3)}`);

  // Test 2: cast epilogue only
  const castEpi = { ops: [{ kind: "cast" as const, toDtype: "f32" as const }], additionalInputCount: 0, outputDtype: "f32" as const };
  const castProd = await benchShape(device, queue, "prod-cast", {
    config: config32, transposeMode: "NN", dtype: "f16", dtypeB: "f16", epilogue: castEpi,
  }, m, n, k, false, 0);
  const castTile = await benchShape(device, queue, "tile-cast", {
    config: config32, transposeMode: "NN", dtype: "f16", dtypeB: "f16", epilogue: castEpi,
  }, m, n, k, true, 0);
  console.log(`Cast:     prod=${castProd.toFixed(3)}ms  tile=${castTile.toFixed(3)}ms  ratio=${(castTile/castProd).toFixed(3)}`);

  // Test 3: cast+bias epilogue (the actual problem case)
  const biasEpi = { ops: [{ kind: "cast" as const, toDtype: "f32" as const }, { kind: "bias" as const, inputIndex: 0 }], additionalInputCount: 1, outputDtype: "f32" as const };
  const biasProd = await benchShape(device, queue, "prod-bias", {
    config: config32, transposeMode: "NN", dtype: "f16", dtypeB: "f16", epilogue: biasEpi,
  }, m, n, k, false, 1);
  const biasTile = await benchShape(device, queue, "tile-bias", {
    config: config32, transposeMode: "NN", dtype: "f16", dtypeB: "f16", epilogue: biasEpi,
  }, m, n, k, true, 1);
  console.log(`Cast+Bias: prod=${biasProd.toFixed(3)}ms  tile=${biasTile.toFixed(3)}ms  ratio=${(biasTile/biasProd).toFixed(3)}`);

  // Test 4: same shape at 64x64x8 (the config used for square_large bare)
  const config64 = { tileM: 64, tileN: 64, tileK: 8, threadTileM: 4, threadTileN: 4, useSubgroups: false, vectorWidth: 1 };
  const big64Prod = await benchShape(device, queue, "prod-64x64", {
    config: config64, transposeMode: "NN", dtype: "f16", dtypeB: "f16", epilogue: biasEpi,
  }, m, n, k, false, 1);
  const big64Tile = await benchShape(device, queue, "tile-64x64", {
    config: config64, transposeMode: "NN", dtype: "f16", dtypeB: "f16", epilogue: biasEpi,
  }, m, n, k, true, 1);
  console.log(`64x64 Cast+Bias: prod=${big64Prod.toFixed(3)}ms  tile=${big64Tile.toFixed(3)}ms  ratio=${(big64Tile/big64Prod).toFixed(3)}`);

  process.exit(0);
}

main();
