/**
 * Benchmark the tile-IR forward attention kernel.
 * Measures GPU execution time with timestamp queries.
 */
import { initWebGPU, webgpuBackend } from "../src/backend/webgpu";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";
import type { TileKernelSpec } from "../src/backend/webgpu/tile-ir";
import { tiledGrid } from "../src/backend/webgpu/tile-ir";

await initWebGPU();
const device = webgpuBackend.device!;

const BR = 64;
const BC = 32;
const F32_NEG_MAX = -3.4028234663852886e38;

function makeForwardAttentionSpec(headDim: number): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BR;
  return {
    name: `tileAttnFwd_D${D}`,
    workgroupSize: WG,
    autoBarriers: true,
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      O: { storage: "read_write", type: "f32" },
      L: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      is_causal: "u32",
    },
    grid: tiledGrid({
      x: { uniform: "seq_len", tileSize: BR },
      y: "num_heads",
      z: "batch_size",
    }),
    kernel(ctx) {
      const tidx = ctx.localIndex();
      const qBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);
      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const isCausal = ctx.uniform("is_causal");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");
      const qRow = qBlock.mul(ctx.u32(BR)).add(tidx);
      const valid = qRow.lt(N);
      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(N);
      const qBase = bhOff.add(qRow.mul(Dim));
      const Q = ctx.tileLoad(
        "Q",
        { kind: "thread", base: qBase, stride: ctx.u32(1) },
        { rows: 1, cols: D, guard: valid },
      );
      const mPrev = ctx.full(1, 1, F32_NEG_MAX);
      const lPrev = ctx.full(1, 1, 0);
      const oAcc = ctx.zeros(1, D);
      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));
      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));
        const offsR = ctx.arange(kvStart, BC);
        const offsD = ctx.arange(ctx.u32(0), D);
        const tilePtr = ctx.tilePtr(
          bhOff,
          offsR.outer(Dim),
          offsD.inner(ctx.u32(1)),
        );
        const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
        const K = ctx.load2D("K", tilePtr, tileMask);
        const scores = ctx.dot(Q, K.T());
        ctx.range(0, BC, (j) => {
          const kvPos = kvStart.add(j);
          const isActive = valid
            .and(kvPos.lt(N))
            .and(isCausal.eq(ctx.u32(0)).or(kvPos.le(qRow)));
          scores.set(
            j,
            isActive.select(scores.get(j).mul(scale), ctx.f32(F32_NEG_MAX)),
          );
        });
        const mNew = scores.max(1);
        const mMax = mNew.max(mPrev);
        const correction = mPrev.sub(mMax).exp();
        oAcc.mul_(correction);
        lPrev.mul_(correction);
        scores.sub_(mMax);
        scores.exp_();
        lPrev.add_(scores.sum(1));
        mPrev.assign(mMax);
        const V = ctx.load2D("V", tilePtr, tileMask, { reuseShared: K });
        ctx.dotAccum(scores, V, oAcc);
      });
      ctx.ifThen(valid, () => {
        const l = lPrev.get(ctx.u32(0));
        const invL = l.gt(ctx.f32(0)).select(ctx.f32(1).div(l), ctx.f32(0));
        oAcc.mul_(invL);
        ctx.tileStore("O", oAcc, { base: qBase, stride: ctx.u32(1) });
        const m = mPrev.get(ctx.u32(0));
        const lse = m.add(l.max(ctx.f32(1e-10)).log());
        ctx.emitStore("L", bhOffL.add(qRow), lse);
      });
    },
  };
}

const wgsl = compileTileKernel(makeForwardAttentionSpec(64));

// Create pipeline
const module = device.createShaderModule({ code: wgsl });
const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module, entryPoint: "main" },
});

// Setup: B=1, H=12, N=512, D=64
const B = 1,
  H = 12,
  N = 512,
  D = 64;
const qkvSize = B * H * N * D;
const lSize = B * H * N;

const qBuf = device.createBuffer({
  size: qkvSize * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const kBuf = device.createBuffer({
  size: qkvSize * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const vBuf = device.createBuffer({
  size: qkvSize * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const oBuf = device.createBuffer({
  size: qkvSize * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const lBuf = device.createBuffer({
  size: lSize * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

// Fill with random data
const data = new Float32Array(qkvSize);
for (let i = 0; i < qkvSize; i++) data[i] = (Math.random() - 0.5) * 0.1;
device.queue.writeBuffer(qBuf, 0, data);
device.queue.writeBuffer(kBuf, 0, data);
device.queue.writeBuffer(vBuf, 0, data);

const scale = 1 / Math.sqrt(D);
const scaleU32 = new Uint32Array(new Float32Array([scale]).buffer)[0];

// Uniform buffer
const uniformData = new Uint32Array([B, H, N, D, scaleU32, 1 /* causal */]);
const uniformBuf = device.createBuffer({
  size: uniformData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuf, 0, uniformData);

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: qBuf } },
    { binding: 1, resource: { buffer: kBuf } },
    { binding: 2, resource: { buffer: vBuf } },
    { binding: 3, resource: { buffer: oBuf } },
    { binding: 4, resource: { buffer: lBuf } },
    { binding: 5, resource: { buffer: uniformBuf } },
  ],
});

const gridX = Math.ceil(N / BR);

// Timestamp queries
const hasTimestamp = device.features.has("timestamp-query");
let querySet: GPUQuerySet | null = null;
let queryBuf: GPUBuffer | null = null;
let readBuf: GPUBuffer | null = null;

if (hasTimestamp) {
  querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  queryBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  readBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
}

// Warmup
for (let i = 0; i < 5; i++) {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(gridX, H, B);
  pass.end();
  device.queue.submit([enc.finish()]);
}
await device.queue.onSubmittedWorkDone();

// Benchmark
const ITERS = 20;
const times: number[] = [];

if (hasTimestamp) {
  for (let i = 0; i < ITERS; i++) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass({
      timestampWrites: {
        querySet: querySet!,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      },
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(gridX, H, B);
    pass.end();
    enc.resolveQuerySet(querySet!, 0, 2, queryBuf!, 0);
    enc.copyBufferToBuffer(queryBuf!, 0, readBuf!, 0, 16);
    device.queue.submit([enc.finish()]);
    await readBuf!.mapAsync(GPUMapMode.READ);
    const ts = new BigInt64Array(readBuf!.getMappedRange().slice(0));
    readBuf!.unmap();
    const ns = Number(ts[1] - ts[0]);
    times.push(ns / 1000); // µs
  }
} else {
  // CPU timing fallback
  for (let i = 0; i < ITERS; i++) {
    const start = performance.now();
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(gridX, H, B);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    times.push((performance.now() - start) * 1000);
  }
}

times.sort((a, b) => a - b);
const median = times[Math.floor(times.length / 2)];
const min = times[0];
const max = times[times.length - 1];
const p10 = times[Math.floor(times.length * 0.1)];
const p90 = times[Math.floor(times.length * 0.9)];

console.log(`Forward attention: B=${B}, H=${H}, N=${N}, D=${D}, causal=true`);
console.log(
  `  ${hasTimestamp ? "GPU timestamp" : "CPU wall"} timing (${ITERS} iterations):`,
);
console.log(`  Median: ${median.toFixed(0)}µs`);
console.log(`  Min: ${min.toFixed(0)}µs, Max: ${max.toFixed(0)}µs`);
console.log(`  P10: ${p10.toFixed(0)}µs, P90: ${p90.toFixed(0)}µs`);

// Cleanup
qBuf.destroy();
kBuf.destroy();
vBuf.destroy();
oBuf.destroy();
lBuf.destroy();
uniformBuf.destroy();
if (queryBuf) queryBuf.destroy();
if (readBuf) readBuf.destroy();
if (querySet) querySet.destroy();

process.exit(0);
