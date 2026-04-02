/**
 * Tile-IR Attention Benchmark + Correctness Test
 *
 * 1. Correctness: small shapes (N=8..128, D=8..64) vs CPU reference
 * 2. Performance: DistilGPT-2 shapes (B=1, H=12, N=512, D=64)
 */
import { getWebGPUDevice, initWebGPU, syncWebGPU } from "../src/backend/webgpu";
import {
  makeBackwardDKVSpec,
  makeBackwardDQSpec,
  makeDPrecomputeSpec,
  makeForwardAttentionSpec,
} from "../src/backend/webgpu/attention-tile-ir";
import type {
  GPUBindGroup,
  GPUBuffer,
  GPUComputePipeline,
  GPUDevice,
  GPUQueue,
} from "../src/backend/webgpu/gpu-types";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";

const BUF = {
  STORAGE: 0x0080,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
};

// ============================================================================
// CPU Reference Implementations
// ============================================================================

function cpuAttentionForward(
  Q: Float32Array,
  K: Float32Array,
  V: Float32Array,
  B: number,
  H: number,
  N: number,
  D: number,
  scale: number,
  isCausal: boolean,
): { O: Float32Array; L: Float32Array } {
  const O = new Float32Array(B * H * N * D);
  const L = new Float32Array(B * H * N);

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const bhOff = (b * H + h) * N * D;
      const bhN = (b * H + h) * N;

      for (let i = 0; i < N; i++) {
        // Compute scores = Q[i] @ K[j]^T * scale
        const scores = new Float32Array(N);
        for (let j = 0; j < N; j++) {
          let s = 0;
          for (let d = 0; d < D; d++) {
            s += Q[bhOff + i * D + d] * K[bhOff + j * D + d];
          }
          s *= scale;
          if (isCausal && j > i) s = -3.402823e38;
          scores[j] = s;
        }

        // Softmax
        let maxVal = -Infinity;
        for (let j = 0; j < N; j++) maxVal = Math.max(maxVal, scores[j]);
        let sumExp = 0;
        for (let j = 0; j < N; j++) {
          scores[j] = Math.exp(scores[j] - maxVal);
          sumExp += scores[j];
        }
        for (let j = 0; j < N; j++) scores[j] /= sumExp;

        // Output = P @ V
        for (let d = 0; d < D; d++) {
          let o = 0;
          for (let j = 0; j < N; j++) {
            o += scores[j] * V[bhOff + j * D + d];
          }
          O[bhOff + i * D + d] = o;
        }

        // Logsumexp
        L[bhN + i] = maxVal + Math.log(sumExp);
      }
    }
  }

  return { O, L };
}

function cpuDPrecompute(
  dO: Float32Array,
  O: Float32Array,
  totalRows: number,
  D: number,
): Float32Array {
  const result = new Float32Array(totalRows);
  for (let i = 0; i < totalRows; i++) {
    let s = 0;
    for (let d = 0; d < D; d++) {
      s += dO[i * D + d] * O[i * D + d];
    }
    result[i] = s;
  }
  return result;
}

/**
 * CPU reference: backward dQ
 * dQ[i,d] = sum_j { ds[i,j] * K[j,d] }
 * where ds[i,j] = P[i,j] * (dP[i,j] - D_val[i]) * scale
 *   dP[i,j] = sum_d { dO[i,d] * V[j,d] }
 *   D_val[i] = sum_d { dO[i,d] * O[i,d] }
 */
function cpuBackwardDQ(
  Q: Float32Array,
  K: Float32Array,
  V: Float32Array,
  O: Float32Array,
  L: Float32Array,
  dO: Float32Array,
  B: number,
  H: number,
  N: number,
  D: number,
  scale: number,
  isCausal: boolean,
): Float32Array {
  const dQ = new Float32Array(B * H * N * D);

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const bhOff = (b * H + h) * N * D;
      const bhN = (b * H + h) * N;

      // Precompute D_val[i] = dot(dO[i], O[i])
      const D_val = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let d = 0; d < D; d++)
          s += dO[bhOff + i * D + d] * O[bhOff + i * D + d];
        D_val[i] = s;
      }

      for (let i = 0; i < N; i++) {
        // Recompute scores and softmax for row i
        const scores = new Float32Array(N);
        for (let j = 0; j < N; j++) {
          let s = 0;
          for (let d = 0; d < D; d++)
            s += Q[bhOff + i * D + d] * K[bhOff + j * D + d];
          s *= scale;
          if (isCausal && j > i) s = -3.402823e38;
          scores[j] = s;
        }
        // Softmax using L for stability
        const lse = L[bhN + i];
        const P = new Float32Array(N);
        for (let j = 0; j < N; j++) P[j] = Math.exp(scores[j] - lse);

        // dP[i,j] = sum_d dO[i,d] * V[j,d]
        // ds[i,j] = P[i,j] * (dP[i,j] - D_val[i]) * scale
        // dQ[i,d] += ds[i,j] * K[j,d]
        for (let j = 0; j < N; j++) {
          if (isCausal && j > i) continue;
          let dp = 0;
          for (let d = 0; d < D; d++)
            dp += dO[bhOff + i * D + d] * V[bhOff + j * D + d];
          const ds = P[j] * (dp - D_val[i]) * scale;
          for (let d = 0; d < D; d++) {
            dQ[bhOff + i * D + d] += ds * K[bhOff + j * D + d];
          }
        }
      }
    }
  }
  return dQ;
}

/**
 * CPU reference: backward dK, dV
 * dK[j,d] = sum_i { ds[i,j] * Q[i,d] }
 * dV[j,d] = sum_i { P[i,j] * dO[i,d] }
 */
function cpuBackwardDKV(
  Q: Float32Array,
  K: Float32Array,
  V: Float32Array,
  O: Float32Array,
  L: Float32Array,
  dO: Float32Array,
  B: number,
  H: number,
  N: number,
  D: number,
  scale: number,
  isCausal: boolean,
): { dK: Float32Array; dV: Float32Array } {
  const dK = new Float32Array(B * H * N * D);
  const dV = new Float32Array(B * H * N * D);

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const bhOff = (b * H + h) * N * D;
      const bhN = (b * H + h) * N;

      // Precompute D_val[i] = dot(dO[i], O[i])
      const D_val = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let d = 0; d < D; d++)
          s += dO[bhOff + i * D + d] * O[bhOff + i * D + d];
        D_val[i] = s;
      }

      for (let j = 0; j < N; j++) {
        for (let i = 0; i < N; i++) {
          if (isCausal && j > i) continue;

          // Recompute P[i,j]
          let score = 0;
          for (let d = 0; d < D; d++)
            score += Q[bhOff + i * D + d] * K[bhOff + j * D + d];
          score *= scale;
          if (isCausal && j > i) score = -3.402823e38;
          const lse = L[bhN + i];
          const p = Math.exp(score - lse);

          // dP[i,j] = sum_d dO[i,d] * V[j,d]
          let dp = 0;
          for (let d = 0; d < D; d++)
            dp += dO[bhOff + i * D + d] * V[bhOff + j * D + d];
          const ds = p * (dp - D_val[i]) * scale;

          // dK[j,d] += ds * Q[i,d]
          // dV[j,d] += p * dO[i,d]
          for (let d = 0; d < D; d++) {
            dK[bhOff + j * D + d] += ds * Q[bhOff + i * D + d];
            dV[bhOff + j * D + d] += p * dO[bhOff + i * D + d];
          }
        }
      }
    }
  }
  return { dK, dV };
}

// ============================================================================
// GPU Dispatch Helpers
// ============================================================================

function createPipeline(device: GPUDevice, wgsl: string) {
  const mod = device.createShaderModule({ code: wgsl });
  return device.createComputePipeline({
    layout: "auto",
    compute: { module: mod, entryPoint: "main" },
  });
}

function createBuffer(
  device: GPUDevice,
  size: number,
  usage: number,
  data?: ArrayBufferView,
): GPUBuffer {
  const buf = device.createBuffer({
    size: Math.max(16, size),
    usage,
  });
  if (data) device.queue.writeBuffer(buf, 0, data);
  return buf;
}

function packUniforms(
  spec: ReturnType<typeof makeForwardAttentionSpec>,
  values: Record<string, number>,
): Uint8Array {
  const entries = Object.entries(spec.uniforms);
  const paddedCount = Math.ceil(entries.length / 4) * 4;
  const buf = new ArrayBuffer(paddedCount * 4);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  for (let i = 0; i < entries.length; i++) {
    const [name, type] = entries[i];
    if (type === "f32") f32[i] = values[name];
    else u32[i] = values[name];
  }
  return new Uint8Array(buf);
}

function dispatch(
  device: GPUDevice,
  queue: GPUQueue,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  grid: number[],
): void {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(grid[0] || 1, grid[1] || 1, grid[2] || 1);
  pass.end();
  queue.submit([enc.finish()]);
}

async function readBuffer(
  device: GPUDevice,
  queue: GPUQueue,
  buf: GPUBuffer,
  size: number,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: Math.max(16, size),
    usage: BUF.COPY_DST | 0x0001, // MAP_READ
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, size);
  queue.submit([enc.finish()]);
  await staging.mapAsync(1); // MAP_READ
  const data = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

// ============================================================================
// Scale encoding: f32 → u32 bitcast
// ============================================================================
function f32ToU32Bits(val: number): number {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  f[0] = val;
  return u[0];
}

// ============================================================================
// Correctness Tests
// ============================================================================

async function testForwardCorrectness(
  device: GPUDevice,
  queue: GPUQueue,
  B: number,
  H: number,
  N: number,
  D: number,
  isCausal: boolean,
): Promise<{ pass: boolean; maxErr: number }> {
  const scale = 1.0 / Math.sqrt(D);
  const totalElements = B * H * N * D;

  // Generate deterministic test data
  const qData = new Float32Array(totalElements);
  const kData = new Float32Array(totalElements);
  const vData = new Float32Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    qData[i] = Math.sin(i * 0.1) * 0.5;
    kData[i] = Math.cos(i * 0.13) * 0.5;
    vData[i] = Math.sin(i * 0.17 + 1) * 0.5;
  }

  // CPU reference
  const { O: cpuO, L: cpuL } = cpuAttentionForward(
    qData,
    kData,
    vData,
    B,
    H,
    N,
    D,
    scale,
    isCausal,
  );

  // GPU buffers
  const qBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    qData,
  );
  const kBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    kData,
  );
  const vBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    vData,
  );
  const oBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_SRC,
  );
  const lBuf = createBuffer(device, B * H * N * 4, BUF.STORAGE | BUF.COPY_SRC);

  // Compile and dispatch
  const spec = makeForwardAttentionSpec(D);
  const wgsl = compileTileKernel(spec);
  const pipeline = createPipeline(device, wgsl);

  const uniforms = {
    batch_size: B,
    num_heads: H,
    seq_len: N,
    head_dim: D,
    scale_u32: f32ToU32Bits(scale),
    is_causal: isCausal ? 1 : 0,
  };
  const configData = packUniforms(spec, uniforms);
  const configBuf = createBuffer(
    device,
    configData.byteLength,
    BUF.UNIFORM | BUF.COPY_DST,
    configData,
  );

  const entries = [
    { binding: 0, resource: { buffer: qBuf } },
    { binding: 1, resource: { buffer: kBuf } },
    { binding: 2, resource: { buffer: vBuf } },
    { binding: 3, resource: { buffer: oBuf } },
    { binding: 4, resource: { buffer: lBuf } },
    { binding: 5, resource: { buffer: configBuf } },
  ];
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const grid = spec.grid(uniforms);
  dispatch(device, queue, pipeline, bindGroup, grid);
  await syncWebGPU();

  // Read back
  const gpuO = await readBuffer(device, queue, oBuf, totalElements * 4);
  const gpuL = await readBuffer(device, queue, lBuf, B * H * N * 4);

  // Compare
  let maxErrO = 0;
  for (let i = 0; i < totalElements; i++) {
    maxErrO = Math.max(maxErrO, Math.abs(gpuO[i] - cpuO[i]));
  }

  let maxErrL = 0;
  for (let i = 0; i < B * H * N; i++) {
    maxErrL = Math.max(maxErrL, Math.abs(gpuL[i] - cpuL[i]));
  }

  // Cleanup
  qBuf.destroy();
  kBuf.destroy();
  vBuf.destroy();
  oBuf.destroy();
  lBuf.destroy();
  configBuf.destroy();

  const maxErr = Math.max(maxErrO, maxErrL);
  return { pass: maxErr < 1e-2, maxErr };
}

async function testDPrecomputeCorrectness(
  device: GPUDevice,
  queue: GPUQueue,
  totalRows: number,
  D: number,
): Promise<{ pass: boolean; maxErr: number }> {
  const totalElements = totalRows * D;

  const dOData = new Float32Array(totalElements);
  const oData = new Float32Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    dOData[i] = Math.sin(i * 0.1) * 0.3;
    oData[i] = Math.cos(i * 0.17) * 0.3;
  }

  const cpuResult = cpuDPrecompute(dOData, oData, totalRows, D);

  const dOBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    dOData,
  );
  const oBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    oData,
  );
  const dBuf = createBuffer(device, totalRows * 4, BUF.STORAGE | BUF.COPY_SRC);

  const spec = makeDPrecomputeSpec(D);
  const wgsl = compileTileKernel(spec);
  const pipeline = createPipeline(device, wgsl);

  const uniforms = { total_rows: totalRows, head_dim: D };
  const configData = packUniforms(spec, uniforms);
  const configBuf = createBuffer(
    device,
    configData.byteLength,
    BUF.UNIFORM | BUF.COPY_DST,
    configData,
  );

  const entries = [
    { binding: 0, resource: { buffer: dOBuf } },
    { binding: 1, resource: { buffer: oBuf } },
    { binding: 2, resource: { buffer: dBuf } },
    { binding: 3, resource: { buffer: configBuf } },
  ];
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const grid = spec.grid(uniforms);
  dispatch(device, queue, pipeline, bindGroup, grid);
  await syncWebGPU();

  const gpuResult = await readBuffer(device, queue, dBuf, totalRows * 4);

  let maxErr = 0;
  for (let i = 0; i < totalRows; i++) {
    maxErr = Math.max(maxErr, Math.abs(gpuResult[i] - cpuResult[i]));
  }

  dOBuf.destroy();
  oBuf.destroy();
  dBuf.destroy();
  configBuf.destroy();

  return { pass: maxErr < 1e-3, maxErr };
}

async function testBackwardDQCorrectness(
  device: GPUDevice,
  queue: GPUQueue,
  B: number,
  H: number,
  N: number,
  D: number,
  isCausal: boolean,
): Promise<{ pass: boolean; maxErr: number }> {
  const scale = 1.0 / Math.sqrt(D);
  const totalElements = B * H * N * D;
  const totalN = B * H * N;

  // Generate deterministic test data
  const qData = new Float32Array(totalElements);
  const kData = new Float32Array(totalElements);
  const vData = new Float32Array(totalElements);
  const dOData = new Float32Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    qData[i] = Math.sin(i * 0.1) * 0.5;
    kData[i] = Math.cos(i * 0.13) * 0.5;
    vData[i] = Math.sin(i * 0.17 + 1) * 0.5;
    dOData[i] = Math.cos(i * 0.19 + 2) * 0.3;
  }

  // Get O, L from forward pass (CPU reference)
  const { O: oData, L: lData } = cpuAttentionForward(
    qData,
    kData,
    vData,
    B,
    H,
    N,
    D,
    scale,
    isCausal,
  );
  // D_val from D-precompute (CPU reference)
  const dValData = cpuDPrecompute(dOData, oData, totalN, D);
  // dQ reference
  const cpuDQ = cpuBackwardDQ(
    qData,
    kData,
    vData,
    oData,
    lData,
    dOData,
    B,
    H,
    N,
    D,
    scale,
    isCausal,
  );

  // GPU buffers
  const qBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    qData,
  );
  const kBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    kData,
  );
  const vBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    vData,
  );
  const lBuf = createBuffer(
    device,
    totalN * 4,
    BUF.STORAGE | BUF.COPY_DST,
    lData,
  );
  const dBuf = createBuffer(
    device,
    totalN * 4,
    BUF.STORAGE | BUF.COPY_DST,
    dValData,
  );
  const dOBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    dOData,
  );
  const dQBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_SRC,
  );

  const spec = makeBackwardDQSpec(D);
  const wgsl = compileTileKernel(spec);
  const pipeline = createPipeline(device, wgsl);

  const uniforms = {
    batch_size: B,
    num_heads: H,
    seq_len: N,
    head_dim: D,
    scale_u32: f32ToU32Bits(scale),
    is_causal: isCausal ? 1 : 0,
  };
  const configData = packUniforms(spec, uniforms);
  const configBuf = createBuffer(
    device,
    configData.byteLength,
    BUF.UNIFORM | BUF.COPY_DST,
    configData,
  );

  const entries = [
    { binding: 0, resource: { buffer: qBuf } },
    { binding: 1, resource: { buffer: kBuf } },
    { binding: 2, resource: { buffer: vBuf } },
    { binding: 3, resource: { buffer: lBuf } },
    { binding: 4, resource: { buffer: dBuf } },
    { binding: 5, resource: { buffer: dOBuf } },
    { binding: 6, resource: { buffer: dQBuf } },
    { binding: 7, resource: { buffer: configBuf } },
  ];
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const grid = spec.grid(uniforms);
  dispatch(device, queue, pipeline, bindGroup, grid);
  await syncWebGPU();

  const gpuDQ = await readBuffer(device, queue, dQBuf, totalElements * 4);

  let maxErr = 0;
  for (let i = 0; i < totalElements; i++) {
    maxErr = Math.max(maxErr, Math.abs(gpuDQ[i] - cpuDQ[i]));
  }

  qBuf.destroy();
  kBuf.destroy();
  vBuf.destroy();
  lBuf.destroy();
  dBuf.destroy();
  dOBuf.destroy();
  dQBuf.destroy();
  configBuf.destroy();

  return { pass: maxErr < 5e-2, maxErr };
}

async function testBackwardDKVCorrectness(
  device: GPUDevice,
  queue: GPUQueue,
  B: number,
  H: number,
  N: number,
  D: number,
  isCausal: boolean,
): Promise<{ pass: boolean; maxErr: number }> {
  const scale = 1.0 / Math.sqrt(D);
  const totalElements = B * H * N * D;
  const totalN = B * H * N;

  const qData = new Float32Array(totalElements);
  const kData = new Float32Array(totalElements);
  const vData = new Float32Array(totalElements);
  const dOData = new Float32Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    qData[i] = Math.sin(i * 0.1) * 0.5;
    kData[i] = Math.cos(i * 0.13) * 0.5;
    vData[i] = Math.sin(i * 0.17 + 1) * 0.5;
    dOData[i] = Math.cos(i * 0.19 + 2) * 0.3;
  }

  const { O: oData, L: lData } = cpuAttentionForward(
    qData,
    kData,
    vData,
    B,
    H,
    N,
    D,
    scale,
    isCausal,
  );
  const dValData = cpuDPrecompute(dOData, oData, totalN, D);
  const { dK: cpuDK, dV: cpuDV } = cpuBackwardDKV(
    qData,
    kData,
    vData,
    oData,
    lData,
    dOData,
    B,
    H,
    N,
    D,
    scale,
    isCausal,
  );

  const qBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    qData,
  );
  const kBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    kData,
  );
  const vBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    vData,
  );
  const lBuf = createBuffer(
    device,
    totalN * 4,
    BUF.STORAGE | BUF.COPY_DST,
    lData,
  );
  const dBuf = createBuffer(
    device,
    totalN * 4,
    BUF.STORAGE | BUF.COPY_DST,
    dValData,
  );
  const dOBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_DST,
    dOData,
  );
  const dKBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_SRC,
  );
  const dVBuf = createBuffer(
    device,
    totalElements * 4,
    BUF.STORAGE | BUF.COPY_SRC,
  );

  const spec = makeBackwardDKVSpec(D);
  const wgsl = compileTileKernel(spec);
  const pipeline = createPipeline(device, wgsl);

  const uniforms = {
    batch_size: B,
    num_heads: H,
    seq_len: N,
    head_dim: D,
    scale_u32: f32ToU32Bits(scale),
    is_causal: isCausal ? 1 : 0,
  };
  const configData = packUniforms(spec, uniforms);
  const configBuf = createBuffer(
    device,
    configData.byteLength,
    BUF.UNIFORM | BUF.COPY_DST,
    configData,
  );

  const entries = [
    { binding: 0, resource: { buffer: qBuf } },
    { binding: 1, resource: { buffer: kBuf } },
    { binding: 2, resource: { buffer: vBuf } },
    { binding: 3, resource: { buffer: lBuf } },
    { binding: 4, resource: { buffer: dBuf } },
    { binding: 5, resource: { buffer: dOBuf } },
    { binding: 6, resource: { buffer: dKBuf } },
    { binding: 7, resource: { buffer: dVBuf } },
    { binding: 8, resource: { buffer: configBuf } },
  ];
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const grid = spec.grid(uniforms);
  dispatch(device, queue, pipeline, bindGroup, grid);
  await syncWebGPU();

  const gpuDK = await readBuffer(device, queue, dKBuf, totalElements * 4);
  const gpuDV = await readBuffer(device, queue, dVBuf, totalElements * 4);

  let maxErrK = 0,
    maxErrV = 0;
  for (let i = 0; i < totalElements; i++) {
    maxErrK = Math.max(maxErrK, Math.abs(gpuDK[i] - cpuDK[i]));
    maxErrV = Math.max(maxErrV, Math.abs(gpuDV[i] - cpuDV[i]));
  }

  qBuf.destroy();
  kBuf.destroy();
  vBuf.destroy();
  lBuf.destroy();
  dBuf.destroy();
  dOBuf.destroy();
  dKBuf.destroy();
  dVBuf.destroy();
  configBuf.destroy();

  const maxErr = Math.max(maxErrK, maxErrV);
  return { pass: maxErr < 5e-2, maxErr };
}

// ============================================================================
// Benchmark
// ============================================================================

async function benchKernel(
  device: GPUDevice,
  queue: GPUQueue,
  _label: string,
  wgsl: string,
  bufferSizes: Record<string, number>,
  configData: Uint8Array,
  grid: number[],
  warmup: number,
  iters: number,
): Promise<number> {
  const pipeline = createPipeline(device, wgsl);

  // Create storage buffers
  const buffers: Record<string, GPUBuffer> = {};
  const entries: Array<{
    binding: number;
    resource: { buffer: GPUBuffer };
  }> = [];
  let binding = 0;
  for (const [name, size] of Object.entries(bufferSizes)) {
    const buf = device.createBuffer({
      size: Math.max(16, size),
      usage: BUF.STORAGE | BUF.COPY_SRC | BUF.COPY_DST,
    });
    buffers[name] = buf;
    entries.push({ binding, resource: { buffer: buf } });
    binding++;
  }

  // Config buffer
  const configBuf = createBuffer(
    device,
    configData.byteLength,
    BUF.UNIFORM | BUF.COPY_DST,
    configData,
  );
  entries.push({ binding, resource: { buffer: configBuf } });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  // Warmup
  for (let i = 0; i < warmup; i++) {
    dispatch(device, queue, pipeline, bindGroup, grid);
  }
  await syncWebGPU();

  // Timed iterations
  const times: number[] = [];
  for (let i = 0; i < iters; i++) {
    const start = performance.now();
    dispatch(device, queue, pipeline, bindGroup, grid);
    await syncWebGPU();
    times.push(performance.now() - start);
  }

  // Cleanup
  for (const buf of Object.values(buffers)) buf.destroy();
  configBuf.destroy();

  times.sort((a, b) => a - b);
  return times[Math.floor(iters / 2)]; // median
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  await initWebGPU();
  const ctx = getWebGPUDevice();
  if (!ctx) {
    console.error("No WebGPU device");
    process.exit(1);
  }
  const { device, queue } = ctx;

  const WARMUP = parseInt(process.env.BENCH_WARMUP || "5", 10);
  const ITERS = parseInt(process.env.BENCH_ITERS || "20", 10);

  // ---- Correctness Tests ----
  console.log("=== Correctness Tests ===\n");

  const correctnessTests = [
    // Forward attention
    {
      name: "Fwd N=8 D=8 non-causal",
      fn: () => testForwardCorrectness(device, queue, 1, 1, 8, 8, false),
    },
    {
      name: "Fwd N=8 D=16 causal",
      fn: () => testForwardCorrectness(device, queue, 1, 1, 8, 16, true),
    },
    {
      name: "Fwd N=64 D=64 non-causal",
      fn: () => testForwardCorrectness(device, queue, 1, 1, 64, 64, false),
    },
    {
      name: "Fwd N=64 D=64 causal",
      fn: () => testForwardCorrectness(device, queue, 1, 1, 64, 64, true),
    },
    {
      name: "Fwd N=128 D=64 causal (multi-tile)",
      fn: () => testForwardCorrectness(device, queue, 1, 1, 128, 64, true),
    },
    {
      name: "Fwd B=2 H=4 N=32 D=64",
      fn: () => testForwardCorrectness(device, queue, 2, 4, 32, 64, false),
    },
    // D-precompute
    {
      name: "DPre 128 rows D=64",
      fn: () => testDPrecomputeCorrectness(device, queue, 128, 64),
    },
    {
      name: "DPre 512 rows D=64",
      fn: () => testDPrecomputeCorrectness(device, queue, 512, 64),
    },
    // Backward dQ
    {
      name: "BwdDQ N=8 D=8 non-causal",
      fn: () => testBackwardDQCorrectness(device, queue, 1, 1, 8, 8, false),
    },
    {
      name: "BwdDQ N=8 D=16 causal",
      fn: () => testBackwardDQCorrectness(device, queue, 1, 1, 8, 16, true),
    },
    {
      name: "BwdDQ N=64 D=64 causal",
      fn: () => testBackwardDQCorrectness(device, queue, 1, 1, 64, 64, true),
    },
    // Backward dKV
    {
      name: "BwdDKV N=8 D=8 non-causal",
      fn: () => testBackwardDKVCorrectness(device, queue, 1, 1, 8, 8, false),
    },
    {
      name: "BwdDKV N=8 D=16 causal",
      fn: () => testBackwardDKVCorrectness(device, queue, 1, 1, 8, 16, true),
    },
    {
      name: "BwdDKV N=64 D=64 causal",
      fn: () => testBackwardDKVCorrectness(device, queue, 1, 1, 64, 64, true),
    },
  ];

  let allPass = true;
  for (const t of correctnessTests) {
    const { pass, maxErr } = await t.fn();
    const status = pass ? "PASS" : "FAIL";
    console.log(`  ${status}  ${t.name}  (maxErr=${maxErr.toExponential(2)})`);
    if (!pass) allPass = false;
  }

  console.log(
    `\n${allPass ? "All correctness tests passed!" : "SOME TESTS FAILED"}\n`,
  );

  // ---- Benchmarks ----
  console.log(
    "=== Performance Benchmarks (DistilGPT-2: B=1, H=12, N=512, D=64) ===\n",
  );

  const B = 1,
    H = 12,
    N = 512,
    D = 64;
  const scale = 1.0 / Math.sqrt(D);
  const totalBHND = B * H * N * D;
  const totalBHN = B * H * N;

  // Forward
  {
    const spec = makeForwardAttentionSpec(D);
    const wgsl = compileTileKernel(spec);
    const uniforms = {
      batch_size: B,
      num_heads: H,
      seq_len: N,
      head_dim: D,
      scale_u32: f32ToU32Bits(scale),
      is_causal: 1,
    };
    const configData = packUniforms(spec, uniforms);
    const grid = spec.grid(uniforms);

    const t = await benchKernel(
      device,
      queue,
      "Forward",
      wgsl,
      {
        Q: totalBHND * 4,
        K: totalBHND * 4,
        V: totalBHND * 4,
        O: totalBHND * 4,
        L: totalBHN * 4,
      },
      configData,
      grid,
      WARMUP,
      ITERS,
    );
    console.log(
      `  Forward:        ${t.toFixed(3)} ms  (${grid[0]}×${grid[1]}×${grid[2]} workgroups)`,
    );
  }

  // D-precompute
  {
    const spec = makeDPrecomputeSpec(D);
    const wgsl = compileTileKernel(spec);
    const uniforms = { total_rows: totalBHN, head_dim: D };
    const configData = packUniforms(spec, uniforms);
    const grid = spec.grid(uniforms);

    const t = await benchKernel(
      device,
      queue,
      "DPrecompute",
      wgsl,
      {
        dO: totalBHND * 4,
        Out: totalBHND * 4,
        D_val: totalBHN * 4,
      },
      configData,
      grid,
      WARMUP,
      ITERS,
    );
    console.log(
      `  D-precompute:   ${t.toFixed(3)} ms  (${grid[0]} workgroups)`,
    );
  }

  // Backward dQ
  {
    const spec = makeBackwardDQSpec(D);
    const wgsl = compileTileKernel(spec);
    const uniforms = {
      batch_size: B,
      num_heads: H,
      seq_len: N,
      head_dim: D,
      scale_u32: f32ToU32Bits(scale),
      is_causal: 1,
    };
    const configData = packUniforms(spec, uniforms);
    const grid = spec.grid(uniforms);

    const t = await benchKernel(
      device,
      queue,
      "Backward dQ",
      wgsl,
      {
        Q: totalBHND * 4,
        K: totalBHND * 4,
        V: totalBHND * 4,
        L_buf: totalBHN * 4,
        D_buf: totalBHN * 4,
        dO: totalBHND * 4,
        dQ: totalBHND * 4,
      },
      configData,
      grid,
      WARMUP,
      ITERS,
    );
    console.log(
      `  Backward dQ:    ${t.toFixed(3)} ms  (${grid[0]}×${grid[1]}×${grid[2]} workgroups)`,
    );
  }

  // Backward dKV
  {
    const spec = makeBackwardDKVSpec(D);
    const wgsl = compileTileKernel(spec);
    const uniforms = {
      batch_size: B,
      num_heads: H,
      seq_len: N,
      head_dim: D,
      scale_u32: f32ToU32Bits(scale),
      is_causal: 1,
    };
    const configData = packUniforms(spec, uniforms);
    const grid = spec.grid(uniforms);

    const t = await benchKernel(
      device,
      queue,
      "Backward dKV",
      wgsl,
      {
        Q: totalBHND * 4,
        K: totalBHND * 4,
        V: totalBHND * 4,
        L_buf: totalBHN * 4,
        D_buf: totalBHN * 4,
        dO: totalBHND * 4,
        dK: totalBHND * 4,
        dV: totalBHND * 4,
      },
      configData,
      grid,
      WARMUP,
      ITERS,
    );
    console.log(
      `  Backward dKV:   ${t.toFixed(3)} ms  (${grid[0]}×${grid[1]}×${grid[2]} workgroups)`,
    );
  }

  console.log("\nDone.");
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
