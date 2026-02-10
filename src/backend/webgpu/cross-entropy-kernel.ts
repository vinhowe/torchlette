/**
 * Fused Cross-Entropy Kernel
 *
 * Single-dispatch forward and backward kernels that replace the 9-op
 * decomposed cross-entropy (max, sub, exp, sum, log, sub, gather, neg, mean).
 *
 * Uses 256-thread workgroup-level parallel reduction in shared memory.
 * Each workgroup handles one row (one sample in the batch).
 *
 * Forward:  logits [B, V] + targets [B] → loss [B]
 * Backward: logits [B, V] + targets [B] + grad_output [B] → grad_logits [B, V]
 */

import {
  dispatchComputePass,
  getWebGPUDevice,
  trackSharedEncoderWrite,
  allocateOutputBuffer,
  cachedCreateBindGroup,
} from "./index";

// WebGPU type definitions (runtime, not importable at compile time)
type GPUBuffer = {
  size: number;
  usage: number;
  destroy(): void;
};

type GPUDevice = {
  createShaderModule(descriptor: { code: string }): GPUShaderModule;
  createComputePipeline(descriptor: {
    layout: "auto";
    compute: { module: GPUShaderModule; entryPoint: string };
  }): GPUComputePipeline;
  createBindGroup(descriptor: {
    layout: GPUBindGroupLayout;
    entries: Array<{
      binding: number;
      resource: { buffer: GPUBuffer; offset?: number; size?: number };
    }>;
  }): GPUBindGroup;
  createBuffer(descriptor: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): GPUBuffer;
  queue: {
    writeBuffer(
      buffer: GPUBuffer,
      offset: number,
      data: ArrayBufferView,
    ): void;
  };
};

type GPUShaderModule = object;
type GPUComputePipeline = {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
};
type GPUBindGroupLayout = object;
type GPUBindGroup = object;

const GPUBufferUsage = {
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};

const WORKGROUP_SIZE = 256;

// ============================================================================
// Pipeline Cache
// ============================================================================

const pipelineCache = new Map<string, GPUComputePipeline>();

function getOrCreatePipeline(
  device: GPUDevice,
  key: string,
  code: string,
): GPUComputePipeline {
  let pipeline = pipelineCache.get(key);
  if (!pipeline) {
    const module = device.createShaderModule({ code });
    pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    pipelineCache.set(key, pipeline);
  }
  return pipeline;
}

// ============================================================================
// WGSL Shaders
// ============================================================================

function crossEntropyForwardShader(): string {
  return `
struct CEConfig {
  batch_size: u32,
  vocab_size: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss: array<f32>;
@group(0) @binding(3) var<uniform> config: CEConfig;

var<workgroup> sdata: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  let tid = lid.x;
  let V = config.vocab_size;
  let base = row * V;

  // Phase 1: Parallel max reduction
  var local_max = -3.402823e+38;
  for (var i = tid; i < V; i += ${WORKGROUP_SIZE}u) {
    local_max = max(local_max, logits[base + i]);
  }
  sdata[tid] = local_max;
  workgroupBarrier();

  for (var s = ${WORKGROUP_SIZE / 2}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      sdata[tid] = max(sdata[tid], sdata[tid + s]);
    }
    workgroupBarrier();
  }
  let row_max = sdata[0];

  // Phase 2: Parallel sum-exp reduction
  var local_sum = 0.0;
  for (var i = tid; i < V; i += ${WORKGROUP_SIZE}u) {
    local_sum += exp(logits[base + i] - row_max);
  }
  sdata[tid] = local_sum;
  workgroupBarrier();

  for (var s = ${WORKGROUP_SIZE / 2}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    workgroupBarrier();
  }
  let log_sum_exp = log(sdata[0]);

  // Phase 3: Output (thread 0 only)
  if (tid == 0u) {
    let t = u32(targets[row]);
    loss[row] = -(logits[base + t] - row_max - log_sum_exp);
  }
}
`;
}

function crossEntropyBackwardShader(): string {
  return `
struct CEConfig {
  batch_size: u32,
  vocab_size: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_logits: array<f32>;
@group(0) @binding(4) var<uniform> config: CEConfig;

var<workgroup> sdata: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  let tid = lid.x;
  let V = config.vocab_size;
  let base = row * V;

  // Phase 1: Parallel max reduction (recompute from forward)
  var local_max = -3.402823e+38;
  for (var i = tid; i < V; i += ${WORKGROUP_SIZE}u) {
    local_max = max(local_max, logits[base + i]);
  }
  sdata[tid] = local_max;
  workgroupBarrier();

  for (var s = ${WORKGROUP_SIZE / 2}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      sdata[tid] = max(sdata[tid], sdata[tid + s]);
    }
    workgroupBarrier();
  }
  let row_max = sdata[0];

  // Phase 2: Parallel sum-exp reduction (recompute from forward)
  var local_sum = 0.0;
  for (var i = tid; i < V; i += ${WORKGROUP_SIZE}u) {
    local_sum += exp(logits[base + i] - row_max);
  }
  sdata[tid] = local_sum;
  workgroupBarrier();

  for (var s = ${WORKGROUP_SIZE / 2}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    workgroupBarrier();
  }
  let log_sum_exp = log(sdata[0]);

  // Phase 3: Write gradients (all threads participate)
  // grad_logits[b,v] = grad_output[b] * (softmax[b,v] - one_hot(target[b])[v])
  let t = u32(targets[row]);
  let g = grad_output[row];
  for (var i = tid; i < V; i += ${WORKGROUP_SIZE}u) {
    let softmax_i = exp(logits[base + i] - row_max - log_sum_exp);
    let one_hot_i = select(0.0, 1.0, i == t);
    grad_logits[base + i] = g * (softmax_i - one_hot_i);
  }
}
`;
}

// ============================================================================
// Config Buffer (cached per batch_size x vocab_size)
// ============================================================================

const configCache = new Map<string, GPUBuffer>();

function getOrCreateConfigBuffer(
  device: GPUDevice,
  batchSize: number,
  vocabSize: number,
): GPUBuffer {
  const key = `${batchSize}:${vocabSize}`;
  let buf = configCache.get(key);
  if (!buf) {
    buf = device.createBuffer({
      size: 8, // 2 x u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    configCache.set(key, buf);
  }
  const data = new Uint32Array([batchSize, vocabSize]);
  device.queue.writeBuffer(buf, 0, new Uint8Array(data.buffer));
  return buf;
}

// ============================================================================
// Dispatch Functions
// ============================================================================

/**
 * Dispatch fused cross-entropy forward kernel.
 * logits [B, V] + targets [B] → loss [B]
 */
export function dispatchCrossEntropyForward(
  logitsBuffer: GPUBuffer,
  targetsBuffer: GPUBuffer,
  batchSize: number,
  vocabSize: number,
): GPUBuffer {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device as unknown as GPUDevice;

  const outputSizeBytes = batchSize * 4; // f32
  const outBuffer = allocateOutputBuffer(outputSizeBytes) as unknown as GPUBuffer;

  const configBuf = getOrCreateConfigBuffer(device, batchSize, vocabSize);

  const pipeline = getOrCreatePipeline(
    device,
    "crossEntropyFwd",
    crossEntropyForwardShader(),
  );

  const bindGroup = cachedCreateBindGroup(device as any, pipeline as any,
    [logitsBuffer, targetsBuffer, outBuffer, configBuf] as any) as any;

  trackSharedEncoderWrite(logitsBuffer as any);
  trackSharedEncoderWrite(targetsBuffer as any);
  trackSharedEncoderWrite(outBuffer as any);

  dispatchComputePass(
    pipeline as any,
    bindGroup as any,
    batchSize, // one workgroup per row
  );

  return outBuffer;
}

/**
 * Dispatch fused cross-entropy backward kernel.
 * logits [B, V] + targets [B] + grad_output [B] → grad_logits [B, V]
 */
export function dispatchCrossEntropyBackward(
  logitsBuffer: GPUBuffer,
  targetsBuffer: GPUBuffer,
  gradOutputBuffer: GPUBuffer,
  batchSize: number,
  vocabSize: number,
): GPUBuffer {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device as unknown as GPUDevice;

  const outputSizeBytes = batchSize * vocabSize * 4; // f32
  const outBuffer = allocateOutputBuffer(outputSizeBytes) as unknown as GPUBuffer;

  const configBuf = getOrCreateConfigBuffer(device, batchSize, vocabSize);

  const pipeline = getOrCreatePipeline(
    device,
    "crossEntropyBwd",
    crossEntropyBackwardShader(),
  );

  const bindGroup = cachedCreateBindGroup(device as any, pipeline as any,
    [logitsBuffer, targetsBuffer, gradOutputBuffer, outBuffer, configBuf] as any) as any;

  trackSharedEncoderWrite(logitsBuffer as any);
  trackSharedEncoderWrite(targetsBuffer as any);
  trackSharedEncoderWrite(gradOutputBuffer as any);
  trackSharedEncoderWrite(outBuffer as any);

  dispatchComputePass(
    pipeline as any,
    bindGroup as any,
    batchSize, // one workgroup per row
  );

  return outBuffer;
}
