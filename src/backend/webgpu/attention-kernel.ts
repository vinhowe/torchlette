/**
 * Fused FlashAttention Kernel
 *
 * Single-dispatch forward and backward kernels that replace the decomposed
 * attention path (Q@K^T + scale + mask + softmax + attn@V).
 *
 * Uses online softmax (FlashAttention algorithm) to avoid materializing
 * the full N×N attention matrix.
 *
 * Forward:  Q,K,V [B,H,N,D] → O [B,H,N,D] + L [B,H,N] (logsumexp)
 * Backward: Q,K,V,L,dO → dQ,dK,dV
 *
 * All WGSL generation is handled by tile-IR (attention-tile-ir.ts).
 * Uses vec4 register/shared arrays and subgroup cooperative dot products.
 */

import {
  dispatchComputePass,
  allocateOutputBuffer,
  cachedCreateBindGroup,
  getPipeline,
} from "./index";
import { requireContext } from "./webgpu-state";

import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { trackBuffers } from "./wgsl-helpers";
import { compileTileKernel } from "./tile-compiler";
import {
  makeForwardAttentionSpec,
  makeDPrecomputeSpec,
  makeBackwardDQSpec,
  makeBackwardDKVSpec,
} from "./attention-tile-ir";

// Cache compiled tile-IR WGSL per headDim to avoid recompilation
const tileIRWGSLCache = new Map<string, string>();
function getTileIRWGSL(key: string, specFactory: () => import("./tile-ir").TileKernelSpec): string {
  let wgsl = tileIRWGSLCache.get(key);
  if (!wgsl) {
    wgsl = compileTileKernel(specFactory());
    tileIRWGSLCache.set(key, wgsl);
  }
  return wgsl;
}

// Tiling parameters (must match attention-tile-ir.ts)
const BR = 64;  // Q rows per workgroup

// ============================================================================
// Config Buffer Cache
// ============================================================================

const configCache = new Map<string, GPUBuffer>();

function getOrCreateConfigBuffer(
  device: GPUDevice,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: number,
): GPUBuffer {
  const key = `${batchSize}:${numHeads}:${seqLen}:${headDim}:${scale}:${isCausal}`;
  let buf = configCache.get(key);
  if (buf) return buf;

  buf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false,
  });

  const data = new ArrayBuffer(32);
  const u32View = new Uint32Array(data);
  const f32View = new Float32Array(data);
  u32View[0] = batchSize;
  u32View[1] = numHeads;
  u32View[2] = seqLen;
  u32View[3] = headDim;
  f32View[4] = scale;
  u32View[5] = isCausal;
  u32View[6] = 0; // pad
  u32View[7] = 0; // pad
  device.queue.writeBuffer(buf, 0, new Uint8Array(data));
  return buf;
}

// ============================================================================
// Dispatch Functions
// ============================================================================

/**
 * Dispatch FlashAttention forward kernel.
 * Q,K,V: [B, H, N, D] → O: [B, H, N, D], L: [B, H, N]
 */
export function dispatchFlashAttentionForward(
  qBuffer: GPUBuffer,
  kBuffer: GPUBuffer,
  vBuffer: GPUBuffer,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: boolean,
): { outputBuffer: GPUBuffer; logsumexpBuffer: GPUBuffer } {
  const ctx = requireContext();
  const device = ctx.device;

  const outputSizeBytes = batchSize * numHeads * seqLen * headDim * 4; // f32
  const logsumexpSizeBytes = batchSize * numHeads * seqLen * 4; // f32

  const outBuffer = allocateOutputBuffer(outputSizeBytes);
  const lseBuffer = allocateOutputBuffer(logsumexpSizeBytes);

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const wgsl = getTileIRWGSL(`fwd:${headDim}`, () => makeForwardAttentionSpec(headDim));
  const pipelineKey = `faFwd:tile:${headDim}`;
  const pipeline = getPipeline(ctx, pipelineKey, wgsl);

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [qBuffer, kBuffer, vBuffer, outBuffer, lseBuffer, configBuf]);

  trackBuffers(qBuffer, kBuffer, vBuffer, outBuffer, lseBuffer);

  const numQBlocks = Math.ceil(seqLen / BR);
  dispatchComputePass(
    pipeline,
    bindGroup,
    numQBlocks,
    numHeads,
    batchSize,
  );

  return { outputBuffer: outBuffer, logsumexpBuffer: lseBuffer };
}

/**
 * Dispatch FlashAttention backward D precompute kernel.
 * dO,O: [B,H,N,D] → D: [B,H,N]
 */
export function dispatchFlashAttentionBackwardD(
  dOBuffer: GPUBuffer,
  oBuffer: GPUBuffer,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: boolean,
): GPUBuffer {
  const ctx = requireContext();
  const device = ctx.device;

  const totalRows = batchSize * numHeads * seqLen;
  const outputSizeBytes = totalRows * 4; // f32

  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const wgsl = getTileIRWGSL(`bwdD:${headDim}`, () => makeDPrecomputeSpec(headDim));
  const pipelineKey = `faBwdD:tile:${headDim}`;
  const pipeline = getPipeline(ctx, pipelineKey, wgsl);

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [dOBuffer, oBuffer, outBuffer, configBuf]);

  trackBuffers(dOBuffer, oBuffer, outBuffer);

  dispatchComputePass(
    pipeline,
    bindGroup,
    totalRows,
  );

  return outBuffer;
}

/**
 * Dispatch FlashAttention backward dQ kernel.
 * Q,K,V,L,D,dO → dQ: [B,H,N,D]
 */
export function dispatchFlashAttentionBackwardDQ(
  qBuffer: GPUBuffer,
  kBuffer: GPUBuffer,
  vBuffer: GPUBuffer,
  lBuffer: GPUBuffer,
  dBuffer: GPUBuffer,
  dOBuffer: GPUBuffer,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: boolean,
): GPUBuffer {
  const ctx = requireContext();
  const device = ctx.device;

  const outputSizeBytes = batchSize * numHeads * seqLen * headDim * 4;

  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const wgsl = getTileIRWGSL(`bwdDQ:${headDim}`, () => makeBackwardDQSpec(headDim));
  const pipelineKey = `faBwdDQ:tile:${headDim}`;
  const pipeline = getPipeline(ctx, pipelineKey, wgsl);

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, outBuffer, configBuf]);

  trackBuffers(qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, outBuffer);

  const numQBlocks = Math.ceil(seqLen / BR);
  dispatchComputePass(
    pipeline,
    bindGroup,
    numQBlocks,
    numHeads,
    batchSize,
  );

  return outBuffer;
}

/**
 * Reset all module-local mutable state (pipeline cache, config buffer cache).
 */
export function resetAttentionKernelState(): void {
  configCache.clear();
  tileIRWGSLCache.clear();
}

/**
 * Dispatch FlashAttention backward dKV kernel.
 * Q,K,V,L,D,dO → dK: [B,H,N,D], dV: [B,H,N,D]
 */
export function dispatchFlashAttentionBackwardDKV(
  qBuffer: GPUBuffer,
  kBuffer: GPUBuffer,
  vBuffer: GPUBuffer,
  lBuffer: GPUBuffer,
  dBuffer: GPUBuffer,
  dOBuffer: GPUBuffer,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: boolean,
): { dKBuffer: GPUBuffer; dVBuffer: GPUBuffer } {
  const ctx = requireContext();
  const device = ctx.device;

  const outputSizeBytes = batchSize * numHeads * seqLen * headDim * 4;

  const dKBuffer = allocateOutputBuffer(outputSizeBytes);
  const dVBuffer = allocateOutputBuffer(outputSizeBytes);

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const BC_BW = 64;
  const wgsl = getTileIRWGSL(`bwdDKV:${headDim}`, () => makeBackwardDKVSpec(headDim));
  const pipelineKey = `faBwdDKV:tile:${headDim}`;
  const pipeline = getPipeline(ctx, pipelineKey, wgsl);

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, dKBuffer, dVBuffer, configBuf]);

  trackBuffers(qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, dKBuffer, dVBuffer);

  const numKVBlocks = Math.ceil(seqLen / BC_BW);
  dispatchComputePass(
    pipeline,
    bindGroup,
    numKVBlocks,
    numHeads,
    batchSize,
  );

  return { dKBuffer, dVBuffer };
}
