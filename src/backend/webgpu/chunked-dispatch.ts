/**
 * Generic flat-element chunking abstraction for ops that exceed
 * maxStorageBufferBindingSize. Extracts the shared scaffolding
 * (chunk layout, 2D dispatch, per-chunk bind group creation) into
 * two reusable functions.
 *
 * Used by: binary, unary, where, cast, stridedScatterCopy/Add.
 * NOT used by: gather/scatterAdd (dimension-specific) or
 * contiguousChunked (2D tile chunking).
 */

import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { WORKGROUP_SIZE, MAX_WORKGROUPS_PER_DIM } from "./shape-utils";
import { requireContext } from "./gpu-context";
import { dispatchComputePass, getPipeline } from "./dispatch";
import {
  createUniformBuffer,
  releaseUniformBuffer,
  profiledCreateBindGroup,
} from "./bind-group-cache";

// ============================================================================
// Types
// ============================================================================

export interface FlatChunkLayout {
  elementsPerChunk: number;
  numChunks: number;
  use2D: boolean;
  gridSizeX: number;
}

export interface ChunkedBinding {
  buffer: GPUBuffer;
  /** 'scalar' binds the full (small) buffer each chunk; 'chunked' binds a sub-range. */
  mode: "scalar" | "chunked";
  /** Per-binding bytes-per-element override (e.g. cast source has different bpe than output). */
  bytesPerElement?: number;
}

// ============================================================================
// Layout computation
// ============================================================================

/**
 * Compute the flat chunk layout for a given total element count and device limits.
 *
 * @param totalElements   Total output elements.
 * @param maxBytesPerElement  Largest bytes-per-element across all chunked bindings
 *                            (determines how many elements fit in one binding).
 * @param maxBindingSize  `device.limits.maxStorageBufferBindingSize`.
 * @param minAlignment    `device.limits.minStorageBufferOffsetAlignment`.
 * @param elementsPerAlignment  Override alignment granularity in elements.
 *                              Defaults to `minAlignment / maxBytesPerElement`.
 *                              Cast uses `lcm(srcEpa, dstEpa)`.
 */
export function computeFlatChunkLayout(
  totalElements: number,
  maxBytesPerElement: number,
  maxBindingSize: number,
  minAlignment: number,
  elementsPerAlignment?: number,
): FlatChunkLayout {
  const epa = elementsPerAlignment ?? (minAlignment / maxBytesPerElement);
  const maxElementsPerChunk = Math.floor(maxBindingSize / maxBytesPerElement);
  const elementsPerChunk =
    Math.floor(maxElementsPerChunk / epa) * epa;

  const numChunks = Math.ceil(totalElements / elementsPerChunk);

  const maxWorkgroups = Math.ceil(elementsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;
  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;

  return { elementsPerChunk, numChunks, use2D, gridSizeX };
}

// ============================================================================
// Chunked dispatch
// ============================================================================

/**
 * Dispatch a shader across flat element chunks.
 *
 * Binding order convention: `[...inputs, output, params]` — matches all
 * existing chunked shaders.
 *
 * Each chunk creates a bind group where scalar inputs are bound in full and
 * chunked inputs (and the output) are bound as sub-ranges at the appropriate
 * byte offset.  A params uniform buffer is created per chunk (default: single
 * `chunkSize` u32; override via `createChunkParams`).
 */
export function dispatchFlatChunked(config: {
  key: string;
  shader: string;
  layout: FlatChunkLayout;
  inputs: ChunkedBinding[];
  outBuffer: GPUBuffer;
  outBytesPerElement: number;
  totalElements: number;
  createChunkParams?: (
    device: GPUDevice,
    chunkSize: number,
    chunkStart: number,
  ) => GPUBuffer;
  releaseChunkParams?: (buf: GPUBuffer) => void;
}): void {
  const ctx = requireContext();
  const { layout, inputs, outBuffer, outBytesPerElement, totalElements } = config;
  const pipeline = getPipeline(ctx, config.key, config.shader);
  const createParams = config.createChunkParams ?? defaultCreateParams;
  const releaseParams = config.releaseChunkParams ?? releaseUniformBuffer;

  for (let chunk = 0; chunk < layout.numChunks; chunk++) {
    const chunkStart = chunk * layout.elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + layout.elementsPerChunk, totalElements);
    const chunkSize = chunkEnd - chunkStart;

    const paramsBuffer = createParams(ctx.device, chunkSize, chunkStart);

    // Build bind group entries: [...inputs, output, params]
    const entries: Array<{
      binding: number;
      resource: { buffer: GPUBuffer; offset?: number; size?: number };
    }> = [];
    let bindingIdx = 0;

    for (const input of inputs) {
      if (input.mode === "scalar") {
        entries.push({ binding: bindingIdx, resource: { buffer: input.buffer } });
      } else {
        const bpe = input.bytesPerElement ?? outBytesPerElement;
        entries.push({
          binding: bindingIdx,
          resource: {
            buffer: input.buffer,
            offset: chunkStart * bpe,
            size: chunkSize * bpe,
          },
        });
      }
      bindingIdx++;
    }

    // Output (always chunked)
    entries.push({
      binding: bindingIdx,
      resource: {
        buffer: outBuffer,
        offset: chunkStart * outBytesPerElement,
        size: chunkSize * outBytesPerElement,
      },
    });
    bindingIdx++;

    // Params
    entries.push({ binding: bindingIdx, resource: { buffer: paramsBuffer } });

    const bindGroup = profiledCreateBindGroup(ctx.device, {
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });

    const chunkWorkgroups = Math.ceil(chunkSize / WORKGROUP_SIZE);
    const dispatchX = layout.use2D
      ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM)
      : chunkWorkgroups;
    const dispatchY = layout.use2D
      ? Math.ceil(chunkWorkgroups / dispatchX)
      : 1;

    dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);

    releaseParams(paramsBuffer);
  }
}

function defaultCreateParams(
  device: GPUDevice,
  chunkSize: number,
  _chunkStart: number,
): GPUBuffer {
  return createUniformBuffer(device, chunkSize);
}
