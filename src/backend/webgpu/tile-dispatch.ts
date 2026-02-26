/**
 * Tile Kernel Dispatch
 *
 * Connects tile IR kernels to the existing WebGPU dispatch infrastructure:
 * defineKernel (pipeline caching), cachedCreateBindGroup, dispatchComputePass,
 * trackSharedEncoderWrite, allocateOutputBuffer.
 *
 * Usage:
 * ```typescript
 * const myKernel = createTileKernelDispatcher(mySpec);
 * const outBuf = myKernel.dispatch(
 *   { x: xBuffer, weight: wBuffer, bias: bBuffer, output: outBuffer },
 *   { numRows, featureDim, eps },
 * );
 * ```
 */

import {
  dispatchComputePass,
  trackSharedEncoderWrite,
  cachedCreateBindGroup,
  getPipeline,
} from "./index";
import { requireContext } from "./webgpu-state";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { compileTileKernel } from "./tile-compiler";
import type { TileKernelSpec, UniformType } from "./tile-ir";

// ============================================================================
// Config Buffer Packing
// ============================================================================

/**
 * Pack uniform values into a Uint8Array for writeBuffer.
 * Fields are packed in declaration order, padded to 16-byte alignment.
 */
function packUniforms(
  spec: TileKernelSpec,
  values: Record<string, number>,
): { data: Uint8Array; sizeBytes: number } {
  const entries = Object.entries(spec.uniforms);
  const fieldCount = entries.length;
  // Pad to 16-byte alignment
  const paddedFieldCount = Math.ceil(fieldCount / 4) * 4;
  const sizeBytes = paddedFieldCount * 4;

  const buffer = new ArrayBuffer(sizeBytes);
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);

  for (let i = 0; i < entries.length; i++) {
    const [name, type] = entries[i];
    const val = values[name];
    if (val === undefined) {
      throw new Error(`Missing uniform value: ${name}`);
    }
    if (type === "f32") {
      f32[i] = val;
    } else {
      u32[i] = val;
    }
  }

  return { data: new Uint8Array(buffer), sizeBytes };
}

/**
 * Build a cache key from uniform values.
 * Only dimension-like uniforms (u32, i32) are included — f32 values like eps
 * change the config buffer data but not the pipeline or struct layout.
 */
function uniformCacheKey(spec: TileKernelSpec, values: Record<string, number>): string {
  const parts: string[] = [];
  for (const [name, type] of Object.entries(spec.uniforms)) {
    if (type !== "f32") {
      parts.push(`${values[name]}`);
    }
  }
  return parts.join(":");
}

// ============================================================================
// Tile Kernel Instance
// ============================================================================

export interface TileKernelInstance {
  /**
   * Dispatch the kernel. Buffers must match the binding names in the spec.
   * Uniforms must provide all declared uniform values.
   */
  dispatch(
    buffers: Record<string, GPUBuffer>,
    uniforms: Record<string, number>,
  ): void;

  /** Get the compiled WGSL source (for debugging/testing). */
  getWGSL(): string;

  /** Reset cached pipelines and config buffers. */
  reset(): void;
}

/**
 * Create a dispatchable tile kernel from a spec.
 *
 * The returned instance caches the pipeline and config buffers via defineKernel.
 * Each dispatch call updates config data and encodes on the shared encoder.
 */
export function createTileKernelDispatcher(spec: TileKernelSpec): TileKernelInstance {
  // Inline pipeline + config buffer caching (replaces defineKernel)
  const configCache = new Map<string, GPUBuffer>();

  // Compile WGSL once (the spec is static)
  let cachedWGSL: string | null = null;
  function getWGSL(): string {
    if (!cachedWGSL) {
      cachedWGSL = compileTileKernel(spec);
    }
    return cachedWGSL;
  }

  function getConfigBuffer(device: GPUDevice, key: string, sizeBytes: number, data: ArrayBufferView): GPUBuffer {
    let buf = configCache.get(key);
    if (!buf) {
      buf = device.createBuffer({
        size: sizeBytes,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      configCache.set(key, buf);
    }
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  return {
    dispatch(buffers: Record<string, GPUBuffer>, uniforms: Record<string, number>): void {
      const ctx = requireContext();
      const device = ctx.device;

      // Pipeline (cached via context.pipelines)
      const pipeline = getPipeline(ctx, spec.name, getWGSL());

      // Build buffer array in binding order
      const bindingNames = Object.keys(spec.bindings);
      const bufferArray: GPUBuffer[] = [];
      for (const name of bindingNames) {
        const buf = buffers[name];
        if (!buf) {
          throw new Error(`Missing buffer for binding: ${name}`);
        }
        bufferArray.push(buf);
      }

      // Config uniform buffer (only when uniforms are declared)
      const hasUniforms = Object.keys(spec.uniforms).length > 0;
      if (hasUniforms) {
        const configKey = uniformCacheKey(spec, uniforms);
        const { data, sizeBytes } = packUniforms(spec, uniforms);
        const configBuf = getConfigBuffer(device, configKey, sizeBytes, data);
        bufferArray.push(configBuf);
      }

      // Bind group
      const bindGroup = cachedCreateBindGroup(device, pipeline, bufferArray);

      // Track writes
      for (const buf of bufferArray) {
        trackSharedEncoderWrite(buf);
      }

      // Compute grid dimensions
      const grid = spec.grid(uniforms);

      // Dispatch
      dispatchComputePass(pipeline, bindGroup, ...grid);
    },

    getWGSL,

    reset(): void {
      configCache.clear();
      cachedWGSL = null;
    },
  };
}
