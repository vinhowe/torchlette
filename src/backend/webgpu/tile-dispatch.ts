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
  cachedCreateBindGroup,
  profiledCreateBindGroup,
} from "./bind-group-cache";
import { computeFlatChunkLayout } from "./chunked-dispatch";
import { dispatchComputePass, getPipeline } from "./dispatch";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { dtypeBytes } from "./shape-utils";
import type { AutotuneOptions } from "./tile-autotune";
import { autotuneTileKernel, getDefaultConfig } from "./tile-autotune";
import { compileTileKernel } from "./tile-compiler";
import type { AutotuneConfig, DataType, TileKernelSpec } from "./tile-ir";
import { resolveGrid } from "./tile-ir";
import { requireContext, trackSharedEncoderWrite } from "./webgpu-state";

// ============================================================================
// Chunked Binding Configuration
// ============================================================================

export interface ChunkedBindingConfig {
  /** Which bindings are chunked vs scalar. Key = binding name from spec. */
  modes: Record<string, "scalar" | "chunked">;
  /** Per-binding bytes-per-element override (e.g. cast source has different bpe). */
  bytesPerElement?: Record<string, number>;
  /** The uniform field name that holds the element count (patched per chunk). */
  sizeUniform: string;
  /** Total elements across all chunks. */
  totalElements: number;
  /** Max bytes per element across all chunked bindings (for chunk layout). */
  maxBytesPerElement: number;
  /** Override elements-per-alignment (for cast with mixed dtypes). */
  elementsPerAlignment?: number;
}

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
function uniformCacheKey(
  spec: TileKernelSpec,
  values: Record<string, number>,
): string {
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

  /**
   * Dispatch the kernel across flat element chunks for tensors exceeding
   * maxStorageBufferBindingSize. Same WGSL kernel is used for every chunk;
   * WebGPU offset/size on bind group entries makes index 0 map to chunk start.
   */
  dispatchChunked(
    buffers: Record<string, GPUBuffer>,
    uniforms: Record<string, number>,
    chunking: ChunkedBindingConfig,
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
export function createTileKernelDispatcher(
  spec: TileKernelSpec,
): TileKernelInstance {
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

  function getConfigBuffer(
    device: GPUDevice,
    key: string,
    sizeBytes: number,
    data: ArrayBufferView,
  ): GPUBuffer {
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

  // Shared binding assembly: iterates bindings in declaration order,
  // inserting the uniform config buffer at uniformBindingIndex.
  function buildBindings(
    bindingNames: string[],
    buffers: Record<string, GPUBuffer>,
    configBuf: GPUBuffer | null,
    visitor: (bindingIndex: number, name: string, buf: GPUBuffer) => void,
    visitUniform: (bindingIndex: number, buf: GPUBuffer) => void,
  ): void {
    const uniformIdx = spec.uniformBindingIndex;
    let bindingIndex = 0;
    for (let i = 0; i < bindingNames.length; i++) {
      if (
        configBuf &&
        uniformIdx !== undefined &&
        bindingIndex === uniformIdx
      ) {
        visitUniform(bindingIndex, configBuf);
        bindingIndex++;
      }
      const name = bindingNames[i];
      const buf = buffers[name];
      if (!buf) throw new Error(`Missing buffer for binding: ${name}`);
      visitor(bindingIndex, name, buf);
      bindingIndex++;
    }
    if (configBuf && (uniformIdx === undefined || bindingIndex <= uniformIdx)) {
      visitUniform(bindingIndex, configBuf);
    }
  }

  function prepareConfigBuffer(
    device: GPUDevice,
    uniforms: Record<string, number>,
    keyPrefix = "",
  ): GPUBuffer | null {
    if (Object.keys(spec.uniforms).length === 0) return null;
    const configKey = keyPrefix + uniformCacheKey(spec, uniforms);
    const { data, sizeBytes } = packUniforms(spec, uniforms);
    return getConfigBuffer(device, configKey, sizeBytes, data);
  }

  return {
    dispatch(
      buffers: Record<string, GPUBuffer>,
      uniforms: Record<string, number>,
    ): void {
      const ctx = requireContext();
      const device = ctx.device;
      const wgsl = getWGSL();
      const pipeline = getPipeline(ctx, wgsl, wgsl);

      const configBuf = prepareConfigBuffer(device, uniforms);
      const bindingNames = Object.keys(spec.bindings);
      const bufferArray: GPUBuffer[] = [];

      buildBindings(
        bindingNames,
        buffers,
        configBuf,
        (_idx, _name, buf) => bufferArray.push(buf),
        (_idx, buf) => bufferArray.push(buf),
      );

      const bindGroup = cachedCreateBindGroup(device, pipeline, bufferArray);
      for (const buf of bufferArray) trackSharedEncoderWrite(buf);

      const grid = resolveGrid(spec)(uniforms);
      dispatchComputePass(
        pipeline,
        bindGroup,
        ...(grid as [number, number, number]),
      );
    },

    dispatchChunked(
      buffers: Record<string, GPUBuffer>,
      uniforms: Record<string, number>,
      chunking: ChunkedBindingConfig,
    ): void {
      const ctx = requireContext();
      const device = ctx.device;

      const limits = device.limits;
      const maxBindingSize =
        limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
      const minAlignment =
        (limits as Record<string, number>)?.minStorageBufferOffsetAlignment ??
        256;

      const layout = computeFlatChunkLayout(
        chunking.totalElements,
        chunking.maxBytesPerElement,
        maxBindingSize,
        minAlignment,
        chunking.elementsPerAlignment,
      );

      const wgsl = getWGSL();
      const pipeline = getPipeline(ctx, wgsl, wgsl);
      const bindingNames = Object.keys(spec.bindings);
      const gridFn = resolveGrid(spec);

      for (let chunk = 0; chunk < layout.numChunks; chunk++) {
        const chunkStart = chunk * layout.elementsPerChunk;
        const chunkEnd = Math.min(
          chunkStart + layout.elementsPerChunk,
          chunking.totalElements,
        );
        const chunkSize = chunkEnd - chunkStart;

        const patchedUniforms = {
          ...uniforms,
          [chunking.sizeUniform]: chunkSize,
        };
        const configBuf = prepareConfigBuffer(
          device,
          patchedUniforms,
          "chunked:",
        );

        type Entry = {
          binding: number;
          resource: { buffer: GPUBuffer; offset?: number; size?: number };
        };
        const entries: Entry[] = [];

        buildBindings(
          bindingNames,
          buffers,
          configBuf,
          (idx, name, buf) => {
            const mode = chunking.modes[name] ?? "chunked";
            if (mode === "scalar") {
              entries.push({ binding: idx, resource: { buffer: buf } });
            } else {
              const bpe =
                chunking.bytesPerElement?.[name] ??
                dtypeBytes(spec.bindings[name].type);
              entries.push({
                binding: idx,
                resource: {
                  buffer: buf,
                  offset: chunkStart * bpe,
                  size: chunkSize * bpe,
                },
              });
            }
          },
          (idx, buf) =>
            entries.push({ binding: idx, resource: { buffer: buf } }),
        );

        const bindGroup = profiledCreateBindGroup(device, {
          layout: pipeline.getBindGroupLayout(0),
          entries,
        });
        for (const entry of entries)
          trackSharedEncoderWrite(entry.resource.buffer);

        dispatchComputePass(
          pipeline,
          bindGroup,
          ...(gridFn(patchedUniforms) as [number, number, number]),
        );
      }
    },

    getWGSL,

    reset(): void {
      configCache.clear();
      cachedWGSL = null;
    },
  };
}

// ============================================================================
// Auto-Tuning Dispatcher
// ============================================================================

export interface AutoTileKernelInstance {
  dispatch(
    buffers: Record<string, GPUBuffer>,
    uniforms: Record<string, number>,
  ): void;
  getConfig(): Record<string, number>;
  tune(
    buffers: Record<string, GPUBuffer>,
    uniforms: Record<string, number>,
    options?: AutotuneOptions,
  ): Promise<void>;
  reset(): void;
}

/**
 * Create a tile kernel dispatcher with autotuning support.
 * Starts with default config; call `tune()` to benchmark and select the best.
 */
export function createAutoTileKernelDispatcher(
  autoConfig: AutotuneConfig,
): AutoTileKernelInstance {
  let currentConfig = getDefaultConfig(autoConfig);
  let dispatcher = createTileKernelDispatcher(
    autoConfig.factory(currentConfig),
  );

  return {
    dispatch(buffers, uniforms) {
      dispatcher.dispatch(buffers, uniforms);
    },
    getConfig() {
      return { ...currentConfig };
    },
    async tune(buffers, uniforms, options) {
      const result = await autotuneTileKernel(
        autoConfig,
        buffers,
        uniforms,
        options,
      );
      currentConfig = result.config;
      dispatcher = createTileKernelDispatcher(
        autoConfig.factory(currentConfig),
      );
    },
    reset() {
      currentConfig = getDefaultConfig(autoConfig);
      dispatcher = createTileKernelDispatcher(
        autoConfig.factory(currentConfig),
      );
    },
  };
}
