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
import { profiledCreateBindGroup } from "./bind-group-cache";
import { computeFlatChunkLayout } from "./chunked-dispatch";
import { requireContext } from "./webgpu-state";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { compileTileKernel } from "./tile-compiler";
import type { TileKernelSpec, UniformType, AutotuneConfig, DataType } from "./tile-ir";
import { resolveGrid } from "./tile-ir";
import { autotuneTileKernel, getDefaultConfig, type AutotuneOptions } from "./tile-autotune";
import { analyzeAccessPatterns, reportAccessPatterns } from "./tile-access-analysis";

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

/** Bytes per element for a tile-IR DataType. */
function dataTypeBpe(dt: DataType): number {
  return dt === "f16" ? 2 : 4;
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

      // Pipeline (cached via context.pipelines — key by WGSL content for uniqueness)
      const wgsl = getWGSL();
      const pipeline = getPipeline(ctx, wgsl, wgsl);

      // Build buffer array in binding order, respecting uniformBindingIndex
      const bindingNames = Object.keys(spec.bindings);
      const bufferArray: GPUBuffer[] = [];
      const hasUniforms = Object.keys(spec.uniforms).length > 0;
      const uniformIdx = spec.uniformBindingIndex;
      let configBuf: GPUBuffer | null = null;

      if (hasUniforms) {
        const configKey = uniformCacheKey(spec, uniforms);
        const { data, sizeBytes } = packUniforms(spec, uniforms);
        configBuf = getConfigBuffer(device, configKey, sizeBytes, data);
      }

      let bindingIndex = 0;
      for (let i = 0; i < bindingNames.length; i++) {
        // Insert uniform at the specified binding index
        if (configBuf && uniformIdx !== undefined && bindingIndex === uniformIdx) {
          bufferArray.push(configBuf);
          bindingIndex++;
        }
        const buf = buffers[bindingNames[i]];
        if (!buf) {
          throw new Error(`Missing buffer for binding: ${bindingNames[i]}`);
        }
        bufferArray.push(buf);
        bindingIndex++;
      }

      // Append uniform at end if not already inserted
      if (configBuf && (uniformIdx === undefined || bindingIndex <= uniformIdx)) {
        bufferArray.push(configBuf);
      }

      // Bind group
      const bindGroup = cachedCreateBindGroup(device, pipeline, bufferArray);

      // Track writes
      for (const buf of bufferArray) {
        trackSharedEncoderWrite(buf);
      }

      // Compute grid dimensions (explicit or auto-inferred)
      const gridFn = resolveGrid(spec);
      const grid = gridFn(uniforms);

      // Dispatch
      dispatchComputePass(pipeline, bindGroup, ...grid);
    },

    dispatchChunked(
      buffers: Record<string, GPUBuffer>,
      uniforms: Record<string, number>,
      chunking: ChunkedBindingConfig,
    ): void {
      const ctx = requireContext();
      const device = ctx.device;

      const limits = device.limits;
      const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
      const minAlignment = (limits as Record<string, number>)?.minStorageBufferOffsetAlignment ?? 256;

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
      const hasUniforms = Object.keys(spec.uniforms).length > 0;
      const uniformIdx = spec.uniformBindingIndex;
      const gridFn = resolveGrid(spec);

      for (let chunk = 0; chunk < layout.numChunks; chunk++) {
        const chunkStart = chunk * layout.elementsPerChunk;
        const chunkEnd = Math.min(chunkStart + layout.elementsPerChunk, chunking.totalElements);
        const chunkSize = chunkEnd - chunkStart;

        // Patch the size uniform for this chunk
        const patchedUniforms = { ...uniforms, [chunking.sizeUniform]: chunkSize };

        // Pack uniforms into config buffer (cached by content)
        let configBuf: GPUBuffer | null = null;
        if (hasUniforms) {
          const configKey = `chunked:${uniformCacheKey(spec, patchedUniforms)}`;
          const { data, sizeBytes } = packUniforms(spec, patchedUniforms);
          configBuf = getConfigBuffer(device, configKey, sizeBytes, data);
        }

        // Build bind group entries with offset/size for chunked bindings
        const entries: Array<{
          binding: number;
          resource: { buffer: GPUBuffer; offset?: number; size?: number };
        }> = [];

        let bindingIndex = 0;

        for (let i = 0; i < bindingNames.length; i++) {
          // Insert uniform at the specified binding index
          if (configBuf && uniformIdx !== undefined && bindingIndex === uniformIdx) {
            entries.push({ binding: bindingIndex, resource: { buffer: configBuf } });
            bindingIndex++;
          }

          const name = bindingNames[i];
          const buf = buffers[name];
          if (!buf) {
            throw new Error(`Missing buffer for binding: ${name}`);
          }

          const mode = chunking.modes[name] ?? "chunked";
          if (mode === "scalar") {
            entries.push({ binding: bindingIndex, resource: { buffer: buf } });
          } else {
            const bpe = chunking.bytesPerElement?.[name] ?? dataTypeBpe(spec.bindings[name].type);
            entries.push({
              binding: bindingIndex,
              resource: {
                buffer: buf,
                offset: chunkStart * bpe,
                size: chunkSize * bpe,
              },
            });
          }
          bindingIndex++;
        }

        // Append uniform at end if not already inserted
        if (configBuf && (uniformIdx === undefined || bindingIndex <= uniformIdx)) {
          entries.push({ binding: bindingIndex, resource: { buffer: configBuf } });
        }

        const bindGroup = profiledCreateBindGroup(device, {
          layout: pipeline.getBindGroupLayout(0),
          entries,
        });

        // Track writes
        for (const entry of entries) {
          trackSharedEncoderWrite(entry.resource.buffer);
        }

        // Compute grid for this chunk's element count
        const grid = gridFn(patchedUniforms);

        // Dispatch
        dispatchComputePass(pipeline, bindGroup, ...grid);
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
// Auto-Tunable Tile Kernel Dispatcher
// ============================================================================

export interface AutoTileKernelInstance {
  /** Dispatch with cached best config (or default if not yet tuned). */
  dispatch(
    buffers: Record<string, GPUBuffer>,
    uniforms: Record<string, number>,
  ): void;

  /** Run autotuning for given buffers/uniforms. Caches result. */
  tune(
    buffers: Record<string, GPUBuffer>,
    uniforms: Record<string, number>,
    options?: AutotuneOptions,
  ): Promise<void>;

  /** Get the WGSL source for the current (best or default) config. */
  getWGSL(): string;

  /** Get the current config values. */
  getConfig(): Record<string, number>;

  /** Reset cached state and tuning results. */
  reset(): void;
}

/**
 * Create a self-tuning tile kernel dispatcher from an AutotuneConfig.
 *
 * - `dispatch()` uses the cached best config (or default)
 * - `tune()` runs autotuning, caches the winner, rebuilds the dispatcher
 */
export function createAutoTileKernelDispatcher(
  autoConfig: AutotuneConfig,
): AutoTileKernelInstance {
  let currentConfig = getDefaultConfig(autoConfig);
  let dispatcher = createTileKernelDispatcher(autoConfig.factory(currentConfig));

  return {
    dispatch(buffers: Record<string, GPUBuffer>, uniforms: Record<string, number>): void {
      dispatcher.dispatch(buffers, uniforms);
    },

    async tune(
      buffers: Record<string, GPUBuffer>,
      uniforms: Record<string, number>,
      options?: AutotuneOptions,
    ): Promise<void> {
      const result = await autotuneTileKernel(autoConfig, buffers, uniforms, options);
      currentConfig = result.config;
      dispatcher = createTileKernelDispatcher(autoConfig.factory(currentConfig));
    },

    getWGSL(): string {
      return dispatcher.getWGSL();
    },

    getConfig(): Record<string, number> {
      return { ...currentConfig };
    },

    reset(): void {
      currentConfig = getDefaultConfig(autoConfig);
      dispatcher = createTileKernelDispatcher(autoConfig.factory(currentConfig));
    },
  };
}
