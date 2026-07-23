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

import type { LazyIRNode } from "../../graph/types";
import {
  cachedCreateBindGroup,
  profiledCreateBindGroup,
} from "./bind-group-cache";
import { computeFlatChunkLayout } from "./chunked-dispatch";
import { dispatchComputePass, getPipeline } from "./dispatch";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE, dtypeBytes } from "./shape-utils";
import type { AutotuneOptions } from "./tile-autotune";
import { autotuneTileKernel, getDefaultConfig } from "./tile-autotune";
import { compileTileKernel } from "./tile-compiler";
import type { AutotuneConfig, TileKernelSpec } from "./tile-ir";
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
  const i32 = new Int32Array(buffer);

  for (let i = 0; i < entries.length; i++) {
    const [name, type] = entries[i];
    const val = values[name];
    if (val === undefined) {
      throw new Error(`Missing uniform value: ${name}`);
    }
    if (type === "f32") {
      f32[i] = val;
    } else if (type === "i32") {
      i32[i] = val;
    } else {
      u32[i] = val;
    }
  }

  return { data: new Uint8Array(buffer), sizeBytes };
}

/**
 * Build a cache key from uniform values, keyed on ALL uniform values (u32/i32
 * AND f32). A cached config buffer is shared by every dispatch with the same
 * key; a compiled-plan replay binds that buffer by IDENTITY and never rewrites
 * it (for STATIC configs — no volatile repack). So two configs whose f32 bytes
 * differ (e.g. inc-2a's now-static Adam `weight_decay`: 0.1 for one param
 * group, 0.0 for another) MUST get DISTINCT buffers — otherwise the
 * last-written config silently trains the other group's params under compiled
 * replay (the frozen-scalar-family failure mode; Gate 4 caught it). f32 fields
 * were excluded historically because per-step-varying f32s (Adam step_size)
 * rode a TAG_UNIFORM volatile repack that rewrote the shared buffer every
 * dispatch — inc-2a retired that (config is static), so the exclusion is no
 * longer safe. Volatile kernels are unaffected: distinct buffers still get
 * rewritten per dispatch, just fewer aliasing collisions.
 */
function uniformCacheKey(
  spec: TileKernelSpec,
  values: Record<string, number>,
): string {
  const parts: string[] = [];
  for (const name of Object.keys(spec.uniforms)) {
    parts.push(`${values[name]}`);
  }
  return parts.join(":");
}

// ============================================================================
// Tile Kernel Instance
// ============================================================================

/**
 * Re-derive the kernel's uniform values from the CURRENT step's plan node.
 * Provide this for any uniform that carries a per-step-varying VALUE (Adam's
 * bias-corrected step_size, GradScaler's inv_scale). Without it, a compiled
 * plan replays the config buffer's frozen record-time contents — the lowered
 * path rewrites configs on every dispatch, but replays bind them as-is.
 *
 * The repack receives the node executing when the config was recorded (its
 * payload is rebuilt by the frontend each step) and must return the FULL
 * uniform record (static fields included).
 */
export type VolatileUniformRepack = (
  node: LazyIRNode,
) => Record<string, number>;

/** Stage-4 plan/encode: a tile dispatch fully described without encoding. */
export interface TileKernelPlan {
  pipeline: GPUComputePipeline;
  /** Binding names in bind-group order; null marks the uniform config slot. */
  bindingOrder: (string | null)[];
  grid: [number, number, number];
  /** The instance's cached config buffer for these uniforms (null when the
   *  spec has no uniforms, or the cache has no entry yet — generators run
   *  post-execution, so a missing entry means a different uniform key). */
  configBuffer: GPUBuffer | null;
}

/**
 * Stage-4 chunked plan/encode: a chunked tile dispatch (>maxStorageBufferBinding
 * Size) fully described WITHOUT encoding. This is the SINGLE SOURCE for both the
 * execution path (dispatchChunked) and the stream generator (generateChunked
 * Binary/Unary), following the planChunkedFullReduction precedent: the per-chunk
 * geometry (numChunks, sub-range byte windows, per-chunk grid + packed uniforms)
 * is derived once from the capability-forced split `numChunks = f(bytes,
 * maxStorageBufferBindingSize)` so the recorded and generated command streams
 * cannot drift. Chunk geometry is step-invariant for a fixed-size buffer, so the
 * per-chunk params are baked (no volatile repack) — exactly as the chunked
 * full-reduction bakes them.
 */
export interface TileKernelChunkedPlan {
  pipeline: GPUComputePipeline;
  numChunks: number;
  /** Binding names in bind-group order; null marks the uniform config slot. */
  bindingOrder: (string | null)[];
  chunks: Array<{
    /** Per-binding sub-range aligned to bindingOrder (null = whole buffer /
     *  scalar binding / uniform slot). */
    ranges: (({ offset: number; size: number }) | null)[];
    /** Packed uniform bytes for this chunk (chunkSize patched into sizeUniform). */
    paramsData: Uint8Array;
    grid: [number, number, number];
  }>;
}

export interface TileKernelInstance {
  /**
   * Derive the dispatch plan WITHOUT encoding or mutating the config cache.
   * Single source: the same pipeline/binding/grid computations dispatch()
   * performs (stream generation consumes this; see stream-generate.ts).
   */
  plan(uniforms: Record<string, number>): TileKernelPlan;

  /**
   * Dispatch the kernel. Buffers must match the binding names in the spec.
   * Uniforms must provide all declared uniform values.
   */
  dispatch(
    buffers: Record<string, GPUBuffer>,
    uniforms: Record<string, number>,
    volatileRepack?: VolatileUniformRepack,
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
    volatileRepack?: VolatileUniformRepack,
  ): void;

  /**
   * Derive the chunked dispatch plan WITHOUT encoding or mutating the config
   * cache. Single source shared with dispatchChunked (both consume
   * computeChunkGeometry) so the stream generator's per-chunk commands match the
   * recorded ones by construction. See TileKernelChunkedPlan.
   */
  planChunked(
    uniforms: Record<string, number>,
    chunking: ChunkedBindingConfig,
  ): TileKernelChunkedPlan;

  /** Get the compiled WGSL source (for debugging/testing). */
  getWGSL(): string;

  /**
   * Stage-4 stream generation: build the volatile-uniform PACK function
   * (`node => packUniforms(spec, volatileRepack(node)).data`) the generated
   * TAG_UNIFORM carries. A kernel with per-step-varying config (Adam's bias-
   * corrected step_size) MUST carry this packer, not a no-op — else the config
   * freezes at the build-time value (the frozen-step_size class).
   */
  volatilePack(
    volatileRepack: VolatileUniformRepack,
  ): (node: LazyIRNode) => ArrayBufferView;

  /** Reset cached pipelines and config buffers. */
  reset(): void;
}

/**
 * Compute the bind-group order (spec bindings interleaved with the uniform
 * config slot) for a spec. Shared by plan() and computeChunkGeometry so their
 * null-placement for the uniform slot cannot drift.
 */
function computeBindingOrder(
  spec: TileKernelSpec,
  hasConfig: boolean,
): (string | null)[] {
  const bindingNames = Object.keys(spec.bindings);
  const bindingOrder: (string | null)[] = [];
  const uniformIdx = spec.uniformBindingIndex;
  let bindingIndex = 0;
  for (let i = 0; i < bindingNames.length; i++) {
    if (hasConfig && uniformIdx !== undefined && bindingIndex === uniformIdx) {
      bindingOrder.push(null);
      bindingIndex++;
    }
    bindingOrder.push(bindingNames[i]);
    bindingIndex++;
  }
  if (hasConfig && (uniformIdx === undefined || bindingIndex <= uniformIdx)) {
    bindingOrder.push(null);
  }
  return bindingOrder;
}

/** Per-chunk geometry shared by dispatchChunked (execution) and planChunked
 *  (stream generation) — the SINGLE SOURCE for the capability-forced split
 *  `numChunks = f(bytes, maxStorageBufferBindingSize)`. Returns, per chunk, the
 *  sub-range byte windows (aligned to bindingOrder), the chunk element count,
 *  and the per-chunk uniform record with `sizeUniform` patched. Neither caller
 *  packs or dispatches here, so this touches no config cache / recording. */
interface ChunkGeometry {
  numChunks: number;
  bindingOrder: (string | null)[];
  chunks: Array<{
    chunkSize: number;
    ranges: (({ offset: number; size: number }) | null)[];
    patchedUniforms: Record<string, number>;
  }>;
}

function computeChunkGeometry(
  spec: TileKernelSpec,
  uniforms: Record<string, number>,
  chunking: ChunkedBindingConfig,
  limits: { maxStorageBufferBindingSize?: number } | undefined,
): ChunkGeometry {
  const maxBindingSize =
    limits?.maxStorageBufferBindingSize ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE;
  const minAlignment =
    (limits as Record<string, number> | undefined)
      ?.minStorageBufferOffsetAlignment ?? 256;

  const layout = computeFlatChunkLayout(
    chunking.totalElements,
    chunking.maxBytesPerElement,
    maxBindingSize,
    minAlignment,
    chunking.elementsPerAlignment,
  );

  const hasConfig = Object.keys(spec.uniforms).length > 0;
  const bindingOrder = computeBindingOrder(spec, hasConfig);

  const chunks: ChunkGeometry["chunks"] = [];
  for (let chunk = 0; chunk < layout.numChunks; chunk++) {
    const chunkStart = chunk * layout.elementsPerChunk;
    const chunkEnd = Math.min(
      chunkStart + layout.elementsPerChunk,
      chunking.totalElements,
    );
    const chunkSize = chunkEnd - chunkStart;
    const ranges: (({ offset: number; size: number }) | null)[] =
      bindingOrder.map((name) => {
        if (name === null) return null; // uniform config slot
        const mode = chunking.modes[name] ?? "chunked";
        if (mode === "scalar") return null; // whole buffer
        const bpe =
          chunking.bytesPerElement?.[name] ??
          dtypeBytes(spec.bindings[name].type);
        return { offset: chunkStart * bpe, size: chunkSize * bpe };
      });
    chunks.push({
      chunkSize,
      ranges,
      patchedUniforms: { ...uniforms, [chunking.sizeUniform]: chunkSize },
    });
  }
  return { numChunks: layout.numChunks, bindingOrder, chunks };
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
  const configCache = new Map<
    string,
    { buffer: GPUBuffer; lastData: Uint8Array }
  >();

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
    data: Uint8Array,
  ): GPUBuffer {
    let entry = configCache.get(key);
    if (!entry) {
      entry = {
        buffer: device.createBuffer({
          size: sizeBytes,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
        lastData: data.slice(),
      };
      configCache.set(key, entry);
    }
    if (entry.lastData.length === data.length) {
      entry.lastData.set(data);
    } else {
      entry.lastData = data.slice();
    }
    device.queue.writeBuffer(entry.buffer, 0, data);
    return entry.buffer;
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
    plan(uniforms: Record<string, number>): TileKernelPlan {
      const ctx = requireContext();
      const wgsl = getWGSL();
      const pipeline = getPipeline(ctx, wgsl, wgsl);
      const bindingNames = Object.keys(spec.bindings);
      const hasConfig = Object.keys(spec.uniforms).length > 0;
      let configBuffer: GPUBuffer | null = null;
      if (hasConfig) {
        const configKey = uniformCacheKey(spec, uniforms);
        configBuffer = configCache.get(configKey)?.buffer ?? null;
      }
      const bindingOrder: (string | null)[] = [];
      const uniformIdx = spec.uniformBindingIndex;
      let bindingIndex = 0;
      for (let i = 0; i < bindingNames.length; i++) {
        if (
          hasConfig &&
          uniformIdx !== undefined &&
          bindingIndex === uniformIdx
        ) {
          bindingOrder.push(null);
          bindingIndex++;
        }
        bindingOrder.push(bindingNames[i]);
        bindingIndex++;
      }
      if (
        hasConfig &&
        (uniformIdx === undefined || bindingIndex <= uniformIdx)
      ) {
        bindingOrder.push(null);
      }
      const g = resolveGrid(spec)(uniforms);
      const grid: [number, number, number] = [g[0] ?? 1, g[1] ?? 1, g[2] ?? 1];
      return { pipeline, bindingOrder, grid, configBuffer };
    },

    dispatch(
      buffers: Record<string, GPUBuffer>,
      uniforms: Record<string, number>,
      _volatileRepack?: VolatileUniformRepack,
    ): void {
      const ctx = requireContext();
      const device = ctx.device;
      const wgsl = getWGSL();
      const pipeline = getPipeline(ctx, wgsl, wgsl);

      const configBuf = prepareConfigBuffer(device, uniforms, "");
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
      _volatileRepack?: VolatileUniformRepack,
    ): void {
      const ctx = requireContext();
      const device = ctx.device;

      // Single source: derive the per-chunk geometry (same split planChunked
      // and the stream generator consume). This path adds only the config-buffer
      // + volatile-repack handling on top.
      const geom = computeChunkGeometry(spec, uniforms, chunking, device.limits);

      const wgsl = getWGSL();
      const pipeline = getPipeline(ctx, wgsl, wgsl);
      const bindingNames = Object.keys(spec.bindings);
      const gridFn = resolveGrid(spec);

      for (const chunk of geom.chunks) {
        const patchedUniforms = chunk.patchedUniforms;
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
        // The geometry's ranges are aligned to bindingOrder (uniform slot = the
        // null entries); buildBindings walks the spec bindings + inserts the
        // config buffer at the same index, so we index geom.ranges by the same
        // interleaved position via a shared cursor.
        let orderIdx = 0;
        buildBindings(
          bindingNames,
          buffers,
          configBuf,
          (idx, _name, buf) => {
            const range = chunk.ranges[orderIdx++];
            entries.push({
              binding: idx,
              resource: range
                ? { buffer: buf, offset: range.offset, size: range.size }
                : { buffer: buf },
            });
          },
          (idx, buf) => {
            orderIdx++; // skip the uniform slot's (null) range entry
            entries.push({ binding: idx, resource: { buffer: buf } });
          },
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

    planChunked(
      uniforms: Record<string, number>,
      chunking: ChunkedBindingConfig,
    ): TileKernelChunkedPlan {
      const ctx = requireContext();
      const wgsl = getWGSL();
      const pipeline = getPipeline(ctx, wgsl, wgsl);
      const gridFn = resolveGrid(spec);
      const geom = computeChunkGeometry(
        spec,
        uniforms,
        chunking,
        ctx.device.limits,
      );
      return {
        pipeline,
        numChunks: geom.numChunks,
        bindingOrder: geom.bindingOrder,
        chunks: geom.chunks.map((c) => ({
          ranges: c.ranges,
          // Bake this chunk's params (chunk geometry is step-invariant for a
          // fixed buffer — no volatile repack, exactly as the chunked full
          // reduction bakes them). The generator emits one params slot per chunk.
          paramsData: packUniforms(spec, c.patchedUniforms).data,
          grid: gridFn(c.patchedUniforms) as [number, number, number],
        })),
      };
    },

    getWGSL,

    volatilePack(volatileRepack: VolatileUniformRepack) {
      // Identical to the pack recorded in getConfigBuffer (line ~257), so a
      // generated TAG_UNIFORM replays byte-for-byte what the recorded path
      // would write each step.
      return (node: LazyIRNode): ArrayBufferView =>
        packUniforms(spec, volatileRepack(node)).data;
    },

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
    volatileRepack?: VolatileUniformRepack,
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
    dispatch(buffers, uniforms, volatileRepack) {
      dispatcher.dispatch(buffers, uniforms, volatileRepack);
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
