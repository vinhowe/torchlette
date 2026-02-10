/**
 * Matmul dispatch: kernel selection, pipeline caching, and execution.
 */

import { submitOrCollect, getSharedEncoderInstance, getCurrentOpLabel, createParamsBuffer as sharedCreateParamsBuffer, releaseParamsBuffer, cachedCreateBindGroup } from "../index";
import { profileApiCall, getTimestampWrites } from "../profiler";
import {
  type CodegenOptions,
  type EpilogueConfig,
  generateTiledMatmulShader,
  getShaderCacheKey,
} from "./codegen";
import {
  classifyShape,
  DEFAULT_CONFIG,
  type DType,
  getSubgroupSupport,
  getTransposeMode,
  getWorkgroupSize,
  type MatmulKernelConfig,
  type ShapeClass,
  validateConfig,
} from "./types";
import { autotune, type BenchmarkFn, cacheTuningResult } from "./autotune";

// GPU types (matching index.ts)
type GPUBuffer = {
  getMappedRange(): ArrayBuffer;
  mapAsync(mode: number): Promise<void>;
  unmap(): void;
  destroy(): void;
};

type GPUComputePipeline = {
  getBindGroupLayout(index: number): unknown;
};

type GPUComputePass = {
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
  setBindGroup(index: number, group: unknown): void;
  setPipeline(pipeline: GPUComputePipeline): void;
};

type GPUCommandEncoder = {
  beginComputePass(descriptor?: any): GPUComputePass;
  finish(): unknown;
};

type GPUQueue = {
  submit(commands: unknown[]): void;
  writeBuffer(buffer: GPUBuffer, offset: number, data: ArrayBufferView): void;
  onSubmittedWorkDone(): Promise<void>;
};

type GPUDevice = {
  createBindGroup(descriptor: {
    layout: unknown;
    entries: Array<{ binding: number; resource: { buffer: GPUBuffer } }>;
  }): unknown;
  createBuffer(descriptor: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): GPUBuffer;
  createCommandEncoder(): GPUCommandEncoder;
  createComputePipeline(descriptor: {
    layout: "auto";
    compute: { module: unknown; entryPoint: string };
  }): GPUComputePipeline;
  createShaderModule(descriptor: { code: string }): unknown;
  queue: GPUQueue;
};

const GPUBufferUsage = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};

/**
 * Pipeline cache for compiled matmul kernels.
 */
const pipelineCache = new Map<string, GPUComputePipeline>();

/**
 * Tuning results cache (in-memory only for now).
 */
const tuningCache = new Map<string, MatmulKernelConfig>();

/**
 * Global autotune mode counter.
 * When > 0, matmul dispatch will run autotuning for new shapes.
 * Uses a counter to support nested compile regions and lazy execution.
 */
let autotuneCounter = 0;

/**
 * Shapes currently being autotuned (to prevent recursive autotuning).
 */
const autotuningInProgress = new Set<string>();

/**
 * Enable or disable autotune mode.
 * When enabled=true, increments the counter.
 * When enabled=false, resets the counter to 0 (used by tests).
 */
export function setAutotuneEnabled(enabled: boolean): void {
  if (enabled) {
    autotuneCounter++;
  } else {
    autotuneCounter = 0;
  }
}

/**
 * Check if autotune mode is enabled.
 */
export function isAutotuneEnabled(): boolean {
  return autotuneCounter > 0;
}

/**
 * Clear the pipeline cache.
 */
export function clearPipelineCache(): void {
  pipelineCache.clear();
}

/**
 * Get tuning cache key.
 */
function getTuningKey(shapeClass: ShapeClass, dtype: DType): string {
  return `${shapeClass}_${dtype}`;
}

/**
 * Get the best kernel config for a shape class (from cache or default).
 */
export function getConfigForShape(
  shapeClass: ShapeClass,
  dtype: DType,
): MatmulKernelConfig {
  const key = getTuningKey(shapeClass, dtype);
  const cached = tuningCache.get(key);
  if (cached) {
    return cached;
  }
  return DEFAULT_CONFIG;
}

/**
 * Store a tuning result.
 */
export function setTuningResult(
  shapeClass: ShapeClass,
  dtype: DType,
  config: MatmulKernelConfig,
): void {
  const key = getTuningKey(shapeClass, dtype);
  tuningCache.set(key, config);
}

/**
 * Clear the dispatch tuning cache.
 * Called by tests to reset state between runs.
 */
export function clearDispatchTuningCache(): void {
  tuningCache.clear();
}

/**
 * Run autotuning for a specific shape if autotune mode is enabled and no cached result exists.
 * Returns the best config (from autotune, cache, or default).
 *
 * This is called during matmul dispatch when autotune is enabled.
 */
async function autotuneIfNeeded(
  device: GPUDevice,
  queue: GPUQueue,
  m: number,
  n: number,
  k: number,
  dtype: DType,
): Promise<MatmulKernelConfig> {
  const shapeClass = classifyShape(m, n, k, 1);
  const key = getTuningKey(shapeClass, dtype);

  // Already have a tuned config
  const cached = tuningCache.get(key);
  if (cached) {
    return cached;
  }

  // Not in autotune mode - use default
  if (!isAutotuneEnabled()) {
    return DEFAULT_CONFIG;
  }

  // Prevent recursive autotuning (autotune benchmarks call dispatchTiledMatmul)
  if (autotuningInProgress.has(key)) {
    return DEFAULT_CONFIG;
  }

  // Check if subgroups are supported
  const subgroupSupport = getSubgroupSupport();
  const includeSubgroups = subgroupSupport?.supported ?? false;

  // Mark as in progress
  autotuningInProgress.add(key);

  try {
    // Create benchmark function
    const benchmarkFn: BenchmarkFn = async (config, warmup, iters) => {
      // Create test buffers
      const aSize = m * k * 4;
      const bSize = k * n * 4;
      const outSize = m * n * 4;

      const aBuffer = device.createBuffer({
        size: aSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const bBuffer = device.createBuffer({
        size: bSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const outBuffer = device.createBuffer({
        size: outSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      try {
        // Warmup iterations
        for (let i = 0; i < warmup; i++) {
          dispatchTiledMatmulInternal({
            device,
            queue,
            a: aBuffer,
            b: bBuffer,
            out: outBuffer,
            m,
            n,
            k,
            config,
            dtype,
          });
        }
        await queue.onSubmittedWorkDone();

        // Timed iterations
        const times: number[] = [];
        for (let i = 0; i < iters; i++) {
          const start = performance.now();
          dispatchTiledMatmulInternal({
            device,
            queue,
            a: aBuffer,
            b: bBuffer,
            out: outBuffer,
            m,
            n,
            k,
            config,
            dtype,
          });
          await queue.onSubmittedWorkDone();
          times.push(performance.now() - start);
        }

        // Return median time
        times.sort((a, b) => a - b);
        return times[Math.floor(times.length / 2)];
      } finally {
        aBuffer.destroy();
        bBuffer.destroy();
        outBuffer.destroy();
      }
    };

    // Run autotune
    const result = await autotune(benchmarkFn, m, n, k, dtype, {
      maxTrials: 12, // Reasonable trial count for interactive use
      warmupIters: 2,
      timingIters: 3,
    }, includeSubgroups);

    // Cache the result in both caches (dispatch cache for lookups, autotune cache for TuneResult)
    tuningCache.set(key, result.config);
    cacheTuningResult(result);
    return result.config;
  } finally {
    autotuningInProgress.delete(key);
  }
}

/**
 * Pre-tune matmul kernels for a list of shapes.
 * This is called before plan execution to ensure all matmul ops
 * have tuned configs before they're dispatched.
 *
 * @param device GPU device
 * @param queue GPU queue
 * @param shapes Array of [m, n, k] tuples to tune
 * @param dtype Data type
 */
export async function pretuneMatmulShapes(
  device: GPUDevice,
  queue: GPUQueue,
  shapes: Array<[number, number, number]>,
  dtype: DType = "f32",
): Promise<void> {
  if (!isAutotuneEnabled()) {
    return;
  }

  // Autotune each unique shape
  const seenKeys = new Set<string>();
  for (const [m, n, k] of shapes) {
    const shapeClass = classifyShape(m, n, k, 1);
    const key = getTuningKey(shapeClass, dtype);

    // Skip if already tuned or seen
    if (tuningCache.has(key) || seenKeys.has(key)) {
      continue;
    }
    seenKeys.add(key);

    // Autotune this shape
    await autotuneIfNeeded(device, queue, m, n, k, dtype);
  }
}

/**
 * Get or create a compute pipeline for the given options.
 */
function getOrCreatePipeline(
  device: GPUDevice,
  options: CodegenOptions,
): GPUComputePipeline {
  const cacheKey = getShaderCacheKey(options);
  const cached = pipelineCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const shaderCode = generateTiledMatmulShader(options);
  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

/**
 * Pack matmul parameters into a Uint32Array for the shared params buffer pool.
 */
function packMatmulParams(
  m: number,
  n: number,
  k: number,
  lda: number,
  ldb: number,
  ldc: number,
  alpha: number,
  batchSize: number,
  batchStrideA: number,
  batchStrideB: number,
  batchStrideC: number,
): Uint32Array {
  // Params struct: m, n, k, lda, ldb, ldc, alpha, batchSize, batchStrideA, batchStrideB, batchStrideC
  // 10 u32 + 1 f32 = 48 bytes (aligned to 16)
  const data = new ArrayBuffer(48);
  const u32View = new Uint32Array(data);
  const f32View = new Float32Array(data);

  u32View[0] = m;
  u32View[1] = n;
  u32View[2] = k;
  u32View[3] = lda;
  u32View[4] = ldb;
  u32View[5] = ldc;
  f32View[6] = alpha;
  u32View[7] = batchSize;
  u32View[8] = batchStrideA;
  u32View[9] = batchStrideB;
  u32View[10] = batchStrideC;

  return u32View;
}

/**
 * Matmul dispatch options for internal use.
 */
export type DispatchMatmulOptions = {
  device: GPUDevice;
  queue: GPUQueue;
  a: GPUBuffer;
  b: GPUBuffer;
  out: GPUBuffer;
  m: number;
  n: number;
  k: number;
  batchSize?: number;
  /** Batch stride for A (0 for broadcasting, m*k for full batch) */
  batchStrideA?: number;
  /** Batch stride for B (0 for broadcasting, k*n for full batch) */
  batchStrideB?: number;
  /** Batch stride for output (always m*n unless broadcasting) */
  batchStrideC?: number;
  transA?: boolean;
  transB?: boolean;
  alpha?: number;
  dtype?: DType;
  /** dtype for input B (defaults to dtype) */
  dtypeB?: DType;
  config?: MatmulKernelConfig;
  epilogue?: EpilogueConfig;
  epilogueInputs?: GPUBuffer[];
  /** Cast input A from this wider dtype during tile load (e.g. read f32, cast to f16) */
  inputCastA?: DType;
  /** Cast input B from this wider dtype during tile load (e.g. read f32, cast to f16) */
  inputCastB?: DType;
};

/**
 * Internal matmul dispatch that requires a config.
 * Used by autotuning benchmarks to avoid recursive autotuning.
 */
function dispatchTiledMatmulInternal(options: DispatchMatmulOptions & { config: MatmulKernelConfig }): void {
  const {
    device,
    queue,
    a,
    b,
    out,
    m,
    n,
    k,
    batchSize = 1,
    transA = false,
    transB = false,
    alpha = 1.0,
    dtype = "f32",
    dtypeB,
    config,
    epilogue,
    epilogueInputs = [],
  } = options;

  // Validate config
  validateConfig(config);

  // Get transpose mode
  const transposeMode = getTransposeMode(transA, transB);

  // Compute leading dimensions
  const lda = transA ? m : k;
  const ldb = transB ? k : n;
  const ldc = n;

  // Compute batch strides
  const batchStrideA = options.batchStrideA ?? m * k;
  const batchStrideB = options.batchStrideB ?? k * n;
  const batchStrideC = options.batchStrideC ?? m * n;

  // Build codegen options
  const codegenOptions: CodegenOptions = {
    config,
    transposeMode,
    dtype,
    dtypeB,
    epilogue,
    batched: batchSize > 1,
  };

  // Get or create pipeline
  const pipeline = getOrCreatePipeline(device, codegenOptions);

  // Create params buffer via shared pool
  const paramsData = packMatmulParams(m, n, k, lda, ldb, ldc, alpha, batchSize, batchStrideA, batchStrideB, batchStrideC);
  const paramsBuffer = sharedCreateParamsBuffer(device as any, paramsData);

  // Build flat buffer array for cached bind group
  const bgBuffers = [a, b, out, paramsBuffer, ...epilogueInputs];
  const bindGroup = cachedCreateBindGroup(device as any, pipeline as any, bgBuffers as any) as any;

  // Compute dispatch dimensions
  const workgroupsX = Math.ceil(n / config.tileN);
  const workgroupsY = Math.ceil(m / config.tileM);
  const workgroupsZ = batchSize;

  // Encode and submit (batch/shared-encoder mode aware)
  const sharedEnc = getSharedEncoderInstance();
  const opLabel = getCurrentOpLabel() ?? "matmul";
  if (sharedEnc) {
    const tsWrites = getTimestampWrites(opLabel);
    const pass = sharedEnc.beginComputePass(tsWrites ? { timestampWrites: tsWrites } : undefined);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
  } else {
    const encoder = device.createCommandEncoder();
    const tsWrites = getTimestampWrites(opLabel);
    const pass = encoder.beginComputePass(tsWrites ? { timestampWrites: tsWrites } : undefined);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
    submitOrCollect(encoder.finish());
  }

  // Release params buffer back to shared pool
  releaseParamsBuffer(paramsBuffer as any);
}

/**
 * Dispatch a tiled matmul operation.
 *
 * This is the main entry point for executing matmul on WebGPU.
 */
export function dispatchTiledMatmul(options: DispatchMatmulOptions): void {
  const {
    device,
    queue,
    a,
    b,
    out,
    m,
    n,
    k,
    batchSize = 1,
    transA = false,
    transB = false,
    alpha = 1.0,
    dtype = "f32",
    dtypeB,
    epilogue,
    epilogueInputs = [],
    inputCastA,
    inputCastB,
  } = options;

  // Select kernel config
  const shapeClass = classifyShape(m, n, k, batchSize);
  const config = options.config ?? getConfigForShape(shapeClass, dtype);

  // Validate config
  validateConfig(config);

  // Get transpose mode
  const transposeMode = getTransposeMode(transA, transB);

  // Compute leading dimensions
  // For non-transposed: lda = K (A is M x K), ldb = N (B is K x N)
  // For transposed A: lda = M (A^T is K x M stored as M x K)
  // For transposed B: ldb = K (B^T is N x K stored as K x N)
  const lda = transA ? m : k;
  const ldb = transB ? k : n;
  const ldc = n;

  // Compute batch strides (0 means broadcast - same data for all batches)
  const batchStrideA = options.batchStrideA ?? m * k;
  const batchStrideB = options.batchStrideB ?? k * n;
  const batchStrideC = options.batchStrideC ?? m * n;

  // Build codegen options
  const codegenOptions: CodegenOptions = {
    config,
    transposeMode,
    dtype,
    dtypeB,
    epilogue,
    batched: batchSize > 1,
    inputCastA,
    inputCastB,
  };

  // Get or create pipeline
  const pipeline = getOrCreatePipeline(device, codegenOptions);

  // Create params buffer via shared pool
  const paramsData = packMatmulParams(m, n, k, lda, ldb, ldc, alpha, batchSize, batchStrideA, batchStrideB, batchStrideC);
  const paramsBuffer = sharedCreateParamsBuffer(device as any, paramsData);

  // Build flat buffer array for cached bind group
  const bgBuffers = [a, b, out, paramsBuffer, ...epilogueInputs];
  const bindGroup = cachedCreateBindGroup(device as any, pipeline as any, bgBuffers as any) as any;

  // Compute dispatch dimensions
  // const _wgSize = getWorkgroupSize(config);
  const workgroupsX = Math.ceil(n / config.tileN);
  const workgroupsY = Math.ceil(m / config.tileM);
  const workgroupsZ = batchSize;

  // Encode and submit (batch/shared-encoder mode aware)
  const sharedEnc = getSharedEncoderInstance();
  const opLabel = getCurrentOpLabel() ?? "matmul";
  if (sharedEnc) {
    const tsWrites = getTimestampWrites(opLabel);
    const pass = sharedEnc.beginComputePass(tsWrites ? { timestampWrites: tsWrites } : undefined);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
  } else {
    const encoder = device.createCommandEncoder();
    const tsWrites = getTimestampWrites(opLabel);
    const pass = encoder.beginComputePass(tsWrites ? { timestampWrites: tsWrites } : undefined);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
    submitOrCollect(encoder.finish());
  }

  // Release params buffer back to shared pool
  releaseParamsBuffer(paramsBuffer as any);
}

/**
 * Compute output shape for matmul.
 */
export function computeMatmulOutputShape(
  aShape: number[],
  bShape: number[],
  transA: boolean,
  transB: boolean,
): number[] {
  if (aShape.length < 2 || bShape.length < 2) {
    throw new Error("matmul requires at least 2D tensors");
  }

  // Get matrix dimensions (last 2 dims)
  const aRank = aShape.length;
  const bRank = bShape.length;

  let m: number, ka: number, kb: number, n: number;

  if (transA) {
    ka = aShape[aRank - 2];
    m = aShape[aRank - 1];
  } else {
    m = aShape[aRank - 2];
    ka = aShape[aRank - 1];
  }

  if (transB) {
    n = bShape[bRank - 2];
    kb = bShape[bRank - 1];
  } else {
    kb = bShape[bRank - 2];
    n = bShape[bRank - 1];
  }

  if (ka !== kb) {
    throw new Error(`matmul shape mismatch: k dimensions ${ka} vs ${kb}`);
  }

  // Handle batch dimensions (broadcast)
  const aBatch = aShape.slice(0, -2);
  const bBatch = bShape.slice(0, -2);

  // Broadcast batch dimensions
  const maxBatchRank = Math.max(aBatch.length, bBatch.length);
  const outBatch: number[] = [];

  for (let i = 0; i < maxBatchRank; i++) {
    const aDim = aBatch[aBatch.length - 1 - i] ?? 1;
    const bDim = bBatch[bBatch.length - 1 - i] ?? 1;

    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error(`batch dimensions not broadcastable: ${aDim} vs ${bDim}`);
    }

    outBatch.unshift(Math.max(aDim, bDim));
  }

  return [...outBatch, m, n];
}

/**
 * Compute total batch size from batch dimensions.
 */
export function computeBatchSize(batchDims: number[]): number {
  return batchDims.reduce((acc, dim) => acc * dim, 1);
}

/**
 * Compute batch strides for A and B with proper broadcasting.
 * Returns stride of 0 for broadcast dimensions (where input batch dim is 1).
 */
export function computeBatchStrides(
  aShape: number[],
  bShape: number[],
  outBatchDims: number[],
  m: number,
  n: number,
  k: number,
): { strideA: number; strideB: number; strideC: number } {
  const aBatchDims = aShape.slice(0, -2);
  const bBatchDims = bShape.slice(0, -2);
  const outBatchSize = computeBatchSize(outBatchDims);

  // If no batch dimensions, all strides are 0 (single matrix)
  if (outBatchSize <= 1) {
    return { strideA: 0, strideB: 0, strideC: 0 };
  }

  // Compute stride for A
  // If A has no batch dims or all batch dims are 1, stride is 0 (broadcast)
  const aBatchSize = computeBatchSize(aBatchDims);
  const strideA = aBatchSize === 1 ? 0 : m * k;

  // Compute stride for B
  const bBatchSize = computeBatchSize(bBatchDims);
  const strideB = bBatchSize === 1 ? 0 : k * n;

  // Output always has full batch stride
  const strideC = m * n;

  return { strideA, strideB, strideC };
}
