/**
 * Matmul dispatch: kernel selection, pipeline caching, and execution.
 */

import { ENV } from "../../../core/env";
import {
  cachedCreateBindGroup,
  releaseParamsBuffer,
  createParamsBuffer as sharedCreateParamsBuffer,
} from "../bind-group-cache";
import { dispatchComputePass } from "../dispatch";
import type {
  GPUBuffer,
  GPUComputePipeline,
  GPUDevice,
  GPUQueue,
} from "../gpu-types";
import { GPUBufferUsage } from "../gpu-types";
import { getWarmupPipeline, recordPipeline } from "../pipeline-warmup";
import { F32_BYTES } from "../shape-utils";
import { getCurrentOpLabel } from "../shared-encoder";
import { compileTileKernel } from "../tile-compiler";
import { splitWorkgroups2d, type TileKernelSpec } from "../tile-ir";
import { onTeardown } from "../webgpu-state";
import { cacheTuningResult } from "./autotune";
import {
  computeGemvRoute,
  type GemvKernelOptions,
  gemvSupportsEpilogue,
  generateGemvShaderTileIR,
  getGemvShaderCacheKey,
} from "./gemv";
import {
  generateKSplitReductionShaderTileIR,
  generateTiledMatmulShaderTileIR,
} from "./tile-matmul";
import {
  type CodegenOptions,
  classifyShape,
  DEFAULT_CONFIG,
  type DType,
  type EpilogueConfig,
  getShaderCacheKey,
  getSubgroupSupport,
  getTransposeMode,
  type MatmulKernelConfig,
  validateConfig,
} from "./types";
import {
  getMatmulVariant,
  MATMUL_VARIANTS,
  type MatmulVariantChoice,
  type MatmulVariantContext,
} from "./variants";

/**
 * Pipeline cache for compiled matmul kernels.
 */
const pipelineCache = new Map<string, GPUComputePipeline>();

/**
 * Tuning results cache (in-memory only for now).
 */
const tuningCache = new Map<string, MatmulKernelConfig>();

/**
 * Per-shape tuning cache: exact (M,N,K,dtype) → winning (variant, choice).
 * Used for bare matmuls only (epilogue matmuls use shape-class defaults).
 * Populated by the autotuner (which benchmarks across ALL applicable
 * variants × their candidates) or seeded via seedPerShapeMatmulChoice.
 */
const perShapeTuningCache = new Map<string, MatmulVariantChoice>();

function getPerShapeKey(m: number, n: number, k: number, dtype: DType): string {
  return `${m}_${n}_${k}_${dtype}`;
}

/** Count of autotune searches actually executed (cache misses). Debug/tests. */
let autotuneRunCount = 0;
export function getAutotuneRunCount(): number {
  return autotuneRunCount;
}

/** Read the tuned (variant, choice) for an exact shape, if any. Debug/tests. */
export function getPerShapeMatmulChoice(
  m: number,
  n: number,
  k: number,
  dtype: DType,
): MatmulVariantChoice | undefined {
  return perShapeTuningCache.get(getPerShapeKey(m, n, k, dtype));
}

/** Seed a per-shape winner without benchmarking (tests / pre-seeded caches). */
export function seedPerShapeMatmulChoice(
  m: number,
  n: number,
  k: number,
  dtype: DType,
  choice: MatmulVariantChoice,
): void {
  perShapeTuningCache.set(getPerShapeKey(m, n, k, dtype), choice);
}

/** Clear per-shape tuned winners (tests). */
export function clearPerShapeMatmulChoices(): void {
  perShapeTuningCache.clear();
}

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
 * Tiled choice for a ctx: shape-class tuning cache → heuristic defaults.
 */
function tiledChoiceForContext(
  ctx: MatmulVariantContext,
): Extract<MatmulVariantChoice, { variant: "tiled" }> {
  const shapeClass = classifyShape(ctx.m, ctx.n, ctx.k, ctx.batchSize);
  const key = `${shapeClass}_${ctx.dtypeA}_${ctx.hasEpilogue ? "epilogue" : "bare"}`;
  const cached = tuningCache.get(key);
  if (cached) {
    return { variant: "tiled", config: cached };
  }
  return getMatmulVariant("tiled").defaultChoice(ctx) as Extract<
    MatmulVariantChoice,
    { variant: "tiled" }
  >;
}

/**
 * Select the (variant, choice) for a matmul described by ctx.
 *
 * Order: explicit caller config (pins the tiled variant) → per-shape tuned
 * winner (bare matmuls only; re-gated on isApplicable so a winner tuned in
 * one context never forces an inapplicable variant) → first applicable
 * registry variant's heuristic default (for tiled, via the shape-class
 * tuning cache first). With autotune off the caches are empty and this
 * reduces exactly to the pre-registry heuristics.
 */
function selectMatmulChoice(
  ctx: MatmulVariantContext,
  explicitConfig?: MatmulKernelConfig,
): MatmulVariantChoice {
  if (explicitConfig) {
    return { variant: "tiled", config: explicitConfig };
  }
  // Per-shape tuned winner (bare matmuls only — same guard as pre-registry)
  if (!ctx.hasEpilogue) {
    const hit = perShapeTuningCache.get(
      getPerShapeKey(ctx.m, ctx.n, ctx.k, ctx.dtypeA),
    );
    if (hit && getMatmulVariant(hit.variant).isApplicable(ctx)) {
      return hit;
    }
  }
  const variant = MATMUL_VARIANTS.find((v) => v.isApplicable(ctx));
  if (!variant || variant.name === "tiled") {
    return tiledChoiceForContext(ctx);
  }
  return variant.defaultChoice(ctx);
}

/**
 * Reset all module-local mutable state (pipeline cache, tuning caches, autotune state).
 */
export function resetMatmulState(): void {
  pipelineCache.clear();
  tuningCache.clear();
  perShapeTuningCache.clear();
  autotuneCounter = 0;
  autotuneRunCount = 0;
  autotuningInProgress.clear();
  gemvDispatchCount = 0;
  gemvEpilogueDispatchCount = 0;
}
onTeardown(resetMatmulState);

/**
 * Clear the dispatch tuning cache.
 * Called by tests to reset state between runs.
 */
export function clearDispatchTuningCache(): void {
  tuningCache.clear();
}

/**
 * Build a bench plan for a candidate (variant, choice). Reuses the SAME plan
 * seams real execution uses (planTiledMatmul with an explicit config pins the
 * tiled variant, incl. its K-split/swapGrid decisions; planGemvRowMatmul is
 * the GEMV route) so the autotuner measures exactly what would dispatch.
 * Returns null when the choice degenerates for this geometry.
 */
function buildBenchPlan(
  device: GPUDevice,
  queue: GPUQueue,
  ctx: MatmulVariantContext,
  choice: MatmulVariantChoice,
): MatmulStandardPlan | MatmulKSplitPlan | null {
  if (choice.variant === "gemv") {
    return planGemvRowMatmul(
      device,
      ctx.n,
      ctx.k,
      ctx.transB,
      1.0,
      ctx.dtypeA,
      ctx.dtypeB,
      choice.wgSize,
      choice.rowsPerWg,
    );
  }
  return planTiledMatmul({
    device,
    queue,
    a: undefined as unknown as GPUBuffer, // geometry-only at plan time
    b: undefined as unknown as GPUBuffer,
    out: undefined as unknown as GPUBuffer,
    m: ctx.m,
    n: ctx.n,
    k: ctx.k,
    transA: ctx.transA,
    transB: ctx.transB,
    dtype: ctx.dtypeA,
    dtypeB: ctx.dtypeB !== ctx.dtypeA ? ctx.dtypeB : undefined,
    config: choice.config,
  });
}

/** Encode + submit one bench execution of a plan (standard or K-split).
 *  Raw bind groups / own encoder — self-contained by design, so benching
 *  never pollutes the shared params sequence or bind-group caches. */
function benchmarkPlanOnce(
  device: GPUDevice,
  queue: GPUQueue,
  plan: MatmulStandardPlan | MatmulKSplitPlan,
  a: GPUBuffer,
  b: GPUBuffer,
  out: GPUBuffer,
  paramsBuffer: GPUBuffer,
  kSplitTempBuffer?: GPUBuffer,
  reduceParamsBuffer?: GPUBuffer,
): void {
  const encoder = device.createCommandEncoder();
  if (plan.kSplit) {
    const matmulBG = device.createBindGroup({
      layout: plan.ksplitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: kSplitTempBuffer as GPUBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });
    const reduceBG = device.createBindGroup({
      layout: plan.reducePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: kSplitTempBuffer as GPUBuffer } },
        { binding: 1, resource: { buffer: out } },
        { binding: 2, resource: { buffer: reduceParamsBuffer as GPUBuffer } },
      ],
    });
    const pass1 = encoder.beginComputePass();
    pass1.setPipeline(plan.ksplitPipeline);
    pass1.setBindGroup(0, matmulBG);
    pass1.dispatchWorkgroups(...plan.ksplitDispatch);
    pass1.end();
    const pass2 = encoder.beginComputePass();
    pass2.setPipeline(plan.reducePipeline);
    pass2.setBindGroup(0, reduceBG);
    pass2.dispatchWorkgroups(...plan.reduceDispatch);
    pass2.end();
  } else {
    const bindGroup = device.createBindGroup({
      layout: plan.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: out } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(plan.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(plan.dispatchX, plan.dispatchY, plan.dispatchZ);
    pass.end();
  }
  queue.submit([encoder.finish()]);
}

/** Benchmark a plan: 2 warmups + 3 timed executions, median ms. Creates and
 *  destroys its own params/temp buffers (per-candidate — params differ). */
async function benchmarkPlanMedian(
  device: GPUDevice,
  queue: GPUQueue,
  plan: MatmulStandardPlan | MatmulKSplitPlan,
  a: GPUBuffer,
  b: GPUBuffer,
  out: GPUBuffer,
): Promise<number> {
  const makeUniform = (data: Uint32Array): GPUBuffer => {
    const buf = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(buf, 0, data);
    return buf;
  };

  let paramsBuffer: GPUBuffer;
  let kSplitTempBuffer: GPUBuffer | undefined;
  let reduceParamsBuffer: GPUBuffer | undefined;
  if (plan.kSplit) {
    paramsBuffer = makeUniform(plan.ksplitParamsData);
    reduceParamsBuffer = makeUniform(plan.reduceParamsData);
    kSplitTempBuffer = device.createBuffer({
      size: plan.tempBytes,
      usage: GPUBufferUsage.STORAGE,
    });
  } else {
    paramsBuffer = makeUniform(plan.paramsData);
  }

  try {
    // Warmup
    for (let i = 0; i < 2; i++) {
      benchmarkPlanOnce(
        device,
        queue,
        plan,
        a,
        b,
        out,
        paramsBuffer,
        kSplitTempBuffer,
        reduceParamsBuffer,
      );
    }
    await queue.onSubmittedWorkDone?.();

    // Timed iterations
    const times: number[] = [];
    for (let i = 0; i < 3; i++) {
      const start = performance.now();
      benchmarkPlanOnce(
        device,
        queue,
        plan,
        a,
        b,
        out,
        paramsBuffer,
        kSplitTempBuffer,
        reduceParamsBuffer,
      );
      await queue.onSubmittedWorkDone?.();
      times.push(performance.now() - start);
    }
    times.sort((x, y) => x - y);
    return times[Math.floor(times.length / 2)];
  } finally {
    paramsBuffer.destroy();
    kSplitTempBuffer?.destroy();
    reduceParamsBuffer?.destroy();
  }
}

/**
 * Run autotuning for a specific shape if no cached result exists: benchmark
 * ALL applicable registry variants × their candidates and cache the winning
 * (variant, choice) per exact shape. Called via pretuneMatmulShapes when
 * autotune mode is enabled.
 */
async function autotuneIfNeeded(
  device: GPUDevice,
  queue: GPUQueue,
  m: number,
  n: number,
  k: number,
  dtype: DType,
): Promise<MatmulVariantChoice> {
  const perShapeKey = getPerShapeKey(m, n, k, dtype);

  // Already have a per-shape tuned winner
  const perShapeCached = perShapeTuningCache.get(perShapeKey);
  if (perShapeCached) {
    return perShapeCached;
  }

  // Prevent recursive autotuning
  if (autotuningInProgress.has(perShapeKey)) {
    return { variant: "tiled", config: DEFAULT_CONFIG };
  }

  // Bench context: bare, unbatched, NN, f32-vs-f32 (same as pre-registry).
  const ctx: MatmulVariantContext = {
    m,
    n,
    k,
    batchSize: 1,
    dtypeA: dtype,
    dtypeB: dtype,
    transA: false,
    transB: false,
    hasEpilogue: false,
    epiloguePresent: false,
    hasInputCast: false,
    hasExplicitConfig: false,
    subgroupSupported: getSubgroupSupport()?.supported ?? false,
  };

  autotuningInProgress.add(perShapeKey);

  try {
    const shapeClass = classifyShape(m, n, k, 1);

    // Test buffers shared across all variants and candidates
    const aBuffer = device.createBuffer({
      size: m * k * F32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bBuffer = device.createBuffer({
      size: k * n * F32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outBuffer = device.createBuffer({
      size: m * n * F32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const flops = 2 * m * n * k;
    // Fallback winner if every candidate fails: the tiled heuristic default.
    const tiledDefault = getMatmulVariant("tiled").defaultChoice(
      ctx,
    ) as Extract<MatmulVariantChoice, { variant: "tiled" }>;
    let bestChoice: MatmulVariantChoice = tiledDefault;
    let bestGflops = 0;
    // Best TILED result feeds the shape-class cache (existing consumers).
    let bestTiledConfig = tiledDefault.config;
    let bestTiledGflops = 0;

    try {
      for (const variant of MATMUL_VARIANTS) {
        if (!variant.isApplicable(ctx)) continue;
        for (const choice of variant.candidates(ctx)) {
          try {
            const plan = buildBenchPlan(device, queue, ctx, choice);
            if (!plan) continue;
            const medianMs = await benchmarkPlanMedian(
              device,
              queue,
              plan,
              aBuffer,
              bBuffer,
              outBuffer,
            );
            const gflops = flops / (medianMs * 1e6);
            if (ENV.TORCHLETTE_DEBUG_AUTOTUNE === "1") {
              const desc =
                choice.variant === "gemv"
                  ? `gemv wg${choice.wgSize}`
                  : `tiled ${choice.config.tileM}x${choice.config.tileN}x${choice.config.tileK} t${choice.config.threadTileM}x${choice.config.threadTileN}${choice.config.useSubgroups ? " sg" : ""}`;
              console.log(
                `[autotune] ${m}x${n}x${k} ${dtype} ${desc}: ${medianMs.toFixed(3)}ms (${gflops.toFixed(1)} GFLOP/s)`,
              );
            }
            if (gflops > bestGflops) {
              bestGflops = gflops;
              bestChoice = choice;
            }
            if (choice.variant === "tiled" && gflops > bestTiledGflops) {
              bestTiledGflops = gflops;
              bestTiledConfig = choice.config;
            }
          } catch {
            // Skip failed candidates
          }
        }
      }
    } finally {
      aBuffer.destroy();
      bBuffer.destroy();
      outBuffer.destroy();
    }

    // Cache the overall winner per exact shape (variant + choice)…
    perShapeTuningCache.set(perShapeKey, bestChoice);
    autotuneRunCount++;
    if (ENV.TORCHLETTE_DEBUG_AUTOTUNE === "1") {
      console.log(
        `[autotune] ${m}x${n}x${k} ${dtype} WINNER: ${JSON.stringify(bestChoice)} (${bestGflops.toFixed(1)} GFLOP/s)`,
      );
    }
    // …and the best tiled config per shape class (pre-registry consumers).
    cacheTuningResult({
      config: bestTiledConfig,
      gflopsPerSec: bestTiledGflops,
      medianMs:
        bestTiledGflops > 0 ? flops / (bestTiledGflops * 1e6) : Infinity,
      shapeClass,
      dtype,
    });

    return bestChoice;
  } finally {
    autotuningInProgress.delete(perShapeKey);
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
  // Autotuning requires either env var TORCHLETTE_AUTOTUNE=1 or programmatic
  // setAutotuneEnabled(true) (used by compile({ autotune: true }))
  const envEnabled = ENV.TORCHLETTE_AUTOTUNE === "1";
  if (!envEnabled && !isAutotuneEnabled()) {
    return;
  }

  // Autotune each unique shape (per-shape key, not per-shape-class)
  const seenKeys = new Set<string>();
  for (const [m, n, k] of shapes) {
    const key = getPerShapeKey(m, n, k, dtype);

    // Skip if already tuned or seen
    if (perShapeTuningCache.has(key) || seenKeys.has(key)) {
      continue;
    }
    seenKeys.add(key);

    // Autotune this shape
    await autotuneIfNeeded(device, queue, m, n, k, dtype);
  }
}

/** Shared pipeline-cache-or-compile logic for matmul and K-split reduction. */
function cachedPipeline(
  device: GPUDevice,
  cache: Map<string, GPUComputePipeline>,
  cacheKey: string,
  generateShader: () => string,
): GPUComputePipeline {
  const cached = cache.get(cacheKey);
  if (cached) return cached;

  const warmed = getWarmupPipeline(cacheKey);
  if (warmed) {
    cache.set(cacheKey, warmed);
    return warmed;
  }

  const shaderCode = generateShader();
  recordPipeline(cacheKey, shaderCode);
  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  cache.set(cacheKey, pipeline);
  return pipeline;
}

/**
 * Get or create a compute pipeline for the given options.
 */
function getOrCreatePipeline(
  device: GPUDevice,
  options: CodegenOptions,
): GPUComputePipeline {
  return cachedPipeline(device, pipelineCache, getShaderCacheKey(options), () =>
    generateTiledMatmulShaderTileIR(options),
  );
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
type DispatchMatmulOptions = {
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

// --- K-split infrastructure ---

/** Cached temp buffers for K-split partial results, keyed by byte size. */
const kSplitTempBufferCache = new Map<number, GPUBuffer>();

/** Get or create a persistent temp buffer for K-split partials. */
function getKSplitTempBuffer(device: GPUDevice, byteSize: number): GPUBuffer {
  let buf = kSplitTempBufferCache.get(byteSize);
  if (!buf) {
    buf = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    kSplitTempBufferCache.set(byteSize, buf);
  }
  return buf;
}

/** Stage-4 stream generation: read-only lookup of the cached K-split temp
 *  buffer (the generator binds it as a persistent slot — same buffer object
 *  the recording's dispatch referenced). Null if not yet allocated; the
 *  generator runs post-recording so a hit is guaranteed for any dispatched
 *  K-split shape. Never allocates. */
export function lookupKSplitTempBuffer(byteSize: number): GPUBuffer | null {
  return kSplitTempBufferCache.get(byteSize) ?? null;
}

/** Pipeline cache for K-split reduction shaders. */
const reductionPipelineCache = new Map<string, GPUComputePipeline>();

/** Get or create a reduction pipeline. */
function getOrCreateReductionPipeline(
  device: GPUDevice,
  kSplitCount: number,
  outputDtype: DType,
): GPUComputePipeline {
  return cachedPipeline(
    device,
    reductionPipelineCache,
    `ksplit_reduce_${kSplitCount}_${outputDtype}`,
    () => generateKSplitReductionShaderTileIR(kSplitCount, outputDtype),
  );
}

/** Minimum workgroups before considering K-split. */
// K-split engages while the base grid is below the TARGET occupancy (128
// workgroups), not below an arbitrary lower cliff: a [1,2048]×[2048,2048]
// decode matmul launches only 64 single-row workgroups (<1 wave on an 80-SM
// GPU) and was denied splitting by the old `>= 64` cutoff — conflating
// workgroup count with occupancy. Shapes already at/above target never split
// (training-scale grids are far above it), so their configs are unchanged.
const KSPLIT_MIN_WORKGROUPS_THRESHOLD = 128;
/** Minimum K dimension for K-split to be worthwhile. */
const KSPLIT_MIN_K = 512;
/** Target total workgroups after K-split. */
const KSPLIT_TARGET_WORKGROUPS = 128;
/** Maximum K-split factor. */
const KSPLIT_MAX_FACTOR = 32;

/**
 * Determine K-split factor, or 0 if K-split is not beneficial.
 */
function computeKSplitFactor(
  baseWorkgroups: number,
  k: number,
  tileK: number,
  batchSize: number,
  hasEpilogue: boolean,
): number {
  // K-split only for unbatched matmuls without epilogue
  if (batchSize > 1 || hasEpilogue) return 0;
  // Only when underutilizing GPU
  if (baseWorkgroups >= KSPLIT_MIN_WORKGROUPS_THRESHOLD) return 0;
  // Only when K is large enough to split
  if (k < KSPLIT_MIN_K) return 0;

  const desired = Math.ceil(KSPLIT_TARGET_WORKGROUPS / baseWorkgroups);
  const maxByK = Math.floor(k / tileK); // each split needs at least tileK elements
  return Math.min(desired, maxByK, KSPLIT_MAX_FACTOR);
}

/**
 * Dispatch a tiled matmul operation.
 *
 * This is the main entry point for executing matmul on WebGPU.
 * Automatically uses K-split when the output tile grid is too small
 * to saturate the GPU (e.g., small-M backward matmuls).
 */
/**
 * Stage-4 plan/encode split for the STANDARD (non-K-split) matmul path:
 * pipeline + params bytes + dispatch dims + epilogue-input count, computed
 * from geometry alone (no GPU alloc/encode). Returns {kSplit:true} when the
 * K-split path applies — the stream generator treats that as uncovered
 * (op-internal temp + two dispatches, a later increment). dispatchTiledMatmul
 * consumes the SAME plan, so pipeline/bindings/workgroups can't drift.
 * Binding order: [a, b, out, params, ...epilogueInputs].
 */
export interface MatmulStandardPlan {
  kSplit: false;
  pipeline: GPUComputePipeline;
  paramsData: Uint32Array;
  dispatchX: number;
  dispatchY: number;
  dispatchZ: number;
  numEpilogueInputs: number;
  /** Profiler label suffix (e.g. "_gemv" for the M=1 GEMV kernel). */
  label?: string;
}

/**
 * K-split plan: two dispatches over a cached f32 partials temp buffer (keyed
 * by byte size, persistent — NOT a recordAlloc'd buffer, so it's a
 * persistent slot in the stream). (1) K-split matmul a,b → temp[P*M*N];
 * (2) reduction temp → out with alpha. The generator looks the temp up by
 * tempBytes (lookupKSplitTempBuffer) and binds it as a persistent slot.
 */
export interface MatmulKSplitPlan {
  kSplit: true;
  tempBytes: number;
  ksplitPipeline: GPUComputePipeline;
  ksplitParamsData: Uint32Array;
  ksplitDispatch: [number, number, number];
  reducePipeline: GPUComputePipeline;
  reduceParamsData: Uint32Array;
  reduceDispatch: [number, number, number];
  /** Profiler label suffix (e.g. "_gemv" for the M=1 GEMV kernel). */
  label?: string;
}

// --- GEMV (M=1) routing ---

/** Count of dispatches routed to the GEMV kernels (probe/debug hook). */
let gemvDispatchCount = 0;
export function getGemvDispatchCount(): number {
  return gemvDispatchCount;
}

/** Count of GEMV dispatches with a fused epilogue seam (probe/debug hook). */
let gemvEpilogueDispatchCount = 0;
export function getGemvEpilogueDispatchCount(): number {
  return gemvEpilogueDispatchCount;
}

/** Pack the GEMV uniform config {n, k, alpha, split_k} (16 bytes). */
function packGemvParams(
  n: number,
  k: number,
  alpha: number,
  splitK: number,
): Uint32Array {
  const data = new ArrayBuffer(16);
  const u32View = new Uint32Array(data);
  u32View[0] = n;
  u32View[1] = k;
  new Float32Array(data)[2] = alpha;
  u32View[3] = splitK;
  return u32View;
}

let quantGemvDispatchCount = 0;
/** Count of quantized (int8) GEMV dispatches (probe/gate hook). */
export function getQuantGemvDispatchCount(): number {
  return quantGemvDispatchCount;
}

/**
 * Weight-only int8-grouped GEMV: y[1,N] = a[1,K] @ dequant(packedW[N,K], scales).
 * The B operand is a packed int8 weight ([N,K/4] u32) + per-group f16 scales
 * ([N,K/G] u32-packed). Dequant is fused into the NT row-dot load path — no
 * dequantized weight tensor is ever materialized (the residency win). This is
 * a self-contained eager dispatch (like matmulChunked): inference-only, noGrad,
 * so it stays out of the autograd/compiled-plan replay machinery.
 * See docs/quantization-design.md. Binds [a, b, out, params, b_scales].
 */
export function dispatchQuantizedGemvNT(
  device: GPUDevice,
  aBuffer: GPUBuffer,
  packedWeight: GPUBuffer,
  scales: GPUBuffer,
  outBuffer: GPUBuffer,
  n: number,
  k: number,
  groupSize: number,
  outputDtype: DType = "f32",
  alpha = 1.0,
): void {
  // NT route geometry (single source: computeGemvRoute). Quant path is scalar
  // (no vec4) and never K-splits (a scale is per K-group, not per partial).
  const route = computeGemvRoute(n, k, /* transB */ true);
  if (!route) {
    throw new Error(`dispatchQuantizedGemvNT: no NT route for N=${n} K=${k}`);
  }
  const kernelOpts: GemvKernelOptions = {
    mode: "nt",
    dtypeA: "f32",
    dtypeB: "f16", // logical weight dtype (informational; packed as u32)
    outputDtype,
    kSplit: false,
    rowsPerWg: route.rowsPerWg, // wgSize defaults to GEMV_DEFAULT_WG_SIZE
    vec4: false,
    quantB: { scheme: "int8-grouped", groupSize },
  };
  const pipeline = cachedPipeline(
    device,
    pipelineCache,
    getGemvShaderCacheKey(kernelOpts),
    () => generateGemvShaderTileIR(kernelOpts),
  );
  const paramsBuffer = sharedCreateParamsBuffer(
    device,
    packGemvParams(n, k, alpha, 1),
  );
  const bindGroup = cachedCreateBindGroup(device, pipeline, [
    aBuffer,
    packedWeight,
    outBuffer,
    paramsBuffer,
    scales,
  ]);
  quantGemvDispatchCount++;
  dispatchComputePass(
    pipeline,
    bindGroup,
    route.dispatch[0],
    route.dispatch[1],
    route.dispatch[2],
    (getCurrentOpLabel() ?? "matmul") + "_gemv_q8",
  );
  releaseParamsBuffer(paramsBuffer);
}

/**
 * Explicit dequant of an int8-grouped packed weight to f32 [N, K] — the M>1 /
 * prefill fallback (docs/quantization-design.md phase 2). The tiled quant path
 * was not built (phase-1 outcome), so a consumer that can't take the packed
 * operand (M>1) gets this EXPLICIT dequant, then runs the stock matmul on the
 * result — declared, never a silent dequantize-materialize. One thread per
 * output element; mirrors the GEMV kernel's unpack (unpackInt8Snorm · f16
 * group scale). Single source for the dequant mapping: quantize.ts.
 * Binds [bq, out, params, b_scales].
 */
let dequantI8DispatchCount = 0;
/** Count of explicit int8 dequant dispatches (probe/gate hook). */
export function getDequantI8DispatchCount(): number {
  return dequantI8DispatchCount;
}

export function dispatchDequantizeInt8Grouped(
  device: GPUDevice,
  packedWeight: GPUBuffer,
  scales: GPUBuffer,
  outBuffer: GPUBuffer,
  n: number,
  k: number,
  groupSize: number,
): void {
  dequantI8DispatchCount++;
  const gShift = Math.log2(groupSize);
  if (!Number.isInteger(gShift)) {
    throw new Error(`dequant: groupSize ${groupSize} must be a power of two`);
  }
  const spec = createDequantInt8Kernel(groupSize);
  const cacheKey = `dequant_i8g${groupSize}`;
  const pipeline = cachedPipeline(device, pipelineCache, cacheKey, () =>
    compileTileKernel(spec),
  );
  const paramsBuffer = sharedCreateParamsBuffer(
    device,
    new Uint32Array([n, k, 0, 0]),
  );
  const bindGroup = cachedCreateBindGroup(device, pipeline, [
    packedWeight,
    outBuffer,
    paramsBuffer,
    scales,
  ]);
  const total = n * k;
  const totalWg = Math.ceil(total / 256);
  const gx = Math.min(totalWg, 65535);
  const gy = Math.ceil(totalWg / 65535);
  dispatchComputePass(
    pipeline,
    bindGroup,
    gx,
    gy,
    1,
    (getCurrentOpLabel() ?? "matmul") + "_dequant_i8",
  );
  releaseParamsBuffer(paramsBuffer);
}

/** Per-element dequant tile kernel: out[e] = unpackInt8Snorm(bq)·scale. */
function createDequantInt8Kernel(groupSize: number): TileKernelSpec {
  const gShift = Math.log2(groupSize);
  return {
    name: "dequantI8Grouped",
    workgroupSize: 256,
    enableF16: true,
    uniformBindingIndex: 2,
    bindings: {
      b: { storage: "read", type: "u32" },
      out: { storage: "read_write", type: "f32" },
      b_scales: { storage: "read", type: "u32" },
    },
    uniforms: { n: "u32", k: "u32", pad0: "u32", pad1: "u32" },
    grid: (u) => splitWorkgroups2d(Math.ceil((u.n * u.k) / 256)),
    kernel(ctx) {
      const n = ctx.emitLet("n", ctx.uniform("n"));
      const k = ctx.emitLet("k", ctx.uniform("k"));
      const total = ctx.emitLet("total", n.mul(k));
      // 2D-flattened global element index.
      const e = ctx.emitLet(
        "e",
        ctx.rowIndex2d().mul(ctx.u32(256)).add(ctx.localIndex()),
      );
      ctx.ifThen(e.lt(total), () => {
        const row = ctx.emitLet("row", e.div(k));
        const col = ctx.emitLet("col", e.mod(k));
        const wordsPerRow = ctx.emitLet("words_pr", k.shr(ctx.u32(2)));
        const groupsPerRow = ctx.emitLet("groups_pr", k.shr(ctx.u32(gShift)));
        const word = ctx.load(
          "b",
          row.mul(wordsPerRow).add(col.shr(ctx.u32(2))),
        );
        const qn = word.unpackInt8Snorm(col.and(ctx.u32(3)));
        const sIdx = ctx.emitLet(
          "s_idx",
          row.mul(groupsPerRow).add(col.shr(ctx.u32(gShift))),
        );
        const sWord = ctx.load("b_scales", sIdx.shr(ctx.u32(1)));
        const scale = sWord.unpackHalf(sIdx.and(ctx.u32(1)));
        ctx.emitStore("out", e, qn.mul(scale));
      });
    },
  };
}

/**
 * Route an M=1, batch=1, no-epilogue matmul to the dedicated GEMV tile-IR
 * kernels. Returns a plan shaped exactly like the tiled paths (standard
 * single dispatch, or K-split partials + the shared reduction pass) so the
 * dispatch tail, compiled-plan recording, and the stage-4 stream generator
 * all consume it unchanged. Returns null to fall through to the tiled path.
 * Opt-out: TORCHLETTE_GEMV=0.
 */
function planGemvRowMatmul(
  device: GPUDevice,
  n: number,
  k: number,
  transB: boolean,
  alpha: number,
  dtype: DType,
  dtypeB: DType,
  wgSize?: number,
  rowsPerWg?: number,
  epilogue?: EpilogueConfig,
  epilogueInputCount = 0,
): MatmulStandardPlan | MatmulKSplitPlan | null {
  const route = computeGemvRoute(n, k, transB, wgSize, rowsPerWg);
  if (!route) return null;
  const hasEpilogueOps = !!epilogue && epilogue.ops.length > 0;
  if (hasEpilogueOps) {
    // Epilogue applies to the final output at the kernel's seam — never to
    // K-split partials — and must be reconstructible as bias/unary.
    if (route.splitK >= 2) return null;
    if (!gemvSupportsEpilogue(epilogue)) return null;
    if (epilogueInputCount !== epilogue.additionalInputCount) return null;
  }
  const outputDtype: DType = hasEpilogueOps
    ? epilogue.outputDtype
    : dtype === "f32" || dtypeB === "f32"
      ? "f32"
      : "f16";
  const kernelOpts: GemvKernelOptions = {
    mode: route.mode,
    dtypeA: dtype,
    dtypeB,
    outputDtype,
    kSplit: route.splitK >= 2,
    wgSize,
    rowsPerWg: route.rowsPerWg,
    vec4: route.vec4,
    epilogue: hasEpilogueOps ? epilogue : undefined,
  };
  const pipeline = cachedPipeline(
    device,
    pipelineCache,
    getGemvShaderCacheKey(kernelOpts),
    () => generateGemvShaderTileIR(kernelOpts),
  );
  if (route.splitK >= 2) {
    const reduceParamsBuf = new ArrayBuffer(8);
    const reduceU32 = new Uint32Array(reduceParamsBuf);
    new Float32Array(reduceParamsBuf)[1] = alpha;
    reduceU32[0] = n;
    return {
      kSplit: true,
      tempBytes: route.splitK * n * F32_BYTES,
      ksplitPipeline: pipeline,
      ksplitParamsData: packGemvParams(n, k, 1.0, route.splitK),
      ksplitDispatch: route.dispatch,
      reducePipeline: getOrCreateReductionPipeline(
        device,
        route.splitK,
        outputDtype,
      ),
      reduceParamsData: reduceU32,
      reduceDispatch: [Math.ceil(n / 256), 1, 1],
      label: "_gemv",
    };
  }
  return {
    kSplit: false,
    pipeline,
    paramsData: packGemvParams(n, k, alpha, 1),
    dispatchX: route.dispatch[0],
    dispatchY: route.dispatch[1],
    dispatchZ: route.dispatch[2],
    numEpilogueInputs: hasEpilogueOps ? epilogueInputCount : 0,
    label: hasEpilogueOps ? "_gemv_epi" : "_gemv",
  };
}

export function planTiledMatmul(
  options: DispatchMatmulOptions,
): MatmulStandardPlan | MatmulKSplitPlan {
  const {
    device,
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

  const hasEpilogue =
    !!epilogue &&
    epilogue.ops.length > 0 &&
    epilogue.ops.some((op) => op.kind !== "none");

  // Variant selection (data-driven registry). Lives HERE — the single seam
  // both dispatchTiledMatmul and the stage-4 stream generator consume — so
  // the executed path and generated streams can't disagree.
  const ctx: MatmulVariantContext = {
    m,
    n,
    k,
    batchSize,
    dtypeA: dtype,
    dtypeB: dtypeB ?? dtype,
    transA,
    transB,
    hasEpilogue,
    epiloguePresent: !!epilogue || epilogueInputs.length > 0,
    epilogue,
    hasInputCast: !!inputCastA || !!inputCastB,
    hasExplicitConfig: !!options.config,
    subgroupSupported: getSubgroupSupport()?.supported ?? false,
  };
  const choice = selectMatmulChoice(ctx, options.config);

  if (choice.variant === "gemv") {
    const gemvPlan = planGemvRowMatmul(
      device,
      n,
      k,
      transB,
      alpha,
      dtype,
      dtypeB ?? dtype,
      choice.wgSize,
      choice.rowsPerWg,
      epilogue,
      epilogueInputs.length,
    );
    if (gemvPlan) return gemvPlan;
    // Route degenerated for this choice — fall through to the tiled family.
  }

  const config =
    choice.variant === "tiled"
      ? choice.config
      : tiledChoiceForContext(ctx).config;
  validateConfig(config);
  const transposeMode = getTransposeMode(transA, transB);
  const lda = transA ? m : k;
  const ldb = transB ? k : n;
  const ldc = n;
  const batchStrideA = options.batchStrideA ?? m * k;
  const batchStrideB = options.batchStrideB ?? k * n;
  const batchStrideC = options.batchStrideC ?? m * n;
  const workgroupsX = Math.ceil(n / config.tileN);
  const workgroupsY = Math.ceil(m / config.tileM);
  const baseWorkgroups = workgroupsX * workgroupsY;
  const kSplitFactor = computeKSplitFactor(
    baseWorkgroups,
    k,
    config.tileK,
    batchSize,
    hasEpilogue,
  );
  if (kSplitFactor >= 2) {
    const outputDtype =
      epilogue?.outputDtype ??
      (dtype === "f32" || (dtypeB ?? dtype) === "f32" ? "f32" : dtype);
    const totalElements = m * n;
    const tempBytes = kSplitFactor * totalElements * F32_BYTES;
    const ksplitPipeline = getOrCreatePipeline(device, {
      config,
      transposeMode,
      dtype,
      dtypeB,
      batched: false,
      inputCastA,
      inputCastB,
      kSplit: kSplitFactor,
    });
    const ksplitParamsData = packMatmulParams(
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      1.0,
      1,
      batchStrideA,
      batchStrideB,
      batchStrideC,
    );
    const reducePipeline = getOrCreateReductionPipeline(
      device,
      kSplitFactor,
      outputDtype as DType,
    );
    const reduceParamsBuf = new ArrayBuffer(8);
    const reduceU32 = new Uint32Array(reduceParamsBuf);
    new Float32Array(reduceParamsBuf)[1] = alpha;
    reduceU32[0] = totalElements;
    return {
      kSplit: true,
      tempBytes,
      ksplitPipeline,
      ksplitParamsData,
      ksplitDispatch: [workgroupsX, workgroupsY, kSplitFactor],
      reducePipeline,
      reduceParamsData: reduceU32,
      reduceDispatch: [Math.ceil(totalElements / 256), 1, 1],
    };
  }

  const swapGrid =
    batchSize === 1 && workgroupsX > workgroupsY * 4 ? true : undefined;
  const codegenOptions: CodegenOptions = {
    config,
    transposeMode,
    dtype,
    dtypeB,
    epilogue,
    batched: batchSize > 1,
    inputCastA,
    inputCastB,
    swapGrid,
  };
  const pipeline = getOrCreatePipeline(device, codegenOptions);
  const paramsData = packMatmulParams(
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    batchSize,
    batchStrideA,
    batchStrideB,
    batchStrideC,
  );
  return {
    kSplit: false,
    pipeline,
    paramsData,
    dispatchX: swapGrid ? workgroupsY : workgroupsX,
    dispatchY: swapGrid ? workgroupsX : workgroupsY,
    dispatchZ: batchSize,
    numEpilogueInputs: epilogueInputs.length,
  };
}

export function dispatchTiledMatmul(options: DispatchMatmulOptions): void {
  const { device, a, b, out, epilogueInputs = [] } = options;

  // ALL selection decisions (variant, config, K-split, grid) are
  // single-sourced in planTiledMatmul (the stream generator consumes the
  // same plan — standard or K-split or GEMV).
  const plan = planTiledMatmul(options);
  if (plan.label?.startsWith("_gemv")) {
    gemvDispatchCount++;
    if (plan.label === "_gemv_epi") gemvEpilogueDispatchCount++;
  }

  if (plan.kSplit) {
    // --- K-split path: two dispatches over the cached partials temp. ---
    const opLabel = (getCurrentOpLabel() ?? "matmul") + (plan.label ?? "");
    const tempBuffer = getKSplitTempBuffer(device, plan.tempBytes);

    const kSplitParamsBuffer = sharedCreateParamsBuffer(
      device,
      plan.ksplitParamsData,
    );
    const kSplitBindGroup = cachedCreateBindGroup(device, plan.ksplitPipeline, [
      a,
      b,
      tempBuffer,
      kSplitParamsBuffer,
    ]);
    dispatchComputePass(
      plan.ksplitPipeline,
      kSplitBindGroup,
      plan.ksplitDispatch[0],
      plan.ksplitDispatch[1],
      plan.ksplitDispatch[2],
      opLabel,
    );
    releaseParamsBuffer(kSplitParamsBuffer);

    const reduceParamsBuffer = sharedCreateParamsBuffer(
      device,
      plan.reduceParamsData,
    );
    const reduceBindGroup = cachedCreateBindGroup(device, plan.reducePipeline, [
      tempBuffer,
      out,
      reduceParamsBuffer,
    ]);
    dispatchComputePass(
      plan.reducePipeline,
      reduceBindGroup,
      plan.reduceDispatch[0],
      plan.reduceDispatch[1],
      plan.reduceDispatch[2],
      opLabel + "_ksplit_reduce",
    );
    releaseParamsBuffer(reduceParamsBuffer);
    return;
  }

  const std = plan;
  const paramsBuffer = sharedCreateParamsBuffer(device, std.paramsData);
  const bgBuffers = [a, b, out, paramsBuffer, ...epilogueInputs];
  const bindGroup = cachedCreateBindGroup(device, std.pipeline, bgBuffers);
  dispatchComputePass(
    std.pipeline,
    bindGroup,
    std.dispatchX,
    std.dispatchY,
    std.dispatchZ,
    std.label ? (getCurrentOpLabel() ?? "matmul") + std.label : undefined,
  );
  releaseParamsBuffer(paramsBuffer);
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
