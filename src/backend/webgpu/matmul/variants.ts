/**
 * Data-driven matmul kernel-variant registry.
 *
 * The logical matmul op has multiple kernel FAMILIES (variants) — the tiled
 * shared-memory kernel family and the dedicated M=1 GEMV kernels — each with
 * its own parameter space. This module makes "which kernel family, with which
 * parameters" a data + measurement decision (cuBLASLt algorithm enumeration /
 * Inductor benchmark-selection style) instead of ad-hoc branches:
 *
 *  - `isApplicable(ctx)`  — geometry/capability gate (pure, plan-time safe);
 *  - `defaultChoice(ctx)` — the heuristic pick used when autotune is off
 *                           (MUST reproduce the pre-registry behavior);
 *  - `candidates(ctx)`    — the search space the autotuner benchmarks.
 *
 * Selection order = registry order (first applicable variant wins the
 * heuristic path). The autotuner benchmarks ALL applicable variants ×
 * candidates and caches the winning (variant, choice) per exact shape; the
 * dispatcher's cache hit short-circuits straight to the winner (subject to a
 * re-check of `isApplicable` — a choice tuned for one context, e.g. bare
 * unbatched, is never forced onto a context where its variant can't run).
 *
 * Device limits: the only limit the selection depends on is the guaranteed
 * WebGPU maxComputeWorkgroupsPerDimension, folded into computeGemvRoute;
 * subgroup capability enters via ctx.subgroupSupported (a config FLAG axis
 * inside the tiled variant — `useSubgroups` — not a separate variant, since
 * subgroup codegen is a parameter of the tiled shader generator).
 *
 * NOT exported above the matmul module: dispatch.ts is the only consumer.
 */

import { ENV } from "../../../core/env";
import { generateNeighborConfigs, getDefaultConfigForShape } from "./autotune";
import {
  computeGemvRoute,
  GEMV_DEFAULT_WG_SIZE,
  GEMV_WG_SIZE_PARAM,
} from "./gemv";
import {
  classifyShape,
  DEFAULT_CONFIG,
  type DType,
  type MatmulKernelConfig,
} from "./types";

/**
 * Everything variant selection is allowed to look at. Geometry + dtypes +
 * epilogue/cast presence + capability. Pure data — safe at plan/lowering
 * time (the stage-4 stream generator runs the same selection).
 */
export interface MatmulVariantContext {
  m: number;
  n: number;
  k: number;
  batchSize: number;
  dtypeA: DType;
  dtypeB: DType;
  transA: boolean;
  transB: boolean;
  /** Non-trivial epilogue ops present (tiled config selection keys on this). */
  hasEpilogue: boolean;
  /** ANY epilogue object or epilogue inputs present (GEMV requires none). */
  epiloguePresent: boolean;
  /** inputCastA/inputCastB present (GEMV kernels don't implement load-casts). */
  hasInputCast: boolean;
  /** Caller pinned an explicit MatmulKernelConfig (forces the tiled variant). */
  hasExplicitConfig: boolean;
  /** Subgroup ops available (tiled `useSubgroups` config axis). */
  subgroupSupported: boolean;
}

/** A concrete, dispatchable selection: variant + its variant-specific params. */
export type MatmulVariantChoice =
  | {
      variant: "tiled";
      config: MatmulKernelConfig;
    }
  | {
      variant: "gemv";
      /** Threads per workgroup; the NN route geometry (grid, split-k) and the
       *  shader are both derived from this at plan time via computeGemvRoute
       *  — single source, nothing else in the choice can disagree with it. */
      wgSize: number;
    };

export interface MatmulVariant {
  readonly name: MatmulVariantChoice["variant"];
  /** Can this variant execute the op described by ctx at all? */
  isApplicable(ctx: MatmulVariantContext): boolean;
  /** Heuristic pick (autotune off). Must match pre-registry behavior. */
  defaultChoice(ctx: MatmulVariantContext): MatmulVariantChoice;
  /** Autotune search space for this ctx (default choice included). */
  candidates(ctx: MatmulVariantContext): MatmulVariantChoice[];
}

const tiledConfigKey = (c: MatmulKernelConfig): string =>
  `${c.tileM}_${c.tileN}_${c.tileK}_${c.threadTileM}_${c.threadTileN}_${c.vectorWidth}_${c.useSubgroups}`;

/**
 * GEMV variant: M=1, batch=1, bare (no epilogue/casts), opt-out via
 * TORCHLETTE_GEMV=0. Mirrors exactly the pre-registry guard in
 * planTiledMatmul + computeGemvRoute's geometry bail-outs (the "tiled
 * gemv_row config wins here" threshold lives in the route, single-sourced).
 */
const gemvVariant: MatmulVariant = {
  name: "gemv",
  isApplicable(ctx) {
    return (
      ctx.m === 1 &&
      ctx.batchSize === 1 &&
      !ctx.epiloguePresent &&
      !ctx.hasInputCast &&
      !ctx.hasExplicitConfig &&
      ENV.TORCHLETTE_GEMV !== "0" &&
      computeGemvRoute(ctx.n, ctx.k, ctx.transB) !== null
    );
  },
  defaultChoice() {
    return { variant: "gemv", wgSize: GEMV_DEFAULT_WG_SIZE };
  },
  candidates(ctx) {
    // One TuneParam: wgSize (shared with gemvAutotuneConfig — same source).
    // A candidate wgSize must itself yield a valid route (the NN route's
    // split-k / grid thresholds depend on it).
    return GEMV_WG_SIZE_PARAM.values
      .filter((w) => computeGemvRoute(ctx.n, ctx.k, ctx.transB, w) !== null)
      .map((wgSize) => ({ variant: "gemv" as const, wgSize }));
  },
};

/** Tiled shared-memory kernel family. Always applicable (the fallback). */
const tiledVariant: MatmulVariant = {
  name: "tiled",
  isApplicable() {
    return true;
  },
  defaultChoice(ctx) {
    return {
      variant: "tiled",
      config: getDefaultConfigForShape(
        classifyShape(ctx.m, ctx.n, ctx.k, ctx.batchSize),
        ctx.hasEpilogue,
        ctx.m,
        ctx.n,
        ctx.k,
      ),
    };
  },
  candidates(ctx) {
    const base = (this.defaultChoice(ctx) as { config: MatmulKernelConfig })
      .config;
    const configs = generateNeighborConfigs(base, ctx.subgroupSupported);
    // Always include the conservative DEFAULT_CONFIG in the search set.
    if (
      !configs.some((c) => tiledConfigKey(c) === tiledConfigKey(DEFAULT_CONFIG))
    ) {
      configs.push(DEFAULT_CONFIG);
    }
    return configs.map((config) => ({ variant: "tiled" as const, config }));
  },
};

/**
 * The registry: ordered — the heuristic path picks the FIRST applicable
 * variant, so more-specialized variants come first.
 */
export const MATMUL_VARIANTS: readonly MatmulVariant[] = [
  gemvVariant,
  tiledVariant,
];

export function getMatmulVariant(
  name: MatmulVariantChoice["variant"],
): MatmulVariant {
  const v = MATMUL_VARIANTS.find((x) => x.name === name);
  if (!v) throw new Error(`Unknown matmul variant: ${name}`);
  return v;
}
