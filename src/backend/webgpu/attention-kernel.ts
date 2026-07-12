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
 * Kernel specs use the tile-IR block API: cooperative loads, block dot products
 * (auto-vec4), block reductions, and block stores. No manual vec4 or shared
 * memory code. Auto-CSE handles all sub-expression sharing.
 */

import { ENV } from "../../core/env";
import {
  type AttentionKernelRole,
  BC_BW,
  BR,
  realizeAttentionSpec,
} from "../../schedule/attention-skeleton";
import type { AttnModifierSpec } from "../types";
import { cachedCreateBindGroup } from "./bind-group-cache";
import { allocateOutputBuffer } from "./buffer-arena";
import { dispatchComputePass, getPipeline } from "./dispatch";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { compileTileKernel } from "./tile-compiler";
import type { TileKernelSpec } from "./tile-ir";
import {
  onTeardown,
  requireContext,
  trackSharedEncoderWrite,
} from "./webgpu-state";

// ============================================================================
// Attention modifier seams (task #64 — FlexAttention-class score/mask mods)
//
// Modifiers arrive as DATA (AttnModifierSpec, backend/types.ts) and are
// lowered here to tile-IR expressions at the kernels' declared seam points
// (spec.seams + ctx.applySeam — the #62 machinery):
//   - "attn_score": wraps the scaled pre-softmax score (fwd + both bwd
//     recompute sites). Args: qIdx, kvIdx, head, batch (u32 BlockExprs).
//   - "attn_mask": wraps the active-position predicate (AND'ed onto the
//     bounds check). Same args. Inactive positions get F32_NEG_MAX (fwd)
//     / p=0 (bwd), exactly like the legacy causal branch.
//
// A modifier's STRUCTURE (which kinds) is template identity → part of every
// WGSL/pipeline/config cache key via attnModifierKey(), the single source.
// Numeric params (cap, window) are uniform DATA, not key material — except
// in the config-buffer key, which keys uniform CONTENT (like scale already
// does).
//
// CAUSALITY IS A MODIFIER (commit B): the legacy is_causal uniform and its
// triplicated inline predicate were deleted; normalizeAttnModifier folds the
// public isCausal flag into a causal maskMod at every entry. Bit-parity of
// the two paths was gated BEFORE the deletion
// (test/webgpu/attention-modifier-parity.spec.ts, fwd + dQ/dK/dV exact).
//
// Backward for scoreMod is INFERENCE-FIRST: paired-derivative design lives
// in docs/attention-modifier-seams-design.md §2; until implemented,
// dispatching a backward kernel with a scoreMod throws. Mask mods need no
// derivative (constant structure) and are wired in backward.
// ============================================================================

/** Canonical structural key for a modifier — SINGLE SOURCE for every cache
 *  seam (WGSL cache, pipeline cache, config-buffer cache, and the decode
 *  bucketKey fragment). "" = null modifier = legacy keys, byte-stable. */
export function attnModifierKey(mod?: AttnModifierSpec): string {
  if (!mod) return "";
  const parts: string[] = [];
  if (mod.scoreMod) parts.push(`s.${mod.scoreMod.kind}`);
  for (const m of mod.maskMods ?? []) parts.push(`m.${m.kind}`);
  return parts.join("+");
}

export function hasCausalMask(mod?: AttnModifierSpec): boolean {
  return (mod?.maskMods ?? []).some((m) => m.kind === "causal");
}

const _f32BitsBuf = new Float32Array(1);
const _f32BitsU32 = new Uint32Array(_f32BitsBuf.buffer);
function f32Bits(x: number): number {
  _f32BitsBuf[0] = x;
  return _f32BitsU32[0];
}

/** Uniform-content words contributed by modifier params (packed into the
 *  config buffer's words 5..7, after B,H,N,D,scale). ORDER IS THE CONTRACT
 *  with modifierUniformFields: scoreMod param first, then mask-mod params in
 *  maskMods order — declaration order == pack order. */
function modifierParamWords(mod?: AttnModifierSpec): number[] {
  if (!mod) return [];
  const words: number[] = [];
  if (mod.scoreMod) words.push(f32Bits(mod.scoreMod.cap));
  for (const m of mod.maskMods ?? []) {
    if (m.kind === "slidingWindow") words.push(m.window >>> 0);
  }
  return words;
}

/** True when the modifier adds nothing — canonicalized to undefined so `{}`
 *  and undefined share the legacy templates (a `{}`-keyed spec would emit
 *  different WGSL under the SAME "" cache key — the collision this guards). */
function isNullModifier(mod?: AttnModifierSpec): boolean {
  return !mod || (!mod.scoreMod && !mod.maskMods?.length);
}

/**
 * Canonicalize (isCausal, modifier) at every public dispatch/plan entry —
 * the SINGLE place the two composition surfaces meet. Since the legacy
 * is_causal uniform branch was deleted (commit B; exonerated bit-exactly by
 * test/webgpu/attention-modifier-parity.spec.ts), causality ALWAYS travels
 * as a causal maskMod: isCausal=true folds INTO the modifier (structural).
 * Null modifier + isCausal=false canonicalizes to undefined ({} ≡ undefined
 * — guards the ""-key WGSL collision).
 */
function normalizeAttnModifier(
  isCausal: boolean,
  mod?: AttnModifierSpec,
): { mod?: AttnModifierSpec } {
  if (isNullModifier(mod)) {
    return isCausal ? { mod: { maskMods: [{ kind: "causal" }] } } : {};
  }
  const m = mod as AttnModifierSpec;
  const maskMods =
    isCausal && !hasCausalMask(m)
      ? [{ kind: "causal" } as const, ...(m.maskMods ?? [])]
      : (m.maskMods ?? []);
  // Canonical order (sort by kind): semantically identical compositions
  // share one template regardless of declaration order. Duplicate kinds
  // would collide on uniform field names — reject loudly.
  const sorted = [...maskMods].sort((a, b) => a.kind.localeCompare(b.kind));
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].kind === sorted[i - 1].kind) {
      throw new Error(
        `attention modifier has duplicate maskMod kind '${sorted[i].kind}'`,
      );
    }
  }
  return { mod: { ...m, maskMods: sorted } };
}

/** Append the modifier fragment to a cache key (WGSL or pipeline). */
function withModKey(base: string, mod?: AttnModifierSpec): string {
  const k = attnModifierKey(mod);
  return k ? `${base}:${k}` : base;
}

/** Backward is inference-first for score modifiers: the paired-derivative
 *  ("attn_dscore") emission is designed (doc §2) but not implemented. Mask
 *  mods are constant structure (no derivative) and are supported. */
export function assertBackwardSupportsModifier(mod?: AttnModifierSpec): void {
  if (mod?.scoreMod) {
    throw new Error(
      `attention backward with scoreMod '${mod.scoreMod.kind}' is not ` +
        `implemented (inference-first; see docs/attention-modifier-seams-design.md §2)`,
    );
  }
}

// ============================================================================
// WGSL Cache & Config Buffer Cache
// ============================================================================

const tileIRWGSLCache = new Map<string, string>();
function getTileIRWGSL(key: string, specFactory: () => TileKernelSpec): string {
  let wgsl = tileIRWGSLCache.get(key);
  if (!wgsl) {
    wgsl = compileTileKernel(specFactory());
    tileIRWGSLCache.set(key, wgsl);
  } else if (ENV.TORCHLETTE_STRICT_GPU === "1") {
    // Seam assertion (#64 iv): the key is DERIVED from the modifier spec
    // (attnModifierKey), so a hit whose fresh emission differs means the key
    // fn and the emission fn drifted — the cache-collision class that
    // produces correct-looking-but-wrong kernels. Strict mode pays a
    // recompile per hit to make that drift LOUD.
    const fresh = compileTileKernel(specFactory());
    if (fresh !== wgsl) {
      throw new Error(
        `attention WGSL cache collision under key '${key}': the modifier ` +
          `key fn and the kernel emission drifted (attnModifierKey vs ` +
          `buildAttentionSeams/modifierUniformFields)`,
      );
    }
  }
  return wgsl;
}

const configCache = new Map<string, GPUBuffer>();

/** Config-buffer cache key: uniform CONTENT identity (dims + scale +
 *  modifier structure + modifier param values; causality lives in the
 *  modifier since commit B). Single source — used by
 *  getOrCreateConfigBuffer and lookupAttentionConfigBuffer. */
function attentionConfigKey(
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  mod?: AttnModifierSpec,
): string {
  const base = `${batchSize}:${numHeads}:${seqLen}:${headDim}:${scale}`;
  const modKey = attnModifierKey(mod);
  if (!modKey) return base;
  return `${base}:${modKey}:${modifierParamWords(mod).join(",")}`;
}

function getOrCreateConfigBuffer(
  device: GPUDevice,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  mod?: AttnModifierSpec,
): GPUBuffer {
  const key = attentionConfigKey(
    batchSize,
    numHeads,
    seqLen,
    headDim,
    scale,
    mod,
  );
  let buf = configCache.get(key);
  if (buf) return buf;

  buf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false,
  });

  // Layout: [0..4] = B,H,N,D,scale; [5..7] = modifier param words (causality
  // is template structure since commit B, not uniform content).
  const data = new ArrayBuffer(32);
  const u32View = new Uint32Array(data);
  const f32View = new Float32Array(data);
  u32View[0] = batchSize;
  u32View[1] = numHeads;
  u32View[2] = seqLen;
  u32View[3] = headDim;
  f32View[4] = scale;
  const params = modifierParamWords(mod);
  for (let i = 0; i < params.length; i++) {
    if (5 + i > 7)
      throw new Error("attention modifier params exceed config pad words");
    u32View[5 + i] = params[i];
  }
  device.queue.writeBuffer(buf, 0, new Uint8Array(data));
  // Cache it (the "getOrCreate" contract — was missing, so every attention
  // dispatch re-created + re-uploaded this 32-byte uniform; caching is a
  // per-step perf win AND gives the stream generator a stable buffer
  // identity to bind as a persistent slot via lookupAttentionConfigBuffer).
  configCache.set(key, buf);
  return buf;
}

// ============================================================================
// Kernel Specs (tile-IR) — ABSORBED into the schedule module (§7 P4 cutover-flip)
// ============================================================================
//
// The four fused-attention kernel bodies (forward, D-precompute, backward-dQ,
// backward-dKV) were RELOCATED into src/schedule/attention-skeleton.ts, where
// the ScheduleState now OWNS the loop-nest / K-V staging / online-softmax /
// backward-recompute structure. The live dispatch/plan sites below route their
// spec factories through `realizeAttentionSpec` (the schedule chokepoint) — the
// schedule object is the sole WGSL writer at the dispatch seam. The retired
// `make*Spec` factories (and their body-only seam/uniform/name helpers) are gone;
// the byte differential (test/schedule/attention-differential.spec.ts) guards the
// LIVE path.

// ============================================================================
// Dispatch Functions
// ============================================================================

/** Shared attention dispatch: config buffer + WGSL cache + pipeline + bind
 *  group + tracking. `isCausal`/`mod` must already be normalized
 *  (normalizeAttnModifier) by the caller. A modifier is template identity:
 *  its key fragment goes into BOTH the WGSL cache key and the pipeline key
 *  (same emitted-source, same key — diverging them is the cache-collision
 *  class this file's header warns about). */
function dispatchAttention(
  wgslKey: string,
  pipelinePrefix: string,
  specFactory: () => TileKernelSpec,
  headDim: number,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  scale: number,
  // Template identity vs config content: templateMod keys the WGSL/pipeline
  // caches (undefined for D-precompute — no seam sites, one template);
  // configMod keys the SHARED config buffer all four dispatches bind (its
  // pad words carry the modifier params, so its key always carries the mod).
  templateMod: AttnModifierSpec | undefined,
  configMod: AttnModifierSpec | undefined,
  buffers: GPUBuffer[], // all data buffers (config appended automatically)
  ...grid: number[]
): void {
  const ctx = requireContext();
  const configBuf = getOrCreateConfigBuffer(
    ctx.device,
    batchSize,
    numHeads,
    seqLen,
    headDim,
    scale,
    configMod,
  );
  const wgsl = getTileIRWGSL(withModKey(wgslKey, templateMod), specFactory);
  const pipeline = getPipeline(
    ctx,
    withModKey(`${pipelinePrefix}:tile:${headDim}`, templateMod),
    wgsl,
  );
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
    ...buffers,
    configBuf,
  ]);
  for (const b of buffers) trackSharedEncoderWrite(b);
  dispatchComputePass(pipeline, bindGroup, grid[0], grid[1] ?? 1, grid[2] ?? 1);
}

// ============================================================================
// Stage-4 stream generation: plan helpers (no GPU dispatch). Each attention
// dispatch resolves its pipeline through the SAME getTileIRWGSL+getPipeline
// caches the dispatcher uses, and binds the shared CACHED config buffer
// (getOrCreateConfigBuffer — a persistent slot, looked up read-only here).
// All four attention dispatches in a step share one config buffer (same
// B/H/N/D/scale/causal). The ephemeral D-precompute is a normal recorded
// allocateOutputBuffer (allocKind 1), not a persistent workspace.
// ============================================================================

export interface AttentionStepPlan {
  pipeline: import("./gpu-types").GPUComputePipeline;
  configBuffer: GPUBuffer | null;
  grid: [number, number, number];
}

/** Read-only lookup of the cached attention config buffer (persistent slot
 *  in the stream). Null if not yet created (guaranteed hit post-recording).
 *  Expects a NORMALIZED modifier — plan* entries normalize. */
export function lookupAttentionConfigBuffer(
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  mod?: AttnModifierSpec,
): GPUBuffer | null {
  return (
    configCache.get(
      attentionConfigKey(batchSize, numHeads, seqLen, headDim, scale, mod),
    ) ?? null
  );
}

function planAttentionStep(
  wgslKey: string,
  pipelinePrefix: string,
  specFactory: () => TileKernelSpec,
  headDim: number,
  mod: AttnModifierSpec | undefined,
  configBuffer: GPUBuffer | null,
  grid: [number, number, number],
): AttentionStepPlan {
  const ctx = requireContext();
  const wgsl = getTileIRWGSL(withModKey(wgslKey, mod), specFactory);
  const pipeline = getPipeline(
    ctx,
    withModKey(`${pipelinePrefix}:tile:${headDim}`, mod),
    wgsl,
  );
  return { pipeline, configBuffer, grid };
}

export function planFlashAttentionForward(
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: boolean,
  modifier?: AttnModifierSpec,
): { plan: AttentionStepPlan; outBytes: number; lseBytes: number } {
  const norm = normalizeAttnModifier(isCausal, modifier);
  const cfg = lookupAttentionConfigBuffer(
    batchSize,
    numHeads,
    seqLen,
    headDim,
    scale,
    norm.mod,
  );
  return {
    plan: planAttentionStep(
      `fwd:${headDim}`,
      "faFwd",
      () => realizeAttentionSpec("forward", headDim, norm.mod),
      headDim,
      norm.mod,
      cfg,
      [Math.ceil(seqLen / BR), numHeads, batchSize],
    ),
    outBytes: batchSize * numHeads * seqLen * headDim * 4,
    lseBytes: batchSize * numHeads * seqLen * 4,
  };
}

export function planFlashAttentionBackward(
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: boolean,
  modifier?: AttnModifierSpec,
): {
  dPlan: AttentionStepPlan;
  dqPlan: AttentionStepPlan;
  dkvPlan: AttentionStepPlan;
  dBytes: number;
  dqBytes: number;
  dkvBytes: number;
} {
  const norm = normalizeAttnModifier(isCausal, modifier);
  assertBackwardSupportsModifier(norm.mod);
  const cfg = lookupAttentionConfigBuffer(
    batchSize,
    numHeads,
    seqLen,
    headDim,
    scale,
    norm.mod,
  );
  const bhnd = batchSize * numHeads * seqLen * headDim * 4;
  return {
    dPlan: planAttentionStep(
      `bwdD:${headDim}`,
      "faBwdD",
      () => realizeAttentionSpec("dPrecompute", headDim),
      headDim,
      undefined, // D-precompute has no score/mask sites — one template
      cfg,
      [batchSize * numHeads * seqLen, 1, 1],
    ),
    dqPlan: planAttentionStep(
      `bwdDQ:${headDim}`,
      "faBwdDQ",
      () => realizeAttentionSpec("backwardDQ", headDim, norm.mod),
      headDim,
      norm.mod,
      cfg,
      [Math.ceil(seqLen / BR), numHeads, batchSize],
    ),
    dkvPlan: planAttentionStep(
      `bwdDKV:${headDim}`,
      "faBwdDKV",
      () => realizeAttentionSpec("backwardDKV", headDim, norm.mod),
      headDim,
      norm.mod,
      cfg,
      [Math.ceil(seqLen / BC_BW), numHeads, batchSize],
    ),
    dBytes: batchSize * numHeads * seqLen * 4,
    dqBytes: bhnd,
    dkvBytes: bhnd,
  };
}

/** Q,K,V: [B, H, N, D] → O: [B, H, N, D], L: [B, H, N] */
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
  modifier?: AttnModifierSpec,
): { outputBuffer: GPUBuffer; logsumexpBuffer: GPUBuffer } {
  const norm = normalizeAttnModifier(isCausal, modifier);
  const outBuffer = allocateOutputBuffer(
    batchSize * numHeads * seqLen * headDim * 4,
  );
  const lseBuffer = allocateOutputBuffer(batchSize * numHeads * seqLen * 4);
  dispatchAttention(
    `fwd:${headDim}`,
    "faFwd",
    () => realizeAttentionSpec("forward", headDim, norm.mod),
    headDim,
    batchSize,
    numHeads,
    seqLen,
    scale,
    norm.mod,
    norm.mod,
    [qBuffer, kBuffer, vBuffer, outBuffer, lseBuffer],
    Math.ceil(seqLen / BR),
    numHeads,
    batchSize,
  );
  return { outputBuffer: outBuffer, logsumexpBuffer: lseBuffer };
}

/** dO,O: [B,H,N,D] → D: [B,H,N] */
export function dispatchFlashAttentionBackwardD(
  dOBuffer: GPUBuffer,
  oBuffer: GPUBuffer,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: boolean,
  modifier?: AttnModifierSpec,
): GPUBuffer {
  const norm = normalizeAttnModifier(isCausal, modifier);
  assertBackwardSupportsModifier(norm.mod);
  const outBuffer = allocateOutputBuffer(batchSize * numHeads * seqLen * 4);
  dispatchAttention(
    `bwdD:${headDim}`,
    "faBwdD",
    () => realizeAttentionSpec("dPrecompute", headDim),
    headDim,
    batchSize,
    numHeads,
    seqLen,
    scale,
    undefined, // D-precompute: no seam sites — one template for all mods
    norm.mod,
    [dOBuffer, oBuffer, outBuffer],
    batchSize * numHeads * seqLen,
  );
  return outBuffer;
}

/** Q,K,V,L,D,dO → dQ: [B,H,N,D] */
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
  modifier?: AttnModifierSpec,
): GPUBuffer {
  const norm = normalizeAttnModifier(isCausal, modifier);
  assertBackwardSupportsModifier(norm.mod);
  const outBuffer = allocateOutputBuffer(
    batchSize * numHeads * seqLen * headDim * 4,
  );
  dispatchAttention(
    `bwdDQ:${headDim}`,
    "faBwdDQ",
    () => realizeAttentionSpec("backwardDQ", headDim, norm.mod),
    headDim,
    batchSize,
    numHeads,
    seqLen,
    scale,
    norm.mod,
    norm.mod,
    [qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, outBuffer],
    Math.ceil(seqLen / BR),
    numHeads,
    batchSize,
  );
  return outBuffer;
}

/** Q,K,V,L,D,dO → dK: [B,H,N,D], dV: [B,H,N,D] */
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
  modifier?: AttnModifierSpec,
): { dKBuffer: GPUBuffer; dVBuffer: GPUBuffer } {
  const norm = normalizeAttnModifier(isCausal, modifier);
  assertBackwardSupportsModifier(norm.mod);
  const dKBuffer = allocateOutputBuffer(
    batchSize * numHeads * seqLen * headDim * 4,
  );
  const dVBuffer = allocateOutputBuffer(
    batchSize * numHeads * seqLen * headDim * 4,
  );
  dispatchAttention(
    `bwdDKV:${headDim}`,
    "faBwdDKV",
    () => realizeAttentionSpec("backwardDKV", headDim, norm.mod),
    headDim,
    batchSize,
    numHeads,
    seqLen,
    scale,
    norm.mod,
    norm.mod,
    [qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, dKBuffer, dVBuffer],
    Math.ceil(seqLen / BC_BW),
    numHeads,
    batchSize,
  );
  return { dKBuffer, dVBuffer };
}

/** Reset all module-local mutable state (pipeline cache, config buffer cache). */
export function resetAttentionKernelState(): void {
  configCache.clear();
  tileIRWGSLCache.clear();
}
onTeardown(resetAttentionKernelState);
