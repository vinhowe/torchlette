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

import type { AttnModifierSpec } from "../types";
import { cachedCreateBindGroup } from "./bind-group-cache";
import { allocateOutputBuffer } from "./buffer-arena";
import { dispatchComputePass, getPipeline } from "./dispatch";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { F32_NEG_MAX, WORKGROUP_SIZE } from "./shape-utils";
import { compileTileKernel } from "./tile-compiler";
import type {
  BlockExpr,
  KernelContext,
  SeamFn,
  TileKernelSpec,
} from "./tile-ir";
import { tiledGrid } from "./tile-ir";
import {
  onTeardown,
  requireContext,
  trackSharedEncoderWrite,
} from "./webgpu-state";

// ============================================================================
// Tiling Parameters
// ============================================================================

const BR = 64; // Q rows per workgroup (forward, dQ)
const BC = 32; // KV rows per tile (forward, dQ)
const BQ_BW = 16; // Q rows per tile (backward dKV)
const BC_BW = 64; // KV rows per workgroup (backward dKV)

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

function hasCausalMask(mod?: AttnModifierSpec): boolean {
  return (mod?.maskMods ?? []).some((m) => m.kind === "causal");
}

/** Build the seam functions for a modifier (undefined for the null modifier
 *  — applySeam is identity, the kernel is the bare bounds-checked softmax). */
function buildAttentionSeams(
  mod: AttnModifierSpec | undefined,
): Record<string, SeamFn> | undefined {
  if (!mod) return undefined;
  const seams: Record<string, SeamFn> = {};
  const maskMods = mod.maskMods ?? [];
  if (maskMods.length > 0) {
    seams.attn_mask = (
      ctx: KernelContext,
      active: BlockExpr,
      args: Record<string, BlockExpr>,
    ) => {
      let a = active;
      for (const m of maskMods) {
        if (m.kind === "causal") {
          a = a.and(args.kvIdx.le(args.qIdx));
        } else if (m.kind === "slidingWindow") {
          // active iff kv > q − window, computed u32-underflow-safe as
          // kv + window > q. The window value is uniform DATA (mod_window)
          // — same template serves any window size.
          a = a.and(args.kvIdx.add(ctx.uniform("mod_window")).gt(args.qIdx));
        } else {
          throw new Error(
            `attention maskMod '${(m as { kind: string }).kind}' not implemented in kernel emission`,
          );
        }
      }
      return a;
    };
  }
  if (mod.scoreMod) {
    if (mod.scoreMod.kind !== "softcap") {
      throw new Error(
        `attention scoreMod '${(mod.scoreMod as { kind: string }).kind}' not implemented in kernel emission`,
      );
    }
    // Logit soft-cap: s' = cap · tanh(s / cap) (Gemma-2). Emitted in f32 —
    // modifier arithmetic stays f32 under f16 QKV (mandatory f16 gate).
    // Backward's paired "attn_dscore" (1 − (s'/cap)²) is inference-first —
    // backward entries throw via assertBackwardSupportsModifier.
    seams.attn_score = (ctx: KernelContext, sVal: BlockExpr) => {
      const cap = ctx.uniform("mod_softcap").bitcastTo("f32");
      return sVal.div(cap).tanh().mul(cap);
    };
  }
  return seams;
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

/** Uniform struct fields for modifier params — paired with
 *  modifierParamWords (same order; see its doc). Spread after scale_u32 in
 *  each seam-site spec's uniforms. */
function modifierUniformFields(mod?: AttnModifierSpec): Record<string, "u32"> {
  if (!mod) return {};
  const fields: Record<string, "u32"> = {};
  if (mod.scoreMod) fields.mod_softcap = "u32";
  for (const m of mod.maskMods ?? []) {
    if (m.kind === "slidingWindow") fields.mod_window = "u32";
  }
  return fields;
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
function assertBackwardSupportsModifier(mod?: AttnModifierSpec): void {
  if (mod?.scoreMod) {
    throw new Error(
      `attention backward with scoreMod '${mod.scoreMod.kind}' is not ` +
        `implemented (inference-first; see docs/attention-modifier-seams-design.md §2)`,
    );
  }
}

/** WGSL-identifier-safe name fragment ("" for null modifier). */
function modNameFragment(mod?: AttnModifierSpec): string {
  const k = attnModifierKey(mod);
  return k ? `_${k.replace(/[^A-Za-z0-9]/g, "_")}` : "";
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
// Kernel Specs (tile-IR)
// ============================================================================

function makeForwardAttentionSpec(
  headDim: number,
  mod?: AttnModifierSpec,
): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BR;

  return {
    name: `tileAttnFwd_D${D}${modNameFragment(mod)}`,
    workgroupSize: WG,
    autoBarriers: true,
    seams: buildAttentionSeams(mod),
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      O: { storage: "read_write", type: "f32" },
      L: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      ...modifierUniformFields(mod),
    },
    grid: tiledGrid({
      x: { uniform: "seq_len", tileSize: BR },
      y: "num_heads",
      z: "batch_size",
    }),

    kernel(ctx) {
      const tidx = ctx.localIndex();
      const qBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const qRow = qBlock.mul(ctx.u32(BR)).add(tidx);
      const valid = qRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(N);
      const qBase = bhOff.add(qRow.mul(Dim));

      const Q = ctx.tileLoad(
        "Q",
        {
          kind: "thread",
          base: qBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const mPrev = ctx.full(1, 1, F32_NEG_MAX);
      const lPrev = ctx.full(1, 1, 0);
      const oAcc = ctx.zeros(1, D);

      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));

        const offsR = ctx.arange(kvStart, BC);
        const offsD = ctx.arange(ctx.u32(0), D);
        const tilePtr = ctx.tilePtr(
          bhOff,
          offsR.outer(Dim),
          offsD.inner(ctx.u32(1)),
        );
        const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
        const K = ctx.load2D("K", tilePtr, tileMask);

        const scores = ctx.dot(Q, K.T());

        ctx.range(0, BC, (j) => {
          const kvPos = kvStart.add(j);
          const seamArgs = {
            qIdx: qRow,
            kvIdx: kvPos,
            head: hIdx,
            batch: bIdx,
          };
          // Seam "attn_mask": mask predicates (incl. causal) are structural
          // modifier emissions AND'ed onto the bounds check.
          const isActive = ctx.applySeam(
            "attn_mask",
            valid.and(kvPos.lt(N)),
            seamArgs,
          );
          // Seam "attn_score": wraps the scaled pre-softmax score.
          const s = ctx.applySeam(
            "attn_score",
            scores.get(j).mul(scale),
            seamArgs,
          );
          scores.set(j, isActive.select(s, ctx.f32(F32_NEG_MAX)));
        });

        const mNew = scores.max(1);
        const mMax = mNew.max(mPrev);
        const correction = mPrev.sub(mMax).exp();

        oAcc.mul_(correction);
        lPrev.mul_(correction);

        scores.sub_(mMax);
        scores.exp_();
        lPrev.add_(scores.sum(1));
        mPrev.assign(mMax);

        const V = ctx.load2D("V", tilePtr, tileMask, { reuseShared: K });
        ctx.dotAccum(scores, V, oAcc);
      });

      ctx.ifThen(valid, () => {
        const l = lPrev.get(ctx.u32(0));
        const invL = l.gt(ctx.f32(0)).select(ctx.f32(1).div(l), ctx.f32(0));
        oAcc.mul_(invL);
        ctx.tileStore("O", oAcc, { base: qBase, stride: ctx.u32(1) });

        const m = mPrev.get(ctx.u32(0));
        const lse = m.add(l.max(ctx.f32(1e-10)).log());
        ctx.emitStore("L", bhOffL.add(qRow), lse);
      });
    },
  };
}

function makeDPrecomputeSpec(headDim: number): TileKernelSpec {
  const D = headDim;
  const WG = WORKGROUP_SIZE;

  return {
    name: `tileAttnDPrecompute_D${D}`,
    workgroupSize: WG,
    bindings: {
      dO: { storage: "read", type: "f32" },
      Out: { storage: "read", type: "f32" },
      D_val: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
    },
    grid: (u) => [u.batch_size * u.num_heads * u.seq_len],

    kernel(ctx) {
      const row = ctx.programId(0);
      const tid = ctx.localIndex();
      const Dim = ctx.uniform("head_dim");
      const base = row.mul(Dim);

      const dotProd = ctx.wgReduce("sum", tid, Dim, WG, (i) =>
        ctx.load("dO", base.add(i)).mul(ctx.load("Out", base.add(i))),
      );
      ctx.guardedStore("D_val", tid.eq(ctx.u32(0)), row, dotProd);
    },
  };
}

function makeBackwardDQSpec(
  headDim: number,
  mod?: AttnModifierSpec,
): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BR;

  return {
    name: `tileAttnBwdDQ_D${D}${modNameFragment(mod)}`,
    workgroupSize: WG,
    autoBarriers: true,
    seams: buildAttentionSeams(mod),
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      L_buf: { storage: "read", type: "f32" },
      D_buf: { storage: "read", type: "f32" },
      dO: { storage: "read", type: "f32" },
      dQ: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      ...modifierUniformFields(mod),
    },
    grid: tiledGrid({
      x: { uniform: "seq_len", tileSize: BR },
      y: "num_heads",
      z: "batch_size",
    }),

    kernel(ctx) {
      const tidx = ctx.localIndex();
      const qBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const qRow = qBlock.mul(ctx.u32(BR)).add(tidx);
      const valid = qRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(N);
      const rowBase = bhOff.add(qRow.mul(Dim));

      const Q = ctx.tileLoad(
        "Q",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const dO = ctx.tileLoad(
        "dO",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const lVar = ctx.emitVar("_Li", "f32", ctx.f32(0));
      const dVar = ctx.emitVar("_Di", "f32", ctx.f32(0));
      ctx.ifThen(valid, () => {
        lVar.set(ctx.load("L_buf", bhOffL.add(qRow)));
        dVar.set(ctx.load("D_buf", bhOffL.add(qRow)));
      });

      const dqAcc = ctx.zeros(1, D);

      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));

        const offsR = ctx.arange(kvStart, BC);
        const offsD = ctx.arange(ctx.u32(0), D);
        const tilePtr = ctx.tilePtr(
          bhOff,
          offsR.outer(Dim),
          offsD.inner(ctx.u32(1)),
        );
        const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
        const K = ctx.load2D("K", tilePtr, tileMask);
        const V = ctx.load2D("V", tilePtr, tileMask);

        // Fused single-loop: compute score, p, ds per KV-row, accumulate dQ inline
        ctx.range(0, BC, (j) => {
          const kvPos = kvStart.add(j);
          const seamArgs = {
            qIdx: qRow,
            kvIdx: kvPos,
            head: hIdx,
            batch: bIdx,
          };
          const isActive = ctx.applySeam(
            "attn_mask",
            valid.and(kvPos.lt(N)),
            seamArgs,
          );
          // Flash-style recompute: raw score is rebuilt from Q·K, so the
          // score modifier's forward AND its local derivative (the
          // "attn_dscore" seam, chain factor d(modded)/d(raw)) are available
          // inline — no extra saved tensor.
          const s = ctx.dotRow(Q, K, j).mul(scale);
          const sMod = ctx.applySeam("attn_score", s, seamArgs);
          const dov = ctx.dotRow(dO, V, j);
          const p = isActive.select(sMod.sub(lVar.get()).exp(), ctx.f32(0));
          const ds = ctx.applySeam("attn_dscore", p.mul(dov.sub(dVar.get())), {
            ...seamArgs,
            raw: s,
            modded: sMod,
          });
          ctx.accumRow(dqAcc, ds, K, j);
        });
      });

      ctx.ifThen(valid, () => {
        dqAcc.mul_(scale);
        ctx.tileStore("dQ", dqAcc, { base: rowBase, stride: ctx.u32(1) });
      });
    },
  };
}

function makeBackwardDKVSpec(
  headDim: number,
  mod?: AttnModifierSpec,
): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BC_BW;

  return {
    name: `tileAttnBwdDKV_D${D}${modNameFragment(mod)}`,
    workgroupSize: WG,
    autoBarriers: true,
    seams: buildAttentionSeams(mod),
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      L_buf: { storage: "read", type: "f32" },
      D_buf: { storage: "read", type: "f32" },
      dO: { storage: "read", type: "f32" },
      dK: { storage: "read_write", type: "f32" },
      dV: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      ...modifierUniformFields(mod),
    },
    grid: tiledGrid({
      x: { uniform: "seq_len", tileSize: BC_BW },
      y: "num_heads",
      z: "batch_size",
    }),

    kernel(ctx) {
      const tidx = ctx.localIndex();
      const kvBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const kvRow = kvBlock.mul(ctx.u32(BC_BW)).add(tidx);
      const valid = kvRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(N);
      const rowBase = bhOff.add(kvRow.mul(Dim));

      const K = ctx.tileLoad(
        "K",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const V = ctx.tileLoad(
        "V",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const dkAcc = ctx.zeros(1, D);
      const dvAcc = ctx.zeros(1, D);

      const lTile = ctx.sharedArray("L_tile", BQ_BW, "f32");
      const dTile = ctx.sharedArray("D_tile", BQ_BW, "f32");

      const numQTiles = N.add(ctx.u32(BQ_BW - 1)).div(ctx.u32(BQ_BW));

      ctx.forRange(ctx.u32(0), numQTiles, (qt) => {
        const qStart = qt.mul(ctx.u32(BQ_BW));

        // Causal tile-skip: Q-tiles entirely above the diagonal contribute
        // nothing — baked structurally when a causal maskMod is present (the
        // affine-mask → loop-bound rule's precedent; window bounds land with
        // #64 iii). Workgroup-uniform (qStart/kvBlock only).
        const skipTile = hasCausalMask(mod)
          ? qStart.add(ctx.u32(BQ_BW - 1)).lt(kvBlock.mul(ctx.u32(BC_BW)))
          : undefined;

        const emitTileBody = () => {
          const offsR = ctx.arange(qStart, BQ_BW);
          const offsD = ctx.arange(ctx.u32(0), D);
          const tilePtr = ctx.tilePtr(
            bhOff,
            offsR.outer(Dim),
            offsD.inner(ctx.u32(1)),
          );
          const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
          const QTile = ctx.load2D("Q", tilePtr, tileMask);
          const dOTile = ctx.load2D("dO", tilePtr, tileMask);

          ctx.ifThen(tidx.lt(ctx.u32(BQ_BW)), () => {
            const qi = qStart.add(tidx);
            const inBounds = qi.lt(N);
            const lIdx = bhOffL.add(qi);
            lTile.write(
              tidx,
              inBounds.select(ctx.load("L_buf", lIdx), ctx.f32(0)),
            );
            dTile.write(
              tidx,
              inBounds.select(ctx.load("D_buf", lIdx), ctx.f32(0)),
            );
          });

          ctx.barrier();

          // Fused single-loop: compute score, p, ds per Q-row, accumulate dK/dV inline
          ctx.range(0, BQ_BW, (j) => {
            const qi = qStart.add(j);
            const seamArgs = {
              qIdx: qi,
              kvIdx: kvRow,
              head: hIdx,
              batch: bIdx,
            };
            const isActive = ctx.applySeam(
              "attn_mask",
              valid.and(qi.lt(N)),
              seamArgs,
            );
            const s = ctx.dotRow(K, QTile, j).mul(scale);
            const sMod = ctx.applySeam("attn_score", s, seamArgs);
            const p = isActive.select(
              sMod.sub(lTile.read(j)).exp(),
              ctx.f32(0),
            );
            const dov = ctx.dotRow(V, dOTile, j);
            // "attn_dscore" applies BEFORE the trailing d(raw)/d(QK) scale
            // factor — the chain multiplies d(modded)/d(raw) into dS first.
            const ds = ctx
              .applySeam("attn_dscore", p.mul(dov.sub(dTile.read(j))), {
                ...seamArgs,
                raw: s,
                modded: sMod,
              })
              .mul(scale);
            ctx.accumRow(dkAcc, ds, QTile, j);
            ctx.accumRow(dvAcc, p, dOTile, j);
          });

          ctx.barrier();
        };
        if (skipTile) ctx.ifThen(skipTile.not(), emitTileBody);
        else emitTileBody();
      });

      ctx.ifThen(valid, () => {
        ctx.tileStore("dK", dkAcc, { base: rowBase, stride: ctx.u32(1) });
        ctx.tileStore("dV", dvAcc, { base: rowBase, stride: ctx.u32(1) });
      });
    },
  };
}

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
      () => makeForwardAttentionSpec(headDim, norm.mod),
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
      () => makeDPrecomputeSpec(headDim),
      headDim,
      undefined, // D-precompute has no score/mask sites — one template
      cfg,
      [batchSize * numHeads * seqLen, 1, 1],
    ),
    dqPlan: planAttentionStep(
      `bwdDQ:${headDim}`,
      "faBwdDQ",
      () => makeBackwardDQSpec(headDim, norm.mod),
      headDim,
      norm.mod,
      cfg,
      [Math.ceil(seqLen / BR), numHeads, batchSize],
    ),
    dkvPlan: planAttentionStep(
      `bwdDKV:${headDim}`,
      "faBwdDKV",
      () => makeBackwardDKVSpec(headDim, norm.mod),
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
    () => makeForwardAttentionSpec(headDim, norm.mod),
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
    () => makeDPrecomputeSpec(headDim),
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
    () => makeBackwardDQSpec(headDim, norm.mod),
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
    () => makeBackwardDKVSpec(headDim, norm.mod),
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
