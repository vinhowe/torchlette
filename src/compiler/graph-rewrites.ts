/**
 * Graph Rewrite Passes
 *
 * Simplification passes run before pattern detection in the graph compiler.
 * Each pass identifies and bypasses redundant nodes, reducing the graph
 * before fusion and epilogue detection.
 *
 * The GraphPass interface formalizes each pass as a composable function.
 * Passes are run in order via runPasses(), which returns per-pass stats.
 */

import type { DType } from "../backend/types";
import { isViewOp } from "../executor/lowered-plan";
import type { LazyIRNode, LazyRef } from "../graph/types";

// ============================================================================
// Types
// ============================================================================

export interface RewriteContext {
  planNodes: LazyIRNode[];
  consumers: Map<number, LazyIRNode[]>;
  consumerCount: Map<number, number>;
}

/** A composable graph rewrite pass. */
export interface GraphPass {
  name: string;
  run(ctx: RewriteContext, bypassed: Set<number>): number;
}

// ============================================================================
// Pass Registry
// ============================================================================

/** All simplification passes in execution order. */
export const SIMPLIFICATION_PASSES: GraphPass[] = [
  { name: "identity-casts", run: eliminateIdentityCasts },
  { name: "redundant-contiguous", run: eliminateRedundantContiguous },
  { name: "algebraic-identities", run: eliminateAlgebraicIdentities },
  { name: "fuse-sum-reshape", run: fuseSumReshape },
  { name: "cse", run: eliminateCommonSubexpressions },
  { name: "dce", run: eliminateDeadCode },
];

/** Run a sequence of passes, returning per-pass elimination counts. */
export function runPasses(
  ctx: RewriteContext,
  bypassed: Set<number>,
  passes: GraphPass[],
): Map<string, number> {
  const stats = new Map<string, number>();
  for (const pass of passes) {
    stats.set(pass.name, pass.run(ctx, bypassed));
  }
  return stats;
}

// ============================================================================
// Rewrite Passes
// ============================================================================

/** Eliminate identity casts: cast(x, dtype=x.dtype) → bypass. */
export function eliminateIdentityCasts(
  ctx: RewriteContext,
  bypassed: Set<number>,
): number {
  let count = 0;
  for (const node of ctx.planNodes) {
    if (node.op !== "cast" || node.result) continue;
    const payload = node.payload as { dtype?: DType } | undefined;
    if (!payload?.dtype) continue;
    const inputRef = node.inputs[0];
    if (!inputRef) continue;

    let inputDtype: DType | undefined;
    if (inputRef.kind === "pending") inputDtype = inputRef.node.dtype;
    else if (inputRef.kind === "materialized")
      inputDtype = inputRef.storage.backendTensor.dtype ?? "f32";

    if (inputDtype !== payload.dtype) continue;
    redirectConsumers(node, inputRef, ctx);
    bypassed.add(node.id);
    count++;
  }
  return count;
}

/**
 * Eliminate redundant contiguous: contiguous(x) → x when x always produces
 * contiguous output (all compute ops do; only view ops can be non-contiguous).
 */
export function eliminateRedundantContiguous(
  ctx: RewriteContext,
  bypassed: Set<number>,
): number {
  let count = 0;
  for (const node of ctx.planNodes) {
    if (node.op !== "contiguous" || node.result) continue;
    const inputRef = node.inputs[0];
    if (!inputRef || inputRef.kind !== "pending") continue;
    if (!isViewOp(inputRef.node.op)) {
      redirectConsumers(node, inputRef, ctx);
      bypassed.add(node.id);
      count++;
    }
  }
  return count;
}

/** Try to extract a constant scalar value from a LazyRef. */
function tryGetScalarValue(ref: LazyRef): number | null {
  if (ref.kind !== "pending") return null;
  const node = ref.node;
  if (node.op !== "full") return null;
  const totalElements = node.shape.reduce((a, b) => a * b, 1);
  if (totalElements !== 1) return null;
  const payload = node.payload as { fillValue: number } | undefined;
  if (!payload || typeof payload.fillValue !== "number") return null;
  return payload.fillValue;
}

const IDENTITY_RULES: Record<string, { value: number; commutative: boolean }> =
  {
    mul: { value: 1, commutative: true },
    add: { value: 0, commutative: true },
    sub: { value: 0, commutative: false },
    div: { value: 1, commutative: false },
  };

/** Eliminate algebraic identities: mul(x,1)→x, add(x,0)→x, div(x,1)→x, sub(x,0)→x. */
export function eliminateAlgebraicIdentities(
  ctx: RewriteContext,
  bypassed: Set<number>,
): number {
  let count = 0;
  for (const node of ctx.planNodes) {
    if (node.result || node.inputs.length !== 2) continue;
    const rule = IDENTITY_RULES[node.op];
    if (!rule) continue;

    const val1 = tryGetScalarValue(node.inputs[1]);
    if (val1 === rule.value) {
      redirectConsumers(node, node.inputs[0], ctx);
      bypassed.add(node.id);
      count++;
    } else if (rule.commutative) {
      const val0 = tryGetScalarValue(node.inputs[0]);
      if (val0 === rule.value) {
        redirectConsumers(node, node.inputs[1], ctx);
        bypassed.add(node.id);
        count++;
      }
    }
  }
  return count;
}

// ============================================================================
// Sum+Reshape Fusion Pass
// ============================================================================

/**
 * Fuse sum(keepdim:true) → reshape into sum(keepdim:false).
 *
 * _sumToShape() generates sum(dims, keepdim:true) then reshape(targetShape)
 * to handle rank reduction (e.g., [1,512,768] → sum → [1,1,768] → reshape → [768]).
 * The reshape is metadata-only but still dispatches. Since the backend recomputes
 * output shape from (inputShape, dim, keepdim), changing keepdim to false and
 * updating the sum node's shape produces identical output bytes with no reshape.
 */
export function fuseSumReshape(
  ctx: RewriteContext,
  bypassed: Set<number>,
): number {
  let count = 0;
  for (const node of ctx.planNodes) {
    if (node.op !== "reshape" || node.result || bypassed.has(node.id)) continue;
    const inputRef = node.inputs[0];
    if (!inputRef || inputRef.kind !== "pending") continue;

    const sumNode = inputRef.node;
    if (sumNode.op !== "sum" && sumNode.op !== "mean") continue;
    if (bypassed.has(sumNode.id)) continue;

    const payload = sumNode.payload as
      | { dim?: number | number[] | null; keepdim?: boolean }
      | undefined;
    if (!payload?.keepdim) continue;

    // Only fuse single-consumer sums (multi-consumer would change other readers' shape)
    if ((ctx.consumerCount.get(sumNode.id) ?? 0) > 1) continue;

    // Verify same total elements (reshape just squeezes size-1 dims)
    const sumElements = sumNode.shape.reduce((a, b) => a * b, 1);
    const reshapeElements = node.shape.reduce((a, b) => a * b, 1);
    if (sumElements !== reshapeElements) continue;

    // Mutate sum node: output the reshaped shape directly
    sumNode.shape = node.shape;
    payload.keepdim = false;

    // Bypass the reshape
    redirectConsumers(node, { kind: "pending", node: sumNode }, ctx);
    bypassed.add(node.id);
    count++;
  }
  return count;
}

// ============================================================================
// CSE Pass
// ============================================================================

/** Ops that must never be CSE'd due to side effects or non-determinism. */
const NON_CSE_OPS = new Set([
  "rand",
  "randn",
  "bernoulli",
  "fusedAttentionForward",
  "fusedAttentionBackward",
  "fusedLayerNormForward",
  "fusedLayerNormBackwardGradX",
  "fusedLayerNormBackwardGradWeightBias",
  "fusedRMSNormForward",
  "fusedRMSNormBackwardGradX",
  "fusedRMSNormBackwardGradWeight",
  "fusedCrossEntropyForward",
  "fusedCrossEntropyBackward",
  "adamStep",
  "stridedScatterCopy",
  "stridedScatterAdd",
  "unscaleGrad",
]);

/** Ops whose payload is too large to include in CSE keys (e.g., full tensor data). */
const LARGE_PAYLOAD_OPS = new Set(["tensorFromArray", "zeros", "arange"]);

/**
 * Compute a structural key for a node.
 * Key format: `op|shape|dtype|ref0|ref1|...|payload_hash`
 *
 * Input refs are encoded as:
 *   - pending:   `p:nodeId` (the canonical node for already-CSE'd chains)
 *   - materialized: `m:storageId`
 *   - scalar:    `s:value:dtype`
 *
 * Nodes with no inputs and large payloads (tensorFromArray, zeros, arange)
 * are excluded from CSE entirely — they're data sources with unique identity.
 */
function structuralKey(node: LazyIRNode): string | null {
  // Data source ops with no pending inputs can't be CSE'd meaningfully
  if (node.inputs.length === 0) return null;

  let key = `${node.op}|${node.shape.join(",")}|${node.dtype}`;
  for (const ref of node.inputs) {
    if (ref.kind === "pending") {
      // outputIndex MUST be part of the key: two refs to the same multi-output
      // node but different output slots (e.g. fusedAttentionBackward dQ/dK/dV at
      // indices 0/1/2) are DIFFERENT values. Omitting it makes structurally
      // identical consumers of distinct outputs collide, so CSE merges them and
      // every consumer collapses onto a single output — a silent, correctness-
      // breaking gradient bug for any multi-output op (see tools/sdpa2-diff.ts).
      key += `|p:${ref.node.id}:${ref.outputIndex ?? 0}`;
    } else if (ref.kind === "materialized") {
      key += `|m:${ref.storage.id}`;
    } else {
      key += `|s:${ref.value}:${ref.dtype}`;
    }
  }
  if (node.payload !== undefined && !LARGE_PAYLOAD_OPS.has(node.op)) {
    // Known small payloads: cast {dtype}, full {fillValue}, reduction {dim, keepdim}
    key += `|${JSON.stringify(node.payload)}`;
  }
  return key;
}

/**
 * Eliminate common subexpressions: when two nodes have identical
 * (op, shape, dtype, inputs, payload), redirect the second's consumers
 * to the first and bypass the second.
 *
 * Non-deterministic ops (RNG) and side-output ops (adam, fused kernels)
 * are excluded.
 */
export function eliminateCommonSubexpressions(
  ctx: RewriteContext,
  bypassed: Set<number>,
): number {
  const canonical = new Map<string, LazyIRNode>();
  let count = 0;
  for (const node of ctx.planNodes) {
    if (node.result || bypassed.has(node.id) || NON_CSE_OPS.has(node.op))
      continue;
    const key = structuralKey(node);
    if (key === null) continue; // Data source ops — not eligible for CSE
    const existing = canonical.get(key);
    if (existing) {
      redirectConsumers(node, { kind: "pending", node: existing }, ctx);
      bypassed.add(node.id);
      count++;
    } else {
      canonical.set(key, node);
    }
  }
  return count;
}

// ============================================================================
// DCE Pass
// ============================================================================

/**
 * Eliminate dead code: remove nodes with zero consumers that aren't the
 * plan output or already-materialized. Iterates until fixed point since
 * removing a node may make its inputs dead.
 */
export function eliminateDeadCode(
  ctx: RewriteContext,
  bypassed: Set<number>,
): number {
  let count = 0;
  const lastNode = ctx.planNodes[ctx.planNodes.length - 1];
  let changed = true;
  while (changed) {
    changed = false;
    for (const node of ctx.planNodes) {
      if (bypassed.has(node.id) || node.result) continue;
      if (node === lastNode) continue; // Never eliminate plan output
      const consumers = ctx.consumerCount.get(node.id) ?? 0;
      if (consumers === 0) {
        bypassed.add(node.id);
        // Decrement consumer counts of this node's inputs
        for (const ref of node.inputs) {
          if (ref.kind === "pending") {
            const c = ctx.consumerCount.get(ref.node.id) ?? 0;
            if (c > 0) ctx.consumerCount.set(ref.node.id, c - 1);
          }
        }
        count++;
        changed = true;
      }
    }
  }
  return count;
}

// ============================================================================
// Helpers
// ============================================================================

/** Redirect all consumers of `node` to use `replacementRef` instead. */
function redirectConsumers(
  node: LazyIRNode,
  replacementRef: LazyRef,
  ctx: RewriteContext,
): void {
  const nodeConsumers = ctx.consumers.get(node.id);
  if (!nodeConsumers) return;

  for (const consumer of nodeConsumers) {
    for (let i = 0; i < consumer.inputs.length; i++) {
      const ref = consumer.inputs[i];
      if (ref.kind === "pending" && ref.node.id === node.id) {
        consumer.inputs[i] = replacementRef;
      }
    }
  }

  if (replacementRef.kind === "pending") {
    const inputId = replacementRef.node.id;
    const count = ctx.consumerCount.get(inputId) ?? 0;
    const nodeConsumerCount = ctx.consumerCount.get(node.id) ?? 0;
    ctx.consumerCount.set(inputId, count + nodeConsumerCount - 1);
    const inputConsumers = ctx.consumers.get(inputId) ?? [];
    const filtered = inputConsumers.filter((c) => c.id !== node.id);
    for (const consumer of nodeConsumers) {
      if (!filtered.some((c) => c.id === consumer.id)) {
        filtered.push(consumer);
      }
    }
    ctx.consumers.set(inputId, filtered);
  }

  ctx.consumerCount.set(node.id, 0);
  ctx.consumers.set(node.id, []);
}
