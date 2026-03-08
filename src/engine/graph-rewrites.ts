/**
 * Graph Rewrite Passes
 *
 * Simplification passes run before pattern detection in the graph compiler.
 * Each pass identifies and bypasses redundant nodes, reducing the graph
 * before fusion and epilogue detection.
 */

import type { DType } from "../backend/types";
import type { LazyIRNode, LazyRef } from "./lazy-types";
import { isViewOp } from "./lowered-plan";

// ============================================================================
// Types
// ============================================================================

export interface RewriteContext {
  planNodes: LazyIRNode[];
  consumers: Map<number, LazyIRNode[]>;
  consumerCount: Map<number, number>;
}

// ============================================================================
// Rewrite Passes
// ============================================================================

/** Eliminate identity casts: cast(x, dtype=x.dtype) → bypass. */
export function eliminateIdentityCasts(
  ctx: RewriteContext,
  bypassed: Set<number>,
): void {
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
  }
}

/**
 * Eliminate redundant contiguous: contiguous(x) → x when x always produces
 * contiguous output (all compute ops do; only view ops can be non-contiguous).
 */
export function eliminateRedundantContiguous(
  ctx: RewriteContext,
  bypassed: Set<number>,
): void {
  for (const node of ctx.planNodes) {
    if (node.op !== "contiguous" || node.result) continue;
    const inputRef = node.inputs[0];
    if (!inputRef || inputRef.kind !== "pending") continue;
    if (!isViewOp(inputRef.node.op)) {
      redirectConsumers(node, inputRef, ctx);
      bypassed.add(node.id);
    }
  }
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
): void {
  for (const node of ctx.planNodes) {
    if (node.result || node.inputs.length !== 2) continue;
    const rule = IDENTITY_RULES[node.op];
    if (!rule) continue;

    const val1 = tryGetScalarValue(node.inputs[1]);
    if (val1 === rule.value) {
      redirectConsumers(node, node.inputs[0], ctx);
      bypassed.add(node.id);
    } else if (rule.commutative) {
      const val0 = tryGetScalarValue(node.inputs[0]);
      if (val0 === rule.value) {
        redirectConsumers(node, node.inputs[1], ctx);
        bypassed.add(node.id);
      }
    }
  }
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
