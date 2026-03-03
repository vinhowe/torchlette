/**
 * Graph-Level Rewrite Passes
 *
 * Simplifies the LazyIRNode graph before pattern detection.
 * Each pass identifies nodes that can be bypassed (identity ops)
 * and redirects their consumers to use the original input instead.
 *
 * Passes run during the cache-miss path of analyzeGraph(), after
 * the consumer/producer maps are built but before pattern detection.
 *
 * SAFETY: These passes mutate LazyRef objects in consumer nodes.
 * This is safe because:
 * - LazyIRNodes are created fresh each step
 * - The plan fingerprint is computed before rewrites
 * - On cache hit, the template reconstructs from position indices
 */

import type { DType } from "../backend/types";
import type { LazyIRNode, LazyRef } from "./lazy-types";
import { isViewOp } from "./lowered-plan";

/**
 * Context provided to rewrite passes.
 */
interface RewriteContext {
  /** Plan nodes in current order. */
  planNodes: LazyIRNode[];
  /** Map from node ID → consumer nodes. */
  consumers: Map<number, LazyIRNode[]>;
  /** Map from node ID → number of consumers. */
  consumerCount: Map<number, number>;
}

/**
 * Run all rewrite passes on the plan graph.
 * Returns the set of node IDs that were bypassed (should be excluded from execution).
 */
export function runRewritePasses(ctx: RewriteContext): Set<number> {
  const bypassed = new Set<number>();
  eliminateIdentityCasts(ctx, bypassed);
  eliminateRedundantContiguous(ctx, bypassed);
  eliminateAlgebraicIdentities(ctx, bypassed);
  return bypassed;
}

/**
 * Eliminate identity casts: cast(x, dtype=x.dtype) → bypass.
 *
 * When a cast node's target dtype matches its input's dtype, the cast
 * is a no-op. Redirect all consumers to use the cast's input directly.
 * This can enable fusion chains that were previously broken by the cast.
 */
function eliminateIdentityCasts(
  ctx: RewriteContext,
  bypassed: Set<number>,
): void {
  for (const node of ctx.planNodes) {
    if (node.op !== "cast") continue;
    if (node.result) continue; // Already computed

    const payload = node.payload as { dtype?: DType } | undefined;
    if (!payload?.dtype) continue;

    // Check if input dtype matches output dtype
    const inputRef = node.inputs[0];
    if (!inputRef) continue;

    let inputDtype: DType | undefined;
    if (inputRef.kind === "pending") {
      inputDtype = inputRef.node.dtype;
    } else if (inputRef.kind === "materialized") {
      inputDtype = inputRef.storage.backendTensor.dtype ?? "f32";
    }

    if (inputDtype !== payload.dtype) continue;

    // This is an identity cast — redirect consumers
    redirectConsumers(node, inputRef, ctx);
    bypassed.add(node.id);
  }
}

/**
 * Eliminate redundant contiguous: contiguous(x) → x when x always produces contiguous output.
 *
 * Only view ops (reshape, transpose, permute, expand, narrow) can produce
 * non-contiguous output. All compute ops (matmul, elementwise, reductions,
 * data sources) always produce contiguous tensors, so contiguous after them
 * is a no-op.
 */
function eliminateRedundantContiguous(
  ctx: RewriteContext,
  bypassed: Set<number>,
): void {
  for (const node of ctx.planNodes) {
    if (node.op !== "contiguous") continue;
    if (node.result) continue;

    const inputRef = node.inputs[0];
    if (!inputRef || inputRef.kind !== "pending") continue;

    const inputNode = inputRef.node;
    // If input op always produces contiguous output, this contiguous is redundant
    if (!isViewOp(inputNode.op)) {
      redirectConsumers(node, inputRef, ctx);
      bypassed.add(node.id);
    }
  }
}

/**
 * Redirect all consumers of `node` to use `replacementRef` instead.
 * Mutates LazyRef objects in consumer nodes' inputs arrays.
 */
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

  // Update consumer counts: consumers of node now become consumers of the input
  if (replacementRef.kind === "pending") {
    const inputId = replacementRef.node.id;
    const count = ctx.consumerCount.get(inputId) ?? 0;
    const nodeConsumerCount = ctx.consumerCount.get(node.id) ?? 0;
    // The input gains the node's consumers but loses the node as a consumer
    ctx.consumerCount.set(inputId, count + nodeConsumerCount - 1);
    // Also update the consumers list
    const inputConsumers = ctx.consumers.get(inputId) ?? [];
    // Remove node from input's consumers
    const filtered = inputConsumers.filter((c) => c.id !== node.id);
    // Add node's consumers to input's consumers
    for (const consumer of nodeConsumers) {
      if (!filtered.some((c) => c.id === consumer.id)) {
        filtered.push(consumer);
      }
    }
    ctx.consumers.set(inputId, filtered);
  }

  // Clear node's consumer tracking
  ctx.consumerCount.set(node.id, 0);
  ctx.consumers.set(node.id, []);
}

/**
 * Try to extract a constant scalar value from a LazyRef.
 * Returns the value if the ref points to a `full` node with a single element,
 * or null otherwise.
 */
function tryGetScalarValue(ref: LazyRef): number | null {
  if (ref.kind !== "pending") return null;
  const node = ref.node;
  if (node.op !== "full") return null;
  // full node must produce a scalar (0-d) or single-element tensor
  const totalElements = node.shape.reduce((a, b) => a * b, 1);
  if (totalElements !== 1) return null;
  const payload = node.payload as { fillValue: number } | undefined;
  if (!payload || typeof payload.fillValue !== "number") return null;
  return payload.fillValue;
}

/**
 * Eliminate algebraic identities:
 *   mul(x, 1) → x,  mul(1, x) → x
 *   add(x, 0) → x,  add(0, x) → x
 *   div(x, 1) → x
 *   sub(x, 0) → x
 *
 * These appear in AMP scaling paths, optimizer computations, and
 * autograd where scalar constants flow through the graph.
 */
function eliminateAlgebraicIdentities(
  ctx: RewriteContext,
  bypassed: Set<number>,
): void {
  for (const node of ctx.planNodes) {
    if (node.result) continue;
    if (node.inputs.length !== 2) continue;

    const ref0 = node.inputs[0];
    const ref1 = node.inputs[1];

    if (node.op === "mul") {
      // mul(x, 1) → x
      const val1 = tryGetScalarValue(ref1);
      if (val1 === 1) {
        redirectConsumers(node, ref0, ctx);
        bypassed.add(node.id);
        continue;
      }
      // mul(1, x) → x
      const val0 = tryGetScalarValue(ref0);
      if (val0 === 1) {
        redirectConsumers(node, ref1, ctx);
        bypassed.add(node.id);
      }
    } else if (node.op === "add") {
      // add(x, 0) → x
      const val1 = tryGetScalarValue(ref1);
      if (val1 === 0) {
        redirectConsumers(node, ref0, ctx);
        bypassed.add(node.id);
        continue;
      }
      // add(0, x) → x
      const val0 = tryGetScalarValue(ref0);
      if (val0 === 0) {
        redirectConsumers(node, ref1, ctx);
        bypassed.add(node.id);
      }
    } else if (node.op === "sub") {
      // sub(x, 0) → x
      const val1 = tryGetScalarValue(ref1);
      if (val1 === 0) {
        redirectConsumers(node, ref0, ctx);
        bypassed.add(node.id);
      }
    } else if (node.op === "div") {
      // div(x, 1) → x
      const val1 = tryGetScalarValue(ref1);
      if (val1 === 1) {
        redirectConsumers(node, ref0, ctx);
        bypassed.add(node.id);
      }
    }
  }
}
