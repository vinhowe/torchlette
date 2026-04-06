/**
 * Substitution: given a matched root and a replacement ref, rewire the plan
 * so every consumer of the root now uses the replacement. New nodes (usually
 * created inside the rewrite function) are spliced into the plan just before
 * the matched root, so their outputs are produced before any downstream
 * consumer executes.
 *
 * The substitution itself does NOT bypass or delete the old chain — it just
 * drops the consumer links. Dead-code elimination (a separate pass) removes
 * orphaned nodes afterward, decrementing consumer counts along the way.
 */
import { createLazyIRNode } from "../../graph/node-factory";
import type {
  DeviceKind,
  DType,
} from "../../backend/types";
import type {
  LazyIRNode,
  LazyOpCode,
  LazyRef,
} from "../../graph/types";
import type { Bindings } from "./pattern";

// ============================================================================
// Rewrite context
// ============================================================================

/** Passed to rewrite functions. Tracks new nodes for the substitution engine. */
export interface RewriteContext {
  /** Create a new LazyIR node; the substitution engine will insert it. */
  createNode(
    op: LazyOpCode,
    inputs: LazyRef[],
    shape: number[],
    dtype: DType,
    device: DeviceKind,
    payload?: unknown,
  ): LazyIRNode;
  /** Shorthand: wrap a node as a pending ref. */
  pendingRef(node: LazyIRNode, outputIndex?: number): LazyRef;
  /** Mark a node as explicitly dead — the caller knows its output is no
   *  longer needed. Used by in-place mutation rules to mark the upstream
   *  chain (e.g., the original matmul/transpose nodes) as removable. */
  markDead(node: LazyIRNode): void;
}

/** A rewrite receives bindings from the matcher and returns the ref that
 *  replaces the matched root. */
export type RewriteFn = (
  bindings: Bindings,
  ctx: RewriteContext,
  root: import("../../graph/types").LazyIRNode,
) => LazyRef;

// ============================================================================
// Builder-context (tracks new nodes)
// ============================================================================

/** Internal: builds a RewriteContext that records all created nodes. */
export function makeRewriteContext(): {
  ctx: RewriteContext;
  newNodes: LazyIRNode[];
  killedNodes: Set<LazyIRNode>;
} {
  const newNodes: LazyIRNode[] = [];
  const killedNodes = new Set<LazyIRNode>();
  const ctx: RewriteContext = {
    createNode(op, inputs, shape, dtype, device, payload) {
      const node = createLazyIRNode(op, inputs, shape, dtype, device, payload);
      newNodes.push(node);
      return node;
    },
    pendingRef(node, outputIndex) {
      return outputIndex !== undefined
        ? { kind: "pending", node, outputIndex }
        : { kind: "pending", node };
    },
    markDead(node) {
      killedNodes.add(node);
    },
  };
  return { ctx, newNodes, killedNodes };
}

// ============================================================================
// Consumer map maintenance
// ============================================================================

export interface ConsumerMaps {
  consumers: Map<number, LazyIRNode[]>;
  consumerCount: Map<number, number>;
}

/** Record that `consumer` is now a consumer of the node/ref `producerRef`. */
function addConsumer(
  maps: ConsumerMaps,
  producerRef: LazyRef,
  consumer: LazyIRNode,
): void {
  if (producerRef.kind !== "pending") return;
  const id = producerRef.node.id;
  const list = maps.consumers.get(id) ?? [];
  list.push(consumer);
  maps.consumers.set(id, list);
  maps.consumerCount.set(id, (maps.consumerCount.get(id) ?? 0) + 1);
}

// ============================================================================
// Apply substitution
// ============================================================================

/** Apply one substitution to a plan. Mutates the plan, consumers, and
 *  consumerCount in place.
 *
 *  - `newNodes` are inserted into `plan` immediately before `matchedRoot`.
 *  - Every input ref of every new node is registered with the consumer
 *    maps so downstream passes (e.g., DCE) see correct consumer counts.
 *  - Every consumer of `matchedRoot` is rewired to use `replacement`.
 *  - `matchedRoot`'s consumer list is cleared (it's now orphan).
 *
 *  Does NOT:
 *  - Remove `matchedRoot` or its upstream chain from the plan (DCE's job).
 *  - Add `matchedRoot.id` to any `bypassed` set (DCE's job).
 *
 *  Requirements:
 *  - `matchedRoot` must already be in `plan`.
 *  - `replacement` must not be the same ref as the matched root output
 *    (would cause an infinite loop).
 */
export function applySubstitution(
  plan: LazyIRNode[],
  matchedRoot: LazyIRNode,
  replacement: LazyRef,
  newNodes: readonly LazyIRNode[],
  maps: ConsumerMaps,
  /** Snapshot of matchedRoot.inputs BEFORE the rewrite fn ran. Needed for
   *  the in-place mutation case so we can decrement consumer counts on
   *  the old producers. If omitted, matchedRoot.inputs is used. */
  preInputs?: readonly LazyRef[],
): void {
  // 1. Find insertion point for new nodes (before matchedRoot).
  const rootIdx = plan.indexOf(matchedRoot);
  if (rootIdx < 0) {
    throw new Error(
      `applySubstitution: matchedRoot #${matchedRoot.id} not in plan`,
    );
  }

  // 2. Splice new nodes into plan.
  if (newNodes.length > 0) {
    plan.splice(rootIdx, 0, ...newNodes);
  }

  // 3. Register consumer relationships for each new node's inputs.
  for (const node of newNodes) {
    for (const inputRef of node.inputs) {
      addConsumer(maps, inputRef, node);
    }
  }

  // In-place mutation case: replacement refers back to matchedRoot.
  // The rewrite has mutated matchedRoot's op/inputs in place. We need
  // to transfer matchedRoot's consumer membership from its OLD inputs
  // to its NEW inputs.
  if (
    replacement.kind === "pending" &&
    replacement.node.id === matchedRoot.id
  ) {
    const oldInputs = preInputs ?? matchedRoot.inputs;
    // Decrement consumer counts on OLD inputs (matchedRoot is no longer
    // one of their consumers).
    for (const inputRef of oldInputs) {
      if (inputRef.kind !== "pending") continue;
      const id = inputRef.node.id;
      const list = maps.consumers.get(id);
      if (!list) continue;
      const i = list.findIndex((n) => n === matchedRoot);
      if (i >= 0) {
        list.splice(i, 1);
        maps.consumers.set(id, list);
        maps.consumerCount.set(id, list.length);
      }
    }
    // Add matchedRoot as consumer of its NEW inputs.
    for (const inputRef of matchedRoot.inputs) {
      if (inputRef.kind !== "pending") continue;
      const id = inputRef.node.id;
      const list = maps.consumers.get(id) ?? [];
      if (!list.includes(matchedRoot)) {
        list.push(matchedRoot);
        maps.consumers.set(id, list);
        maps.consumerCount.set(id, list.length);
      }
    }
    return;
  }

  // 4. Rewire consumers of matchedRoot → replacement.
  const rootConsumers = maps.consumers.get(matchedRoot.id) ?? [];
  for (const consumer of rootConsumers) {
    for (let i = 0; i < consumer.inputs.length; i++) {
      const ref = consumer.inputs[i];
      if (ref.kind === "pending" && ref.node.id === matchedRoot.id) {
        consumer.inputs[i] = replacement;
      }
    }
  }

  // 5. Update consumer maps: move rootConsumers → replacement's producer.
  if (replacement.kind === "pending") {
    const targetId = replacement.node.id;
    const existing = maps.consumers.get(targetId) ?? [];
    for (const c of rootConsumers) {
      if (!existing.includes(c)) existing.push(c);
    }
    maps.consumers.set(targetId, existing);
    maps.consumerCount.set(targetId, existing.length);
  }
  maps.consumers.set(matchedRoot.id, []);
  maps.consumerCount.set(matchedRoot.id, 0);
}
