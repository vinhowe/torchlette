/**
 * Rewrite engine: apply a set of Rules to a plan, iterating to fixed point.
 *
 * Each rule has a Pattern + RewriteFn. The engine walks the plan forward,
 * tries each rule at each node, and applies the first match. Fixed-point
 * iteration lets rules chain: one rule's output can trigger another rule.
 *
 * The engine is pure-ish: it mutates `plan` and the consumer maps, but
 * doesn't touch any module-level state except via `createLazyIRNode`
 * (for fresh node IDs).
 *
 * # Rewrite output contract
 *
 * A rule's `rewrite` function returns a `LazyRef` that replaces the
 * matched root. There are two supported forms:
 *
 * 1. **New-node replacement** (usual case): return a ref to a freshly-
 *    created node (via `ctx.createNode` / `ctx.pendingRef`). The engine
 *    splices the new nodes into the plan before the matched root and
 *    rewires every downstream consumer to reference the new ref.
 *
 * 2. **In-place mutation**: return a ref to the matched root itself,
 *    after mutating its `op`, `inputs`, and/or `payload` in place. This
 *    preserves the node's identity (its id). Use this when an external
 *    system — notably torchlette's pending-RuntimeTensor registry —
 *    holds references by node id and shouldn't need to be re-pointed.
 *    The engine detects this case via `replacement.node.id === root.id`
 *    and transfers consumer-map membership from the root's pre-rewrite
 *    inputs to its new inputs. The rule is responsible for:
 *      - Ensuring the mutated node's shape/dtype still matches what
 *        downstream consumers expect.
 *      - Ensuring the mutated node's pattern no longer matches the
 *        rule that produced it (to avoid infinite re-matching).
 *      - Calling `ctx.markDead` on any upstream nodes whose outputs
 *        are orphaned by the mutation but can't be auto-detected as
 *        dead (e.g., nodes with pending RuntimeTensors).
 *
 * Either form may create helper nodes via `ctx.createNode`; those are
 * spliced into the plan regardless of which form the rewrite takes.
 */
import type { LazyIRNode, LazyRef } from "../../graph/types";
import { match } from "./matcher";
import type { Bindings, Pattern } from "./pattern";
import {
  applySubstitution,
  type ConsumerMaps,
  makeRewriteContext,
  type RewriteFn,
} from "./substitute";

// ============================================================================
// Types
// ============================================================================

export interface Rule {
  readonly name: string;
  readonly pattern: Pattern;
  readonly rewrite: RewriteFn;
  /** Cross-capture predicate that runs AFTER a successful match. Returning
   *  false rejects the match. Receives the matched root node for access
   *  to its payload/shape/dtype. */
  readonly check?: (bindings: Bindings, root: LazyIRNode) => boolean;
}

export interface ApplyStats {
  /** Total rewrites applied. */
  applied: number;
  /** Per-rule application counts. */
  byRule: Map<string, number>;
  /** Number of fixed-point passes that produced changes. */
  passes: number;
}

// ============================================================================
// Engine
// ============================================================================

/** Apply `rules` to `plan`, iterating to fixed point. Mutates `plan` and
 *  `maps` in place.
 *  - `replaced`: ids of nodes whose outputs were rewired away.
 *  - `killed`: nodes explicitly marked dead by a rule (via ctx.markDead).
 *    Callers can physically remove these from plan.nodes. */
export function applyRules(
  plan: LazyIRNode[],
  rules: readonly Rule[],
  maps: ConsumerMaps,
  options?: {
    maxPasses?: number;
    replaced?: Set<number>;
    killed?: Set<LazyIRNode>;
  },
): ApplyStats {
  const maxPasses = options?.maxPasses ?? 10;
  const replaced = options?.replaced ?? new Set<number>();
  const killed = options?.killed ?? new Set<LazyIRNode>();
  const stats: ApplyStats = {
    applied: 0,
    byRule: new Map(),
    passes: 0,
  };

  for (let pass = 0; pass < maxPasses; pass++) {
    const before = stats.applied;

    // Walk FORWARD so producers are processed before consumers. A rewrite
    // at position i rewires i's consumers (at higher indices) to use the
    // replacement ref — when we later reach those consumers, their inputs
    // already reflect upstream rewrites, and duplicate matches along a
    // chain are avoided.
    //
    // When a rewrite inserts N new nodes at position i, the matched root
    // shifts to i+N. We leave i unchanged so the next iteration visits the
    // first inserted node; new nodes may themselves trigger rewrites.
    for (let i = 0; i < plan.length; i++) {
      const node = plan[i];
      // Skip nodes whose output has already been rewired — they're orphans
      // in the semantic graph and should not be re-matched.
      if (replaced.has(node.id)) continue;
      const rootRef: LazyRef = { kind: "pending", node };

      for (const rule of rules) {
        const bindings = match(rule.pattern, rootRef);
        if (!bindings) continue;
        if (rule.check && !rule.check(bindings, node)) continue;

        // Snapshot the matched root's inputs BEFORE calling the rewrite.
        // In-place mutation rules change node.inputs; we need the pre-rewrite
        // inputs to correctly decrement consumer counts on the old producers.
        const preInputs = node.inputs.slice();

        // Build replacement nodes + ref.
        const { ctx, newNodes, killedNodes } = makeRewriteContext();
        const replacement = rule.rewrite(bindings, ctx, node);
        for (const k of killedNodes) killed.add(k);

        // In-place mutation case: a rule may MUTATE the matched root in
        // place (change its op/inputs) and return a ref to it. applySubstitution
        // handles that specially. To avoid infinite rematching, the mutated
        // node's new op/shape must no longer match the rule's pattern.
        applySubstitution(plan, node, replacement, newNodes, maps, preInputs);
        replaced.add(node.id);
        stats.applied++;
        stats.byRule.set(
          rule.name,
          (stats.byRule.get(rule.name) ?? 0) + 1,
        );
        // After a successful rewrite, don't try more rules on the now-orphan
        // node. Break out and let the next iteration handle the new nodes.
        break;
      }
    }

    stats.passes++;
    if (stats.applied === before) break; // fixed point
  }

  return stats;
}
