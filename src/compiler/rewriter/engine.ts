/**
 * Rewrite engine: apply a set of Rules to a plan, iterating to fixed point.
 *
 * Each rule has a Pattern + RewriteFn. The engine walks the plan in reverse,
 * tries each rule at each node, and applies the first match. Fixed-point
 * iteration lets rules chain: one rule's output can trigger another rule.
 *
 * The engine is pure-ish: it mutates `plan` and the consumer maps, but
 * doesn't touch any module-level state except via `createLazyIRNode`
 * (for fresh node IDs).
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
   *  false rejects the match. Useful for shape/dtype constraints that span
   *  multiple bindings. */
  readonly check?: (bindings: Bindings) => boolean;
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
 *  `maps` in place. `replaced` (optional) collects the ids of nodes whose
 *  outputs were rewired away — callers can pass the same set to DCE or
 *  other passes. */
export function applyRules(
  plan: LazyIRNode[],
  rules: readonly Rule[],
  maps: ConsumerMaps,
  options?: { maxPasses?: number; replaced?: Set<number> },
): ApplyStats {
  const maxPasses = options?.maxPasses ?? 10;
  const replaced = options?.replaced ?? new Set<number>();
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
        if (rule.check && !rule.check(bindings)) continue;

        // Build replacement nodes + ref.
        const { ctx, newNodes } = makeRewriteContext();
        const replacement = rule.rewrite(bindings, ctx);

        // Guard: replacement must not be the same as matched root.
        if (
          replacement.kind === "pending" &&
          replacement.node.id === node.id
        ) {
          // Malformed rule — would cause infinite loop. Skip.
          continue;
        }

        applySubstitution(plan, node, replacement, newNodes, maps);
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
