/**
 * Plan-level rewrite: apply DSL rules directly to an ExecutionPlan's node
 * list BEFORE the template cache sees it. Node-insertion rewrites require
 * this — if we ran them inside compileTemplate, the cached finalPerm would
 * reference inserted-node positions that don't exist in the fresh plan.nodes
 * handed in on cache hit.
 *
 * Trade-off: rules run every step instead of once per template. For the
 * small rule set we currently have (single-digit rewrites per plan) this
 * is negligible — a few dozen microseconds of pattern matching.
 */
import type { ExecutionPlan, LazyIRNode } from "../../graph/types";
import { applyRules, type Rule } from "./engine";
import type { ConsumerMaps } from "./substitute";

/** Apply `rules` to `plan.nodes` in place. Returns number of rewrites.
 *  `externalNodeIds` is the set of nodes whose outputs are externally
 *  referenced (by pending RuntimeTensors); they're protected from removal. */
export function rewritePlan(
  plan: ExecutionPlan,
  rules: readonly Rule[],
  externalNodeIds?: Set<number>,
): number {
  if (rules.length === 0) return 0;

  // Build consumer maps from the current plan.
  const consumers = new Map<number, LazyIRNode[]>();
  const consumerCount = new Map<number, number>();
  for (const node of plan.nodes) {
    for (const ref of node.inputs) {
      if (ref.kind === "pending") {
        const id = ref.node.id;
        const list = consumers.get(id) ?? [];
        list.push(node);
        consumers.set(id, list);
        consumerCount.set(id, (consumerCount.get(id) ?? 0) + 1);
      }
    }
  }

  const maps: ConsumerMaps = { consumers, consumerCount };
  const killed = new Set<LazyIRNode>();
  const stats = applyRules(plan.nodes, rules, maps, { killed });
  if (stats.applied === 0) return 0;

  // Compact: physically remove explicitly-killed nodes and any transitively
  // dead nodes (0 in-plan consumers after removal). Killed nodes may still
  // be in externalNodeIds (their pending tensors were autograd saved-for-
  // backward artifacts); trust the rule's assertion that they're no longer
  // needed.
  const dead = new Set<number>();
  const worklist: LazyIRNode[] = [];
  for (const n of killed) {
    dead.add(n.id);
    worklist.push(n);
  }
  while (worklist.length > 0) {
    const node = worklist.pop()!;
    for (const ref of node.inputs) {
      if (ref.kind !== "pending") continue;
      const id = ref.node.id;
      const c = (consumerCount.get(id) ?? 0) - 1;
      consumerCount.set(id, c);
      if (c > 0) continue;
      if (dead.has(id)) continue;
      if (externalNodeIds?.has(id)) continue;
      dead.add(id);
      worklist.push(ref.node);
    }
  }
  if (dead.size > 0) {
    const live = plan.nodes.filter((n) => !dead.has(n.id));
    plan.nodes.length = 0;
    plan.nodes.push(...live);
  }

  if (process.env.TORCHLETTE_LOG_DSL) {
    const parts: string[] = [];
    for (const [name, count] of stats.byRule) parts.push(`${name}=${count}`);
    console.log(
      `[dsl-rewrite] nodes=${plan.nodes.length} ${parts.join(" ")} removed=${dead.size}`,
    );
  }
  return stats.applied;
}
