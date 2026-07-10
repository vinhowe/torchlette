/**
 * Synthetic plan corpus for the fusion-decision differential (islands I2).
 *
 * Each case is a deterministic LazyIRNode plan covering one branch of the
 * detector's decision space. The committed fixture
 * (test/fixtures/fusion-decisions.json) pins the decisions; the policy
 * re-expression (I2a) must reproduce them byte-identically (null test), and
 * the gap-spanning extension (I2b) regenerates the fixture INTENTIONALLY with
 * the decision delta visible in the diff.
 */
import {
  detectFusionGroups,
  segmentPlanForExecution,
} from "../../src/compiler/fusion-detect";
import {
  createPendingRef,
  createScalarRef,
  LazyIRNode,
  type LazyRef,
} from "../../src/graph/types";

export interface CorpusCase {
  name: string;
  nodes: LazyIRNode[];
  options?: {
    maxStorageBuffers?: number;
    enableMultiOutput?: boolean;
    excludedIds?: Set<number>;
  };
  externalNodeIds?: Set<number>;
}

class PlanBuilder {
  nodes: LazyIRNode[] = [];
  private nextId = 1;

  n(
    op: string,
    inputs: (LazyIRNode | LazyRef)[],
    shape: number[],
    dtype = "f32",
  ): LazyIRNode {
    const refs: LazyRef[] = inputs.map((i) =>
      i instanceof LazyIRNode ? createPendingRef(i) : i,
    );
    const node = new LazyIRNode(
      this.nextId++,
      op as never,
      refs,
      shape,
      dtype as never,
      "webgpu",
    );
    this.nodes.push(node);
    return node;
  }
}

export function buildCorpus(): CorpusCase[] {
  const cases: CorpusCase[] = [];
  const OPTS = { maxStorageBuffers: 8, enableMultiOutput: true };

  // 1. Simple consecutive chain — one fused island.
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8, 4]);
    const a = p.n("relu", [x], [8, 4]);
    const b = p.n("exp", [a], [8, 4]);
    p.n("neg", [b], [8, 4]);
    cases.push({ name: "consecutive-chain", nodes: p.nodes, options: OPTS });
  }

  // 2. THE STRANDED CASE — chain broken by a group-DEPENDENT reshape, then
  //    more fusibles continuing DIRECTLY from the chain (independent of the
  //    reshape). Today: flush → two groups. I2b: one gapped island.
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8, 4]);
    const a = p.n("relu", [x], [8, 4]);
    const v = p.n("reshape", [a], [32]); // depends on the chain → flush today
    const c = p.n("exp", [a], [8, 4]); // continues from the CHAIN, not the view
    p.n("neg", [c], [8, 4]);
    p.n("sum", [v], [], "f32"); // keeps the reshape externally consumed
    cases.push({ name: "stranded-dependent-gap", nodes: p.nodes, options: OPTS });
  }

  // 3. Independent non-fusible gap — passes through, chain unbroken.
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8, 4]);
    const y = p.n("leafInput", [], [4, 4]);
    const a = p.n("relu", [x], [8, 4]);
    p.n("matmul", [y, y], [4, 4]); // independent of the chain
    const b = p.n("exp", [a], [8, 4]);
    p.n("neg", [b], [8, 4]);
    cases.push({ name: "independent-gap", nodes: p.nodes, options: OPTS });
  }

  // 4. Two independent components inside one consecutive run — union-find split.
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8, 4]);
    const y = p.n("leafInput", [], [8, 4]);
    const a1 = p.n("relu", [x], [8, 4]);
    const b1 = p.n("exp", [y], [8, 4]); // separate component, interleaved
    const a2 = p.n("neg", [a1], [8, 4]);
    p.n("sqrt", [b1], [8, 4]);
    p.n("abs", [a2], [8, 4]);
    cases.push({ name: "two-components", nodes: p.nodes, options: OPTS });
  }

  // 5. Externally-referenced intermediate, SAME shape — multi-output promotion.
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8, 4]);
    const a = p.n("relu", [x], [8, 4]);
    const b = p.n("exp", [a], [8, 4]);
    p.n("neg", [b], [8, 4]);
    p.n("sum", [a], [], "f32"); // external consumer of intermediate `a`
    cases.push({ name: "promoted-intermediate", nodes: p.nodes, options: OPTS });
  }

  // 6. Externally-referenced intermediate, DIFFERENT shape from primary —
  //    split path (cannot promote).
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8, 4]);
    const s = p.n("leafInput", [], [1]);
    const a = p.n("relu", [x], [8, 4]);
    const b = p.n("add", [a, s], [8, 4]);
    const c = p.n("mul", [b, b], [8, 4]);
    const d = p.n("exp", [s], [1]); // different-shape member, referenced below
    p.n("neg", [c], [8, 4]);
    p.n("sum", [d], [], "f32");
    cases.push({ name: "split-mixed-shape", nodes: p.nodes, options: OPTS });
  }

  // 7. Buffer-limit overflow — many distinct external inputs, small budget.
  {
    const p = new PlanBuilder();
    const leaves = Array.from({ length: 6 }, () => p.n("leafInput", [], [4]));
    let acc = p.n("add", [leaves[0], leaves[1]], [4]);
    for (let i = 2; i < 6; i++) acc = p.n("add", [acc, leaves[i]], [4]);
    p.n("neg", [acc], [4]);
    cases.push({
      name: "buffer-limit-split",
      nodes: p.nodes,
      options: { maxStorageBuffers: 4, enableMultiOutput: true },
    });
  }

  // 8. Same-shape singletons separated by barriers — global singleton batching.
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8]);
    const y = p.n("leafInput", [], [8]);
    const a = p.n("relu", [x], [8]);
    p.n("sum", [a], [], "f32"); // barrier
    const b = p.n("exp", [y], [8]);
    p.n("sum", [b], [], "f32"); // barrier
    cases.push({ name: "singleton-batching", nodes: p.nodes, options: OPTS });
  }

  // 9. Excluded (claimed) ids — the excluded fusible acts as a barrier.
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8, 4]);
    const a = p.n("relu", [x], [8, 4]);
    const b = p.n("gelu", [a], [8, 4]); // will be excluded (epilogue-claimed)
    const c = p.n("exp", [b], [8, 4]);
    p.n("neg", [c], [8, 4]);
    cases.push({
      name: "excluded-claimed",
      nodes: p.nodes,
      options: { ...OPTS, excludedIds: new Set([b.id]) },
    });
  }

  // 10. Scalar (0-d) refs — broadcast independently, no union through them.
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8]);
    const y = p.n("leafInput", [], [8]);
    const a = p.n("add", [x, createScalarRef(2, "f32")], [8]);
    const b = p.n("mul", [y, createScalarRef(3, "f32")], [8]);
    p.n("neg", [a], [8]);
    p.n("abs", [b], [8]);
    cases.push({ name: "scalar-refs", nodes: p.nodes, options: OPTS });
  }

  // 11. READINESS NO-CASE — after a group-dependent gap, a fusible node reads
  //     a producer that comes AFTER the forced-emission point. Must NOT join
  //     the earlier chain (would execute before its input exists) — in
  //     today's code AND under the I2b extension.
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8, 4]);
    const a = p.n("relu", [x], [8, 4]);
    const v = p.n("reshape", [a], [32]); // dependent gap → emission point
    const w = p.n("sum", [v], [], "f32"); // post-gap producer
    const c = p.n("add", [a, w], [8, 4]); // reads post-gap producer → NOT ready
    p.n("neg", [c], [8, 4]);
    cases.push({ name: "readiness-no-case", nodes: p.nodes, options: OPTS });
  }

  // 12. External node ids (saved-for-backward) force intermediate handling.
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8, 4]);
    const a = p.n("relu", [x], [8, 4]);
    const b = p.n("exp", [a], [8, 4]);
    p.n("neg", [b], [8, 4]);
    cases.push({
      name: "external-saved",
      nodes: p.nodes,
      options: OPTS,
      externalNodeIds: new Set([a.id]),
    });
  }

  // 13. Multi-dependent-gap chain — two dependent view gaps, fusibles after
  //     each continuing from the chain (I2b: all one island; today: three).
  {
    const p = new PlanBuilder();
    const x = p.n("leafInput", [], [8, 4]);
    const a = p.n("relu", [x], [8, 4]);
    p.n("reshape", [a], [32]);
    const b = p.n("exp", [a], [8, 4]);
    p.n("reshape", [b], [4, 8]);
    const c = p.n("sqrt", [b], [8, 4]);
    p.n("neg", [c], [8, 4]);
    cases.push({ name: "multi-dependent-gap", nodes: p.nodes, options: OPTS });
  }

  return cases;
}

/** Serialize the decisions for one corpus case (positions, not ids). */
export function decisionsFor(c: CorpusCase): unknown {
  const idToPos = new Map<number, number>();
  c.nodes.forEach((n, i) => idToPos.set(n.id, i));
  const pos = (n: LazyIRNode) => idToPos.get(n.id) as number;

  const det = detectFusionGroups(c.nodes, c.externalNodeIds, c.options);
  const groups = det.groups.map((g) => ({
    planIndices: [...g.planIndices],
    output: pos(g.outputNode),
    additional: (g.additionalOutputNodes ?? []).map(pos),
    needed: (g.neededIntermediates ?? []).map(pos),
    externalInputCount: g.externalInputs.length,
  }));

  const segments = segmentPlanForExecution(
    c.nodes,
    c.externalNodeIds,
    c.options,
  ).map((s) =>
    s.kind === "fused"
      ? { kind: s.kind, positions: s.group.nodes.map(pos) }
      : { kind: s.kind, positions: s.nodes.map(pos) },
  );

  return { name: c.name, stats: det.stats, groups, segments };
}
