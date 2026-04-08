/**
 * Unified Graph Compiler
 *
 * Consolidates pattern detection systems into a single `analyzeGraph()` call
 * with priority-ordered pattern detectors:
 *
 *  1. Matmul epilogue chains       (priority 100)
 *  2. Row-program fusion           (priority 70)
 *  3. Elementwise fusion            (priority 40)
 *
 * The analysis phase runs once per structural fingerprint and produces a
 * `GraphAnalysisResult` consumed by executor-lowered.ts. Results are
 * cached in the FusionAnalysisTemplate.
 */

import type { DType } from "../backend/types";
import type { LazyIRNode } from "../graph/types";
import {
  type ExecutionSegment,
  isFusibleOp,
  reorderPlanForFusion,
  segmentPlanForExecution,
} from "./fusion-detect";
import { runPasses, SIMPLIFICATION_PASSES } from "./graph-rewrites";
import {
  detectMatmulEpilogueCore,
  type MatmulEpiloguePlan,
  type MatmulPrologueInfo,
} from "./matmul-epilogue";
import { detectRowPrograms } from "./row-program-detect";
import type { RowProgramMatch } from "./row-program-types";

// ============================================================================
// Types
// ============================================================================

/** Result of the unified graph analysis. */
interface GraphAnalysisResult {
  /** Plan nodes in final (reordered) execution order. */
  planNodes: LazyIRNode[];

  /** Execution segments (fused and sequential). */
  segments: ExecutionSegment[];

  /** Node IDs claimed by matmul epilogue chains. */
  epilogueClaimedIds: Set<number>;

  /** Node IDs claimed by matmul prologues (absorbed casts). */
  prologueClaimedIds: Set<number>;

  /** Matmul ID → epilogue chain node IDs. */
  matmulEpilogueChains: Map<number, number[]>;

  /** Matmul ID → prologue info array. */
  matmulPrologues: Map<number, MatmulPrologueInfo[]>;

  /** Consumer count map (nodeId → number of consumers in the plan). */
  consumerCount: Map<number, number>;

  /** Node IDs bypassed by graph rewrites (identity casts, redundant contiguous). */
  rewriteBypassedIds: Set<number>;

  /** Pre-computed matmul epilogue directives (matmulNodeId → full plan with prologues). */
  matmulDirectives: Map<number, MatmulEpiloguePlan>;

  /** Detected row-program matches (multi-reduction → single kernel). */
  rowProgramMatches: RowProgramMatch[];
}

// ============================================================================
// Matmul Epilogue Detection
// ============================================================================

/**
 * Detect matmul epilogue chains from the plan.
 * Walks forward from each matmul node to find cast→bias→activation chains.
 *
 * This is extracted from the inline code in executor-lowered.ts.
 */
function detectMatmulEpilogueChains(
  planNodes: LazyIRNode[],
  consumers: Map<number, LazyIRNode[]>,
  consumerCount: Map<number, number>,
  nodePosition: Map<number, number>,
  externalNodeIds?: Set<number>,
): {
  epilogueClaimedIds: Set<number>;
  prologueClaimedIds: Set<number>;
  matmulEpilogueChains: Map<number, number[]>;
  matmulPrologues: Map<number, MatmulPrologueInfo[]>;
} {
  const epilogueClaimedIds = new Set<number>();
  const prologueClaimedIds = new Set<number>();
  const matmulEpilogueChains = new Map<number, number[]>();
  const matmulPrologues = new Map<number, MatmulPrologueInfo[]>();

  for (let mi = 0; mi < planNodes.length; mi++) {
    const node = planNodes[mi];
    if (node.op !== "matmul") continue;

    const matmulPos = mi;
    const chainIds: number[] = [];
    let current = node;
    let additionalInputCount = 0;

    for (let depth = 0; depth < 4; depth++) {
      const cc = consumerCount.get(current.id) ?? 0;
      if (cc !== 1) break;
      if (externalNodeIds?.has(current.id)) {
        break;
      }

      const nexts = consumers.get(current.id);
      if (!nexts || nexts.length !== 1) break;
      const next = nexts[0];

      if (next.inputs.length === 0) break;
      let chainInputIdx = 0;
      const primary = next.inputs[0];
      if (primary.kind !== "pending" || primary.node.id !== current.id) {
        if (
          (next.op === "add" || next.op === "mul") &&
          next.inputs.length === 2
        ) {
          const alt = next.inputs[1];
          if (alt.kind === "pending" && alt.node.id === current.id) {
            chainInputIdx = 1;
          } else {
            break;
          }
        } else {
          break;
        }
      }

      // Skip reshape that only removes leading size-1 dims
      if (next.op === "reshape") {
        const curShape = current.shape;
        const nextShape = next.shape;
        if (
          curShape.length === nextShape.length + 1 &&
          curShape[0] === 1 &&
          curShape.slice(1).every((d: number, i: number) => d === nextShape[i])
        ) {
          chainIds.push(next.id);
          current = next;
          continue;
        }
        break;
      }

      let ok = false;
      if (next.op === "cast") ok = true;
      else if (
        (next.op === "add" || next.op === "mul" || next.op === "sub") &&
        next.inputs.length === 2
      ) {
        if (additionalInputCount >= 4) break;
        const secondary = next.inputs[chainInputIdx === 0 ? 1 : 0];
        if (secondary.kind === "materialized") {
          ok = true;
        } else if (secondary.kind === "pending") {
          const secPos = nodePosition.get(secondary.node.id);
          if (secPos !== undefined && secPos < matmulPos) {
            ok = true;
          }
        }
        if (ok) additionalInputCount++;
      } else if (isFusibleOp(next.op) && next.inputs.length === 1) {
        ok = true;
      }

      if (!ok) break;

      chainIds.push(next.id);
      current = next;
    }

    if (chainIds.length > 0) {
      matmulEpilogueChains.set(node.id, chainIds);
      for (const id of chainIds) epilogueClaimedIds.add(id);
    }

    // Detect input-side cast prologues (inference only)
    if (!externalNodeIds || externalNodeIds.size === 0) {
      const prologuesForNode: MatmulPrologueInfo[] = [];
      for (const idx of [0, 1] as const) {
        const inputRef = node.inputs[idx];
        if (inputRef.kind !== "pending") continue;
        const castNode = inputRef.node;
        if (castNode.op !== "cast") continue;
        if ((consumerCount.get(castNode.id) ?? 0) !== 1) continue;
        const castPayload = castNode.payload as { dtype: DType } | undefined;
        if (!castPayload) continue;
        const toDtype = castPayload.dtype;
        const castInput = castNode.inputs[0];
        if (!castInput) continue;
        let fromDtype: DType;
        if (castInput.kind === "pending") {
          fromDtype = castInput.node.dtype;
        } else if (castInput.kind === "materialized") {
          fromDtype = castInput.storage.backendTensor.dtype ?? "f32";
        } else {
          continue;
        }
        if (fromDtype !== "f32" || toDtype !== "f16") continue;

        prologuesForNode.push({
          inputIndex: idx,
          castNodeId: castNode.id,
          originalInputRef: castInput,
          fromDtype,
          toDtype,
        });
        prologueClaimedIds.add(castNode.id);
      }
      if (prologuesForNode.length > 0) {
        matmulPrologues.set(node.id, prologuesForNode);
      }
    }
  }

  return {
    epilogueClaimedIds,
    prologueClaimedIds,
    matmulEpilogueChains,
    matmulPrologues,
  };
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * Analyze a computation graph and produce a unified analysis result.
 *
 * Runs the following detectors in priority order:
 *  1. Matmul epilogue chains (claims cast/bias/activation after matmul)
 *  2. Compound patterns (claims softmax/log_softmax decomposition)
 *  3. Reduction preamble/epilogue (claims elementwise ops adjacent to reductions)
 *  4. Elementwise fusion (claims fusible chains from remaining nodes)
 *
 * Reduction execution still happens inline in segment-executors.ts;
 * graph-compiler handles claiming so preamble nodes aren't stolen by
 * elementwise fusion.
 *
 * @param planNodes - Original plan nodes in topological order
 * @param externalNodeIds - Node IDs with external references (saved-for-backward)
 * @param maxStorageBuffers - Device storage buffer limit for fusion group sizing
 */
export function analyzeGraph(
  planNodes: LazyIRNode[],
  externalNodeIds?: Set<number>,
  maxStorageBuffers?: number,
): GraphAnalysisResult {
  // Reorder plan to cluster fusible chains together
  let reorderedNodes = planNodes;
  if (planNodes.length > 2) {
    reorderedNodes = reorderPlanForFusion(planNodes);
  }

  // Build consumer, position, and ID lookup maps in a single pass
  const consumers = new Map<number, LazyIRNode[]>();
  const consumerCount = new Map<number, number>();
  const nodePosition = new Map<number, number>();
  const nodeById = new Map<number, LazyIRNode>();
  for (let i = 0; i < reorderedNodes.length; i++) {
    const node = reorderedNodes[i];
    nodePosition.set(node.id, i);
    nodeById.set(node.id, node);
    for (const input of node.inputs) {
      if (input.kind === "pending") {
        const producerId = input.node.id;
        consumerCount.set(producerId, (consumerCount.get(producerId) ?? 0) + 1);
        if (!consumers.has(producerId)) consumers.set(producerId, []);
        consumers.get(producerId)!.push(node);
      }
    }
  }

  // --- Graph rewrites: simplify before pattern detection ---
  const rewriteCtx = { planNodes: reorderedNodes, consumers, consumerCount };
  const rewriteBypassedIds = new Set<number>();
  const passStats = runPasses(
    rewriteCtx,
    rewriteBypassedIds,
    SIMPLIFICATION_PASSES,
  );
  // Un-bypass external nodes: they need individual results for materialization.
  if (externalNodeIds) {
    for (const id of externalNodeIds) {
      rewriteBypassedIds.delete(id);
    }
  }

  // Log pass stats when TORCHLETTE_LOG_REWRITES=1
  if (
    typeof process !== "undefined" &&
    process.env?.TORCHLETTE_LOG_REWRITES === "1" &&
    rewriteBypassedIds.size > 0
  ) {
    const parts: string[] = [];
    for (const [name, count] of passStats) {
      if (count > 0) parts.push(`${name}=${count}`);
    }
    // Collect per-op breakdown of bypassed nodes
    const opCounts: Record<string, number> = {};
    for (const node of reorderedNodes) {
      if (rewriteBypassedIds.has(node.id)) {
        opCounts[node.op] = (opCounts[node.op] ?? 0) + 1;
      }
    }
    const opParts = Object.entries(opCounts)
      .sort((a, b) => b[1] - a[1])
      .map(([op, n]) => `${op}×${n}`);
    console.log(
      `[graph-rewrites] ${reorderedNodes.length} nodes → ${parts.join(", ")} (${rewriteBypassedIds.size} bypassed: ${opParts.join(", ")})`,
    );
  }

  // --- Priority 100: Matmul epilogue chains ---
  const {
    epilogueClaimedIds,
    prologueClaimedIds,
    matmulEpilogueChains,
    matmulPrologues,
  } = detectMatmulEpilogueChains(
    reorderedNodes,
    consumers,
    consumerCount,
    nodePosition,
    externalNodeIds,
  );

  // Debug: show epilogue chains for large plans
  if (
    typeof process !== "undefined" &&
    process.env?.TORCHLETTE_DEBUG_FUSION === "1" &&
    reorderedNodes.length > 200 &&
    matmulEpilogueChains.size > 0
  ) {
    console.log(
      `[fusion-debug] ${reorderedNodes.length} nodes: ${matmulEpilogueChains.size} matmul epilogue chains`,
    );
    for (const [mmId, chainIds] of matmulEpilogueChains) {
      const mm = nodeById.get(mmId);
      const chainOps = chainIds
        .map((id) => {
          const n = nodeById.get(id);
          return n ? `${n.op}(${JSON.stringify(n.shape)})` : `?#${id}`;
        })
        .join(" → ");
      console.log(
        `  matmul#${mmId} ${JSON.stringify(mm?.shape)} → ${chainOps}`,
      );
    }
  }

  // Relocate epilogue chain nodes after their matmul
  if (epilogueClaimedIds.size > 0) {
    const unclaimed = reorderedNodes.filter(
      (n) => !epilogueClaimedIds.has(n.id),
    );
    const relocated: LazyIRNode[] = [];
    for (const n of unclaimed) {
      relocated.push(n);
      const chain = matmulEpilogueChains.get(n.id);
      if (chain) {
        for (const id of chain) {
          const chainNode = nodeById.get(id);
          if (chainNode) relocated.push(chainNode);
        }
      }
    }
    reorderedNodes = relocated;

    // Validate topological order after relocation
    const posAfterReloc = new Map<number, number>();
    for (let i = 0; i < reorderedNodes.length; i++) {
      posAfterReloc.set(reorderedNodes[i].id, i);
    }
    for (let i = 0; i < reorderedNodes.length; i++) {
      const node = reorderedNodes[i];
      for (const inp of node.inputs) {
        if (inp.kind === "pending") {
          const depPos = posAfterReloc.get(inp.node.id);
          if (depPos !== undefined && depPos > i) {
            console.error(
              `[TOPOSORT VIOLATION] node ${node.id} op=${node.op} at pos ${i} ` +
              `depends on ${inp.node.id} op=${inp.node.op} at pos ${depPos} ` +
              `(epilogueClaimed=${epilogueClaimedIds.has(inp.node.id)})`,
            );
          }
        }
      }
    }
  }

  // Build matmul directives: full epilogue plans for execution.
  // This runs detectMatmulEpilogueCore() once during analysis so that
  // segment-executors can look up plans instead of re-detecting.
  const matmulDirectives = new Map<number, MatmulEpiloguePlan>();
  if (matmulEpilogueChains.size > 0 || matmulPrologues.size > 0) {
    for (let i = 0; i < reorderedNodes.length; i++) {
      const node = reorderedNodes[i];
      if (node.op !== "matmul") continue;
      const hasChain = matmulEpilogueChains.has(node.id);
      const prologues = matmulPrologues.get(node.id);
      if (!hasChain && !prologues) continue;

      let plan = hasChain
        ? detectMatmulEpilogueCore(
            reorderedNodes,
            i,
            consumerCount,
            externalNodeIds,
          )
        : null;

      // If we have prologues but no epilogue, create a minimal plan
      // so the matmul goes through the epilogue dispatch path with prologue support.
      if (!plan && prologues && prologues.length > 0) {
        plan = {
          consumedCount: 1,
          epilogueOps: [],
          epilogueInputRefs: [],
          outputDtype: node.dtype,
          outputNode: node,
        };
      }

      if (plan) {
        if (prologues && prologues.length > 0) {
          plan.prologues = prologues;
        }
        matmulDirectives.set(node.id, plan);
      }
    }
  }

  // --- Priority 70: Row-program fusion (multi-reduction → single kernel) ---
  const rowProgramMatches = detectRowPrograms(
    reorderedNodes,
    consumerCount,
    consumers,
    externalNodeIds,
    new Set([...epilogueClaimedIds, ...prologueClaimedIds]),
  );
  const rowProgramClaimedIds = new Set<number>();
  for (const match of rowProgramMatches) {
    for (const id of match.coveredNodeIds) {
      rowProgramClaimedIds.add(id);
    }
  }

  // --- Priority 40: Elementwise fusion (via segmentPlanForExecution) ---
  // Bypassed nodes are excluded from fusion (they become view-like pass-throughs)
  if (
    typeof process !== "undefined" &&
    process.env?.TORCHLETTE_DEBUG_FUSION === "1" &&
    reorderedNodes.length > 200
  ) {
    // Show which gelu ops are claimed and by which system
    for (const n of reorderedNodes) {
      if (n.op === "gelu") {
        const epi = epilogueClaimedIds.has(n.id);
        const pro = prologueClaimedIds.has(n.id);
        const row = rowProgramClaimedIds.has(n.id);
        const rew = rewriteBypassedIds.has(n.id);
        if (epi || pro || row || rew) {
          console.log(
            `[fusion-debug] gelu#${n.id} ${JSON.stringify(n.shape)} claimed by: ${epi ? "epilogue " : ""}${pro ? "prologue " : ""}${row ? "rowProgram " : ""}${rew ? "rewrite" : ""}`,
          );
        }
      }
    }
  }

  let allClaimedIds: Set<number> | undefined;
  if (
    epilogueClaimedIds.size > 0 ||
    prologueClaimedIds.size > 0 ||
    rowProgramClaimedIds.size > 0 ||
    rewriteBypassedIds.size > 0
  ) {
    allClaimedIds = new Set([
      ...epilogueClaimedIds,
      ...prologueClaimedIds,
      ...rowProgramClaimedIds,
      ...rewriteBypassedIds,
    ]);
  }
  const segments = segmentPlanForExecution(reorderedNodes, externalNodeIds, {
    maxStorageBuffers,
    enableMultiOutput: true,
    excludedIds: allClaimedIds,
  });

  // Debug dump: show fusion break patterns for large backward plans
  if (
    typeof process !== "undefined" &&
    process.env?.TORCHLETTE_DEBUG_FUSION === "1" &&
    reorderedNodes.length > 200
  ) {
    let totalFusible = 0;
    const runLengths: number[] = [];
    let runLen = 0;
    for (const n of reorderedNodes) {
      if (isFusibleOp(n.op) && !allClaimedIds?.has(n.id)) {
        totalFusible++;
        runLen++;
      } else {
        if (runLen > 0) runLengths.push(runLen);
        runLen = 0;
      }
    }
    if (runLen > 0) runLengths.push(runLen);
    runLengths.sort((a, b) => b - a);

    let fusedNodes = 0;
    for (const seg of segments) {
      if (seg.kind === "fused" && seg.group.nodes.length >= 2)
        fusedNodes += seg.group.nodes.length;
    }

    console.log(
      `\n[fusion-debug] ${reorderedNodes.length} nodes, ${totalFusible} fusible (unclaimed), ${fusedNodes} fused`,
    );
    console.log(
      `  Consecutive runs (top 15): ${runLengths.slice(0, 15).join(", ")}`,
    );

    // Show what breaks fusible runs at [*,*,3072]
    let breaks = 0;
    for (let i = 1; i < reorderedNodes.length && breaks < 8; i++) {
      const prev = reorderedNodes[i - 1];
      const cur = reorderedNodes[i];
      const prevFusible = isFusibleOp(prev.op) && !allClaimedIds?.has(prev.id);
      const curFusible = isFusibleOp(cur.op) && !allClaimedIds?.has(cur.id);
      const prevIs3072 =
        prev.shape.length >= 2 && prev.shape[prev.shape.length - 1] === 3072;
      if (prevFusible && !curFusible && prevIs3072) {
        breaks++;
        console.log(
          `  Break #${breaks} at pos ${i}: ${cur.op} ${cur.dtype} ${JSON.stringify(cur.shape)}${allClaimedIds?.has(cur.id) ? " [CLAIMED]" : ""}${externalNodeIds?.has(cur.id) ? " [EXTERNAL]" : ""}`,
        );
        // Show context
        for (
          let j = Math.max(0, i - 2);
          j < Math.min(reorderedNodes.length, i + 3);
          j++
        ) {
          const n = reorderedNodes[j];
          const f = isFusibleOp(n.op) ? "F" : ".";
          const c = allClaimedIds?.has(n.id) ? "C" : " ";
          const e = externalNodeIds?.has(n.id) ? "E" : " ";
          console.log(
            `    [${j}] ${f}${c}${e} ${n.op.padEnd(14)} ${n.dtype.padEnd(4)} ${JSON.stringify(n.shape)}`,
          );
        }
      }
    }

    // Shape distribution of fusible ops
    const shapeGroups = new Map<
      string,
      { count: number; ops: Map<string, number> }
    >();
    for (const n of reorderedNodes) {
      if (isFusibleOp(n.op) && !allClaimedIds?.has(n.id)) {
        const key = JSON.stringify(n.shape);
        let g = shapeGroups.get(key);
        if (!g) {
          g = { count: 0, ops: new Map() };
          shapeGroups.set(key, g);
        }
        g.count++;
        g.ops.set(n.op, (g.ops.get(n.op) ?? 0) + 1);
      }
    }
    console.log(`  Fusible by shape:`);
    for (const [shape, g] of [...shapeGroups.entries()].sort(
      (a, b) => b[1].count - a[1].count,
    )) {
      const ops = [...g.ops.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([o, c]) => `${o}:${c}`)
        .join(" ");
      console.log(`    ${shape.padEnd(18)} ${g.count} ops (${ops})`);
    }
  }

  return {
    planNodes: reorderedNodes,
    segments,
    epilogueClaimedIds,
    prologueClaimedIds,
    matmulEpilogueChains,
    matmulPrologues,
    consumerCount,
    rewriteBypassedIds,
    matmulDirectives,
    rowProgramMatches,
  };
}

// Graph rewrite passes imported from ./graph-rewrites.ts
