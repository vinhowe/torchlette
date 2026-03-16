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
import {
  type ExecutionSegment,
  isFusibleOp,
  reorderPlanForFusion,
  segmentPlanForExecution,
} from "./fusion-detect";
import { runPasses, SIMPLIFICATION_PASSES } from "./graph-rewrites";
import type { LazyIRNode } from "../graph/types";
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
      if (externalNodeIds?.has(current.id) && !chainIds.includes(current.id)) {
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
    epilogueClaimedIds: allClaimedIds,
  });

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
