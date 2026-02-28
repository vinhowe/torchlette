/**
 * Unified Graph Compiler
 *
 * Consolidates the 4 scattered pattern detection systems into a single
 * `analyzeGraph()` call with priority-ordered pattern detectors:
 *
 *  1. Matmul epilogue chains  (priority 100)
 *  2. Compound patterns        (priority 80)
 *  3. Reduction preamble/epilogue (detected during execution, not here yet)
 *  4. Elementwise fusion       (priority 40)
 *
 * The analysis phase runs once per structural fingerprint and produces a
 * `GraphAnalysisResult` consumed by executor-optimized.ts. Results are
 * cached in the FusionAnalysisTemplate.
 */

import type { DType } from "../backend/types";
import type { LazyIRNode, LazyRef } from "./lazy-types";
import type { MatmulPrologueInfo } from "./matmul-epilogue";
import type { CompoundMatch } from "./compound-patterns";
import { detectCompoundPatterns } from "./compound-patterns";
import { runRewritePasses } from "./graph-rewrites";
import {
  reorderPlanForFusion,
  segmentPlanForExecution,
  type ExecutionSegment,
} from "./fusion-detect";

// ============================================================================
// Types
// ============================================================================

/** Result of the unified graph analysis. */
export interface GraphAnalysisResult {
  /** Plan nodes in final (reordered) execution order. */
  planNodes: LazyIRNode[];

  /** Execution segments (fused and sequential). */
  segments: ExecutionSegment[];

  /** Node IDs claimed by matmul epilogue chains. */
  epilogueClaimedIds: Set<number>;

  /** Node IDs claimed by matmul prologues (absorbed casts). */
  prologueClaimedIds: Set<number>;

  /** Node IDs claimed by compound patterns (softmax, etc.). */
  compoundClaimedIds: Set<number>;

  /** Matmul ID → epilogue chain node IDs. */
  matmulEpilogueChains: Map<number, number[]>;

  /** Matmul ID → prologue info array. */
  matmulPrologues: Map<number, MatmulPrologueInfo[]>;

  /** Detected compound pattern matches. */
  compoundMatches: CompoundMatch[];

  /** Consumer count map (nodeId → number of consumers in the plan). */
  consumerCount: Map<number, number>;

  /** Node IDs bypassed by graph rewrites (identity casts, redundant contiguous). */
  rewriteBypassedIds: Set<number>;
}

// ============================================================================
// Matmul Epilogue Detection
// ============================================================================

/**
 * Detect matmul epilogue chains from the plan.
 * Walks forward from each matmul node to find cast→bias→activation chains.
 *
 * This is extracted from the inline code in executor-optimized.ts.
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
        if ((next.op === "add" || next.op === "mul") && next.inputs.length === 2) {
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
        if (curShape.length === nextShape.length + 1
            && curShape[0] === 1
            && curShape.slice(1).every((d: number, i: number) => d === nextShape[i])) {
          chainIds.push(next.id);
          current = next;
          continue;
        }
        break;
      }

      let ok = false;
      if (next.op === "cast") ok = true;
      else if ((next.op === "add" || next.op === "mul") && next.inputs.length === 2) {
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
      } else if (next.op === "relu" || next.op === "silu" || next.op === "sigmoid" || next.op === "tanh" || next.op === "gelu") {
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

  return { epilogueClaimedIds, prologueClaimedIds, matmulEpilogueChains, matmulPrologues };
}

/**
 * Relocate epilogue chain nodes to appear immediately after their matmul.
 */
function relocateEpilogueNodes(
  planNodes: LazyIRNode[],
  epilogueClaimedIds: Set<number>,
  matmulEpilogueChains: Map<number, number[]>,
  nodeById: Map<number, LazyIRNode>,
): LazyIRNode[] {
  const unclaimed = planNodes.filter(n => !epilogueClaimedIds.has(n.id));
  const relocated: LazyIRNode[] = [];
  for (const n of unclaimed) {
    relocated.push(n);
    const chain = matmulEpilogueChains.get(n.id);
    if (chain) {
      for (const id of chain) {
        relocated.push(nodeById.get(id)!);
      }
    }
  }
  return relocated;
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
 *  3. Elementwise fusion (claims fusible chains from remaining nodes)
 *
 * Reduction preamble/epilogue detection still happens during execution
 * (in segment-executors.ts) for now. It will be moved here in a future phase.
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

  // Build consumer/producer maps
  const consumers = new Map<number, LazyIRNode[]>();
  const consumerCount = new Map<number, number>();
  for (const node of reorderedNodes) {
    for (const input of node.inputs) {
      if (input.kind === "pending") {
        const producerId = input.node.id;
        consumerCount.set(producerId, (consumerCount.get(producerId) ?? 0) + 1);
        if (!consumers.has(producerId)) consumers.set(producerId, []);
        consumers.get(producerId)!.push(node);
      }
    }
  }

  const nodePosition = new Map<number, number>();
  for (let i = 0; i < reorderedNodes.length; i++) {
    nodePosition.set(reorderedNodes[i].id, i);
  }

  const nodeById = new Map<number, LazyIRNode>();
  for (const n of reorderedNodes) nodeById.set(n.id, n);

  // --- Graph rewrites: simplify before pattern detection ---
  const rewriteBypassedIds = runRewritePasses({
    planNodes: reorderedNodes,
    consumers,
    consumerCount,
  });

  // --- Priority 100: Matmul epilogue chains ---
  const {
    epilogueClaimedIds,
    prologueClaimedIds,
    matmulEpilogueChains,
    matmulPrologues,
  } = detectMatmulEpilogueChains(
    reorderedNodes, consumers, consumerCount, nodePosition, externalNodeIds,
  );

  // Relocate epilogue chain nodes after their matmul
  if (epilogueClaimedIds.size > 0) {
    reorderedNodes = relocateEpilogueNodes(
      reorderedNodes, epilogueClaimedIds, matmulEpilogueChains, nodeById,
    );
  }

  // --- Priority 80: Compound patterns (softmax, log_softmax) ---
  const compoundMatches = detectCompoundPatterns(
    reorderedNodes, consumerCount, consumers, externalNodeIds,
  );
  const compoundClaimedIds = new Set<number>();
  for (const match of compoundMatches) {
    for (const id of match.coveredNodeIds) {
      compoundClaimedIds.add(id);
    }
  }

  // --- Priority 40: Elementwise fusion (via segmentPlanForExecution) ---
  // Bypassed nodes are excluded from fusion (they become view-like pass-throughs)
  let allClaimedIds: Set<number> | undefined;
  if (epilogueClaimedIds.size > 0 || prologueClaimedIds.size > 0 ||
      compoundClaimedIds.size > 0 || rewriteBypassedIds.size > 0) {
    allClaimedIds = new Set([
      ...epilogueClaimedIds, ...prologueClaimedIds,
      ...compoundClaimedIds, ...rewriteBypassedIds,
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
    compoundClaimedIds,
    matmulEpilogueChains,
    matmulPrologues,
    compoundMatches,
    consumerCount,
    rewriteBypassedIds,
  };
}
