/**
 * Unified Graph Compiler
 *
 * Consolidates the 4 scattered pattern detection systems into a single
 * `analyzeGraph()` call with priority-ordered pattern detectors:
 *
 *  1. Matmul epilogue chains       (priority 100)
 *  2. Compound patterns             (priority 80)
 *  3. Reduction preamble/epilogue   (priority 60 — claiming only)
 *  4. Elementwise fusion            (priority 40)
 *
 * The analysis phase runs once per structural fingerprint and produces a
 * `GraphAnalysisResult` consumed by executor-optimized.ts. Results are
 * cached in the FusionAnalysisTemplate.
 */

import type { DType } from "../backend/types";
import type { CompoundMatch } from "./compound-patterns";
import { detectCompoundPatterns } from "./compound-patterns";
import {
  type ExecutionSegment,
  isFusibleOp,
  reorderPlanForFusion,
  segmentPlanForExecution,
} from "./fusion-detect";
import type { LazyIRNode, LazyRef } from "./lazy-types";
import { isViewOp } from "./lowered-plan";
import type { MatmulPrologueInfo } from "./matmul-epilogue";
import {
  detectReductionEpilogue,
  detectReductionFusion,
  detectReductionPreamble,
} from "./reduction-preamble";

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
  eliminateIdentityCasts(rewriteCtx, rewriteBypassedIds);
  eliminateRedundantContiguous(rewriteCtx, rewriteBypassedIds);
  eliminateAlgebraicIdentities(rewriteCtx, rewriteBypassedIds);

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

  // --- Priority 80: Compound patterns (softmax, log_softmax) ---
  const compoundMatches = detectCompoundPatterns(
    reorderedNodes,
    consumerCount,
    consumers,
    externalNodeIds,
  );
  const compoundClaimedIds = new Set<number>();
  for (const match of compoundMatches) {
    for (const id of match.coveredNodeIds) {
      compoundClaimedIds.add(id);
    }
  }

  // --- Priority 60: Reduction preamble/epilogue claiming ---
  // Scan for elementwise→reduction and reduction→elementwise patterns.
  // Claim their nodes so they're excluded from elementwise fusion (P40),
  // ensuring preamble/epilogue ops stay in sequential segments where
  // inline detection can fuse them into reduction kernels.
  const reductionClaimedIds = new Set<number>();
  for (let i = 0; i < reorderedNodes.length; i++) {
    const node = reorderedNodes[i];
    if (reductionClaimedIds.has(node.id)) continue;
    if (epilogueClaimedIds.has(node.id) || compoundClaimedIds.has(node.id))
      continue;

    // Try combined preamble+epilogue first
    if (isFusibleOp(node.op)) {
      const fusion = detectReductionFusion(
        reorderedNodes,
        i,
        consumerCount,
        externalNodeIds,
      );
      if (fusion) {
        for (const n of fusion.preambleChain) reductionClaimedIds.add(n.id);
        for (const n of fusion.epilogueChain) reductionClaimedIds.add(n.id);
        continue;
      }
      // Try preamble-only
      const preamble = detectReductionPreamble(
        reorderedNodes,
        i,
        consumerCount,
      );
      if (preamble) {
        for (const n of preamble.preambleChain) reductionClaimedIds.add(n.id);
        continue;
      }
    }

    // Try epilogue-only for reduction nodes
    if (node.op === "sum" || node.op === "mean" || node.op === "max") {
      const epilogue = detectReductionEpilogue(
        reorderedNodes,
        i,
        consumerCount,
        externalNodeIds,
      );
      if (epilogue) {
        // Claim epilogue chain nodes (not the reduction itself — it's the anchor)
        for (let j = 1; j < epilogue.consumedCount; j++) {
          reductionClaimedIds.add(reorderedNodes[i + j].id);
        }
      }
    }
  }

  // --- Priority 40: Elementwise fusion (via segmentPlanForExecution) ---
  // Bypassed nodes are excluded from fusion (they become view-like pass-throughs)
  let allClaimedIds: Set<number> | undefined;
  if (
    epilogueClaimedIds.size > 0 ||
    prologueClaimedIds.size > 0 ||
    compoundClaimedIds.size > 0 ||
    reductionClaimedIds.size > 0 ||
    rewriteBypassedIds.size > 0
  ) {
    allClaimedIds = new Set([
      ...epilogueClaimedIds,
      ...prologueClaimedIds,
      ...compoundClaimedIds,
      ...reductionClaimedIds,
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
    compoundClaimedIds,
    matmulEpilogueChains,
    matmulPrologues,
    compoundMatches,
    consumerCount,
    rewriteBypassedIds,
  };
}

// ============================================================================
// Graph Rewrite Passes (inlined from graph-rewrites.ts)
// ============================================================================

interface RewriteContext {
  planNodes: LazyIRNode[];
  consumers: Map<number, LazyIRNode[]>;
  consumerCount: Map<number, number>;
}

/** Eliminate identity casts: cast(x, dtype=x.dtype) → bypass. */
function eliminateIdentityCasts(
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
function eliminateRedundantContiguous(
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
function eliminateAlgebraicIdentities(
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
