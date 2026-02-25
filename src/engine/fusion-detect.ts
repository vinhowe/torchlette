/**
 * Fusion Group Detection for Lazy Execution Plans
 *
 * This module detects opportunities for operator fusion in lazy execution plans.
 * Per spec §15, elementwise ops can be fused into single GPU kernels.
 *
 * Fusion Rules:
 * - Consecutive elementwise ops can be fused
 * - Random ops break fusion (non-deterministic side effects)
 * - Non-elementwise ops (matmul, reduce, gather) break fusion
 * - Fused groups must have 2+ ops to be worthwhile
 */

import type { DType } from "../backend/types";
import type {
  FusedInput,
  FusedKernelRecipe,
  FusedNode,
  FusedOutput,
} from "../backend/webgpu/fusion-codegen";
import { isRandomOp } from "./ir-optimize";
import type { LazyIRNode, LazyRef } from "./lazy";

/**
 * Ops that can be fused into elementwise kernels.
 */
const FUSIBLE_OPS = new Set([
  // Arithmetic (binary elementwise)
  "add",
  "sub",
  "mul",
  "div",
  "pow",
  // Math (unary elementwise)
  "sqrt",
  "rsqrt",
  "neg",
  "abs",
  "exp",
  "log",
  "sin",
  "cos",
  "floor",
  "ceil",
  "round",
  "sign",
  // Activations (unary elementwise)
  "relu",
  "sigmoid",
  "tanh",
  "gelu",
  "gelu_tanh",
  "gelu_erf",
  "silu",
  "softplus",
  // Comparisons (binary elementwise)
  "eq",
  "ne",
  "lt",
  "le",
  "gt",
  "ge",
  // Ternary (elementwise)
  "where",
  // Type cast (elementwise)
  "cast",
  // Finite check (unary elementwise, returns f32 0/1)
  "isfinite",
  // Note: "max" and "min" are NOT included here because the runtime
  // uses them as reduction ops (single input + dim option), not as
  // binary elementwise ops. The registry's max/min are binary elementwise
  // but the lazy engine's max/min are reductions.
]);

/**
 * Check if an op can be fused.
 */
export function isFusibleOp(op: string): boolean {
  return FUSIBLE_OPS.has(op);
}

/**
 * A detected fusion group in a lazy plan.
 */
export interface FusionGroup {
  /** Nodes to fuse (in execution order) */
  nodes: LazyIRNode[];
  /** Indices in the original plan */
  planIndices: number[];
  /** External inputs (materialized refs or nodes outside the group) */
  externalInputs: LazyRef[];
  /** Output node (last node in group) */
  outputNode: LazyIRNode;
  /** Intermediate nodes that are additional outputs (externally referenced) */
  additionalOutputNodes?: LazyIRNode[];
  /** Intermediates consumed externally that couldn't be promoted to additional
   *  outputs (shape mismatch or binding limit). These need sequential
   *  re-execution after the fused kernel to set their results. */
  neededIntermediates?: LazyIRNode[];
}

/**
 * Result of fusion detection.
 */
interface FusionDetectionResult {
  /** Detected fusion groups */
  groups: FusionGroup[];
  /** Stats about detection */
  stats: {
    totalNodes: number;
    fusibleNodes: number;
    groupCount: number;
    nodesInGroups: number;
  };
}

/**
 * Process a candidate sub-group for fusion, checking external references
 * and splitting as needed. This is the core logic extracted to handle
 * each connected component separately.
 */
function processCandidate(
  subNodes: LazyIRNode[],
  subIndices: number[],
  subNodeIds: Set<number>,
  internalDeps: number[][],
  allPlanNodes: LazyIRNode[],
  externalNodeIds: Set<number> | undefined,
  enableMultiOutput: boolean,
  maxBuffers: number,
  outGroups: FusionGroup[],
): void {
  // Find intermediate nodes referenced from outside the group
  const externallyReferenced = new Set<number>(); // positions within subNodes
  for (let k = 0; k < subNodes.length - 1; k++) {
    const intermediateId = subNodes[k].id;
    // Check if any node outside the group references this intermediate
    for (const node of allPlanNodes) {
      if (subNodeIds.has(node.id)) continue;
      for (const input of node.inputs) {
        if (input.kind === "pending" && input.node.id === intermediateId) {
          externallyReferenced.add(k);
        }
      }
    }
    // Also mark if this intermediate has a live external tensor
    // (e.g., saved-for-backward references)
    if (externalNodeIds?.has(intermediateId)) {
      externallyReferenced.add(k);
    }
  }

  // Try multi-output: keep group intact with additional outputs
  if (enableMultiOutput && externallyReferenced.size > 0) {
    const primaryOutputNode = subNodes[subNodes.length - 1];
    const primaryShape = primaryOutputNode.shape;

    // All additional outputs must have the same shape as primary
    const additionalNodes: LazyIRNode[] = [];
    let shapesMatch = true;
    for (const pos of externallyReferenced) {
      const node = subNodes[pos];
      if (
        node.shape.length !== primaryShape.length ||
        !node.shape.every((d, i) => d === primaryShape[i])
      ) {
        shapesMatch = false;
        break;
      }
      additionalNodes.push(node);
    }

    // Check binding limit: non-inlined externalInputs + 1 (primary) + N (additional) <= maxBuffers
    const externalInputs = collectExternalInputs(subNodes, subNodeIds);
    const nonInlinedInputs = externalInputs.filter(ref => !isInlinableScalar(ref).inlinable).length;
    const totalBindings = nonInlinedInputs + 1 + additionalNodes.length;

    if (shapesMatch && totalBindings <= maxBuffers) {
      // Keep group intact with multi-output
      outGroups.push({
        nodes: subNodes,
        planIndices: subIndices,
        externalInputs,
        outputNode: primaryOutputNode,
        additionalOutputNodes: additionalNodes.length > 0 ? additionalNodes : undefined,
      });
      return; // Skip split logic
    }
  }

  // Fall back to split logic
  const splitAfter = new Set(externallyReferenced);

  // Propagate splits for cross-boundary intra-group references.
  // If there's a split after position s, any node at position j > s
  // that depends on a node at position i where i < s (i.e., i is NOT the
  // output of the sub-group ending at s) needs i to also be a split point.
  // Iterate until stable.
  let changed = true;
  while (changed) {
    changed = false;
    const sortedSplits = [...splitAfter].sort((a, b) => a - b);
    for (let k = 0; k < subNodes.length; k++) {
      for (const depPos of internalDeps[k]) {
        // Find which sub-group boundary separates k from depPos
        // depPos < k, and if there's a split s where depPos < s+1 <= k
        // (i.e., depPos is not the output of its sub-group), we need to
        // add depPos as a split point too.
        for (const s of sortedSplits) {
          // s is the last index of a sub-group (split after s)
          if (depPos < s && s < k) {
            // depPos is an intermediate in the sub-group [start..s],
            // but k (in a later sub-group) depends on it.
            // depPos must become a split point (its sub-group's output).
            if (!splitAfter.has(depPos)) {
              splitAfter.add(depPos);
              changed = true;
            }
          }
        }
      }
    }
  }

  // Split the candidate group at the identified positions
  if (splitAfter.size === 0) {
    // No splits needed - use the whole group
    const externalInputs = collectExternalInputs(subNodes, subNodeIds);
    outGroups.push({
      nodes: subNodes,
      planIndices: subIndices,
      externalInputs,
      outputNode: subNodes[subNodes.length - 1],
    });
  } else {
    // Split into sub-groups at identified positions
    let start = 0;
    const sortedSplits = [...splitAfter].sort((a, b) => a - b);
    for (const splitIdx of sortedSplits) {
      const end = splitIdx + 1;
      const splitNodes = subNodes.slice(start, end);
      const splitIndices = subIndices.slice(start, end);
      if (splitNodes.length >= 2) {
        const splitNodeIds = new Set(splitNodes.map((n) => n.id));
        const externalInputs = collectExternalInputs(splitNodes, splitNodeIds);
        outGroups.push({
          nodes: splitNodes,
          planIndices: splitIndices,
          externalInputs,
          outputNode: splitNodes[splitNodes.length - 1],
        });
      }
      start = end;
    }
    // Remaining nodes after last split
    const remaining = subNodes.slice(start);
    const remainingIndices = subIndices.slice(start);
    if (remaining.length >= 2) {
      const remainingIds = new Set(remaining.map((n) => n.id));
      const externalInputs = collectExternalInputs(remaining, remainingIds);
      outGroups.push({
        nodes: remaining,
        planIndices: remainingIndices,
        externalInputs,
        outputNode: remaining[remaining.length - 1],
      });
    }
  }
}

/**
 * Detect fusion opportunities in a lazy execution plan.
 *
 * @param nodes - Nodes in the plan (topologically sorted)
 * @returns Detected fusion groups
 */
/**
 * Phase 1: Scan for consecutive fusible ops, flush at dependency breaks.
 * Non-fusible nodes that don't depend on the current candidate pass through.
 */
function buildCandidateGroups(
  nodes: LazyIRNode[],
  epilogueClaimedIds: Set<number> | undefined,
): { candidateGroups: Array<{ nodes: LazyIRNode[]; indices: number[] }>; fusibleCount: number } {
  const candidateGroups: Array<{ nodes: LazyIRNode[]; indices: number[] }> = [];
  let currentGroup: { nodes: LazyIRNode[]; indices: number[] } | null = null;
  const candidateNodeIds = new Set<number>();
  let fusibleCount = 0;

  const flushCandidate = () => {
    if (currentGroup && currentGroup.nodes.length >= 2) {
      candidateGroups.push(currentGroup);
    }
    currentGroup = null;
    candidateNodeIds.clear();
  };

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    if (isFusibleOp(node.op) && !epilogueClaimedIds?.has(node.id)) {
      fusibleCount++;
      if (!currentGroup) {
        currentGroup = { nodes: [], indices: [] };
        candidateNodeIds.clear();
      }
      currentGroup.nodes.push(node);
      currentGroup.indices.push(i);
      candidateNodeIds.add(node.id);
    } else if (currentGroup && !hasPendingInputIn(node, candidateNodeIds)) {
      // Independent non-fusible node — passes through as prereq
    } else {
      flushCandidate();
    }
  }
  flushCandidate();

  return { candidateGroups, fusibleCount };
}

/**
 * Phase 2: Split candidate groups into connected components via Union-Find,
 * process each component, and batch independent singletons (Phase 2b).
 */
function splitCandidatesByComponent(
  candidateGroups: Array<{ nodes: LazyIRNode[]; indices: number[] }>,
  nodes: LazyIRNode[],
  externalNodeIds: Set<number> | undefined,
  enableMultiOutput: boolean,
  maxBuffers: number,
): FusionGroup[] {
  const groups: FusionGroup[] = [];

  for (const candidate of candidateGroups) {
    // Build position maps
    const nodeIdToPos = new Map<number, number>();
    for (let k = 0; k < candidate.nodes.length; k++) {
      nodeIdToPos.set(candidate.nodes[k].id, k);
    }

    // Union-Find for connected component decomposition
    const parent: number[] = [];
    const ufRank: number[] = [];
    for (let k = 0; k < candidate.nodes.length; k++) {
      parent[k] = k;
      ufRank[k] = 0;
    }
    const find = (x: number): number => {
      if (parent[x] !== x) parent[x] = find(parent[x]);
      return parent[x];
    };
    const union = (a: number, b: number): void => {
      const ra = find(a);
      const rb = find(b);
      if (ra === rb) return;
      if (ufRank[ra] < ufRank[rb]) {
        parent[ra] = rb;
      } else if (ufRank[ra] > ufRank[rb]) {
        parent[rb] = ra;
      } else {
        parent[rb] = ra;
        ufRank[ra]++;
      }
    };

    // Union nodes connected by internal data dependencies
    // Skip 0-d (scalar) inputs — they broadcast independently
    for (let k = 0; k < candidate.nodes.length; k++) {
      for (const input of candidate.nodes[k].inputs) {
        if (input.kind === "pending") {
          const depPos = nodeIdToPos.get(input.node.id);
          if (depPos !== undefined && input.node.shape.length > 0) {
            union(k, depPos);
          }
        }
      }
    }

    // Group nodes by their root (connected component)
    const componentMap = new Map<number, number[]>();
    for (let k = 0; k < candidate.nodes.length; k++) {
      const root = find(k);
      if (!componentMap.has(root)) componentMap.set(root, []);
      componentMap.get(root)!.push(k);
    }

    // Process each connected component separately
    const singletonPositions: number[] = [];
    for (const positions of componentMap.values()) {
      if (positions.length < 2) {
        singletonPositions.push(positions[0]);
        continue;
      }

      positions.sort((a, b) => a - b);

      const subNodes = positions.map((p) => candidate.nodes[p]);
      const subIndices = positions.map((p) => candidate.indices[p]);
      const subNodeIds = new Set(subNodes.map((n) => n.id));

      const subNodeIdToPos = new Map<number, number>();
      for (let k = 0; k < subNodes.length; k++) {
        subNodeIdToPos.set(subNodes[k].id, k);
      }
      const internalDeps: number[][] = [];
      for (let k = 0; k < subNodes.length; k++) {
        const deps: number[] = [];
        for (const input of subNodes[k].inputs) {
          if (input.kind === "pending") {
            const pos = subNodeIdToPos.get(input.node.id);
            if (pos !== undefined && pos < k) {
              deps.push(pos);
            }
          }
        }
        internalDeps.push(deps);
      }

      processCandidate(
        subNodes, subIndices, subNodeIds, internalDeps,
        nodes, externalNodeIds, enableMultiOutput, maxBuffers, groups,
      );
    }

    // Phase 2b: Batch independent single-node components into multi-output groups
    if (enableMultiOutput && singletonPositions.length >= 2) {
      const byShapeDtype = new Map<string, number[]>();
      for (const pos of singletonPositions) {
        const node = candidate.nodes[pos];
        const key = `${(node.shape ?? [1]).join(",")}_${node.dtype ?? "f32"}`;
        if (!byShapeDtype.has(key)) byShapeDtype.set(key, []);
        byShapeDtype.get(key)!.push(pos);
      }

      for (const sameShapePositions of byShapeDtype.values()) {
        if (sameShapePositions.length < 2) continue;
        sameShapePositions.sort((a, b) => a - b);

        let batchStart = 0;
        while (batchStart < sameShapePositions.length) {
          const batchPositions: number[] = [];
          const seenInputs = new Set<string>();

          for (let j = batchStart; j < sameShapePositions.length; j++) {
            const node = candidate.nodes[sameShapePositions[j]];

            let newInputCount = 0;
            for (const input of node.inputs) {
              if (input.kind === "scalar") continue;
              const inputKey = input.kind === "materialized"
                ? `s:${input.storage.id}`
                : `p:${input.node.id}`;
              if (!seenInputs.has(inputKey)) newInputCount++;
            }

            const wouldBeOutputs = batchPositions.length + 1;
            const wouldBeInputs = seenInputs.size + newInputCount;
            if (wouldBeInputs + wouldBeOutputs > maxBuffers && batchPositions.length >= 2) {
              break;
            }

            batchPositions.push(sameShapePositions[j]);
            for (const input of node.inputs) {
              if (input.kind === "scalar") continue;
              const inputKey = input.kind === "materialized"
                ? `s:${input.storage.id}`
                : `p:${input.node.id}`;
              seenInputs.add(inputKey);
            }
          }

          if (batchPositions.length < 2) break;

          batchStart += batchPositions.length;

          const batchNodes = batchPositions.map(p => candidate.nodes[p]);
          const batchIndices = batchPositions.map(p => candidate.indices[p]);
          const batchNodeIds = new Set(batchNodes.map(n => n.id));
          const externalInputs = collectExternalInputs(batchNodes, batchNodeIds);

          groups.push({
            nodes: batchNodes,
            planIndices: batchIndices,
            externalInputs,
            outputNode: batchNodes[batchNodes.length - 1],
            additionalOutputNodes: batchNodes.slice(0, -1),
          });
        }
      }
    }
  }

  return groups;
}

/**
 * Phase 3: Split groups that exceed the storage buffer binding limit.
 */
function splitGroupsByBufferLimit(groups: FusionGroup[], maxBuffers: number): FusionGroup[] {
  const finalGroups: FusionGroup[] = [];
  for (const group of groups) {
    const numOutputs = 1 + (group.additionalOutputNodes?.length ?? 0);
    const maxInputsForGroup = maxBuffers - numOutputs;
    const nonInlinableInputs = group.externalInputs.filter(ref => !isInlinableScalar(ref).inlinable).length;
    if (nonInlinableInputs <= maxInputsForGroup) {
      finalGroups.push(group);
      continue;
    }

    const subGroups = splitGroupByInputLimit(group, maxInputsForGroup);
    for (const sub of subGroups) {
      finalGroups.push(sub);
    }
  }
  return finalGroups;
}

/**
 * Phase 4: Collect unfused fusible singletons and batch same-shape/dtype
 * nodes into multi-output groups, respecting ordering constraints.
 */
function batchGlobalSingletons(
  nodes: LazyIRNode[],
  finalGroups: FusionGroup[],
  epilogueClaimedIds: Set<number> | undefined,
  maxBuffers: number,
): void {
  const groupedNodeIds = new Set<number>();
  for (const g of finalGroups) {
    for (const n of g.nodes) groupedNodeIds.add(n.id);
  }

  const singletons: Array<{ node: LazyIRNode; planIndex: number }> = [];
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    if (isFusibleOp(node.op) && !groupedNodeIds.has(node.id) && !epilogueClaimedIds?.has(node.id)) {
      singletons.push({ node, planIndex: i });
    }
  }

  if (singletons.length < 2) return;

  const byShapeDtype = new Map<string, Array<{ node: LazyIRNode; planIndex: number }>>();
  for (const s of singletons) {
    const key = `${(s.node.shape ?? [1]).join(",")}_${s.node.dtype ?? "f32"}`;
    if (!byShapeDtype.has(key)) byShapeDtype.set(key, []);
    byShapeDtype.get(key)!.push(s);
  }

  // Build earliest-consumer-position map
  const earliestConsumer = new Map<number, number>();
  for (let i = 0; i < nodes.length; i++) {
    for (const input of nodes[i].inputs) {
      if (input.kind === "pending") {
        const prev = earliestConsumer.get(input.node.id);
        if (prev === undefined || i < prev) {
          earliestConsumer.set(input.node.id, i);
        }
      }
    }
  }

  for (const sameBucket of byShapeDtype.values()) {
    if (sameBucket.length < 2) continue;
    sameBucket.sort((a, b) => a.planIndex - b.planIndex);

    let batchStart = 0;
    while (batchStart < sameBucket.length) {
      const batchEntries: Array<{ node: LazyIRNode; planIndex: number }> = [];
      const batchNodeIds = new Set<number>();
      const seenInputs = new Set<string>();
      let batchMaxPos = -1;
      let batchMinPos = Infinity;
      let endJ = batchStart;

      for (let j = batchStart; j < sameBucket.length; j++) {
        endJ = j + 1;
        const entry = sameBucket[j];
        const node = entry.node;

        const candidateMaxPos = Math.max(batchMaxPos, entry.planIndex);
        const candidateMinPos = Math.min(batchMinPos, entry.planIndex);

        let orderingSafe = true;
        for (const prev of batchEntries) {
          const ec = earliestConsumer.get(prev.node.id);
          if (ec !== undefined && ec <= candidateMaxPos) {
            orderingSafe = false;
            break;
          }
        }
        if (orderingSafe) {
          const ec = earliestConsumer.get(entry.node.id);
          if (ec !== undefined && ec <= candidateMaxPos) {
            orderingSafe = false;
          }
        }
        if (orderingSafe && batchEntries.length > 0) {
          for (let g = candidateMinPos + 1; g < candidateMaxPos; g++) {
            const gapNode = nodes[g];
            if (!gapNode) continue;
            for (const input of gapNode.inputs) {
              if (input.kind === "pending" &&
                  (batchNodeIds.has(input.node.id) || input.node.id === node.id)) {
                orderingSafe = false;
                break;
              }
            }
            if (!orderingSafe) break;
          }
        }
        if (!orderingSafe) break;

        let newInputCount = 0;
        for (const input of node.inputs) {
          if (input.kind === "scalar") continue;
          const inputKey = input.kind === "materialized"
            ? `s:${input.storage.id}`
            : `p:${input.node.id}`;
          if (!seenInputs.has(inputKey)) newInputCount++;
        }

        const wouldBeOutputs = batchEntries.length + 1;
        const wouldBeInputs = seenInputs.size + newInputCount;
        if (wouldBeInputs + wouldBeOutputs > maxBuffers && batchEntries.length >= 2) {
          break;
        }

        batchEntries.push(entry);
        batchNodeIds.add(node.id);
        batchMaxPos = candidateMaxPos;
        batchMinPos = candidateMinPos;
        for (const input of node.inputs) {
          if (input.kind === "scalar") continue;
          const inputKey = input.kind === "materialized"
            ? `s:${input.storage.id}`
            : `p:${input.node.id}`;
          seenInputs.add(inputKey);
        }
      }

      if (batchEntries.length < 2) {
        batchStart = endJ;
        continue;
      }

      batchStart = endJ;

      const batchNodes = batchEntries.map(e => e.node);
      const batchIndices = batchEntries.map(e => e.planIndex);
      const externalInputs = collectExternalInputs(batchNodes, batchNodeIds);

      finalGroups.push({
        nodes: batchNodes,
        planIndices: batchIndices,
        externalInputs,
        outputNode: batchNodes[batchNodes.length - 1],
        additionalOutputNodes: batchNodes.slice(0, -1),
      });
    }
  }
}

/**
 * Phase 5: Promote externally-referenced intermediates to additional outputs.
 */
function promoteIntermediates(
  nodes: LazyIRNode[],
  finalGroups: FusionGroup[],
  externalNodeIds: Set<number> | undefined,
  maxBuffers: number,
): void {
  // Build reverse dependency map for O(1) lookups
  const consumers = new Map<number, number[]>();
  for (const planNode of nodes) {
    for (const input of planNode.inputs) {
      if (input.kind === "pending") {
        let arr = consumers.get(input.node.id);
        if (!arr) { arr = []; consumers.set(input.node.id, arr); }
        arr.push(planNode.id);
      }
    }
  }

  for (const group of finalGroups) {
    const groupNodeIds = new Set(group.nodes.map(n => n.id));
    const outputIds = new Set<number>();
    outputIds.add(group.outputNode.id);
    if (group.additionalOutputNodes) {
      for (const n of group.additionalOutputNodes) outputIds.add(n.id);
    }

    const neededIntermediates: LazyIRNode[] = [];
    for (const gnode of group.nodes) {
      if (outputIds.has(gnode.id)) continue;
      let isExternal = externalNodeIds?.has(gnode.id) ?? false;
      if (!isExternal) {
        const cons = consumers.get(gnode.id);
        if (cons) {
          for (const cid of cons) {
            if (!groupNodeIds.has(cid)) { isExternal = true; break; }
          }
        }
      }
      if (isExternal) neededIntermediates.push(gnode);
    }

    if (neededIntermediates.length > 0) {
      const primaryShape = group.outputNode.shape;
      const promotable = neededIntermediates.filter(
        n => n.shape.length === primaryShape.length && n.shape.every((d, i) => d === primaryShape[i]),
      );
      const currentOutputs = 1 + (group.additionalOutputNodes?.length ?? 0);
      const nonInlinedExtInputs = group.externalInputs.filter(ref => !isInlinableScalar(ref).inlinable).length;
      const availableSlots = maxBuffers - nonInlinedExtInputs - currentOutputs;

      if (promotable.length === neededIntermediates.length && promotable.length <= availableSlots) {
        group.additionalOutputNodes = [
          ...(group.additionalOutputNodes ?? []),
          ...promotable,
        ];
      } else {
        group.neededIntermediates = neededIntermediates;
      }
    }
  }
}

export function detectFusionGroups(
  nodes: LazyIRNode[],
  externalNodeIds?: Set<number>,
  options?: { maxStorageBuffers?: number; enableMultiOutput?: boolean; epilogueClaimedIds?: Set<number> },
): FusionDetectionResult {
  const maxBuffers = options?.maxStorageBuffers ?? Infinity;
  const enableMultiOutput = options?.enableMultiOutput ?? false;
  const epilogueClaimedIds = options?.epilogueClaimedIds;

  // Phase 1: Build candidate groups of consecutive fusible ops
  const { candidateGroups, fusibleCount } = buildCandidateGroups(nodes, epilogueClaimedIds);

  // Phase 2: Split into connected components, process each, batch singletons
  const groups = splitCandidatesByComponent(
    candidateGroups, nodes, externalNodeIds, enableMultiOutput, maxBuffers,
  );

  // Phase 3: Split groups exceeding storage buffer limit
  const finalGroups = splitGroupsByBufferLimit(groups, maxBuffers);

  // Phase 4: Global singleton batching
  if (enableMultiOutput) {
    batchGlobalSingletons(nodes, finalGroups, epilogueClaimedIds, maxBuffers);
  }

  // Phase 5: Promote externally-referenced intermediates to outputs
  promoteIntermediates(nodes, finalGroups, externalNodeIds, maxBuffers);

  return {
    groups: finalGroups,
    stats: {
      totalNodes: nodes.length,
      fusibleNodes: fusibleCount,
      groupCount: finalGroups.length,
      nodesInGroups: finalGroups.reduce((sum, g) => sum + g.nodes.length, 0),
    },
  };
}

/**
 * Split a fusion group into sub-groups where each has at most
 * `maxExternalInputs` external inputs.
 *
 * Greedy approach: iterate nodes in order, tracking which inputs are
 * internal (produced by a prior node in the current sub-group) vs external.
 * When adding a node would push external input count over the limit,
 * emit the current sub-group and start a new one.
 */
function splitGroupByInputLimit(
  group: FusionGroup,
  maxExternalInputs: number,
): FusionGroup[] {
  const result: FusionGroup[] = [];

  // Track which nodes were additional outputs in the original group
  const originalAdditionalIds = new Set<number>();
  if (group.additionalOutputNodes) {
    for (const n of group.additionalOutputNodes) originalAdditionalIds.add(n.id);
  }

  let subNodes: LazyIRNode[] = [];
  let subIndices: number[] = [];
  let subNodeIds = new Set<number>();
  // Track which storage/node IDs are external for the current sub-group
  // Only count non-inlinable inputs toward the binding limit
  let externalStorageIds = new Set<number>();
  let externalPendingIds = new Set<number>();

  const flushSubGroup = () => {
    if (subNodes.length >= 2) {
      const externalInputs = collectExternalInputs(subNodes, subNodeIds);
      // Preserve additional outputs from the original group that are
      // intermediates in this sub-group (not the output node)
      const outputNode = subNodes[subNodes.length - 1];
      const additionalOutputNodes: LazyIRNode[] = [];
      for (let j = 0; j < subNodes.length - 1; j++) {
        if (originalAdditionalIds.has(subNodes[j].id)) {
          additionalOutputNodes.push(subNodes[j]);
        }
      }
      result.push({
        nodes: subNodes,
        planIndices: subIndices,
        externalInputs,
        outputNode,
        additionalOutputNodes: additionalOutputNodes.length > 0 ? additionalOutputNodes : undefined,
      });
    }
    subNodes = [];
    subIndices = [];
    subNodeIds = new Set();
    externalStorageIds = new Set();
    externalPendingIds = new Set();
  };

  for (let k = 0; k < group.nodes.length; k++) {
    const node = group.nodes[k];
    const planIdx = group.planIndices[k];

    // Count how many NEW external inputs this node would introduce.
    // Inlinable scalar constants don't consume binding slots.
    let newExternalCount = 0;
    for (const input of node.inputs) {
      if (input.kind === "materialized") {
        if (!externalStorageIds.has(input.storage.id)) {
          newExternalCount++;
        }
      } else if (input.kind === "pending") {
        if (!subNodeIds.has(input.node.id) && !externalPendingIds.has(input.node.id)) {
          // Skip inlinable scalar constants — they won't consume a binding
          const check = isInlinableScalar(input);
          if (!check.inlinable) {
            newExternalCount++;
          }
        }
      }
    }

    const currentExternalCount = externalStorageIds.size + externalPendingIds.size;
    if (currentExternalCount + newExternalCount > maxExternalInputs && subNodes.length >= 2) {
      // Adding this node would exceed the limit — flush current sub-group
      flushSubGroup();
      // Re-count this node's externals against the fresh sub-group
    }

    // Add node to current sub-group
    subNodes.push(node);
    subIndices.push(planIdx);
    subNodeIds.add(node.id);

    // Update external input tracking (only non-inlinable inputs)
    for (const input of node.inputs) {
      if (input.kind === "materialized") {
        externalStorageIds.add(input.storage.id);
      } else if (input.kind === "pending") {
        if (!subNodeIds.has(input.node.id)) {
          const check = isInlinableScalar(input);
          if (!check.inlinable) {
            externalPendingIds.add(input.node.id);
          }
        }
      }
    }
  }

  flushSubGroup();

  // Phase 2: Cross-boundary intermediate promotion.
  // After splitting, the later sub-group may depend on an intermediate of the
  // earlier sub-group that isn't the output or an additional output. Detect
  // these cross-boundary dependencies and promote them as additional outputs.
  if (result.length >= 2) {
    for (let g = 0; g < result.length - 1; g++) {
      const earlier = result[g];
      const later = result[g + 1];
      const earlierNodeIds = new Set(earlier.nodes.map(n => n.id));
      const earlierOutputIds = new Set<number>();
      earlierOutputIds.add(earlier.outputNode.id);
      if (earlier.additionalOutputNodes) {
        for (const n of earlier.additionalOutputNodes) earlierOutputIds.add(n.id);
      }

      // Find nodes in the later group that depend on intermediates of the earlier group
      for (const laterNode of later.nodes) {
        for (const input of laterNode.inputs) {
          if (input.kind !== "pending") continue;
          const depId = input.node.id;
          if (!earlierNodeIds.has(depId)) continue;
          if (earlierOutputIds.has(depId)) continue;

          // depId is an intermediate of the earlier group consumed by the later group
          // Check if it can be promoted as an additional output
          const depNode = earlier.nodes.find(n => n.id === depId);
          if (!depNode) continue;

          const primaryShape = earlier.outputNode.shape;
          const shapeMatch = depNode.shape.length === primaryShape.length &&
            depNode.shape.every((d, i) => d === primaryShape[i]);

          if (shapeMatch) {
            // Check binding limit: externalInputs + currentOutputs + 1 <= maxExternalInputs + currentOutputs
            const currentOutputs = 1 + (earlier.additionalOutputNodes?.length ?? 0);
            const availableSlots = (maxExternalInputs + currentOutputs) - earlier.externalInputs.length - currentOutputs;
            if (availableSlots > 0) {
              if (!earlier.additionalOutputNodes) earlier.additionalOutputNodes = [];
              earlier.additionalOutputNodes.push(depNode);
              earlierOutputIds.add(depId);
            }
          }
        }
      }
    }
  }

  return result;
}

/**
 * Collect external inputs for a fusion group.
 * External inputs are:
 * - Materialized refs (already computed tensors)
 * - Pending refs to nodes outside the group
 */
function collectExternalInputs(
  groupNodes: LazyIRNode[],
  groupNodeIds: Set<number>,
): LazyRef[] {
  const external: LazyRef[] = [];
  // Use separate tracking for materialized vs pending to avoid ID namespace collisions
  // (a storage.id could equal a node.id, causing a valid input to be skipped)
  const seenStorage = new Set<number>();
  const seenNodes = new Set<number>();
  // Track scalar refs by value+dtype to deduplicate
  const seenScalars = new Set<string>();

  for (const node of groupNodes) {
    for (const input of node.inputs) {
      if (input.kind === "scalar") {
        // Scalar refs are always external but don't consume bindings (inlined as constants)
        const key = `${input.value}_${input.dtype}`;
        if (!seenScalars.has(key)) {
          seenScalars.add(key);
          external.push(input);
        }
      } else if (input.kind === "materialized") {
        // Already computed tensor - always external
        if (!seenStorage.has(input.storage.id)) {
          seenStorage.add(input.storage.id);
          external.push(input);
        }
      } else if (input.kind === "pending") {
        // Pending node - check if it's in our group
        if (!groupNodeIds.has(input.node.id) && !seenNodes.has(input.node.id)) {
          seenNodes.add(input.node.id);
          external.push(input);
        }
      }
    }
  }

  return external;
}

/**
 * Convert a fusion group to a FusedKernelRecipe for codegen.
 *
 * @param group - The detected fusion group
 * @returns Recipe suitable for fusion-codegen
 */
export function groupToRecipe(group: FusionGroup): FusedKernelRecipe {
  const nodeIds = group.nodes.map((n) => n.id);
  const nodeSet = new Set(nodeIds);

  // Build input mapping
  const inputMap = new Map<number, number>(); // LazyRef ID -> negative index
  const inputs: FusedInput[] = [];

  // Counter for synthetic scalar ref IDs (must not collide with node/storage IDs)
  let scalarIdCounter = -1000000;

  const registerInput = (ref: LazyRef): number => {
    let id: number;
    if (ref.kind === "scalar") {
      // Scalar refs use a synthetic ID based on value+dtype
      const key = `scalar_${ref.value}_${ref.dtype}`;
      // Check if we've already registered this exact scalar
      for (const [existingId, existingIdx] of inputMap) {
        const inp = inputs[-existingIdx - 1];
        if (inp?.isInlinedConstant && inp.inlinedValue === ref.value) {
          return existingIdx;
        }
      }
      id = scalarIdCounter--;
    } else {
      id = ref.kind === "materialized" ? -ref.storage.id : ref.node.id;
    }
    if (inputMap.has(id)) {
      return inputMap.get(id)!;
    }
    const idx = inputs.length;
    inputMap.set(id, -(idx + 1));

    if (ref.kind === "scalar") {
      // Scalar refs are always inlined constants — zero binding slots
      inputs.push({
        id,
        index: idx,
        shape: [],
        dtype: ref.dtype,
        isInlinedConstant: true,
        inlinedValue: ref.value,
      });
    } else if (ref.kind === "materialized") {
      inputs.push({
        id,
        index: idx,
        shape: ref.storage.backendTensor.shape.slice(),
        dtype: (ref.storage.backendTensor.dtype as DType) ?? "f32",
      });
    } else {
      // Check if this pending ref is an inlinable scalar constant
      const inlineCheck = isInlinableScalar(ref);
      const inputEntry: FusedInput = {
        id: ref.node.id,
        index: idx,
        shape: ref.node.shape ?? [1],
        dtype: (ref.node.dtype as DType) ?? "f32",
      };
      if (inlineCheck.inlinable) {
        inputEntry.isInlinedConstant = true;
        inputEntry.inlinedValue = inlineCheck.value;
      }
      inputs.push(inputEntry);
    }
    return -(idx + 1);
  };

  // Build fused nodes
  const fusedNodes: FusedNode[] = [];
  for (const node of group.nodes) {
    const mappedInputs: number[] = [];

    for (const input of node.inputs) {
      if (input.kind === "pending" && nodeSet.has(input.node.id)) {
        // Internal reference
        mappedInputs.push(input.node.id);
      } else {
        // External input
        const negIdx = registerInput(input);
        mappedInputs.push(negIdx);
      }
    }

    // Translate generic "cast" op to specific "cast_{dtype}" for codegen registry
    let fusedOp = node.op;
    if (node.op === "cast" && node.dtype) {
      fusedOp = `cast_${node.dtype}`;
    }

    // Check if node is any output (primary or additional)
    const isAdditionalOutput = group.additionalOutputNodes?.some((n) => n.id === node.id) ?? false;
    fusedNodes.push({
      id: node.id,
      op: fusedOp,
      inputs: mappedInputs,
      shape: node.shape ?? [1],
      dtype: (node.dtype as DType) ?? "f32",
      isOutput: node === group.outputNode || isAdditionalOutput,
    });
  }

  // Build outputs: primary output at index 0, then additional outputs
  const outputNode = group.outputNode;
  const outputs: FusedOutput[] = [
    {
      nodeId: outputNode.id,
      index: 0,
      shape: outputNode.shape ?? [1],
      dtype: (outputNode.dtype as DType) ?? "f32",
    },
  ];

  if (group.additionalOutputNodes) {
    for (let i = 0; i < group.additionalOutputNodes.length; i++) {
      const addNode = group.additionalOutputNodes[i];
      outputs.push({
        nodeId: addNode.id,
        index: i + 1,
        shape: addNode.shape ?? [1],
        dtype: (addNode.dtype as DType) ?? "f32",
      });
    }
  }

  return {
    id: `lazy_fused_${nodeIds.join("_")}`,
    nodes: fusedNodes,
    inputs,
    outputs,
  };
}

/**
 * Execution plan segment - either fusible or sequential.
 */
export type ExecutionSegment =
  | { kind: "fused"; group: FusionGroup; recipe: FusedKernelRecipe }
  | { kind: "sequential"; nodes: LazyIRNode[] };

/**
 * Segment a lazy plan into fusible and sequential parts.
 *
 * @param nodes - Nodes in the plan
 * @returns Segments for execution
 */
export function segmentPlanForExecution(
  nodes: LazyIRNode[],
  externalNodeIds?: Set<number>,
  options?: { maxStorageBuffers?: number; enableMultiOutput?: boolean; epilogueClaimedIds?: Set<number> },
): ExecutionSegment[] {
  const { groups } = detectFusionGroups(nodes, externalNodeIds, options);
  const segments: ExecutionSegment[] = [];

  // Build index lookup for fusion groups
  const indexToGroup = new Map<number, FusionGroup>();
  for (const group of groups) {
    for (const idx of group.planIndices) {
      indexToGroup.set(idx, group);
    }
  }

  // Track which groups have been emitted.
  const emittedGroups = new Set<FusionGroup>();

  // Pre-compute max plan index for each group so we can emit at the last
  // member's position. This ensures all gap nodes (including those in other
  // groups) are processed before the fused segment.
  const groupMaxIdx = new Map<FusionGroup, number>();
  for (const group of groups) {
    groupMaxIdx.set(group, Math.max(...group.planIndices));
  }

  let i = 0;
  while (i < nodes.length) {
    const group = indexToGroup.get(i);

    if (group) {
      if (!emittedGroups.has(group) && i === groupMaxIdx.get(group)) {
        // Last member — emit the fused segment now.
        // All gap nodes (sequential or other groups) between the first and
        // last members have already been processed by earlier iterations.
        emittedGroups.add(group);
        const recipe = groupToRecipe(group);
        segments.push({ kind: "fused", group, recipe });
      }
      // Skip this group member (will be handled by fused dispatch)
      i++;
    } else {
      // Sequential execution - collect consecutive non-fused nodes
      const seqNodes: LazyIRNode[] = [];
      while (i < nodes.length && !indexToGroup.has(i)) {
        seqNodes.push(nodes[i]);
        i++;
      }
      if (seqNodes.length > 0) {
        segments.push({ kind: "sequential", nodes: seqNodes });
      }
    }
  }

  return segments;
}

/**
 * Check if a plan has any fusible opportunities (consecutive fusible ops).
 */
export function hasFusionOpportunities(nodes: LazyIRNode[]): boolean {
  let consecutiveFusible = 0;
  for (const node of nodes) {
    if (isFusibleOp(node.op)) {
      consecutiveFusible++;
      if (consecutiveFusible >= 2) {
        return true;
      }
    } else {
      consecutiveFusible = 0;
    }
  }
  return false;
}

/**
 * Relaxed pre-check: returns true if the plan has 2+ fusible ops anywhere
 * (not necessarily consecutive). Used as a cheaper gate before reordering.
 */
export function hasFusionPotential(nodes: LazyIRNode[]): boolean {
  let count = 0;
  for (const node of nodes) {
    if (isFusibleOp(node.op) && ++count >= 2) return true;
  }
  return false;
}

/**
 * Check if a LazyRef is a scalar constant that can be
 * inlined as a WGSL literal (avoiding a storage buffer binding).
 *
 * Eligible refs:
 * - Scalar LazyRef (kind: "scalar") → value is carried directly
 * - Pending refs pointing to scalar constant nodes:
 *   - tensorFromArray with totalElements === 1 → value = payload.values[0]
 *   - full with totalElements === 1 → value = payload.fillValue
 *   - zeros with totalElements === 1 → value = 0.0
 *
 * Materialized refs already have GPU buffers and can't be inlined without readback.
 */
function isInlinableScalar(ref: LazyRef): { inlinable: true; value: number } | { inlinable: false } {
  // Scalar refs are trivially inlinable
  if (ref.kind === "scalar") {
    return { inlinable: true, value: ref.value };
  }
  if (ref.kind !== "pending") return { inlinable: false };
  const node = ref.node;
  const totalElements = (node.shape ?? [1]).reduce((a: number, b: number) => a * b, 1);
  if (totalElements !== 1) return { inlinable: false };
  // Don't inline if the node already has a result (it's been materialized)
  if (node.result) return { inlinable: false };

  switch (node.op) {
    case "tensorFromArray": {
      const payload = node.payload as { values: number[] } | undefined;
      if (payload?.values && payload.values.length === 1) {
        return { inlinable: true, value: payload.values[0] };
      }
      return { inlinable: false };
    }
    case "full": {
      const payload = node.payload as { fillValue: number } | undefined;
      if (payload && typeof payload.fillValue === "number") {
        return { inlinable: true, value: payload.fillValue };
      }
      return { inlinable: false };
    }
    case "zeros": {
      return { inlinable: true, value: 0.0 };
    }
    default:
      return { inlinable: false };
  }
}

/**
 * Check if a node has any pending input whose node ID is in the given set.
 */
function hasPendingInputIn(node: LazyIRNode, nodeIdSet: Set<number>): boolean {
  for (const input of node.inputs) {
    if (input.kind === "pending" && nodeIdSet.has(input.node.id)) return true;
  }
  return false;
}

/**
 * Priority selection for Kahn's algorithm:
 * P0 = fusible node continuing current chain (has pending input in chain),
 * P1 = fusible node (new chain),
 * P2 = non-fusible.
 * Ties broken by original DFS position (determinism).
 */
function selectBestForFusion(
  ready: Set<number>,
  chainNodeIds: Set<number>,
  nodeById: Map<number, LazyIRNode>,
  originalPos: Map<number, number>,
): number {
  let bestId = -1;
  let bestPriority = 3;
  let bestPos = Infinity;

  for (const id of ready) {
    const node = nodeById.get(id)!;
    const fusible = isFusibleOp(node.op);
    let priority: number;

    if (fusible && chainNodeIds.size > 0 && hasPendingInputIn(node, chainNodeIds)) {
      priority = 0; // Chain continuation
    } else if (fusible) {
      priority = 1; // New chain
    } else {
      priority = 2; // Non-fusible
    }

    const pos = originalPos.get(id)!;
    if (priority < bestPriority || (priority === bestPriority && pos < bestPos)) {
      bestId = id;
      bestPriority = priority;
      bestPos = pos;
    }
  }
  return bestId;
}

/**
 * Reorder a lazy execution plan to cluster fusible dependency chains together.
 * Uses Kahn's algorithm (topological sort) with a priority function that
 * prefers emitting fusible nodes adjacent to each other.
 *
 * This is a pure optimization — the output is always a valid topological order,
 * so all existing downstream code works unchanged.
 */
export function reorderPlanForFusion(nodes: LazyIRNode[]): LazyIRNode[] {
  if (nodes.length <= 2) return nodes;

  // 1. Build maps: nodeById, originalPosition (for deterministic tie-breaking)
  const nodeById = new Map<number, LazyIRNode>();
  const originalPos = new Map<number, number>();
  for (let i = 0; i < nodes.length; i++) {
    nodeById.set(nodes[i].id, nodes[i]);
    originalPos.set(nodes[i].id, i);
  }

  // 2. Compute in-degree + successor lists (only edges within the plan)
  const inDegree = new Map<number, number>();
  const successors = new Map<number, number[]>();
  for (const node of nodes) {
    inDegree.set(node.id, 0);
    successors.set(node.id, []);
  }
  for (const node of nodes) {
    for (const input of node.inputs) {
      if (input.kind === "pending" && nodeById.has(input.node.id)) {
        inDegree.set(node.id, inDegree.get(node.id)! + 1);
        successors.get(input.node.id)!.push(node.id);
      }
    }
  }

  // 3. Initialize ready set with in-degree 0 nodes
  const ready = new Set<number>();
  for (const node of nodes) {
    if (inDegree.get(node.id) === 0) ready.add(node.id);
  }

  // 4. Kahn's with fusion-aware priority
  const result: LazyIRNode[] = [];
  // Track the set of fusible node IDs that form the "current chain"
  let chainNodeIds = new Set<number>();

  while (ready.size > 0) {
    const best = selectBestForFusion(ready, chainNodeIds, nodeById, originalPos);
    ready.delete(best);
    result.push(nodeById.get(best)!);

    const bestNode = nodeById.get(best)!;
    if (isFusibleOp(bestNode.op)) {
      chainNodeIds.add(best);
    } else {
      chainNodeIds = new Set();
    }

    for (const succId of successors.get(best)!) {
      const newDeg = inDegree.get(succId)! - 1;
      inDegree.set(succId, newDeg);
      if (newDeg === 0) ready.add(succId);
    }
  }

  return result;
}

/**
 * Compute a structural fingerprint for a lazy plan.
 *
 * Two plans with the same fingerprint have identical op types, shapes, dtypes,
 * and dependency structure (relative input positions). This enables caching
 * fusion analysis results across steps for compiled regions.
 *
 * Uses FNV-1a hashing with structural encoding. Includes external node
 * positions and scalar constant values in the fingerprint.
 */
export function computePlanFingerprint(
  nodes: LazyIRNode[],
  externalNodeIds?: Set<number>,
): number {
  // Map node IDs to plan positions for relative input encoding
  const idToPos = new Map<number, number>();
  for (let i = 0; i < nodes.length; i++) {
    idToPos.set(nodes[i].id, i);
  }

  let h = 0x811c9dc5; // FNV-1a 32-bit offset basis
  const prime = 0x01000193;

  const hashByte = (b: number) => {
    h ^= b & 0xff;
    h = Math.imul(h, prime);
  };
  const hashInt = (v: number) => {
    hashByte(v & 0xff);
    hashByte((v >>> 8) & 0xff);
    hashByte((v >>> 16) & 0xff);
    hashByte((v >>> 24) & 0xff);
  };
  const hashStr = (s: string) => {
    for (let j = 0; j < s.length; j++) hashByte(s.charCodeAt(j));
    hashByte(0); // null terminator
  };

  // Hash node count as discriminator
  hashInt(nodes.length);

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];

    // Op
    hashStr(node.op);

    // Shape
    hashInt(node.shape.length);
    for (const dim of node.shape) hashInt(dim);

    // Dtype
    hashStr(node.dtype ?? "f32");

    // Inputs: encode as relative positions
    hashInt(node.inputs.length);
    for (const inp of node.inputs) {
      if (inp.kind === "pending") {
        const pos = idToPos.get(inp.node.id);
        hashInt(pos !== undefined ? pos : -1);
        hashByte(0x01); // pending marker
      } else if (inp.kind === "scalar") {
        hashByte(0x02); // scalar marker
        // Hash scalar value as float64 bytes
        const buf = new Float64Array(1);
        buf[0] = inp.value;
        const bytes = new Uint8Array(buf.buffer);
        for (let b = 0; b < 8; b++) hashByte(bytes[b]);
      } else {
        hashByte(0x03); // materialized marker
        // Hash shape+dtype (not storage.id, which changes across steps)
        const mShape = inp.storage.backendTensor.shape;
        hashInt(mShape.length);
        for (const d of mShape) hashInt(d);
        hashStr((inp.storage.backendTensor.dtype as string) ?? "f32");
      }
    }

    // External status
    if (externalNodeIds?.has(node.id)) {
      hashByte(0xEE);
    }

    // Payload: for ops with structural payload (cast dtype, etc.)
    if (node.payload) {
      const p = node.payload as Record<string, unknown>;
      if (p.dtype) hashStr(String(p.dtype));
      if (typeof p.dim === "number") hashInt(p.dim);
      if (typeof p.k === "number") hashInt(p.k);
    }
  }

  return h >>> 0; // ensure unsigned
}
