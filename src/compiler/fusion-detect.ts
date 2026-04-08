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

import { ensureDType, type DType } from "../backend/types";
import type {
  FusedInput,
  FusedKernelRecipe,
  FusedNode,
  FusedOutput,
} from "../backend/webgpu/fusion-types";
import { shapesEqual } from "../core/shape";
import type { LazyIRNode, LazyRef } from "../graph/types";
import { OP_REGISTRY } from "../ops/registry";

/**
 * Ops that can be fused into elementwise kernels.
 * Derived from OP_REGISTRY (single source of truth).
 *
 * Exclusions:
 * - cast_* variants: the lazy engine uses "cast" (mapped to cast_f16 etc. at codegen time)
 * - max, min: the lazy engine uses these as reductions (arity 1 + dim), not binary elementwise
 */
const FUSIBLE_OPS = new Set<string>();
for (const [name, def] of Object.entries(OP_REGISTRY)) {
  if (def.fusible && !name.startsWith("cast_")) {
    FUSIBLE_OPS.add(name);
  }
}
// "cast" is fusible (mapped to cast_f16 etc. at codegen time)
FUSIBLE_OPS.add("cast");
// min/max are reduction ops — NOT in the registry (use minimum/maximum for
// binary elementwise). No need to delete them from FUSIBLE_OPS.

/**
 * Check if an op can be fused.
 */
export function isFusibleOp(op: string): boolean {
  return FUSIBLE_OPS.has(op);
}

/** Collect node IDs into a Set. */
function nodeIdSet(nodes: LazyIRNode[]): Set<number> {
  return new Set(nodes.map((n) => n.id));
}

/** Build a map from node IDs to their index in an array. */
export function buildIdPositionMap(nodes: LazyIRNode[]): Map<number, number> {
  const map = new Map<number, number>();
  for (let k = 0; k < nodes.length; k++) {
    map.set(nodes[k].id, k);
  }
  return map;
}

/** Collect output + additional output node IDs for a fusion group. */
function groupOutputIds(group: {
  outputNode: LazyIRNode;
  additionalOutputNodes?: LazyIRNode[];
}): Set<number> {
  const ids = new Set([group.outputNode.id]);
  if (group.additionalOutputNodes) {
    for (const n of group.additionalOutputNodes) ids.add(n.id);
  }
  return ids;
}

/** Shape+dtype key for grouping nodes by compatible shape. */
function shapeDtypeKey(node: LazyIRNode): string {
  return `${(node.shape ?? [1]).join(",")}_${node.dtype ?? "f32"}`;
}

/** Count external inputs that require a storage buffer binding (not inlinable scalars). */
function countNonInlinableInputs(inputs: LazyRef[]): number {
  return inputs.filter((ref) => !isInlinableScalar(ref).inlinable).length;
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
      if (!shapesEqual(node.shape, primaryShape)) {
        shapesMatch = false;
        break;
      }
      additionalNodes.push(node);
    }

    // Check binding limit: non-inlined externalInputs + 1 (primary) + N (additional) <= maxBuffers
    const externalInputs = collectExternalInputs(subNodes, subNodeIds);
    const totalBindings =
      countNonInlinableInputs(externalInputs) + 1 + additionalNodes.length;

    if (shapesMatch && totalBindings <= maxBuffers) {
      // Keep group intact with multi-output
      outGroups.push({
        nodes: subNodes,
        planIndices: subIndices,
        externalInputs,
        outputNode: primaryOutputNode,
        additionalOutputNodes:
          additionalNodes.length > 0 ? additionalNodes : undefined,
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
        const splitNodeIds = nodeIdSet(splitNodes);
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
      const remainingIds = nodeIdSet(remaining);
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
  excludedIds: Set<number> | undefined,
): {
  candidateGroups: Array<{ nodes: LazyIRNode[]; indices: number[] }>;
  fusibleCount: number;
} {
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
    if (isFusibleOp(node.op) && !excludedIds?.has(node.id)) {
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
    const nodeIdToPos = buildIdPositionMap(candidate.nodes);

    // Union-Find for connected component decomposition (path compression only)
    const parent = Array.from({ length: candidate.nodes.length }, (_, i) => i);
    const find = (x: number): number => {
      if (parent[x] !== x) parent[x] = find(parent[x]);
      return parent[x];
    };
    const union = (a: number, b: number): void => {
      const ra = find(a);
      const rb = find(b);
      if (ra !== rb) parent[rb] = ra;
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
      componentMap.get(root)?.push(k);
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
      const subNodeIds = nodeIdSet(subNodes);

      const subNodeIdToPos = buildIdPositionMap(subNodes);
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
        subNodes,
        subIndices,
        subNodeIds,
        internalDeps,
        nodes,
        externalNodeIds,
        enableMultiOutput,
        maxBuffers,
        groups,
      );
    }

    // Phase 2b: Batch independent single-node components into multi-output groups
    if (enableMultiOutput && singletonPositions.length >= 2) {
      const byShapeDtype = new Map<string, number[]>();
      for (const pos of singletonPositions) {
        const node = candidate.nodes[pos];
        const key = shapeDtypeKey(node);
        if (!byShapeDtype.has(key)) byShapeDtype.set(key, []);
        byShapeDtype.get(key)?.push(pos);
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
            addInputKeys(node, seenInputs);
            if (
              seenInputs.size + batchPositions.length + 1 > maxBuffers &&
              batchPositions.length >= 2
            ) {
              break;
            }
            batchPositions.push(sameShapePositions[j]);
          }

          if (batchPositions.length < 2) break;
          batchStart += batchPositions.length;
          const batchNodes = batchPositions.map((p) => candidate.nodes[p]);
          const batchIndices = batchPositions.map((p) => candidate.indices[p]);
          const batchNodeIds = nodeIdSet(batchNodes);
          groups.push({
            nodes: batchNodes,
            planIndices: batchIndices,
            externalInputs: collectExternalInputs(batchNodes, batchNodeIds),
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
function splitGroupsByBufferLimit(
  groups: FusionGroup[],
  maxBuffers: number,
): FusionGroup[] {
  const finalGroups: FusionGroup[] = [];
  for (const group of groups) {
    const numOutputs = 1 + (group.additionalOutputNodes?.length ?? 0);
    const maxInputsForGroup = maxBuffers - numOutputs;
    if (countNonInlinableInputs(group.externalInputs) <= maxInputsForGroup) {
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
  excludedIds: Set<number> | undefined,
  maxBuffers: number,
): void {
  const groupedNodeIds = new Set<number>();
  for (const g of finalGroups) {
    for (const n of g.nodes) groupedNodeIds.add(n.id);
  }

  const singletons: Array<{ node: LazyIRNode; planIndex: number }> = [];
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    if (
      isFusibleOp(node.op) &&
      !groupedNodeIds.has(node.id) &&
      !excludedIds?.has(node.id)
    ) {
      singletons.push({ node, planIndex: i });
    }
  }

  if (singletons.length < 2) return;

  const byShapeDtype = new Map<
    string,
    Array<{ node: LazyIRNode; planIndex: number }>
  >();
  for (const s of singletons) {
    const key = shapeDtypeKey(s.node);
    if (!byShapeDtype.has(key)) byShapeDtype.set(key, []);
    byShapeDtype.get(key)?.push(s);
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
              if (
                input.kind === "pending" &&
                (batchNodeIds.has(input.node.id) || input.node.id === node.id)
              ) {
                orderingSafe = false;
                break;
              }
            }
            if (!orderingSafe) break;
          }
        }
        if (!orderingSafe) break;

        addInputKeys(node, seenInputs);
        if (
          seenInputs.size + batchEntries.length + 1 > maxBuffers &&
          batchEntries.length >= 2
        ) {
          break;
        }

        batchEntries.push(entry);
        batchNodeIds.add(node.id);
        batchMaxPos = candidateMaxPos;
        batchMinPos = candidateMinPos;
      }

      if (batchEntries.length < 2) {
        batchStart = endJ;
        continue;
      }

      batchStart = endJ;

      const batchNodes = batchEntries.map((e) => e.node);
      const batchIndices = batchEntries.map((e) => e.planIndex);
      finalGroups.push({
        nodes: batchNodes,
        planIndices: batchIndices,
        externalInputs: collectExternalInputs(batchNodes, batchNodeIds),
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
        if (!arr) {
          arr = [];
          consumers.set(input.node.id, arr);
        }
        arr.push(planNode.id);
      }
    }
  }

  for (const group of finalGroups) {
    const groupNodeIds = nodeIdSet(group.nodes);
    const outputIds = groupOutputIds(group);

    const neededIntermediates: LazyIRNode[] = [];
    for (const gnode of group.nodes) {
      if (outputIds.has(gnode.id)) continue;
      let isExternal = externalNodeIds?.has(gnode.id) ?? false;
      if (!isExternal) {
        const cons = consumers.get(gnode.id);
        if (cons) {
          for (const cid of cons) {
            if (!groupNodeIds.has(cid)) {
              isExternal = true;
              break;
            }
          }
        }
      }
      if (isExternal) neededIntermediates.push(gnode);
    }

    if (neededIntermediates.length > 0) {
      const primaryShape = group.outputNode.shape;
      const promotable = neededIntermediates.filter((n) =>
        shapesEqual(n.shape, primaryShape),
      );
      const currentOutputs = 1 + (group.additionalOutputNodes?.length ?? 0);
      const availableSlots =
        maxBuffers -
        countNonInlinableInputs(group.externalInputs) -
        currentOutputs;

      if (
        promotable.length === neededIntermediates.length &&
        promotable.length <= availableSlots
      ) {
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
  options?: {
    maxStorageBuffers?: number;
    enableMultiOutput?: boolean;
    excludedIds?: Set<number>;
  },
): FusionDetectionResult {
  const maxBuffers = options?.maxStorageBuffers ?? Infinity;
  const enableMultiOutput = options?.enableMultiOutput ?? false;
  const excludedIds = options?.excludedIds;

  // Phase 1: Build candidate groups of consecutive fusible ops
  const { candidateGroups, fusibleCount } = buildCandidateGroups(
    nodes,
    excludedIds,
  );

  // Phase 2: Split into connected components, process each, batch singletons
  const groups = splitCandidatesByComponent(
    candidateGroups,
    nodes,
    externalNodeIds,
    enableMultiOutput,
    maxBuffers,
  );

  // Phase 3: Split groups exceeding storage buffer limit
  const finalGroups = splitGroupsByBufferLimit(groups, maxBuffers);

  // Phase 4: Global singleton batching
  if (enableMultiOutput) {
    batchGlobalSingletons(nodes, finalGroups, excludedIds, maxBuffers);
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
    for (const n of group.additionalOutputNodes)
      originalAdditionalIds.add(n.id);
  }

  let subNodes: LazyIRNode[] = [];
  let subIndices: number[] = [];
  let subNodeIds = new Set<number>();
  let seenExternals = new Set<string>();

  const flushSubGroup = () => {
    if (subNodes.length >= 2) {
      const externalInputs = collectExternalInputs(subNodes, subNodeIds);
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
        additionalOutputNodes:
          additionalOutputNodes.length > 0 ? additionalOutputNodes : undefined,
      });
    }
    subNodes = [];
    subIndices = [];
    subNodeIds = new Set();
    seenExternals = new Set();
  };

  for (let k = 0; k < group.nodes.length; k++) {
    const node = group.nodes[k];
    const planIdx = group.planIndices[k];

    const newExternalCount = countNewExternals(
      node,
      subNodeIds,
      seenExternals,
      false,
    );
    if (
      seenExternals.size + newExternalCount > maxExternalInputs &&
      subNodes.length >= 2
    ) {
      flushSubGroup();
    }

    subNodes.push(node);
    subIndices.push(planIdx);
    subNodeIds.add(node.id);
    countNewExternals(node, subNodeIds, seenExternals, true);
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
      const earlierNodeIds = nodeIdSet(earlier.nodes);
      const earlierOutputIds = groupOutputIds(earlier);

      // Find nodes in the later group that depend on intermediates of the earlier group
      for (const laterNode of later.nodes) {
        for (const input of laterNode.inputs) {
          if (input.kind !== "pending") continue;
          const depId = input.node.id;
          if (!earlierNodeIds.has(depId)) continue;
          if (earlierOutputIds.has(depId)) continue;

          // depId is an intermediate of the earlier group consumed by the later group
          // Check if it can be promoted as an additional output
          const depNode = earlier.nodes.find((n) => n.id === depId);
          if (!depNode) continue;

          if (shapesEqual(depNode.shape, earlier.outputNode.shape)) {
            // Check binding limit: externalInputs + currentOutputs + 1 <= maxExternalInputs + currentOutputs
            const currentOutputs =
              1 + (earlier.additionalOutputNodes?.length ?? 0);
            const availableSlots =
              maxExternalInputs +
              currentOutputs -
              earlier.externalInputs.length -
              currentOutputs;
            if (availableSlots > 0) {
              if (!earlier.additionalOutputNodes)
                earlier.additionalOutputNodes = [];
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
 * - Scalar refs (inlined as constants, don't consume bindings)
 *
 * Uses prefixed string keys for dedup (s: storage, p: pending, v: scalar)
 * to avoid ID namespace collisions between storage.id and node.id.
 */
export function collectExternalInputs(
  groupNodes: LazyIRNode[],
  groupNodeIds: Set<number>,
): LazyRef[] {
  const external: LazyRef[] = [];
  const seen = new Set<string>();

  for (const node of groupNodes) {
    for (const input of node.inputs) {
      if (input.kind === "pending" && groupNodeIds.has(input.node.id)) continue;
      const key =
        input.kind === "scalar"
          ? `v:${input.value}_${input.dtype}`
          : inputKey(input)!;
      if (!seen.has(key)) {
        seen.add(key);
        external.push(input);
      }
    }
  }

  return external;
}

/**
 * Convert a fusion group to a FusedKernelRecipe for codegen.
 *
 * @param group - The detected fusion group
 * @returns Recipe suitable for fusion-tile-ir
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
      // Check if we've already registered this exact scalar
      for (const [, existingIdx] of inputMap) {
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
      return inputMap.get(id) as number;
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
        dtype: ensureDType(ref.storage.backendTensor.dtype),
      });
    } else {
      // Check if this pending ref is an inlinable scalar constant
      const inlineCheck = isInlinableScalar(ref);
      const inputEntry: FusedInput = {
        id: ref.node.id,
        index: idx,
        shape: ref.node.shape ?? [1],
        dtype: ensureDType(ref.node.dtype),
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
      fusedOp = `cast_${node.dtype}` as typeof fusedOp;
    }

    // Check if node is any output (primary or additional)
    const isAdditionalOutput =
      group.additionalOutputNodes?.some((n) => n.id === node.id) ?? false;
    fusedNodes.push({
      id: node.id,
      op: fusedOp,
      inputs: mappedInputs,
      shape: node.shape ?? [1],
      dtype: ensureDType(node.dtype),
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
      dtype: ensureDType(outputNode.dtype),
    },
  ];

  if (group.additionalOutputNodes) {
    for (let i = 0; i < group.additionalOutputNodes.length; i++) {
      const addNode = group.additionalOutputNodes[i];
      outputs.push({
        nodeId: addNode.id,
        index: i + 1,
        shape: addNode.shape ?? [1],
        dtype: ensureDType(addNode.dtype),
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
 * Execution plan segment - fusible, sequential, or reduction.
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
  options?: {
    maxStorageBuffers?: number;
    enableMultiOutput?: boolean;
    excludedIds?: Set<number>;
  },
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

  // Build node ID → group for direct group membership check
  const nodeIdToGroup = new Map<number, FusionGroup>();
  for (const group of groups) {
    for (const node of group.nodes) {
      nodeIdToGroup.set(node.id, group);
    }
  }

  // Track which groups have been emitted.
  const emittedGroups = new Set<FusionGroup>();

  // Pre-compute max plan index for each group so we can emit at the last
  // member's position. Gap-dependency check below handles cases where gap
  // nodes depend on group outputs.
  const groupMaxIdx = new Map<FusionGroup, number>();
  for (const group of groups) {
    groupMaxIdx.set(group, Math.max(...group.planIndices));
  }

  /** Check if index i belongs to any group. */
  function isGroupMember(idx: number): boolean {
    return indexToGroup.has(idx);
  }

  let i = 0;
  while (i < nodes.length) {
    const group = indexToGroup.get(i);

    if (group) {
      if (!emittedGroups.has(group) && i === groupMaxIdx.get(group)) {
        emittedGroups.add(group);
        const recipe = groupToRecipe(group);
        segments.push({ kind: "fused", group, recipe });
      }
      i++;
    } else {
      // Sequential execution - collect consecutive non-grouped nodes.
      // Before adding a node, check if it depends on a not-yet-emitted
      // fusion group. If so, flush that group first so its outputs are
      // available. This handles "gap nodes" between non-contiguous group
      // members that depend on earlier group members.
      const seqNodes: LazyIRNode[] = [];
      while (i < nodes.length && !isGroupMember(i)) {
        const node = nodes[i];
        // Check if this gap node (or any ancestor) depends on a
        // not-yet-emitted fusion group. Walk the dependency chain.
        const visited = new Set<number>();
        const stack = [node];
        while (stack.length > 0) {
          const cur = stack.pop()!;
          if (visited.has(cur.id)) continue;
          visited.add(cur.id);
          for (const inp of cur.inputs) {
            if (inp.kind !== "pending") continue;
            const depGroup = nodeIdToGroup.get(inp.node.id);
            if (depGroup && !emittedGroups.has(depGroup)) {
              if (seqNodes.length > 0) {
                segments.push({ kind: "sequential", nodes: [...seqNodes] });
                seqNodes.length = 0;
              }
              emittedGroups.add(depGroup);
              segments.push({ kind: "fused", group: depGroup, recipe: groupToRecipe(depGroup) });
            } else if (!depGroup && !inp.node.result) {
              // Non-group, non-materialized ancestor — keep walking
              stack.push(inp.node);
            }
          }
        }
        seqNodes.push(node);
        i++;
      }
      if (seqNodes.length > 0) {
        segments.push({ kind: "sequential", nodes: seqNodes });
      }
    }
  }

  return segments;
}

/** Stable key for a non-scalar LazyRef (for input dedup tracking). */
function inputKey(input: LazyRef): string | null {
  if (input.kind === "scalar") return null;
  return input.kind === "materialized"
    ? `s:${input.storage.id}`
    : `p:${input.node.id}`;
}

/** Add a node's non-scalar input keys to a tracking set and return the count of newly added keys. */
function addInputKeys(node: LazyIRNode, seenInputs: Set<string>): number {
  let added = 0;
  for (const input of node.inputs) {
    const k = inputKey(input);
    if (k !== null && !seenInputs.has(k)) {
      seenInputs.add(k);
      added++;
    }
  }
  return added;
}

/**
 * Count (and optionally track) non-inlinable external inputs a node would introduce.
 * Skips scalars and inlinable pending refs that won't consume binding slots.
 * When `track` is true, adds newly-seen keys to `seen` for incremental tracking.
 */
function countNewExternals(
  node: LazyIRNode,
  groupNodeIds: Set<number>,
  seen: Set<string>,
  track: boolean,
): number {
  let count = 0;
  for (const input of node.inputs) {
    if (input.kind === "scalar") continue;
    if (input.kind === "pending") {
      if (groupNodeIds.has(input.node.id)) continue;
      if (isInlinableScalar(input).inlinable) continue;
    }
    const key = inputKey(input)!;
    if (!seen.has(key)) {
      count++;
      if (track) seen.add(key);
    }
  }
  return count;
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
function isInlinableScalar(
  ref: LazyRef,
): { inlinable: true; value: number } | { inlinable: false } {
  // Scalar refs are trivially inlinable
  if (ref.kind === "scalar") {
    return { inlinable: true, value: ref.value };
  }
  if (ref.kind !== "pending") return { inlinable: false };
  const node = ref.node;
  const totalElements = (node.shape ?? [1]).reduce(
    (a: number, b: number) => a * b,
    1,
  );
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
    const node = nodeById.get(id) as LazyIRNode;
    const fusible = isFusibleOp(node.op);
    let priority: number;

    if (
      fusible &&
      chainNodeIds.size > 0 &&
      hasPendingInputIn(node, chainNodeIds)
    ) {
      priority = 0; // Chain continuation
    } else if (fusible) {
      priority = 1; // New chain
    } else {
      priority = 2; // Non-fusible
    }

    const pos = originalPos.get(id) as number;
    if (
      priority < bestPriority ||
      (priority === bestPriority && pos < bestPos)
    ) {
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
        inDegree.set(node.id, (inDegree.get(node.id) as number) + 1);
        successors.get(input.node.id)?.push(node.id);
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
    const best = selectBestForFusion(
      ready,
      chainNodeIds,
      nodeById,
      originalPos,
    );
    ready.delete(best);
    const bestNode = nodeById.get(best) as LazyIRNode;
    result.push(bestNode);
    if (isFusibleOp(bestNode.op)) {
      chainNodeIds.add(best);
    } else {
      chainNodeIds = new Set();
    }

    for (const succId of successors.get(best) as number[]) {
      const newDeg = (inDegree.get(succId) as number) - 1;
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
      hashByte(0xee);
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
