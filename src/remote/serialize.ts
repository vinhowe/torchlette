/**
 * Plan serialization / deserialization for remote execution.
 *
 * Pure functions over graph types. No side effects, no backend coupling.
 */

import type { DType } from "../backend/types";
import type { ExecutionPlan, LazyIRNode, LazyRef } from "../graph/types";
import type {
  HandleRef,
  InlineTensorBytes,
  SerializedNode,
  SerializedPlan,
  SerializedRef,
} from "./wire";

/**
 * Function that maps a server-resident storage id to a HandleRef the remote
 * peer can understand. Injected by the caller so serialization stays pure.
 *
 * In a real client, this consults a local `Map<storageId, HandleRef>` maintained
 * by the remote engine. In the round-trip test, we pass an identity-ish function.
 */
export type StorageIdToHandle = (storageId: number) => HandleRef;

/**
 * Inverse: look up a StorageHandle given the HandleRef the peer sent back.
 * Used during deserialization on the receiving side.
 */
export type HandleToStorage = (
  handle: HandleRef,
) => import("../graph/types").StorageHandle;

export interface SerializeOptions {
  /** Map a local materialized StorageHandle.id to a HandleRef the peer understands. */
  resolveHandle: StorageIdToHandle;
}

export interface DeserializeOptions {
  /**
   * Resolve a HandleRef into a local StorageHandle. Required if the incoming
   * plan contains any materialized refs. May be undefined for self-contained
   * plans (all tensorFromArray / creation nodes).
   */
  resolveHandle?: HandleToStorage;
  /**
   * Allocate a fresh node id. Defaults to a local counter.
   * Allows the receiver to integrate node ids with its own lifetime tracking.
   */
  allocNodeId?: () => number;
}

// ============================================================================
// Serialize
// ============================================================================

export function serializePlan(
  plan: ExecutionPlan,
  options: SerializeOptions,
): SerializedPlan {
  // Build a node → idx map. plan.nodes is already in topological order.
  const nodeToIdx = new Map<LazyIRNode, number>();
  for (let i = 0; i < plan.nodes.length; i++) {
    nodeToIdx.set(plan.nodes[i], i);
  }

  const externalHandles = new Set<HandleRef>();
  const serializedNodes: SerializedNode[] = plan.nodes.map((node, idx) => {
    const inputs = node.inputs.map((ref) =>
      serializeRef(ref, nodeToIdx, options.resolveHandle, externalHandles),
    );
    const out: SerializedNode = {
      idx,
      op: node.op,
      inputs,
      shape: node.shape.slice(),
      dtype: node.dtype,
      device: node.device,
      payload:
        node.payload === undefined
          ? undefined
          : encodePayload(node.payload),
    };
    if (node.module !== undefined) out.module = node.module;
    if (node.isCheckpointBoundary) out.isCheckpointBoundary = true;
    return out;
  });

  return {
    version: 1,
    nodes: serializedNodes,
    externalHandles: [...externalHandles],
    outputNodes: plan.outputIndices,
  };
}

function serializeRef(
  ref: LazyRef,
  nodeToIdx: Map<LazyIRNode, number>,
  resolveHandle: StorageIdToHandle,
  externalHandles: Set<HandleRef>,
): SerializedRef {
  if (ref.kind === "scalar") {
    return { kind: "scalar", value: ref.value, dtype: ref.dtype };
  }
  if (ref.kind === "materialized") {
    const handle = resolveHandle(ref.storage.id);
    externalHandles.add(handle);
    return { kind: "materialized", handle };
  }
  // pending
  const idx = nodeToIdx.get(ref.node);
  if (idx === undefined) {
    // Node was already executed (has a result) but excluded from the plan
    // by buildMergedPlan's skipExecuted optimization. Treat as materialized.
    if (ref.node.result) {
      const handle = resolveHandle(ref.node.result.id);
      externalHandles.add(handle);
      return { kind: "materialized", handle };
    }
    throw new Error(
      `serializePlan: pending ref points to node id=${ref.node.id} op=${ref.node.op} not in plan`,
    );
  }
  const out: SerializedRef = { kind: "pending", nodeIdx: idx };
  if (ref.outputIndex !== undefined && ref.outputIndex !== 0) {
    out.outputIndex = ref.outputIndex;
  }
  return out;
}

// ============================================================================
// Deserialize
// ============================================================================

export function deserializePlan(
  wire: SerializedPlan,
  options: DeserializeOptions = {},
): ExecutionPlan {
  if (wire.version !== 1) {
    throw new Error(`deserializePlan: unsupported version ${wire.version}`);
  }
  const allocId = options.allocNodeId ?? makeDefaultIdAllocator();
  const builtNodes: LazyIRNode[] = [];

  for (let i = 0; i < wire.nodes.length; i++) {
    const wn = wire.nodes[i];
    if (wn.idx !== i) {
      throw new Error(
        `deserializePlan: node at position ${i} has idx=${wn.idx} (expected dense ordering)`,
      );
    }
    const inputs = wn.inputs.map((ref) =>
      deserializeRef(ref, builtNodes, options.resolveHandle),
    );
    const node: LazyIRNode = {
      id: allocId(),
      op: wn.op,
      inputs,
      shape: wn.shape.slice(),
      dtype: wn.dtype,
      device: wn.device,
      payload: wn.payload === undefined ? undefined : decodePayload(wn.payload),
    };
    if (wn.module !== undefined) node.module = wn.module;
    if (wn.isCheckpointBoundary) node.isCheckpointBoundary = true;
    builtNodes.push(node);
  }

  return { nodes: builtNodes, outputIndices: wire.outputNodes };
}

function deserializeRef(
  ref: SerializedRef,
  built: LazyIRNode[],
  resolveHandle: HandleToStorage | undefined,
): LazyRef {
  if (ref.kind === "scalar") {
    return { kind: "scalar", value: ref.value, dtype: ref.dtype };
  }
  if (ref.kind === "materialized") {
    if (!resolveHandle) {
      throw new Error(
        `deserializePlan: plan references handle ${ref.handle} but no resolveHandle was provided`,
      );
    }
    return { kind: "materialized", storage: resolveHandle(ref.handle) };
  }
  // pending
  const node = built[ref.nodeIdx];
  if (!node) {
    throw new Error(
      `deserializePlan: pending ref targets nodeIdx=${ref.nodeIdx} but only ${built.length} nodes have been built`,
    );
  }
  return ref.outputIndex !== undefined && ref.outputIndex !== 0
    ? { kind: "pending", node, outputIndex: ref.outputIndex }
    : { kind: "pending", node };
}

// ============================================================================
// Payload encoding (Float32Array support)
// ============================================================================

/** Walks a payload, replacing Float32Array with InlineTensorBytes wrappers. */
function encodePayload(payload: unknown): unknown {
  return mapDeep(payload, (v) => {
    if (v instanceof Float32Array) {
      return {
        __inlineTensor: true,
        dtype: "f32" as DType,
        values: Array.from(v),
      } satisfies InlineTensorBytes;
    }
    if (
      v instanceof Int32Array ||
      v instanceof Uint32Array ||
      v instanceof Uint8Array
    ) {
      const dtype: DType =
        v instanceof Int32Array
          ? "i32"
          : v instanceof Uint32Array
            ? "u32"
            : "bool";
      return {
        __inlineTensor: true,
        dtype,
        values: Array.from(v),
      } satisfies InlineTensorBytes;
    }
    return undefined; // no transform; continue walking
  });
}

/** Inverse of encodePayload: rebuild typed arrays from InlineTensorBytes wrappers. */
function decodePayload(payload: unknown): unknown {
  return mapDeep(payload, (v) => {
    if (
      v !== null &&
      typeof v === "object" &&
      (v as InlineTensorBytes).__inlineTensor === true
    ) {
      const { dtype, values } = v as InlineTensorBytes;
      if (dtype === "f32") return new Float32Array(values);
      if (dtype === "i32") return new Int32Array(values);
      if (dtype === "u32") return new Uint32Array(values);
      if (dtype === "bool") return new Uint8Array(values);
      // f16 has no JS typed array; carry as Float32Array by convention.
      return new Float32Array(values);
    }
    return undefined;
  });
}

/**
 * Structural deep-map. Visits nested objects/arrays. If `transform` returns
 * a non-undefined value, that value replaces the node (and we stop
 * recursing into it). Returns a new object tree; does not mutate the input.
 *
 * Typed arrays are opaque to this walker except via the transform hook.
 */
function mapDeep(value: unknown, transform: (v: unknown) => unknown): unknown {
  const replaced = transform(value);
  if (replaced !== undefined) return replaced;
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map((v) => mapDeep(v, transform));
  // Leave typed arrays (not handled by transform) untouched. JSON-safety is
  // the caller's responsibility if a payload carries one we don't recognize.
  if (ArrayBuffer.isView(value)) return value;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    out[k] = mapDeep(v, transform);
  }
  return out;
}

// ============================================================================
// Default id allocator for deserialized nodes
// ============================================================================

function makeDefaultIdAllocator(): () => number {
  // Start at a high value to avoid colliding with nodes created locally
  // before deserialize is called. Not strictly required — the receiver
  // should pass its own allocator integrated with its node-id counter.
  let next = 1_000_000_000;
  return () => next++;
}
