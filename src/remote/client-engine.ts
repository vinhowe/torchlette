/**
 * Client-side remote engine: wraps a Torchlette whose runtime is patched so
 * every plan execution and readback goes through a Transport instead of a
 * local backend. User code calls `api.matmul(a, b)`, `loss.backward()`,
 * `tensor.item()` etc. exactly like local Torchlette; the plans build up
 * locally, then get shipped as SerializedPlan JSON to whatever implements
 * the Transport.
 *
 * Handle bridging: the server issues opaque HandleRef strings for each
 * output; the client allocates a local StorageHandle (with a stub
 * BackendTensor that never computes) and maps its id → HandleRef in
 * ClientHandleMap. Subsequent plans that reference the tensor become
 * `materialized` refs whose storage id is translated back to a HandleRef
 * during serialization.
 */

import { cpuBackend } from "../backend/cpu";
import { registerBackend } from "../backend/registry";
import type { BackendTensor, DType } from "../backend/types";
import { buildMergedPlan } from "../executor/plan-builder";
import type { Tensor as FrontendTensor } from "../frontend/tensor";
import { Torchlette } from "../frontend/torchlette";
import { createStorageHandle, releaseNodeInputRefs, retainPlanInputRefs } from "../graph/node-factory";
import { storageTracker } from "../graph/storage-tracker";
import type { LazyIRNode, StorageHandle } from "../graph/types";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import {
  clearDisposedPendingNodeIds,
  getAllPendingTensors,
  materializePendingTensors,
} from "../runtime/tensor";
import type {
  DownloadParams,
  DownloadResult,
  ExecuteParams,
  ExecuteResult,
  ReadScalarParams,
  ReadScalarResult,
  ReleaseParams,
  ReleaseResult,
  UploadParams,
  UploadResult,
} from "./rpc";
import { serializePlan } from "./serialize";
import type { HandleRef } from "./wire";

// ============================================================================
// Transport
// ============================================================================

/**
 * A Transport speaks the five remote-training RPC methods. Implementations
 * include: a WebSocket client (browser/Node), an HTTP/2 client, an
 * in-process fake for tests, or anything else that can move plans and bytes
 * between the caller and a handle-registry owner.
 */
export interface Transport {
  execute(params: ExecuteParams): Promise<ExecuteResult>;
  upload(params: UploadParams): Promise<UploadResult>;
  download(params: DownloadParams): Promise<DownloadResult>;
  readScalar(params: ReadScalarParams): Promise<ReadScalarResult>;
  release(params: ReleaseParams): Promise<ReleaseResult>;
}

// ============================================================================
// Client-side handle registry
// ============================================================================

/** Maps local StorageHandle.id ↔ server-issued HandleRef. */
export class ClientHandleMap {
  private readonly idToHandle = new Map<number, HandleRef>();
  private readonly handleToId = new Map<HandleRef, number>();

  bind(storageId: number, handle: HandleRef): void {
    this.idToHandle.set(storageId, handle);
    this.handleToId.set(handle, storageId);
  }

  getHandle(storageId: number): HandleRef | undefined {
    return this.idToHandle.get(storageId);
  }

  requireHandle(storageId: number): HandleRef {
    const h = this.idToHandle.get(storageId);
    if (!h) {
      throw new Error(
        `ClientHandleMap: no HandleRef for storage id=${storageId}. ` +
          `This storage was not produced by a prior remote execute.`,
      );
    }
    return h;
  }

  unbind(storageId: number): HandleRef | undefined {
    const h = this.idToHandle.get(storageId);
    if (h) {
      this.idToHandle.delete(storageId);
      this.handleToId.delete(h);
    }
    return h;
  }

  size(): number {
    return this.idToHandle.size;
  }

  entries(): IterableIterator<[number, HandleRef]> {
    return this.idToHandle.entries();
  }
}

// ============================================================================
// Stub backend tensors — client never executes against them.
// ============================================================================

function makeStub(shape: number[], dtype: DType): BackendTensor {
  return {
    shape,
    dtype,
    ownsBuffer: false,
    toArray(): number[] {
      throw new Error(
        "Stub backendTensor: data lives on the server. Use async " +
          "tensor.cpu() / tensor.item() instead of .toArray().",
      );
    },
  };
}

// ============================================================================
// RemoteEngine
// ============================================================================

export interface RemoteEngineStats {
  executes: number;
  nodesShipped: number;
  downloads: number;
  scalarReads: number;
  bytesUp: number;
  bytesDown: number;
  handlesReleased: number;
  /** Per-phase cumulative timing (ms). */
  serializeMs: number;
  transportMs: number;
  bookkeepingMs: number;
}

export interface RemoteEngine {
  /** Torchlette frontend — user code calls api.matmul, loss.backward, etc. */
  torch: Torchlette;
  /** Client-side mapping of local storage ids to server HandleRefs. */
  handles: ClientHandleMap;
  /** Transport used to ship plans and read results. */
  transport: Transport;
  /** Cumulative stats across the session. */
  stats: RemoteEngineStats;
  /**
   * Release server-side handles not in the `keep` set. Call at end of
   * each training step. Pass all tensors that persist across steps:
   * `[...optimizer.getAllKeepTensors(), ...model.persistentTensors()]`.
   */
  markStep(keep: FrontendTensor[]): Promise<number>;
  /**
   * Pre-upload pending `tensorFromArray` tensors via binary transport,
   * bypassing JSON plan serialization. Call once before the first training
   * step to eliminate the cold-start payload (~40-60MB of inline weights
   * for a large model → ~100KB of handle refs in subsequent plans).
   *
   * For each tensor: extracts the Float32Array from its pending node's
   * payload, uploads via `transport.upload()` (binary frame), and patches
   * the node to look "already executed" so `buildMergedPlan` skips it.
   */
  preUpload(tensors: FrontendTensor[]): Promise<number>;
}

export interface CreateRemoteEngineOptions {
  /**
   * Device tag used on the client-side Torchlette. This only affects how
   * the client labels nodes — the server's backend does the actual compute.
   * Default: "cpu".
   */
  device?: "cpu" | "webgpu";
}

/**
 * Create a Torchlette instance whose runtime executes every plan through
 * the given Transport. The client does no computation locally; it only
 * builds autograd graphs and serializes them.
 */
export function createRemoteEngine(
  transport: Transport,
  options: CreateRemoteEngineOptions = {},
): RemoteEngine {
  // The client's Torchlette needs *some* registered backend to satisfy
  // initialization; we register CPU as a no-op target. All force methods
  // below are overridden, so the backend is never actually invoked.
  registerBackend(cpuBackend);

  const torch = new Torchlette(options.device ?? "cpu");
  const handles = new ClientHandleMap();
  // Track storage ID → node for clearing stale results on handle release.
  const storageToNode = new Map<number, LazyIRNode>();
  // Node IDs that failed serialization (stale local refs). Excluded from
  // future forceAllPending plans to prevent unbounded plan growth.
  const failedNodeIds = new Set<number>();
  const stats: RemoteEngineStats = {
    executes: 0,
    nodesShipped: 0,
    downloads: 0,
    scalarReads: 0,
    bytesUp: 0,
    bytesDown: 0,
    handlesReleased: 0,
    serializeMs: 0,
    transportMs: 0,
    bookkeepingMs: 0,
  };

  const runtime = torch.runtime;

  /** Ship one plan to the server, patching outputs as local stub storages. */
  async function shipPlan(plan: { nodes: LazyIRNode[] }): Promise<void> {
    if (plan.nodes.length === 0) return;

    const t0 = performance.now();
    const wire = serializePlan(plan, {
      resolveHandle: (id: number) => handles.requireHandle(id),
    });
    stats.executes++;
    stats.nodesShipped += plan.nodes.length;
    stats.bytesUp += JSON.stringify(wire).length;
    const t1 = performance.now();
    stats.serializeMs += t1 - t0;

    const result = await transport.execute({ plan: wire });
    const t2 = performance.now();
    stats.transportMs += t2 - t1;

    // Bind each server HandleRef to a fresh local stub StorageHandle so
    // subsequent plans can reference these outputs as materialized inputs.
    for (const [idxStr, handleRef] of Object.entries(result.outputs)) {
      const idx = Number(idxStr);
      const node = plan.nodes[idx];
      // If the node had a previous result (e.g., local CPU storage from
      // model init), bind that old storage ID to the same server handle.
      // Other nodes may hold materialized refs to the old storage.
      if (node.result && !handles.getHandle(node.result.id)) {
        handles.bind(node.result.id, handleRef);
      }
      const stub = makeStub(node.shape, node.dtype);
      const storage = createStorageHandle(node.device, stub);
      handles.bind(storage.id, handleRef);
      storageToNode.set(storage.id, node);
      node.result = storage;
    }
    // Bind multi-output side results (adamStep m/v, fusedAttention lse/rng, etc.)
    if (result.sideOutputs) {
      for (const [key, handleRef] of Object.entries(result.sideOutputs)) {
        const [idxStr, outIdxStr] = key.split(":");
        const node = plan.nodes[Number(idxStr)];
        const outIdx = Number(outIdxStr);
        const stub = makeStub(node.shape, node.dtype);
        const storage = createStorageHandle(node.device, stub);
        handles.bind(storage.id, handleRef);
        if (!node.results) node.results = [];
        node.results[outIdx] = storage;
      }
    }
    stats.bookkeepingMs += performance.now() - t2;
  }

  function collectPendingRoots(
    tensors: readonly RuntimeTensor[],
  ): LazyIRNode[] {
    const roots: LazyIRNode[] = [];
    for (const t of tensors) {
      if (t.isMaterialized() || t.disposed) continue;
      const ref = t.lazyRef;
      if (ref.kind === "pending") roots.push(ref.node);
    }
    return roots;
  }

  function materializeRemaining(tensors: readonly RuntimeTensor[]): void {
    for (const t of tensors) {
      if (t.isMaterialized() || t.disposed) continue;
      const ref = t.lazyRef;
      if (ref.kind === "pending") {
        const idx = ref.outputIndex ?? 0;
        const storage = idx === 0 ? ref.node.result : ref.node.results?.[idx];
        if (storage) t._materialize(storage);
      }
    }
  }

  function postExecuteBookkeeping(
    plan: { nodes: LazyIRNode[] },
    tensors: readonly RuntimeTensor[],
  ): void {
    for (const node of plan.nodes) {
      if (node.result) {
        materializePendingTensors(node.id, node.result, node.results);
      }
    }
    clearDisposedPendingNodeIds();
    materializeRemaining(tensors);
  }

  // Override runtime force methods. These are not normally callable from
  // outside the engine, but they're the interception points where
  // autograd + user code delegate to the backend.

  // biome-ignore lint/suspicious/noExplicitAny: overriding private methods
  (runtime as any).forceAllMerged = async (
    ...tensors: RuntimeTensor[]
  ): Promise<void> => {
    const roots = collectPendingRoots(tensors);
    if (roots.length === 0) return;
    const plan = buildMergedPlan(roots);
    if (plan.nodes.length === 0) return;
    retainPlanInputRefs(plan.nodes);
    await shipPlan(plan);
    postExecuteBookkeeping(plan, tensors);
    for (const node of plan.nodes) {
      releaseNodeInputRefs(node);
      node.result = undefined;
    }
  };

  // biome-ignore lint/suspicious/noExplicitAny: overriding private methods
  (runtime as any).force = async (tensor: RuntimeTensor): Promise<void> => {
    if (tensor.isMaterialized() || tensor.disposed) return;
    if (tensor.lazyRef.kind !== "pending") return;
    // biome-ignore lint/suspicious/noExplicitAny: reusing patched method
    await (runtime as any).forceAllMerged(tensor);
  };

  // forceAllPending is NOT overridden — the engine's built-in version
  // handles the full lifecycle (retain, execute via hook, materialize,
  // release, clear). We just need the execution hook to be set.
  (runtime as any)._executionHook = async (
    plan: { nodes: LazyIRNode[] },
  ): Promise<void> => {
    if (plan.nodes.length === 0) return;
    await shipPlan(plan);
  };

  // biome-ignore lint/suspicious/noExplicitAny: overriding public methods
  (runtime as any).cpu = async (a: RuntimeTensor): Promise<number[]> => {
    // biome-ignore lint/suspicious/noExplicitAny: calling patched method
    await (runtime as any).force(a);
    const storage = (a as unknown as { lazyRef: { storage: StorageHandle } })
      .lazyRef.storage;
    const handle = handles.requireHandle(storage.id);
    stats.downloads++;
    const result = await transport.download({ handle });
    stats.bytesDown += result.values.length * 4; // rough: f32
    return result.values;
  };

  // biome-ignore lint/suspicious/noExplicitAny: overriding public methods
  (runtime as any).item = async (a: RuntimeTensor): Promise<number> => {
    // biome-ignore lint/suspicious/noExplicitAny: calling patched method
    await (runtime as any).force(a);
    const storage = (a as unknown as { lazyRef: { storage: StorageHandle } })
      .lazyRef.storage;
    const handle = handles.requireHandle(storage.id);
    stats.scalarReads++;
    const result = await transport.readScalar({ handle });
    return result.value;
  };

  async function markStep(keep: FrontendTensor[]): Promise<number> {
    // Collect storage IDs to keep: explicit keep list + anything
    // referenced by a materialized ref in the live lazy graph.
    const keepIds = new Set<number>();

    // 1. Keep tensors from the explicit keep list.
    for (const t of keep) {
      const ref = t._unwrap().lazyRef;
      if (ref.kind === "materialized") {
        keepIds.add(ref.storage.id);
      }
    }

    // 2. Scan ALL live pending tensors' node chains for materialized
    //    input refs. These storages are frozen in the lazy graph and
    //    must not be released — the serializer would fail.
    const pending = getAllPendingTensors();
    const visited = new Set<LazyIRNode>();
    const scanNode = (node: LazyIRNode) => {
      if (visited.has(node)) return;
      visited.add(node);
      for (const ref of node.inputs) {
        if (ref.kind === "materialized") {
          keepIds.add(ref.storage.id);
        } else if (ref.kind === "pending") {
          scanNode(ref.node);
        }
      }
    };
    for (const rt of pending) {
      if (rt.disposed) continue;
      const ref = rt.lazyRef;
      if (ref.kind === "pending") scanNode(ref.node);
    }

    const toRelease: HandleRef[] = [];
    const idsToUnbind: number[] = [];
    for (const [id, handle] of handles.entries()) {
      if (!keepIds.has(id)) {
        toRelease.push(handle);
        idsToUnbind.push(id);
      }
    }

    if (toRelease.length > 0) {
      await transport.release({ handles: toRelease });
      for (const id of idsToUnbind) {
        handles.unbind(id);
        storageTracker.unregister(id);
        // Clear stale node.result so the lazy graph doesn't hold
        // materialized refs to released storages. The node will be
        // re-executed from scratch in the next plan.
        const node = storageToNode.get(id);
        if (node && node.result?.id === id) {
          node.result = undefined;
        }
        storageToNode.delete(id);
      }
      stats.handlesReleased += toRelease.length;
    }
    return toRelease.length;
  }

  async function preUpload(tensors: FrontendTensor[]): Promise<number> {
    let uploaded = 0;
    for (const t of tensors) {
      const rt = t._unwrap();
      const ref = rt.lazyRef;

      // Already has a server handle — skip.
      if (ref.kind === "materialized" && handles.getHandle(ref.storage.id)) {
        continue;
      }

      // Get the data to upload. For pending tensorFromArray nodes, read
      // from the payload. For already-materialized CPU tensors, read from
      // the backend tensor.
      let values: number[];
      let shape: number[];
      let dtype: string;

      if (ref.kind === "pending") {
        const node = ref.node;
        if (node.op !== "tensorFromArray") continue;
        const payload = node.payload as
          | { values: Float32Array | Int32Array | Uint32Array | number[] }
          | undefined;
        if (!payload?.values) continue;
        values = Array.isArray(payload.values)
          ? payload.values
          : Array.from(payload.values);
        shape = node.shape;
        dtype = node.dtype;
      } else if (ref.kind === "materialized") {
        // CPU-materialized tensor — read data back from the backend tensor.
        const bt = ref.storage.backendTensor;
        values = bt.toArray();
        shape = bt.shape;
        dtype = bt.dtype;
      } else {
        continue;
      }

      const result = await transport.upload({
        values,
        shape,
        dtype,
      });

      // Bind the server handle to BOTH the old storage ID (so materialized
      // refs from other nodes resolve) and a new stub storage.
      if (ref.kind === "materialized") {
        handles.bind(ref.storage.id, result.handle);
      }

      if (ref.kind === "pending") {
        const node = ref.node;
        const stub = makeStub(shape, dtype as any);
        const storage = createStorageHandle(node.device, stub);
        handles.bind(storage.id, result.handle);
        node.result = storage;
        rt._materialize(storage);
      }

      uploaded++;
    }
    return uploaded;
  }

  return { torch, handles, transport, stats, markStep, preUpload };
}
