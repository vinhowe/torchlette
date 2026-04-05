/**
 * Client-side RemoteRuntimeEngine setup. Wraps a local Torchlette whose
 * runtime is patched to route every plan execution + readback through the
 * RpcClient. Everything computes server-side; the client only builds plans.
 *
 * Key mechanism: client maintains Map<localStorageId, HandleRef>. When
 * serializing a plan, materialized refs are translated via this map. When
 * `execute` returns HandleRefs for output nodes, we allocate fresh local
 * stub StorageHandles and register them in the map — subsequent plans can
 * reference them as materialized inputs.
 */

import { cpuBackend } from "../../../src/backend/cpu/index.ts";
import { registerBackend } from "../../../src/backend/registry.ts";
import type { BackendTensor } from "../../../src/backend/types.ts";
import { buildMergedPlan } from "../../../src/executor/plan-builder.ts";
import type { Tensor as FrontendTensor } from "../../../src/frontend/tensor.ts";
import { Torchlette } from "../../../src/frontend/torchlette.ts";
import { createStorageHandle } from "../../../src/graph/node-factory.ts";
import { storageTracker } from "../../../src/graph/storage-tracker.ts";
import type { LazyIRNode, StorageHandle } from "../../../src/graph/types.ts";
import type { Tensor as RuntimeTensor } from "../../../src/runtime/tensor.ts";
import {
  clearDisposedPendingNodeIds,
  getAllPendingTensors,
  materializePendingTensors,
} from "../../../src/runtime/tensor.ts";
import { serializePlan } from "../../../src/remote/serialize.ts";
import type { HandleRef, NodeIdx } from "../../../src/remote/wire.ts";
import type { RpcClient } from "./transport.ts";

registerBackend(cpuBackend);

// ============================================================================
// Stub backend tensors — client never computes, never reads from these.
// ============================================================================

function makeStub(shape: number[], dtype: import("../../../src/backend/types.ts").DType): BackendTensor {
  return {
    shape,
    dtype,
    ownsBuffer: false,
    toArray(): number[] {
      throw new Error(
        "Stub backendTensor: data lives on the server. Use async api.cpu() / api.item() instead of .toArray().",
      );
    },
  };
}

// ============================================================================
// Client-side handle registry
// ============================================================================

export class ClientHandleMap {
  /** storageId → HandleRef on server. */
  private readonly idToHandle = new Map<number, HandleRef>();
  /** Inverse, for release tracking. */
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

  allHandles(): HandleRef[] {
    return [...this.idToHandle.values()];
  }
}

// ============================================================================
// Patch Torchlette's runtime with remote execution
// ============================================================================

export interface RemoteEngine {
  torch: Torchlette;
  handles: ClientHandleMap;
  rpc: RpcClient;
  /**
   * Release server-side handles for any storage NOT held by one of the
   * `keep` tensors. Call at end of each training step to prevent unbounded
   * server memory growth.
   */
  markStep(keep: FrontendTensor[]): Promise<number>;
  stats: {
    executes: number;
    nodesShipped: number;
    downloads: number;
    scalarReads: number;
    bytesUp: number;
    bytesDown: number;
    handlesReleased: number;
  };
}

export function createRemoteEngine(rpc: RpcClient): RemoteEngine {
  const torch = new Torchlette("cpu");
  const handles = new ClientHandleMap();
  const stats = {
    executes: 0,
    nodesShipped: 0,
    downloads: 0,
    scalarReads: 0,
    bytesUp: 0,
    bytesDown: 0,
    handlesReleased: 0,
  };

  const runtime = torch.runtime;

  /** Ship one plan to the server and patch its outputs as local storages. */
  async function shipPlan(plan: { nodes: LazyIRNode[] }): Promise<void> {
    if (plan.nodes.length === 0) return;

    // Serialize, using the client map to translate materialized refs.
    const wire = serializePlan(plan, {
      resolveHandle: (id: number) => handles.requireHandle(id),
    });

    stats.executes++;
    stats.nodesShipped += plan.nodes.length;
    const json = JSON.stringify({ plan: wire });
    stats.bytesUp += json.length;

    // Execute on server.
    const result = await rpc.execute({ plan: wire });

    // For each output node in the plan, allocate a local stub StorageHandle
    // and bind it to the server's HandleRef. Update the plan's nodes so the
    // engine's bookkeeping finds them via node.result.
    for (const [idxStr, handleRef] of Object.entries(result.outputs)) {
      const idx = Number(idxStr);
      const node = plan.nodes[idx];
      const stub = makeStub(node.shape, node.dtype);
      const storage = createStorageHandle(node.device, stub);
      handles.bind(storage.id, handleRef);
      node.result = storage;
    }
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

  // biome-ignore lint/suspicious/noExplicitAny: overriding private methods
  (runtime as any).forceAllMerged = async (
    ...tensors: RuntimeTensor[]
  ): Promise<void> => {
    const pendingRoots = collectPendingRoots(tensors);
    if (pendingRoots.length === 0) return;
    const plan = buildMergedPlan(pendingRoots);
    if (plan.nodes.length === 0) return;
    await shipPlan(plan);
    postExecuteBookkeeping(plan, tensors);
  };

  // biome-ignore lint/suspicious/noExplicitAny: overriding private methods
  (runtime as any).force = async (tensor: RuntimeTensor): Promise<void> => {
    if (tensor.isMaterialized() || tensor.disposed) return;
    if (tensor.lazyRef.kind !== "pending") return;
    // biome-ignore lint/suspicious/noExplicitAny: reusing patched method
    await (runtime as any).forceAllMerged(tensor);
  };

  // biome-ignore lint/suspicious/noExplicitAny: overriding private methods
  (runtime as any).forceAllPending = async (): Promise<void> => {
    const pendingTensors = getAllPendingTensors();
    if (pendingTensors.length === 0) return;
    const pendingRoots = collectPendingRoots(pendingTensors);
    if (pendingRoots.length === 0) return;
    const plan = buildMergedPlan(pendingRoots, /* skipExecuted */ true);
    if (plan.nodes.length === 0) return;
    await shipPlan(plan);
    postExecuteBookkeeping(plan, pendingTensors);
    for (const node of plan.nodes) {
      node.result = undefined;
    }
  };

  // biome-ignore lint/suspicious/noExplicitAny: overriding public methods
  (runtime as any).cpu = async (a: RuntimeTensor): Promise<number[]> => {
    // biome-ignore lint/suspicious/noExplicitAny: calling patched method
    await (runtime as any).force(a);
    const storage = (a as unknown as { lazyRef: { storage: StorageHandle } })
      .lazyRef.storage;
    const handle = handles.requireHandle(storage.id);
    stats.downloads++;
    const result = await rpc.download({ handle });
    stats.bytesDown += JSON.stringify(result.values).length;
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
    const result = await rpc.readScalar({ handle });
    return result.value;
  };

  async function markStep(keep: FrontendTensor[]): Promise<number> {
    // Force all params to materialize any pending updates (e.g. copy_ after
    // an optimizer step). Without this, a param's lazyRef is still pending
    // and we can't identify its storage for the keep set — which leads us
    // to release storages that the pending chain still references.
    const runtimeTensors = keep.map((t) => t._unwrap());
    // biome-ignore lint/suspicious/noExplicitAny: calling patched method
    await (runtime as any).forceAllMerged(...runtimeTensors);

    // Collect storage IDs of all "keep" tensors.
    const keepIds = new Set<number>();
    for (const t of keep) {
      const ref = t._unwrap().lazyRef;
      if (ref.kind === "materialized") {
        keepIds.add(ref.storage.id);
      }
    }

    // biome-ignore lint/suspicious/noExplicitAny: reading private field for cleanup
    const idToHandle = (handles as any).idToHandle as Map<number, HandleRef>;
    const toRelease: HandleRef[] = [];
    const idsToUnbind: number[] = [];
    for (const [id, handle] of idToHandle) {
      if (!keepIds.has(id)) {
        toRelease.push(handle);
        idsToUnbind.push(id);
      }
    }

    if (toRelease.length > 0) {
      await rpc.release({ handles: toRelease });
      for (const id of idsToUnbind) handles.unbind(id);
      stats.handlesReleased += toRelease.length;
    }
    return toRelease.length;
  }

  void storageTracker; // keep import even if unused at runtime

  return { torch, handles, rpc, stats, markStep };
}
