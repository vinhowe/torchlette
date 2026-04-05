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
import { createStorageHandle } from "../graph/node-factory";
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
   * Release server-side handles for any storage NOT held by one of the
   * `keep` tensors. Call at end of each training step to prevent unbounded
   * server memory growth. Returns the number of handles released.
   *
   * **`keep` must include every tensor the caller still holds a reference
   * to past this boundary** — model parameters, optimizer state, any
   * batch/target tensors created outside the step, etc. Anything not in
   * `keep` is assumed to be a step-scoped intermediate safe to drop.
   * Forgetting a persistent tensor will break the next plan that
   * references it with "no HandleRef for storage id=…".
   */
  markStep(keep: FrontendTensor[]): Promise<number>;
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
  const stats: RemoteEngineStats = {
    executes: 0,
    nodesShipped: 0,
    downloads: 0,
    scalarReads: 0,
    bytesUp: 0,
    bytesDown: 0,
    handlesReleased: 0,
  };

  const runtime = torch.runtime;

  /** Ship one plan to the server, patching outputs as local stub storages. */
  async function shipPlan(plan: { nodes: LazyIRNode[] }): Promise<void> {
    if (plan.nodes.length === 0) return;

    const wire = serializePlan(plan, {
      resolveHandle: (id: number) => handles.requireHandle(id),
    });
    stats.executes++;
    stats.nodesShipped += plan.nodes.length;
    // Rough cost accounting; not the same as the transport's real payload.
    stats.bytesUp += JSON.stringify(wire).length;

    const result = await transport.execute({ plan: wire });

    // Bind each server HandleRef to a fresh local stub StorageHandle so
    // subsequent plans can reference these outputs as materialized inputs.
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
    const pending = getAllPendingTensors();
    if (pending.length === 0) return;
    const roots = collectPendingRoots(pending);
    if (roots.length === 0) return;
    const plan = buildMergedPlan(roots, /* skipExecuted */ true);
    if (plan.nodes.length === 0) return;
    await shipPlan(plan);
    postExecuteBookkeeping(plan, pending);
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
    // Force all params to materialize any pending updates (e.g. from a
    // copy_ after an optimizer step). Without this, a param's lazyRef is
    // still pending and we can't identify its storage for the keep set —
    // we'd release storages that the pending chain still references.
    const runtimeTensors = keep.map((t) => t._unwrap());
    // biome-ignore lint/suspicious/noExplicitAny: calling patched method
    await (runtime as any).forceAllMerged(...runtimeTensors);

    const keepIds = new Set<number>();
    for (const t of keep) {
      const ref = t._unwrap().lazyRef;
      if (ref.kind === "materialized") {
        keepIds.add(ref.storage.id);
      }
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
      for (const id of idsToUnbind) handles.unbind(id);
      stats.handlesReleased += toRelease.length;
    }
    return toRelease.length;
  }

  return { torch, handles, transport, stats, markStep };
}
