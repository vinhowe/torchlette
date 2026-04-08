/**
 * Create executionHook + readHook for transparent remote training.
 *
 * Uses the execution-hook design (merged in worktree-remote-training-design):
 * the RuntimeEngine runs its full lifecycle (beginStep/endStep, tidy,
 * snapshotForStep, storageTracker) natively. The hook only replaces
 * WHERE the plan executes — on the remote GPU server via WebSocket.
 *
 * Stubs carry their HandleRef and a download callback, so readHook
 * can download transparently when cpu()/item() are called.
 */

import type { BackendTensor, DType } from "../../../../src/backend/types";
import { createStorageHandle } from "../../../../src/graph/node-factory";
import type { ExecutionPlan, LazyIRNode } from "../../../../src/graph/types";
import type { ExecutionHook, ReadHook } from "../../../../src/runtime/engine";
import { serializePlan } from "../../../../src/remote/serialize";
import { getPendingNodeIds } from "../../../../src/runtime/tensor";
import type { HandleRef } from "../../../../src/remote/wire";
import type { RpcClient } from "./remote-transport";

/** Extended stub that carries the server handle + download callback. */
interface RemoteStub extends BackendTensor {
  _handleRef: HandleRef;
  _download: () => Promise<number[]>;
}

function isRemoteStub(bt: BackendTensor): bt is RemoteStub {
  return "_handleRef" in bt && "_download" in bt;
}

export interface RemoteHooksStats {
  executes: number;
  nodesShipped: number;
  bytesUp: number;
  bytesDown: number;
}

/**
 * Build an executionHook + readHook pair backed by an RpcClient.
 *
 * The executionHook:
 *   1. Serializes the plan (materialized refs → HandleRef lookups).
 *   2. Ships to server via transport.execute().
 *   3. Creates stubs for each output, binding handle + download callback.
 *   4. Replaces node.result with the stub storage.
 *
 * The readHook:
 *   Detects RemoteStub BackendTensors and downloads via transport.download().
 */
export function createRemoteHooks(transport: RpcClient): {
  executionHook: ExecutionHook;
  readHook: ReadHook;
  stats: RemoteHooksStats;
} {
  // Maps local storage id → server HandleRef (for serialization of
  // materialized refs that reference prior remote outputs).
  // Entries are NEVER deleted: lazy nodes can hold materialized refs to
  // destroyed storages across step boundaries (e.g., Adam's input refs to
  // old param weights). Server-side handles are released at session close.
  const handleMap = new Map<number, HandleRef>();

  const stats: RemoteHooksStats = {
    executes: 0,
    nodesShipped: 0,
    bytesUp: 0,
    bytesDown: 0,
  };

  const executionHook: ExecutionHook = async (plan: ExecutionPlan) => {
    if (plan.nodes.length === 0) return;

    // Tell the server which nodes need results (have pending RuntimeTensors).
    // This lets the server fuse intermediates without losing output nodes.
    const pendingIds = getPendingNodeIds();
    const outputNodes: LazyIRNode[] = [];
    for (const node of plan.nodes) {
      if (pendingIds.has(node.id)) outputNodes.push(node);
    }

    const wire = serializePlan(plan, {
      resolveHandle: (storageId: number) => {
        const h = handleMap.get(storageId);
        if (!h) {
          throw new Error(
            `RemoteHook: no HandleRef for storage id=${storageId}`,
          );
        }
        return h;
      },
      outputNodes,
    });
    stats.executes++;
    stats.nodesShipped += plan.nodes.length;
    const payload = JSON.stringify(wire);
    stats.bytesUp += payload.length;

    const result = await transport.execute({ plan: wire });

    // Create a stub storage for a server-resident handle and register it.
    function bindStub(
      handleRef: HandleRef,
      shape: number[],
      dtype: DType,
      device: "cpu" | "webgpu",
    ) {
      const stub: RemoteStub = {
        shape: [...shape],
        dtype,
        ownsBuffer: true,
        _handleRef: handleRef,
        _download: async () => {
          const res = await transport.download({ handle: handleRef });
          stats.bytesDown += res.values.length * 4;
          return res.values;
        },
        toArray(): number[] {
          throw new Error(
            "Remote stub: data lives on the server. Use async cpu()/item().",
          );
        },
        destroy() {
          // No-op: don't remove HandleRefs or release server handles.
          // Lazy nodes can hold materialized refs to destroyed storages
          // across step boundaries (Adam input → old param weight).
          // Server releases all handles at session close.
        },
      };
      const stubStorage = createStorageHandle(device, stub);
      handleMap.set(stubStorage.id, handleRef);
      return stubStorage;
    }

    // Bind primary outputs.
    for (const [idxStr, handleRef] of Object.entries(result.outputs)) {
      const idx = Number(idxStr);
      const node = plan.nodes[idx];
      node.result = bindStub(handleRef, node.shape, node.dtype, node.device);
    }

    // Bind multi-output side results (adamStep m/v, fusedAttention lse/rng, etc.).
    if (result.sideOutputs) {
      for (const [key, handleRef] of Object.entries(result.sideOutputs)) {
        const [idxStr, outIdxStr] = key.split(":");
        const node = plan.nodes[Number(idxStr)];
        const outIdx = Number(outIdxStr);
        if (!node.results) node.results = [];
        node.results[outIdx] = bindStub(
          handleRef,
          node.shape,
          node.dtype,
          node.device,
        );
      }
    }
  };

  const readHook: ReadHook = async (bt: BackendTensor) => {
    if (!isRemoteStub(bt)) {
      // Local tensor (e.g., tensorFromArray on CPU). Read normally.
      return bt.toArray();
    }
    return bt._download();
  };

  /** No-op — handle release is deferred to session close. */
  async function flushPendingRelease(): Promise<void> {}

  return { executionHook, readHook, stats, flushPendingRelease };
}
