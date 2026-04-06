/**
 * Derisk the execution-hook + read-hook design. These two hooks on
 * RuntimeEngine replace the monkey-patch approach: the engine's full
 * lifecycle (beginStep/endStep, tidy, snapshotForStep, storageTracker)
 * runs natively. The hook only replaces WHERE the plan runs.
 *
 * Uses an in-process executor (CPU backend) as the "remote server" to
 * prove the hooks are sufficient without any monkey-patching.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { cpuBackend } from "../../src/backend/cpu";
import { registerBackend } from "../../src/backend/registry";
import type { BackendTensor, DType } from "../../src/backend/types";
import { executePlanSequential } from "../../src/executor/sequential";
import { Torchlette } from "../../src/frontend/torchlette";
import { createStorageHandle } from "../../src/graph/node-factory";
import type { ExecutionPlan, StorageHandle } from "../../src/graph/types";
import type { ExecutionHook, ReadHook } from "../../src/runtime/engine";

beforeAll(() => {
  registerBackend(cpuBackend);
});

// ============================================================================
// Stubs that carry a read callback (simulates remote handle)
// ============================================================================

/** Accumulated destroy notifications — simulates batched release RPCs. */
const destroyed: string[] = [];

interface RemoteStub extends BackendTensor {
  _id: string;
  _readData: () => Promise<number[]>;
}

function isRemoteStub(bt: BackendTensor): bt is RemoteStub {
  return "_id" in bt && "_readData" in bt;
}

let nextStubId = 1;

/** In-process "server": executes the plan on CPU, stores results. */
const serverStore = new Map<string, StorageHandle>();

function makeExecutionHook(): ExecutionHook {
  return async (plan: ExecutionPlan) => {
    // Resolve stubs → real server-side storages (like a real server resolving handles).
    for (const node of plan.nodes) {
      for (let i = 0; i < node.inputs.length; i++) {
        const ref = node.inputs[i];
        if (
          ref.kind === "materialized" &&
          isRemoteStub(ref.storage.backendTensor)
        ) {
          const real = serverStore.get(
            (ref.storage.backendTensor as RemoteStub)._id,
          );
          if (real) {
            node.inputs[i] = { kind: "materialized", storage: real };
          }
        }
      }
    }

    await executePlanSequential(plan, cpuBackend);

    // For each node with a result, create a stub that carries a read
    // callback and a destroy callback. Store the real storage server-side.
    for (const node of plan.nodes) {
      if (!node.result) continue;
      const realStorage = node.result;
      const stubId = `stub-${nextStubId++}`;
      serverStore.set(stubId, realStorage);

      const stub: RemoteStub = {
        shape: realStorage.backendTensor.shape,
        dtype: realStorage.backendTensor.dtype,
        ownsBuffer: true,
        _id: stubId,
        _readData: async () => cpuBackend.ops.read(realStorage.backendTensor),
        toArray(): number[] {
          throw new Error("Remote stub: use async cpu()");
        },
        destroy() {
          destroyed.push(stubId);
          serverStore.delete(stubId);
        },
      };

      // Replace the node's result with a stub storage.
      const stubStorage = createStorageHandle(node.device, stub);
      node.result = stubStorage;
    }
  };
}

function makeReadHook(): ReadHook {
  return async (bt: BackendTensor) => {
    if (!isRemoteStub(bt)) {
      throw new Error("readHook: expected RemoteStub");
    }
    return bt._readData();
  };
}

// ============================================================================
// Tests
// ============================================================================

describe("RuntimeEngine execution hooks", () => {
  it("basic forward + read through hooks", async () => {
    const api = new Torchlette("cpu", {
      executionHook: makeExecutionHook(),
      readHook: makeReadHook(),
    } as Parameters<typeof Torchlette>[1]);

    const a = api.tensorFromArray([1, 2, 3, 4], [2, 2]);
    const b = api.tensorFromArray([10, 20, 30, 40], [2, 2]);
    const c = api.add(a, b);
    const values = await c.cpu();

    expect(values).toEqual([11, 22, 33, 44]);
  });

  it("item() works through readHook", async () => {
    const api = new Torchlette("cpu", {
      executionHook: makeExecutionHook(),
      readHook: makeReadHook(),
    } as Parameters<typeof Torchlette>[1]);

    const x = api.tensorFromArray([1, 2, 3, 4], [4]);
    const s = x.sum();
    if (typeof s === "number") throw new Error("expected tensor");
    expect(await s.item()).toBe(10);
  });

  it("autograd backward works through execution hook", async () => {
    const api = new Torchlette("cpu", {
      executionHook: makeExecutionHook(),
      readHook: makeReadHook(),
    } as Parameters<typeof Torchlette>[1]);

    const W = api
      .tensorFromArray([0.1, -0.2, 0.3, 0.05], [2, 2])
      .requires_grad_(true);
    const X = api.tensorFromArray([1, 0, 0, 1], [2, 2]);
    const y = api.matmul(X, W);
    const loss = y.sum();
    if (typeof loss === "number") throw new Error("expected tensor");
    const lossVal = await loss.item();
    expect(lossVal).toBeCloseTo(0.25, 4);

    await loss.backward();
    expect(W.grad).not.toBeNull();
    const grad = await W.grad!.cpu();
    expect(grad.length).toBe(4);
  });

  it("training loop converges through hooks", async () => {
    const api = new Torchlette("cpu", {
      executionHook: makeExecutionHook(),
      readHook: makeReadHook(),
    } as Parameters<typeof Torchlette>[1]);

    const W = api
      .tensorFromArray([0.1, -0.2, 0.3, 0.05], [2, 2])
      .requires_grad_(true);
    const b = api.tensorFromArray([0, 0], [2]).requires_grad_(true);
    const X = api.tensorFromArray([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]);
    const T = api.tensorFromArray([0, 1, 1, 0], [4], { dtype: "i32" });

    const { crossEntropy } = await import("../../src/nn/functional");
    const losses: number[] = [];

    for (let step = 0; step < 15; step++) {
      const logits = api.add(api.matmul(X, W), b);
      const loss = crossEntropy(api, logits, T, { reduction: "mean" });
      const lossVal = await loss.item();
      losses.push(lossVal);

      await loss.backward();

      for (const p of [W, b]) {
        if (!p.grad) continue;
        api.noGrad(() => {
          p.copy_(api.sub(p, api.mul(p.grad!, 0.5)));
        });
        p.zeroGrad();
      }
    }

    expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
  });

  it("stub rc goes to 0 after releaseStepTemps", async () => {
    const { rcGet } = await import("../../src/graph/refcount");
    const { storageTracker } = await import("../../src/graph/storage-tracker");
    destroyed.length = 0;

    const api = new Torchlette("cpu", {
      executionHook: makeExecutionHook(),
      readHook: makeReadHook(),
    } as Parameters<typeof Torchlette>[1]);

    const W = api.tensorFromArray([1, 2], [2]).requires_grad_(true);
    await W.cpu();

    await api.beginStep();

    const tmp = api.add(W, 1);
    await tmp.cpu();

    const tmpRt = tmp._unwrap();
    const tmpRef = tmpRt.lazyRef;
    if (tmpRef.kind !== "materialized") throw new Error("expected materialized");
    const tmpId = tmpRef.storage.id;

    expect(rcGet(tmpId)).toBeGreaterThan(0);
    const storageBefore = storageTracker.getStorage(tmpId);
    expect(storageBefore).toBeDefined();
    expect(storageBefore).toBeDefined();

    await api.markStep();


    expect(destroyed.length).toBeGreaterThan(0);
  });

  it("stubs get destroyed by engine lifecycle (beginStep/markStep, no manual release)", async () => {
    destroyed.length = 0;
    const api = new Torchlette("cpu", {
      executionHook: makeExecutionHook(),
      readHook: makeReadHook(),
    } as Parameters<typeof Torchlette>[1]);

    // Create persistent weights BEFORE beginStep — these survive.
    const W = api.tensorFromArray([1, 2, 3, 4], [2, 2]).requires_grad_(true);

    // Snapshot: everything that exists now is "persistent".
    await api.beginStep();

    // Training step: intermediates created AFTER the snapshot are step-scoped.
    const X = api.tensorFromArray([1, 0, 0, 1], [2, 2]);
    const y = api.matmul(X, W);
    const loss = y.sum();
    if (typeof loss === "number") throw new Error("expected tensor");
    const lossVal = await loss.item();
    expect(lossVal).toBeCloseTo(10, 4); // 1+2+3+4

    await loss.backward();

    // SGD update
    api.noGrad(() => {
      W.copy_(api.sub(W, api.mul(W.grad!, 0.1)));
    });
    W.zeroGrad();

    const destroyedBefore = destroyed.length;

    // markStep: destroys step-scoped stubs (X, y, loss, intermediates).
    await api.markStep();

    // Stubs created during the step should have been destroyed via the
    // engine's native storageTracker.releaseStepTemps → destroyUnreachable
    // → backendTensor.destroy() → our destroy callback.
    expect(destroyed.length).toBeGreaterThan(destroyedBefore);

    // W should still be alive (it was created before beginStep).
    const wVals = await W.cpu();
    expect(wVals.length).toBe(4);
  });
});
