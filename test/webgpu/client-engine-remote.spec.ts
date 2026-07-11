/**
 * RemoteEngine exercised against an in-process Transport that runs plans on
 * the local WebGPU backend. No WebSocket — just the full client/server wire
 * dance, compressed into one process, to verify the engine builds plans,
 * routes execution through the transport, and reads back values correctly.
 *
 * This test lives in the webgpu project because createRemoteEngine builds
 * its client-side Torchlette as a webgpu device — the cpu device path was
 * removed because it caused the client to construct lazy graphs that the
 * server's WebGPU executor mis-handled (silent ~30% slow-convergence).
 * The in-process transport here mirrors what the real server does: a real
 * webgpu executePlanSequential against the registered webgpu backend.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { initWebGPU, webgpuBackend } from "../../src/backend/webgpu";
import { registerBackend } from "../../src/backend/registry";
import { executePlanSequential } from "../../src/executor/sequential";
import {
  createStorageHandle,
  getNextStorageId,
} from "../../src/graph/node-factory";
import { rcRelease, rcRetain } from "../../src/graph/refcount";
import { storageTracker } from "../../src/graph/storage-tracker";
import type { StorageHandle } from "../../src/graph/types";
import { createRemoteEngine, type Transport } from "../../src/remote/client-engine";
import { deserializePlan } from "../../src/remote/serialize";
import type { HandleRef } from "../../src/remote/wire";
import { cpuOnly } from "../helpers/webgpu";

beforeAll(async () => {
  if (cpuOnly) return;
  const ok = await initWebGPU();
  if (!ok) throw new Error("WebGPU init failed");
  registerBackend(webgpuBackend);
});

// ============================================================================
// In-process Transport: the "server" lives in the same process.
// ============================================================================

class InProcessTransport implements Transport {
  private readonly registry = new Map<HandleRef, StorageHandle>();
  private nextHandle = 1;

  // Each registry entry is ONE ownership of its StorageHandle, accounted via
  // the engine's shared refcount system — exactly as the real server session
  // does (examples/remote-training-demo/server.ts allocHandle/release). Without
  // it, server-side plan-output storages sit at rc=0 in the module-global
  // tracker that the CLIENT engine (same process) shares, and the client's next
  // destroyUnreachable() reaps them under the transport's live registry → a
  // cross-"engine" reclaimed-read under STRICT_LIFETIME (task #74).
  private alloc(storage: StorageHandle): HandleRef {
    const h = `h${this.nextHandle++}`;
    this.registry.set(h, storage);
    rcRetain(storage.id, "session.handle");
    return h;
  }

  private free(h: HandleRef): boolean {
    const storage = this.registry.get(h);
    if (!storage) return false;
    this.registry.delete(h);
    rcRelease(storage.id, "session.handle");
    return true;
  }

  async execute(params: {
    plan: import("../../src/remote/wire").SerializedPlan;
    releases?: HandleRef[];
  }): Promise<{ outputs: Record<number, HandleRef> }> {
    // Process piggybacked releases first, mirroring the real server.
    if (params.releases && params.releases.length > 0) {
      for (const h of params.releases) this.free(h);
    }
    const plan = deserializePlan(params.plan, {
      resolveHandle: (h) => {
        const s = this.registry.get(h);
        if (!s) throw new Error(`no storage for handle ${h}`);
        return s;
      },
    });
    // Snapshot the storage-id watermark, execute, retain outputs, then drop
    // this plan's step-scoped intermediates (rc still 0) — the real server's
    // force → materialize → destroyUnreachableSince sequence.
    const sinceId = getNextStorageId();
    await executePlanSequential(plan, webgpuBackend);

    const outputSet = params.plan.outputNodes
      ? new Set(params.plan.outputNodes)
      : null;
    const outputs: Record<number, HandleRef> = {};
    for (let i = 0; i < plan.nodes.length; i++) {
      const node = plan.nodes[i];
      if (!node.result) continue;
      if (outputSet === null || outputSet.has(i)) {
        outputs[i] = this.alloc(node.result);
      }
    }
    storageTracker.destroyUnreachableSince(sinceId);
    return { outputs };
  }

  async upload(params: {
    values: number[];
    shape: number[];
  }): Promise<{ handle: HandleRef }> {
    const bt = webgpuBackend.ops.tensorFromArray(params.values, params.shape);
    const storage = createStorageHandle("webgpu", bt);
    return { handle: this.alloc(storage) };
  }

  async download(params: {
    handle: HandleRef;
  }): Promise<{ values: number[] }> {
    const s = this.registry.get(params.handle);
    if (!s) throw new Error(`no storage for handle ${params.handle}`);
    const values = await webgpuBackend.ops.read(s.backendTensor);
    return { values };
  }

  async readScalar(params: {
    handle: HandleRef;
  }): Promise<{ value: number }> {
    const { values } = await this.download({ handle: params.handle });
    if (values.length !== 1) {
      throw new Error(`readScalar: tensor has ${values.length} elements`);
    }
    return { value: values[0] };
  }

  async release(params: {
    handles: HandleRef[];
  }): Promise<{ releasedCount: number }> {
    let n = 0;
    for (const h of params.handles) {
      if (this.free(h)) n++;
    }
    if (n > 0) storageTracker.destroyUnreachable();
    return { releasedCount: n };
  }

  handleCount(): number {
    return this.registry.size;
  }
}

// ============================================================================
// Tests
// ============================================================================

describe.skipIf(cpuOnly)("RemoteEngine (in-process Transport)", () => {
  it("runs a forward pass and reads the result via the transport", async () => {
    const transport = new InProcessTransport();
    const { torch } = createRemoteEngine(transport);

    const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
    const b = torch.tensorFromArray([10, 20, 30, 40], [2, 2]);
    const c = torch.add(a, b);
    const values = await c.cpu();

    expect(values).toEqual([11, 22, 33, 44]);
  });

  it("reads scalars via readScalar (not full download)", async () => {
    const transport = new InProcessTransport();
    const { torch, stats } = createRemoteEngine(transport);

    const x = torch.tensorFromArray([1, 2, 3, 4], [4]);
    const sum = x.sum();
    if (typeof sum === "number") throw new Error("sum should be Tensor");
    const result = await sum.item();

    expect(result).toBe(10);
    expect(stats.scalarReads).toBe(1);
    expect(stats.downloads).toBe(0);
  });

  it("handle bridging: plan N outputs become plan N+1 inputs", async () => {
    const transport = new InProcessTransport();
    const { torch, handles } = createRemoteEngine(transport);

    // Plan 1: create a + b
    const a = torch.tensorFromArray([1, 2, 3], [3]);
    const b = torch.tensorFromArray([10, 20, 30], [3]);
    const sum = torch.add(a, b);
    await sum.cpu(); // forces plan 1

    const handleCountAfterPlan1 = handles.size();
    expect(handleCountAfterPlan1).toBeGreaterThan(0);

    // Plan 2: references sum from plan 1 via a materialized ref
    const scaled = torch.mul(sum, 2);
    const result = await scaled.cpu();

    expect(result).toEqual([22, 44, 66]);
  });

  it("markStep releases handles not bound to `keep` tensors", async () => {
    const transport = new InProcessTransport();
    const engine = createRemoteEngine(transport);
    const { torch, handles, markStep, flushReleases } = engine;

    // Build a plan with several outputs, only one of which we "keep".
    const a = torch.tensorFromArray([1, 2, 3, 4], [4]).requires_grad_(true);
    const b = torch.add(a, 10);
    const c = torch.mul(b, 2);
    const d = torch.sub(c, 1);
    await d.cpu();

    const before = handles.size();
    expect(before).toBeGreaterThan(1);

    // Dispose temporaries — only `a` remains alive.
    const released = await markStep([a]);
    expect(released).toBeGreaterThan(0);
    expect(handles.size()).toBeLessThan(before);
    expect(engine.stats.handlesReleased).toBe(released);
    // markStep defers the release RPC for piggyback. Flush explicitly so
    // we can assert the server registry has caught up.
    await flushReleases();
    expect(transport.handleCount()).toBe(handles.size());
  });

  it("trains a tiny MLP through the transport (loss drops)", async () => {
    const transport = new InProcessTransport();
    const engine = createRemoteEngine(transport);
    const { torch, markStep } = engine;

    const opts = { device: "webgpu" as const };
    const W = torch
      .tensorFromArray([0.1, -0.2, 0.3, 0.05], [2, 2], opts)
      .requires_grad_(true);
    const b = torch
      .tensorFromArray([0, 0], [2], opts)
      .requires_grad_(true);
    const X = torch.tensorFromArray(
      [0, 0, 0, 1, 1, 0, 1, 1],
      [4, 2],
      opts,
    );
    const T = torch.tensorFromArray([0, 1, 1, 0], [4], { ...opts, dtype: "i32" });

    const losses: number[] = [];
    const lr = 0.5;
    for (let step = 0; step < 10; step++) {
      const logits = torch.add(torch.matmul(X, W), b); // [4, 2]
      // Manual cross-entropy via logsumexp trick
      const lse = torch.add(logits, 0);
      const maxL = lse.max({ dim: -1, keepdim: true });
      if (typeof maxL === "number") throw new Error("max scalar");
      const centered = torch.sub(logits, maxL);
      const expC = torch.exp(centered);
      const sumExp = expC.sum({ dim: -1, keepdim: true });
      if (typeof sumExp === "number") throw new Error("sumExp scalar");
      const logSum = torch.log(sumExp);
      const logProbs = torch.sub(centered, logSum); // [4, 2]
      // Gather targets along dim 1
      const tReshaped = torch.reshape(T, [4, 1]);
      const selected = torch.gather(logProbs, tReshaped, { dim: 1 });
      const loss = torch.neg(selected.mean());
      if (typeof loss === "number") throw new Error("loss scalar");
      losses.push(await loss.item());
      await loss.backward();

      for (const p of [W, b]) {
        if (!p.grad) throw new Error("missing grad");
        torch.noGrad(() => {
          const updated = torch.sub(p, torch.mul(p.grad!, lr));
          p.copy_(updated);
        });
        p.zeroGrad();
      }
      await markStep([W, b, X, T]);
    }

    expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
    // Handle registry includes keep tensors + any storages referenced by
    // materialized refs in the live lazy graph (conservative retention).
    expect(engine.handles.size()).toBeLessThanOrEqual(10);
  });

  it("preUpload sends tensorFromArray data via binary upload, not plan JSON", async () => {
    const transport = new InProcessTransport();
    const engine = createRemoteEngine(transport);
    const { torch, preUpload } = engine;

    // Create a large-ish tensor inline.
    const bigValues = Array.from({ length: 1024 }, (_, i) => i * 0.001);
    const W = torch.tensorFromArray(bigValues, [32, 32]).requires_grad_(true);

    // Pre-upload it.
    const count = await preUpload([W]);
    expect(count).toBe(1);
    // W should now be materialized.
    expect(engine.handles.size()).toBe(1);

    // Use W in a computation — it should be referenced as a materialized
    // handle, NOT re-sent as inline data in the plan.
    const x = torch.tensorFromArray(
      Array.from({ length: 32 }, () => 1),
      [1, 32],
    );
    const y = torch.matmul(x, W);
    const result = await y.cpu();

    // Verify correctness: row 0 of W is [0, 0.001, 0.002, ..., 0.031]
    // x = [1,1,...,1], so y[0,j] = sum(W[:,j]) = sum of column j
    expect(result.length).toBe(32);
    expect(result[0]).toBeCloseTo(
      bigValues.filter((_, i) => i % 32 === 0).reduce((a, b) => a + b, 0),
      4,
    );
  });

  it("preUpload skips already-materialized tensors", async () => {
    const transport = new InProcessTransport();
    const { torch, preUpload } = createRemoteEngine(transport);

    const a = torch.tensorFromArray([1, 2, 3], [3]);
    await a.cpu(); // forces → now materialized
    const count = await preUpload([a]);
    expect(count).toBe(0); // skipped, already materialized
  });

  it("transport.release() keeps client and server registries in sync", async () => {
    const transport = new InProcessTransport();
    const { torch, handles, markStep, flushReleases } =
      createRemoteEngine(transport);

    const a = torch.tensorFromArray([1, 2, 3], [3]).requires_grad_(true);
    const b = torch.add(a, 1);
    const c = torch.add(b, 2);
    await c.cpu();

    await markStep([a]);
    // markStep defers the release RPC for piggyback. Flush explicitly so
    // the server registry catches up.
    await flushReleases();
    expect(handles.size()).toBe(transport.handleCount());
  });
});
