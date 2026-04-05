/**
 * RemoteEngine exercised against an in-process Transport that runs plans on
 * the local CPU backend. No WebSocket — just the full client/server wire
 * dance, compressed into one process, to verify the engine builds plans,
 * routes execution through the transport, and reads back values correctly.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { cpuBackend } from "../../src/backend/cpu";
import { registerBackend } from "../../src/backend/registry";
import { executePlanSequential } from "../../src/executor/sequential";
import { createStorageHandle } from "../../src/graph/node-factory";
import type { StorageHandle } from "../../src/graph/types";
import { createRemoteEngine, type Transport } from "../../src/remote/client-engine";
import { deserializePlan } from "../../src/remote/serialize";
import type { HandleRef } from "../../src/remote/wire";

beforeAll(() => {
  registerBackend(cpuBackend);
});

// ============================================================================
// In-process Transport: the "server" lives in the same process.
// ============================================================================

class InProcessTransport implements Transport {
  private readonly registry = new Map<HandleRef, StorageHandle>();
  private nextHandle = 1;

  private alloc(storage: StorageHandle): HandleRef {
    const h = `h${this.nextHandle++}`;
    this.registry.set(h, storage);
    return h;
  }

  async execute(params: {
    plan: import("../../src/remote/wire").SerializedPlan;
  }): Promise<{ outputs: Record<number, HandleRef> }> {
    const plan = deserializePlan(params.plan, {
      resolveHandle: (h) => {
        const s = this.registry.get(h);
        if (!s) throw new Error(`no storage for handle ${h}`);
        return s;
      },
    });
    await executePlanSequential(plan, cpuBackend);

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
    return { outputs };
  }

  async upload(params: {
    values: number[];
    shape: number[];
  }): Promise<{ handle: HandleRef }> {
    const bt = cpuBackend.ops.tensorFromArray(params.values, params.shape);
    const storage = createStorageHandle("cpu", bt);
    return { handle: this.alloc(storage) };
  }

  async download(params: {
    handle: HandleRef;
  }): Promise<{ values: number[] }> {
    const s = this.registry.get(params.handle);
    if (!s) throw new Error(`no storage for handle ${params.handle}`);
    const values = await cpuBackend.ops.read(s.backendTensor);
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
      if (this.registry.delete(h)) n++;
    }
    return { releasedCount: n };
  }

  handleCount(): number {
    return this.registry.size;
  }
}

// ============================================================================
// Tests
// ============================================================================

describe("RemoteEngine (in-process Transport)", () => {
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
    const { torch, handles, markStep } = engine;

    // Build a plan with several outputs, only one of which we "keep".
    const a = torch.tensorFromArray([1, 2, 3, 4], [4]).requires_grad_(true);
    const b = torch.add(a, 10);
    const c = torch.mul(b, 2);
    const d = torch.sub(c, 1);
    await d.cpu();

    const before = handles.size();
    expect(before).toBeGreaterThan(1);

    // Only `a` is a parameter we want to keep.
    const released = await markStep([a]);
    expect(released).toBeGreaterThan(0);
    expect(handles.size()).toBeLessThan(before);
    expect(engine.stats.handlesReleased).toBe(released);
    expect(transport.handleCount()).toBe(handles.size());
  });

  it("trains a tiny MLP through the transport (loss drops)", async () => {
    const transport = new InProcessTransport();
    const engine = createRemoteEngine(transport);
    const { torch, markStep } = engine;

    const opts = { device: "cpu" as const };
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
      // Keep must include every tensor persisting across steps: params AND
      // the batch/target tensors created outside the loop.
      await markStep([W, b, X, T]);
    }

    expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
    // Handle registry should be stable at the `keep` count after markStep.
    expect(engine.handles.size()).toBeLessThanOrEqual(4);
  });

  it("transport.release() keeps client and server registries in sync", async () => {
    const transport = new InProcessTransport();
    const { torch, handles, markStep } = createRemoteEngine(transport);

    const a = torch.tensorFromArray([1, 2, 3], [3]).requires_grad_(true);
    const b = torch.add(a, 1);
    const c = torch.add(b, 2);
    await c.cpu();

    await markStep([a]);
    expect(handles.size()).toBe(transport.handleCount());
  });
});
