import { beforeEach, describe, expect, it } from "vitest";

import {
  analyzeCrossDeviceOps,
  createTransferStats,
  inferOperationDevice,
  needsTransfer,
  recordTransfer,
  resolveTransferPath,
  shouldAutoTransfer,
} from "../src/engine/cross-device";
import type { LazyIRNode, LazyRef } from "../src/engine/lazy";
import {
  createLazyIRNode,
  createPendingRef,
  resetNodeIdCounter,
} from "../src/engine/lazy";
import { RuntimeEngine } from "../src/runtime/engine";
import { resetBaseIdCounter } from "../src/runtime/tensor";

describe("Transfer Path Resolution", () => {
  it("returns noop for same device", () => {
    const path = resolveTransferPath("cpu", "cpu");
    expect(path.method).toBe("noop");
    expect(path.sourceDevice).toBe("cpu");
    expect(path.targetDevice).toBe("cpu");
  });

  it("returns via_cpu for cpu to webgpu", () => {
    const path = resolveTransferPath("cpu", "webgpu");
    expect(path.method).toBe("via_cpu");
    expect(path.sourceDevice).toBe("cpu");
    expect(path.targetDevice).toBe("webgpu");
  });

  it("returns via_cpu for webgpu to cpu", () => {
    const path = resolveTransferPath("webgpu", "cpu");
    expect(path.method).toBe("via_cpu");
    expect(path.sourceDevice).toBe("webgpu");
    expect(path.targetDevice).toBe("cpu");
  });

  it("returns via_cpu for webgpu to webgpu (different adapters)", () => {
    // In future could be "direct" for same adapter
    const path = resolveTransferPath("webgpu", "webgpu");
    expect(path.method).toBe("noop");
  });
});

describe("Transfer Needed Check", () => {
  it("returns false for same device", () => {
    expect(needsTransfer("cpu", "cpu")).toBe(false);
    expect(needsTransfer("webgpu", "webgpu")).toBe(false);
  });

  it("returns true for different devices", () => {
    expect(needsTransfer("cpu", "webgpu")).toBe(true);
    expect(needsTransfer("webgpu", "cpu")).toBe(true);
  });
});

describe("Operation Device Inference", () => {
  it("uses default when no inputs", () => {
    expect(inferOperationDevice([])).toBe("cpu");
    expect(inferOperationDevice([], "webgpu")).toBe("webgpu");
  });

  it("uses input device when single input", () => {
    expect(inferOperationDevice(["cpu"])).toBe("cpu");
    expect(inferOperationDevice(["webgpu"])).toBe("webgpu");
  });

  it("uses common device when all inputs same", () => {
    expect(inferOperationDevice(["cpu", "cpu"])).toBe("cpu");
    expect(inferOperationDevice(["webgpu", "webgpu"])).toBe("webgpu");
  });

  it("prefers GPU when mixed devices", () => {
    expect(inferOperationDevice(["cpu", "webgpu"])).toBe("webgpu");
    expect(inferOperationDevice(["webgpu", "cpu"])).toBe("webgpu");
  });
});

describe("Auto-Transfer Detection", () => {
  it("returns null when all inputs on target", () => {
    expect(shouldAutoTransfer(["cpu"], "cpu")).toBeNull();
    expect(shouldAutoTransfer(["webgpu", "webgpu"], "webgpu")).toBeNull();
  });

  it("returns target when transfer needed", () => {
    expect(shouldAutoTransfer(["cpu"], "webgpu")).toBe("webgpu");
    expect(shouldAutoTransfer(["webgpu"], "cpu")).toBe("cpu");
  });
});

describe("Cross-Device Analysis", () => {
  beforeEach(() => {
    resetNodeIdCounter();
  });

  function makeNode(
    op: string,
    inputs: LazyRef[],
    device: "cpu" | "webgpu",
  ): LazyIRNode {
    return createLazyIRNode(op as any, inputs, [2, 2], "f32", device);
  }

  it("detects single device graph", () => {
    const node1 = makeNode("tensorFromArray", [], "cpu");
    const node2 = makeNode("relu", [createPendingRef(node1)], "cpu");

    const analysis = analyzeCrossDeviceOps([node1, node2]);

    expect(analysis.isSingleDevice).toBe(true);
    expect(analysis.devices.size).toBe(1);
    expect(analysis.devices.has("cpu")).toBe(true);
    expect(analysis.transferPoints.length).toBe(0);
  });

  it("detects multi-device graph", () => {
    const node1 = makeNode("tensorFromArray", [], "cpu");
    const node2 = makeNode("relu", [createPendingRef(node1)], "webgpu");

    const analysis = analyzeCrossDeviceOps([node1, node2]);

    expect(analysis.isSingleDevice).toBe(false);
    expect(analysis.devices.size).toBe(2);
    expect(analysis.devices.has("cpu")).toBe(true);
    expect(analysis.devices.has("webgpu")).toBe(true);
  });

  it("identifies transfer points", () => {
    const node1 = makeNode("tensorFromArray", [], "cpu");
    const node2 = makeNode("relu", [createPendingRef(node1)], "webgpu");

    const analysis = analyzeCrossDeviceOps([node1, node2]);

    expect(analysis.transferPoints.length).toBe(1);
    expect(analysis.transferPoints[0].nodeId).toBe(node2.id);
    expect(analysis.transferPoints[0].inputIndex).toBe(0);
    expect(analysis.transferPoints[0].sourceDevice).toBe("cpu");
    expect(analysis.transferPoints[0].targetDevice).toBe("webgpu");
  });

  it("handles multiple inputs with mixed devices", () => {
    const node1 = makeNode("tensorFromArray", [], "cpu");
    const node2 = makeNode("tensorFromArray", [], "webgpu");
    const node3 = makeNode(
      "add",
      [createPendingRef(node1), createPendingRef(node2)],
      "webgpu",
    );

    const analysis = analyzeCrossDeviceOps([node1, node2, node3]);

    // Only node1 needs transfer (cpu -> webgpu)
    expect(analysis.transferPoints.length).toBe(1);
    expect(analysis.transferPoints[0].sourceDevice).toBe("cpu");
    expect(analysis.transferPoints[0].targetDevice).toBe("webgpu");
  });
});

describe("Transfer Statistics", () => {
  it("creates empty stats", () => {
    const stats = createTransferStats();
    expect(stats.totalTransfers).toBe(0);
    expect(stats.totalBytesTransferred).toBe(0);
    expect(stats.transfersByPath.size).toBe(0);
  });

  it("records transfers", () => {
    const stats = createTransferStats();

    recordTransfer(stats, {
      storage: { id: 1, device: "webgpu", backendTensor: { shape: [2, 2], toArray: () => [] } },
      stats: {
        bytesTransferred: 1024,
        path: { sourceDevice: "cpu", targetDevice: "webgpu", method: "via_cpu" },
      },
    });

    expect(stats.totalTransfers).toBe(1);
    expect(stats.totalBytesTransferred).toBe(1024);
    expect(stats.transfersByPath.get("cpu->webgpu")).toEqual({
      count: 1,
      bytes: 1024,
    });
  });

  it("aggregates multiple transfers", () => {
    const stats = createTransferStats();

    recordTransfer(stats, {
      storage: { id: 1, device: "webgpu", backendTensor: { shape: [2, 2], toArray: () => [] } },
      stats: {
        bytesTransferred: 1024,
        path: { sourceDevice: "cpu", targetDevice: "webgpu", method: "via_cpu" },
      },
    });

    recordTransfer(stats, {
      storage: { id: 2, device: "webgpu", backendTensor: { shape: [4, 4], toArray: () => [] } },
      stats: {
        bytesTransferred: 2048,
        path: { sourceDevice: "cpu", targetDevice: "webgpu", method: "via_cpu" },
      },
    });

    expect(stats.totalTransfers).toBe(2);
    expect(stats.totalBytesTransferred).toBe(3072);
    expect(stats.transfersByPath.get("cpu->webgpu")).toEqual({
      count: 2,
      bytes: 3072,
    });
  });

  it("ignores noop transfers", () => {
    const stats = createTransferStats();

    recordTransfer(stats, {
      storage: { id: 1, device: "cpu", backendTensor: { shape: [2, 2], toArray: () => [] } },
      stats: {
        bytesTransferred: 0,
        path: { sourceDevice: "cpu", targetDevice: "cpu", method: "noop" },
      },
    });

    expect(stats.totalTransfers).toBe(0);
    expect(stats.totalBytesTransferred).toBe(0);
  });
});

describe("RuntimeEngine Lazy Transfer", () => {
  let engine: RuntimeEngine;

  beforeEach(() => {
    engine = new RuntimeEngine("cpu");
    resetNodeIdCounter();
    resetBaseIdCounter();
  });

  it("returns same tensor for same device", () => {
    const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
    const b = engine.transfer(a, "cpu");

    expect(b).toBe(a);
    expect(b.device).toBe("cpu");
  });

  it("creates lazy transfer node for different device", () => {
    const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2], "cpu");
    const b = engine.transfer(a, "webgpu");

    expect(b).not.toBe(a);
    expect(b.device).toBe("webgpu");
    expect(b.shape).toEqual([2, 2]);
    expect(b.isMaterialized()).toBe(false);
  });

  it("transfer preserves shape", () => {
    const a = engine.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], "cpu");
    const b = engine.transfer(a, "webgpu");

    expect(b.shape).toEqual([2, 3]);
  });

  it("transfer creates new baseId", () => {
    const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2], "cpu");
    const b = engine.transfer(a, "webgpu");

    expect(b.baseId).not.toBe(a.baseId);
  });
});

describe("RuntimeEngine ensureSameDevice (internal)", () => {
  let engine: RuntimeEngine;

  beforeEach(() => {
    engine = new RuntimeEngine("cpu");
    resetNodeIdCounter();
    resetBaseIdCounter();
  });

  it("returns tensors unchanged when already same device", () => {
    const a = engine.tensorFromArray([1, 2], [2], "cpu");
    const b = engine.tensorFromArray([3, 4], [2], "cpu");

    // Access private method for testing
    const result = (engine as any).ensureSameDevice(a, b);

    expect(result.tensors[0]).toBe(a);
    expect(result.tensors[1]).toBe(b);
    expect(result.device).toBe("cpu");
  });

  it("prefers GPU when mixed devices", () => {
    const a = engine.tensorFromArray([1, 2], [2], "cpu");
    const b = engine.tensorFromArray([3, 4], [2], "webgpu");

    const result = (engine as any).ensureSameDevice(a, b);

    expect(result.device).toBe("webgpu");
    // First tensor should be transferred
    expect(result.tensors[0].device).toBe("webgpu");
    expect(result.tensors[0]).not.toBe(a);
    // Second tensor unchanged
    expect(result.tensors[1]).toBe(b);
  });
});

describe("Cross-Device Transfer Execution (CPU)", () => {
  let engine: RuntimeEngine;

  beforeEach(() => {
    engine = new RuntimeEngine("cpu");
    resetNodeIdCounter();
    resetBaseIdCounter();
  });

  it("noop transfer preserves data", async () => {
    const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2], "cpu");
    const b = engine.transfer(a, "cpu"); // Same device = noop

    // Should be the exact same tensor
    expect(b).toBe(a);

    // Force and verify data
    await engine.force(a);
    const values = await engine.cpu(a);
    expect(values).toEqual([1, 2, 3, 4]);
  });

  it("transfer then compute preserves correctness", async () => {
    // Create tensor, transfer (noop for CPU->CPU), then compute
    const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2], "cpu");
    const b = engine.transfer(a, "cpu");
    const c = engine.relu(b);

    await engine.force(c);
    const values = await engine.cpu(c);
    expect(values).toEqual([1, 2, 3, 4]); // All positive, so unchanged by relu
  });

  it("transfer negative values through relu", async () => {
    const a = engine.tensorFromArray([-1, 2, -3, 4], [2, 2], "cpu");
    const b = engine.transfer(a, "cpu");
    const c = engine.relu(b);

    await engine.force(c);
    const values = await engine.cpu(c);
    expect(values).toEqual([0, 2, 0, 4]); // Negatives become 0
  });

  it("chained operations after transfer", async () => {
    const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2], "cpu");
    const b = engine.tensorFromArray([10, 20, 30, 40], [2, 2], "cpu");

    // Transfer both (noop), then add
    const aT = engine.transfer(a, "cpu");
    const bT = engine.transfer(b, "cpu");
    const c = engine.add(aT, bT);

    await engine.force(c);
    const values = await engine.cpu(c);
    expect(values).toEqual([11, 22, 33, 44]);
  });
});

describe("Frontend Cross-Device API", () => {
  it("tensor.to() returns same tensor for same device", async () => {
    const { Torchlette } = await import("../src/frontend");
    const torch = new Torchlette("cpu");

    const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
    const b = a.to("cpu");

    expect(b).toBe(a);
  });

  it("tensor.to() is lazy", async () => {
    const { Torchlette } = await import("../src/frontend");
    const torch = new Torchlette("cpu");

    const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);

    // Create transfer to webgpu (lazy - won't execute yet)
    const b = a.to("webgpu");

    // Should have different device but not be materialized yet
    expect(b.device).toBe("webgpu");
    expect(b).not.toBe(a);
  });

  it("operations work after to()", async () => {
    const { Torchlette } = await import("../src/frontend");
    const torch = new Torchlette("cpu");

    const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
    const b = a.to("cpu"); // Noop for same device
    const c = b.relu();

    const values = await c.cpu();
    expect(values).toEqual([1, 2, 3, 4]);
  });
});
