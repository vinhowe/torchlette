/**
 * End-to-end derisking: train a tiny MLP on XOR with every forward/backward/
 * optimizer plan serialized through JSON and re-executed. Compare loss
 * trajectory and final weights bit-exact against a local-only baseline.
 *
 * If this passes, autograd, saved-for-backward tensors, in-place ops (copy_),
 * and handle bridging across plan boundaries all survive the wire.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { cpuBackend } from "../../src/backend/cpu";
import { registerBackend } from "../../src/backend/registry";
import { executePlanSequential } from "../../src/executor/sequential";
import { buildMergedPlan } from "../../src/executor/plan-builder";
import { Torchlette } from "../../src/frontend/torchlette";
import { storageTracker } from "../../src/graph/storage-tracker";
import type { DeviceKind } from "../../src/backend/types";
import type { LazyIRNode } from "../../src/graph/types";
import type { Tensor as RuntimeTensor } from "../../src/runtime/tensor";
import {
  clearDisposedPendingNodeIds,
  getAllPendingTensors,
  materializePendingTensors,
} from "../../src/runtime/tensor";
import { deserializePlan, serializePlan } from "../../src/remote/serialize";
import type { HandleRef, SerializedPlan } from "../../src/remote/wire";

beforeAll(() => {
  registerBackend(cpuBackend);
});

// ============================================================================
// In-process round-trip patch
// ============================================================================

const resolveHandle = (id: number): HandleRef => `h:${id}`;
const lookupHandle = (h: HandleRef) => {
  const id = Number(h.slice(2));
  const s = storageTracker.getStorage(id);
  if (!s) throw new Error(`no storage for id=${id}`);
  return s;
};

async function roundTripAndExecute(plan: {
  nodes: LazyIRNode[];
}): Promise<void> {
  if (plan.nodes.length === 0) return;
  const wire = serializePlan(plan, { resolveHandle });
  const parsed = JSON.parse(JSON.stringify(wire)) as SerializedPlan;
  const clone = deserializePlan(parsed, { resolveHandle: lookupHandle });
  await executePlanSequential(clone, cpuBackend);
  for (let i = 0; i < plan.nodes.length; i++) {
    plan.nodes[i].result = clone.nodes[i].result;
    if (clone.nodes[i].results) plan.nodes[i].results = clone.nodes[i].results;
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

function patchEngineForRoundTrip(torch: Torchlette): void {
  const runtime = torch.runtime;

  // biome-ignore lint/suspicious/noExplicitAny: overriding private methods
  (runtime as any).forceAllMerged = async (
    ...tensors: RuntimeTensor[]
  ): Promise<void> => {
    const roots = collectPendingRoots(tensors);
    if (roots.length === 0) return;
    const plan = buildMergedPlan(roots);
    if (plan.nodes.length === 0) return;
    await roundTripAndExecute(plan);
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
    await roundTripAndExecute(plan);
    postExecuteBookkeeping(plan, pending);
    for (const node of plan.nodes) {
      node.result = undefined;
    }
  };
}

// ============================================================================
// Tiny MLP training loop
// ============================================================================

const XOR_INPUTS = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const XOR_TARGETS = [0, 1, 1, 0];

interface TrainConfig {
  seed: number;
  steps: number;
  lr: number;
  hidden: number;
  device: DeviceKind;
  init: { W1: number[]; b1: number[]; W2: number[]; b2: number[] };
}

function makePrng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return ((s >>> 0) / 0x100000000) * 2 - 1;
  };
}

function generateInit(seed: number, hidden: number): TrainConfig["init"] {
  const rng = makePrng(seed);
  const scale = 0.5;
  const draw = (n: number) => Array.from({ length: n }, () => rng() * scale);
  return {
    W1: draw(2 * hidden),
    b1: new Array(hidden).fill(0),
    W2: draw(hidden * 1),
    b2: new Array(1).fill(0),
  };
}

async function runTraining(
  api: Torchlette,
  cfg: TrainConfig,
): Promise<{ losses: number[]; finalWeights: number[][] }> {
  const { hidden, steps, lr, device, init } = cfg;
  const opts = { device };

  const W1 = api
    .tensorFromArray(init.W1, [2, hidden], opts)
    .requires_grad_(true);
  const b1 = api
    .tensorFromArray(init.b1, [hidden], opts)
    .requires_grad_(true);
  const W2 = api
    .tensorFromArray(init.W2, [hidden, 1], opts)
    .requires_grad_(true);
  const b2 = api.tensorFromArray(init.b2, [1], opts).requires_grad_(true);

  const X = api.tensorFromArray(XOR_INPUTS.flat(), [4, 2], opts);
  const T = api.tensorFromArray(XOR_TARGETS, [4, 1], opts);

  const losses: number[] = [];
  for (let step = 0; step < steps; step++) {
    const h = api.relu(api.add(api.matmul(X, W1), b1));
    const y = api.add(api.matmul(h, W2), b2);
    const diff = api.sub(y, T);
    const sq = api.mul(diff, diff);
    const loss = sq.mean();
    if (typeof loss === "number") throw new Error("loss should be Tensor");

    losses.push(await loss.item());

    await loss.backward();

    for (const p of [W1, b1, W2, b2]) {
      if (!p.grad) throw new Error("missing grad");
      api.noGrad(() => {
        const updated = api.sub(p, api.mul(p.grad!, lr));
        p.copy_(updated);
      });
      p.zeroGrad();
    }
  }

  const finalWeights = [
    await W1.cpu(),
    await b1.cpu(),
    await W2.cpu(),
    await b2.cpu(),
  ];
  return { losses, finalWeights };
}

// ============================================================================
// The actual test
// ============================================================================

describe("remote wire format: full training loop through round-trip", () => {
  it("XOR MLP: bit-exact loss and weights vs local-only baseline", async () => {
    const hidden = 4;
    const init = generateInit(12345, hidden);
    const cfg: TrainConfig = {
      seed: 12345,
      steps: 20,
      lr: 0.1,
      hidden,
      device: "cpu",
      init,
    };

    // Local baseline.
    const baseline = new Torchlette("cpu");
    const baselineResult = await runTraining(baseline, cfg);

    // Round-trip: same code, every plan serializes through JSON before exec.
    const rtrip = new Torchlette("cpu");
    patchEngineForRoundTrip(rtrip);
    const rtripResult = await runTraining(rtrip, cfg);

    // Bit-exact comparison.
    expect(rtripResult.losses).toEqual(baselineResult.losses);
    expect(rtripResult.finalWeights[0]).toEqual(baselineResult.finalWeights[0]);
    expect(rtripResult.finalWeights[1]).toEqual(baselineResult.finalWeights[1]);
    expect(rtripResult.finalWeights[2]).toEqual(baselineResult.finalWeights[2]);
    expect(rtripResult.finalWeights[3]).toEqual(baselineResult.finalWeights[3]);

    // Sanity: training actually did something.
    expect(rtripResult.losses[cfg.steps - 1]).toBeLessThan(
      rtripResult.losses[0] * 0.6,
    );
  });
});
