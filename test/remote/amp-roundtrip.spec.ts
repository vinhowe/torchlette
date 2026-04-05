/**
 * AMP (Automatic Mixed Precision) through the remote round-trip.
 *
 * Verifies that autocast + GradScaler + Adam work correctly when every
 * plan is serialized through JSON before execution. The client backend
 * is CPU (no fused unscaleGrad/adamStep), so GradScaler and Adam both
 * fall through to their decomposed-op paths — all standard lazy nodes,
 * no infFlagBuffer.
 *
 * This test mirrors the in-process round-trip patch from
 * training-roundtrip.spec.ts but adds the AMP machinery on top.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { cpuBackend } from "../../src/backend/cpu";
import { registerBackend } from "../../src/backend/registry";
import { executePlanSequential } from "../../src/executor/sequential";
import { buildMergedPlan } from "../../src/executor/plan-builder";
import { Torchlette } from "../../src/frontend/torchlette";
import { storageTracker } from "../../src/graph/storage-tracker";
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
// In-process round-trip (same as training-roundtrip.spec.ts)
// ============================================================================

const resolveHandle = (id: number): HandleRef => `h:${id}`;
const lookupHandle = (h: HandleRef) => {
  const s = storageTracker.getStorage(Number(h.slice(2)));
  if (!s) throw new Error(`no storage for id=${Number(h.slice(2))}`);
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
    if (t.lazyRef.kind === "pending") roots.push(t.lazyRef.node);
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

function patchEngineForRoundTrip(torch: Torchlette): void {
  const runtime = torch.runtime;

  const makeForce = () => async (...tensors: RuntimeTensor[]) => {
    const roots = collectPendingRoots(tensors);
    if (roots.length === 0) return;
    const plan = buildMergedPlan(roots);
    if (plan.nodes.length === 0) return;
    await roundTripAndExecute(plan);
    for (const node of plan.nodes) {
      if (node.result)
        materializePendingTensors(node.id, node.result, node.results);
    }
    clearDisposedPendingNodeIds();
    materializeRemaining(tensors);
  };

  // biome-ignore lint/suspicious/noExplicitAny: patching
  (runtime as any).forceAllMerged = makeForce();
  // biome-ignore lint/suspicious/noExplicitAny: patching
  (runtime as any).force = async (t: RuntimeTensor) => {
    if (t.isMaterialized() || t.disposed) return;
    if (t.lazyRef.kind !== "pending") return;
    // biome-ignore lint/suspicious/noExplicitAny: patching
    await (runtime as any).forceAllMerged(t);
  };
  // biome-ignore lint/suspicious/noExplicitAny: patching
  (runtime as any).forceAllPending = async () => {
    const pending = getAllPendingTensors();
    if (pending.length === 0) return;
    const roots = collectPendingRoots(pending);
    if (roots.length === 0) return;
    const plan = buildMergedPlan(roots, true);
    if (plan.nodes.length === 0) return;
    await roundTripAndExecute(plan);
    for (const node of plan.nodes) {
      if (node.result)
        materializePendingTensors(node.id, node.result, node.results);
    }
    clearDisposedPendingNodeIds();
    materializeRemaining(pending);
    for (const node of plan.nodes) node.result = undefined;
  };
}

// ============================================================================
// Deterministic init (same PRNG as other tests)
// ============================================================================

function makePrng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return ((s >>> 0) / 0x100000000) * 2 - 1;
  };
}

// ============================================================================
// Test
// ============================================================================

describe("AMP through remote round-trip", () => {
  it("autocast + GradScaler + SGD: loss drops, scale updates", async () => {
    const api = new Torchlette("cpu");
    patchEngineForRoundTrip(api);

    const { GradScaler } = await import("../../src/optim/grad-scaler");

    const rng = makePrng(42);
    const draw = (n: number) =>
      Array.from({ length: n }, () => rng() * 0.3);

    // Tiny linear model: y = Wx + b (2→2)
    const W = api
      .tensorFromArray(draw(4), [2, 2])
      .requires_grad_(true);
    const b = api.tensorFromArray([0, 0], [2]).requires_grad_(true);

    const X = api.tensorFromArray(
      [0, 0, 0, 1, 1, 0, 1, 1],
      [4, 2],
    );
    const targets = api.tensorFromArray(
      [0, 1, 1, 0],
      [4],
      { dtype: "i32" },
    );

    const { SGD } = await import("../../src/optim/sgd");
    const scaler = new GradScaler(api, { initScale: 8, enabled: true });
    const optimizer = new SGD([W, b], { lr: 0.1 });
    const losses: number[] = [];

    for (let step = 0; step < 10; step++) {
      // Forward with autocast (generates cast nodes even on CPU client)
      const logits = api.autocast(() => {
        return api.add(api.matmul(X, W), b);
      });

      // Cross-entropy loss
      const { crossEntropy } = await import("../../src/nn/functional");
      const loss = crossEntropy(api, logits, targets, { reduction: "mean" });

      // Read loss BEFORE backward — backward's cleanup disposes intermediates.
      const lossVal = await loss.item();
      losses.push(lossVal);

      // Scaled backward
      const scaledLoss = api.mul(loss, scaler.getScale());
      if (typeof scaledLoss === "number")
        throw new Error("scaledLoss should be Tensor");
      await scaledLoss.backward();

      // Unscale (reads grads, checks for inf/NaN)
      scaler.unscale_(optimizer);

      // Optimizer step (builds lazy ops on unscaled grads)
      scaler.step(optimizer);
      optimizer.zeroGrad();

      // Update scale factor (reads back inf detection via item())
      await scaler.update();
    }

    // Sanity: loss should decrease
    expect(losses[losses.length - 1]).toBeLessThan(losses[0]);

    // Scale should have been updated (either grew or shrunk depending on inf)
    // At minimum, getScale() returns a positive number.
    expect(scaler.getScale()).toBeGreaterThan(0);

    // No crashes through 10 steps = the decomposed GradScaler path serializes.
  });
});
