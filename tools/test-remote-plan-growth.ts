/**
 * Test that plan sizes stay stable across training steps when using
 * an execution hook (simulating the remote training path).
 *
 * The key invariant: plan sizes should NOT grow linearly per step.
 * If they do, stale node.result is accumulating.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";
import type { ExecutionPlan, LazyIRNode } from "../src/graph/types";

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA * 2 + 1, S = 10, B = 32;

  const planSizes: number[][] = []; // per step, list of plan sizes

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    executionHook: async (plan: ExecutionPlan, backend) => {
      if (plan.nodes.length === 0) return;
      // Track plan size
      if (planSizes.length > 0) {
        planSizes[planSizes.length - 1].push(plan.nodes.length);
      }
      // Simulate server path: toposort + executePlanOptimized with externalNodeIds
      const { getPendingNodeIds } = await import("../src/runtime/tensor");
      const { executePlanOptimized } = await import("../src/executor/executor");
      const pendingIds = getPendingNodeIds();
      const externalNodeIds = new Set<number>();
      for (const node of plan.nodes) {
        if (pendingIds.has(node.id)) externalNodeIds.add(node.id);
      }
      // Toposort (same as server does on deserialized plans)
      const nodeSet = new Set(plan.nodes);
      const sorted: LazyIRNode[] = [];
      const visited = new Set<LazyIRNode>();
      function visit(node: LazyIRNode) {
        if (visited.has(node)) return;
        visited.add(node);
        for (const ref of node.inputs) {
          if (ref.kind === "pending" && nodeSet.has(ref.node)) visit(ref.node);
        }
        sorted.push(node);
      }
      for (const node of plan.nodes) visit(node);
      plan.nodes = sorted;

      if (externalNodeIds.size > 0) {
        await executePlanOptimized(plan, backend, {
          enableFusion: true,
          externalNodeIds,
        });
      } else {
        const { executePlanSequential } = await import("../src/executor/sequential");
        await executePlanSequential(plan, backend, { enableEarlyRelease: false });
      }
    },
  });
  api.manualSeed(42);
  const m = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });
  const o = new Adam(m.parameters(), { lr: 1e-3 });

  const STEPS = 8;
  for (let step = 0; step < STEPS; step++) {
    planSizes.push([]);
    await api.beginStep();
    const b = generateBatchWithCompartments({ seqLen: S, batchSize: B }, 2);
    const t = api.tensorFromArray(b.tokens, [B, S], { dtype: "i32" });
    const g = api.tensorFromArray(b.targets as any, [B * (S - 1)], { dtype: "i32" });
    const l = api.tidy(() => {
      const f = m.forward(t);
      const lg = f.logits.narrow(1, 0, S - 1).contiguous().reshape([B * (S - 1), V]);
      const x = crossEntropy(api, lg, g); api.keep(x); return x;
    });
    t.dispose(); g.dispose();
    const v = await l.item();
    await l.backward(); l.dispose();
    o.step(); o.zeroGrad();
    api.endStep();
  }

  // Report
  console.log("Plan sizes per step:");
  let maxTotal = 0;
  for (let i = 0; i < planSizes.length; i++) {
    const total = planSizes[i].reduce((a, b) => a + b, 0);
    maxTotal = Math.max(maxTotal, total);
    console.log(`  step ${i}: ${planSizes[i].join(" + ")} = ${total} total nodes`);
  }

  // Check: steps 2+ should have similar total plan sizes (within 10% of step 1)
  const step1Total = planSizes[1].reduce((a, b) => a + b, 0);
  const lastTotal = planSizes[STEPS - 1].reduce((a, b) => a + b, 0);
  const growth = lastTotal / step1Total;
  console.log(`\nGrowth factor (step ${STEPS-1} / step 1): ${growth.toFixed(2)}x`);
  if (growth > 1.15) {
    console.error(`FAIL: Plans growing! ${step1Total} → ${lastTotal} (${((growth - 1) * 100).toFixed(0)}% growth)`);
    process.exit(1);
  } else {
    console.log("PASS: Plan sizes stable across steps");
  }
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
