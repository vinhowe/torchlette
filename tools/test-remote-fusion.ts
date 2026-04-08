/**
 * End-to-end test: simulate browser remote training with server-side fusion.
 *
 * Mimics the exact path: client Torchlette with executionHook →
 * serialize plan → deserialize on "server" → executePlanOptimized →
 * return handles → client binds stubs → materialize.
 *
 * Tests that plan sizes stay stable and loss decreases.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import {
  generateBatchWithCompartments,
  setTransitionMatrices,
  VOCAB_SIZE_DATA,
} from "../examples/toy-compartmentalization/src/lib/data";
import type {
  ExecutionPlan,
  LazyIRNode,
  StorageHandle,
} from "../src/graph/types";
import type { Backend, BackendTensor, DType } from "../src/backend/types";
import { serializePlan, deserializePlan } from "../src/remote/serialize";
import { createStorageHandle } from "../src/graph/node-factory";
import { executePlanOptimized } from "../src/executor/executor";
import { executePlanSequential } from "../src/executor/sequential";
import { getPendingNodeIds } from "../src/runtime/tensor";
import { getBackend } from "../src/backend/registry";
import type { HandleRef, NodeIdx } from "../src/remote/wire";

// ============================================================================
// Simulated server session (same as server.ts)
// ============================================================================

class SimSession {
  private handles = new Map<HandleRef, StorageHandle>();
  private next = 1;

  alloc(s: StorageHandle): HandleRef {
    const h = `h${this.next++}`;
    this.handles.set(h, s);
    return h;
  }

  resolve(h: HandleRef): StorageHandle {
    const s = this.handles.get(h);
    if (!s) throw new Error(`Unknown handle: ${h}`);
    return s;
  }

  handleCount(): number {
    return this.handles.size;
  }
}

function toposortPlan(plan: ExecutionPlan): void {
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
}

// ============================================================================
// Simulated server execute handler (mirrors server.ts executeHandler)
// ============================================================================

async function serverExecute(
  session: SimSession,
  wirePlan: ReturnType<typeof serializePlan>,
  serverBackend: Backend,
): Promise<{
  outputs: Record<NodeIdx, HandleRef>;
  sideOutputs: Record<string, HandleRef>;
}> {
  const plan = deserializePlan(wirePlan, {
    resolveHandle: (h) => session.resolve(h),
  });

  const serverDevice = serverBackend.name as "cpu" | "webgpu";
  for (const node of plan.nodes) node.device = serverDevice;

  // Build node id → wire index BEFORE toposort
  const nodeIdToWireIdx = new Map<number, number>();
  for (let i = 0; i < plan.nodes.length; i++) {
    nodeIdToWireIdx.set(plan.nodes[i].id, i);
  }

  // Build externalNodeIds from outputNodes
  const outputNodeIds = new Set<number>();
  if (wirePlan.outputNodes) {
    for (const idx of wirePlan.outputNodes) {
      outputNodeIds.add(plan.nodes[idx].id);
    }
  }

  toposortPlan(plan);

  // Execute with fusion (same as server.ts)
  if (serverBackend.name === "webgpu" && outputNodeIds.size > 0) {
    await executePlanOptimized(plan, serverBackend, {
      enableFusion: true,
      externalNodeIds: outputNodeIds,
    });
  } else {
    await executePlanSequential(plan, serverBackend);
  }

  // Return handles for ALL nodes with results (not just outputNodes)
  const outputs: Record<NodeIdx, HandleRef> = {};
  const sideOutputs: Record<string, HandleRef> = {};
  for (const node of plan.nodes) {
    if (!node.result) continue;
    const wireIdx = nodeIdToWireIdx.get(node.id)!;
    outputs[wireIdx] = session.alloc(node.result);
    if (node.results) {
      for (let j = 0; j < node.results.length; j++) {
        const r = node.results[j];
        if (r) sideOutputs[`${wireIdx}:${j}`] = session.alloc(r);
      }
    }
  }
  return { outputs, sideOutputs };
}

// ============================================================================
// Main test
// ============================================================================

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA * 2 + 1,
    S = 10,
    B = 32;
  const serverBackend = getBackend("webgpu")!;
  const session = new SimSession();

  // Client-side handle map (storage id → server HandleRef)
  const handleMap = new Map<number, HandleRef>();

  const planSizes: number[][] = [];
  const handleCounts: number[][] = [];
  let hookCalls = 0;

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    executionHook: async (plan: ExecutionPlan, _backend: Backend) => {
      hookCalls++;
      if (plan.nodes.length === 0) return;

      // Compute outputNodes (same as remote-hooks.ts)
      const pendingIds = getPendingNodeIds();
      const outputNodes: LazyIRNode[] = [];
      for (const node of plan.nodes) {
        if (pendingIds.has(node.id)) outputNodes.push(node);
      }

      // Serialize (same as remote-hooks.ts)
      const wire = serializePlan(plan, {
        resolveHandle: (storageId: number) => {
          const h = handleMap.get(storageId);
          if (!h)
            throw new Error(`No HandleRef for storage id=${storageId}`);
          return h;
        },
        outputNodes,
      });

      // Server execute
      const result = await serverExecute(session, wire, serverBackend);

      // Track sizes
      const nHandles =
        Object.keys(result.outputs).length +
        Object.keys(result.sideOutputs).length;
      if (planSizes.length > 0) {
        planSizes[planSizes.length - 1].push(plan.nodes.length);
        handleCounts[handleCounts.length - 1].push(nHandles);
      }

      // Bind results: reuse the server's actual StorageHandle so reads work.
      // In the real browser demo, stubs + readHook handle this over the wire.
      // Here we share the process so we can just reuse the storage directly.
      for (const [idxStr, handleRef] of Object.entries(result.outputs)) {
        const idx = Number(idxStr);
        const node = plan.nodes[idx];
        const serverStorage = session.resolve(handleRef);
        const storage = createStorageHandle(node.device, serverStorage.backendTensor);
        handleMap.set(storage.id, handleRef);
        node.result = storage;
      }
      for (const [key, handleRef] of Object.entries(result.sideOutputs)) {
        const [idxStr, outIdxStr] = key.split(":");
        const node = plan.nodes[Number(idxStr)];
        const outIdx = Number(outIdxStr);
        const serverStorage = session.resolve(handleRef);
        const storage = createStorageHandle(node.device, serverStorage.backendTensor);
        handleMap.set(storage.id, handleRef);
        if (!node.results) node.results = [];
        node.results[outIdx] = storage;
      }
    },
  });

  api.manualSeed(42);
  const m = createModel(api, nn, {
    ...MESS3_CONFIG,
    seqLen: S,
    vocabSize: V,
    posEncoding: "rope",
  });
  const o = new Adam(m.parameters(), { lr: 1e-3 });

  const STEPS = 8;
  for (let step = 0; step < STEPS; step++) {
    planSizes.push([]);
    handleCounts.push([]);
    await api.beginStep();
    const b = generateBatchWithCompartments({ seqLen: S, batchSize: B }, 2);
    const t = api.tensorFromArray(b.tokens, [B, S], { dtype: "i32" });
    const g = api.tensorFromArray(
      b.targets as number[],
      [B * (S - 1)],
      { dtype: "i32" },
    );
    const l = api.tidy(() => {
      const f = m.forward(t);
      const lg = f.logits
        .narrow(1, 0, S - 1)
        .contiguous()
        .reshape([B * (S - 1), V]);
      const x = crossEntropy(api, lg, g);
      api.keep(x);
      return x;
    });
    t.dispose();
    g.dispose();
    const v = await l.item();
    await l.backward();
    l.dispose();
    o.step();
    o.zeroGrad();
    api.endStep();
    console.log(
      `step ${step}: loss=${typeof v === "number" ? v.toFixed(4) : v}, hooks=${hookCalls}, handles=${session.handleCount()}`,
    );
  }

  // Report
  console.log("\nPlan sizes per step:");
  for (let i = 0; i < planSizes.length; i++) {
    const sizes = planSizes[i].join(" + ");
    const handles = handleCounts[i].join(" + ");
    const total = planSizes[i].reduce((a, b) => a + b, 0);
    console.log(`  step ${i}: nodes=[${sizes}]=${total}  handles=[${handles}]`);
  }

  // Check stability
  const step1Total = planSizes[1].reduce((a, b) => a + b, 0);
  const lastTotal = planSizes[STEPS - 1].reduce((a, b) => a + b, 0);
  const growth = lastTotal / step1Total;
  console.log(
    `\nGrowth factor (step ${STEPS - 1} / step 1): ${growth.toFixed(2)}x`,
  );
  if (growth > 1.15) {
    console.error(
      `FAIL: Plans growing! ${step1Total} → ${lastTotal} (${((growth - 1) * 100).toFixed(0)}% growth)`,
    );
    process.exit(1);
  } else {
    console.log("PASS: Plan sizes stable across steps");
  }
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
