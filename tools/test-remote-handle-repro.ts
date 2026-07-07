/**
 * Reproduce "no HandleRef for storage id=X" in remote hooks path.
 * Uses a mock execution hook with REAL stub lifecycle (including destroy()
 * that removes HandleRefs) — matching remote-hooks.ts exactly.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";
import { createStorageHandle } from "../src/graph/node-factory";
import type { ExecutionPlan } from "../src/graph/types";
import type { BackendTensor } from "../src/backend/types";
import { serializePlan } from "../src/remote/serialize";

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA * 2 + 1, S = 10, B = 32;

  const handleMap = new Map<number, string>();
  let nextHandle = 1;
  let destroyedCount = 0;

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    executionHook: async (plan: ExecutionPlan) => {
      if (plan.nodes.length === 0) return;

      // Serialize — this is where "no HandleRef" would throw
      try {
        serializePlan(plan, {
          resolveHandle: (storageId: number) => {
            const h = handleMap.get(storageId);
            if (!h) {
              console.error(`\nNO HANDLE for storage id=${storageId}`);
              for (let i = 0; i < plan.nodes.length; i++) {
                for (const ref of plan.nodes[i].inputs) {
                  if (ref.kind === "materialized" && ref.storage.id === storageId) {
                    console.error(`  Referenced by node[${i}] op=${plan.nodes[i].op} shape=${plan.nodes[i].shape}`);
                  }
                }
              }
              throw new Error(`MockRemoteHook: no HandleRef for storage id=${storageId}`);
            }
            return h;
          },
        });
      } catch (e) {
        console.error("Serialization failed:", (e as Error).message);
      }

      // Execute locally
      const { executePlanSequential } = await import("../src/executor/sequential");
      const backend = api.runtime.getBackend("webgpu");
      await executePlanSequential(plan, backend, { enableEarlyRelease: false });

      // Only create stubs for output nodes — exactly like remote-hooks.ts
      const outputs = plan.outputIndices ?? new Set<number>();
      let registered = 0;
      for (let i = 0; i < plan.nodes.length; i++) {
        if (!outputs.has(i)) continue;
        const node = plan.nodes[i];
        if (!node.result) continue;

        const handleRef = `h${nextHandle++}`;
        const realBt = node.result.backendTensor;
        const stub: BackendTensor = {
          shape: [...node.shape],
          dtype: node.dtype,
          ownsBuffer: true,
          toArray() { return realBt.toArray(); },
          destroy() {
            handleMap.delete(stubStorage.id);
            destroyedCount++;
          },
        };
        const stubStorage = createStorageHandle(node.device, stub);
        handleMap.set(stubStorage.id, handleRef);
        node.result = stubStorage;
        registered++;

        // Side outputs
        if (node.results) {
          for (let j = 0; j < node.results.length; j++) {
            const r = node.results[j];
            if (!r) continue;
            const sideHandleRef = `h${nextHandle++}`;
            const sideRealBt = r.backendTensor;
            const sideStub: BackendTensor = {
              shape: [...node.shape],
              dtype: node.dtype,
              ownsBuffer: true,
              toArray() { return sideRealBt.toArray(); },
              destroy() {
                handleMap.delete(sideStubStorage.id);
                destroyedCount++;
              },
            };
            const sideStubStorage = createStorageHandle(node.device, sideStub);
            handleMap.set(sideStubStorage.id, sideHandleRef);
            node.results[j] = sideStubStorage;
            registered++;
          }
        }
      }
      console.log(`  hook: ${plan.nodes.length} nodes, ${outputs.size} outputs, ${registered} stubs, destroyed=${destroyedCount}`);
    },
  });
  api.manualSeed(42);
  const m = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });
  const o = new Adam(m.parameters(), { lr: 1e-3 });

  console.log(`handleMap size after init: ${handleMap.size}`);

  for (let step = 0; step < 3; step++) {
    await api.beginStep();
    console.log(`\n=== Step ${step} ===`);
    console.log(`handleMap: ${handleMap.size}, destroyed: ${destroyedCount}`);

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
    console.log(`loss: ${v.toFixed(6)}, handleMap: ${handleMap.size}, destroyed: ${destroyedCount}`);

    await l.backward();
    console.log(`after bwd: handleMap: ${handleMap.size}, destroyed: ${destroyedCount}`);
    l.dispose();

    o.step(); o.zeroGrad();
    api.endStep();
    console.log(`after endStep: handleMap: ${handleMap.size}, destroyed: ${destroyedCount}`);
  }

  console.log("\nDONE — no handle errors");
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
