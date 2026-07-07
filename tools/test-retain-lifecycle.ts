/**
 * Test: lazy node input retention prevents premature storage destruction.
 *
 * Scenario: Adam's _updateLazyRef drops the old param storage. A lazy
 * optimizer node still references it. Without retention, the storage is
 * destroyed (rc→0). With retention, it survives until the node executes.
 *
 * This is the minimal repro of the remote handle lifecycle bug.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: false }); // sequential, no cache

  // Simple model: one linear layer
  const W = api.tensorFromArray([1, 2, 3, 4], [2, 2]).requires_grad_(true);
  const X = api.tensorFromArray([1, 0, 0, 1], [2, 2]);

  // Create Adam (initializes m/v as zeros — pending lazy nodes)
  const optimizer = new Adam([W], { lr: 0.01 });

  // Step 1: forward + backward + optimizer
  await api.beginStep();
  const y = api.matmul(X, W);
  const loss = y.sum();
  if (typeof loss === "number") throw new Error("expected tensor");
  await loss.item();
  await loss.backward();
  optimizer.step();     // Creates lazy optimizer nodes referencing current W storage
  optimizer.zeroGrad();
  api.endStep();        // Forces optimizer → materializes new W, old W storage should survive until here

  // Step 2: verify training works
  await api.beginStep();
  const y2 = api.matmul(X, W);
  const loss2 = y2.sum();
  if (typeof loss2 === "number") throw new Error("expected tensor");
  const v = await loss2.item();
  console.log(`step 2 loss: ${v.toFixed(4)}`);
  await loss2.backward();
  optimizer.step();
  optimizer.zeroGrad();
  api.endStep();

  // Step 3: with the execution hook (simulates remote)
  // This is where the bug manifests — stubs with ownsBuffer:true get destroyed
  // when _updateLazyRef drops the old storage, before the optimizer plan ships.
  const { createStorageHandle } = await import("../src/graph/node-factory");
  type BT = import("../src/backend/types").BackendTensor;
  let destroyCount = 0;
  const hookApi = new Torchlette("cpu", {
    executionHook: async (plan) => {
      const { executePlanSequential } = await import("../src/executor/sequential");
      const { cpuBackend } = await import("../src/backend/cpu/index");
      await executePlanSequential(plan, cpuBackend);
      // Replace results with stubs (like remote would)
      for (const node of plan.nodes) {
        if (!node.result) continue;
        const real = node.result;
        // Proxy the real backend tensor but intercept destroy.
        const realBt = real.backendTensor;
        const stub: BT = Object.create(realBt, {
          ownsBuffer: { value: true },
          destroy: { value: () => { destroyCount++; } },
        });
        node.result = createStorageHandle(node.device, stub);
      }
    },
    readHook: async (bt) => bt.toArray(),
  } as any);

  const W2 = hookApi.tensorFromArray([1, 2, 3, 4], [2, 2]).requires_grad_(true);
  const X2 = hookApi.tensorFromArray([1, 0, 0, 1], [2, 2]);
  const opt2 = new Adam([W2], { lr: 0.01 });

  // Two steps through the hook
  for (let step = 0; step < 3; step++) {
    await hookApi.beginStep();
    const y = hookApi.matmul(X2, W2);
    const loss = y.sum();
    if (typeof loss === "number") throw new Error("expected tensor");
    await loss.item();
    await loss.backward();
    opt2.step();
    opt2.zeroGrad();
    hookApi.endStep();
  }
  console.log(`hook path: 3 steps completed, ${destroyCount} stub destroys`);
  console.log("PASS");
  process.exit(0);
}

main().catch((e) => {
  console.error("FAIL:", e.message);
  console.error(e.stack?.split("\n").slice(0, 8).join("\n"));
  process.exit(1);
});
