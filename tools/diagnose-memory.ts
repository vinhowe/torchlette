/**
 * Diagnose GPU memory lifecycle during LoRA training.
 * Tracks storage tracker state and buffer pool state at every phase
 * of the training step to identify where memory grows.
 */

// Access storage tracker internals for diagnostics
import { storageTracker } from "../src/graph/storage-tracker";
import {
  Adam,
  GradScaler,
  getGPUMemoryStats,
  initWebGPU,
  nn,
  Torchlette,
} from "../src/index";
import {
  getTensorDebugStats,
  resetTensorDebugStats,
} from "../src/runtime/tensor";

async function main() {
  await initWebGPU();
  const api = new Torchlette();

  // Create a tiny model to exercise the full training path:
  // Linear(32, 64) -> ReLU -> Linear(64, 10) -> CrossEntropy
  const W1 = api.randn([32, 64], { device: "webgpu", requiresGrad: true });
  const W2 = api.randn([64, 10], { device: "webgpu", requiresGrad: true });
  const params = [W1, W2];

  const optimizer = new Adam(params, { lr: 1e-3 }, api);
  // No GradScaler - Dawn lacks shader-f16 on this machine

  function snapshot(label: string) {
    const st = storageTracker.stats();
    const gpu = getGPUMemoryStats();
    const rt = getTensorDebugStats();
    console.log(
      `  [${label}] storages: total=${st.totalStorages} reachable=${st.reachableStorages} unreachable=${st.unreachableStorages} | ` +
        `GPU: ${(gpu.currentBytes / 1024 / 1024).toFixed(1)}MB buffers=${gpu.currentBuffers} | ` +
        `RT: created=${rt.created} disposed=${rt.disposed} materialized=${rt.materialized}`,
    );
  }

  for (let step = 0; step < 5; step++) {
    console.log(`\n=== Step ${step} ===`);
    resetTensorDebugStats();

    snapshot("pre-beginStep");
    await api.beginStep();
    snapshot("post-beginStep");

    // Create batch data
    const x = api.randn([4, 32], { device: "webgpu" });
    const targetData = Array.from({ length: 4 }, () =>
      Math.floor(Math.random() * 10),
    );
    const target = api.tensorFromArray(targetData, [4], {
      device: "webgpu",
      dtype: "i32",
    });

    // Forward pass in tidy (no compile, no AMP)
    const loss = api.tidy(() => {
      const h1 = api.relu(api.matmul(x, W1));
      const logits = api.matmul(h1, W2);
      const l = nn.crossEntropy(api, logits, target);
      api.keep(l);
      return l;
    });

    snapshot("post-forward");

    // Get loss value
    const lossVal = await loss.item();
    console.log(`  loss = ${lossVal.toFixed(4)}`);
    snapshot("post-item");

    // Backward
    await loss.backward();
    snapshot("post-backward");

    // Optimizer step (no GradScaler, no clip — avoid pow dispatch issue)
    optimizer.step();
    optimizer.zeroGrad();
    snapshot("post-optimizer");

    // Dispose batch tensors
    x.dispose();
    target.dispose();
    snapshot("post-dispose-batch");

    // End step
    api.endStep();
    snapshot("post-endStep");

    // Mark step (GPU fence + cleanup)
    await api.markStep();
    snapshot("post-markStep");
  }

  console.log("\n=== Final storage tracker state ===");
  const finalStats = storageTracker.stats();
  console.log(`Total storages: ${finalStats.totalStorages}`);
  console.log(`Reachable: ${finalStats.reachableStorages}`);
  console.log(`Unreachable: ${finalStats.unreachableStorages}`);

  const gpuFinal = getGPUMemoryStats();
  console.log(`GPU buffers: ${gpuFinal.currentBuffers}`);
  console.log(
    `GPU memory: ${(gpuFinal.currentBytes / 1024 / 1024).toFixed(1)}MB`,
  );

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
