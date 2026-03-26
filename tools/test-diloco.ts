/**
 * DiLoCo Local Simulation
 *
 * Simulates K workers doing DiLoCo training on a tiny GPT-2 model,
 * all in one process. Each worker trains on a different data shard
 * for H steps, then pseudo-gradients are averaged and the outer
 * optimizer updates global params.
 *
 * Compares against a single-worker baseline with the same total steps
 * to verify DiLoCo matches or exceeds single-worker quality.
 */

import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { e3m0Dequantize, e3m0Quantize } from "../src/distributed";
import { Torchlette } from "../src/frontend/torchlette";
import { nn } from "../src/nn";
import { Adam } from "../src/optim";

// Tiny GPT-2 config for fast iteration
const TINY_CONFIG = {
  vocabSize: 256, // byte-level
  blockSize: 64,
  numLayers: 2,
  numHeads: 2,
  embedDim: 64,
  dropoutRate: 0,
};

// Training config
const K = 2; // number of simulated workers
const H = 50; // inner steps between syncs
const OUTER_ROUNDS = 5; // number of DiLoCo sync rounds
const SEQ_LEN = 32;
const BATCH_SIZE = 1;
const LR = 1e-3;

/** Generate random byte-level text data. */
function generateData(length: number): number[] {
  const data: number[] = [];
  // Simple repeating pattern so the model can learn something
  const pattern = "the cat sat on the mat. the dog ran in the park. ";
  for (let i = 0; i < length; i++) {
    data.push(pattern.charCodeAt(i % pattern.length));
  }
  return data;
}

/** Create a tiny GPT-2 model. */
async function createModel(api: Torchlette) {
  // Import model class dynamically
  const { GPT2WithLoRA } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora"
  );
  const loraConfig = { rank: 4, alpha: 4 };
  const model = new GPT2WithLoRA(api, TINY_CONFIG, loraConfig, "webgpu");
  return model;
}

/** Run one forward+backward step, return loss value. */
async function trainStep(
  api: Torchlette,
  model: any,
  optimizer: Adam,
  tokens: number[],
  offset: number,
): Promise<number> {
  const seqLen = SEQ_LEN;
  const start = offset % (tokens.length - seqLen - 1);
  const inputData = tokens.slice(start, start + seqLen);
  const targetData = tokens.slice(start + 1, start + seqLen + 1);

  await api.beginStep();

  const input = api.tensorFromArray(inputData, [1, seqLen], {
    device: "webgpu",
  });
  const target = api.tensorFromArray(targetData, [1, seqLen], {
    device: "webgpu",
  });

  const loss = api.tidy(() => {
    const { loss: l } = model.forwardWithLoss(input, target);
    api.keep(l);
    return l;
  });

  const lossVal = await loss.item();
  await loss.backward();
  optimizer.step();
  optimizer.zeroGrad();

  input.dispose();
  target.dispose();
  api.endStep();
  await api.markStep();

  return lossVal;
}

async function main() {
  console.log("=== DiLoCo Local Simulation ===\n");
  console.log(
    `Config: ${K} workers, H=${H} inner steps, ${OUTER_ROUNDS} rounds`,
  );
  console.log(
    `Model: tiny GPT-2 (${TINY_CONFIG.numLayers}L, ${TINY_CONFIG.embedDim}d)\n`,
  );

  const ok = await initWebGPU();
  if (!ok) {
    console.error("WebGPU not available");
    process.exit(1);
  }

  // Generate data shards (different data per worker)
  const shards: number[][] = [];
  for (let w = 0; w < K; w++) {
    const data = generateData(2000);
    // Rotate data to give each worker a different view
    const rotated = [...data.slice(w * 500), ...data.slice(0, w * 500)];
    shards.push(rotated);
  }

  // ================================================================
  // Baseline: single worker, K*H*OUTER_ROUNDS total steps
  // ================================================================
  console.log("--- Baseline (single worker) ---");
  {
    const api = new Torchlette("webgpu", { enableFusion: true });
    const model = await createModel(api);
    model.train(true);
    const params = model.getLoRAParameters();
    const optimizer = new Adam(params, { lr: LR }, api);
    const totalSteps = K * H * OUTER_ROUNDS;
    const losses: number[] = [];

    for (let step = 0; step < totalSteps; step++) {
      const l = await trainStep(
        api,
        model,
        optimizer,
        shards[0],
        step * SEQ_LEN,
      );
      losses.push(l);
      if (step % 50 === 0 || step === totalSteps - 1) {
        const recent = losses.slice(-10);
        const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
        console.log(`  step ${step}: loss=${avg.toFixed(4)}`);
      }
    }

    const finalAvg =
      losses.slice(-20).reduce((a, b) => a + b, 0) /
      Math.min(20, losses.length);
    console.log(`  Final avg loss: ${finalAvg.toFixed(4)}\n`);
  }

  // ================================================================
  // DiLoCo: K workers, H inner steps, OUTER_ROUNDS sync rounds
  // ================================================================
  console.log("--- DiLoCo (simulated distributed) ---");
  {
    const api = new Torchlette("webgpu", { enableFusion: true });
    const model = await createModel(api);
    model.train(true);
    const params = model.getLoRAParameters();

    const { NesterovOuterOptimizer } = await import(
      "../src/distributed/outer-optimizer"
    );
    const outerOpt = new NesterovOuterOptimizer(api, {
      lr: 1.0,
      momentum: 0,
    });

    const allLosses: number[] = [];

    for (let round = 0; round < OUTER_ROUNDS; round++) {
      // Save the global snapshot (CPU copy of all params)
      const globalSnapshot: Float32Array[] = [];
      for (const p of params) {
        globalSnapshot.push(new Float32Array(await p.cpu()));
      }

      // Simulate K workers doing H inner steps each.
      // Each worker starts from the same global params and trains independently.
      const workerPseudoGrads: Float32Array[][] = [];

      for (let w = 0; w < K; w++) {
        // Restore params to global snapshot before each worker
        await api.beginStep();
        for (let i = 0; i < params.length; i++) {
          const snap = api.tensorFromArray(
            Array.from(globalSnapshot[i]),
            params[i].shape,
            { device: params[i].device },
          );
          api.copy_(params[i], snap);
          // Don't dispose — let markStep handle cleanup
        }
        api.endStep();
        await api.markStep();

        // Fresh Adam per worker (each worker has independent optimizer state)
        const workerOpt = new Adam(params, { lr: LR }, api);

        const workerLosses: number[] = [];
        for (let step = 0; step < H; step++) {
          const offset = (round * H + step) * SEQ_LEN + w * 1000;
          const l = await trainStep(api, model, workerOpt, shards[w], offset);
          workerLosses.push(l);
        }
        allLosses.push(...workerLosses);

        const avg =
          workerLosses.reduce((a, b) => a + b, 0) / workerLosses.length;
        console.log(
          `  round ${round}, worker ${w}: avg loss=${avg.toFixed(4)}`,
        );

        // Compute pseudo-gradients: local_params - global_snapshot
        const pseudoGrads: Float32Array[] = [];
        for (let i = 0; i < params.length; i++) {
          const localData = await params[i].cpu();
          const delta = new Float32Array(localData.length);
          for (let j = 0; j < delta.length; j++) {
            delta[j] = localData[j] - globalSnapshot[i][j];
          }
          pseudoGrads.push(delta);
        }
        workerPseudoGrads.push(pseudoGrads);
      }

      // Restore to global snapshot before applying outer update
      await api.beginStep();
      for (let i = 0; i < params.length; i++) {
        const snap = api.tensorFromArray(
          Array.from(globalSnapshot[i]),
          params[i].shape,
          { device: params[i].device },
        );
        api.copy_(params[i], snap);
        snap.dispose();
      }
      api.endStep();
      await api.markStep();

      // Average pseudo-gradients across workers
      const avgPseudoGrads: Float32Array[] = [];
      for (let p = 0; p < params.length; p++) {
        const size = workerPseudoGrads[0][p].length;
        const avg = new Float32Array(size);
        for (let w = 0; w < K; w++) {
          const wg = workerPseudoGrads[w][p];
          for (let i = 0; i < size; i++) {
            avg[i] += wg[i] / K;
          }
        }
        avgPseudoGrads.push(avg);
      }

      // Test E3M0 compression on the pseudo-gradients
      let totalOrigBytes = 0;
      let totalCompBytes = 0;
      for (const pg of avgPseudoGrads) {
        // Pad to multiple of 8
        const padded = new Float32Array(Math.ceil(pg.length / 8) * 8);
        padded.set(pg);
        const { codes, scales } = e3m0Quantize(padded);
        const restored = e3m0Dequantize(codes, scales, padded.length);
        totalOrigBytes += pg.length * 4;
        totalCompBytes += codes.byteLength + scales.byteLength;
      }
      console.log(
        `  round ${round}: compression ${(totalOrigBytes / totalCompBytes).toFixed(1)}x (${(totalOrigBytes / 1024).toFixed(0)}KB → ${(totalCompBytes / 1024).toFixed(0)}KB)`,
      );

      // Restore to global snapshot before applying outer update
      await api.beginStep();
      for (let i = 0; i < params.length; i++) {
        const snap = api.tensorFromArray(
          Array.from(globalSnapshot[i]),
          params[i].shape,
          { device: params[i].device },
        );
        api.copy_(params[i], snap);
      }
      api.endStep();
      await api.markStep();

      // Apply outer Nesterov update (CPU-side math, writes back to GPU)
      const avgTensors = avgPseudoGrads.map((pg, i) =>
        api.tensorFromArray(Array.from(pg), params[i].shape, {
          device: "webgpu",
        }),
      );
      await outerOpt.step(params, avgTensors);
      for (const t of avgTensors) t.dispose();
      await api.markStep();
    }

    const finalAvg =
      allLosses.slice(-20).reduce((a, b) => a + b, 0) /
      Math.min(20, allLosses.length);
    console.log(`\n  Final avg loss: ${finalAvg.toFixed(4)}`);

    outerOpt.dispose();
  }

  console.log("\n=== Done ===");
  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
