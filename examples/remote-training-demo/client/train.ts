/**
 * Training loop for the tiny transformer. Runs on top of any Torchlette —
 * the point is that the remote engine transparently ships plans to the
 * server. User code doesn't know or care.
 */

import { crossEntropy } from "../../../src/nn/functional.ts";
import type { Tensor, Torchlette } from "../../../src/frontend/torchlette.ts";
import type { Dataset, ModelConfig, TransformerModel } from "./model.ts";
import { forward, parameters, sampleBatch } from "./model.ts";
import type { RemoteEngine } from "../../../src/remote/client-engine.ts";

export interface TrainConfig {
  lr: number;
  batchSize: number;
  seqLen: number;
  seed: number;
}

/** A single gradient-descent step; returns the loss value (RPC). */
export async function trainStep(
  api: Torchlette,
  model: TransformerModel,
  ds: Dataset,
  params: Tensor[],
  cfg: TrainConfig,
  rng: () => number,
): Promise<number> {
  const { inputs, targets } = sampleBatch(ds, cfg.batchSize, cfg.seqLen, rng);

  const inputTensor = api.tensorFromArray(
    inputs,
    [cfg.batchSize, cfg.seqLen],
    { device: "cpu", dtype: "i32" },
  );
  const targetTensor = api.tensorFromArray(
    targets,
    [cfg.batchSize * cfg.seqLen],
    { device: "cpu", dtype: "i32" },
  );

  const logits = forward(api, model, inputTensor);
  // logits [B, T, V] → reshape to [B*T, V]
  const V = model.config.vocabSize;
  const flatLogits = api.reshape(logits, [cfg.batchSize * cfg.seqLen, V]);

  const loss = crossEntropy(api, flatLogits, targetTensor, { reduction: "mean" });
  const lossVal = await loss.item();

  await loss.backward();

  // SGD step.
  for (const p of params) {
    if (!p.grad) continue;
    api.noGrad(() => {
      const updated = api.sub(p, api.mul(p.grad!, cfg.lr));
      p.copy_(updated);
    });
    p.zeroGrad();
  }

  return lossVal;
}

/**
 * Run a training loop. Calls `onStep` after each step with loss + cumulative
 * stats. Yields control (via queueMicrotask/setTimeout(0)) between steps so
 * the UI can paint.
 */
export async function trainLoop(
  api: Torchlette,
  model: TransformerModel,
  ds: Dataset,
  cfg: TrainConfig,
  remote: RemoteEngine,
  onStep: (step: number, loss: number, elapsedMs: number) => void,
  shouldStop: () => boolean,
): Promise<void> {
  const params = parameters(model);
  let s = cfg.seed >>> 0 || 1;
  const rng = () => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return ((s >>> 0) / 0x100000000) * 2 - 1;
  };

  const t0 = performance.now();
  let step = 0;
  while (!shouldStop()) {
    const loss = await trainStep(api, model, ds, params, cfg, rng);
    const elapsed = performance.now() - t0;
    onStep(step, loss, elapsed);

    // Release old handles. Keep only handles bound to live params.
    await remote.markStep(params);

    step++;
    // Yield to UI
    await new Promise((r) => setTimeout(r, 0));
  }
}

export function modelConfigSmall(vocabSize: number): ModelConfig {
  return {
    vocabSize,
    blockSize: 16,
    embedDim: 32,
    numHeads: 4,
    numLayers: 2,
    mlpRatio: 2,
  };
}

