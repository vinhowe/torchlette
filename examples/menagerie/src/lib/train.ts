/**
 * Minimal in-browser full-model training loop for Menagerie. Plain AdamW +
 * gradient clipping (no AMP/GradScaler for the MVP). Follows the validated
 * step ceremony: beginStep → forwardWithLoss → backward → clip → step → zeroGrad
 * → endStep → markStep (see gpt2-lora-trainer/trainer.ts).
 */
import type { Torchlette } from "torchlette";
import { Adam } from "torchlette";
import { clipGradNorm_ } from "torchlette/nn";
import type { GPT2WithLoRA } from "gpt2-browser";

export interface TrainConfig {
  steps: number;
  batchSize: number;
  seqLength: number;
  lr: number;
  weightDecay: number;
}

export interface StepReport {
  step: number;
  loss: number;
  tokensSeen: number;
}

export interface TrainResult {
  steps: number;
  tokensSeen: number;
  finalLoss: number;
  wallclockMs: number;
}

/**
 * Build one (input, target) batch of contiguous windows from a flat token
 * stream, advancing `cursor` with wraparound. target is input shifted by 1.
 */
function makeBatch(
  tokens: number[],
  cursor: number,
  batchSize: number,
  seqLength: number,
): { inputData: number[]; targetData: number[]; nextCursor: number } {
  const inputData: number[] = [];
  const targetData: number[] = [];
  const span = seqLength + 1;
  const usable = Math.max(1, tokens.length - span);
  let c = cursor;
  for (let b = 0; b < batchSize; b++) {
    const start = c % usable;
    for (let t = 0; t < seqLength; t++) {
      inputData.push(tokens[start + t]);
      targetData.push(tokens[start + t + 1]);
    }
    c += seqLength;
  }
  return { inputData, targetData, nextCursor: c };
}

export async function trainModel(opts: {
  api: Torchlette;
  model: GPT2WithLoRA;
  tokens: number[];
  config: TrainConfig;
  onStep?: (r: StepReport) => void;
  shouldStop?: () => boolean;
}): Promise<TrainResult> {
  const { api, model, tokens, config, onStep, shouldStop } = opts;
  const { steps, batchSize, seqLength, lr, weightDecay } = config;

  if (tokens.length < seqLength + 2) {
    throw new Error(
      `Not enough tokens (${tokens.length}) for seqLength ${seqLength}. Use a bigger dataset slice.`,
    );
  }

  const params = model.getBaseParameters();
  const optimizer = new Adam(
    params,
    { lr, weightDecay, adamW: weightDecay > 0 },
    api,
  );

  const tokensPerStep = batchSize * seqLength;
  let cursor = 0;
  let finalLoss = NaN;
  const t0 = performance.now();

  for (let step = 0; step < steps; step++) {
    if (shouldStop?.()) break;

    const { inputData, targetData, nextCursor } = makeBatch(
      tokens,
      cursor,
      batchSize,
      seqLength,
    );
    cursor = nextCursor;

    await api.beginStep();

    const input = api.tensorFromArray(inputData, [batchSize, seqLength], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [batchSize, seqLength], { device: "webgpu" });

    const { loss } = model.forwardWithLoss(input, target);
    // Read the loss value BEFORE backward(): backward() clears the autograd
    // graph and disposes the loss tensor, so reading it afterward throws
    // "Tensor has been disposed". Materializing it here doesn't affect backward.
    const lossVal = (await loss.cpu())[0];
    finalLoss = lossVal;
    await loss.backward();

    await clipGradNorm_(api, params, 1.0);
    optimizer.step();
    optimizer.zeroGrad();

    input.dispose();
    target.dispose();

    api.endStep();
    await api.markStep();

    onStep?.({ step, loss: lossVal, tokensSeen: (step + 1) * tokensPerStep });
  }

  return {
    steps,
    tokensSeen: steps * tokensPerStep,
    finalLoss,
    wallclockMs: performance.now() - t0,
  };
}
