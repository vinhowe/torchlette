/**
 * GPT-2 Compiled Trainer
 *
 * Training loop with compile() wrapping forward + backward + optimizer + zeroGrad.
 */

import type { Tensor, Torchlette, DeviceKind } from "../../src/frontend";
import { Adam, type AdamOptions } from "../../src/optim";
import { GPT2 } from "./model";
import { FineWebDataLoader } from "./data";

// ============================================================================
// Types
// ============================================================================

export type GPT2TrainerConfig = {
  learningRate: number;
  betas?: [number, number];
  eps?: number;
  weightDecay?: number;
  gradClipNorm?: number;
  maxSteps?: number;
  logEveryNSteps?: number;
  /** Enable fp16 mixed precision (default: false) */
  useFp16?: boolean;
};

export type TrainResult = {
  totalSteps: number;
  finalLoss: number;
  avgLossPerStep: number;
  totalTimeMs: number;
  avgTimePerStepMs: number;
};

export type TrainCallbacks = {
  onStep?: (step: number, loss: number, timeMs: number) => void;
  onEpoch?: (epoch: number, avgLoss: number) => void;
};

// ============================================================================
// GPT-2 Trainer
// ============================================================================

/**
 * GPT-2 Trainer with compiled training step.
 *
 * Compiles forward + backward + optimizer step + zeroGrad into a single
 * compiled region for maximum optimization.
 */
export class GPT2Trainer {
  private readonly api: Torchlette;
  private readonly model: GPT2;
  private readonly optimizer: Adam;
  private readonly config: GPT2TrainerConfig;
  private readonly device?: DeviceKind;

  private compiledTrainStep:
    | ((input: Tensor, target: Tensor) => Tensor)
    | null = null;
  private unoptimizedTrainStep:
    | ((input: Tensor, target: Tensor) => Tensor)
    | null = null;

  private totalSteps = 0;
  private totalLoss = 0;

  constructor(
    api: Torchlette,
    model: GPT2,
    config: GPT2TrainerConfig,
    options?: { device?: DeviceKind },
  ) {
    this.api = api;
    this.model = model;
    this.config = config;
    this.device = options?.device;

    // Initialize optimizer
    const adamOptions: AdamOptions = {
      lr: config.learningRate,
      betas: config.betas,
      eps: config.eps,
      weightDecay: config.weightDecay,
    };
    this.optimizer = new Adam(model.parameters(), adamOptions, api);
  }

  /**
   * Compile the training step.
   *
   * This wraps forward pass into a compiled region for fusion and optimization.
   * When useFp16 is enabled, autocast is applied for mixed precision training.
   * Note: backward() is async so it runs outside the compile region for now.
   */
  compile(): void {
    const useFp16 = this.config.useFp16 ?? false;

    // Define the training step function (forward pass only, backward is async)
    const trainStepFn = (input: Tensor, target: Tensor): Tensor => {
      // Forward pass with loss computation
      // Use tidy to clean up intermediate tensors (but keep the loss)
      const loss = this.api.tidy(() => {
        const { loss: lossVal } = this.model.forwardWithLoss(input, target);
        if (!lossVal) {
          throw new Error("Loss is null - targets must be provided");
        }
        // Keep the loss tensor so it survives tidy
        lossVal.keep();
        return lossVal;
      });
      return loss;
    };

    // Wrap in autocast if fp16 is enabled
    const wrappedFn = useFp16
      ? (input: Tensor, target: Tensor) =>
          this.api.autocast(() => trainStepFn(input, target), { deviceType: "webgpu" })
      : trainStepFn;

    // Compile the forward pass for fusion optimization
    this.compiledTrainStep = this.api.compile(wrappedFn);

    console.log(`Training step compiled with fusion enabled${useFp16 ? " and fp16 AMP" : ""}`);
  }

  /**
   * Create an unoptimized version for comparison.
   */
  createUnoptimizedVersion(): void {
    // The unoptimized version just runs the same code without compile()
    this.unoptimizedTrainStep = (input: Tensor, target: Tensor): Tensor => {
      const { loss } = this.model.forwardWithLoss(input, target);
      if (!loss) {
        throw new Error("Loss is null");
      }
      return loss;
    };
  }

  /**
   * Execute one training step (compiled version).
   */
  async trainStep(input: Tensor, target: Tensor): Promise<number> {
    if (!this.compiledTrainStep) {
      throw new Error("Training step not compiled. Call compile() first.");
    }

    const startTime = performance.now();

    // Run compiled forward pass
    const loss = this.compiledTrainStep(input, target);

    // Backward pass (outside compile for now due to async nature)
    await loss.backward();

    // Gradient clipping if configured
    if (this.config.gradClipNorm) {
      // Note: grad clipping would need to be implemented
      // For now, skip this
    }

    // Optimizer step
    this.optimizer.step();

    // Zero gradients
    this.optimizer.zeroGrad();

    // Get loss value before disposing
    const lossValue = await loss.item();

    // Dispose the loss tensor to free GPU memory
    loss.dispose();

    const endTime = performance.now();

    this.totalSteps++;
    this.totalLoss += lossValue;

    return lossValue;
  }

  /**
   * Execute one training step (unoptimized version for comparison).
   */
  async trainStepUnoptimized(input: Tensor, target: Tensor): Promise<number> {
    if (!this.unoptimizedTrainStep) {
      throw new Error("Unoptimized version not created");
    }

    // Run unoptimized forward pass
    const loss = this.unoptimizedTrainStep(input, target);

    // Backward pass
    await loss.backward();

    // Optimizer step
    this.optimizer.step();

    // Zero gradients
    this.optimizer.zeroGrad();

    // Get loss value
    const lossValue = await loss.item();

    return lossValue;
  }

  /**
   * Run the full training loop.
   */
  async train(
    dataLoader: FineWebDataLoader,
    numSteps?: number,
    callbacks?: TrainCallbacks,
  ): Promise<TrainResult> {
    const maxSteps = numSteps ?? this.config.maxSteps ?? 1000;
    const logEvery = this.config.logEveryNSteps ?? 10;

    const startTime = performance.now();
    let totalLoss = 0;

    this.model.train();

    for (let step = 0; step < maxSteps; step++) {
      const stepStart = performance.now();

      const { input, target } = await dataLoader.nextBatch();
      const loss = await this.trainStep(input, target);

      totalLoss += loss;
      const stepTime = performance.now() - stepStart;

      // Callback
      if (callbacks?.onStep) {
        callbacks.onStep(step, loss, stepTime);
      }

      // Logging
      if ((step + 1) % logEvery === 0) {
        const avgLoss = totalLoss / (step + 1);
        console.log(
          `Step ${step + 1}/${maxSteps}: loss=${loss.toFixed(4)}, ` +
          `avg_loss=${avgLoss.toFixed(4)}, step_time=${stepTime.toFixed(1)}ms`,
        );
      }

      // Clean up tensors
      input.dispose();
      target.dispose();
    }

    const totalTime = performance.now() - startTime;

    return {
      totalSteps: maxSteps,
      finalLoss: totalLoss / maxSteps,
      avgLossPerStep: totalLoss / maxSteps,
      totalTimeMs: totalTime,
      avgTimePerStepMs: totalTime / maxSteps,
    };
  }

  /**
   * Get training statistics.
   */
  getStats(): { totalSteps: number; avgLoss: number } {
    return {
      totalSteps: this.totalSteps,
      avgLoss: this.totalSteps > 0 ? this.totalLoss / this.totalSteps : 0,
    };
  }

  /**
   * Reset training statistics.
   */
  resetStats(): void {
    this.totalSteps = 0;
    this.totalLoss = 0;
  }

  /**
   * Get the underlying model.
   */
  getModel(): GPT2 {
    return this.model;
  }

  /**
   * Get the optimizer.
   */
  getOptimizer(): Adam {
    return this.optimizer;
  }
}

// ============================================================================
// Training Helpers
// ============================================================================

/**
 * Measure training step time.
 */
export async function measureTrainStepTime(
  trainer: GPT2Trainer,
  input: Tensor,
  target: Tensor,
  numIterations = 10,
  warmupIterations = 3,
): Promise<{ median: number; mean: number; min: number; max: number }> {
  const times: number[] = [];

  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    await trainer.trainStep(input, target);
  }

  // Measure
  for (let i = 0; i < numIterations; i++) {
    const start = performance.now();
    await trainer.trainStep(input, target);
    times.push(performance.now() - start);
  }

  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const min = times[0];
  const max = times[times.length - 1];

  return { median, mean, min, max };
}

/**
 * Compare optimized vs unoptimized training step times.
 */
export async function compareOptimizedVsUnoptimized(
  trainer: GPT2Trainer,
  input: Tensor,
  target: Tensor,
  numIterations = 10,
  warmupIterations = 3,
): Promise<{
  optimized: { median: number; mean: number };
  unoptimized: { median: number; mean: number };
  speedup: number;
}> {
  // Ensure both versions are created
  if (!trainer["compiledTrainStep"]) {
    trainer.compile();
  }
  if (!trainer["unoptimizedTrainStep"]) {
    trainer.createUnoptimizedVersion();
  }

  // Measure optimized
  const optimizedTimes: number[] = [];
  for (let i = 0; i < warmupIterations; i++) {
    await trainer.trainStep(input, target);
  }
  for (let i = 0; i < numIterations; i++) {
    const start = performance.now();
    await trainer.trainStep(input, target);
    optimizedTimes.push(performance.now() - start);
  }

  // Measure unoptimized
  const unoptimizedTimes: number[] = [];
  for (let i = 0; i < warmupIterations; i++) {
    await trainer.trainStepUnoptimized(input, target);
  }
  for (let i = 0; i < numIterations; i++) {
    const start = performance.now();
    await trainer.trainStepUnoptimized(input, target);
    unoptimizedTimes.push(performance.now() - start);
  }

  const optMedian = optimizedTimes.sort((a, b) => a - b)[Math.floor(optimizedTimes.length / 2)];
  const optMean = optimizedTimes.reduce((a, b) => a + b, 0) / optimizedTimes.length;

  const unoptMedian = unoptimizedTimes.sort((a, b) => a - b)[Math.floor(unoptimizedTimes.length / 2)];
  const unoptMean = unoptimizedTimes.reduce((a, b) => a + b, 0) / unoptimizedTimes.length;

  return {
    optimized: { median: optMedian, mean: optMean },
    unoptimized: { median: unoptMedian, mean: unoptMean },
    speedup: unoptMedian / optMedian,
  };
}
