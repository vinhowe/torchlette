/**
 * LoRA Trainer for GPT-2.
 *
 * Handles the training loop with progress callbacks for UI integration.
 * Supports:
 * - AMP (Automatic Mixed Precision) for reduced memory and faster training
 * - Gradient checkpointing to trade compute for memory
 */

import type { FrontendTensor as Tensor, Torchlette } from 'torchlette';
import { Adam, GradScaler } from 'torchlette';
import type { GPT2WithLoRA } from './gpt2-lora';
import type { GPT2Tokenizer } from './tokenizer';

export type TrainingConfig = {
  maxSteps: number;
  batchSize: number;
  seqLength: number;
  learningRate: number;
  /** Enable Automatic Mixed Precision (f16 compute, f32 accumulate) */
  useAMP?: boolean;
  /** Enable gradient checkpointing (trade compute for memory) */
  useCheckpointing?: boolean;
};

export type TrainingCallbacks = {
  onStepStart?: (step: number) => void;
  onStepEnd?: (step: number, loss: number, timeMs: number) => void;
  shouldStop?: () => boolean;
};

export type TrainingResult = {
  finalLoss: number;
  totalSteps: number;
  totalTimeMs: number;
};

/**
 * Simple data loader for training text.
 */
class SimpleDataLoader {
  private tokens: Uint32Array;
  private currentIdx = 0;

  constructor(
    private api: Torchlette,
    tokens: number[],
    private batchSize: number,
    private seqLength: number
  ) {
    this.tokens = new Uint32Array(tokens);
  }

  hasMore(): boolean {
    return this.tokens.length >= this.batchSize * (this.seqLength + 1);
  }

  reset(): void {
    this.currentIdx = 0;
  }

  nextBatch(): { input: Tensor; target: Tensor } {
    const inputData: number[] = [];
    const targetData: number[] = [];

    for (let b = 0; b < this.batchSize; b++) {
      // Wrap around if we've exhausted the data
      if (this.currentIdx + this.seqLength + 1 > this.tokens.length) {
        this.currentIdx = Math.floor(Math.random() * (this.tokens.length - this.seqLength - 1));
      }

      for (let i = 0; i < this.seqLength; i++) {
        inputData.push(this.tokens[this.currentIdx + i]);
        targetData.push(this.tokens[this.currentIdx + i + 1]);
      }

      // Move to next random position for variety
      this.currentIdx = Math.floor(Math.random() * (this.tokens.length - this.seqLength - 1));
    }

    return {
      input: this.api.tensorFromArray(inputData, [this.batchSize, this.seqLength], {
        device: 'webgpu',
      }),
      target: this.api.tensorFromArray(targetData, [this.batchSize, this.seqLength], {
        device: 'webgpu',
      }),
    };
  }
}

/**
 * LoRA Trainer class.
 */
export class LoRATrainer {
  private model: GPT2WithLoRA;
  private tokenizer: GPT2Tokenizer;
  private api: Torchlette;
  private optimizer: Adam | null = null;
  private gradScaler: GradScaler | null = null;

  constructor(
    api: Torchlette,
    model: GPT2WithLoRA,
    tokenizer: GPT2Tokenizer
  ) {
    this.api = api;
    this.model = model;
    this.tokenizer = tokenizer;
  }

  /**
   * Train the LoRA adapter on the given text.
   */
  async train(
    trainingText: string,
    config: TrainingConfig,
    callbacks: TrainingCallbacks = {}
  ): Promise<TrainingResult> {
    const useAMP = config.useAMP ?? false;
    const useCheckpointing = config.useCheckpointing ?? false;

    // Tokenize training text
    const tokens = this.tokenizer.encode(trainingText);

    if (tokens.length < config.batchSize * (config.seqLength + 1)) {
      throw new Error(
        `Training text too short. Need at least ${config.batchSize * (config.seqLength + 1)} tokens, got ${tokens.length}`
      );
    }

    // Create data loader
    const dataLoader = new SimpleDataLoader(
      this.api,
      tokens,
      config.batchSize,
      config.seqLength
    );

    // Create optimizer for LoRA parameters only
    const loraParams = this.model.getLoRAParameters();
    this.optimizer = new Adam(loraParams, { lr: config.learningRate }, this.api);

    // Create gradient scaler for AMP
    if (useAMP) {
      this.gradScaler = new GradScaler(this.api, {
        initScale: 65536.0,
        growthFactor: 2.0,
        backoffFactor: 0.5,
        growthInterval: 2000,
      });
    }

    // Enable checkpointing on model if requested
    if (useCheckpointing) {
      this.model.enableCheckpointing(true);
    }

    // Set model to training mode
    this.model.train(true);

    let totalLoss = 0;
    let lastLoss = 0;
    const startTime = performance.now();

    // Create compiled forward function for AMP, fusion, and memory planning
    // The compile() region enables:
    // - AMP transforms (f16 compute for matmuls)
    // - Kernel fusion (fuse elementwise ops)
    // - Memory planning (buffer reuse)
    const model = this.model;
    const api = this.api;

    // Compiled forward pass - returns the loss tensor directly
    const compiledForwardWithLoss = useAMP
      ? api.compile((batchInput: Tensor, batchTarget: Tensor) => {
          // autocast inside compile enables f16 for matmuls
          return api.autocast(() => {
            const { loss } = model.forwardWithLoss(batchInput, batchTarget);
            return loss;
          }, { deviceType: 'webgpu' });
        })
      : null;

    for (let step = 0; step < config.maxSteps; step++) {
      // Check for early stop
      if (callbacks.shouldStop?.()) {
        break;
      }

      callbacks.onStepStart?.(step);
      const stepStart = performance.now();

      // Get batch
      const { input, target } = dataLoader.nextBatch();

      // Forward pass with loss - use compiled region for AMP
      let loss: Tensor;
      if (useAMP && compiledForwardWithLoss) {
        // Run forward pass in compiled region with AMP + fusion + memory planning
        loss = compiledForwardWithLoss(input, target);

        // Scale loss for gradient scaling
        loss = this.gradScaler!.scale(loss);
      } else {
        const result = this.model.forwardWithLoss(input, target);
        loss = result.loss;
      }

      // Backward pass
      await loss.backward();

      // Optimizer step - with gradient unscaling for AMP
      if (useAMP && this.gradScaler) {
        await this.gradScaler.unscale_(this.optimizer!);
        const stepped = await this.gradScaler.step(this.optimizer!);
        this.gradScaler.update();

        if (!stepped) {
          // Skip this step due to NaN/Inf gradients
          console.warn(`Step ${step}: Skipped due to NaN/Inf gradients, scale=${this.gradScaler.getScale()}`);
        }
      } else {
        this.optimizer!.step();
      }
      this.optimizer!.zeroGrad();

      // Get loss value (unscale if AMP was used)
      let lossValue = await loss.item();
      if (useAMP && this.gradScaler) {
        lossValue = lossValue / this.gradScaler.getScale();
      }
      totalLoss += lossValue;
      lastLoss = lossValue;

      // Explicitly dispose batch tensors to free memory
      input.dispose();
      target.dispose();

      const stepTime = performance.now() - stepStart;

      callbacks.onStepEnd?.(step, lossValue, stepTime);

      // Memory cleanup - finalize the step and free intermediate tensors
      await this.api.markStep();
    }

    // Set model to eval mode and disable checkpointing
    this.model.train(false);
    if (useCheckpointing) {
      this.model.enableCheckpointing(false);
    }

    const totalTime = performance.now() - startTime;

    return {
      finalLoss: lastLoss,
      totalSteps: config.maxSteps,
      totalTimeMs: totalTime,
    };
  }

  /**
   * Get the trained LoRA parameters for export.
   */
  getLoRAWeights(): Map<string, { data: Float32Array; shape: number[] }> {
    const weights = new Map<string, { data: Float32Array; shape: number[] }>();
    const loraParams = this.model.getLoRAParameters();

    // LoRA params are ordered: [loraA_0, loraB_0, loraA_1, loraB_1, ...]
    const numLayers = loraParams.length / 2;

    for (let i = 0; i < numLayers; i++) {
      const loraA = loraParams[i * 2];
      const loraB = loraParams[i * 2 + 1];

      weights.set(`h.${i}.attn.c_attn.lora_A`, {
        data: new Float32Array([]), // Will be filled by async read
        shape: loraA.shape,
      });
      weights.set(`h.${i}.attn.c_attn.lora_B`, {
        data: new Float32Array([]),
        shape: loraB.shape,
      });
    }

    return weights;
  }

  /**
   * Export LoRA weights asynchronously.
   */
  async exportLoRAWeights(): Promise<Map<string, { data: Float32Array; shape: number[] }>> {
    const weights = new Map<string, { data: Float32Array; shape: number[] }>();
    const loraParams = this.model.getLoRAParameters();

    const numLayers = loraParams.length / 2;

    for (let i = 0; i < numLayers; i++) {
      const loraA = loraParams[i * 2];
      const loraB = loraParams[i * 2 + 1];

      const loraAData = await loraA.cpu();
      const loraBData = await loraB.cpu();

      weights.set(`h.${i}.attn.c_attn.lora_A`, {
        data: new Float32Array(loraAData),
        shape: loraA.shape,
      });
      weights.set(`h.${i}.attn.c_attn.lora_B`, {
        data: new Float32Array(loraBData),
        shape: loraB.shape,
      });
    }

    return weights;
  }
}
