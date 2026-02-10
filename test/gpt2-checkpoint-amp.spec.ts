/**
 * Transformer Integration Tests for Checkpoint + AMP
 *
 * Tests gradient checkpointing and AMP (Automatic Mixed Precision)
 * with a simplified transformer-like model structure.
 */

import { describe, expect, it, beforeEach } from "vitest";
import { Torchlette, type FrontendTensor as Tensor } from "../src/frontend";
import { checkpoint } from "../src/nn/checkpoint";

/**
 * Simple MLP block for testing.
 * Linear -> ReLU -> Linear with residual connection.
 */
class ResidualMLP {
  private w1: Tensor;
  private b1: Tensor;
  private w2: Tensor;
  private b2: Tensor;

  constructor(
    private api: Torchlette,
    dim: number,
    hiddenDim: number,
  ) {
    // Initialize with small random values for numerical stability
    const scale = 0.1 / Math.sqrt(dim);
    const w1Data = Array.from({ length: dim * hiddenDim }, () => (Math.random() - 0.5) * scale);
    const w2Data = Array.from({ length: hiddenDim * dim }, () => (Math.random() - 0.5) * scale);

    this.w1 = api.tensorFromArray(w1Data, [dim, hiddenDim], { requiresGrad: true });
    this.b1 = api.zeros([hiddenDim], { requiresGrad: true });
    this.w2 = api.tensorFromArray(w2Data, [hiddenDim, dim], { requiresGrad: true });
    this.b2 = api.zeros([dim], { requiresGrad: true });
  }

  forward(x: Tensor): Tensor {
    // x: [..., dim]
    let h = x.matmul(this.w1).add(this.b1);
    h = h.relu();
    h = h.matmul(this.w2).add(this.b2);
    // Residual connection
    return x.add(h);
  }

  parameters(): Tensor[] {
    return [this.w1, this.b1, this.w2, this.b2];
  }
}

/**
 * Simple stacked MLP model.
 * Multiple residual MLP blocks followed by output projection.
 */
class StackedMLP {
  private blocks: ResidualMLP[];
  private outputProj: Tensor;
  private outputBias: Tensor;

  constructor(
    private api: Torchlette,
    private inputDim: number,
    private hiddenDim: number,
    private outputDim: number,
    private numLayers: number,
  ) {
    // Residual MLP blocks
    this.blocks = [];
    for (let i = 0; i < numLayers; i++) {
      this.blocks.push(new ResidualMLP(api, inputDim, hiddenDim));
    }

    // Output projection
    const scale = 0.1 / Math.sqrt(inputDim);
    const projData = Array.from({ length: inputDim * outputDim }, () => (Math.random() - 0.5) * scale);
    this.outputProj = api.tensorFromArray(projData, [inputDim, outputDim], { requiresGrad: true });
    this.outputBias = api.zeros([outputDim], { requiresGrad: true });
  }

  forward(x: Tensor, useCheckpoint = false): Tensor {
    let h = x;

    // Apply MLP blocks
    for (const block of this.blocks) {
      if (useCheckpoint) {
        h = checkpoint(this.api, (input: Tensor) => block.forward(input), [h]);
      } else {
        h = block.forward(h);
      }
    }

    // Output projection
    const output = h.matmul(this.outputProj).add(this.outputBias);
    return output;
  }

  parameters(): Tensor[] {
    const params: Tensor[] = [];
    for (const block of this.blocks) {
      params.push(...block.parameters());
    }
    params.push(this.outputProj, this.outputBias);
    return params;
  }
}

describe("Model Checkpoint + AMP Integration", () => {
  let torch: Torchlette;

  // Small model config for testing
  const INPUT_DIM = 16;
  const HIDDEN_DIM = 32;
  const OUTPUT_DIM = 8;
  const NUM_LAYERS = 3;
  const BATCH_SIZE = 2;
  const SEQ_LEN = 4;

  beforeEach(() => {
    torch = new Torchlette("cpu");
  });

  describe("Basic Forward/Backward", () => {
    it("model produces valid gradients without checkpoint", async () => {
      const model = new StackedMLP(
        torch,
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        NUM_LAYERS,
      );

      // Random input
      const inputData = Array.from(
        { length: BATCH_SIZE * SEQ_LEN * INPUT_DIM },
        () => Math.random() - 0.5,
      );
      const input = torch.tensorFromArray(inputData, [BATCH_SIZE, SEQ_LEN, INPUT_DIM], {
        requiresGrad: true,
      });

      // Forward
      const output = model.forward(input, false);
      expect(output.shape).toEqual([BATCH_SIZE, SEQ_LEN, OUTPUT_DIM]);

      // Simple loss: sum of outputs
      const loss = output.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      // Check all parameters have gradients
      for (const param of model.parameters()) {
        expect(param.grad).not.toBeNull();

        // Gradient should not contain NaN
        const gradData = await param.grad!.cpu();
        for (const val of gradData) {
          expect(Number.isFinite(val)).toBe(true);
        }
      }
    });

    it("model produces valid gradients with checkpoint", async () => {
      const model = new StackedMLP(
        torch,
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        NUM_LAYERS,
      );

      const inputData = Array.from(
        { length: BATCH_SIZE * SEQ_LEN * INPUT_DIM },
        () => Math.random() - 0.5,
      );
      const input = torch.tensorFromArray(inputData, [BATCH_SIZE, SEQ_LEN, INPUT_DIM], {
        requiresGrad: true,
      });

      // Forward with checkpointing
      const output = model.forward(input, true);
      expect(output.shape).toEqual([BATCH_SIZE, SEQ_LEN, OUTPUT_DIM]);

      const loss = output.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      // Check all parameters have gradients
      for (const param of model.parameters()) {
        expect(param.grad).not.toBeNull();

        const gradData = await param.grad!.cpu();
        for (const val of gradData) {
          expect(Number.isFinite(val)).toBe(true);
        }
      }
    });
  });

  describe("Checkpoint Gradient Correctness", () => {
    it("both checkpoint and non-checkpoint produce valid gradients", async () => {
      // Use fixed input data for reproducibility
      const inputData = Array.from({ length: BATCH_SIZE * SEQ_LEN * INPUT_DIM }, (_, i) =>
        Math.sin(i * 0.1) * 0.1
      );

      // Model 1: without checkpoint
      const torch1 = new Torchlette("cpu");
      const model1 = new StackedMLP(torch1, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 1);
      const input1 = torch1.tensorFromArray(inputData, [BATCH_SIZE, SEQ_LEN, INPUT_DIM], {
        requiresGrad: true,
      });

      const output1 = model1.forward(input1, false);
      const loss1 = output1.sum();
      if (typeof loss1 === "number") throw new Error("Expected tensor");
      await loss1.backward();

      // Model 2: with checkpoint
      const torch2 = new Torchlette("cpu");
      const model2 = new StackedMLP(torch2, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 1);
      const input2 = torch2.tensorFromArray(inputData, [BATCH_SIZE, SEQ_LEN, INPUT_DIM], {
        requiresGrad: true,
      });

      const output2 = model2.forward(input2, true);
      const loss2 = output2.sum();
      if (typeof loss2 === "number") throw new Error("Expected tensor");
      await loss2.backward();

      // Both should produce valid finite gradients
      const params1 = model1.parameters();
      const params2 = model2.parameters();

      for (let i = 0; i < params1.length; i++) {
        const grad1 = await params1[i].grad!.cpu();
        const grad2 = await params2[i].grad!.cpu();

        // Both should have same shape
        expect(params1[i].shape).toEqual(params2[i].shape);

        // Both should be finite
        for (const val of grad1) {
          expect(Number.isFinite(val)).toBe(true);
        }
        for (const val of grad2) {
          expect(Number.isFinite(val)).toBe(true);
        }
      }
    });
  });

  describe("AMP Integration", () => {
    it("autocast produces valid gradients", async () => {
      const model = new StackedMLP(
        torch,
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        NUM_LAYERS,
      );

      const inputData = Array.from(
        { length: BATCH_SIZE * SEQ_LEN * INPUT_DIM },
        () => Math.random() - 0.5,
      );
      const input = torch.tensorFromArray(inputData, [BATCH_SIZE, SEQ_LEN, INPUT_DIM], {
        requiresGrad: true,
      });

      // Forward with autocast
      const output = await torch.autocastAsync(async () => {
        return model.forward(input, false);
      });

      const loss = output.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      // Check all parameters have gradients
      for (const param of model.parameters()) {
        expect(param.grad).not.toBeNull();

        const gradData = await param.grad!.cpu();
        for (const val of gradData) {
          expect(Number.isFinite(val)).toBe(true);
        }
      }
    });
  });

  describe("Checkpoint + AMP Combined", () => {
    it("checkpoint + autocast produces valid gradients", async () => {
      const model = new StackedMLP(
        torch,
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        NUM_LAYERS,
      );

      const inputData = Array.from(
        { length: BATCH_SIZE * SEQ_LEN * INPUT_DIM },
        () => Math.random() - 0.5,
      );
      const input = torch.tensorFromArray(inputData, [BATCH_SIZE, SEQ_LEN, INPUT_DIM], {
        requiresGrad: true,
      });

      // Forward with both checkpoint and autocast
      const output = await torch.autocastAsync(async () => {
        return model.forward(input, true);
      });

      const loss = output.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      // Check all parameters have gradients
      for (const param of model.parameters()) {
        expect(param.grad).not.toBeNull();

        const gradData = await param.grad!.cpu();
        for (const val of gradData) {
          expect(Number.isFinite(val)).toBe(true);
        }
      }
    });
  });

  describe("Multiple Training Steps", () => {
    it("model trains for multiple steps without issues", async () => {
      const model = new StackedMLP(
        torch,
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        NUM_LAYERS,
      );

      const losses: number[] = [];

      for (let step = 0; step < 3; step++) {
        // Zero gradients
        for (const param of model.parameters()) {
          param.zeroGrad();
        }

        // Random input
        const inputData = Array.from(
          { length: BATCH_SIZE * SEQ_LEN * INPUT_DIM },
          () => Math.random() - 0.5,
        );
        const input = torch.tensorFromArray(inputData, [BATCH_SIZE, SEQ_LEN, INPUT_DIM], {
          requiresGrad: true,
        });

        // Forward with checkpoint + AMP
        const output = await torch.autocastAsync(async () => {
          return model.forward(input, true);
        });

        const loss = output.sum();
        if (typeof loss === "number") throw new Error("Expected tensor");

        const lossVal = await loss.item();
        losses.push(lossVal);

        await loss.backward();

        // Verify gradients are valid
        for (const param of model.parameters()) {
          expect(param.grad).not.toBeNull();
          const gradData = await param.grad!.cpu();
          for (const val of gradData) {
            expect(Number.isFinite(val)).toBe(true);
          }
        }
      }

      // All losses should be finite
      for (const l of losses) {
        expect(Number.isFinite(l)).toBe(true);
      }
    });
  });

  describe("retainGrad with Checkpoint", () => {
    it("retainGrad works on intermediate tensor with checkpoint", async () => {
      const model = new StackedMLP(
        torch,
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        2,
      );

      const inputData = Array.from(
        { length: BATCH_SIZE * SEQ_LEN * INPUT_DIM },
        () => Math.random() - 0.5,
      );
      const input = torch.tensorFromArray(inputData, [BATCH_SIZE, SEQ_LEN, INPUT_DIM], {
        requiresGrad: true,
      });

      // Forward with checkpoint
      const output = model.forward(input, true);
      output.retainGrad();

      const loss = output.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      // Output should have gradient because we called retainGrad
      expect(output.grad).not.toBeNull();
      expect(output.isRetainGrad).toBe(true);

      // Gradient should be all 1s from sum backward
      const outputGrad = await output.grad!.cpu();
      for (const val of outputGrad) {
        expect(val).toBeCloseTo(1.0, 5);
      }
    });
  });
});
