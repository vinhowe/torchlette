/**
 * Embedding layer for token lookups.
 * Similar to PyTorch's nn.Embedding.
 */

import type { Tensor, Torchlette, DeviceKind } from "../frontend";
import { Module } from "./module";
import { sizeOf } from "../core/shape";

export type EmbeddingOptions = {
  /** Device to create parameters on */
  device?: DeviceKind;
};

/**
 * A lookup table that stores embeddings of a fixed dictionary and size.
 *
 * This module is often used to store word embeddings and retrieve them using indices.
 * The input to the module is a list of indices, and the output is the corresponding
 * word embeddings.
 *
 * @example
 * ```ts
 * const embedding = new Embedding(api, 50257, 768);  // vocab size 50257, embed dim 768
 * const tokens = api.tensorFromArray([0, 1, 2, 3], [1, 4]);  // batch=1, seqLen=4
 * const embedded = embedding.forward(tokens);  // [1, 4, 768]
 * ```
 */
export class Embedding extends Module {
  readonly numEmbeddings: number;
  readonly embeddingDim: number;
  readonly weight: Tensor;

  constructor(
    api: Torchlette,
    numEmbeddings: number,
    embeddingDim: number,
    options?: EmbeddingOptions,
  ) {
    super(api);
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;

    const device = options?.device;

    // Initialize with standard normal distribution
    this.weight = api.randn([numEmbeddings, embeddingDim], {
      requiresGrad: true,
      device,
    });
  }

  /**
   * Forward pass: lookup embeddings for input indices.
   *
   * @param input - Tensor of token indices, shape [...] (any shape)
   * @returns Tensor of embeddings, shape [..., embeddingDim]
   */
  forward(input: Tensor): Tensor {
    // input: [...] containing token indices
    // weight: [numEmbeddings, embeddingDim]
    // output: [..., embeddingDim]

    const inputShape = input.shape;
    const numElements = sizeOf(inputShape);

    // Flatten input to [numElements]
    const flatInput = input.reshape([numElements]);

    // Expand indices to [numElements, embeddingDim] for gather
    // Each index is repeated embeddingDim times across dim 1
    const expandedInput = flatInput
      .reshape([numElements, 1])
      .expand([numElements, this.embeddingDim])
      .contiguous(); // Required: gather doesn't handle strided tensors correctly

    // Gather from weight: output[i][j] = weight[expandedInput[i][j]][j]
    // Since expandedInput[i][j] = flatInput[i] for all j,
    // this gives us weight[flatInput[i]][j] for each position
    const gathered = this.weight.gather(expandedInput, { dim: 0 });

    // Reshape to [..., embeddingDim]
    const outputShape = [...inputShape, this.embeddingDim];
    return gathered.reshape(outputShape);
  }

  /**
   * Get all learnable parameters.
   */
  parameters(): Tensor[] {
    return [this.weight];
  }
}
