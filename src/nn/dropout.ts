/**
 * Dropout module.
 */

import type { Tensor, Torchlette } from "../frontend";
import { dropout as dropoutFn } from "./functional";
import { Module } from "./module";

export type DropoutOptions = {
  /** Probability of an element to be zeroed (default: 0.5) */
  p?: number;
};

/**
 * Dropout layer.
 *
 * During training, randomly zeroes elements with probability p using a Bernoulli
 * distribution, and scales the remaining elements by 1/(1-p) to maintain expected value.
 *
 * During evaluation, returns the input unchanged.
 *
 * @example
 * ```ts
 * const dropout = new Dropout(api, { p: 0.5 });
 * dropout.train();  // Enable dropout
 * const output = dropout.forward(input);
 *
 * dropout.eval();   // Disable dropout
 * const evalOutput = dropout.forward(input);  // Returns input unchanged
 * ```
 */
export class Dropout extends Module {
  private readonly p: number;

  constructor(api: Torchlette, options?: DropoutOptions) {
    super(api);
    this.p = options?.p ?? 0.5;
    if (this.p < 0 || this.p > 1) {
      throw new Error(`Dropout probability must be between 0 and 1, got ${this.p}`);
    }
  }

  forward(input: Tensor): Tensor {
    return dropoutFn(this.api, input, this.p, this.training);
  }
}
