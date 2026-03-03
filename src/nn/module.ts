/**
 * Base Module class for neural network layers.
 * Similar to PyTorch's nn.Module.
 */

import type { Tensor, Torchlette } from "../frontend";

export abstract class Module {
  protected readonly api: Torchlette;
  private trainingMode = true;
  private _modules = new Map<string, Module>();

  constructor(api: Torchlette) {
    this.api = api;
  }

  /**
   * Check if module is in training mode.
   */
  get training(): boolean {
    return this.trainingMode;
  }

  /**
   * Register a buffer (non-parameter persistent tensor) on this module.
   * The tensor is set as a property on `this` for direct access.
   */
  registerBuffer(name: string, tensor: Tensor): void {
    Object.defineProperty(this, name, {
      value: tensor,
      writable: true,
      configurable: true,
    });
  }

  /**
   * Register a child module for recursive train()/eval() propagation.
   */
  registerModule(name: string, module: Module): void {
    this._modules.set(name, module);
  }

  /**
   * Set module to training mode.
   * In training mode, dropout is active, etc.
   * Recursively sets all registered child modules.
   */
  train(mode = true): this {
    this.trainingMode = mode;
    for (const child of this._modules.values()) {
      child.train(mode);
    }
    return this;
  }

  /**
   * Set module to evaluation mode.
   * Equivalent to train(false).
   */
  eval(): this {
    return this.train(false);
  }

  /**
   * Forward pass. Subclasses must implement this.
   */
  abstract forward(input: Tensor): Tensor;
}
