/**
 * Base Module class for neural network layers.
 * Similar to PyTorch's nn.Module.
 */

import type { Tensor, Torchlette } from "../frontend";

export abstract class Module {
  protected readonly api: Torchlette;
  private trainingMode = true;
  private _buffers = new Map<string, Tensor>();
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
   * The tensor is stored in _buffers and also set as a property on `this`.
   */
  registerBuffer(name: string, tensor: Tensor): void {
    this._buffers.set(name, tensor);
    Object.defineProperty(this, name, { value: tensor, writable: true, configurable: true });
  }

  /**
   * Register a child module for recursive train()/eval() propagation.
   */
  registerModule(name: string, module: Module): void {
    this._modules.set(name, module);
  }

  /**
   * Return all registered buffers, optionally recursing into child modules.
   */
  buffers(recurse = true): Tensor[] {
    const result = [...this._buffers.values()];
    if (recurse) {
      for (const child of this._modules.values()) {
        result.push(...child.buffers(true));
      }
    }
    return result;
  }

  /**
   * Return all registered child modules.
   */
  modules(): Module[] {
    return [...this._modules.values()];
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

  /**
   * Callable interface - calls forward().
   */
  call(input: Tensor): Tensor {
    return this.forward(input);
  }
}
