/**
 * Base Module class for neural network layers.
 * Similar to PyTorch's nn.Module.
 */

import type { DeviceKind, Tensor, Torchlette } from "../frontend/torchlette";

export abstract class Module {
  protected readonly api: Torchlette;
  private trainingMode = true;
  private _modules = new Map<string, Module>();
  private _params = new Map<string, Tensor>();
  private _buffers = new Map<string, Tensor>();

  constructor(api: Torchlette) {
    this.api = api;
    // Return a Proxy that auto-registers Module-typed property assignments.
    // registerParameter() uses Object.defineProperty which does NOT trigger
    // the Proxy set trap, so parameters are unaffected.
    return new Proxy(this, {
      set(target, prop, value, receiver) {
        if (value instanceof Module && typeof prop === "string") {
          target._modules.set(prop, value);
        }
        return Reflect.set(target, prop, value, receiver);
      },
    });
  }

  /**
   * Check if module is in training mode.
   */
  get training(): boolean {
    return this.trainingMode;
  }

  /**
   * Register a learnable parameter on this module.
   * The tensor is stored in the params map AND set as a property on `this`.
   */
  registerParameter(name: string, tensor: Tensor | null): void {
    if (tensor !== null) {
      this._params.set(name, tensor);
      // DECLARE persistent state (task #70 D3): a module param IS long-lived
      // registered state. Registering it here — the single enumeration seam every
      // param passes through — makes it gen-independent persistent (REG), strictly
      // more robust than relying on the transient step snapshot alone (a param
      // cannot fall out of persistence because a concurrent test perturbed the
      // boundary gen). No duplicate walk: this rides the existing enumeration.
      this.api.registerState(tensor);
    }
    Object.defineProperty(this, name, {
      value: tensor,
      writable: true,
      configurable: true,
    });
  }

  /**
   * Register a buffer (non-parameter persistent tensor) on this module.
   * The tensor is set as a property on `this` for direct access.
   */
  registerBuffer(name: string, tensor: Tensor): void {
    this._buffers.set(name, tensor);
    // DECLARE persistent state (task #70 D3): a buffer is a non-parameter
    // persistent tensor — register it as REG state, same as a param.
    this.api.registerState(tensor);
    Object.defineProperty(this, name, {
      value: tensor,
      writable: true,
      configurable: true,
    });
  }

  /**
   * Register a child module for recursive train()/eval() propagation.
   * Usually not needed — assigning a Module-typed property auto-registers it.
   * Use this for indexed children (e.g., `registerModule("0", child)`).
   */
  registerModule(name: string, module: Module): void {
    this._modules.set(name, module);
  }

  /** Iterate over direct child modules. */
  protected children(): IterableIterator<Module> {
    return this._modules.values();
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
   * Get all learnable parameters, recursively.
   * Collects own parameters then recurses into registered child modules.
   */
  parameters(): Tensor[] {
    const result: Tensor[] = [];
    for (const param of this._params.values()) {
      result.push(param);
    }
    for (const child of this._modules.values()) {
      result.push(...child.parameters());
    }
    return result;
  }

  /**
   * Get all named parameters with dotted key paths, recursively.
   * E.g. "layers.0.attn.weight" for nested modules.
   */
  namedParameters(prefix = ""): [string, Tensor][] {
    const result: [string, Tensor][] = [];
    const p = prefix ? prefix + "." : "";
    for (const [name, param] of this._params) {
      result.push([p + name, param]);
    }
    for (const [name, child] of this._modules) {
      result.push(...child.namedParameters(p + name));
    }
    return result;
  }

  /**
   * Return the module's state as a flat dictionary of named parameters.
   * Keys are dotted paths like "layers.0.weight".
   */
  stateDict(): Record<string, Tensor> {
    return Object.fromEntries(this.namedParameters());
  }

  /**
   * Load parameters from a state dictionary using in-place copy.
   * Keys must match those from stateDict().
   */
  loadStateDict(stateDict: Record<string, Tensor>): void {
    for (const [name, param] of this.namedParameters()) {
      const src = stateDict[name];
      if (!src) {
        throw new Error(`Missing key in state_dict: ${name}`);
      }
      param.copy_(src);
    }
  }

  /**
   * Move all parameters and buffers to the given device.
   * Returns this module for chaining.
   */
  to(device: DeviceKind): this {
    // to() REPLACES the wrapper (a fresh moved tensor), so re-register the new
    // wrapper as REG state (task #70 D3); the old wrapper's registration dies with
    // it (WeakSet). This is the one wholesale-replacement case §D3 calls out.
    for (const [name, param] of this._params) {
      const moved = this.api.registerState(this.api.to(param, device));
      this._params.set(name, moved);
      (this as Record<string, unknown>)[name] = moved;
    }
    for (const [name, buffer] of this._buffers) {
      const moved = this.api.registerState(this.api.to(buffer, device));
      this._buffers.set(name, moved);
      (this as Record<string, unknown>)[name] = moved;
    }
    for (const child of this._modules.values()) {
      child.to(device);
    }
    return this;
  }

  /**
   * Forward pass. Subclasses must implement this.
   */
  abstract forward(input: Tensor): Tensor;
}
