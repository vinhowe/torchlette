/**
 * ModuleList — an ordered container of child modules.
 * Similar to PyTorch's nn.ModuleList.
 */

import type { Tensor, Torchlette } from "../frontend/torchlette";
import { Module } from "./module";

export class ModuleList extends Module {
  private readonly _list: Module[] = [];

  constructor(api: Torchlette, modules?: Module[]) {
    super(api);
    if (modules) for (const m of modules) this.append(m);
  }

  append(module: Module): this {
    this.registerModule(String(this._list.length), module);
    this._list.push(module);
    return this;
  }

  get length(): number {
    return this._list.length;
  }

  get(i: number): Module {
    return this._list[i];
  }

  [Symbol.iterator](): Iterator<Module> {
    return this._list[Symbol.iterator]();
  }

  forward(_input: Tensor): Tensor {
    throw new Error("ModuleList does not implement forward()");
  }
}
