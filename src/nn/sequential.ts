/**
 * Sequential — chains modules in order.
 * Similar to PyTorch's nn.Sequential.
 */

import type { Tensor, Torchlette } from "../frontend";
import { Module } from "./module";

export class Sequential extends Module {
  constructor(api: Torchlette, ...modules: Module[]) {
    super(api);
    for (let i = 0; i < modules.length; i++) {
      this.registerModule(String(i), modules[i]);
    }
  }

  forward(input: Tensor): Tensor {
    let x = input;
    for (const child of this.children()) {
      x = child.forward(x);
    }
    return x;
  }
}
