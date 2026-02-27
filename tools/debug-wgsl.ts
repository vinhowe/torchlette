import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";
import { makeSumDimSpec } from "../src/backend/webgpu/reduction-tile-ir";

// Reproduce: sum dim=1 of [2,3] tensor
const inputShape = [2, 3];
const inputStrides = [3, 1];
const normalizedDims = [1];
const outShape = [2];
const outStrides = [1];
const inputToOutDim = [0, -1];

const spec = makeSumDimSpec(inputShape, inputStrides, normalizedDims,
  outShape, outStrides, inputToOutDim, false /* sequential */);

const wgsl = compileTileKernel(spec);
console.log("=== Generated WGSL ===");
console.log(wgsl);
