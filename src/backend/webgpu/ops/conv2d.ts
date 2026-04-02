/**
 * Conv2d op: direct 2D convolution using tile-IR.
 *
 * Each thread computes one output element by sliding the kernel window
 * over input spatial positions. Uses blockLoad for automatic padding
 * (OOB reads return 0).
 *
 * Layout: NCHW (batch, channels, height, width) — same as PyTorch.
 */

import type { BackendTensor } from "../../types";
import { resolveOutputBuffer } from "../buffer-arena";
import { requireContext } from "../gpu-context";
import type { GPUBuffer } from "../gpu-types";
import { asGPUTensor } from "../gpu-types";
import { WORKGROUP_SIZE } from "../shape-utils";
import { createTensor } from "../tensor";
import {
  createTileKernelDispatcher,
  type TileKernelInstance,
} from "../tile-dispatch";
import type { DataType, TileKernelSpec } from "../tile-ir";
import { elementwiseGrid } from "../tile-ir";

const WG = WORKGROUP_SIZE; // 256

export interface Conv2dOptions {
  stride?: number | [number, number];
  padding?: number | [number, number];
  hasBias?: boolean;
  outBuffer?: GPUBuffer;
}

// ============================================================================
// Output shape computation
// ============================================================================

function conv2dOutputShape(
  inputShape: number[],
  weightShape: number[],
  stride: [number, number],
  padding: [number, number],
): [number, number, number, number] {
  const [N, , H, W] = inputShape;
  const [Cout, , KH, KW] = weightShape;
  const outH = Math.floor((H + 2 * padding[0] - KH) / stride[0] + 1);
  const outW = Math.floor((W + 2 * padding[1] - KW) / stride[1] + 1);
  return [N, Cout, outH, outW];
}

function normalizeParam(
  p: number | [number, number] | undefined,
  fallback: number,
): [number, number] {
  if (p === undefined) return [fallback, fallback];
  if (typeof p === "number") return [p, p];
  return p;
}

// ============================================================================
// Tile-IR kernel spec
// ============================================================================

function makeConv2dSpec(
  kH: number,
  kW: number,
  Cin: number,
  hasBias: boolean,
  dtype: DataType = "f32",
): TileKernelSpec {
  const bindings: Record<
    string,
    { storage: "read" | "read_write"; type: DataType }
  > = {
    input: { storage: "read", type: dtype },
    weight: { storage: "read", type: dtype },
    out: { storage: "read_write", type: dtype },
  };
  if (hasBias) {
    bindings.bias = { storage: "read", type: dtype };
  }

  return {
    name: `conv2d_${kH}x${kW}_c${Cin}${hasBias ? "_bias" : ""}`,
    workgroupSize: WG,
    bindings,
    uniforms: {
      N: "u32",
      Cin: "u32",
      H: "u32",
      W: "u32",
      Cout: "u32",
      outH: "u32",
      outW: "u32",
      strideH: "u32",
      strideW: "u32",
      padH: "u32",
      padW: "u32",
      totalOut: "u32",
    },
    grid: elementwiseGrid(WG, { elementUniform: "totalOut" }),
    kernel(ctx) {
      const idx = ctx.elementIndex(WG, "totalOut");

      const outW = ctx.uniform("outW");
      const outH = ctx.uniform("outH");
      const Cout = ctx.uniform("Cout");
      const H = ctx.uniform("H");
      const W = ctx.uniform("W");

      // Decompose flat index → (n, co, oh, ow)
      const ow = idx.mod(outW);
      const rest1 = idx.div(outW);
      const oh = rest1.mod(outH);
      const rest2 = rest1.div(outH);
      const co = rest2.mod(Cout);
      const n = rest2.div(Cout);

      // Initialize accumulator
      const acc = ctx.emitVar(
        "acc",
        "f32",
        hasBias ? ctx.load("bias", co) : ctx.f32(0),
      );

      // Base input spatial positions (before kernel offset)
      const baseH = oh.mul(ctx.uniform("strideH"));
      const baseW = ow.mul(ctx.uniform("strideW"));
      const padH = ctx.uniform("padH");
      const padW = ctx.uniform("padW");

      // Input strides: [Cin*H*W, H*W, W, 1]
      const inputHW = H.mul(W);
      const inputCHW = ctx.uniform("Cin").mul(inputHW);
      const inputBatchBase = n.mul(inputCHW);

      // Weight strides: [Cin*KH*KW, KH*KW, KW, 1]
      const cKHKW = ctx.u32(Cin * kH * kW);
      const weightCoBase = co.mul(cKHKW);

      // Loop over input channels and kernel spatial positions
      // Compile-time unroll for kH × kW (small kernels), runtime loop for Cin
      ctx.forRange(ctx.u32(0), ctx.uniform("Cin"), (ci) => {
        const inputCBase = inputBatchBase.add(ci.mul(inputHW));
        const weightCBase = weightCoBase.add(ci.mul(ctx.u32(kH * kW)));

        for (let ky = 0; ky < kH; ky++) {
          for (let kx = 0; kx < kW; kx++) {
            // Input position with padding
            const ih = baseH.add(ctx.u32(ky)).sub(padH);
            const iw = baseW.add(ctx.u32(kx)).sub(padW);

            // Bounds check: ih in [0, H) and iw in [0, W)
            // Note: since ih/iw are u32, negative values wrap to large positive,
            // so a single < check handles both bounds
            const inBounds = ih.lt(H).and(iw.lt(W));

            const inputIdx = inputCBase.add(ih.mul(W)).add(iw);
            const inputVal = inBounds.select(
              ctx.load("input", inputIdx),
              ctx.f32(0),
            );

            const weightIdx = weightCBase.add(ctx.u32(ky * kW + kx));
            const weightVal = ctx.load("weight", weightIdx);

            acc.addAssign(inputVal.mul(weightVal));
          }
        }
      });

      ctx.emitStore("out", idx, acc.get());
    },
  };
}

// ============================================================================
// Kernel dispatcher cache
// ============================================================================

const dispatcherCache = new Map<string, TileKernelInstance>();

function getDispatcher(
  kH: number,
  kW: number,
  Cin: number,
  hasBias: boolean,
): TileKernelInstance {
  const key = `conv2d_${kH}x${kW}_c${Cin}${hasBias ? "_b" : ""}`;
  let d = dispatcherCache.get(key);
  if (!d) {
    d = createTileKernelDispatcher(makeConv2dSpec(kH, kW, Cin, hasBias));
    dispatcherCache.set(key, d);
  }
  return d;
}

// ============================================================================
// Public dispatch function
// ============================================================================

export function conv2d(
  _input: BackendTensor,
  _weight: BackendTensor,
  _bias: BackendTensor | undefined,
  options?: Conv2dOptions,
): BackendTensor {
  const input = asGPUTensor(_input);
  const weight = asGPUTensor(_weight);
  const bias = _bias ? asGPUTensor(_bias) : undefined;

  if (input.shape.length !== 4)
    throw new Error(`conv2d: input must be 4D [N,C,H,W], got ${input.shape}`);
  if (weight.shape.length !== 4)
    throw new Error(
      `conv2d: weight must be 4D [Cout,Cin,kH,kW], got ${weight.shape}`,
    );

  const [N, Cin, H, W] = input.shape;
  const [Cout, CinK, kH, kW] = weight.shape;
  if (Cin !== CinK)
    throw new Error(`conv2d: input channels ${Cin} != weight channels ${CinK}`);

  const stride = normalizeParam(options?.stride, 1);
  const padding = normalizeParam(options?.padding, 0);
  const hasBias = !!bias;

  if (hasBias && (bias?.shape.length !== 1 || bias?.shape[0] !== Cout)) {
    throw new Error(`conv2d: bias must be [Cout=${Cout}], got ${bias?.shape}`);
  }

  const outShape = conv2dOutputShape(
    input.shape,
    weight.shape,
    stride,
    padding,
  );
  const [, , outH, outW] = outShape;
  const totalOut = N * Cout * outH * outW;

  const ctx = requireContext();
  const outBuffer = options?.outBuffer
    ? options.outBuffer
    : resolveOutputBuffer(ctx.device, totalOut * 4, [
        input.buffer,
        weight.buffer,
      ]);

  const dispatcher = getDispatcher(kH, kW, Cin, hasBias);

  const buffers: Record<string, GPUBuffer> = {
    input: input.buffer,
    weight: weight.buffer,
    out: outBuffer,
  };
  if (hasBias) {
    buffers.bias = bias?.buffer;
  }

  dispatcher.dispatch(buffers, {
    N,
    Cin,
    H,
    W,
    Cout,
    outH,
    outW,
    strideH: stride[0],
    strideW: stride[1],
    padH: padding[0],
    padW: padding[1],
    totalOut,
  });

  return createTensor(outShape, outBuffer);
}
