/**
 * Tests for the public tile-IR custom kernel API.
 * Verifies that users can write, compile, and dispatch custom GPU kernels
 * using the exported TileKernelSpec / compileTileKernel / createTileKernelDispatcher.
 */
import { describe, it, expect, beforeAll } from "vitest";
import {
  type TileKernelSpec,
  KernelContext,
  compileTileKernel,
  createTileKernelDispatcher,
  elementwiseGrid,
  ceilDivGrid,
  singleWorkgroup,
  initWebGPU,
  getWebGPUInitError,
  webgpuBackend,
} from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

describe("compileTileKernel", () => {
  it("compiles a vector-add spec to valid WGSL", () => {
    const WG = 64;
    const spec: TileKernelSpec = {
      name: "vecadd",
      workgroupSize: WG,
      bindings: [
        { name: "a", direction: "in" },
        { name: "b", direction: "in" },
        { name: "out", direction: "out" },
      ],
      uniforms: { size: "u32" as const },
      grid: elementwiseGrid(WG),
      kernel(ctx: KernelContext) {
        const idx = ctx.globalId(0);
        ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
        const a = ctx.load("a", idx);
        const b = ctx.load("b", idx);
        ctx.emitStore("out", idx, a.add(b));
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("@compute");
    expect(wgsl).toContain("@workgroup_size(64");
    expect(wgsl).toContain("@group(0)");
    // Should have 3 storage bindings + 1 uniform
    const bindingMatches = wgsl.match(/@binding\(\d+\)/g);
    expect(bindingMatches!.length).toBe(4); // a, b, out, uniforms
  });

  it("compiles a masked kernel using blockLoad/blockStore", () => {
    const WG = 64;
    const spec: TileKernelSpec = {
      name: "masked_scale",
      workgroupSize: WG,
      bindings: [
        { name: "src", direction: "in" },
        { name: "out", direction: "out" },
      ],
      uniforms: { size: "u32" as const, scale: "f32" as const },
      grid: elementwiseGrid(WG),
      kernel(ctx: KernelContext) {
        const idx = ctx.globalId(0);
        const size = ctx.uniform("size");
        const val = ctx.blockLoad("src", idx, size);
        ctx.blockStore("out", idx, size, val.mul(ctx.uniform("scale")));
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("@compute");
    // blockLoad produces a select (mask ? load : 0.0)
    expect(wgsl).toContain("select");
  });
});

describe("grid helpers", () => {
  it("elementwiseGrid computes correct workgroup counts", () => {
    const grid = elementwiseGrid(64);
    // 100 elements / 64 threads per WG = ceil(100/64) = 2
    expect(grid({ total_elements: 100 })).toEqual([2]);
    // exact fit
    expect(grid({ total_elements: 128 })).toEqual([2]);
    // 1 element => 1 workgroup
    expect(grid({ total_elements: 1 })).toEqual([1]);
  });

  it("ceilDivGrid divides correctly", () => {
    const grid = ceilDivGrid(256);
    expect(grid({ total_elements: 1024 })).toEqual([4]);
    expect(grid({ total_elements: 1025 })).toEqual([5]);
  });

  it("singleWorkgroup always returns [1]", () => {
    const grid = singleWorkgroup();
    expect(grid({})).toEqual([1]);
    expect(grid({ anything: 999 })).toEqual([1]);
  });
});

describe.skipIf(cpuOnly)("createTileKernelDispatcher", () => {
  beforeAll(async () => {
    const ready = await initWebGPU();
    if (!ready) throw new Error(`WebGPU init failed: ${getWebGPUInitError()}`);
  });

  it("dispatches a custom vector-add kernel end-to-end", async () => {
    const WG = 64;
    const spec: TileKernelSpec = {
      name: "custom_vecadd",
      workgroupSize: WG,
      bindings: [
        { name: "a", direction: "in" },
        { name: "b", direction: "in" },
        { name: "out", direction: "out" },
      ],
      uniforms: { size: "u32" as const },
      grid: elementwiseGrid(WG),
      kernel(ctx: KernelContext) {
        const idx = ctx.globalId(0);
        ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
        const a = ctx.load("a", idx);
        const b = ctx.load("b", idx);
        ctx.emitStore("out", idx, a.add(b));
      },
    };

    const dispatcher = createTileKernelDispatcher(spec);

    const N = 128;
    const aData = Array.from({ length: N }, (_, i) => i);
    const bData = Array.from({ length: N }, (_, i) => i * 2);

    const aTensor = webgpuBackend.ops.tensorFromArray(aData, [N]);
    const bTensor = webgpuBackend.ops.tensorFromArray(bData, [N]);

    const outBuf = dispatcher.dispatch(
      { a: (aTensor as any).buffer, b: (bTensor as any).buffer },
      { size: N },
    );

    // Read result back
    const result = await webgpuBackend.ops.read({
      buffer: outBuf,
      shape: [N],
      strides: [1],
      offset: 0,
      dtype: "f32",
      isContiguous: true,
    });

    const expected = aData.map((a, i) => a + bData[i]);
    expect(result.length).toBe(N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 4);
    }
  });
});
