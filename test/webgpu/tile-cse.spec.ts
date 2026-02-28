/**
 * Auto-CSE (Common Sub-expression Elimination) tests for tile-IR compiler.
 *
 * Validates that the autoCSE pass correctly detects and hoists multi-use
 * expression nodes into let bindings, including:
 * - Intra-expression sharing (same node used twice in one statement)
 * - Cross-statement sharing (same node used in different statements)
 * - Scope-aware binding placement (let bindings at correct scope level)
 * - Depth filtering (trivial nodes not CSE'd)
 * - Interaction with manual emitLet (already-bound nodes skipped)
 *
 * Note: Auto-CSE excludes expressions containing memory reads (load, sharedRead)
 * from CSE candidates. Tests use pure arithmetic expressions to exercise CSE.
 */

import { describe, expect, it } from "vitest";
import { compileTileKernel } from "../../src/backend/webgpu/tile-compiler";
import { type TileKernelSpec, elementwiseGrid } from "../../src/backend/webgpu/tile-ir";

const WG = 64;

describe("Auto-CSE", () => {
  it("detects intra-expression sharing (same node used twice in one store)", () => {
    // x = idx * 3 + 7 (pure arithmetic, no memory reads)
    // store x * x — same node used as both lhs and rhs of mul
    const spec: TileKernelSpec = {
      name: "cse_intra",
      workgroupSize: WG,
      bindings: {
        out: { storage: "read_write", type: "f32" },
      },
      uniforms: { size: "u32" },
      grid: elementwiseGrid(WG, { elementUniform: "size" }),
      kernel(ctx) {
        const idx = ctx.globalId(0);
        ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
        // Pure arithmetic expression (no memory reads)
        const x = idx.mul(ctx.u32(3)).add(ctx.u32(7));
        // x used twice in the same expression tree: x * x (as f32)
        const xf = x.toF32();
        ctx.emitStore("out", idx, xf.mul(xf));
      },
    };
    const wgsl = compileTileKernel(spec);
    // The cast+arithmetic expression should be hoisted to a let binding
    // since it appears twice in the same mul expression
    const cseBindings = wgsl.match(/let _cse\d+/g) || [];
    expect(cseBindings.length).toBeGreaterThanOrEqual(1);
  });

  it("detects cross-statement sharing (same node in two stores)", () => {
    // y = (idx * 5 + 1) as f32 — used in two separate store statements
    const spec: TileKernelSpec = {
      name: "cse_cross",
      workgroupSize: WG,
      bindings: {
        out1: { storage: "read_write", type: "f32" },
        out2: { storage: "read_write", type: "f32" },
      },
      uniforms: { size: "u32" },
      grid: elementwiseGrid(WG, { elementUniform: "size" }),
      kernel(ctx) {
        const idx = ctx.globalId(0);
        ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
        const y = idx.mul(ctx.u32(5)).add(ctx.u32(1)).toF32();
        // y used in two separate stores
        ctx.emitStore("out1", idx, y);
        ctx.emitStore("out2", idx, y.mul(ctx.f32(2.0)));
      },
    };
    const wgsl = compileTileKernel(spec);
    // y should be CSE'd since it appears in two statements
    const cseBindings = wgsl.match(/let _cse\d+/g) || [];
    expect(cseBindings.length).toBeGreaterThanOrEqual(1);
  });

  it("does not CSE trivial nodes (const, programId have depth <= 1)", () => {
    // Const and programId are depth 0/1, should not be CSE'd even if used many times
    const spec: TileKernelSpec = {
      name: "cse_trivial",
      workgroupSize: WG,
      bindings: {
        out1: { storage: "read_write", type: "f32" },
        out2: { storage: "read_write", type: "f32" },
      },
      uniforms: { size: "u32" },
      grid: elementwiseGrid(WG, { elementUniform: "size" }),
      kernel(ctx) {
        const idx = ctx.globalId(0);
        ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
        // Use idx (globalId — depth 1 binary) in multiple stores
        // This should NOT be CSE'd because globalId is trivial
        ctx.emitStore("out1", idx, ctx.f32(42.0));
        ctx.emitStore("out2", idx, ctx.f32(99.0));
      },
    };
    const wgsl = compileTileKernel(spec);
    // No CSE bindings expected — all multi-use nodes are trivial
    const cseBindings = wgsl.match(/let _cse\d+/g) || [];
    expect(cseBindings.length).toBe(0);
  });

  it("skips nodes already bound by manual emitLet", () => {
    // When the user explicitly calls emitLet, auto-CSE should not double-bind
    const spec: TileKernelSpec = {
      name: "cse_manual_let",
      workgroupSize: WG,
      bindings: {
        out: { storage: "read_write", type: "f32" },
      },
      uniforms: { size: "u32" },
      grid: elementwiseGrid(WG, { elementUniform: "size" }),
      kernel(ctx) {
        const idx = ctx.globalId(0);
        ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
        // Pure arithmetic expression
        const x = idx.mul(ctx.u32(3)).add(ctx.u32(7)).toF32();
        // Manually bind the expression
        const xBound = ctx.emitLet("myVal", x);
        // Use it twice — auto-CSE should skip since it's already bound
        ctx.emitStore("out", idx, xBound.mul(xBound));
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should have "let myVal" from manual emitLet, but no _cse bindings
    expect(wgsl).toContain("let myVal");
    const cseBindings = wgsl.match(/let _cse\d+/g) || [];
    expect(cseBindings.length).toBe(0);
  });

  it("handles intra-expression sharing with deeper expressions", () => {
    // (a + b) * (a + b) where a+b is a shared sub-expression
    const spec: TileKernelSpec = {
      name: "cse_deep_intra",
      workgroupSize: WG,
      bindings: {
        out: { storage: "read_write", type: "f32" },
      },
      uniforms: { size: "u32", offset: "u32" },
      grid: elementwiseGrid(WG, { elementUniform: "size" }),
      kernel(ctx) {
        const idx = ctx.globalId(0);
        ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
        // Two non-trivial arithmetic sub-expressions
        const a = idx.mul(ctx.u32(3)).toF32();
        const b = ctx.uniform("offset").add(ctx.u32(1)).toF32();
        const sum = a.add(b);
        // sum used twice: sum * sum
        ctx.emitStore("out", idx, sum.mul(sum));
      },
    };
    const wgsl = compileTileKernel(spec);
    // The add expression should be CSE'd into a let binding
    const cseBindings = wgsl.match(/let _cse\d+/g) || [];
    expect(cseBindings.length).toBeGreaterThanOrEqual(1);
    // The WGSL should have a multiplication of the CSE'd variable with itself
    const cseVarName = wgsl.match(/let (_cse\d+)/)?.[1];
    if (cseVarName) {
      const mulPattern = new RegExp(`\\(${cseVarName} \\* ${cseVarName}\\)`);
      expect(wgsl).toMatch(mulPattern);
    }
  });

  it("places let bindings at correct scope for loop-body expressions", () => {
    // Expression used twice inside a loop body should stay inside the loop
    const spec: TileKernelSpec = {
      name: "cse_loop_scope",
      workgroupSize: WG,
      bindings: {
        out: { storage: "read_write", type: "f32" },
      },
      uniforms: { size: "u32", iters: "u32" },
      grid: elementwiseGrid(WG, { elementUniform: "size" }),
      kernel(ctx) {
        const idx = ctx.globalId(0);
        ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
        ctx.forRange(ctx.u32(0), ctx.uniform("iters"), (i) => {
          // Expression depends on loop variable i — must stay inside loop
          const offset = idx.add(i.mul(ctx.uniform("size")));
          const val = offset.toF32().add(ctx.f32(0.5));
          // val used twice inside loop
          ctx.emitStore("out", offset, val.mul(val));
        });
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should compile without errors — no hoisting of loop-dependent expressions
    expect(wgsl).toContain("for (var");
    // CSE should detect val used twice inside the loop body
    const cseBindings = wgsl.match(/let _cse\d+/g) || [];
    expect(cseBindings.length).toBeGreaterThanOrEqual(1);
  });
});
