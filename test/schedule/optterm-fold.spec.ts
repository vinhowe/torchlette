/**
 * R1 — the `OptTerm → tile-IR` fold, catalog coverage (derived-optimizer-realizer
 * campaign; docs/derived-optimizer-realizer-design.md ruling O1, phase R1).
 *
 * PURE COMPILATION (no GPU): the tile-IR codegen is a WGSL compiler, so the fold
 * can be exercised end-to-end (fold → compileTileKernel → WGSL) in the cpu
 * project. The NUMERIC differential against `evalOptTerm` (fold-compiled kernel
 * dispatched on GPU == graph result, ≤1e-6) lives in `tools/optterm-fold-parity.ts`.
 *
 * This gate pins the STRUCTURAL contract over the whole `OPTIMIZER_PROGRAMS`
 * catalog:
 *   (a) every elementwise program (adamw / sgd / sgd_momentum / lion) folds to
 *       compilable WGSL — the fold is total over the elementwise sub-algebra;
 *   (b) EXACTLY the mm-bearing programs (muon) refuse, with a typed, named
 *       `OptimizerPackRefusal` — the refusal is structural (no elementwise tile-IR
 *       node for a contraction), the single source being the same refusal
 *       `assertFlattenable` raises.
 */

import { describe, expect, it } from "vitest";
import { compileTileKernel } from "../../src/backend/webgpu/tile-compiler";
import type {
  BlockExpr,
  KernelContext,
  TileKernelSpec,
} from "../../src/backend/webgpu/tile-ir";
import {
  ADAMW_PROGRAM,
  LION_PROGRAM,
  MUON_PROGRAM,
  OPTIMIZER_PROGRAMS,
  type OptimizerProgram,
  type OptTerm,
  SGD_MOMENTUM_PROGRAM,
  SGD_PROGRAM,
} from "../../src/ops/semantic/optimizer";
import {
  lowerOptTermToTileIR,
  type FoldRoleBindings,
} from "../../src/schedule/optterm-fold";
import {
  OptimizerPackRefusal,
  optTermHasMatmul,
} from "../../src/optim/pack-optimizer";

/** All role names an OptTerm references (the bindings the fold needs). */
function collectRoles(t: OptTerm, into: Set<string> = new Set()): Set<string> {
  switch (t.k) {
    case "role":
      into.add(t.name);
      break;
    case "u":
      collectRoles(t.a, into);
      break;
    case "b":
      collectRoles(t.a, into);
      collectRoles(t.b, into);
      break;
    case "mm":
      collectRoles(t.a, into);
      collectRoles(t.b, into);
      break;
    default:
      break; // "c" — no role
  }
  return into;
}

/** Every output term of a program (state updates + the param update). */
function outputTerms(p: OptimizerProgram): { name: string; expr: OptTerm }[] {
  return [
    ...p.stateUpdates.map((su) => ({ name: su.slot, expr: su.expr })),
    { name: "param", expr: p.paramUpdate },
  ];
}

/** Build a fold-derived elementwise kernel spec: each role is a `read` binding
 *  loaded at idx, each output term is folded and stored. */
function foldKernelSpec(p: OptimizerProgram): TileKernelSpec {
  const roleNames = new Set<string>();
  for (const t of outputTerms(p)) collectRoles(t.expr, roleNames);
  const bindings: TileKernelSpec["bindings"] = {};
  for (const r of roleNames) bindings[r] = { storage: "read", type: "f32" };
  const outs = outputTerms(p).map((t) => `out_${t.name}`);
  for (const o of outs) bindings[o] = { storage: "read_write", type: "f32" };
  const WG = 64;
  return {
    name: `fold_${p.name}`,
    workgroupSize: WG,
    bindings,
    uniforms: { size: "u32" },
    grid: (u: Record<string, number>) => [Math.ceil((u.size ?? 0) / WG)],
    kernel(ctx: KernelContext) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
      const roleBind: Record<string, BlockExpr> = {};
      for (const r of roleNames) roleBind[r] = ctx.load(r, idx);
      const memo = new Map<OptTerm, BlockExpr>();
      for (const t of outputTerms(p)) {
        const val = lowerOptTermToTileIR(
          t.expr,
          ctx,
          roleBind as FoldRoleBindings,
          memo,
        );
        ctx.emitStore(`out_${t.name}`, idx, val);
      }
    },
  };
}

describe("R1 OptTerm→tile-IR fold — catalog coverage", () => {
  const elementwise = [
    ADAMW_PROGRAM,
    SGD_PROGRAM,
    SGD_MOMENTUM_PROGRAM,
    LION_PROGRAM,
  ];

  for (const p of elementwise) {
    it(`folds ${p.name} to compilable WGSL`, () => {
      const wgsl = compileTileKernel(foldKernelSpec(p));
      expect(wgsl).toContain("@compute");
      // The kernel must actually store every output term.
      for (const t of outputTerms(p)) expect(wgsl).toContain(`out_${t.name}`);
    });
  }

  it("REFUSES muon structurally (contraction / Newton–Schulz mm node)", () => {
    // The fold must throw the typed, named refusal — not a bare Error — when it
    // reaches the mm node inside MUON_PROGRAM.paramUpdate.
    let caught: unknown;
    try {
      // The fold executes inside the kernel closure — compile to trigger it.
      compileTileKernel(foldKernelSpec(MUON_PROGRAM));
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(OptimizerPackRefusal);
    expect((caught as Error).name).toBe("OptimizerPackRefusal");
    expect((caught as Error).message).toContain("contraction");
  });

  it("across the catalog, EXACTLY the mm-bearing programs refuse (single source)", () => {
    const refused: string[] = [];
    for (const p of OPTIMIZER_PROGRAMS) {
      let didRefuse = false;
      try {
        compileTileKernel(foldKernelSpec(p));
      } catch (e) {
        expect(e).toBeInstanceOf(OptimizerPackRefusal);
        didRefuse = true;
      }
      const hasMm =
        optTermHasMatmul(p.paramUpdate) ||
        p.stateUpdates.some((su) => optTermHasMatmul(su.expr));
      // The fold refuses IFF the program contains a contraction — the refusal is
      // the SAME structural fact `assertFlattenable` enforces.
      expect(didRefuse).toBe(hasMm);
      if (didRefuse) refused.push(p.name);
    }
    expect(refused).toEqual(["muon"]);
  });

  it("an unbound role throws a clear (non-refusal) error", () => {
    const WG = 64;
    const spec: TileKernelSpec = {
      name: "fold_missing_role",
      workgroupSize: WG,
      bindings: { out: { storage: "read_write", type: "f32" } },
      uniforms: { size: "u32" },
      grid: (u: Record<string, number>) => [Math.ceil((u.size ?? 0) / WG)],
      kernel(ctx: KernelContext) {
        const idx = ctx.globalId(0);
        // Bind NOTHING — folding SGD's `p - lr*g` must fail on the first role.
        const val = lowerOptTermToTileIR(SGD_PROGRAM.paramUpdate, ctx, {});
        ctx.emitStore("out", idx, val);
      },
    };
    expect(() => compileTileKernel(spec)).toThrow(/role '.*' but it was not bound/);
  });
});
