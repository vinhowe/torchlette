/**
 * Cache-key ⇔ content coherence guard (#92) — injected-defect gates.
 *
 * These tests INJECT a key-underspan (a cache key that omits a field the
 * content builder reads) and assert the guard catches it LOUDLY rather than
 * serving a stale artifact. This is the acceptance test for the seam mechanism
 * that closes the key≠content cache class (fusion pipeline cache + row-program
 * kernel cache).
 *
 * The unit tests (guard mechanism) run everywhere. The two integration tests
 * drive the REAL caches and need a GPU device.
 */
import { beforeAll, describe, expect, it } from "vitest";
import { initWebGPU } from "../../src/backend/webgpu";
import {
  makeCacheKeyGuard,
  normalizeWgsl,
  wgslContentEqual,
} from "../../src/backend/webgpu/cache-key-guard";
import { FusionKernelCache } from "../../src/backend/webgpu/fusion-dispatch";
import type { FusedKernelRecipe } from "../../src/backend/webgpu/fusion-types";
import { dispatchRowProgram } from "../../src/backend/webgpu/row-program-dispatch";
import { requireContext } from "../../src/backend/webgpu/webgpu-state";
import type { RowProgram } from "../../src/compiler/row-program-types";
import { cpuOnly } from "../helpers/webgpu";

const SKIP = cpuOnly;

// ============================================================================
// Unit: the guard mechanism
// ============================================================================

// Exact string equality — for the generic-mechanism tests only. The real
// guards use wgslContentEqual (alpha-equivalence); see the normalizeWgsl block.
const exactEqual = (a: string, b: string) => a === b;
const exactDescribe = (a: string, b: string) => `cached ${a} vs fresh ${b}`;

describe("makeCacheKeyGuard mechanism", () => {
  it("passes silently when regenerated content matches the cached content", () => {
    const guard = makeCacheKeyGuard<string>("test", exactEqual, exactDescribe);
    // Same content on every regenerate — coherent, never throws.
    expect(() => guard.check("k", "WGSL-A", () => "WGSL-A")).not.toThrow();
  });

  it("throws LOUDLY on the FIRST hit when the key under-spans (content differs)", () => {
    const guard = makeCacheKeyGuard<string>(
      "fusion-test",
      exactEqual,
      exactDescribe,
    );
    // The key 'k' cached "WGSL-A", but regenerating from the (different) recipe
    // that maps to the SAME key yields "WGSL-B": the key under-spans the codegen.
    // The first hit is always checked, so this throws immediately.
    expect(() => guard.check("k", "WGSL-A", () => "WGSL-B")).toThrow(
      /cache-key incoherence at 'fusion-test'/,
    );
  });

  it("checks the first hit on each distinct key regardless of sampling", () => {
    const guard = makeCacheKeyGuard<string>("t", exactEqual, exactDescribe);
    // A brand-new key's first hit is always checked (counter starts at 0).
    let regenCount = 0;
    guard.check("key-A", "X", () => {
      regenCount++;
      return "X";
    });
    guard.check("key-B", "Y", () => {
      regenCount++;
      return "Y";
    });
    expect(regenCount).toBe(2);
  });
});

// ============================================================================
// WGSL alpha-equivalence (the property byte-comparison lacked)
// ============================================================================
//
// The tile-IR codegen names local bindings from values that vary run-to-run
// without changing semantics (`t<node.id>`, freshVar `_cse<n>`/`_flat<n>`). Two
// legitimate cache hits therefore produce ALPHA-equivalent, byte-different WGSL.
// wgslContentEqual must treat those as equal (else the guard false-fires on
// every real hit and forces a sequential fallback) while still catching a
// genuine STRUCTURAL difference (the underspan class).
describe("normalizeWgsl / wgslContentEqual alpha-equivalence", () => {
  it("treats WGSL differing only in generated local names as equal", () => {
    // Same structure, generated locals renamed (t12/_cse0 vs t58/_cse7).
    const a = "let v0 = in0[_cse0]; let t12 = (v0 * 0.25); out[i] = t12;";
    const b = "let v0 = in0[_cse7]; let t58 = (v0 * 0.25); out[i] = t58;";
    expect(a).not.toEqual(b); // byte-different (the false-positive case)
    expect(wgslContentEqual(a, b)).toBe(true); // alpha-equivalent
    expect(normalizeWgsl(a)).toEqual(normalizeWgsl(b));
  });

  it("still flags a STRUCTURAL difference (different op — a real underspan)", () => {
    const a = "let v0 = in0[_cse0]; let t12 = (v0 * 0.25);";
    const b = "let v0 = in0[_cse0]; let t12 = (v0 + 0.25);"; // * vs + survives
    expect(wgslContentEqual(a, b)).toBe(false);
  });

  it("still flags a swapped operand (renaming cannot hide it)", () => {
    // Positional input locals v0/v1 are NOT generated names; swapping them is
    // a real structural change that must survive normalization.
    const a = "let t1 = (v0 - v1);";
    const b = "let t1 = (v1 - v0);";
    expect(wgslContentEqual(a, b)).toBe(false);
  });

  it("normalizes consistently (same generated name → same placeholder)", () => {
    // t5 used twice must map to ONE placeholder both times.
    const norm = normalizeWgsl("let t5 = a; let t9 = t5 + t5;");
    expect(norm).toBe("let «0» = a; let «1» = «0» + «0»;");
  });
});

// ============================================================================
// Integration: fusion kernel cache (site 1)
// ============================================================================

describe.skipIf(SKIP)("FusionKernelCache key-underspan defect", () => {
  beforeAll(async () => {
    await initWebGPU();
  });

  function reluRecipe(): FusedKernelRecipe {
    return {
      id: "guard_relu",
      nodes: [
        {
          id: 1,
          op: "relu",
          inputs: [-1],
          shape: [64],
          dtype: "f32",
          isOutput: true,
        },
      ],
      inputs: [{ id: 100, index: 0, shape: [64], dtype: "f32" }],
      outputs: [{ nodeId: 1, index: 0, shape: [64], dtype: "f32" }],
    };
  }

  it("catches a STRUCTURAL underspan: same key, structurally different WGSL", () => {
    const ctx = requireContext();
    const cache = new FusionKernelCache();
    const recipe = reluRecipe();

    // First getOrCreate MISSES → codegens the real relu WGSL, caches under KEY.
    const { kernel } = cache.getOrCreate(ctx.device, recipe);

    // Inject a STRUCTURAL underspan: corrupt the cached WGSL so the next hit's
    // regeneration (the real, unchanged codegen) STRUCTURALLY disagrees with
    // what's cached. This models a key that omits a field the codegen bakes in
    // (a stride/op/operand) — the underspan class #92 closes. `structuralMarker`
    // is NOT a generated identifier (no trailing digit / freshVar prefix), so it
    // survives alpha-normalization; the guard MUST throw on the second (HIT)
    // getOrCreate.
    (kernel as { source: string }).source =
      `${kernel.source}\nlet structuralMarker = something_that_differs;`;

    expect(() => cache.getOrCreate(ctx.device, recipe)).toThrow(
      /cache-key incoherence at 'FusionKernelCache'/,
    );
  });
});

// ============================================================================
// Integration: row-program kernel cache (site 2)
// ============================================================================

describe.skipIf(SKIP)("row-program kernelCache key-underspan defect", () => {
  beforeAll(async () => {
    await initWebGPU();
  });

  // Two programs that produce DIFFERENT WGSL (different write bodyExpr) but are
  // handed the SAME cacheKey — a directly injected key-underspan. The first
  // dispatch caches program-A's kernel; the second HITS that key while the
  // guard recompiles program-B's WGSL → mismatch → loud throw.
  function programWithBody(
    cacheKey: string,
    reduceOp: "sum" | "max",
  ): RowProgram {
    return {
      inputs: [{ dtype: "f32" }],
      output: { dtype: "f32" },
      dim: 1,
      phases: [
        {
          kind: "reduce",
          reduceOp,
          bodyExpr: { kind: "input", bufferIndex: 0 },
        },
        {
          kind: "write",
          bodyExpr: { kind: "reduceResult", phaseIndex: 0 },
          scalarOutput: true,
        },
      ],
      cacheKey,
    };
  }

  it("catches two DIFFERENT programs sharing one forced cacheKey", () => {
    const ctx = requireContext();
    const numRows = 4;
    const dimSize = 8;

    const inputBuf = ctx.device.createBuffer({
      size: numRows * dimSize * 4,
      usage: 0x0080 | 0x0008, // STORAGE | COPY_DST
    });

    // Program A (sum) — first dispatch MISSES → caches sum-kernel under KEY.
    const progA = programWithBody("COLLIDE_KEY", "sum");
    dispatchRowProgram(progA, [inputBuf], numRows, dimSize, numRows);

    // Program B (max) — SAME forced cacheKey but different WGSL. Second dispatch
    // HITS the cached sum kernel; the guard recompiles B's WGSL (max) → mismatch.
    const progB = programWithBody("COLLIDE_KEY", "max");
    expect(() =>
      dispatchRowProgram(progB, [inputBuf], numRows, dimSize, numRows),
    ).toThrow(/cache-key incoherence at 'row-program kernelCache'/);
  });
});
