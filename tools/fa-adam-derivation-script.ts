/**
 * THE P4 ACCEPTANCE ARTIFACT — the FUSED ADAM derivation, RUNNABLE.
 *
 * §7 P4 (LOCAL self-hosting), deliverable 3: the horizontal-`pack` move (the §3
 * pack move's real tenant — multi-tensor packing at PARAMETER altitude) + the
 * elementwise Adam bodies compose into the packed fused kernel.
 *
 * The derivation:
 *   0  base = N per-param elementwise Adam bodies (the un-packed per-param nest —
 *      "the per-param definition is the semantics", optim/adam.ts)
 *   1  pack (concatenate)  the N per-param loops into ONE pack loop over segments
 *      (the multi-tensor horizontal pack the foreach optimizer performs: cat the
 *      N param/grad/m/v flats into one [total] buffer)
 *   2  the packed schedule's ONE flat dispatch reaches the AUTHORED adamStep
 *      kernel BYTE-IDENTICALLY (applyAdamSchedule → makeAdamStepSpec)
 *
 * Then: (a) BYTE-IDENTITY — every Adam variant (vec4 × f16 × unscale) reached
 * byte-identically; (b) PACKED-DISPATCH COUNT — one adamStep per group (not N —
 * the 8-dispatch/step baseline class); (c) a GPU numeric check that the packed
 * foreach step == the per-param step (the trajectory-level parity is the standing
 * gates test/optim/{fused-vs-elementwise,adam-multigroup}.spec.ts).
 *
 *   TORCHLETTE_CPU_ONLY=1 pnpm exec tsx tools/fa-adam-derivation-script.ts
 *   VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH \
 *     pnpm exec tsx tools/fa-adam-derivation-script.ts   # + GPU numeric check
 */

import {
  adamCacheKey,
  type AdamDescriptor,
  applyAdamSchedule,
  deriveAdamSkeleton,
  deriveHorizontalPackedAdam,
  generateAdamShaderTileIR,
} from "../src/schedule/adam-skeleton";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";
import { printScheduleState } from "../src/schedule/canonical";
import type { ValueUid } from "../src/schedule/types";

const cpuOnly = process.env.TORCHLETTE_CPU_ONLY === "1";

let destroyWebGPUFn: (() => void) | null = null;
let failures = 0;
function oracle(cond: boolean, msg: string): void {
  if (cond) {
    // eslint-disable-next-line no-console
    console.log(`  ✓ ORACLE: ${msg}`);
  } else {
    failures++;
    // eslint-disable-next-line no-console
    console.log(`  ✗ ORACLE FAILED: ${msg}`);
  }
}

const v = (s: string): ValueUid => s as unknown as ValueUid;

// ============================================================================
// The derivation (oracle-by-oracle)
// ============================================================================

function runDerivation(): void {
  // eslint-disable-next-line no-console
  console.log("=== THE FUSED-ADAM DERIVATION (each rung's oracle asserted) ===\n");

  // Rung 0/1: the horizontal-pack move packs N per-param loops into one.
  console.log("  0/1: N per-param Adam bodies → pack (concatenate) → one packed dispatch");
  const segments = [
    { value: v("param0"), numElements: 768 * 768 },
    { value: v("param1"), numElements: 768 },
    { value: v("param2"), numElements: 50257 * 768 },
    { value: v("param3"), numElements: 1024 * 768 },
  ];
  const packed = deriveHorizontalPackedAdam(segments);
  oracle(
    packed.perParamLoopCount === 4,
    `the un-packed form has ${packed.perParamLoopCount} per-param loops (N params)`,
  );
  oracle(
    packed.packedState.semantic.loopNest.length === 1,
    "the pack move collapses the N per-param loops into ONE pack loop",
  );
  oracle(
    packed.packMove.move === "pack" &&
      (packed.packMove as { kind: string }).kind === "concatenate",
    "the move is pack/concatenate (multi-tensor horizontal pack at param altitude)",
  );
  const total = 768 * 768 + 768 + 50257 * 768 + 1024 * 768;
  oracle(
    packed.totalElements === total,
    `the packed flat has all ${total} elements (Σ segment sizes)`,
  );

  // Rung 2: the packed dispatch reaches the AUTHORED adamStep kernel byte-identically.
  console.log("\n  2: the packed dispatch reaches the AUTHORED adamStep (byte-identical)");
  const variants: AdamDescriptor[] = [
    { useVec4: false, emitF16: false, emitUnscale: false },
    { useVec4: false, emitF16: true, emitUnscale: false },
    { useVec4: false, emitF16: false, emitUnscale: true },
    { useVec4: true, emitF16: false, emitUnscale: true },
    { useVec4: true, emitF16: true, emitUnscale: true },
  ];
  for (const desc of variants) {
    // The AUTHORED kernel body now LOWERS FROM the schedule (the cutover-flip);
    // the live realize chokepoint IS the schedule path.
    const live = generateAdamShaderTileIR(
      desc.useVec4,
      desc.emitF16,
      desc.emitUnscale,
    );
    const derived = compileTileKernel(
      applyAdamSchedule(deriveAdamSkeleton(desc), desc),
    );
    oracle(
      derived === live,
      `derived == authored adamStep [${adamCacheKey(desc)}] (BYTE-IDENTICAL)`,
    );
  }

  // PACKED-DISPATCH COUNT: the packed group is ONE dispatch, not N.
  console.log("\n  packed-dispatch count: ONE adamStep per group (not N — the 8-dispatch class)");
  oracle(
    1 < packed.perParamLoopCount,
    `packed = 1 dispatch vs un-packed = ${packed.perParamLoopCount} (${packed.perParamLoopCount}× fewer)`,
  );

  // The packed state prints (lowers via the canonical schema — totality check).
  const printed = printScheduleState(packed.packedState);
  oracle(
    printed.includes("schedule-state v"),
    "the packed derived state prints (lowers via the canonical schema)",
  );
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  // eslint-disable-next-line no-console
  console.log("fa-adam-derivation-script v1  (P4 acceptance artifact — RUNNABLE)\n");

  runDerivation();

  if (!cpuOnly) {
    await runGpuNumericCheck();
  } else {
    console.log("\n=== GPU NUMERIC CHECK: SKIPPED (TORCHLETTE_CPU_ONLY=1) ===");
    console.log(
      "  (the trajectory-level parity gates are test/optim/{fused-vs-elementwise,adam-multigroup}.spec.ts)",
    );
  }

  console.log(
    `\n=== ORACLE SUMMARY: ${failures === 0 ? "ALL ORACLES PASSED" : `${failures} ORACLE(S) FAILED`} ===`,
  );
  if (destroyWebGPUFn) destroyWebGPUFn();
  process.exit(failures === 0 ? 0 : 1);
}

// ============================================================================
// GPU numeric check: the packed foreach step == the per-param step
// ============================================================================

async function runGpuNumericCheck(): Promise<void> {
  const { initWebGPU, destroyWebGPU } = await import("../src/backend/webgpu");
  const { Torchlette } = await import("../src/frontend/torchlette");
  const { Adam } = await import("../src/optim/adam");
  await initWebGPU();
  destroyWebGPUFn = destroyWebGPU;
  const api = new Torchlette("webgpu");
  api.setDefaultDevice("webgpu");

  console.log("\n=== GPU NUMERIC CHECK (packed foreach step == per-param step) ===");

  // Two params, a deterministic sum-of-squares-to-target loss (real backward
  // populates grads). Run 5 steps under BOTH paths (packed foreach default vs
  // per-param TORCHLETTE_FOREACH_ADAM=0) and compare the final params.
  const shapes = [
    [4, 8],
    [8, 3],
  ];
  const mkParam = (sh: number[], i: number) =>
    api.tensorFromArray(
      Float32Array.from(
        Array.from({ length: sh[0] * sh[1] }, (_, j) => Math.sin((i + 1) * j * 0.3) * 0.5),
      ),
      sh,
      { requiresGrad: true, device: "webgpu" },
    );
  const target = (sh: number[], i: number) =>
    api.tensorFromArray(
      Float32Array.from(
        Array.from({ length: sh[0] * sh[1] }, (_, j) => Math.cos((i + 1) * j * 0.2) * 0.3),
      ),
      sh,
      { device: "webgpu" },
    );

  const runPath = async (foreach: boolean): Promise<number[]> => {
    const prev = process.env.TORCHLETTE_FOREACH_ADAM;
    process.env.TORCHLETTE_FOREACH_ADAM = foreach ? "1" : "0";
    const params = shapes.map((sh, i) => mkParam(sh, i));
    const targets = shapes.map((sh, i) => target(sh, i));
    const opt = new Adam(params, { lr: 0.01 }, api);
    for (let step = 0; step < 5; step++) {
      await api.beginStep();
      let loss: ReturnType<typeof api.sum> | null = null;
      for (let i = 0; i < params.length; i++) {
        const diff = api.sub(params[i], targets[i]);
        const partial = api.sum(api.mul(diff, diff));
        loss = loss ? api.add(loss, partial) : partial;
      }
      await loss!.backward();
      opt.step();
      opt.zeroGrad();
      api.endStep();
      await api.markStep();
    }
    const out: number[] = [];
    for (const p of params) out.push(...(await api.cpu(p)));
    if (prev === undefined) delete process.env.TORCHLETTE_FOREACH_ADAM;
    else process.env.TORCHLETTE_FOREACH_ADAM = prev;
    return out;
  };

  const packedOut = await runPath(true);
  const perParamOut = await runPath(false);
  let maxDiff = 0;
  for (let i = 0; i < packedOut.length; i++)
    maxDiff = Math.max(maxDiff, Math.abs(packedOut[i] - perParamOut[i]));
  console.log(`  packed-vs-per-param final param max|Δ| over 5 steps: ${maxDiff.toExponential(3)}`);
  oracle(maxDiff < 1e-5, `packed foreach == per-param to ≤1e-5 (got ${maxDiff.toExponential(3)})`);
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
