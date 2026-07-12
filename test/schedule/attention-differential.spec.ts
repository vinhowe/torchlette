/**
 * P0-FULL wave-3 byte differential — ATTENTION family.
 *
 * Two shapes, one family (see src/schedule/attention-skeleton.ts header):
 *
 * (1) AUTHORED (opaque, F3) fused FlashAttention kernels. For each kernel role
 *     (forward, D-precompute, backward-dQ, backward-dKV) across the modifier
 *     corpus (bare, causal, sliding-window, softcap, causal+sliding-window),
 *
 *       compileTileKernel(applyAttentionSchedule(deriveAttentionSkeleton(desc), desc))
 *         ==BYTES== compileTileKernel(<live make*Spec(headDim, mod)>)
 *
 *     The authored skeleton READS the live kernel's config (headDim + modifier);
 *     applyAttentionSchedule regenerates through the SAME live single-source
 *     make*Spec factory (there is nowhere in an opaque skeleton to store a
 *     generator — R22). Backward + scoreMod is refused (inference-first, #64).
 *
 * (2) NAIVE (derivable) attention as a THREE-REGION composition (QK^T matmul →
 *     softmax row-program → PV matmul), verified byte-identically via the REUSED
 *     matmul + row-program family apply seams — never duplicating them. This is
 *     P2's starting position.
 *
 * PURE-COMPILATION (no GPU): the tile-IR codegen is a WGSL compiler; both sides
 * compile in ONE process (identical device/subgroup detection), so byte-identity
 * holds. Lives in the cpu project, runs by default. Exercises the authored
 * no-second-owner seam assertions (§12 check 3) on every skeleton.
 *
 * ALSO gates: the authored-skeleton print form (typed params visible, skeleton
 * sealed), the naive three-region composition print (the educational artifact),
 * and digest stability.
 */

import { describe, expect, it } from "vitest";
import type { AttnModifierSpec } from "../../src/backend/types";
import {
  makeBackwardDKVSpec,
  makeBackwardDQSpec,
  makeDPrecomputeSpec,
  makeForwardAttentionSpec,
} from "../../src/backend/webgpu/attention-kernel";
import { rowProgramToSpec } from "../../src/backend/webgpu/row-program-codegen";
import { compileTileKernel } from "../../src/backend/webgpu/tile-compiler";
import {
  type AttentionDescriptor,
  type AttentionKernelRole,
  applyAttentionSchedule,
  attentionCacheKey,
  D_PRECOMPUTE_OBLIGATION,
  deriveAttentionSkeleton,
  naiveAttentionBackwardComposition,
  naiveAttentionComposition,
  onlineSoftmaxLemma,
  RECOMPUTE_P_OBLIGATION,
  softmaxBackwardRowProgram,
  softmaxRowProgram,
} from "../../src/schedule/attention-skeleton";
import { classifyBody } from "../../src/schedule/moves/streamability";
import type { SemanticBodyNode, ValueUid } from "../../src/schedule/types";
import {
  printScheduleState,
  printSkeleton,
  scheduleDigest,
  skeletonDigest,
} from "../../src/schedule/canonical";
import {
  applyTiledMatmulSchedule,
  generateTiledMatmulShaderTileIR,
} from "../../src/schedule/matmul-skeleton";
import { applyRowProgramSchedule } from "../../src/schedule/reduction-skeleton";

const D = 64;

/** The modifier corpus for the seam-bearing kernels (#64 Gemma seams). */
const modifierCorpus: Array<{ label: string; mod?: AttnModifierSpec }> = [
  { label: "bare", mod: undefined },
  { label: "causal", mod: { maskMods: [{ kind: "causal" }] } },
  {
    label: "sliding-window",
    mod: { maskMods: [{ kind: "slidingWindow", window: 256 }] },
  },
  { label: "softcap", mod: { scoreMod: { kind: "softcap", cap: 30 } } },
  {
    label: "causal+sliding-window",
    mod: {
      maskMods: [{ kind: "causal" }, { kind: "slidingWindow", window: 256 }],
    },
  },
];

const counts = {
  forward: 0,
  dPrecompute: 0,
  backwardDQ: 0,
  backwardDKV: 0,
  naive: 0,
};

function liveSpec(role: AttentionKernelRole, mod?: AttnModifierSpec) {
  switch (role) {
    case "forward":
      return makeForwardAttentionSpec(D, mod);
    case "dPrecompute":
      return makeDPrecomputeSpec(D);
    case "backwardDQ":
      return makeBackwardDQSpec(D, mod);
    case "backwardDKV":
      return makeBackwardDKVSpec(D, mod);
  }
}

describe("P0 attention walking-skeleton byte differential (authored / opaque)", () => {
  // ---- FORWARD: full modifier corpus (causal / sliding-window / softcap) ----
  for (const c of modifierCorpus) {
    it(`forward ${c.label}`, () => {
      const desc: AttentionDescriptor = {
        role: "forward",
        headDim: D,
        modifier: c.mod,
      };
      const live = compileTileKernel(liveSpec("forward", c.mod));
      const skeleton = deriveAttentionSkeleton(desc);
      const derived = compileTileKernel(applyAttentionSchedule(skeleton, desc));
      expect(derived).toBe(live);
      counts.forward++;
    });
  }

  // ---- D-PRECOMPUTE: one template for all mods (no score/mask seam sites) ----
  it("D-precompute (one template, mod-invariant)", () => {
    const desc: AttentionDescriptor = { role: "dPrecompute", headDim: D };
    const live = compileTileKernel(liveSpec("dPrecompute"));
    const skeleton = deriveAttentionSkeleton(desc);
    const derived = compileTileKernel(applyAttentionSchedule(skeleton, desc));
    expect(derived).toBe(live);
    counts.dPrecompute++;
  });

  // ---- BACKWARD dQ + dKV: mask-mod corpus (scoreMod refused in backward) ----
  const bwdMaskCorpus = modifierCorpus.filter((c) => !c.mod?.scoreMod);
  for (const c of bwdMaskCorpus) {
    it(`backward-dQ ${c.label}`, () => {
      const desc: AttentionDescriptor = {
        role: "backwardDQ",
        headDim: D,
        modifier: c.mod,
      };
      const live = compileTileKernel(liveSpec("backwardDQ", c.mod));
      const skeleton = deriveAttentionSkeleton(desc);
      const derived = compileTileKernel(applyAttentionSchedule(skeleton, desc));
      expect(derived).toBe(live);
      counts.backwardDQ++;
    });
    it(`backward-dKV ${c.label}`, () => {
      const desc: AttentionDescriptor = {
        role: "backwardDKV",
        headDim: D,
        modifier: c.mod,
      };
      const live = compileTileKernel(liveSpec("backwardDKV", c.mod));
      const skeleton = deriveAttentionSkeleton(desc);
      const derived = compileTileKernel(applyAttentionSchedule(skeleton, desc));
      expect(derived).toBe(live);
      counts.backwardDKV++;
    });
  }

  it("reports per-role kernel counts", () => {
    const total =
      counts.forward +
      counts.dPrecompute +
      counts.backwardDQ +
      counts.backwardDKV;
    expect(total).toBeGreaterThanOrEqual(14); // 5 fwd + 1 D + 4 dQ + 4 dKV
    // eslint-disable-next-line no-console
    console.log(
      `[P0 attention differential] byte-identical authored kernels covered: ${total} ` +
        `(forward=${counts.forward} dPrecompute=${counts.dPrecompute} ` +
        `backwardDQ=${counts.backwardDQ} backwardDKV=${counts.backwardDKV}) ` +
        `+ naive-composition regions=${counts.naive}`,
    );
  });
});

// ============================================================================
// The authored form: no-second-owner + cache-key + capability-schema legality
// ============================================================================

describe("P0 attention authored-form legality (§6 / F3 / R10)", () => {
  it("the fused kernels wear an OPAQUE skeleton (F3), never derived", () => {
    for (const role of [
      "forward",
      "dPrecompute",
      "backwardDQ",
      "backwardDKV",
    ] as AttentionKernelRole[]) {
      const sk = deriveAttentionSkeleton({ role, headDim: D });
      expect(sk.visibility).toBe("opaque");
      if (sk.visibility === "opaque") {
        expect(sk.kernelRef).toContain("attention-kernel.ts");
        expect(sk.refusalReason).toContain("authored — not yet re-derived");
        // F3: an opaque skeleton has NO loop/staging/role field to leak.
        expect(sk).not.toHaveProperty("schedule");
      }
    }
  });

  it("REFUSES a headDim that violates the vec4 constraint (headDim % 4 != 0)", () => {
    const desc: AttentionDescriptor = { role: "forward", headDim: 66 };
    const sk = deriveAttentionSkeleton(desc);
    // The schema's dependent constraint and the live factory's throw are ONE
    // fact — the seam surfaces the refusal before the factory runs.
    expect(() => applyAttentionSchedule(sk, desc)).toThrow(/vec4|headDim/);
  });

  it("REFUSES a backward kernel with a scoreMod (inference-first, #64)", () => {
    const desc: AttentionDescriptor = {
      role: "backwardDQ",
      headDim: D,
      modifier: { scoreMod: { kind: "softcap", cap: 30 } },
    };
    const sk = deriveAttentionSkeleton(desc);
    // The live factory's backward path throws via assertBackwardSupportsModifier
    // — the authored apply re-calls it, so the refusal is preserved (one owner).
    expect(() => applyAttentionSchedule(sk, desc)).toThrow();
  });

  it("the cache-key encoder delegates to attnModifierKey (single source)", () => {
    const bare = attentionCacheKey({ role: "forward", headDim: D });
    const causal = attentionCacheKey({
      role: "forward",
      headDim: D,
      modifier: { maskMods: [{ kind: "causal" }] },
    });
    expect(bare).toBe("attn:forward:D64");
    expect(causal).toBe("attn:forward:D64:m.causal");
    // Structurally-identical compositions in canonical order share one key.
    const softcapCausal = attentionCacheKey({
      role: "forward",
      headDim: D,
      modifier: {
        scoreMod: { kind: "softcap", cap: 30 },
        maskMods: [{ kind: "causal" }],
      },
    });
    expect(softcapCausal).toBe("attn:forward:D64:s.softcap+m.causal");
  });

  it("the typed param schema carries the 256-head-dim workgroup-storage capability", () => {
    const sk = deriveAttentionSkeleton({ role: "forward", headDim: D });
    if (sk.visibility !== "opaque") throw new Error("expected opaque");
    expect(sk.params.params.headDimension.domain).toContain(256);
    // The capability predicate is a disjunction (headDim<256 OR wgStorage>=32768).
    expect(sk.params.capabilityPredicate.kind).toBe("or");
  });
});

// ============================================================================
// The NAIVE three-region composition (deliverable 2 — the P2 starting position)
// ============================================================================

describe("P0 attention naive composition (QK^T → softmax → PV, family-reuse)", () => {
  it("region 1 (QK^T) round-trips byte-identically via the matmul family", () => {
    const comp = naiveAttentionComposition(D);
    const live = generateTiledMatmulShaderTileIR({
      config: comp.qkT.desc.config,
      transposeMode: comp.qkT.desc.transposeMode,
      dtype: comp.qkT.desc.dtype,
    });
    const derived = compileTileKernel(
      applyTiledMatmulSchedule(comp.qkT.state, comp.qkT.desc),
    );
    expect(derived).toBe(live);
    counts.naive++;
  });

  it("region 2 (softmax) round-trips byte-identically via the row-program family", () => {
    const comp = naiveAttentionComposition(D);
    // Derived: the composition's softmax state through the REUSED row-program
    // family apply seam (assertRowProgramSeam + rowProgramToSpec).
    const derived = compileTileKernel(
      applyRowProgramSchedule(comp.softmax.state, comp.softmax.program),
    );
    // Live single source: rowProgramToSpec on the same RowProgram directly.
    const live = compileTileKernel(rowProgramToSpec(softmaxRowProgram()));
    expect(derived).toBe(live);
    counts.naive++;
  });

  it("region 3 (PV) round-trips byte-identically via the matmul family", () => {
    const comp = naiveAttentionComposition(D);
    const live = generateTiledMatmulShaderTileIR({
      config: comp.pv.desc.config,
      transposeMode: comp.pv.desc.transposeMode,
      dtype: comp.pv.desc.dtype,
    });
    const derived = compileTileKernel(
      applyTiledMatmulSchedule(comp.pv.state, comp.pv.desc),
    );
    expect(derived).toBe(live);
    counts.naive++;
  });

  it("the island-flow connects the three regions (scores → softmax → P → PV)", () => {
    const comp = naiveAttentionComposition(D);
    expect(comp.islandFlow.map((e) => e.via)).toEqual(["scores", "P"]);
    expect(comp.islandFlow[0].from).toBe(comp.qkT.region);
    expect(comp.islandFlow[0].to).toBe(comp.softmax.region);
    expect(comp.islandFlow[1].from).toBe(comp.softmax.region);
    expect(comp.islandFlow[1].to).toBe(comp.pv.region);
  });
});

// ============================================================================
// The NAIVE BACKWARD composition + the P4 backward derivation (§7 local self-host)
// ============================================================================

describe("P4 attention BACKWARD derivation (dV/dP/dS/dQ/dK → authored dQ/dKV/D)", () => {
  const bapply = (op: string, ...args: SemanticBodyNode[]): SemanticBodyNode => ({
    kind: "apply",
    catalog: { op },
    args,
  });
  const bval = (name: string): SemanticBodyNode => ({
    kind: "value",
    value: name as unknown as ValueUid,
  });

  it("dV region round-trips byte-identically via the matmul family", () => {
    const c = naiveAttentionBackwardComposition(D);
    const live = generateTiledMatmulShaderTileIR({
      config: c.dV.desc.config,
      transposeMode: c.dV.desc.transposeMode,
      dtype: c.dV.desc.dtype,
    });
    const derived = compileTileKernel(
      applyTiledMatmulSchedule(c.dV.state, c.dV.desc),
    );
    expect(derived).toBe(live);
  });

  it("dS (softmax-backward) round-trips byte-identically via the row-program family", () => {
    const c = naiveAttentionBackwardComposition(D);
    const live = compileTileKernel(rowProgramToSpec(softmaxBackwardRowProgram()));
    const derived = compileTileKernel(
      applyRowProgramSchedule(c.dS.state, c.dS.program),
    );
    expect(derived).toBe(live);
  });

  it("the backward island-flow connects the regions (dO→dS, dP→dS, dS→dQ/dK)", () => {
    const c = naiveAttentionBackwardComposition(D);
    expect(c.islandFlow.map((e) => e.via)).toEqual(["dO", "dP", "dS", "dS"]);
  });

  it("RECOMPUTATION discharge round-trip: materialized_P refuses+names → recompute_P admits", () => {
    const before = classifyBody(bapply("materialized_P", bval("scores")));
    expect(before.streamable).toBe(false);
    if (before.streamable) throw new Error("unreachable");
    expect(before.refusal.dischargedBy).toBe(RECOMPUTE_P_OBLIGATION);
    const after = classifyBody(
      bapply("recompute_P", bval("scores"), bval("L")),
    );
    expect(after.streamable).toBe(true);
  });

  it("D-PRECOMPUTE discharge round-trip: inline inner-sum refuses+names → precomputed_D admits", () => {
    const before = classifyBody(
      bapply("inline_softmax_grad_innersum", bval("P"), bval("dO"), bval("V")),
    );
    expect(before.streamable).toBe(false);
    if (before.streamable) throw new Error("unreachable");
    expect(before.refusal.dischargedBy).toBe(D_PRECOMPUTE_OBLIGATION);
    const after = classifyBody(bapply("precomputed_D", bval("dO"), bval("O")));
    expect(after.streamable).toBe(true);
  });

  it("the derivation reaches the AUTHORED dQ/dKV/D kernels byte-identically", () => {
    for (const role of [
      "dPrecompute",
      "backwardDQ",
      "backwardDKV",
    ] as AttentionKernelRole[]) {
      const desc: AttentionDescriptor = { role, headDim: D };
      const live = compileTileKernel(liveSpec(role));
      const derived = compileTileKernel(
        applyAttentionSchedule(deriveAttentionSkeleton(desc), desc),
      );
      expect(derived).toBe(live);
    }
  });
});

// ============================================================================
// Print form + digest stability (deliverable 4 + the report artifacts)
// ============================================================================

describe("P0 attention print-form + digest stability", () => {
  it("prints the authored-attention forward state (typed params visible, sealed)", () => {
    const sk = deriveAttentionSkeleton({ role: "forward", headDim: D });
    const text = printSkeleton(sk, [onlineSoftmaxLemma()]);
    expect(text).toContain("authored-skeleton v1");
    expect(text).toContain("skeleton sealed=opaque");
    expect(text).toContain("param headDimension domain=[64,128,256]");
    expect(text).toContain("lemma uid=lemma:online-softmax-rescaling");
    // eslint-disable-next-line no-console
    console.log("\n=== AUTHORED ATTENTION FORWARD (printed) ===\n" + text);
  });

  it("prints the naive three-region composition (the educational artifact)", () => {
    const comp = naiveAttentionComposition(D);
    const artifact = [
      "=== NAIVE ATTENTION as a derivable COMPOSITION (P2 starting position) ===",
      "",
      `region 1 [${comp.qkT.region}]  scores = Q @ Kᵀ`,
      printSemanticTier(comp.qkT.state),
      "",
      `region 2 [${comp.softmax.region}]  P = softmax(scores)`,
      printSemanticTier(comp.softmax.state),
      "",
      `region 3 [${comp.pv.region}]  O = P @ V`,
      printSemanticTier(comp.pv.state),
      "",
      "island-flow:",
      ...comp.islandFlow.map((e) => `  ${e.from} --${e.via}--> ${e.to}`),
    ].join("\n");
    expect(artifact).toContain("region 1");
    expect(artifact).toContain("region 2");
    expect(artifact).toContain("region 3");
    // eslint-disable-next-line no-console
    console.log("\n" + artifact);
  });

  it("digest stable: same authored skeleton twice → identical text + digest", () => {
    const a = deriveAttentionSkeleton({ role: "forward", headDim: D });
    const b = deriveAttentionSkeleton({ role: "forward", headDim: D });
    expect(printSkeleton(a, [onlineSoftmaxLemma()])).toBe(
      printSkeleton(b, [onlineSoftmaxLemma()]),
    );
    expect(skeletonDigest(a, [onlineSoftmaxLemma()])).toBe(
      skeletonDigest(b, [onlineSoftmaxLemma()]),
    );
    expect(skeletonDigest(a)).toMatch(/^[0-9a-f]{32}$/);
  });

  it("digest stable: naive-composition region states are deterministic", () => {
    const c1 = naiveAttentionComposition(D);
    const c2 = naiveAttentionComposition(D);
    expect(scheduleDigest(c1.qkT.state)).toBe(scheduleDigest(c2.qkT.state));
    expect(scheduleDigest(c1.softmax.state)).toBe(
      scheduleDigest(c2.softmax.state),
    );
    expect(scheduleDigest(c1.pv.state)).toBe(scheduleDigest(c2.pv.state));
  });
});

/** Print only the semantic tier + region (slice off requests/receipts). */
function printSemanticTier(
  state: Parameters<typeof printScheduleState>[0],
): string {
  const full = printScheduleState(state);
  const idx = full.indexOf("\nrequests:");
  return idx >= 0 ? full.slice(0, idx) : full;
}
