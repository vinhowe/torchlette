/**
 * CONFORMANCE ENTRY (f, stretch) — kSplit tall-skinny ⊥ epilogue: the TYPED REFUSAL.
 *
 * The corpus also checks NEGATIVE knowledge. The ladder's exercise 7 (tall-skinny
 * matmul + epilogue) reaches, as SOTA-in-closure, kSplit (the fp-reorder lemma
 * splitting K across workers) + epilogue fusion — AND their real incompatibility.
 * A K-split kernel writes RAW f32 partials that a later pass sums; an epilogue
 * (bias/activation/cast) must run ONCE on the summed output, never per split. So a
 * schedule carrying BOTH an epilogue chain AND the kSplit lemma is ILLEGAL. This
 * entry ASSERTS that the grammar REFUSES the illegal composition via a TYPED
 * legality rule on the object (assertTiledSeam's epilogue ⊥ kSplit rule) — the
 * design's "the epilogue⊥kSplit incompatibility becomes a typed legality rule,
 * not a scattered conditional." A corpus that only proved things were reachable
 * would be half a corpus; this entry demonstrates the corpus checks refusals too.
 *
 * BASE: a tall-skinny-shaped tiled matmul descriptor carrying BOTH a bias
 *       epilogue AND kSplit ≥ 2.
 * SCRIPT: applyTiledMatmulSchedule (which runs assertTiledSeam) on that state.
 * OUTCOME (typed-refusal): the seam THROWS with the epilogue ⊥ kSplit legality
 *       message; a kSplit-only state and an epilogue-only state each realize
 *       CLEANLY (the refusal is specific to the illegal COMBINATION, not to either
 *       feature). The negative fact is asserted, not merely narrated.
 *
 * Cite: Osama et al., "Stream-K" (PPoPP 2023) — K-split across workers with a
 *       turnstile combine; the epilogue-once-on-summed-output constraint is the
 *       standard split-K epilogue rule (CUTLASS split-K serial/parallel reduction).
 * Ladder: exercise 7 (rung 7 grid structure — implementation-induced non-commutation).
 */

import type {
  EpilogueConfig,
  MatmulKernelConfig,
} from "../../src/backend/webgpu/matmul/types";
import {
  applyTiledMatmulSchedule,
  deriveTiledMatmulState,
  type TiledMatmulDescriptor,
} from "../../src/schedule/matmul-skeleton";
import type { SemanticRegionUid } from "../../src/schedule/types";
import { type ConformanceModule, runEntry } from "./harness";

const region =
  "region:conformance:ksplit-epilogue" as unknown as SemanticRegionUid;

const config: MatmulKernelConfig = {
  tileM: 32,
  tileN: 32,
  tileK: 16,
  threadTileM: 4,
  threadTileN: 4,
  useSubgroups: false,
  vectorWidth: 1,
};

// A bias epilogue (one additional input, f32 output).
const biasEpilogue: EpilogueConfig = {
  ops: [{ kind: "bias", inputIndex: 0 }],
  additionalInputCount: 1,
  outputDtype: "f32",
};

/** The ILLEGAL combination: epilogue chain AND kSplit ≥ 2. */
const bothDesc: TiledMatmulDescriptor = {
  config,
  transposeMode: "NN",
  dtype: "f32",
  epilogue: biasEpilogue,
  kSplit: 4,
};

/** kSplit only (legal). */
const kSplitOnlyDesc: TiledMatmulDescriptor = {
  config,
  transposeMode: "NN",
  dtype: "f32",
  kSplit: 4,
};

/** Epilogue only (legal). */
const epilogueOnlyDesc: TiledMatmulDescriptor = {
  config,
  transposeMode: "NN",
  dtype: "f32",
  epilogue: biasEpilogue,
};

/** Run a seam and report whether it threw (and the message). */
function seamThrows(desc: TiledMatmulDescriptor): {
  threw: boolean;
  message: string;
} {
  try {
    const state = deriveTiledMatmulState(desc, region);
    applyTiledMatmulSchedule(state, desc);
    return { threw: false, message: "" };
  } catch (err) {
    return {
      threw: true,
      message: err instanceof Error ? err.message : String(err),
    };
  }
}

export const module: ConformanceModule = {
  entry: {
    id: "ksplit-epilogue-refusal",
    technique:
      "kSplit ⊥ epilogue — the typed legality REFUSAL (negative knowledge)",
    citation:
      "Osama et al. 'Stream-K' (PPoPP 2023); CUTLASS split-K reduction — epilogue must run once on the summed output, never per split-partial",
    baseState:
      "tiled matmul carrying BOTH a bias epilogue AND kSplit=4 (the illegal combination)",
    moveScript:
      "applyTiledMatmulSchedule → assertTiledSeam (the epilogue ⊥ kSplit typed legality rule)",
    outcomeKind: "typed-refusal",
    outcome:
      "the seam THROWS the epilogue ⊥ kSplit legality error on the illegal combination; kSplit-only and epilogue-only each realize cleanly (the refusal is specific to the combination, demonstrating the corpus checks NEGATIVE knowledge)",
    ladderRef: "exercise 7 (rung 7 — implementation-induced non-commutation)",
  },

  run(ctx): void {
    // The typed legality rule throws under scheduleStrict() (STRICT_LIFETIME≠0,
    // the default). Pin strict ON so the refusal is asserted deterministically,
    // regardless of ambient env; restore afterward.
    const prev = process.env.TORCHLETTE_STRICT_LIFETIME;
    process.env.TORCHLETTE_STRICT_LIFETIME = "1";
    try {
      // THE REFUSAL — the illegal combination is refused by the typed rule.
      const both = seamThrows(bothDesc);
      ctx.oracle(
        both.threw,
        "the epilogue+kSplit combination is REFUSED — the seam throws (a typed legality rule, not a silent wrong result)",
      );
      ctx.oracle(
        both.threw &&
          both.message.includes("epilogue") &&
          both.message.includes("kSplit"),
        "the refusal message names the epilogue ⊥ kSplit rule (raw f32 partials ⇒ epilogue cannot run per-split)",
      );

      // SPECIFICITY — each feature ALONE is legal; only the COMBINATION is refused.
      const kOnly = seamThrows(kSplitOnlyDesc);
      ctx.oracle(
        !kOnly.threw,
        "kSplit-only realizes CLEANLY (kSplit alone is a legal, in-closure move — the fp-reorder lemma)",
      );
      const eOnly = seamThrows(epilogueOnlyDesc);
      ctx.oracle(
        !eOnly.threw,
        "epilogue-only realizes CLEANLY (epilogue fusion alone is a legal, in-closure structure)",
      );

      ctx.note(
        "NEGATIVE KNOWLEDGE: the corpus asserts a REFUSAL, not a reachability. The epilogue ⊥ kSplit " +
          "edge is implementation-induced non-commutation (a K-split kernel writes raw f32 partials; the " +
          "epilogue must run once on the summed output). The grammar encodes it as ONE typed legality rule " +
          "on the object — the design's improvement over CUTLASS's Pingpong×StreamK incompatibility that " +
          "leaks across scattered template axes.",
      );
    } finally {
      if (prev === undefined) delete process.env.TORCHLETTE_STRICT_LIFETIME;
      else process.env.TORCHLETTE_STRICT_LIFETIME = prev;
    }
  },
};

if (
  process.argv[1] &&
  (process.argv[1].endsWith("ksplit-epilogue-refusal.ts") ||
    process.argv[1].endsWith("ksplit-epilogue-refusal.js"))
) {
  void runEntry(module);
}
