/**
 * CONFORMANCE ENTRY (e) — Coalesced transpose via shared-memory staging + padding.
 *
 * The ladder's exercise 4 endpoint. A naive strided transpose reads OR writes
 * global memory uncoalesced (one axis is contiguous, the other strided — the same
 * bytes cost more by access shape). The classic fix stages the tile through shared
 * memory: read coalesced along one axis into a padded shared tile, then write
 * coalesced along the other — both global accesses are now contiguous, and the
 * +1 column pad breaks the shared-bank conflict. This is the recolor / staging
 * vocabulary: a `recolor` of a global-resident value onto the SHARED staging tier
 * (the tier-crossing that changes an access shape without changing the bytes).
 *
 * BASE: a matmul base whose A operand `in:a` is global-resident (the strided
 *       read side) with a shared A-tile `stage:a_tile` already present.
 * SCRIPT: recolor(in:a → shared, transitionRole = materialization-boundary).
 * OUTCOME (numeric+cost): recolor APPLIES; the value's allocation moves global →
 *       shared (the staging tier); the interior function is unchanged (staging is
 *       a residency decoration, so numeric parity vs the strided path holds); the
 *       inverse restores global. Cost class: uncoalesced global access → coalesced
 *       via shared staging (a bank-conflict-free padded tile) — an access-shape /
 *       coalescing effect, transfer-neutral in total bytes.
 *
 * Cite: Ruetsch & Micikevicius, "Optimizing Matrix Transpose in CUDA" (NVIDIA
 *       2009) — the shared-memory tile + `[TILE][TILE+1]` padding transpose. The
 *       repo's own layout vocabulary (detectSimpleTranspose / staged tiles) is the
 *       WGSL-side instance.
 * Ladder: exercise 4 (rung 2 layout/coalescing — the first layout decoration).
 */

import type { MatmulKernelConfig } from "../../src/backend/webgpu/matmul/types";
import { scheduleDigest } from "../../src/schedule/canonical";
import {
  deriveTiledMatmulState,
  type TiledMatmulDescriptor,
} from "../../src/schedule/matmul-skeleton";
import { applyInverse, applyMove } from "../../src/schedule/moves/moves";
import type {
  ScheduleMove,
  SemanticRegionUid,
  ValueUid,
} from "../../src/schedule/types";
import { type ConformanceModule, runEntry } from "./harness";

const region =
  "region:conformance:coalesced-transpose" as unknown as SemanticRegionUid;
const A_VALUE = "in:a" as unknown as ValueUid;

const config: MatmulKernelConfig = {
  tileM: 32,
  tileN: 32,
  tileK: 16,
  threadTileM: 4,
  threadTileN: 4,
  useSubgroups: false,
  vectorWidth: 1,
};

// NT transpose mode — the A operand is read along a strided axis (the transpose
// side the shared-staging fix targets).
const desc: TiledMatmulDescriptor = {
  config,
  transposeMode: "NT",
  dtype: "f32",
};

export const module: ConformanceModule = {
  entry: {
    id: "coalesced-transpose",
    technique: "Coalesced transpose via shared-memory staging + padding",
    citation:
      "Ruetsch & Micikevicius, 'Optimizing Matrix Transpose in CUDA' (NVIDIA 2009) — shared tile + [TILE][TILE+1] pad; repo instance = staged tiles / detectSimpleTranspose",
    baseState:
      "deriveTiledMatmulState (A operand in:a global-resident; shared stage:a_tile present)",
    moveScript:
      "recolor(in:a → shared, transitionRole=materialization-boundary)",
    outcomeKind: "numeric+cost",
    outcome:
      "recolor APPLIES (in:a global → shared staging tier); interior function unchanged (staging is a residency decoration → numeric parity vs the strided path); inverse restores global; cost class: uncoalesced global access → coalesced via a bank-conflict-free padded shared tile",
    ladderRef:
      "exercise 4 (rung 2 layout/coalescing — the first layout decoration)",
  },

  run(ctx): void {
    const base = deriveTiledMatmulState(desc, region);

    // The A operand starts global-resident (the strided read side); a shared
    // A-tile already exists (the staging destination the fix reads coalesced into).
    const aBefore = base.semantic.values.find((v) => v.uid === A_VALUE);
    ctx.oracle(
      aBefore !== undefined && aBefore.allocation === "global",
      "base A operand in:a is global-resident (the uncoalesced strided-transpose read side)",
    );
    ctx.oracle(
      base.semantic.values.some((v) => v.allocation === "shared"),
      "base has a shared staging tier present (stage:a_tile — the coalescing destination)",
    );

    // THE MOVE — recolor the A operand onto the SHARED staging tier: read it
    // coalesced into shared, so the downstream (transposed) access is coalesced.
    const move: ScheduleMove = {
      move: "recolor",
      value: A_VALUE,
      column: 1,
      tier: "shared",
      transitionRole: "materialization-boundary",
    };
    const outcome = applyMove(base, move);
    ctx.oracle(
      outcome.kind === "applied",
      "recolor(in:a → shared) APPLIES — staging the strided operand through shared memory",
    );
    if (outcome.kind !== "applied") return;

    const staged = outcome.state;
    const aAfter = staged.semantic.values.find((v) => v.uid === A_VALUE);
    ctx.oracle(
      aAfter !== undefined && aAfter.allocation === "shared",
      "the A operand is now shared-resident (the tier crossing that coalesces both global accesses)",
    );

    // SEMANTICS-PRESERVING: staging is a RESIDENCY decoration — where a value
    // lives, not what it computes. The bodies/loops/stores are untouched, so the
    // staged kernel computes the same function as the strided path (numeric parity).
    const stagedSansAlloc = {
      ...staged,
      semantic: {
        ...staged.semantic,
        values: staged.semantic.values.map((v) =>
          v.uid === A_VALUE ? { ...v, allocation: "global" as const } : v,
        ),
      },
    };
    ctx.oracle(
      scheduleDigest(stagedSansAlloc) === scheduleDigest(base),
      "modulo the residency of in:a, the schedule is IDENTICAL — staging changes access shape, not the computed function (numeric parity vs strided)",
    );

    // THE INVERSE — restore global residency (recolor is reversible).
    const restored = applyInverse(staged, outcome.provenance);
    ctx.oracle(
      scheduleDigest(restored) === scheduleDigest(base),
      "applyInverse restores global residency (digest-identical round-trip)",
    );

    // THE COST CLASS — access shape, not byte count. Staging the strided operand
    // through a padded shared tile converts an uncoalesced global access into two
    // coalesced ones; the +1 column pad breaks the shared-bank conflict on the
    // transposed read. Total global bytes are unchanged — the effect is coalescing
    // (memory efficiency at the DRAM burst) + bank-conflict-freedom at shared.
    ctx.note(
      "COST CLASS: access shape / coalescing (rung 2). Staging the strided operand through a " +
        "padded shared tile ([TILE][TILE+1]) makes both global accesses coalesced and the shared " +
        "read bank-conflict-free. The SAME bytes move — only the access pattern changes; this is the " +
        "'same bytes cost differently by access shape' lesson, not a traffic reduction.",
    );
  },
};

if (
  process.argv[1] &&
  (process.argv[1].endsWith("coalesced-transpose.ts") ||
    process.argv[1].endsWith("coalesced-transpose.js"))
) {
  void runEntry(module);
}
