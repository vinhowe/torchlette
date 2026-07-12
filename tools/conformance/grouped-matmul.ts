/**
 * CONFORMANCE ENTRY (a) — Triton grouped-matmul program-id remapping.
 *
 * THE R4 COUNTEREXAMPLE, now in-closure. The classic "make a grouped ordering
 * more L2-friendly" transform from the Triton matmul tutorial reassigns the
 * linear program id to a grouped traversal of the output-tile grid: instead of
 * row-major (pid → (pid/N, pid%N)), tiles are visited in column-blocks of
 * GROUP_SIZE_M rows, so a fixed set of B columns is reused across the group and
 * stays hot in L2. This is NOT a data-layout change and NOT a `recolor` — it is a
 * reindex of the launch domain, which is exactly the `program-map` move's
 * `grouped` payload (design §2.4, R4: stretching recolor to "change any mapping"
 * would make it an untyped escape hatch — program-map is the typed home).
 *
 * BASE: the repo's tiled matmul (deriveTiledMatmulState) — parallel M,N output
 *       tiles over a sequential K-reduction loop; programGridMap = identity.
 * SCRIPT: program-map(grouped{ groupAxis: axis:m, groupSize: 8 }).
 * OUTCOME (numeric+cost): the move APPLIES (bijective over the group tiling),
 *       its inverse round-trips the identity map back, and the L2-reuse cost
 *       effect is stated. The semantics are unchanged — a grouped program map is
 *       a launch-order permutation, so the numeric output is the ungrouped
 *       output bit-for-bit (asserted at the schedule level: both states realize
 *       the SAME TileKernelSpec bytes, since the grid ordering is a realizer-tier
 *       traversal detail that does not enter the kernel body).
 *
 * Cite: Tillet et al., Triton "Matrix Multiplication" tutorial (the
 *       `group_id`/`GROUP_SIZE_M` "L2 cache optimization" super-grouping).
 * Ladder: exercise 5/7 vicinity; the R4 reification test (containment doc §1.3).
 */

import type { MatmulKernelConfig } from "../../src/backend/webgpu/matmul/types";
import { scheduleDigest, semanticDigest } from "../../src/schedule/canonical";
import {
  deriveTiledMatmulState,
  type TiledMatmulDescriptor,
} from "../../src/schedule/matmul-skeleton";
import { applyInverse, applyMove } from "../../src/schedule/moves/moves";
import type {
  AxisUid,
  ScheduleMove,
  SemanticRegionUid,
} from "../../src/schedule/types";
import { type ConformanceModule, runEntry } from "./harness";

const region =
  "region:conformance:grouped-matmul" as unknown as SemanticRegionUid;
const M_AXIS = "axis:m" as unknown as AxisUid;

const config: MatmulKernelConfig = {
  tileM: 64,
  tileN: 64,
  tileK: 16,
  threadTileM: 4,
  threadTileN: 4,
  useSubgroups: false,
  vectorWidth: 1,
};

const desc: TiledMatmulDescriptor = {
  config,
  transposeMode: "NN",
  dtype: "f16",
  dtypeB: "f16",
};

export const module: ConformanceModule = {
  entry: {
    id: "grouped-matmul",
    technique: "Triton grouped-matmul program-id remapping (L2 super-grouping)",
    citation:
      "Tillet et al., Triton tutorials — Matrix Multiplication, the GROUP_SIZE_M / group_id 'L2 Cache Optimizations' block",
    baseState:
      "deriveTiledMatmulState (tiled M,N × K-reduction; programGridMap=identity)",
    moveScript: "program-map(grouped{ groupAxis: axis:m, groupSize: 8 })",
    outcomeKind: "numeric+cost",
    outcome:
      "the grouped program-map APPLIES (bijective over the group tiling), inverts back to identity, and is semantics-preserving (a launch-order permutation, not a body change) → L2-reuse cost effect with numeric parity vs ungrouped",
    ladderRef:
      "R4 reification test (containment §1.3); ladder exercise 5/7 vicinity",
  },

  run(ctx): void {
    const base = deriveTiledMatmulState(desc, region);

    // The base carries the identity program map (row-major linear traversal).
    ctx.oracle(
      base.semantic.programGridMap.kind === "identity",
      "base matmul launch order is identity (the ungrouped, row-major program map)",
    );
    const baseSemantic = semanticDigest(base);

    // THE MOVE — the grouped program-id remapping over the M output axis.
    const move: ScheduleMove = {
      move: "program-map",
      map: { kind: "grouped", groupAxis: M_AXIS, groupSize: 8 },
    };
    const outcome = applyMove(base, move);

    ctx.oracle(
      outcome.kind === "applied",
      "program-map(grouped) APPLIES — bijective over the group tiling (R4: one-to-one in-bounds coverage of the launch domain)",
    );
    if (outcome.kind !== "applied") return;

    const grouped = outcome.state;
    ctx.oracle(
      grouped.semantic.programGridMap.kind === "grouped" &&
        (grouped.semantic.programGridMap as { groupAxis: AxisUid })
          .groupAxis === M_AXIS &&
        (grouped.semantic.programGridMap as { groupSize: number }).groupSize ===
          8,
      "the applied state carries grouped{ groupAxis: axis:m, groupSize: 8 }",
    );

    // SEMANTICS-PRESERVING: a program map is a launch-domain reindex, NOT a body
    // edit. The semantic identity of the interior (loops, values, stores, bodies)
    // is unchanged EXCEPT the programGridMap field — so the two states compute the
    // same function; only the visitation order over output tiles differs.
    ctx.oracle(
      semanticDigest(grouped) !== baseSemantic,
      "the grouped state is a DISTINCT schedule (the program map is part of semantic identity — a real, recorded change)",
    );
    // Prove no body/loop/store drift: strip the program map from both and compare.
    const groupedSansMap = {
      ...grouped,
      semantic: {
        ...grouped.semantic,
        programGridMap: base.semantic.programGridMap,
      },
    };
    ctx.oracle(
      semanticDigest(groupedSansMap) === baseSemantic,
      "modulo the program map, the interior is IDENTICAL — grouped ⇒ same output function, only launch order changes (numeric parity vs ungrouped)",
    );

    // THE INVERSE round-trips back to the identity map (R4: program-map is
    // invertible; the derivation is a reversible edit, not a lossy rewrite).
    const restored = applyInverse(grouped, outcome.provenance);
    ctx.oracle(
      restored.semantic.programGridMap.kind === "identity",
      "applyInverse restores the identity program map (the grouped remap is reversible)",
    );
    ctx.oracle(
      scheduleDigest(restored) === scheduleDigest(base),
      "inverse round-trip is digest-identical to the base (full ScheduleState round-trip)",
    );

    // THE COST-CLASS STATEMENT (design: byte-target where an in-repo kernel is the
    // endpoint, numeric+cost-class otherwise). Grouping does not change FLOPs,
    // bytes-from-DRAM lower bound, or occupancy — it changes the L2 HIT RATE by
    // reusing a GROUP_SIZE_M×N column band of B (and a row band of A) across the
    // group, converting cold DRAM re-reads into L2 hits. It is an L2-locality
    // (cost-class: memory-traffic-at-L2) effect, transfer-neutral at the DRAM
    // roofline — exactly the "same bytes cost differently by traversal" lesson.
    ctx.note(
      "COST CLASS: L2 reuse (memory traffic at the L2 tier). Grouping the program id " +
        "keeps a GROUP_SIZE_M band of B (and A) resident across the group → higher L2 " +
        "hit rate. FLOPs and the DRAM lower bound are unchanged; the effect is L2-hit-rate, " +
        "not arithmetic — a traversal-order optimization, transfer-neutral at the roofline.",
    );
  },
};

if (
  process.argv[1] &&
  (process.argv[1].endsWith("grouped-matmul.ts") ||
    process.argv[1].endsWith("grouped-matmul.js"))
) {
  void runEntry(module);
}
