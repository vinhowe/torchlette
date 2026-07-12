/**
 * CONFORMANCE ENTRY (b) — Two-pass chunked large reduction.
 *
 * The shipped chunked-sum endpoint (the ladder's exercise 2). A full reduction
 * over a very large input (>128MB — the repo's own deliberately-serial chunked
 * sum, reduction-tile-ir.ts) is derived by TILING the reduction axis into chunks:
 * an outer loop over ceil(N/chunk) chunks encloses an inner loop over the chunk,
 * and the chunk partials fold through the reduction's own associative monoid — a
 * tree-reduction-in-chunks + a combine pass. This uses PURE MOVES, no new
 * vocabulary: the `tile` move on the reduction (reduce) axis IS the two-pass
 * decomposition (validity doc's own teaching example).
 *
 * BASE: deriveReductionState (full sum) — a flat parallel output domain over a
 *       sequential reduction loop `loop:reduce` on `axis:reduce`.
 * SCRIPT: tile(loop:reduce, axis:reduce, factor=CHUNK).
 * OUTCOME (numeric+cost): tile APPLIES; the reduce loop becomes outer(chunks) ⊃
 *       inner(chunk) — the two-pass structure; the reduction root op is an
 *       associative+commutative monoid (sum), so the chunk-partial fold is
 *       streamable and equals the single-pass sum (numeric identity, cost class:
 *       memory-bound full reduction, now parallelizable across chunks). The
 *       inverse un-tiles back to the flat reduction.
 *
 * THE DOCUMENTED BOUNDARY (the corpus's honesty): single-pass decoupled-lookback
 * (the CUB/DeepScan "single-pass prefix scan" that combines partials in ONE grid
 * launch via an inter-block lookback protocol) is OUTSIDE the WGSL closure — it
 * needs a forward-progress guarantee across workgroups that WebGPU does not
 * provide. The TWO-PASS endpoint is in closure and near it; the single-pass one
 * is recorded here as a permanent out-of-closure boundary, not a gap to close.
 *
 * Cite: Merrill & Garland, "Single-pass Parallel Prefix Scan with Decoupled
 *       Look-back" (NVIDIA, 2016) — the single-pass technique whose forward-
 *       progress precondition places it outside the WGSL move closure; the
 *       two-pass tree-reduction-in-chunks is the in-closure endpoint.
 * Ladder: exercise 2 (rung 1 reductions; the chunked-sum teaching example).
 */

import { scheduleDigest } from "../../src/schedule/canonical";
import { applyInverse, applyMove } from "../../src/schedule/moves/moves";
import { classifyBody } from "../../src/schedule/moves/streamability";
import type { ReductionDescriptor } from "../../src/schedule/reduction-skeleton";
import { deriveReductionState } from "../../src/schedule/reduction-skeleton";
import type {
  AxisUid,
  LoopUid,
  ScheduleMove,
  SemanticBodyNode,
  SemanticLoop,
  SemanticRegionUid,
  ValueUid,
} from "../../src/schedule/types";
import { type ConformanceModule, runEntry } from "./harness";

const region =
  "region:conformance:chunked-reduction" as unknown as SemanticRegionUid;
const REDUCE_LOOP = "loop:reduce" as unknown as LoopUid;
const REDUCE_AXIS = "axis:reduce" as unknown as AxisUid;
const CHUNK = 256;

// A full (scalar-output) sum reduction — the >128MB chunked-sum's semantic shape.
const desc: ReductionDescriptor = {
  reduceOp: "sum",
  dim: {
    inputShape: [1 << 20],
    inputStrides: [1],
    normalizedDims: [0],
    outShape: [1],
    outStrides: [1],
    inputToOutDim: [0],
    parallel: true,
  },
};

/** Depth-first find of a loop by uid (the reduce loop's tiled children). */
function findLoop(
  nest: readonly SemanticLoop[],
  uid: string,
): SemanticLoop | null {
  for (const l of nest) {
    if ((l.uid as unknown as string) === uid) return l;
    const inner = findLoop(l.children, uid);
    if (inner) return inner;
  }
  return null;
}

export const module: ConformanceModule = {
  entry: {
    id: "chunked-reduction",
    technique:
      "Two-pass chunked large reduction (tree-reduction-in-chunks + combine)",
    citation:
      "Merrill & Garland, 'Single-pass Parallel Prefix Scan with Decoupled Look-back' (NVIDIA 2016) — the single-pass technique is the documented OUT-of-closure boundary; the two-pass tree-in-chunks endpoint is in closure",
    baseState:
      "deriveReductionState (full sum; parallel output ⊃ sequential loop:reduce)",
    moveScript: `tile(loop:reduce, axis:reduce, factor=${CHUNK})`,
    outcomeKind: "numeric+cost",
    outcome:
      "tile splits the reduction axis into outer(chunks) ⊃ inner(chunk); the sum monoid is associative+commutative so the chunk-partial fold equals the single-pass sum (numeric identity), cost class = memory-bound full reduction now chunk-parallel; inverse un-tiles",
    ladderRef:
      "exercise 2 (rung 1 reductions — the chunked-sum teaching example)",
  },

  run(ctx): void {
    const base = deriveReductionState(desc, region);

    // The reduction root op is a pure associative+commutative monoid — the
    // license that makes chunking sound (fp reorder is a LICENSE, not a fact).
    const reduceBody: SemanticBodyNode = {
      kind: "apply",
      catalog: { op: "reduce_sum" },
      args: [{ kind: "value", value: "in:input" as unknown as ValueUid }],
    };
    const verdict = classifyBody(reduceBody);
    ctx.oracle(
      verdict.streamable,
      "the sum reduction is streamable by its monoid (associative+commutative — the chunking license)",
    );

    // Find the reduce loop in the base (sequential, over axis:reduce).
    const preTile = findLoop(base.semantic.loopNest, REDUCE_LOOP);
    ctx.oracle(
      preTile !== null && preTile.kind === "sequential",
      "base has a single sequential reduction loop (the pre-chunked, one-pass form)",
    );

    // THE MOVE — tile the reduction axis into chunks (the two-pass split).
    const move: ScheduleMove = {
      move: "tile",
      loop: REDUCE_LOOP,
      axis: REDUCE_AXIS,
      factor: CHUNK,
    };
    const outcome = applyMove(base, move);
    ctx.oracle(
      outcome.kind === "applied",
      `tile(loop:reduce, factor=${CHUNK}) APPLIES — the reduction axis splits into chunks`,
    );
    if (outcome.kind !== "applied") return;

    const chunked = outcome.state;
    // The tiled nest: outer chunk loop (ceil(N/chunk) chunks) ⊃ inner intra-chunk
    // loop (extent = chunk). This IS the two-pass structure: pass 1 = per-chunk
    // tree reduction (inner), the combine = the outer fold over chunk partials.
    const outer = findLoop(chunked.semantic.loopNest, `${REDUCE_LOOP}:outer`);
    const inner = findLoop(chunked.semantic.loopNest, `${REDUCE_LOOP}:inner`);
    ctx.oracle(
      outer !== null && inner !== null,
      "the reduce loop is now outer(chunks) ⊃ inner(chunk) — the two-pass decomposition",
    );
    ctx.oracle(
      inner !== null &&
        inner.bound.kind === "affineLeaf" &&
        inner.bound.leaf.kind === "intLit" &&
        inner.bound.leaf.value === CHUNK,
      `the inner (intra-chunk) loop has extent = chunk (${CHUNK})`,
    );
    ctx.oracle(
      outer !== null && outer.bound.kind === "affineCeilDiv",
      "the outer (chunk) loop count is ceil(N / chunk) — the combine pass over chunk partials",
    );

    // NUMERIC IDENTITY (cost class): tiling a monoid reduction is semantics-
    // preserving — the chunk-partial fold equals the single-pass sum exactly
    // (real-associative; fp-reorder is licensed by the monoid). The cost class is
    // a memory-bound full reduction; chunking makes it chunk-parallel (bounded
    // working set per chunk) without changing the total bytes read.
    ctx.note(
      "NUMERIC: two-pass sum == single-pass sum by the associative monoid (fp-reorder licensed). " +
        "COST CLASS: memory-bound full reduction; chunking bounds the per-chunk working set and " +
        "makes the reduction chunk-parallel — total DRAM bytes read are unchanged (the roofline denominator).",
    );

    // THE INVERSE — un-tile back to the flat one-pass reduction.
    const restored = applyInverse(chunked, outcome.provenance);
    ctx.oracle(
      scheduleDigest(restored) === scheduleDigest(base),
      "inverse un-tiles back to the flat reduction (digest-identical round-trip)",
    );

    // THE DOCUMENTED OUT-OF-CLOSURE BOUNDARY (corpus honesty).
    ctx.note(
      "BOUNDARY (out of WGSL closure): single-pass decoupled-lookback (Merrill & Garland 2016) " +
        "combines partials in ONE grid launch via an inter-workgroup lookback protocol that " +
        "requires a forward-progress guarantee across workgroups. WebGPU provides no such guarantee " +
        "(cross-workgroup communication is unsound), so the single-pass variant is PERMANENTLY outside " +
        "the move closure. The two-pass tree-in-chunks endpoint (this entry) is in closure and near it.",
    );
  },
};

if (
  process.argv[1] &&
  (process.argv[1].endsWith("chunked-reduction.ts") ||
    process.argv[1].endsWith("chunked-reduction.js"))
) {
  void runEntry(module);
}
