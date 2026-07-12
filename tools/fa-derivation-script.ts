/**
 * THE P2 ACCEPTANCE ARTIFACT — the flashattention derivation as a MOVE-SCRIPT.
 *
 * This is the SPEC the P2 implementation must satisfy (docs/p2-moves-design.md §3;
 * schedule-state-design.md §7 P2 / §8 gate 3). It is checked in NOW as the
 * acceptance artifact; it does NOT RUN the derivation yet — the move-algebra
 * bodies (`stream`, the `fuseGesture` composite transaction) are P2
 * implementation. What it DOES do today, executably:
 *
 *   1. builds the naive three-region composition (the P2 STARTING POSITION,
 *      wave 3's `naiveAttentionComposition`) and prints its base digest;
 *   2. encodes the ordered FA derivation as typed `ScheduleMove`s over the
 *      intra-schedule move grammar (types.ts §13), plus the TWO fuse gestures
 *      (S3 composite transactions — NOT ScheduleState moves, so they are
 *      annotated separately, not in the intra-schedule move list);
 *   3. annotates EACH step's expected legality outcome (succeeds / refuses-first
 *      / discharges-an-obligation), so the implementation has a per-step oracle;
 *   4. prints the whole thing as a canonical, human-readable acceptance script.
 *
 * When P2 lands the move bodies, this same script becomes RUNNABLE: applying the
 * intra-schedule moves to the merged interior schedule reproduces the authored
 * fused-attention state, and the annotated legality outcomes are the assertions.
 *
 *   TORCHLETTE_CPU_ONLY=1 pnpm exec tsx tools/fa-derivation-script.ts
 *
 * The rungs (schedule-state-design.md §7 P2, F17 refusal-first sequence):
 *   0  base = naive three islands (QK^T matmul → softmax row-program → PV matmul)
 *   1  fuse ×2   merge the three naive islands   [S3 composite transaction]
 *   2  tile      the KV loop
 *   3  stream    K/V through shared              [stream move]
 *   4  recolor   accumulator → register
 *   5  stream(softmax) REFUSED — no head/body    [streamability refusal-first, F17]
 *   6  lemma     apply the online-softmax admitted lemma (discharges the obligation)
 *   7  stream    the now-decomposed softmax      [stream move, post-lemma — admitted]
 *   8  program-map  grouped traversal for L2 reuse [R4 reification test]
 */

import {
  naiveAttentionComposition,
  ONLINE_SOFTMAX_OBLIGATION,
} from "../src/schedule/attention-skeleton";
import { printMove, scheduleDigest } from "../src/schedule/canonical";
import type {
  AxisUid,
  LoopUid,
  ScheduleMove,
  ValueUid,
} from "../src/schedule/types";

const loop = (s: string): LoopUid => s as unknown as LoopUid;
const axis = (s: string): AxisUid => s as unknown as AxisUid;
const val = (s: string): ValueUid => s as unknown as ValueUid;

/** The expected legality outcome the P2 implementation must produce per step. */
type ExpectedOutcome =
  | { kind: "succeeds" }
  | { kind: "refuses-first"; namesObligation: string }
  | { kind: "discharges-obligation"; obligation: string }
  | { kind: "composite-transaction"; note: string };

/** One rung of the derivation: a step + its move (or fuse gesture) + oracle. */
interface Rung {
  readonly rung: number;
  readonly label: string;
  /** An intra-schedule move (types.ts §13), or null for a fuse gesture / lemma. */
  readonly move: ScheduleMove | null;
  /** A non-move step's textual form (fuse gesture, lemma application). */
  readonly compositeForm?: string;
  readonly expected: ExpectedOutcome;
}

// ============================================================================
// The FA derivation, rung by rung
// ============================================================================

/**
 * The ordered derivation. The UIDs here are the P2 implementation's targets —
 * the merged interior schedule's KV loop, the K/V/accumulator/softmax values.
 * They are the NAMES the move bodies will resolve; today they anchor the script.
 */
function faDerivation(): readonly Rung[] {
  const kvLoop = loop("loop:attn:kv");
  const kvAxis = axis("axis:attn:kv");
  return [
    {
      rung: 1,
      label: "fuse ×2 — merge the three naive islands into one",
      move: null,
      compositeForm:
        "fuseGesture(P, qkT, softmax) ; fuseGesture(P', qkT+softmax, pv)  " +
        "[S3 composite transaction — islands merge is binary, so 3→1 is two gestures; " +
        "islandFlow scores→softmax→P is the convexity witness]",
      expected: {
        kind: "composite-transaction",
        note: "both merges legal (linear-chain convex union); mints region' twice",
      },
    },
    {
      rung: 2,
      label: "tile the KV loop into outer×inner blocks",
      move: { move: "tile", loop: kvLoop, axis: kvAxis, factor: 32 /* BC */ },
      expected: { kind: "succeeds" },
    },
    {
      rung: 3,
      label: "stream K and V through shared memory",
      move: { move: "stream", value: val("value:attn:kTile"), loop: kvLoop },
      expected: { kind: "succeeds" },
    },
    {
      rung: 4,
      label: "recolor the output accumulator to register residency",
      move: {
        move: "recolor",
        value: val("value:attn:acc"),
        column: 1,
        tier: "register",
        transitionRole: "materialization-boundary",
      },
      expected: { kind: "succeeds" },
    },
    {
      rung: 5,
      label:
        "stream(softmax) — REFUSED (naive softmax has no head/body decomposition)",
      move: { move: "stream", value: val("value:attn:P"), loop: kvLoop },
      expected: {
        kind: "refuses-first",
        namesObligation: ONLINE_SOFTMAX_OBLIGATION,
      },
    },
    {
      rung: 6,
      label:
        "apply the online-softmax admitted lemma (discharges the obligation)",
      move: null,
      compositeForm:
        "applyLemma(onlineSoftmaxLemma) at region:attn — rewrites the softmax box " +
        "div(exp(x-m), sum(exp(x-m))) → online_softmax((m,l,o), correction exp(m_old-m_new))",
      expected: {
        kind: "discharges-obligation",
        obligation: ONLINE_SOFTMAX_OBLIGATION,
      },
    },
    {
      rung: 7,
      label: "stream the now-decomposed softmax (ADMITTED post-lemma)",
      move: { move: "stream", value: val("value:attn:P"), loop: kvLoop },
      expected: { kind: "succeeds" },
    },
    {
      rung: 8,
      label:
        "program-map — grouped traversal for L2 reuse (R4 reification test)",
      move: {
        move: "program-map",
        map: { kind: "grouped", groupAxis: axis("axis:attn:q"), groupSize: 8 },
      },
      expected: { kind: "succeeds" },
    },
  ];
}

// ============================================================================
// Print
// ============================================================================

function printExpected(e: ExpectedOutcome): string {
  switch (e.kind) {
    case "succeeds":
      return "EXPECT succeeds";
    case "refuses-first":
      return `EXPECT refuses-first (names ${e.namesObligation})`;
    case "discharges-obligation":
      return `EXPECT discharges ${e.obligation}`;
    case "composite-transaction":
      return `EXPECT committed composite [${e.note}]`;
  }
}

function main(): void {
  // Rung 0: the base — the naive three-region composition. Digest-anchor it.
  const naive = naiveAttentionComposition(64);
  const qkTDigest = scheduleDigest(naive.qkT.state);
  const softmaxDigest = scheduleDigest(naive.softmax.state);
  const pvDigest = scheduleDigest(naive.pv.state);

  const lines: string[] = [
    "fa-derivation-script v1  (P2 acceptance artifact — SPEC, not yet runnable)",
    "",
    "base: naive three-region attention composition (P2 starting position)",
    `  region qkT     digest=${qkTDigest}`,
    `  region softmax digest=${softmaxDigest}`,
    `  region pv      digest=${pvDigest}`,
    `  islandFlow: ${naive.islandFlow
      .map((e) => `${e.from} -[${e.via}]-> ${e.to}`)
      .join(" ; ")}`,
    "",
    "moves (each annotated with the expected legality outcome the P2 impl must produce):",
  ];

  for (const r of faDerivation()) {
    const form = r.move
      ? printMove(r.move)
      : (r.compositeForm ?? "(non-move step)");
    lines.push(`  ${r.rung}: ${r.label}`);
    lines.push(`       ${form}`);
    lines.push(`       ${printExpected(r.expected)}`);
  }

  lines.push("");
  lines.push(
    "target: the authored fused-attention schedule (skeletonDigest of makeForwardAttentionSpec) —",
  );
  lines.push(
    "        the R24 baseline pin; the derived state after rung 8 must match its computation-shape,",
  );
  lines.push(
    "        at ≤1.5× geomean / ≤2.0× worst-case perf (§8 gate 3, pre-registered).",
  );

  // eslint-disable-next-line no-console
  console.log(lines.join("\n"));

  process.exit(0);
}

main();
