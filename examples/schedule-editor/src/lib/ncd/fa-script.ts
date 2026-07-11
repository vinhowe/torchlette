import {
  applyMove,
  inverseMove,
  napkinCost,
  onlineSoftmaxLemma,
  partitionLegality,
  recolorLegality,
  termHash,
} from "./model";
import type { NcdHistoryEntry, NcdMove, NcdTerm } from "./types";

export interface FaStep {
  id: string;
  label: string;
  makeMove(term: NcdTerm): NcdMove;
  legality(term: NcdTerm, move: NcdMove): { legal: boolean; reason: string };
}

export const FA_STEPS: FaStep[] = [
  {
    id: "admit-online-softmax",
    label: "Admit the online-softmax rescaling lemma",
    makeMove: onlineSoftmaxLemma,
    legality: () => ({
      legal: true,
      reason:
        "The admitted lemma supplies softmax's missing head/body decomposition.",
    }),
  },
  {
    id: "fuse-qk-softmax",
    label: "Recolor scores at column 2 from ℓ0 to ℓ1",
    makeMove: () => ({
      op: "recolor",
      wireId: "scores",
      column: 2,
      before: "l0",
      after: "l1",
    }),
    legality: (term, move) =>
      move.op === "recolor"
        ? recolorLegality(term, move.wireId, move.column, move.after)
        : { legal: false, reason: "Expected recoloring." },
  },
  {
    id: "fuse-softmax-pv",
    label: "Recolor probabilities at column 4 from ℓ0 to ℓ1",
    makeMove: () => ({
      op: "recolor",
      wireId: "probabilities",
      column: 4,
      before: "l0",
      after: "l1",
    }),
    legality: (term, move) =>
      move.op === "recolor"
        ? recolorLegality(term, move.wireId, move.column, move.after)
        : { legal: false, reason: "Expected recoloring." },
  },
  {
    id: "tile-q",
    label: "Group-partition q with g_q=64",
    makeMove: (term) => ({
      op: "partition",
      axisId: "q",
      before: term.decorations.partitions.find((item) => item.axisId === "q"),
      after: { axisId: "q", kind: "group", size: 64, label: "g_q" },
    }),
    legality: (term, move) =>
      move.op === "partition" && move.after
        ? partitionLegality(term, move.after)
        : { legal: false, reason: "Expected group partition." },
  },
  {
    id: "stream-x",
    label: "Stream-partition x with s_x=32",
    makeMove: (term) => ({
      op: "partition",
      axisId: "x",
      before: term.decorations.partitions.find((item) => item.axisId === "x"),
      after: { axisId: "x", kind: "stream", size: 32, label: "s_x" },
    }),
    legality: (term, move) =>
      move.op === "partition" && move.after
        ? partitionLegality(term, move.after)
        : { legal: false, reason: "Expected stream partition." },
  },
];

export function applyFaStep(
  term: NcdTerm,
  stepIndex: number,
): { term: NcdTerm; entry: NcdHistoryEntry } {
  const step = FA_STEPS[stepIndex];
  if (!step) throw new Error(`Unknown FlashAttention step ${stepIndex}`);
  const forward = step.makeMove(term);
  const legal = step.legality(term, forward);
  if (!legal.legal) throw new Error(`${step.label}: ${legal.reason}`);
  const next = applyMove(term, forward);
  return {
    term: next,
    entry: {
      label: step.label,
      forward,
      inverse: inverseMove(forward),
      termHash: termHash(next),
      cost: napkinCost(next),
    },
  };
}

export function deriveFlashAttention(term: NcdTerm): NcdTerm {
  let current = term;
  for (let index = 0; index < FA_STEPS.length; index += 1) {
    current = applyFaStep(current, index).term;
  }
  return current;
}
