/**
 * Cost model: evaluate a scheduling assignment without executing anything.
 *
 * Given live ranges and an assignment (tensor → slot), compute the peak
 * memory: the maximum total bytes in use across all time positions. This
 * is the metric schedulers optimize for.
 */
import type { TensorLiveRange } from "./live-range";

export interface Assignment {
  /** Tensor id → slot id. */
  readonly tensorToSlot: Map<number, number>;
  /** Slot id → size in bytes (max of all tensors assigned to the slot). */
  readonly slotSizes: Map<number, number>;
}

/** Sum of slot sizes currently live at each position, max across positions. */
export function computePeak(
  ranges: Map<number, TensorLiveRange>,
  assignment: Assignment,
): number {
  const maxPos = findMaxPos(ranges);
  if (maxPos < 0) return 0;

  // For each position, sum up sizes of slots that are live there.
  let peak = 0;
  for (let p = 0; p <= maxPos; p++) {
    const liveSlots = new Set<number>();
    for (const r of ranges.values()) {
      if (r.start <= p && p <= r.end) {
        const slot = assignment.tensorToSlot.get(r.id);
        if (slot !== undefined) liveSlots.add(slot);
      }
    }
    let bytes = 0;
    for (const slot of liveSlots) {
      bytes += assignment.slotSizes.get(slot) ?? 0;
    }
    if (bytes > peak) peak = bytes;
  }
  return peak;
}

/** Peak per position (for diagnostics / plotting). */
export function computePeakOverTime(
  ranges: Map<number, TensorLiveRange>,
  assignment: Assignment,
): Array<{ position: number; bytes: number; liveSlots: number }> {
  const maxPos = findMaxPos(ranges);
  const out: Array<{ position: number; bytes: number; liveSlots: number }> = [];
  for (let p = 0; p <= maxPos; p++) {
    const liveSlots = new Set<number>();
    for (const r of ranges.values()) {
      if (r.start <= p && p <= r.end) {
        const slot = assignment.tensorToSlot.get(r.id);
        if (slot !== undefined) liveSlots.add(slot);
      }
    }
    let bytes = 0;
    for (const slot of liveSlots) {
      bytes += assignment.slotSizes.get(slot) ?? 0;
    }
    out.push({ position: p, bytes, liveSlots: liveSlots.size });
  }
  return out;
}

/** Trivial assignment: every tensor its own slot. Baseline for comparisons. */
export function trivialAssignment(
  ranges: Map<number, TensorLiveRange>,
): Assignment {
  const tensorToSlot = new Map<number, number>();
  const slotSizes = new Map<number, number>();
  let nextSlot = 0;
  for (const r of ranges.values()) {
    const slot = nextSlot++;
    tensorToSlot.set(r.id, slot);
    slotSizes.set(slot, r.size);
  }
  return { tensorToSlot, slotSizes };
}

/** Total bytes across all slots in the assignment (memory reserved). */
export function totalBytes(assignment: Assignment): number {
  let total = 0;
  for (const size of assignment.slotSizes.values()) total += size;
  return total;
}

function findMaxPos(ranges: Map<number, TensorLiveRange>): number {
  let max = -1;
  for (const r of ranges.values()) {
    if (r.end > max) max = r.end;
  }
  return max;
}
