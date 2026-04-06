/**
 * Interval graph: a set of intervals [start, end] × size, with queries for
 * overlap and coloring. This is the foundational data structure for memory
 * scheduling — every tensor's live range is an interval, and the scheduler
 * assigns non-overlapping intervals to the same slot.
 *
 * Pure data structure: no knowledge of tensors, nodes, or buffers.
 */

/** An interval with a size. Intervals are inclusive on both ends. */
export interface Interval {
  readonly id: number;
  readonly start: number;
  readonly end: number;
  readonly size: number;
}

/** Do two intervals overlap (share any point in time)? */
export function intervalsOverlap(a: Interval, b: Interval): boolean {
  return a.start <= b.end && b.start <= a.end;
}

/**
 * Collection of intervals with overlap queries and greedy coloring.
 */
export class IntervalGraph {
  private readonly intervals: Interval[] = [];
  private readonly byId = new Map<number, Interval>();

  add(interval: Interval): void {
    if (this.byId.has(interval.id)) {
      throw new Error(`IntervalGraph: duplicate id ${interval.id}`);
    }
    if (interval.start > interval.end) {
      throw new Error(
        `IntervalGraph: interval ${interval.id} has start=${interval.start} > end=${interval.end}`,
      );
    }
    this.intervals.push(interval);
    this.byId.set(interval.id, interval);
  }

  get(id: number): Interval | undefined {
    return this.byId.get(id);
  }

  all(): readonly Interval[] {
    return this.intervals;
  }

  /** Are the two intervals (by id) overlapping? */
  overlaps(a: number, b: number): boolean {
    const ia = this.byId.get(a);
    const ib = this.byId.get(b);
    if (!ia || !ib) return false;
    return intervalsOverlap(ia, ib);
  }

  /** Which intervals are live at this position (inclusive)? */
  liveAt(position: number): readonly Interval[] {
    return this.intervals.filter(
      (i) => i.start <= position && position <= i.end,
    );
  }

  /**
   * Greedy coloring: for each interval (sorted by start), assign the
   * smallest color not already used by an interval it overlaps with.
   * Returns a Map from interval id to color (small non-negative integer).
   *
   * Classic result: for interval graphs, this greedy produces an OPTIMAL
   * chromatic number (equal to the max clique size = max concurrent live
   * intervals). So this really is the minimum number of colors needed.
   */
  color(): Map<number, number> {
    const coloring = new Map<number, number>();
    // Sort by start, then end as tiebreaker.
    const sorted = [...this.intervals].sort(
      (a, b) => a.start - b.start || a.end - b.end,
    );
    for (const interval of sorted) {
      // Find colors already used by intervals that overlap with this one
      // AND have already been colored.
      const usedColors = new Set<number>();
      for (const other of sorted) {
        if (other.id === interval.id) continue;
        if (!coloring.has(other.id)) continue;
        if (intervalsOverlap(interval, other)) {
          usedColors.add(coloring.get(other.id)!);
        }
      }
      // Pick smallest color not in usedColors.
      let color = 0;
      while (usedColors.has(color)) color++;
      coloring.set(interval.id, color);
    }
    return coloring;
  }
}
