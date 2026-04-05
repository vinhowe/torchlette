/**
 * Live range analysis: for each node in a plan, compute the interval over
 * which its output buffer must stay alive (from its position as producer
 * to the maximum position among its in-plan consumers).
 *
 * Externally-referenced nodes (their outputs are consumed by pending
 * RuntimeTensors outside the plan, e.g. gradient outputs / saved-for-
 * backward) get `external: true` and are NOT schedulable — their lifetimes
 * exceed the plan. The scheduler treats them as always-live.
 */
import type { DType } from "../../backend/types";
import type { LazyIRNode } from "../../graph/types";

export interface TensorLiveRange {
  readonly id: number; // == producer node id
  readonly start: number; // producer position
  readonly end: number; // last-consumer position (== start if external-only)
  readonly size: number; // bytes
  readonly external: boolean;
}

/** Size in bytes for a tensor of given shape and dtype. */
export function tensorBytes(shape: readonly number[], dtype: DType): number {
  let count = 1;
  for (const d of shape) count *= d;
  return count * dtypeBytes(dtype);
}

/** Bytes per element for torchlette's dtype set. */
export function dtypeBytes(dtype: DType): number {
  switch (dtype) {
    case "f32":
    case "i32":
    case "u32":
      return 4;
    case "f16":
      return 2;
    case "i8":
    case "u8":
      return 1;
    default:
      // f64 etc. — conservative default
      return 4;
  }
}

/**
 * Compute live ranges for every node in the plan.
 *
 * @param plan - topologically-ordered plan nodes
 * @param externalNodeIds - node ids with pending RuntimeTensors outside
 *   the plan (saved-for-backward, gradient outputs, etc.)
 * @returns map from node id to its live range
 */
export function analyzeLiveRanges(
  plan: readonly LazyIRNode[],
  externalNodeIds?: ReadonlySet<number>,
): Map<number, TensorLiveRange> {
  // Position of each node in the plan.
  const position = new Map<number, number>();
  for (let i = 0; i < plan.length; i++) {
    position.set(plan[i].id, i);
  }

  // First pass: initialize every node's range as [producer_pos, producer_pos].
  const ranges = new Map<number, TensorLiveRange>();
  for (let i = 0; i < plan.length; i++) {
    const node = plan[i];
    const external = externalNodeIds?.has(node.id) ?? false;
    ranges.set(node.id, {
      id: node.id,
      start: i,
      end: i,
      size: tensorBytes(node.shape, node.dtype),
      external,
    });
  }

  // Second pass: extend end to max consumer position.
  for (let i = 0; i < plan.length; i++) {
    const node = plan[i];
    for (const inputRef of node.inputs) {
      if (inputRef.kind !== "pending") continue;
      const producer = ranges.get(inputRef.node.id);
      if (!producer) continue; // producer not in plan (e.g., already executed)
      if (i > producer.end) {
        ranges.set(inputRef.node.id, {
          id: producer.id,
          start: producer.start,
          end: i,
          size: producer.size,
          external: producer.external,
        });
      }
    }
  }

  // External nodes: extend end to plan.length - 1 (they're live past the
  // plan, so for scheduling purposes we treat them as live through the end).
  // Note: we don't use Infinity here because the scheduler reasons about
  // concrete positions. A tensor live through position-max won't be
  // schedulable with anything else anyway.
  const lastPos = plan.length - 1;
  for (const [id, range] of ranges) {
    if (range.external && range.end < lastPos) {
      ranges.set(id, {
        id: range.id,
        start: range.start,
        end: lastPos,
        size: range.size,
        external: true,
      });
    }
  }

  return ranges;
}

/** Compute simple stats about a set of live ranges (for diagnostics). */
export function liveRangeStats(
  ranges: Map<number, TensorLiveRange>,
): {
  count: number;
  external: number;
  totalBytes: number;
  maxConcurrentBytes: number;
  maxConcurrentCount: number;
} {
  let totalBytes = 0;
  let external = 0;
  for (const r of ranges.values()) {
    totalBytes += r.size;
    if (r.external) external++;
  }

  // Sweep to find max concurrent bytes.
  let maxPos = 0;
  for (const r of ranges.values()) maxPos = Math.max(maxPos, r.end);
  let maxConcurrentBytes = 0;
  let maxConcurrentCount = 0;
  for (let p = 0; p <= maxPos; p++) {
    let b = 0;
    let c = 0;
    for (const r of ranges.values()) {
      if (r.start <= p && p <= r.end) {
        b += r.size;
        c++;
      }
    }
    if (b > maxConcurrentBytes) maxConcurrentBytes = b;
    if (c > maxConcurrentCount) maxConcurrentCount = c;
  }

  return {
    count: ranges.size,
    external,
    totalBytes,
    maxConcurrentBytes,
    maxConcurrentCount,
  };
}
