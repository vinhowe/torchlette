/**
 * Stage-4 phase 1: graph-liveness memory planner.
 *
 * Replaces "pin whatever buffers the recording execution happened to
 * allocate" (the planned-buffers mechanism, aa2a7f5) with a DERIVED
 * assignment: slot lifetimes are computed from the command stream, and
 * buffers are assigned by greedy interval allocation over size classes —
 * the same reuse the pool discovered dynamically, now deterministic,
 * plan-owned, and auditable. One owner per buffer; no adoption, no pins,
 * no replay-time fallback heuristics for alloc slots.
 *
 * Validation: structural asserts at build time (no overlapping lifetimes on
 * one buffer; in-place aliases excluded) + the stage-4 ladder (stream
 * differential, fullstack parity, regression, suites, memory measurements).
 */
import { getSizeClass, getSizeForClass } from "../graph/lifetime-analysis";
import type { GpuCommand } from "./compiled-plan";
import {
  TAG_ALLOC,
  TAG_CLEAR,
  TAG_COPY,
  TAG_DISPATCH,
  TAG_UNIFORM,
  TAG_WRITE,
} from "./compiled-plan";

export interface MemoryPlan {
  /** alloc-slot → planner buffer index. */
  assignment: Map<number, number>;
  /** Planner buffer byte sizes (size-class rounded), by buffer index. */
  bufferBytes: number[];
  /** Total planned bytes (sum of bufferBytes). */
  totalBytes: number;
}

/**
 * Compute per-slot lifetimes over the command stream and assign planner
 * buffers by first-fit interval allocation within size classes.
 *
 * Only slots allocated via TAG_ALLOC participate (external / persistent /
 * params / write slots have their own owners). A slot's lifetime spans its
 * alloc command through its last referencing command; a freed buffer is
 * reusable from the NEXT command onward (never within the same command —
 * same-dispatch read/write aliasing is a WebGPU validation error unless a
 * kernel was built for it, and those in-place forms already share a single
 * slot upstream via donation/in-place discipline).
 */
export function planMemory(
  commands: GpuCommand[],
  /** Slots whose buffers ESCAPE the plan as harvested node results —
   *  downstream plans read them after this stream completes, so their
   *  lifetime is the plan's lifetime: never recycled within the stream. */
  resultSlots?: Set<number>,
): MemoryPlan {
  // 1. Lifetimes: per alloc-slot [allocIdx, lastUseIdx] + bytes.
  const allocIdx = new Map<number, number>();
  const lastUse = new Map<number, number>();
  const bytes = new Map<number, number>();
  const touch = (slot: number, i: number) => {
    if (allocIdx.has(slot)) lastUse.set(slot, i);
  };
  for (let i = 0; i < commands.length; i++) {
    const cmd = commands[i];
    switch (cmd.tag) {
      case TAG_ALLOC:
        allocIdx.set(cmd.slot, i);
        lastUse.set(cmd.slot, i);
        bytes.set(cmd.slot, cmd.bytes);
        for (const s of cmd.inputSlots) touch(s, i);
        break;
      case TAG_DISPATCH:
        for (const s of cmd.bindings) touch(s, i);
        break;
      case TAG_COPY:
        touch(cmd.src, i);
        touch(cmd.dst, i);
        break;
      case TAG_CLEAR:
      case TAG_WRITE:
      case TAG_UNIFORM:
        touch(cmd.slot, i);
        break;
      default:
        break;
    }
  }

  // 2. Greedy interval allocation, command order = interval order.
  //    free[sizeClass] = planner buffer indices free at the current point.
  const assignment = new Map<number, number>();
  const bufferBytes: number[] = [];
  const free = new Map<number, number[]>();
  // Slots ordered by alloc position; releases processed before the alloc at
  // the same position would violate the next-command rule, so we release
  // strictly AFTER the releasing command's index (releaseAt = lastUse + 1).
  const slots = [...allocIdx.keys()].sort(
    (a, b) => (allocIdx.get(a) as number) - (allocIdx.get(b) as number),
  );
  // Release queue sorted by releaseAt.
  const releases: Array<{ at: number; bufIdx: number }> = [];
  let releasePos = 0;
  const processReleases = (upto: number) => {
    releases.sort((a, b) => a.at - b.at);
    while (releasePos < releases.length && releases[releasePos].at <= upto) {
      const { bufIdx } = releases[releasePos++];
      const sc = getSizeClass(bufferBytes[bufIdx]);
      let list = free.get(sc);
      if (!list) free.set(sc, (list = []));
      list.push(bufIdx);
    }
  };
  for (const slot of slots) {
    const at = allocIdx.get(slot) as number;
    processReleases(at - 1);
    const sc = getSizeClass(bytes.get(slot) as number);
    const list = free.get(sc);
    let bufIdx: number;
    if (list && list.length > 0) {
      bufIdx = list.pop() as number;
    } else {
      bufIdx = bufferBytes.length;
      bufferBytes.push(getSizeForClass(sc));
    }
    assignment.set(slot, bufIdx);
    if (!resultSlots?.has(slot)) {
      releases.push({ at: (lastUse.get(slot) as number) + 1, bufIdx });
    }
  }

  // 3. Structural audit: no two slots with overlapping lifetimes share a
  //    buffer. O(n²) worst case but slot counts are ~1e3; cheap and always on.
  const byBuf = new Map<number, number[]>();
  for (const [slot, bufIdx] of assignment) {
    let l = byBuf.get(bufIdx);
    if (!l) byBuf.set(bufIdx, (l = []));
    l.push(slot);
  }
  for (const [bufIdx, slotList] of byBuf) {
    slotList.sort((a, b) => (allocIdx.get(a) as number) - (allocIdx.get(b) as number));
    for (let i = 1; i < slotList.length; i++) {
      const prev = slotList[i - 1];
      const cur = slotList[i];
      if (
        resultSlots?.has(prev) ||
        (lastUse.get(prev) as number) >= (allocIdx.get(cur) as number)
      ) {
        throw new Error(
          `[memory-planner] OVERLAP: planner buffer ${bufIdx} assigned to slot ${prev} (live ..${lastUse.get(prev)}) and slot ${cur} (from ${allocIdx.get(cur)})`,
        );
      }
    }
  }

  let totalBytes = 0;
  for (const b of bufferBytes) totalBytes += b;
  return { assignment, bufferBytes, totalBytes };
}
