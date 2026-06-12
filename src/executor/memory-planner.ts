/**
 * Stage-4 phase 1/1.5: graph-liveness memory planner.
 *
 * Replaces "pin whatever buffers the recording execution happened to
 * allocate" (the planned-buffers mechanism, aa2a7f5) with a DERIVED
 * assignment: slot lifetimes are computed from the command stream, and
 * buffers are assigned by greedy interval allocation over size classes —
 * the same reuse the pool discovered dynamically, now deterministic,
 * plan-owned, and auditable.
 *
 * Phase 1.5 (cross-plan packing): plans of one step execute strictly
 * sequentially on the queue, so a buffer that holds only intra-plan-dead
 * data (a TEMP — any non-result alloc) is safe to share with EVERY other
 * plan, in any step order. Temps therefore draw from a step-scoped shared
 * registry of entries; RESULT slots (harvested node results, read by later
 * plans within the step) stay exclusive to their plan for its lifetime.
 * This is the same sharing the pin mechanism got implicitly (a recording
 * could only reuse pool buffers that died before an earlier plan's build —
 * i.e. exactly the earlier plans' dead temps), made deterministic and
 * complete: ALL temps share, not just the ones that happened to flow
 * through the pool at the right moment.
 *
 * Validation: structural asserts at build time (no overlapping lifetimes on
 * one entry; result slots never recycled) + the stage-4 ladder (stream
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

/** One shared planner buffer. The registry owns the GPU buffer; compiled
 *  plans reference entries by index and co-own them via `owners`. */
export interface PlannerEntry {
  /** Size-class-rounded byte size. */
  bytes: number;
  /** Size class (stored, not recomputed — classification stability). */
  sizeClass: number;
  /** Materialized GPU buffer (lazily created at first replay binding). */
  buffer?: GPUBuffer;
  /** Holds some plan's RESULT: live across plans within a step, so the
   *  entry is exclusive to its owning plan while that plan lives. */
  resultHolder: boolean;
  /** Compiled plans whose assignments reference this entry. */
  owners: Set<object>;
  /** Whether the entry is in the shareable free lists. */
  listed: boolean;
}

/**
 * Step-scoped shared buffer registry. Module-singleton (owned by
 * compiled-plan.ts); engine-instance boundaries reset it alongside the
 * template cache (instance boundaries are cache boundaries).
 */
export class PlannerRegistry {
  entries: PlannerEntry[] = [];
  /** Bumped on reset; plans built against an older generation must not
   *  touch the new entries array (index collision). */
  generation = 0;
  /** sizeClass → entry indices shareable across plans (non-result temps).
   *  Membership is persistent: sharing is order-independent (temps are dead
   *  across plan boundaries in both directions), so an entry stays listed
   *  for every subsequently-built plan. */
  private shareable = new Map<number, number[]>();

  /** Copy of the shareable lists, seeding one plan's interval allocation. */
  seedFreeLists(): Map<number, number[]> {
    const out = new Map<number, number[]>();
    for (const [sc, list] of this.shareable) out.set(sc, [...list]);
    return out;
  }

  newEntry(sizeClass: number, resultHolder: boolean): number {
    const idx = this.entries.length;
    this.entries.push({
      bytes: getSizeForClass(sizeClass),
      sizeClass,
      resultHolder,
      owners: new Set(),
      listed: !resultHolder,
    });
    if (!resultHolder) this.pushShareable(sizeClass, idx);
    return idx;
  }

  /** Re-list an entry whose owners all died: its exclusivity constraint
   *  ended with the owning plan, so future plans may temp-claim it. */
  relist(idx: number): void {
    const e = this.entries[idx];
    if (e.listed) return;
    e.resultHolder = false;
    e.listed = true;
    this.pushShareable(e.sizeClass, idx);
  }

  private pushShareable(sizeClass: number, idx: number): void {
    let list = this.shareable.get(sizeClass);
    if (!list) this.shareable.set(sizeClass, (list = []));
    list.push(idx);
  }

  /** Drop all entries (returning them for buffer teardown) and bump the
   *  generation so stale plans can't index into the new array. */
  reset(): PlannerEntry[] {
    const old = this.entries;
    this.entries = [];
    this.shareable = new Map();
    this.generation++;
    return old;
  }
}

export interface MemoryPlan {
  /** alloc-slot → registry entry index. */
  assignment: Map<number, number>;
  /** Unique registry entry indices used by this plan (ownership list). */
  entries: number[];
  /** Bytes of entries newly created for this plan. */
  newBytes: number;
  /** Bytes of pre-existing shared entries reused by this plan. */
  reusedBytes: number;
}

/**
 * Compute per-slot lifetimes over the command stream and assign registry
 * entries by first-fit interval allocation within size classes.
 *
 * Only slots allocated via TAG_ALLOC participate (external / persistent /
 * params / write slots have their own owners). A slot's lifetime spans its
 * alloc command through its last referencing command; a freed entry is
 * reusable from the NEXT command onward (never within the same command —
 * same-dispatch read/write aliasing is a WebGPU validation error unless a
 * kernel was built for it, and those in-place forms already share a single
 * slot upstream via donation/in-place discipline).
 */
export function planMemory(
  commands: GpuCommand[],
  registry: PlannerRegistry,
  /** Slots whose buffers ESCAPE the plan as harvested node results —
   *  downstream plans read them after this stream completes, so their
   *  lifetime is the plan's lifetime: never recycled within the stream,
   *  never assigned to a shared entry. */
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
  //    free[sizeClass] = entry indices free at the current point — seeded
  //    with the registry's shareable entries (free from position 0: temps
  //    of other plans are dead whenever this plan runs).
  const assignment = new Map<number, number>();
  const usedEntries = new Set<number>();
  const free = registry.seedFreeLists();
  let newBytes = 0;
  let reusedBytes = 0;
  // Slots ordered by alloc position; releases processed before the alloc at
  // the same position would violate the next-command rule, so we release
  // strictly AFTER the releasing command's index (releaseAt = lastUse + 1).
  const slots = [...allocIdx.keys()].sort(
    (a, b) => (allocIdx.get(a) as number) - (allocIdx.get(b) as number),
  );
  // Release queue sorted by releaseAt.
  const releases: Array<{ at: number; entryIdx: number }> = [];
  let releasePos = 0;
  const processReleases = (upto: number) => {
    releases.sort((a, b) => a.at - b.at);
    while (releasePos < releases.length && releases[releasePos].at <= upto) {
      const { entryIdx } = releases[releasePos++];
      const sc = registry.entries[entryIdx].sizeClass;
      let list = free.get(sc);
      if (!list) free.set(sc, (list = []));
      list.push(entryIdx);
    }
  };
  for (const slot of slots) {
    const at = allocIdx.get(slot) as number;
    processReleases(at - 1);
    const sc = getSizeClass(bytes.get(slot) as number);
    const isResult = resultSlots?.has(slot) ?? false;
    let entryIdx: number;
    const list = isResult ? undefined : free.get(sc);
    if (list && list.length > 0) {
      entryIdx = list.pop() as number;
      if (!usedEntries.has(entryIdx)) {
        reusedBytes += registry.entries[entryIdx].bytes;
      }
    } else {
      entryIdx = registry.newEntry(sc, isResult);
      newBytes += registry.entries[entryIdx].bytes;
    }
    usedEntries.add(entryIdx);
    assignment.set(slot, entryIdx);
    if (!isResult) {
      releases.push({ at: (lastUse.get(slot) as number) + 1, entryIdx });
    }
  }
  // 3. Structural audit: no two slots with overlapping lifetimes share an
  //    entry, and result slots are never followed by another lifetime.
  //    O(n²) worst case but slot counts are ~1e3; cheap and always on.
  const byEntry = new Map<number, number[]>();
  for (const [slot, entryIdx] of assignment) {
    let l = byEntry.get(entryIdx);
    if (!l) byEntry.set(entryIdx, (l = []));
    l.push(slot);
  }
  for (const [entryIdx, slotList] of byEntry) {
    slotList.sort(
      (a, b) => (allocIdx.get(a) as number) - (allocIdx.get(b) as number),
    );
    for (let i = 1; i < slotList.length; i++) {
      const prev = slotList[i - 1];
      const cur = slotList[i];
      if (
        resultSlots?.has(prev) ||
        (lastUse.get(prev) as number) >= (allocIdx.get(cur) as number)
      ) {
        throw new Error(
          `[memory-planner] OVERLAP: entry ${entryIdx} assigned to slot ${prev} (live ..${lastUse.get(prev)}) and slot ${cur} (from ${allocIdx.get(cur)})`,
        );
      }
    }
    // A result slot must own its entry exclusively within this plan.
    if (
      slotList.length > 1 &&
      slotList.some((s) => resultSlots?.has(s) ?? false)
    ) {
      throw new Error(
        `[memory-planner] RESULT-SHARED: entry ${entryIdx} hosts a result slot alongside other lifetimes (slots ${slotList.join(",")})`,
      );
    }
  }

  return {
    assignment,
    entries: [...usedEntries],
    newBytes,
    reusedBytes,
  };
}
