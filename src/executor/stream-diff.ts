/**
 * Stage-4 phase 0: canonical serialization + diffing of GpuCommand streams.
 *
 * The compile-from-IR migration (docs/stage4-compile-from-ir.md) replaces
 * trace-recorded command streams with streams GENERATED from the lowered
 * plan. The migration's safety net is a differential at the command-stream
 * level: generated and recorded streams for the same plan must match. This
 * module is that seam.
 *
 * Canonical form: per-command lines over ABSTRACT identities — pipelines by
 * first-appearance interning (object identity is stable within one process
 * via the kernel caches), buffers by slot index (never by GPUBuffer — two
 * recordings of one template allocate different buffers but must serialize
 * identically), uniform packs by node index + byte length. This also gives
 * the phase-0 determinism gate: recording the SAME template twice must
 * produce byte-identical canonical streams.
 */
import type {
  CompiledPlan,
  DispatchCommand,
  GpuCommand,
} from "./compiled-plan";
import {
  TAG_ALLOC,
  TAG_BARRIER,
  TAG_CLEAR,
  TAG_COPY,
  TAG_DISPATCH,
  TAG_UNIFORM,
  TAG_WRITE,
} from "./compiled-plan";

export function canonicalizeStream(commands: GpuCommand[]): string[] {
  const pipelineIds = new Map<unknown, number>();
  const pid = (p: unknown): number => {
    let id = pipelineIds.get(p);
    if (id === undefined) {
      id = pipelineIds.size;
      pipelineIds.set(p, id);
    }
    return id;
  };
  const lines: string[] = [];
  for (const cmd of commands) {
    switch (cmd.tag) {
      case TAG_DISPATCH: {
        const d = cmd as DispatchCommand;
        lines.push(
          `dispatch p${pid(d.pipeline)} bind=[${d.bindings.join(",")}] wg=${d.gx}x${d.gy}x${d.gz} label=${d.label ?? "?"}`,
        );
        break;
      }
      case TAG_ALLOC:
        lines.push(
          `alloc slot=${cmd.slot} bytes=${cmd.bytes} kind=${cmd.allocKind} inputs=[${cmd.inputSlots.join(",")}]`,
        );
        break;
      case TAG_COPY:
        lines.push(
          `copy ${cmd.src}+${cmd.srcOffset} -> ${cmd.dst}+${cmd.dstOffset} bytes=${cmd.bytes}`,
        );
        break;
      case TAG_WRITE:
        lines.push(`write slot=${cmd.slot} node=${cmd.nodeIndex}`);
        break;
      case TAG_CLEAR:
        lines.push(`clear slot=${cmd.slot} bytes=${cmd.bytes}`);
        break;
      case TAG_UNIFORM:
        lines.push(`uniform slot=${cmd.slot} node=${cmd.nodeIndex}`);
        break;
      case TAG_BARRIER:
        lines.push("barrier");
        break;
      default:
        lines.push(`unknown tag=${(cmd as { tag: number }).tag}`);
    }
  }
  return lines;
}

export interface StreamDiff {
  equal: boolean;
  /** First diverging command index (or the shorter length on prefix match). */
  firstDivergence?: number;
  a?: string;
  b?: string;
  lengthA: number;
  lengthB: number;
}

export function diffStreams(a: GpuCommand[], b: GpuCommand[]): StreamDiff {
  const ca = canonicalizeStream(a);
  const cb = canonicalizeStream(b);
  const n = Math.min(ca.length, cb.length);
  for (let i = 0; i < n; i++) {
    if (ca[i] !== cb[i]) {
      return {
        equal: false,
        firstDivergence: i,
        a: ca[i],
        b: cb[i],
        lengthA: ca.length,
        lengthB: cb.length,
      };
    }
  }
  if (ca.length !== cb.length) {
    return {
      equal: false,
      firstDivergence: n,
      a: ca[n] ?? "<end>",
      b: cb[n] ?? "<end>",
      lengthA: ca.length,
      lengthB: cb.length,
    };
  }
  return { equal: true, lengthA: ca.length, lengthB: cb.length };
}

export function diffCompiledPlans(
  a: CompiledPlan,
  b: CompiledPlan,
): StreamDiff {
  return diffStreams(a.commands, b.commands);
}

// ============================================================================
// Stage-4 phase 2: segment-aligned diffing — verify each covered action's
// generated commands against the recording even when OTHER actions are
// uncovered. Recorded commands carry the plan-node index that produced them;
// slot numbering differs between the two streams once coverage has gaps, so
// comparison is done MODULO a slot bijection built incrementally (consistent
// across all compared segments — a generated slot must map to exactly one
// recorded slot and vice versa). Pipelines compare by object identity (both
// sides resolve through the same caches).
// ============================================================================

export interface SegmentDiffResult {
  verifiedActions: number;
  verifiedCommands: number;
  divergences: Array<{ nodeIndex: number; detail: string }>;
  /** Generated segments with no recorded counterpart (alignment gap). */
  unmatched: number;
}

export function diffSegmentsAligned(
  genSegments: Array<{ nodeIndex: number; commands: GpuCommand[] }>,
  recorded: GpuCommand[],
): SegmentDiffResult {
  const recByNode = new Map<number, GpuCommand[]>();
  for (const c of recorded) {
    const ni = (c as { nodeIndex?: number }).nodeIndex;
    if (ni === undefined || ni < 0) continue;
    let l = recByNode.get(ni);
    if (!l) recByNode.set(ni, (l = []));
    l.push(c);
  }
  const g2r = new Map<number, number>();
  const r2g = new Map<number, number>();
  const mapSlot = (g: number, r: number): boolean => {
    const eg = g2r.get(g);
    if (eg !== undefined) return eg === r;
    if (r2g.has(r)) return false;
    g2r.set(g, r);
    r2g.set(r, g);
    return true;
  };
  const mapSlots = (g: number[], r: number[]): boolean =>
    g.length === r.length && g.every((gs, i) => mapSlot(gs, r[i]));

  const result: SegmentDiffResult = {
    verifiedActions: 0,
    verifiedCommands: 0,
    divergences: [],
    unmatched: 0,
  };
  for (const seg of genSegments) {
    const rec = recByNode.get(seg.nodeIndex);
    if (!rec) {
      result.unmatched++;
      continue;
    }
    let detail: string | null = null;
    if (rec.length !== seg.commands.length) {
      detail = `command count gen=${seg.commands.length} rec=${rec.length} (rec: ${canonicalizeStream(rec).join(" | ")})`;
    } else {
      for (let i = 0; i < rec.length && !detail; i++) {
        const g = seg.commands[i];
        const r = rec[i];
        if (g.tag !== r.tag) {
          detail = `cmd ${i}: tag gen=${g.tag} rec=${r.tag}`;
          break;
        }
        switch (g.tag) {
          case TAG_ALLOC: {
            const rr = r as typeof g;
            if (g.bytes !== rr.bytes) detail = `cmd ${i}: alloc bytes gen=${g.bytes} rec=${rr.bytes}`;
            else if (g.allocKind !== rr.allocKind) detail = `cmd ${i}: allocKind gen=${g.allocKind} rec=${rr.allocKind}`;
            else if (!mapSlots(g.inputSlots, rr.inputSlots)) detail = `cmd ${i}: alloc inputSlots gen=[${g.inputSlots}] rec=[${rr.inputSlots}]`;
            else if (!mapSlot(g.slot, rr.slot)) detail = `cmd ${i}: alloc slot bijection gen=${g.slot} rec=${rr.slot}`;
            break;
          }
          case TAG_DISPATCH: {
            const rr = r as typeof g;
            if (g.pipeline !== rr.pipeline) detail = `cmd ${i}: pipeline identity differs`;
            else if (g.gx !== rr.gx || g.gy !== rr.gy || g.gz !== rr.gz) detail = `cmd ${i}: workgroups gen=${g.gx}x${g.gy}x${g.gz} rec=${rr.gx}x${rr.gy}x${rr.gz}`;
            else if (!mapSlots(g.bindings, rr.bindings)) detail = `cmd ${i}: bindings gen=[${g.bindings}] rec=[${rr.bindings}]`;
            break;
          }
          case TAG_CLEAR: {
            const rr = r as typeof g;
            if (g.bytes !== rr.bytes) detail = `cmd ${i}: clear bytes gen=${g.bytes} rec=${rr.bytes}`;
            else if (!mapSlot(g.slot, rr.slot)) detail = `cmd ${i}: clear slot bijection`;
            break;
          }
          case TAG_WRITE: {
            const rr = r as typeof g;
            if (g.nodeIndex !== rr.nodeIndex) detail = `cmd ${i}: write nodeIndex gen=${g.nodeIndex} rec=${rr.nodeIndex}`;
            else if (!mapSlot(g.slot, rr.slot)) detail = `cmd ${i}: write slot bijection`;
            break;
          }
          case TAG_COPY: {
            const rr = r as typeof g;
            if (g.bytes !== rr.bytes || g.srcOffset !== rr.srcOffset || g.dstOffset !== rr.dstOffset) detail = `cmd ${i}: copy geometry differs`;
            else if (!mapSlot(g.src, rr.src) || !mapSlot(g.dst, rr.dst)) detail = `cmd ${i}: copy slot bijection`;
            break;
          }
          default:
            break;
        }
      }
    }
    if (detail) {
      result.divergences.push({ nodeIndex: seg.nodeIndex, detail });
    } else {
      result.verifiedActions++;
      result.verifiedCommands += seg.commands.length;
    }
  }
  return result;
}

// ============================================================================
// Stage-4 phase 0: DispatchPlan interface (registry stub — no behavior yet).
// ============================================================================

/**
 * A declarative description of one op's GPU work: everything the stream
 * emitter needs to generate the op's commands WITHOUT executing it.
 * Imperative dispatch paths stay as the lowered-path executors; ops gain
 * `plan()` implementations op-by-op in phases 2-3 (see the design doc).
 */
export interface DispatchPlanStep {
  /** Kernel cache key — resolves to a pipeline via the existing caches. */
  pipelineKey: string;
  workgroups: [number, number, number];
  /** Binding roles in bind-group order: which logical tensor each slot is. */
  bindings: Array<
    | { kind: "input"; index: number }
    | { kind: "output"; index: number }
    | { kind: "temp"; index: number }
    | { kind: "uniform"; index: number }
  >;
  label?: string;
}

export interface DispatchPlanTemp {
  bytes: number;
}

export interface DispatchPlanUniform {
  /** Packed bytes for constant uniforms; volatile ones declare a packer. */
  data?: Uint32Array;
  volatilePack?: (node: unknown) => ArrayBufferView;
}

export interface OpDispatchPlan {
  steps: DispatchPlanStep[];
  temps: DispatchPlanTemp[];
  uniforms: DispatchPlanUniform[];
}

export type OpPlanner = (
  node: unknown,
  inputShapes: number[][],
) => OpDispatchPlan | null;

const opPlanners = new Map<string, OpPlanner>();

/** Register a declarative planner for an op (phases 2-3). */
export function registerOpPlanner(op: string, planner: OpPlanner): void {
  opPlanners.set(op, planner);
}

export function getOpPlanner(op: string): OpPlanner | undefined {
  return opPlanners.get(op);
}

/** Coverage counter: how many ops have declarative planners. */
export function plannerCoverage(): number {
  return opPlanners.size;
}
