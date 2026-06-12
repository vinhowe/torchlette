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
