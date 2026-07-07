/**
 * Step-tape phase-1a MEASUREMENT seams (docs/staged-execution-phase1.md, G0).
 *
 * Two env flags, both default-off, both observation-only (zero behavior
 * change; every call site is guarded by a module-load-time const so the
 * flag-off cost is one boolean test):
 *
 * - TORCHLETTE_TAPE_PROFILE=1 — per-step wall-time accumulators at the exact
 *   seams the step-tape would skip (plan collect / fingerprint / CSE+rewrite
 *   passes / template lookup / compiled-replay JS vs Dawn calls). Consumed by
 *   examples/qwen3/timeline-decode.ts (the G0 table).
 *
 * - TORCHLETTE_TAPE_SLOTDIFF=1 — records, per executed plan, a per-node
 *   payload/scalar byte-image (the §2.4 guard-3 consecutive-step diff
 *   technique). Consumed by examples/qwen3-steering/tape-slot-diff.ts
 *   (the DynamicSlot enumeration).
 *
 * SUNSET: this module, both flags, and every `TAPE_PROFILE`/`TAPE_SLOTDIFF`
 * guard referencing it are DELETED when step-tape phase 1c lands — the tape
 * replay makes these seams unreachable on the measured path, and 1c ships its
 * own guard counters. Measurement instrumentation only; must not grow
 * behavior.
 */

import { ENV } from "./env";

export const TAPE_PROFILE: boolean = ENV.TORCHLETTE_TAPE_PROFILE === "1";
export const TAPE_SLOTDIFF: boolean = ENV.TORCHLETTE_TAPE_SLOTDIFF === "1";

// ---------------------------------------------------------------------------
// Wall-time accumulators (TORCHLETTE_TAPE_PROFILE)
// ---------------------------------------------------------------------------

const acc: Record<string, number> = Object.create(null);
const cnt: Record<string, number> = Object.create(null);

export function tpAdd(key: string, ms: number): void {
  acc[key] = (acc[key] ?? 0) + ms;
  cnt[key] = (cnt[key] ?? 0) + 1;
}

/** Reset all accumulators (harness calls this at each step start). */
export function tpReset(): void {
  for (const k of Object.keys(acc)) delete acc[k];
  for (const k of Object.keys(cnt)) delete cnt[k];
}

/** Snapshot accumulated ms + call counts since the last tpReset(). */
export function tpGet(): {
  ms: Record<string, number>;
  counts: Record<string, number>;
} {
  return { ms: { ...acc }, counts: { ...cnt } };
}

// ---------------------------------------------------------------------------
// Per-plan payload/scalar images (TORCHLETTE_TAPE_SLOTDIFF)
// ---------------------------------------------------------------------------

export interface TpNodeRecord {
  /** Position in plan.nodes. */
  pos: number;
  op: string;
  /** FNV-1a over the stable payload serialization (0 = no payload). */
  payloadHash: number;
  /** Stable payload serialization, capped (for explaining diffs). */
  payloadRepr: string;
  /** Values of scalar-kind input refs, in input order. */
  scalars: number[];
}

export interface TpPlanRecord {
  /** Template fingerprint primary (identifies the plan across steps). */
  fpPrimary: number;
  nodeCount: number;
  nodes: TpNodeRecord[];
}

const planRecords: TpPlanRecord[] = [];

function fnv(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Stable, cap-length payload serialization: sorted keys; typed arrays hash
 *  their full contents but print only a head slice. */
function stableRepr(v: unknown, depth = 0): string {
  if (v === null || v === undefined) return String(v);
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (typeof v === "function") return "<fn>";
  if (ArrayBuffer.isView(v)) {
    const ta = v as unknown as ArrayLike<number> & {
      constructor: { name: string };
    };
    let h = 0x811c9dc5;
    for (let i = 0; i < ta.length; i++) {
      // hash the numeric value's string form — content-sensitive, cheap
      const s = String(ta[i]);
      for (let j = 0; j < s.length; j++) {
        h ^= s.charCodeAt(j);
        h = Math.imul(h, 0x01000193);
      }
    }
    const head = Array.prototype.slice.call(ta, 0, 4).join(",");
    return `${ta.constructor.name}(len=${ta.length},hash=${(h >>> 0).toString(16)},head=[${head}])`;
  }
  if (Array.isArray(v)) {
    if (depth > 4) return "[...]";
    return `[${v.map((x) => stableRepr(x, depth + 1)).join(",")}]`;
  }
  if (typeof v === "object") {
    if (depth > 4) return "{...}";
    const keys = Object.keys(v as object).sort();
    return `{${keys
      .map(
        (k) =>
          `${k}:${stableRepr((v as Record<string, unknown>)[k], depth + 1)}`,
      )
      .join(",")}}`;
  }
  return `<${typeof v}>`;
}

/** Record one executed plan's payload/scalar image. Called from
 *  executePlanOptimized (after fingerprinting) when TAPE_SLOTDIFF is on. */
export function tpRecordPlan(
  fpPrimary: number,
  nodes: ReadonlyArray<{
    op: string;
    payload?: unknown;
    inputs: ReadonlyArray<{ kind: string; value?: number }>;
  }>,
): void {
  const recs: TpNodeRecord[] = [];
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    const scalars: number[] = [];
    for (const inp of n.inputs) {
      if (inp.kind === "scalar") scalars.push(inp.value as number);
    }
    if (n.payload === undefined && scalars.length === 0) continue;
    const repr = n.payload === undefined ? "" : stableRepr(n.payload);
    recs.push({
      pos: i,
      op: n.op,
      payloadHash: repr ? fnv(repr) : 0,
      payloadRepr: repr.length > 220 ? `${repr.slice(0, 220)}…` : repr,
      scalars,
    });
  }
  planRecords.push({ fpPrimary, nodeCount: nodes.length, nodes: recs });
}

/** Take (and clear) all plan records accumulated since the last take. */
export function tpTakePlanRecords(): TpPlanRecord[] {
  const out = planRecords.slice();
  planRecords.length = 0;
  return out;
}
