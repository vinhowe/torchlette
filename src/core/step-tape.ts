/**
 * Step-tape phase-1b RECORDER (docs/staged-execution-phase1.md §2.1/§2.2,
 * as amended by §8) — a PURE OBSERVER of the existing execution path.
 *
 * `TORCHLETTE_STEP_TAPE=record` (default off; read once at module load so the
 * flag-off cost at every seam is a single boolean test). Observation only:
 * nothing reads the tape in 1b — replay + guards are phase 1c. The recorder
 * appends what actually happened during a step (slot writes on the TAG_WRITE
 * stable-buffer path, scalar-table writes, plan executions by template
 * fingerprint, readbacks), and at each markStep() compares the step against
 * the previous one:
 *
 *   - structurally identical (same ordered plan/write/readback signature) AND
 *   - every varying byte in the full payload/scalar image belongs to a
 *     DECLARED slot (TAG_WRITE-written node position, or a scalar-table slot)
 *     ⇒ the pair is ELIGIBLE and a StepTape is stored under its bucketKey.
 *   - any undeclared variance ⇒ the tape is REFUSED with a diagnostic naming
 *     the plan position + op (§2.4 guard 3, the PAYLOAD-THRASH sibling), and
 *     counted. Byte-identical rewrites are NON-slots by construction (they
 *     produce no image diff and, post the §8.4-item-4 scalar-table fix, no
 *     writes either).
 *
 * Guard bookkeeping recorded alongside (consumed by 1c, unread in 1b):
 *   guard 1 — structGen: the lazy-node op-sequence counter delta of the step
 *   guard 2 — bucketKey: derived from the step's structural signature (plan
 *             fingerprints + write positions + readback params); KV-bucket
 *             transitions and steering-structure changes land in different
 *             plan fingerprints, hence different bucketKeys
 *   guard 4 — plan validity: stInvalidateTemplate() is wired to compiled-plan
 *             destruction (CompiledPlan.tapeFp → destroyCompiledPlanBuffers);
 *             tapes referencing an invalidated template are dropped
 *   guard 5 — epoch/regime: tape records the engine epoch and the boundary
 *             regime (stepScopedCleanup); explicit beginStep()/endStep() or a
 *             regime toggle resets the consecutive-step comparator via
 *             stNoteBoundary()
 *
 * The tape references template fingerprints and slot ids ONLY — never
 * buffers, nodes, or storages. Nothing here owns GPU memory.
 *
 * Single-engine by phase-1 scope (like the scalar table's active map): state
 * is module-level; de-singleton is orthogonal (§1).
 *
 * ENV FLAG SUNSET: TORCHLETTE_STEP_TAPE is the CAMPAIGN flag — it survives
 * through 1c (=1 activates replay) and dies at the default-flip decision
 * (phase 2+, its own soak per §2.5).
 */

import { ENV } from "./env";

/**
 * `TORCHLETTE_STEP_TAPE`:
 *   "record" — 1b mode: pure observation, no replay.
 *   "1"      — 1c mode: observe AND replay (recording stays on so skeletons
 *              can be captured and re-recorded after a guard miss).
 * Recording fires for BOTH (replay needs the recorder's eligibility diff to
 * decide which skeletons are replayable — guard 3 is enforced at record time).
 */
export const STEP_TAPE_REPLAY: boolean = ENV.TORCHLETTE_STEP_TAPE === "1";
export const STEP_TAPE_RECORD: boolean =
  ENV.TORCHLETTE_STEP_TAPE === "record" || STEP_TAPE_REPLAY;

/** TAPE_VERIFY=N: every Nth replay-eligible step runs the normal path instead
 *  and cross-checks the skeleton it WOULD have replayed against the freshly
 *  built template (fp + command-stream diff). 0 = off. SUNSET: rides the
 *  TORCHLETTE_STEP_TAPE campaign — dies with it at the default-flip. */
export const STEP_TAPE_VERIFY_N: number = (() => {
  const v = Number(ENV.TORCHLETTE_TAPE_VERIFY ?? "0");
  return Number.isFinite(v) && v > 0 ? Math.floor(v) : 0;
})();

/** STRICT_TAPE=1: throw on any guard miss (steady state) or verify diff — the
 *  CI paranoia mode (§2.4 guard 6). SUNSET: rides TORCHLETTE_STEP_TAPE. */
export const STEP_TAPE_STRICT: boolean = ENV.TORCHLETTE_STRICT_TAPE === "1";

// ---------------------------------------------------------------------------
// Tape schema (§2.1 as amended by §8.4: 4th DynamicSlot source `scalar`)
// ---------------------------------------------------------------------------

export type DynamicSlotSource = "tokenId" | "upload" | "payload" | "scalar";

export interface DynamicSlot {
  /** "w:<fpHex>:<pos>" (TAG_WRITE) | "sc:<fpHex>:<pos>:<inputIndex>" (scalar table). */
  id: string;
  /** Human-readable: op + shape. */
  name: string;
  shape: number[];
  dtype: string;
  /** The recorder emits "upload" | "scalar" — tokenId is semantically an
   *  upload and is not distinguishable at the engine seam (1c may name it). */
  source: DynamicSlotSource;
  /** TAG_WRITE stable-buffer fast path (vs legacy re-executed write). */
  stable?: boolean;
}

export type TapeEntry =
  | { kind: "write"; slot: string }
  | { kind: "plan"; templateId: number }
  | {
      kind: "readback";
      which: "topk" | "read" | "item";
      params?: Record<string, number>;
    };

export interface StepTape {
  bucketKey: string;
  entries: TapeEntry[];
  slots: DynamicSlot[];
  /** Engine epoch at the recording step's boundary (guard 5). */
  epoch: number;
  /** Lazy-node op-sequence counter delta across the recorded step (guard 1). */
  structGen: number;
  /** Boundary regime of the recording step (guard 5). */
  regime: { stepScopedCleanup: boolean };
  /** Template fingerprints referenced (guard-4 invalidation index). */
  templateIds: Set<number>;
  /** Step ordinal (recorder-local) at which this tape was recorded. */
  recordedAtStep: number;
}

// ---------------------------------------------------------------------------
// Fast structural hashing (payload byte-images)
// ---------------------------------------------------------------------------

const F64 = new DataView(new ArrayBuffer(8));

function mix(h: number, x: number): number {
  h ^= x & 0xff;
  h = Math.imul(h, 0x01000193);
  h ^= (x >>> 8) & 0xff;
  h = Math.imul(h, 0x01000193);
  h ^= (x >>> 16) & 0xff;
  h = Math.imul(h, 0x01000193);
  h ^= (x >>> 24) & 0xff;
  return Math.imul(h, 0x01000193);
}

function mixString(h: number, s: string): number {
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h;
}

/** FNV-1a over a stable byte serialization of a payload value. Typed arrays
 *  hash their raw bytes (content-exact, no string building). */
function hashValue(h: number, v: unknown, depth: number): number {
  if (v === null) return mix(h, 0x6e756c);
  if (v === undefined) return mix(h, 0x756e64);
  const t = typeof v;
  if (t === "number") {
    F64.setFloat64(0, v as number);
    h = mix(h, F64.getUint32(0));
    return mix(h, F64.getUint32(4));
  }
  if (t === "boolean") return mix(h, v ? 0x74727565 : 0x66616c73);
  if (t === "string") return mixString(mix(h, 0x737472), v as string);
  if (t === "function") return mix(h, 0x666e); // structural constant
  if (ArrayBuffer.isView(v)) {
    const u8 = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
    h = mixString(mix(h, u8.byteLength), v.constructor.name);
    let i = 0;
    for (; i + 4 <= u8.length; i += 4) {
      h = mix(h, u8[i] | (u8[i + 1] << 8) | (u8[i + 2] << 16) | (u8[i + 3] << 24));
    }
    for (; i < u8.length; i++) {
      h ^= u8[i];
      h = Math.imul(h, 0x01000193);
    }
    return h;
  }
  if (depth > 5) return mix(h, 0x646565);
  if (Array.isArray(v)) {
    h = mix(h, 0x617272 ^ v.length);
    for (const x of v) h = hashValue(h, x, depth + 1);
    return h;
  }
  if (t === "object") {
    const keys = Object.keys(v as object).sort();
    h = mix(h, 0x6f626a ^ keys.length);
    for (const k of keys) {
      h = mixString(h, k);
      h = hashValue(h, (v as Record<string, unknown>)[k], depth + 1);
    }
    return h;
  }
  return mixString(h, String(t));
}

function fnvStr(s: string): number {
  return mixString(0x811c9dc5, s) >>> 0;
}

// ---------------------------------------------------------------------------
// Per-step observation state
// ---------------------------------------------------------------------------

/** Minimal structural view of a plan node (avoids graph-type imports). */
interface NodeLike {
  op: string;
  shape: number[];
  dtype: string;
  payload?: unknown;
  inputs: ReadonlyArray<{ kind: string; value?: unknown }>;
}

/** One node's payload/scalar image (only nodes with payload or scalar refs). */
interface NodeImage {
  pos: number;
  op: string;
  payloadHash: number; // 0 = no payload
  /** Flat [inputIndex, value, ...] pairs for scalar-kind refs. */
  scalars: number[] | null;
}

interface WriteObs {
  pos: number;
  op: string;
  shape: number[];
  dtype: string;
  stable: boolean;
}

interface PlanExecRecord {
  fp: number;
  image: NodeImage[];
  /** TAG_WRITE observations (stream order). */
  writes: WriteObs[];
  /** Node positions written this execution (declared payload coverage). */
  writtenPos: Set<number>;
  /** TAG_UNIFORM volatile-repack observations (§2.1 source "payload": the
   *  config re-packs from the fresh node payload every replay, so payload
   *  variance at these positions is carried as data). */
  uniforms: Array<{ pos: number; op: string }>;
  uniformPos: Set<number>;
  /** Scalar-table writes observed ([pos, inputIndex, value] triples). */
  scalarWrites: number[];
  /** True iff this plan executed via compiled replay. A lowered execution
   *  has no TAG_WRITE/TAG_UNIFORM stream (its uploads run as ordinary op
   *  dispatch), so it cannot declare its slot writes — and only compiled
   *  plans are replayable by a tape anyway. Eligibility requires fully
   *  compiled steps on both sides (mirrors the compiled plan's own
   *  record-on-2nd-execution lifecycle: the first two executions of a fresh
   *  template run lowered, so the warmup pair is counted as such, never
   *  refused). */
  compiled: boolean;
}

interface StepRecord {
  entries: TapeEntry[];
  plans: PlanExecRecord[];
  /** Structural signature (excludes value-conditional scalar writes). */
  structKey: string;
  opSeqEnd: number;
  epoch: number;
  stepScopedCleanup: boolean;
}

/** Declared scalar-table coverage per template fp: "pos:inputIndex" set. */
const declaredScalarSlots = new Map<number, Set<string>>();

let cur: { entries: TapeEntry[]; plans: PlanExecRecord[] } = {
  entries: [],
  plans: [],
};
let activePlan: PlanExecRecord | null = null;
let prev: StepRecord | null = null;
let stepOrdinal = 0;
/** Set by the 1c replay layer for the duration of a REPLAY step: the recorder
 *  must stay blind to it (a replay executes a compiled plan via the tape, not
 *  the observed normal path — its readback would append a plan-less interval
 *  and corrupt the consecutive-step comparator). Reset when the step's
 *  markStep finalizes, leaving `prev` untouched so the next NORMAL step
 *  resumes eligibility cleanly. */
let replaying = false;

// Tape store — in-memory, keyed by bucketKey (engine-scoped; see header).
const tapes = new Map<string, StepTape>();

// Counters + diagnostics
let refusals = 0;
let structureMisses = 0;
/** Set in stEndStep when the just-finalized step produced an eligible tape;
 *  consumed once by the replay layer (1c) to promote a captured skeleton. */
let lastEligible: {
  bucketKey: string;
  fps: number[];
  scalarSlots: Array<{ pos: number; inputIndex: number }>;
} | null = null;
let loweredPairs = 0;
let eligiblePairs = 0;
let stepsObserved = 0;
let boundaryResets = 0;
let planInvalidations = 0;
const refusalDiagnostics: string[] = [];
const MAX_DIAGNOSTICS = 32;
const MAX_WARNINGS = 8;

function hex(fp: number): string {
  return (fp >>> 0).toString(16);
}

// ---------------------------------------------------------------------------
// Recording seams (each call site guarded by STEP_TAPE_RECORD)
// ---------------------------------------------------------------------------

/** Plan execution begins (executor, after fingerprinting, permuted order).
 *  Captures the full payload/scalar byte-image for the guard-3 diff. */
export function stBeginPlan(fp: number, planNodes: readonly NodeLike[]): void {
  const image: NodeImage[] = [];
  for (let i = 0; i < planNodes.length; i++) {
    const n = planNodes[i];
    let scalars: number[] | null = null;
    const inputs = n.inputs;
    for (let ii = 0; ii < inputs.length; ii++) {
      if (inputs[ii].kind === "scalar") {
        (scalars ??= []).push(ii, inputs[ii].value as number);
      }
    }
    if (n.payload === undefined && scalars === null) continue;
    const payloadHash =
      n.payload === undefined
        ? 0
        : hashValue(0x811c9dc5, n.payload, 0) >>> 0 || 1;
    image.push({ pos: i, op: n.op, payloadHash, scalars });
  }
  const rec: PlanExecRecord = {
    fp,
    image,
    writes: [],
    writtenPos: new Set(),
    uniforms: [],
    uniformPos: new Set(),
    scalarWrites: [],
    compiled: false,
  };
  cur.plans.push(rec);
  cur.entries.push({ kind: "plan", templateId: fp });
  activePlan = rec;
}

export function stEndPlan(): void {
  activePlan = null;
}

/** The active plan executed via compiled replay (executeCompiledPlan). */
export function stMarkPlanCompiled(): void {
  if (activePlan) activePlan.compiled = true;
}

/** A TAG_UNIFORM volatile config re-packed from the current node payload. */
export function stRecordUniform(pos: number, node: { op: string }): void {
  const p = activePlan;
  if (!p) return;
  if (!p.uniformPos.has(pos)) {
    p.uniformPos.add(pos);
    p.uniforms.push({ pos, op: node.op });
  }
  cur.entries.push({ kind: "write", slot: `u:${hex(p.fp)}:${pos}` });
}

/** A TAG_WRITE executed (compiled replay; stable fast path or legacy). */
export function stRecordWrite(
  pos: number,
  node: { op: string; shape: number[]; dtype: string },
  stable: boolean,
): void {
  const p = activePlan;
  if (!p) return;
  p.writes.push({
    pos,
    op: node.op,
    shape: node.shape.slice(),
    dtype: node.dtype,
    stable,
  });
  p.writtenPos.add(pos);
  cur.entries.push({ kind: "write", slot: `w:${hex(p.fp)}:${pos}` });
}

/** Scalar-table coverage declaration (every refresh; deduped per fp). */
export function stDeclareScalarSlots(
  slots: ReadonlyArray<{ nodeIndex: number; inputIndex: number }>,
): void {
  const p = activePlan;
  if (!p || declaredScalarSlots.has(p.fp)) return;
  const set = new Set<string>();
  for (const s of slots) set.add(`${s.nodeIndex}:${s.inputIndex}`);
  declaredScalarSlots.set(p.fp, set);
}

/** A scalar-table buffer was actually written (value changed). Recorded as a
 *  write entry but EXCLUDED from the structural signature: scalar writes are
 *  value-conditional by design (the table only writes on change), so their
 *  presence must not make two otherwise-identical steps structurally
 *  different — the DECLARED slot set is the coverage, not the write. */
export function stRecordScalarWrite(
  pos: number,
  inputIndex: number,
  value: number,
): void {
  const p = activePlan;
  if (!p) return;
  p.scalarWrites.push(pos, inputIndex, value);
  cur.entries.push({
    kind: "write",
    slot: `sc:${hex(p.fp)}:${pos}:${inputIndex}`,
  });
}

/** A readback ran (engine seam; readTopK is backend-direct and OUTSIDE plan
 *  replay — §8.4 item 5 — so it is its own tape entry, params included). */
export function stRecordReadback(
  which: "topk" | "read" | "item",
  params?: Record<string, number>,
): void {
  if (replaying) return; // replay steps are invisible to the recorder
  cur.entries.push({ kind: "readback", which, params });
}

/** The 1c replay layer marks a hit replay so the recorder ignores the step. */
export function stMarkReplayStep(): void {
  replaying = true;
}

// ---------------------------------------------------------------------------
// Step finalization: eligibility + refusal (guard 3 at record time)
// ---------------------------------------------------------------------------

function structuralKey(rec: {
  entries: TapeEntry[];
  plans: PlanExecRecord[];
}): string {
  const parts: string[] = [];
  let planIdx = 0;
  for (const e of rec.entries) {
    if (e.kind === "plan") {
      // C/L: a lowered execution of the same template is a different step
      // shape (no TAG_WRITE/TAG_UNIFORM stream) — never pair it with a
      // compiled one.
      parts.push(
        `p:${hex(e.templateId)}:${rec.plans[planIdx++]?.compiled ? "C" : "L"}`,
      );
    } else if (e.kind === "write") {
      if (!e.slot.startsWith("sc:")) parts.push(e.slot);
    } else {
      parts.push(
        `r:${e.which}:${e.params ? Object.keys(e.params).sort().map((k) => `${k}=${e.params![k]}`).join(",") : ""}`,
      );
    }
  }
  return parts.join("|");
}

function refuse(diag: string): void {
  refusals++;
  if (refusalDiagnostics.length < MAX_DIAGNOSTICS) refusalDiagnostics.push(diag);
  if (refusals <= MAX_WARNINGS) {
    console.warn(`[step-tape] REFUSED: ${diag}`);
  }
}

/** Diff prev vs cur images. Returns the varying DECLARED scalar slots, or
 *  null if any undeclared variance was found (tape refused). */
function diffImages(
  prevPlans: PlanExecRecord[],
  curPlans: PlanExecRecord[],
): { scalarVarying: Array<{ fp: number; pos: number; inputIndex: number; op: string }> } | null {
  const scalarVarying: Array<{
    fp: number;
    pos: number;
    inputIndex: number;
    op: string;
  }> = [];
  let clean = true;
  for (let k = 0; k < curPlans.length; k++) {
    const a = prevPlans[k];
    const b = curPlans[k];
    if (a.image.length !== b.image.length) {
      refuse(
        `plan[${k}] fp=0x${hex(b.fp)}: image length ${a.image.length}→${b.image.length} under identical fingerprint (recorder bug or fingerprint gap)`,
      );
      return null;
    }
    const declared = declaredScalarSlots.get(b.fp);
    for (let i = 0; i < b.image.length; i++) {
      const ia = a.image[i];
      const ib = b.image[i];
      if (ia.pos !== ib.pos || ia.op !== ib.op) {
        refuse(
          `plan[${k}] fp=0x${hex(b.fp)} image[${i}]: position/op misaligned (${ia.op}@${ia.pos} vs ${ib.op}@${ib.pos}) under identical fingerprint`,
        );
        return null;
      }
      if (ia.payloadHash !== ib.payloadHash) {
        // Payload variance is declared iff this node position is carried as
        // data in BOTH recording steps: a TAG_WRITE upload or a TAG_UNIFORM
        // volatile repack (§2.1 sources "upload"/"payload").
        const coveredB = b.writtenPos.has(ib.pos) || b.uniformPos.has(ib.pos);
        const coveredA = a.writtenPos.has(ia.pos) || a.uniformPos.has(ia.pos);
        if (!(coveredA && coveredB)) {
          refuse(
            `plan[${k}] fp=0x${hex(b.fp)} node[${ib.pos}] op=${ib.op}: PAYLOAD varies step→step but position is not TAG_WRITE-covered (undeclared variance — the PAYLOAD-THRASH sibling)`,
          );
          clean = false;
        }
        // NO continue: a node can carry BOTH a payload and scalar refs — the
        // scalar comparison below must still run.
      }
      const sa = ia.scalars;
      const sb = ib.scalars;
      if (sa === null && sb === null) continue;
      if (sa === null || sb === null || sa.length !== sb.length) {
        refuse(
          `plan[${k}] fp=0x${hex(b.fp)} node[${ib.pos}] op=${ib.op}: scalar-ref arity changed under identical fingerprint`,
        );
        return null;
      }
      for (let j = 0; j < sb.length; j += 2) {
        const inputIndex = sb[j];
        if (Object.is(sa[j + 1], sb[j + 1])) continue;
        if (declared?.has(`${ib.pos}:${inputIndex}`)) {
          scalarVarying.push({ fp: b.fp, pos: ib.pos, inputIndex, op: ib.op });
        } else {
          refuse(
            `plan[${k}] fp=0x${hex(b.fp)} node[${ib.pos}] op=${ib.op} input[${inputIndex}]: SCALAR value varies (${sa[j + 1]} → ${sb[j + 1]}) but is not a scalar-table slot (undeclared variance)`,
          );
          clean = false;
        }
      }
    }
  }
  return clean ? { scalarVarying } : null;
}

function buildSlots(
  rec: { plans: PlanExecRecord[] },
  scalarVarying: Array<{ fp: number; pos: number; inputIndex: number; op: string }>,
): DynamicSlot[] {
  const slots: DynamicSlot[] = [];
  const seen = new Set<string>();
  for (const p of rec.plans) {
    for (const w of p.writes) {
      const id = `w:${hex(p.fp)}:${w.pos}`;
      if (seen.has(id)) continue;
      seen.add(id);
      slots.push({
        id,
        name: `${w.op}[${w.shape.join(",")}]`,
        shape: w.shape,
        dtype: w.dtype,
        source: "upload",
        stable: w.stable,
      });
    }
  }
  for (const p of rec.plans) {
    for (const u of p.uniforms) {
      const id = `u:${hex(p.fp)}:${u.pos}`;
      if (seen.has(id)) continue;
      seen.add(id);
      slots.push({
        id,
        name: `${u.op}.uniform`,
        shape: [],
        dtype: "u32",
        source: "payload",
      });
    }
  }
  for (const s of scalarVarying) {
    const id = `sc:${hex(s.fp)}:${s.pos}:${s.inputIndex}`;
    if (seen.has(id)) continue;
    seen.add(id);
    slots.push({
      id,
      name: `${s.op}.scalar[${s.inputIndex}]`,
      shape: [],
      dtype: "f32",
      source: "scalar",
    });
  }
  return slots;
}

/** Step boundary (frontend markStep, after sweeps/fence). */
export function stEndStep(info: {
  opSeq: number;
  epoch: number;
  stepScopedCleanup: boolean;
}): void {
  if (replaying) {
    // A replay step: discard whatever leaked into `cur` and leave `prev`
    // (the last real recorded step) intact for the next normal step.
    replaying = false;
    cur = { entries: [], plans: [] };
    activePlan = null;
    return;
  }
  if (cur.entries.length === 0) return; // empty interval — not a step
  stepsObserved++;
  stepOrdinal++;
  const rec: StepRecord = {
    entries: cur.entries,
    plans: cur.plans,
    structKey: structuralKey(cur),
    opSeqEnd: info.opSeq,
    epoch: info.epoch,
    stepScopedCleanup: info.stepScopedCleanup,
  };
  cur = { entries: [], plans: [] };
  const p = prev;
  prev = rec;
  if (!p) return;
  if (
    p.structKey !== rec.structKey ||
    p.stepScopedCleanup !== rec.stepScopedCleanup
  ) {
    structureMisses++;
    return;
  }
  // A step with no plan executions (readback-only interval) has nothing a
  // tape could replay — never store a degenerate plan-less tape.
  if (rec.plans.length === 0) return;
  // Only fully-compiled steps are replayable (see PlanExecRecord.compiled).
  // The structKey encodes C/L per plan, so a matching pair is uniformly
  // compiled or uniformly lowered; the lowered-lowered warmup pair of a
  // fresh template is counted here, never refused.
  if (!rec.plans.every((pl) => pl.compiled)) {
    loweredPairs++;
    return;
  }
  // Two consecutive structurally-identical compiled steps: guard-3 byte-diff.
  const diff = diffImages(p.plans, rec.plans);
  if (!diff) return; // refused (counted + diagnosed)
  eligiblePairs++;
  const bucketKey = `b:${fnvStr(rec.structKey).toString(16)}:${rec.plans.map((pl) => hex(pl.fp)).join("+")}`;
  const templateIds = new Set<number>();
  for (const pl of rec.plans) templateIds.add(pl.fp);
  tapes.set(bucketKey, {
    bucketKey,
    entries: rec.entries,
    slots: buildSlots(rec, diff.scalarVarying),
    epoch: rec.epoch,
    structGen: rec.opSeqEnd - p.opSeqEnd,
    regime: { stepScopedCleanup: rec.stepScopedCleanup },
    templateIds,
    recordedAtStep: stepOrdinal,
  });
  lastEligible = {
    bucketKey,
    fps: [...templateIds],
    scalarSlots: diff.scalarVarying.map((s) => ({
      pos: s.pos,
      inputIndex: s.inputIndex,
    })),
  };
}

/** Consume the just-finalized step's eligibility (1c skeleton promotion).
 *  Returns null unless the LAST stEndStep produced an eligible tape; clears
 *  the flag so each eligible step promotes at most one skeleton. */
export function stConsumeLastEligible(): {
  bucketKey: string;
  fps: number[];
  scalarSlots: Array<{ pos: number; inputIndex: number }>;
} | null {
  const e = lastEligible;
  lastEligible = null;
  return e;
}

// ---------------------------------------------------------------------------
// Invalidation stubs (guards 4/5 — consumed by 1c)
// ---------------------------------------------------------------------------

/** A compiled plan was destroyed/invalidated: drop tapes referencing its
 *  template and reset the comparator if the previous step used it. */
export function stInvalidateTemplate(fp: number): void {
  for (const [key, tape] of tapes) {
    if (tape.templateIds.has(fp)) {
      tapes.delete(key);
      planInvalidations++;
    }
  }
  declaredScalarSlots.delete(fp);
  if (prev?.plans.some((p) => p.fp === fp)) prev = null;
}

/** Boundary-regime perturbation (explicit beginStep/endStep, stepScoped
 *  toggle): the consecutive-step comparator resets — the next tape must be
 *  re-established from two fresh steps under the new regime (guard 5). */
const boundaryReasons: Record<string, number> = {};
export function stNoteBoundary(reason: string): void {
  if (cur.entries.length === 0 && prev === null) return;
  boundaryResets++;
  boundaryReasons[reason] = (boundaryReasons[reason] ?? 0) + 1;
  prev = null;
  cur = { entries: [], plans: [] };
  activePlan = null;
}

// ---------------------------------------------------------------------------
// Accessors (drivers/tests; 1c consumes the tapes)
// ---------------------------------------------------------------------------

export function stGetTapes(): ReadonlyMap<string, StepTape> {
  return tapes;
}

export function stStats(): {
  stepsObserved: number;
  eligiblePairs: number;
  refusals: number;
  structureMisses: number;
  loweredPairs: number;
  boundaryResets: number;
  boundaryReasons: Record<string, number>;
  planInvalidations: number;
  tapeCount: number;
  refusalDiagnostics: string[];
} {
  return {
    stepsObserved,
    eligiblePairs,
    refusals,
    structureMisses,
    loweredPairs,
    boundaryResets,
    boundaryReasons: { ...boundaryReasons },
    planInvalidations,
    tapeCount: tapes.size,
    refusalDiagnostics: refusalDiagnostics.slice(),
  };
}

export function stResetAll(): void {
  tapes.clear();
  declaredScalarSlots.clear();
  cur = { entries: [], plans: [] };
  activePlan = null;
  prev = null;
  stepOrdinal = 0;
  refusals = 0;
  structureMisses = 0;
  loweredPairs = 0;
  eligiblePairs = 0;
  stepsObserved = 0;
  boundaryResets = 0;
  planInvalidations = 0;
  refusalDiagnostics.length = 0;
  lastEligible = null;
  replaying = false;
}
