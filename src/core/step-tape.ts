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
import {
  deriveStepObject,
  type StepObject,
  type StepReceipts,
} from "./step-object";

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
  /** Fingerprints of the plans that carried a checkpoint boundary this step
   *  (task #98 phase 3): the DECLARED recompute segments, in plan order. The
   *  step object's `recomputeRef` is derived from this — a real recompute fact,
   *  distinct from the all-fps `partitionRef`. Empty when the step is not
   *  checkpointed. */
  recomputeFps: number[];
  /** Per-plan islands partition-identity token (task #98 phase 6, ruling 4),
   *  ALIGNED with the ordered plan fps. Each entry is the executor's already-
   *  computed `PlanPartition.boundaryHash` (islands I1, `executor.ts:259`) for
   *  that plan — the detector OWNS membership; this is a read-only PROJECTION,
   *  no second owner. The step object's `partition` facet (`StepPartition`) is
   *  derived from this + the ordered fps. It hashes into NEITHER identity: the
   *  boundaryHash is ALREADY mixed into each plan's fp (I1), so the fp sequence
   *  already carries the partition; this field just surfaces the per-plan token
   *  so the phase-6 null test can assert the projection reproduces it
   *  byte-identically. 0 for a plan with no reified partition (arena-free /
   *  lowered checkpoint plans). */
  partitionHashes: number[];
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
      h = mix(
        h,
        u8[i] | (u8[i + 1] << 8) | (u8[i + 2] << 16) | (u8[i + 3] << 24),
      );
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
  /** Primary result (derived view of results[0]); defined ⇒ the node's value
   *  pre-exists this plan's execution (produced by an earlier plan). */
  result?: unknown;
  /** Checkpoint-segment boundary marker (task #98 phase 3, ruling 3). A plan
   *  carrying any is a recompute-bearing plan — a declared recompute segment. */
  isCheckpointBoundary?: boolean;
}

/** One node's payload/scalar image (only nodes with payload or scalar refs). */
interface NodeImage {
  pos: number;
  op: string;
  payloadHash: number; // 0 = no payload
  /** Flat [inputIndex, value, ...] pairs for scalar-kind refs. */
  scalars: number[] | null;
  /** The node's result PRE-EXISTED at plan begin (shared-node/external class:
   *  this plan never executes it — consumers resolve its CURRENT result buffer
   *  per replay, so its payload is dead HERE; the plan that executes it
   *  carries the payload as a TAG_WRITE and is diffed on its own). Payload
   *  variance at a hadResult position (in BOTH compared steps) is therefore
   *  carried-as-data, not undeclared. [2b G-cover, upload class] */
  hadResult: boolean;
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
  /** True iff this plan carries a checkpoint boundary (task #98 phase 3): it is
   *  a recompute-bearing plan — a DECLARED recompute segment of the step. The
   *  step object's `recomputeRef` is derived from the fps of such plans, making
   *  the recompute facet a real declared fact, not a placeholder over all fps. */
  hasRecompute: boolean;
  /** The executor's already-computed islands partition-identity token for this
   *  plan (task #98 phase 6, ruling 4): `PlanPartition.boundaryHash` (I1). A
   *  read-only PROJECTION of the detector's membership — recorded, never
   *  recomputed here. 0 when no partition was reified (lowered/arena-free plan). */
  partitionHash: number;
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

/** Declared batch-representative coverage per template fp: member node
 *  position → representative node position. A batched op (adam-batch) packs
 *  many nodes into dispatches whose per-step-varying config is TAG_UNIFORM-
 *  repacked from ONE representative node's payload; the member nodes'
 *  payloads are dead at replay. Member payload variance is covered IFF the
 *  representative is TAG_UNIFORM-covered in both compared steps AND the
 *  member's payloadHash EQUALS the representative's within each step — the
 *  agreement assert that makes a batch member with a DIVERGENT config (a
 *  per-group LR that stopped matching the representative) a LOUD refusal
 *  instead of a silent wrong-config replay. [2b G-cover, optimizer class] */
const declaredBatchCover = new Map<number, Map<number, number>>();

let cur: { entries: TapeEntry[]; plans: PlanExecRecord[] } = {
  entries: [],
  plans: [],
};
let activePlan: PlanExecRecord | null = null;

/**
 * [task #98 phase 4 — WITNESS-TIME HARVEST] The cross-plan reads PHYSICALLY
 * OBSERVED during THIS step, keyed by producer template fp → set of "ni:oi"
 * pairs read. Populated by `stObserveWitnessRead` at the cross-plan read seam
 * (`getInputStorage` on a stamped storage — the LOWERED read `observeConsumed`
 * is blind to, §4.1). A step RUNS the whole program (forward + backward +
 * checkpoint recompute), so this set includes the checkpoint-recompute
 * activation read that the generated harvest prunes. At step end it is
 * reconciled into the persistent per-producer `witnessProducer` tracker (K_w=2
 * per producer), which publishes to observed-liveness. Cleared per step. */
let curWitnessReads = new Map<number, Set<string>>();
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
  /** DEDUPED template fps (guard-4 invalidation index). */
  fps: number[];
  /** ORDERED plan execution fp sequence — the multi-plan skeleton must match
   *  candidates to this order (2b surface 1). May repeat fps (a template can
   *  execute more than once per step); NEVER deduped. */
  orderedFps: number[];
  /** Scalar slots carry their plan fp (2b surface 1: route each to the right
   *  plan of a multi-plan step). */
  scalarSlots: Array<{ fp: number; pos: number; inputIndex: number }>;
} | null = null;
let loweredPairs = 0;
let eligiblePairs = 0;
let stepsObserved = 0;
let boundaryResets = 0;
let planInvalidations = 0;
/** [task #98 phase 4] Count of producer observations where THIS step's observed
 *  cross-plan read set DISAGREED with the previous step's for the same producer
 *  (§4.1 rule 3). Zero at steady state — a nonzero count is a nondeterministic-
 *  reader signal (the conservative UNION is still published, so it never ships a
 *  wrong prune; the counter makes the nondeterminism visible for shadow-parity). */
let witnessVariances = 0;
const refusalDiagnostics: string[] = [];
const MAX_DIAGNOSTICS = 32;
const MAX_WARNINGS = 8;

function hex(fp: number): string {
  return (fp >>> 0).toString(16);
}

/** [task #98 phase 4 — WITNESS-TIME HARVEST] Per PRODUCER-template `fp`, the
 *  previous step's observed read set + how many CONSECUTIVE steps it has been
 *  read with that exact set. Keyed by PRODUCER (not the whole-step structure):
 *  under selective checkpointing the recompute READER plans are freshly-
 *  fingerprinted every backward (so no two whole steps are structurally
 *  identical, `stage4 §Task #97` — the reason the whole-step eligibility gate
 *  can never fire on the checkpoint config), but the PRODUCER template of the
 *  cross-plan activation (`0xc98a72f3`, the checkpointed-MLP `contiguous[512,768]`
 *  producer) recurs identically every step. The witness window is therefore
 *  PER PRODUCER: two consecutive steps in which THIS producer was read with the
 *  SAME pair set (§10 ruling 2, K_w=2, applied at the producer stratum). */
const witnessProducer = new Map<
  number,
  { reads: Set<string>; consecutive: number }
>();

/** [task #98 phase 4 — WITNESS-TIME HARVEST] Reconcile THIS step's observed
 *  cross-plan reads (`curW`: producer fp → pair set) against the per-producer
 *  witness tracker, and publish a producer's witnessed harvest once it has been
 *  read with the SAME set in two consecutive steps (K_w=2). Called EVERY step
 *  the recorder finalizes a NORMAL step (not gated on whole-step tape
 *  eligibility — the checkpoint config never reaches that, see above).
 *
 *  Publication is monotone-safe: once a producer's set is published it is only
 *  ever GROWN (a later step observing a new pair republishes the UNION), never
 *  shrunk — a superset keep never causes a UAF (pruning too FEW never crashes),
 *  so the tracker can trust the first two-consecutive agreement and extend it. A
 *  DISAGREEMENT (the producer read a different set than last step) resets the
 *  consecutive count AND, if a set was already published, republishes the UNION
 *  (never drop a witnessed pair) and counts the variance (nondeterministic
 *  reader diagnostic, §4.1 rule 3). */
function reconcileWitnessReads(curW: Map<number, Set<string>>): void {
  if (!witnessHarvestPublisher) return;
  for (const [fp, reads] of curW) {
    const tr = witnessProducer.get(fp);
    if (!tr) {
      // First observation of this producer: seed, not yet publishable (K_w=2).
      witnessProducer.set(fp, { reads: new Set(reads), consecutive: 1 });
      continue;
    }
    const same =
      reads.size === tr.reads.size && [...reads].every((k) => tr.reads.has(k));
    if (same) {
      tr.consecutive++;
      // Two (or more) consecutive identical observations → witnessed. Publish
      // (idempotent — same set each time once stable).
      if (tr.consecutive >= 2) witnessHarvestPublisher(fp, reads);
    } else {
      // The read set changed between two consecutive appearances. Grow the
      // tracker to the UNION (monotone), reset the consecutive count, and — if
      // already published — republish the union so no witnessed pair is dropped.
      const union = new Set<string>(tr.reads);
      for (const k of reads) union.add(k);
      const wasPublished = tr.consecutive >= 2;
      const grew = union.size !== tr.reads.size;
      tr.reads = union;
      tr.consecutive = 1;
      if (grew) {
        witnessVariances++;
        if (refusalDiagnostics.length < MAX_DIAGNOSTICS) {
          refusalDiagnostics.push(
            `[step-tape] witness-variance fp=0x${hex(fp)} grew to ${union.size} pairs (conservative union${wasPublished ? " republished" : ""})`,
          );
        }
      }
      if (wasPublished && grew) witnessHarvestPublisher(fp, union);
    }
  }
}

// ---------------------------------------------------------------------------
// Recording seams (each call site guarded by STEP_TAPE_RECORD)
// ---------------------------------------------------------------------------

/** Plan execution begins (executor, after fingerprinting, permuted order).
 *  Captures the full payload/scalar byte-image for the guard-3 diff. */
export function stBeginPlan(
  fp: number,
  planNodes: readonly NodeLike[],
  partitionHash = 0,
): void {
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
    image.push({
      pos: i,
      op: n.op,
      payloadHash,
      scalars,
      hadResult: n.result !== undefined,
    });
  }
  let hasRecompute = false;
  for (let i = 0; i < planNodes.length; i++) {
    if (planNodes[i].isCheckpointBoundary) {
      hasRecompute = true;
      break;
    }
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
    hasRecompute,
    partitionHash: partitionHash >>> 0,
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

/** Batched-op representative coverage (executor adam-batch seam): the batch's
 *  per-step-varying configs are TAG_UNIFORM-repacked from `rep`'s payload at
 *  replay; `members` are the node positions the batch executed. Re-declared
 *  (overwritten) on every lowered/recording execution of the plan — the
 *  declaration persists per fp so the compiled-pair diff (which never runs
 *  the action) can consult it. */
export function stDeclareBatchCover(members: number[], rep: number): void {
  const p = activePlan;
  if (!p) return;
  let m = declaredBatchCover.get(p.fp);
  if (!m) {
    m = new Map();
    declaredBatchCover.set(p.fp, m);
  } else {
    m.clear();
  }
  for (const pos of members) m.set(pos, rep);
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

/**
 * [task #98 phase 4 — WITNESS-TIME HARVEST] Record a cross-plan read of a
 * STAMPED producer result during a recording step (§4.1 rule 2). Called from
 * the cross-plan read seam (`getInputStorage`) for every stamped storage a plan
 * reads — critically INCLUDING the LOWERED reads `observeConsumed` never sees
 * (the backward pass re-reading a checkpoint-recompute forward activation, the
 * #97 STOP class). The (producerFp, ni, oi) coordinate is exactly the harvest
 * pair coordinate, so the witnessed set can be unioned into the generated
 * harvest's keep set verbatim.
 *
 * Blind during a REPLAY step (a replay executes the tape, not the observed
 * normal path — its reads must not corrupt the witness set the next NORMAL step
 * builds). Cheap: one Map/Set insert, gated by `activePlan` (a read outside a
 * recorded plan — warmup, readback — is not a cross-plan harvest read here). */
export function stObserveWitnessRead(fp: number, ni: number, oi: number): void {
  if (replaying) return;
  if (!activePlan) return;
  let s = curWitnessReads.get(fp);
  if (!s) curWitnessReads.set(fp, (s = new Set()));
  s.add(`${ni}:${oi}`);
}

/** Publisher callback (wired by the executor layer at load — core stays a leaf,
 *  same pattern as observed-liveness's own invalidator callbacks). Delivers a
 *  producer template's WITNESSED harvest set to observed-liveness on eligibility.
 *
 *  The witnessed set PERSISTS across a template's build-from-IR rebuild (an
 *  intrinsic property of the workload, exactly like observed-liveness's own
 *  needed-set — the `templates` map is never cleared on invalidation): only a
 *  fresh eligibility REPLACES it. So there is deliberately no clear-on-invalidate
 *  callback — clearing on rebuild would starve the rebuilt template of its
 *  witnessed keep set until it re-witnessed, re-opening the prune window. */
let witnessHarvestPublisher:
  | ((fp: number, pairs: Set<string>) => void)
  | undefined;
export function setWitnessHarvestPublisher(
  fn: (fp: number, pairs: Set<string>) => void,
): void {
  witnessHarvestPublisher = fn;
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
        `r:${e.which}:${
          e.params
            ? Object.keys(e.params)
                .sort()
                .map((k) => `${k}=${e.params![k]}`)
                .join(",")
            : ""
        }`,
      );
    }
  }
  return parts.join("|");
}

function refuse(diag: string): void {
  refusals++;
  if (refusalDiagnostics.length < MAX_DIAGNOSTICS)
    refusalDiagnostics.push(diag);
  if (refusals <= MAX_WARNINGS) {
    console.warn(`[step-tape] REFUSED: ${diag}`);
  }
}

/** Image entry at a node position (entries are in increasing-pos order). */
function imageAt(p: PlanExecRecord, pos: number): NodeImage | undefined {
  const img = p.image;
  let lo = 0;
  let hi = img.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const v = img[mid].pos;
    if (v === pos) return img[mid];
    if (v < pos) lo = mid + 1;
    else hi = mid - 1;
  }
  return undefined;
}

/** Diff prev vs cur images. Returns the varying DECLARED scalar slots, or
 *  null if any undeclared variance was found (tape refused). */
function diffImages(
  prevPlans: PlanExecRecord[],
  curPlans: PlanExecRecord[],
): {
  scalarVarying: Array<{
    fp: number;
    pos: number;
    inputIndex: number;
    op: string;
  }>;
} | null {
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
        // data in BOTH recording steps, by one of (§2.1 + 2b G-cover):
        //  1. TAG_WRITE upload or TAG_UNIFORM volatile repack at the position
        //     itself (sources "upload"/"payload");
        //  2. dead payload — the node's result PRE-EXISTED this plan in BOTH
        //     steps (shared-node/external class): this plan never reads the
        //     payload; consumers resolve the producer plan's current result
        //     buffer per replay, and the producer plan's own diff covers the
        //     payload as its TAG_WRITE;
        //  3. batch-representative — a declared batch member whose config is
        //     repacked from the representative's payload at replay, with the
        //     member↔representative payloadHash agreement assert per step.
        const coveredB = b.writtenPos.has(ib.pos) || b.uniformPos.has(ib.pos);
        const coveredA = a.writtenPos.has(ia.pos) || a.uniformPos.has(ia.pos);
        if (!(coveredA && coveredB)) {
          if (ia.hadResult && ib.hadResult) {
            // Rule 2: dead payload in this plan (both steps) — covered.
          } else {
            const rep = declaredBatchCover.get(b.fp)?.get(ib.pos);
            const repA = rep !== undefined ? imageAt(a, rep) : undefined;
            const repB = rep !== undefined ? imageAt(b, rep) : undefined;
            const batchCovered =
              rep !== undefined &&
              repA !== undefined &&
              repB !== undefined &&
              a.uniformPos.has(rep) &&
              b.uniformPos.has(rep) &&
              repA.payloadHash === ia.payloadHash &&
              repB.payloadHash === ib.payloadHash;
            if (!batchCovered) {
              refuse(
                rep !== undefined
                  ? `plan[${k}] fp=0x${hex(b.fp)} node[${ib.pos}] op=${ib.op}: PAYLOAD varies and batch-representative coverage failed (rep=${rep} uniform-covered=${a.uniformPos.has(rep) && b.uniformPos.has(rep)}, payload agreement=${repA?.payloadHash === ia.payloadHash && repB?.payloadHash === ib.payloadHash}) — a batch member whose config diverged from its representative would replay WRONG; refusing`
                  : `plan[${k}] fp=0x${hex(b.fp)} node[${ib.pos}] op=${ib.op}: PAYLOAD varies step→step but position is not TAG_WRITE-covered (undeclared variance — the PAYLOAD-THRASH sibling)`,
              );
              clean = false;
            }
          }
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
  scalarVarying: Array<{
    fp: number;
    pos: number;
    inputIndex: number;
    op: string;
  }>,
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
    curWitnessReads = new Map();
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
  // [task #98 phase 4 — WITNESS-TIME HARVEST] Reconcile THIS step's observed
  // cross-plan reads into the per-producer witness tracker and publish any
  // producer witnessed with a stable read set (§4.1). Done for EVERY finalized
  // NORMAL step, keyed by PRODUCER template — NOT gated on whole-step structural
  // identity or tape-replay eligibility. This is load-bearing for the #97
  // checkpoint config: selective-checkpointing steps run recompute segments
  // LOWERED and re-fingerprint the READER plans every backward, so no two whole
  // steps are structurally identical and the whole-step eligibility gate below
  // NEVER fires — yet the PRODUCER template of the cross-plan activation recurs
  // identically every step, and those steps DID run the whole program (forward +
  // backward + recompute), so their reads (incl. the lowered backward read of a
  // checkpoint-recompute activation) are the witnessed harvest the generated
  // prune needs. Rolling the accumulator here empties it for the next step.
  const curWitness = curWitnessReads;
  curWitnessReads = new Map();
  reconcileWitnessReads(curWitness);
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
  // Recompute segments (task #98 phase 3): the fps of plans that carried a
  // checkpoint boundary, in plan order — the DECLARED recompute facet.
  const recomputeFps = rec.plans
    .filter((pl) => pl.hasRecompute)
    .map((pl) => pl.fp);
  // Per-plan islands partition tokens (task #98 phase 6), aligned with the
  // ordered plan fps — the read-only projection of the detector's membership.
  const partitionHashes = rec.plans.map((pl) => pl.partitionHash);
  tapes.set(bucketKey, {
    bucketKey,
    entries: rec.entries,
    slots: buildSlots(rec, diff.scalarVarying),
    epoch: rec.epoch,
    structGen: rec.opSeqEnd - p.opSeqEnd,
    regime: { stepScopedCleanup: rec.stepScopedCleanup },
    templateIds,
    recomputeFps,
    partitionHashes,
    recordedAtStep: stepOrdinal,
  });
  lastEligible = {
    bucketKey,
    fps: [...templateIds],
    orderedFps: rec.plans.map((pl) => pl.fp),
    scalarSlots: diff.scalarVarying.map((s) => ({
      fp: s.fp,
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
  orderedFps: number[];
  scalarSlots: Array<{ fp: number; pos: number; inputIndex: number }>;
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
  // declaredBatchCover deliberately SURVIVES template invalidation: a
  // build-from-IR rebuild (staleness → invalidate → rebuild) never re-runs
  // the executor actions, so the adam-batch seam cannot re-declare — deleting
  // here would starve the rebuilt template's coverage forever (measured: the
  // optimizer template rebuilt ~step 11 and every later pair refused). Safe
  // because the declaration is a HINT verified at every use: the diff requires
  // the representative to be TAG_UNIFORM-covered in both compared steps AND
  // member↔representative payload agreement per step — a stale entry fails
  // those checks and refuses loudly, never replays wrong.
  if (prev?.plans.some((p) => p.fp === fp)) prev = null;
}

/** Boundary-regime perturbation (explicit beginStep/endStep, stepScoped
 *  toggle): the consecutive-step comparator resets — the next tape must be
 *  re-established from two fresh steps under the new regime (guard 5). */
const boundaryReasons: Record<string, number> = {};
let ceremonyWarned = false;
let ceremonyResetsWhileWarm = 0;
export function stNoteBoundary(reason: string): void {
  if (cur.entries.length === 0 && prev === null) return;
  boundaryResets++;
  boundaryReasons[reason] = (boundaryReasons[reason] ?? 0) + 1;
  // [#81] Ceremony-starvation diagnostic (task #81, canonical trigger): a
  // WARM tape store hit by an explicit-ceremony comparator reset means the
  // loop mixes beginStep/endStep with a captured/taped region — the tape will
  // re-trace forever (silent perf starvation, the endStep-remnant class).
  // One-time, loud, names the reason; counted for tests/telemetry.
  if ((reason === "beginStep" || reason === "endStep") && tapes.size > 0) {
    ceremonyResetsWhileWarm++;
    if (!ceremonyWarned) {
      ceremonyWarned = true;
      console.warn(
        `[step-tape] STARVATION: explicit ${reason}() reset the step comparator while ${tapes.size} tape(s) were warm — captured regions will re-trace every call and never replay. Remove beginStep/endStep ceremony from the captured loop (minimal implied-boundary loops are the captured regime; docs/staged-execution-phase2b.md).`,
      );
    }
  }
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

/**
 * [step-object phase 1] DERIVE the StepObject for each witnessed tape (task #98
 * §2.1). This is the single-source read-over the design mandates: the StepObject
 * is a PROJECTION over the recorder's tape store + the live receipt counters —
 * NO second owner (docs/step-object-design.md §2, ruling 1). Reifies, changes
 * NOTHING (the recorder still owns the tapes; this is a read-only view).
 *
 * Receipts hash into NEITHER identity (§2.5); they are the SAME live counters
 * `stStats()` exposes — one source, two shapes.
 */
export function stDeriveStepObjects(): StepObject[] {
  const receipts: StepReceipts = {
    refusals,
    eligiblePairs,
    structureMisses,
    planInvalidations,
    boundaryResets,
  };
  const objs: StepObject[] = [];
  for (const tape of tapes.values())
    objs.push(deriveStepObject(tape, receipts));
  return objs;
}

/**
 * [step-object phase 1] DERIVE the StepObject for ONE tape by bucketKey — the
 * per-step identity lookup (the digest IS the bucketKey, §2.2). Returns null if
 * no tape is witnessed under that key.
 */
export function stDeriveStepObject(bucketKey: string): StepObject | null {
  const tape = tapes.get(bucketKey);
  if (!tape) return null;
  return deriveStepObject(tape, {
    refusals,
    eligiblePairs,
    structureMisses,
    planInvalidations,
    boundaryResets,
  });
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
  /** [#81] ceremony resets that hit a WARM tape store (starvation trigger). */
  ceremonyResetsWhileWarm: number;
  /** [task #98 phase 4] witness-set disagreements between paired witness steps. */
  witnessVariances: number;
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
    ceremonyResetsWhileWarm,
    witnessVariances,
    refusalDiagnostics: refusalDiagnostics.slice(),
  };
}

export function stResetAll(): void {
  tapes.clear();
  declaredScalarSlots.clear();
  declaredBatchCover.clear();
  cur = { entries: [], plans: [] };
  curWitnessReads = new Map();
  witnessProducer.clear();
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
  witnessVariances = 0;
  refusalDiagnostics.length = 0;
  lastEligible = null;
  replaying = false;
  ceremonyWarned = false;
  ceremonyResetsWhileWarm = 0;
}
