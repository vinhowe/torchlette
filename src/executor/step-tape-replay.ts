/**
 * Step-tape phase-1c REPLAY (docs/staged-execution-phase1.md §2.3/§2.4, §3
 * G2–G8). The recorder (src/core/step-tape.ts) is a pure observer; THIS module
 * consumes its eligibility verdicts and replays the decode step program.
 *
 * HARVEST DECISION — option (i) SKELETON-GRAPH (§8.2/§8.4 item 3):
 *   Replay re-dresses a RETAINED planNodes array (the skeleton, captured once
 *   from a recording step) with the fresh per-token upload payloads, then calls
 *   the EXISTING `executeLoweredPlan` — the exact byte-for-byte machinery a
 *   cache-hit uses (refreshScalarTable → inlined-scalar staleness → compiled
 *   replay → result harvest → KV `copy_` side effects). Nothing about the plan,
 *   its buffers, the scalar table, or the harvest is re-implemented; the tape
 *   only skips the JS ABOVE `executeLoweredPlan`: lazy graph build, plan
 *   collect, fingerprint, CSE, rewrites, template lookup (14.55 ms of the 1a
 *   accounting). Slot-direct harvest (option ii) was REJECTED: it would fork the
 *   harvest/sweep lifetime and put the KV-cache `copy_` persistence contract at
 *   risk for ~13 ms more — a smaller win with an intact contract wins (§task B).
 *
 * GUARDS live at the replay DECISION point, all value/identity compares, no
 * re-derivation (§2.4):
 *   1 structGen — enforced by the library seam: the driver's appKey is the
 *     structural declaration (generateChat's loop is constant by construction);
 *     a code-path change (hook toggle, collectHidden) changes the appKey → no
 *     matching skeleton → miss. (The recorder's structGen is retained on the
 *     tape for diagnostics.)
 *   2 bucketKey — appKey encodes model + KV bucket length + steering structure.
 *   3 scalar coverage (the frozen-α defense, §8.4 items 1-2) — value-level:
 *     the driver declares its scalar-slot values (α); any change vs the recorded
 *     baseline ⇒ MISS before replay, so the plan never runs with a stale α. The
 *     miss falls back to the normal path, whose inlined-scalar staleness demotes
 *     and re-records (the mechanism 1b's alphachange probe confirmed).
 *   4 plan validity — the skeleton's compiled plan carries its own
 *     staleness/eviction machinery; `compiledPlan.valid` is checked and a
 *     planner-generation change drops the skeleton (stInvalidateTemplate already
 *     nulls the recorder tape; here we drop the replay skeleton symmetrically).
 *   5 epoch/regime — recorded epoch + stepScopedCleanup must match.
 *
 * ANY miss ⇒ normal path for that step (re-records + re-captures) + counter;
 * STRICT_TAPE=1 throws. A silent stale replay is the failure mode this exists
 * to prevent (§2.4).
 *
 * Single-engine by phase-1 scope (module-level state, like the recorder).
 * SUNSET: rides TORCHLETTE_STEP_TAPE (dies at the default-flip decision).
 */

import type { Backend, DeviceKind, DType } from "../backend/types";
import type { LazyIRNode, StorageHandle } from "../graph/types";
import type { BufferArena } from "../backend/webgpu/buffer-arena";
import type { LoweredPlan } from "./lowered-plan";
import type { GpuCommand } from "./compiled-plan";
import { canonicalizeStream } from "./stream-diff";
import {
  STEP_TAPE_REPLAY,
  STEP_TAPE_STRICT,
  STEP_TAPE_VERIFY_N,
  stConsumeLastEligible,
  stGetTapes,
  stMarkReplayStep,
} from "../core/step-tape";
import { ENV } from "../core/env";
import { getScalarSlotValue } from "../core/scalar-slots";
import { storageTracker } from "../graph/storage-tracker";
import { executeLoweredPlan } from "./executor";
import { assignNodeResult, executeOpSync } from "./op-dispatch";
import {
  noteTemplateReplayed,
  setStepTapeReplayActive,
} from "./observed-liveness";

// ---------------------------------------------------------------------------
// Skeleton store (§8.2 option i)
// ---------------------------------------------------------------------------

interface UploadNode {
  node: LazyIRNode;
  /** Shape signature (e.g. "1,1,1,128") — the driver keys fresh payloads by it. */
  shapeKey: string;
  shape: number[];
  /** [2b] True iff this upload node is a captured-fn ARG (batch x/y): re-dressed
   *  per replay from the fresh arg values. False = an internal constant upload
   *  (e.g. zeroGrad's `zeros`) whose recorded payload is stable — left untouched
   *  by the replay. Always true on the decode path (no training args). */
  isArg: boolean;
}

interface ScalarSlot {
  node: LazyIRNode;
  inputIndex: number;
  /** Recorded value (guard-3 baseline). */
  recorded: number;
}

/** One plan of a (possibly multi-plan) skeleton. A decode step has ONE of
 *  these; a training step has forward/backward/optimizer plans in order. */
interface SkeletonPlan {
  fp: number;
  planNodes: LazyIRNode[];
  loweredPlan: LoweredPlan;
  bufferArena: BufferArena;
  uploads: UploadNode[];
  scalars: ScalarSlot[];
  /** Canonical command stream at capture time (verify determinism/faithfulness). */
  canonical: string[];
  /** [2b] Captured BEFORE the captured call's body began this step — a
   *  DRIVER-level side write (the scheduler's lr-tensor write, forced at the
   *  step-opening markStep). The driver re-creates + re-executes these for
   *  REAL every iteration (fresh pending nodes), so the replay must SKIP the
   *  recorded ghost — re-executing it would clobber the fresh value with the
   *  recorded one (the frozen-LR class, caught by the schedule-through-hits
   *  gate). Structure guard unaffected: the ordered-fp match covers the full
   *  sequence. */
  preBody: boolean;
}

/** [2b] A CROSS-PLAN dataflow link: a later plan's input is a MATERIALIZED ref
 *  to a storage that an EARLIER plan's node produced (e.g. the optimizer plan
 *  reading a grad the backward plan wrote — forced/materialized between plans at
 *  recording). The materialized ref freezes the RECORDING buffer, which is
 *  step-scoped and destroyed at markStep → a dangling read on replay. Re-point
 *  it to the producer node's FRESH result each replay after the producer plan
 *  runs. Matched by storage id at promotion. */
interface CrossPlanLink {
  /** The consumer plan's node and input index holding the materialized ref. */
  consumerPlan: number;
  consumerNode: LazyIRNode;
  inputIndex: number;
  /** The producing node (in an earlier plan) whose fresh result to bind. */
  producerNode: LazyIRNode;
  producerOutputIndex: number;
}

interface Skeleton {
  appKey: string;
  /** ORDERED plan sequence (2b surface 1). Length 1 for decode. */
  plans: SkeletonPlan[];
  epoch: number;
  stepScopedCleanup: boolean;
  /** Output-node refs (2b surface 3): (plan index, node pos) of the tensor(s)
   *  the captured body returned. Decode: [{planIndex:0, pos:last}]. */
  outputRefs: Array<{ planIndex: number; pos: number }>;
  /** Cross-plan materialized-ref links (2b). Empty for decode (single plan). */
  crossPlanLinks: CrossPlanLink[];
  /** [2b rebind] Frozen materialized refs whose OWNING persistent tensor is
   *  known: re-resolved to the owner's CURRENT storage each replay (the
   *  frozen-LR class — replace-and-hold writes move a persistent tensor to a
   *  new storage every step, stranding the recording-era ref). */
  rebinds: Array<{ ref: { kind: string; storage: StorageHandle }; owner: TensorOwner }>;
  /** [2b sched] Per-replay payload re-dress of recorded in-plan scalar-write
   *  chains (see Candidate.scalarDresses). */
  scalarDresses: Array<{ writeNode: LazyIRNode; resultId: number }>;
}

/** One captured compiled plan of the current step, keyed by the active appKey;
 *  the per-step accumulator promotes the ordered list iff the recorder deems
 *  the step eligible. */
interface Candidate {
  appKey: string;
  fp: number;
  planNodes: LazyIRNode[];
  loweredPlan: LoweredPlan;
  bufferArena: BufferArena;
  commands: GpuCommand[];
  epoch: number;
  stepScopedCleanup: boolean;
  /** [2b cross-plan] storage-id → (node, outputIndex) snapshot of THIS plan's
   *  produced results, taken while they are alive (at capture) — promotion runs
   *  after the recording sweep, so node.result is unreadable there. Used to
   *  resolve later plans' materialized refs to their cross-plan producer. */
  resultIds: Map<number, { node: LazyIRNode; oi: number }>;
  /** [2b rebind] materialized-ref → owning RuntimeTensor, captured while the
   *  storage is alive (the owner backref dies with the storage). Replays
   *  re-resolve each ref to the owner's CURRENT storage — persistent tensors
   *  updated by replace-and-hold writes (the scheduler's lr copy_ chain) move
   *  to a NEW storage every step, so the recording-era ref would read the
   *  stale buffer forever (the frozen-LR class; caught by the
   *  schedule-through-hits gate). */
  matOwners: Map<object, TensorOwner>;
  /** [2b sched] Recorded IN-PLAN driver scalar-write chains (the lr ghost):
   *  scatter(dst, tensorFromArray([v])) forced INSIDE the step interval when
   *  the driver's write is still lazy at body force time. The optimizer then
   *  reads the scalar THROUGH this chain, so on replay the value is the
   *  recorded upload payload — frozen. Each entry pairs the recorded WRITE
   *  node (payload re-dressed per replay) with the owning persistent tensor
   *  (source of the CURRENT value via its pending chain). */
  scalarDresses: Array<{ writeNode: LazyIRNode; owner: TensorOwner }>;
}

/** Minimal owner surface (the runtime Tensor): its live lazyRef. */
interface TensorOwner {
  lazyRef: { kind: string; storage?: StorageHandle };
}

const readySkeletons = new Map<string, Skeleton>();
/** Per-step ORDERED accumulator of compiled plans under the active appKey
 *  (2b surface 1: a step is N plans, cleared at each promote boundary). */
let candidates: Candidate[] = [];
/** Output-node hints for the just-traced step: the body's returned tensor(s)'
 *  lazy nodes, set by the capture layer before the promote boundary (2b surface
 *  3). Matched by identity to candidate plans at promotion. */
let pendingOutputNodes: LazyIRNode[] = [];
/** [2b surface 4] Arg-node hints: the captured-fn tensor ARGS' lazy nodes (batch
 *  x/y), in ARG ORDER. Upload slots matching these are re-dressed per replay;
 *  all other upload slots keep their recorded (constant) payload. Empty on the
 *  decode path (every upload is then re-dressed, the historical behavior). */
let pendingArgNodes: LazyIRNode[] = [];
/** [2b] Candidate count when the captured call's BODY began this step — plans
 *  captured before it are driver-level pre-body work (see SkeletonPlan.preBody).
 *  -1 = no body marker this step (decode / raw drivers: nothing is preBody). */
let bodyBeginIndex = -1;

// Active per-step context (set by the driver via the frontend before the step).
let ctxAppKey: string | null = null;
let ctxScalars: number[] = [];
let ctxEpoch = 0;
let ctxStepScopedCleanup = false;

// Counters (§6 observability).
const stats = {
  replays: 0,
  hits: 0,
  missNoTape: 0,
  missScalar: 0,
  missEpoch: 0,
  missValidity: 0,
  missShape: 0,
  invalidations: 0,
  verifies: 0,
  verifyDiffs: 0,
  captures: 0,
  promotions: 0,
};
const diffDiagnostics: string[] = [];

function shapeKeyOf(shape: number[]): string {
  return shape.join(",");
}

function strictThrow(msg: string): void {
  if (STEP_TAPE_STRICT) throw new Error(`[step-tape] STRICT: ${msg}`);
}

// ---------------------------------------------------------------------------
// Per-step context + capture (executor seam)
// ---------------------------------------------------------------------------

export function stSetTapeContext(
  appKey: string,
  scalarValues: number[],
  epoch: number,
  stepScopedCleanup: boolean,
): void {
  ctxAppKey = appKey;
  ctxScalars = scalarValues;
  ctxEpoch = epoch;
  ctxStepScopedCleanup = stepScopedCleanup;
}

export function stTapeReadyFor(appKey: string): boolean {
  return readySkeletons.has(appKey);
}

/** [capture 2a] A skeleton exists AND its compiled plan is still valid — the
 *  gate capture() uses before committing to a hit, so a replay decline (guard
 *  4/5) cannot happen AFTER the captured fn's in-place state already advanced.
 *  An invalid plan drops the skeleton here (symmetric with stTryReplay). */
export function stTapeReplayValid(appKey: string): boolean {
  const sk = readySkeletons.get(appKey);
  if (!sk) return false;
  // Multi-plan (2b surface 4 upfront slot-check): EVERY replayed plan's
  // compiled plan must still be valid before we commit to a hit. preBody plans
  // are never replayed (driver-level work) — their template validity is the
  // driver's normal-path concern, not the tape's.
  for (const pl of sk.plans) {
    if (pl.preBody) continue;
    if (!pl.loweredPlan.compiledPlan?.valid) {
      readySkeletons.delete(appKey);
      stats.invalidations++;
      return false;
    }
  }
  return true;
}

/** [capture 2b surface 3] The capture layer declares the lazy NODES the traced
 *  body returned; matched by identity to candidate plans at promotion. Cleared
 *  each promote boundary with the candidates. */
export function stDeclareOutputNodes(nodes: LazyIRNode[]): void {
  pendingOutputNodes = nodes;
}

/** [2b surface 4] Declare the captured-fn arg nodes (in arg order) so promotion
 *  can mark which upload slots are per-replay warm (vs constant internal). */
export function stDeclareArgNodes(nodes: LazyIRNode[]): void {
  pendingArgNodes = nodes;
}

/** [2b] The captured call's body is about to run (trace path): candidates
 *  accumulated so far this step are driver-level PRE-BODY plans. */
export function stMarkBodyBegin(): void {
  bodyBeginIndex = candidates.length;
}

/**
 * Executor hook: a COMPILED normal step just ran. If the driver declared an
 * appKey for it, stash it as a candidate; if a VERIFY step, cross-check the
 * skeleton we WOULD have replayed against this fresh template.
 */
export function stCaptureCompiledStep(
  fp: number,
  planNodes: LazyIRNode[],
  loweredPlan: LoweredPlan,
  bufferArena: BufferArena | undefined,
  commands: GpuCommand[],
): void {
  if (!STEP_TAPE_REPLAY || ctxAppKey === null || !bufferArena) return;

  // Verify cross-check (§2.4 guard 6 / G2): compare the ready skeleton's
  // recorded stream against the freshly built one for the SAME plan (matched by
  // fp among the skeleton's ordered plans — 2b surface 1: a training verify
  // fires once per plan of the step, each cross-checked against its own
  // recording). A plan whose fp isn't in the skeleton is a structural drift
  // (guard would replay the wrong plan set).
  const ready = readySkeletons.get(ctxAppKey);
  if (STEP_TAPE_VERIFY_N > 0 && ready) {
    stats.verifies++;
    const skPlan = ready.plans[candidates.length];
    let diff: string | null = null;
    if (!skPlan || skPlan.fp !== fp) {
      diff = `appKey=${ctxAppKey}: plan[${candidates.length}] skeleton fp=0x${((skPlan?.fp ?? 0) >>> 0).toString(16)} but normal build fp=0x${(fp >>> 0).toString(16)} (guard would have replayed the WRONG plan)`;
    } else {
      const fresh = canonicalizeStream(commands);
      const n = Math.min(fresh.length, skPlan.canonical.length);
      for (let i = 0; i < n; i++) {
        if (fresh[i] !== skPlan.canonical[i]) {
          diff = `appKey=${ctxAppKey} plan[${candidates.length}] cmd[${i}]: skeleton="${skPlan.canonical[i]}" vs normal="${fresh[i]}"`;
          break;
        }
      }
      if (!diff && fresh.length !== skPlan.canonical.length) {
        diff = `appKey=${ctxAppKey} plan[${candidates.length}]: stream length skeleton=${skPlan.canonical.length} vs normal=${fresh.length}`;
      }
    }
    if (diff) {
      stats.verifyDiffs++;
      if (diffDiagnostics.length < 32) diffDiagnostics.push(diff);
      console.warn(`[step-tape] VERIFY DIFF: ${diff}`);
      strictThrow(`verify diff — ${diff}`);
    }
  }

  // Snapshot this plan's produced storage ids WHILE ALIVE (results get swept
  // before promotion) so cross-plan materialized refs resolve to their producer.
  const resultIds = new Map<number, { node: LazyIRNode; oi: number }>();
  for (const node of planNodes) {
    const results = node.results ?? (node.result ? [node.result] : []);
    for (let oi = 0; oi < results.length; oi++) {
      const r = results[oi];
      if (r) resultIds.set(r.id, { node, oi });
    }
  }
  // [2b sched] In-plan scalar-write ghost chains (see Candidate.scalarDresses):
  // detected NOW (results alive). The chain's own output storage identifies
  // the owning persistent tensor (the scatter dst's old storage has no owner
  // backref by now — replace-and-hold already moved past it).
  const scalarDresses: Candidate["scalarDresses"] = [];
  for (const node of planNodes) {
    if (node.op !== "stridedScatterCopy") continue;
    const n = node.shape.reduce((a: number, b: number) => a * b, 1);
    if (n > 1) continue;
    const srcRef = node.inputs[1] as { kind: string; node?: LazyIRNode };
    if (srcRef?.kind !== "pending" || srcRef.node?.op !== "tensorFromArray")
      continue;
    const out = node.results?.[0] ?? node.result;
    // Owner resolved at PROMOTION: the wrapper only materializes into the
    // chain's output storage (registering the backref) after this force
    // completes; the storage survives the recording sweep because the
    // persistent wrapper holds it.
    if (out) scalarDresses.push({ writeNode: srcRef.node, resultId: out.id });
  }

  // Owner snapshot for materialized refs (see Candidate.matOwners): captured
  // NOW because the storage->tensor backref dies when the storage is swept.
  const matOwners = new Map<object, TensorOwner>();
  for (const node of planNodes) {
    for (const ref of node.inputs) {
      if (ref.kind !== "materialized" || matOwners.has(ref)) continue;
      const owner = storageTracker.ownerOf(ref.storage.id);
      if (owner) matOwners.set(ref, owner as TensorOwner);
      if (ENV.TORCHLETTE_DEBUG_REBIND === "1") {
        const n = ref.storage.backendTensor.shape.reduce(
          (a: number, b: number) => a * b,
          1,
        );
        if (n <= 4)
          console.error(
            `[rebind-dbg] CAPTURE node=${node.op} inputStorage=${ref.storage.id} numel=${n} owner=${owner ? "yes" : "MISSING"}`,
          );
      }
    }
  }

  candidates.push({
    appKey: ctxAppKey,
    fp,
    planNodes,
    loweredPlan,
    bufferArena,
    commands,
    epoch: ctxEpoch,
    stepScopedCleanup: ctxStepScopedCleanup,
    resultIds,
    matOwners,
    scalarDresses,
  });
  stats.captures++;
}

/**
 * Frontend hook at markStep (after stEndStep): if the just-ended step became
 * eligible AND we captured a candidate for its template under the active
 * appKey, promote it to a ready skeleton. Two consecutive structurally
 * identical compiled steps under the same appKey (the recorder's eligibility)
 * — mirrors the compiled plan's own record-on-2nd-execution lifecycle.
 */
export function stPromoteEligibleSkeleton(): void {
  if (!STEP_TAPE_REPLAY) return;
  // The app context is PER-STEP: clear it at the step boundary so any step that
  // does NOT re-declare an appKey (prefill, a training step, any non-decode
  // work) is invisible to capture/verify/replay. Without this, a stale decode
  // appKey leaks into the next generation's prefill and stCaptureCompiledStep
  // verifies the prefill's plan against the decode skeleton (a spurious diff)
  // — and could, in principle, capture a non-decode plan under a decode key.
  // The candidate captured THIS step already holds its own appKey copy, so
  // clearing here does not affect promotion.
  ctxAppKey = null;
  const elig = stConsumeLastEligible();
  const cands = candidates;
  const outNodes = pendingOutputNodes;
  const argNodes = pendingArgNodes;
  const bodyBegin = bodyBeginIndex;
  candidates = [];
  pendingOutputNodes = [];
  pendingArgNodes = [];
  bodyBeginIndex = -1;
  if (!elig || cands.length === 0) {
    return;
  }
  const appKey = cands[0].appKey;
  if (readySkeletons.has(appKey)) return; // already ready — keep the first

  // ORDERED plan-sequence match (2b surface 1): the captured candidate fps must
  // equal the recorder's ORDERED plan fp sequence, in order and length. A
  // mismatch (executor re-segmented, or a plan ran outside the appKey window)
  // aborts promotion → the step re-records next pair.
  if (cands.length !== elig.orderedFps.length) {
    return;
  }
  for (let k = 0; k < cands.length; k++) {
    if (cands[k].fp !== elig.orderedFps[k]) {
      return;
    }
  }

  const tape = stGetTapes().get(elig.bucketKey);
  if (!tape) {
    return;
  }

  // Build one SkeletonPlan per candidate, routing the tape's upload/scalar
  // slots to their owning plan by fpHex (the slot id embeds the plan fp).
  const skPlans: SkeletonPlan[] = [];
  for (let k = 0; k < cands.length; k++) {
    const cand = cands[k];
    const fpHex = (cand.fp >>> 0).toString(16);
    // Upload nodes for THIS plan (ordered by node position → the driver dresses
    // them by ARG ORDER, not shape: a training step's batch x/y share a shape,
    // so shape-keying is ambiguous; ordered re-dressing is unambiguous). Decode
    // (unique shapes, one plan) is the length-1 special case and still works.
    const uploads: UploadNode[] = [];
    const trainingMode = argNodes.length > 0;
    for (const s of tape.slots) {
      if (s.source !== "upload") continue;
      const m = /^w:([0-9a-f]+):(\d+)$/.exec(s.id);
      if (!m || m[1] !== fpHex) continue;
      const pos = Number(m[2]);
      const node = cand.planNodes[pos];
      if (!node) {
        return;
      }
      // isArg: in decode (no argNodes) every upload is re-dressed. In training
      // only the declared arg nodes (batch x/y) are warm; ANY other upload
      // (constant `zeros`/`full`/mask etc.) keeps its recorded payload. An ARG
      // upload MUST be a tensorFromArray (re-dressed by fresh values).
      const isArg = trainingMode ? argNodes.includes(node) : true;
      if (isArg && node.op !== "tensorFromArray") {
        return;
      }
      uploads.push({
        node,
        shapeKey: shapeKeyOf(node.shape),
        shape: node.shape,
        isArg,
      });
    }
    uploads.sort((a, b) => a.node.id - b.node.id);

    const scalars: ScalarSlot[] = [];
    for (const s of elig.scalarSlots) {
      if (s.fp !== cand.fp) continue;
      const node = cand.planNodes[s.pos];
      const ref = node?.inputs[s.inputIndex];
      if (!node || !ref || ref.kind !== "scalar") {
        return;
      }
      scalars.push({ node, inputIndex: s.inputIndex, recorded: ref.value });
    }

    skPlans.push({
      fp: cand.fp,
      planNodes: cand.planNodes,
      loweredPlan: cand.loweredPlan,
      bufferArena: cand.bufferArena,
      uploads,
      scalars,
      canonical: canonicalizeStream(cand.commands),
      preBody: bodyBegin >= 0 && k < bodyBegin,
    });
  }

  // Output-node refs (2b surface 3): find each declared output node BY IDENTITY
  // among the candidate plans → (planIndex, pos). Default (decode, no hint):
  // the LAST node of the LAST plan — the historical lastNode harvest.
  const outputRefs: Array<{ planIndex: number; pos: number }> = [];
  if (outNodes.length > 0) {
    for (const on of outNodes) {
      let found = false;
      for (let pi = 0; pi < skPlans.length && !found; pi++) {
        const pos = skPlans[pi].planNodes.indexOf(on);
        if (pos >= 0) {
          outputRefs.push({ planIndex: pi, pos });
          found = true;
        }
      }
      if (!found) {
        return; // output node not in any captured plan → abort
      }
    }
  } else {
    const last = skPlans[skPlans.length - 1];
    outputRefs.push({
      planIndex: skPlans.length - 1,
      pos: last.planNodes.length - 1,
    });
  }

  // Cross-plan links (2b): index every node's produced storage ids across all
  // plans, then find later-plan MATERIALIZED inputs whose storage a node in an
  // EARLIER plan produced. Those refs freeze the recording's step-scoped buffer
  // (destroyed at markStep); re-point them to the fresh producer result per
  // replay. (Storage ids are monotonic + unique, so an id match is exact.)
  const producerById = new Map<number, { node: LazyIRNode; oi: number; plan: number }>();
  for (let pi = 0; pi < cands.length; pi++) {
    for (const [id, { node, oi }] of cands[pi].resultIds) {
      producerById.set(id, { node, oi, plan: pi });
    }
  }
  const crossPlanLinks: CrossPlanLink[] = [];
  for (let pi = 0; pi < skPlans.length; pi++) {
    for (const node of skPlans[pi].planNodes) {
      for (let ii = 0; ii < node.inputs.length; ii++) {
        const ref = node.inputs[ii];
        if (ref.kind !== "materialized") continue;
        const prod = producerById.get(ref.storage.id);
        // Only a link if the producer is in an EARLIER plan (a within-plan
        // materialized ref is handled by the plan's own execution).
        if (prod && prod.plan < pi) {
          crossPlanLinks.push({
            consumerPlan: pi,
            consumerNode: node,
            inputIndex: ii,
            producerNode: prod.node,
            producerOutputIndex: prod.oi,
          });
        }
      }
    }
  }

  // [2b rebind] Collect owner-backed materialized refs across replayed plans
  // (dedup by ref object; owners snapshotted at capture while storages lived).
  const rebinds: Skeleton["rebinds"] = [];
  const seenRefs = new Set<object>();
  const scalarDresses: Skeleton["scalarDresses"] = [];
  const seenWrites = new Set<LazyIRNode>();
  for (let k = 0; k < cands.length; k++) {
    if (skPlans[k].preBody) continue;
    for (const [ref, owner] of cands[k].matOwners) {
      if (seenRefs.has(ref)) continue;
      seenRefs.add(ref);
      rebinds.push({
        ref: ref as { kind: string; storage: StorageHandle },
        owner,
      });
    }
    for (const d of cands[k].scalarDresses) {
      if (seenWrites.has(d.writeNode)) continue;
      seenWrites.add(d.writeNode);
      const owner = storageTracker.ownerOf(d.resultId);
      if (owner)
        scalarDresses.push({
          writeNode: d.writeNode,
          owner: owner as TensorOwner,
        });
    }
  }

  if (ENV.TORCHLETTE_DEBUG_REBIND === "1") {
    console.error(
      `[rebind-dbg] PROMOTE appKey=${appKey} rebinds=${rebinds.length} dresses=${scalarDresses.length} plans=${skPlans.length}`,
    );
  }
  readySkeletons.set(appKey, {
    appKey,
    plans: skPlans,
    epoch: cands[0].epoch,
    stepScopedCleanup: cands[0].stepScopedCleanup,
    outputRefs,
    crossPlanLinks,
    rebinds,
    scalarDresses,
  });
  stats.promotions++;
}

// ---------------------------------------------------------------------------
// Replay (§2.3)
// ---------------------------------------------------------------------------

/** [2b rebind] Host payload of an UNFORCED driver-level scalar write chain.
 *  Shape: copy_(dst, tensorFromArray([v])) lowers to a pending
 *  stridedScatterCopy whose src input is a pending tensorFromArray carrying
 *  `v` in its payload. Returns the f32 bytes, or undefined when the chain has
 *  any other shape (the rebind then leaves the recorded binding; loud guards
 *  catch a stale read). */
function extractPendingScalarPayload(
  node: LazyIRNode,
): Float32Array | undefined {
  let src: LazyIRNode | undefined;
  if (node.op === "tensorFromArray") {
    src = node;
  } else if (node.op === "stridedScatterCopy") {
    const srcRef = node.inputs[1] as
      | { kind: string; node?: LazyIRNode }
      | undefined;
    if (srcRef?.kind === "pending" && srcRef.node?.op === "tensorFromArray") {
      src = srcRef.node;
    }
  }
  const values = (src?.payload as { values?: ArrayLike<number> } | undefined)
    ?.values;
  if (!values || values.length !== 1) return undefined;
  return Float32Array.from(values as ArrayLike<number>);
}

export interface ReplayOutput {
  resultHandle: StorageHandle;
  shape: number[];
  dtype: DType;
  device: DeviceKind;
}

export interface ReplayResult {
  hit: boolean;
  /** Harvested output tensors (2b surface 3), in declared order. Decode: one
   *  (the logits/last node). */
  outputs?: ReplayOutput[];
}

/**
 * Attempt to replay the current step's tape for `ctxAppKey`. `uploads` supplies
 * the fresh per-step payloads, matched to the skeleton's upload nodes BY ORDER
 * (decode: unique shapes ⇒ order and shape agree; training: x/y share a shape
 * ⇒ order disambiguates). Returns hit=false on ANY guard miss (the driver then
 * runs the normal path). Multi-plan (2b surface 1): all plans replay in order.
 */
export async function stTryReplay(
  uploads: Array<{ shape: number[]; values: Float32Array }>,
  backend: Backend,
): Promise<ReplayResult> {
  if (!STEP_TAPE_REPLAY || ctxAppKey === null) return { hit: false };
  const sk = readySkeletons.get(ctxAppKey);
  if (!sk) {
    stats.missNoTape++;
    return { hit: false };
  }
  stats.replays++;
  const appKey = ctxAppKey;

  // Guard 4: EVERY replayed plan's compiled plan valid (upfront — 2b surface
  // 4). preBody plans are never replayed → not gated.
  for (const pl of sk.plans) {
    if (pl.preBody) continue;
    if (!pl.loweredPlan.compiledPlan?.valid) {
      readySkeletons.delete(appKey);
      stats.missValidity++;
      stats.invalidations++;
      strictThrow(`plan invalidated for appKey=${appKey}`);
      return { hit: false };
    }
  }
  // Guard 5: boundary REGIME (see decode note — regime not epoch equality).
  if (sk.stepScopedCleanup !== ctxStepScopedCleanup) {
    stats.missEpoch++;
    strictThrow(
      `regime drift appKey=${appKey} (stepScopedCleanup ${sk.stepScopedCleanup}->${ctxStepScopedCleanup})`,
    );
    return { hit: false };
  }
  // Guard 3: scalar coverage (value-level, the frozen-α defense). Flattened
  // across plans in plan order (matches how the capture layer collects them).
  const skScalars: ScalarSlot[] = [];
  for (const pl of sk.plans) {
    if (pl.preBody) continue;
    for (const s of pl.scalars) skScalars.push(s);
  }
  if (ctxScalars.length !== skScalars.length) {
    stats.missScalar++;
    return { hit: false };
  }
  for (let i = 0; i < skScalars.length; i++) {
    if (!Object.is(ctxScalars[i], skScalars[i].recorded)) {
      stats.missScalar++;
      strictThrow(
        `scalar slot ${i} changed ${skScalars[i].recorded} -> ${ctxScalars[i]} (frozen-α guard) appKey=${appKey}`,
      );
      return { hit: false };
    }
  }

  // Upfront upload shape check (2b surface 4: no mid-replay throw). Only ARG
  // uploads are re-dressed (decode: all uploads are args; training: only batch
  // x/y — `zeros` etc. keep their recorded constant payload). Flatten arg
  // uploads in a stable global order (plan order, then node id) matching the
  // caller's arg order.
  const argUploadNodes: UploadNode[] = [];
  for (const pl of sk.plans) {
    if (pl.preBody) continue; // driver work — never replayed, never dressed
    for (const u of pl.uploads) if (u.isArg) argUploadNodes.push(u);
  }
  if (uploads.length !== argUploadNodes.length) {
    stats.missShape++;
    strictThrow(
      `upload count ${uploads.length} != skeleton arg uploads ${argUploadNodes.length} appKey=${appKey}`,
    );
    return { hit: false };
  }
  for (let i = 0; i < argUploadNodes.length; i++) {
    const need = argUploadNodes[i].node.shape.reduce((a, b) => a * b, 1);
    if (uploads[i].values.length !== need) {
      stats.missShape++;
      strictThrow(
        `upload ${i} length ${uploads[i].values.length} != ${need} appKey=${appKey}`,
      );
      return { hit: false };
    }
  }
  // Re-dress by ORDER (preserve each node's recorded dtype — token args are int).
  for (let i = 0; i < argUploadNodes.length; i++) {
    const node = argUploadNodes[i].node;
    const p = node.payload as { values?: Float32Array; dtype?: DType } | undefined;
    if (p && "values" in p) {
      p.values = uploads[i].values;
    } else {
      node.payload = { values: uploads[i].values, dtype: "f32" };
    }
  }

  // [2b sched] Re-dress recorded in-plan scalar-write chains from the owning
  // tensor's CURRENT pending payload (the driver's setLR issued since the
  // last markStep). TAG_WRITE re-executes the recorded write node from its
  // payload, so the replayed optimizer reads THIS step's value. Sticky when
  // no pending write exists this step: the last-dressed value IS the current
  // value (scalars only change via writes).
  for (const d of sk.scalarDresses) {
    // Authoritative host value first (the setter notes it — single source at
    // the seam; core/scalar-slots.ts). Chain extraction is the fallback for
    // authors that don't note values; it is only valid while the write is
    // still pending — a FORCED write leaves no chain, and skipping then
    // (sticky dress) goes stale by one step whenever markStep regimes mix
    // (measured: sporadic 3e-3..0.07).
    const noted = getScalarSlotValue(d.owner);
    let bytes: Float32Array | undefined =
      noted !== undefined ? Float32Array.of(noted) : undefined;
    if (!bytes) {
      const cur = d.owner.lazyRef as { kind: string; node?: LazyIRNode };
      if (cur?.kind === "pending" && cur.node)
        bytes = extractPendingScalarPayload(cur.node);
    }
    if (ENV.TORCHLETTE_DEBUG_REBIND === "1")
      console.error(
        `[rebind-dbg] DRESS noted=${noted} bytes=${bytes?.[0] ?? "n/a"}`,
      );
    if (bytes) {
      d.writeNode.payload = { values: bytes, dtype: "f32" };
    }
  }

  // [2b rebind] Re-resolve owner-backed materialized refs to the owner's
  // CURRENT storage. A persistent tensor updated by replace-and-hold writes
  // (the scheduler's lr copy_ chain) moves to a NEW storage every step; the
  // skeleton's recording-era ref would read the stale buffer forever (the
  // frozen-LR class — captured trained with the RECORDING's lr, caught by the
  // schedule-through-hits gate). No-op when the storage is unchanged (params,
  // m/v: in-place planner-fixed buffers).
  for (const rb of sk.rebinds) {
    const cur = rb.owner.lazyRef as {
      kind: string;
      storage?: StorageHandle;
      node?: LazyIRNode;
      outputIndex?: number;
    };
    // The owner's CURRENT storage: a materialized ref carries it directly; a
    // pending ref that was already FORCED (markStep runs before every replay)
    // carries it as the node's result. An unforced pending ref has no current
    // value yet - keep the recorded binding (loud guards catch a stale read).
    let curStorage: StorageHandle | undefined;
    if (cur?.kind === "materialized") {
      curStorage = cur.storage;
    } else if (cur?.kind === "pending" && cur.node) {
      const oi = cur.outputIndex ?? 0;
      curStorage = cur.node.results?.[oi] ?? (oi === 0 ? cur.node.result : undefined);
      // Pending and UNFORCED: no current storage to repoint to. The value
      // itself flows via the scalar DRESS above (authoritative registry /
      // chain payload). NEVER force the chain mid-attempt — forcing here
      // corrupts replay state (measured 12.7-nat divergence).
    }
    // Rebind scope (measured, both directions): (a) SCALAR-STATE refs - the
    // inc-2a optimizer scalars ([1]-shaped f32: lr per group; scale) are
    // updated by DRIVER-level replace-and-hold copy_ chains, so their current
    // storage wanders while the frozen ref reads the recording-era buffer
    // (frozen-LR: 0.48 nats divergence under CosineAnnealingLR); (b) refs
    // whose recorded storage is DESTROYED (dangling). Larger LIVE refs
    // (params/m/v) are NOT rebound: they flow through the planner-consistent
    // binding chain the compiled replays already maintain, and re-pointing
    // them to wrapper-tracked results measurably perturbs the trajectory
    // (0.010 vs noise-floor 1e-3 over 16 steps).
    const numel = rb.ref.storage.backendTensor.shape.reduce(
      (a: number, b: number) => a * b,
      1,
    );
    if (ENV.TORCHLETTE_DEBUG_REBIND === "1") {
      console.error(
        `[rebind-dbg] numel=${numel} cur=${cur?.kind}${curStorage ? "+storage" : ""} refStorage=${rb.ref.storage.id} destroyed=${storageTracker.isDestroyed(rb.ref.storage.id)} same=${curStorage === rb.ref.storage}`,
      );
    }
    // SCALAR-ONLY ([1]-shaped): the optimizer-scalar channel (lr; scale).
    // Larger refs (params/m/v) are NEVER rebound — even when the recorded
    // storage handle is swept, the underlying planner-bound buffers remain
    // the live value chain across replays (main runs at the noise floor with
    // no rebinding at all); re-pointing them to wrapper-tracked storages
    // intermittently aliases planner temps (measured 6e-3..1.06-nat sporadic
    // divergence in the foreach arm, 1-in-3 runs).
    if (numel > 1) continue;
    if (curStorage && curStorage !== rb.ref.storage) {
      // The owner's forced current storage carries the value — repoint.
      rb.ref.storage = curStorage;
    } else if (!curStorage) {
      // Pending-UNFORCED write (the driver's setLR issued since the last
      // markStep): there is no storage to repoint to, but the VALUE is host
      // data — inject it into the recorded binding pre-replay
      // (queue.writeBuffer executes ahead of the replay's submits; prior
      // replays' readers are fenced by the intervening markStep). This is
      // the delivery path for cross-plan consumers of the recorded lr-write
      // ghost's OUTPUT buffer (deleting it measured a consistent ~0.03
      // one-step-stale-lr divergence). Registry first, chain fallback —
      // same sourcing as the dress.
      const noted = getScalarSlotValue(rb.owner);
      const bytes =
        noted !== undefined
          ? Float32Array.of(noted)
          : cur?.kind === "pending" && cur.node
            ? extractPendingScalarPayload(cur.node)
            : undefined;
      const bt = rb.ref.storage.backendTensor as unknown as {
        buffer: GPUBuffer;
        offset?: number;
        dtype?: string;
      };
      const dev = (backend as unknown as { device?: GPUDevice }).device;
      if (bytes && dev && bt.buffer && (bt.dtype ?? "f32") === "f32") {
        dev.queue.writeBuffer(
          bt.buffer,
          (bt.offset ?? 0) * 4,
          bytes.buffer,
          bytes.byteOffset,
          bytes.byteLength,
        );
      }
    }
  }

  // Reset ALL replayed plans' per-step node state up front (results/exec flags
  // undefined) so external-slot resolution + harvest read this replay's
  // buffers, not the prior replay's swept ones. preBody plans are NOT replayed
  // (their driver-level equivalents run for real each iteration) — leave their
  // recorded nodes untouched.
  for (const pl of sk.plans) {
    if (pl.preBody) continue;
    for (const n of pl.planNodes) {
      n.results = undefined;
      n._executed = undefined;
      n._inputsRetained = undefined;
    }
  }

  // Materialize the ARG upload nodes from their fresh payloads BEFORE any plan
  // runs. They are EXTERNAL plan inputs (leaves the compiled plan reads from
  // node.result), not TAG_WRITE-covered (decode's f32 uploads ride the stable
  // buffer; training's i32 token batch does not) — so the replay must produce
  // their result buffer, exactly as the driver's tensorFromArray would on the
  // normal path. (isArg=false uploads keep their recorded payload/result — but
  // those were nulled above; they re-execute inside their plan as internal ops.)
  for (const u of argUploadNodes) {
    const node = u.node;
    const res = executeOpSync(node, [], backend);
    const bt = res instanceof Promise ? await res : res;
    assignNodeResult(node, bt, [], []);
  }

  // Replay every plan in order. For a MULTI-plan step (training) suppress the
  // stage-3 B clear-at-release: a cross-plan intermediate (grad buffer produced
  // by the backward plan, read by the optimizer plan) must live for the whole
  // replay, not be released mid-replay by the last-reader plan (§5 declared
  // lifetime). Single-plan decode has no cross-plan release → flag stays off.
  const multiPlan = sk.plans.length > 1;
  if (multiPlan) setStepTapeReplayActive(true);
  try {
    for (let pi = 0; pi < sk.plans.length; pi++) {
      const pl = sk.plans[pi];
      // preBody plans (driver-level side writes, e.g. the scheduler's lr
      // write) are NOT replayed — the driver's fresh pending nodes execute
      // them for real at each step's markStep; replaying the recorded ghost
      // would clobber the fresh value with the recorded one (frozen-LR).
      if (pl.preBody) continue;
      // Pin the template against idle-retire (see decode note): a replay
      // bypasses the normal path, so mark it replayed to reset only the idle
      // clock.
      noteTemplateReplayed(pl.fp);
      await executeLoweredPlan(
        { nodes: pl.planNodes },
        pl.planNodes,
        pl.loweredPlan,
        backend,
        { bufferArena: pl.bufferArena },
      );
      if (!pl.loweredPlan.compiledPlan?.valid) {
        // A plan fell back to lowered mid-step (staleness). Result is still
        // correct (lowered ran); drop the skeleton so the next step re-records.
        readySkeletons.delete(appKey);
        stats.invalidations++;
      }
      // Re-point any cross-plan materialized refs whose producer plan is THIS
      // one: bind the consumer's frozen ref to the producer's FRESH result (the
      // recording buffer is step-scoped + destroyed; §5 declared lifetime).
      for (const link of sk.crossPlanLinks) {
        if (link.producerNode.result === undefined && !link.producerNode.results)
          continue;
        if (!pl.planNodes.includes(link.producerNode)) continue;
        const fresh =
          link.producerOutputIndex === 0
            ? link.producerNode.result
            : link.producerNode.results?.[link.producerOutputIndex];
        const ref = link.consumerNode.inputs[link.inputIndex];
        if (fresh && ref.kind === "materialized") ref.storage = fresh;
      }
    }
  } finally {
    if (multiPlan) setStepTapeReplayActive(false);
  }

  // Harvest declared outputs (2b surface 3).
  const outputs: ReplayOutput[] = [];
  for (const ref of sk.outputRefs) {
    const node = sk.plans[ref.planIndex].planNodes[ref.pos];
    const result = node?.result;
    if (!result) {
      readySkeletons.delete(appKey);
      stats.missValidity++;
      strictThrow(`replay produced no output for appKey=${appKey}`);
      return { hit: false };
    }
    outputs.push({
      resultHandle: result,
      shape: node.shape,
      dtype: node.dtype,
      device: node.device,
    });
  }
  stats.hits++;
  stMarkReplayStep(); // the upcoming readback + markStep are invisible to the recorder
  return { hit: true, outputs };
}

// ---------------------------------------------------------------------------
// Observability (§6)
// ---------------------------------------------------------------------------

export function stReplayStats(): typeof stats & {
  diffDiagnostics: string[];
  readyTapes: number;
} {
  return {
    ...stats,
    readyTapes: readySkeletons.size,
    diffDiagnostics: diffDiagnostics.slice(),
  };
}

/** [capture 2a] Drop the ready skeleton for one appKey (the §6 cold knob:
 *  capturedFn.invalidate()). The recorder tape is left alone — it will simply
 *  re-promote on the next two eligible steps under the same appKey. */
export function stDropSkeleton(appKey: string): void {
  readySkeletons.delete(appKey);
}

export function stReplayReset(): void {
  readySkeletons.clear();
  candidates = [];
  pendingOutputNodes = [];
  ctxAppKey = null;
  ctxScalars = [];
  for (const k of Object.keys(stats)) (stats as Record<string, number>)[k] = 0;
  diffDiagnostics.length = 0;
}
