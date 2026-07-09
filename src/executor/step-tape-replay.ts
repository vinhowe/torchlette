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
import { executeLoweredPlan } from "./executor";
import { assignNodeResult, executeOpSync } from "./op-dispatch";
import { noteTemplateReplayed } from "./observed-liveness";

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
  // Multi-plan (2b surface 4 upfront slot-check): EVERY plan's compiled plan
  // must still be valid before we commit to a hit.
  for (const pl of sk.plans) {
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

  candidates.push({
    appKey: ctxAppKey,
    fp,
    planNodes,
    loweredPlan,
    bufferArena,
    commands,
    epoch: ctxEpoch,
    stepScopedCleanup: ctxStepScopedCleanup,
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
  candidates = [];
  pendingOutputNodes = [];
  pendingArgNodes = [];
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

  readySkeletons.set(appKey, {
    appKey,
    plans: skPlans,
    epoch: cands[0].epoch,
    stepScopedCleanup: cands[0].stepScopedCleanup,
    outputRefs,
  });
  stats.promotions++;
}

// ---------------------------------------------------------------------------
// Replay (§2.3)
// ---------------------------------------------------------------------------

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

  // Guard 4: EVERY plan's compiled plan valid (upfront — 2b surface 4).
  for (const pl of sk.plans) {
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
  for (const pl of sk.plans) for (const s of pl.scalars) skScalars.push(s);
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
  for (const pl of sk.plans)
    for (const u of pl.uploads) if (u.isArg) argUploadNodes.push(u);
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

  // Reset ALL plans' per-step node state up front (results/exec flags undefined)
  // so external-slot resolution + harvest read this replay's buffers, not the
  // prior replay's swept ones.
  for (const pl of sk.plans) {
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

  // Replay every plan in order.
  for (const pl of sk.plans) {
    // Pin the template against idle-retire (see decode note): a replay bypasses
    // the normal path, so mark it replayed to reset only the idle clock.
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
