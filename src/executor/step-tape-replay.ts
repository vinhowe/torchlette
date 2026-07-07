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

// ---------------------------------------------------------------------------
// Skeleton store (§8.2 option i)
// ---------------------------------------------------------------------------

interface UploadNode {
  node: LazyIRNode;
  /** Shape signature (e.g. "1,1,1,128") — the driver keys fresh payloads by it. */
  shapeKey: string;
  shape: number[];
}

interface ScalarSlot {
  node: LazyIRNode;
  inputIndex: number;
  /** Recorded value (guard-3 baseline). */
  recorded: number;
}

interface Skeleton {
  appKey: string;
  fp: number;
  planNodes: LazyIRNode[];
  loweredPlan: LoweredPlan;
  bufferArena: BufferArena;
  uploads: UploadNode[];
  scalars: ScalarSlot[];
  epoch: number;
  stepScopedCleanup: boolean;
  /** Canonical command stream at capture time (verify determinism/faithfulness). */
  canonical: string[];
  lastNode: LazyIRNode;
}

/** Candidate captured on a compiled normal step, keyed by the active appKey;
 *  promoted to a ready skeleton iff the recorder deems that step eligible. */
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
let candidate: Candidate | null = null;

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
  // recorded stream against the freshly built one for the same appKey.
  const ready = readySkeletons.get(ctxAppKey);
  if (STEP_TAPE_VERIFY_N > 0 && ready) {
    stats.verifies++;
    let diff: string | null = null;
    if (ready.fp !== fp) {
      diff = `appKey=${ctxAppKey}: skeleton fp=0x${(ready.fp >>> 0).toString(16)} but normal build fp=0x${(fp >>> 0).toString(16)} (guard would have replayed the WRONG plan)`;
    } else {
      const fresh = canonicalizeStream(commands);
      const n = Math.min(fresh.length, ready.canonical.length);
      for (let i = 0; i < n; i++) {
        if (fresh[i] !== ready.canonical[i]) {
          diff = `appKey=${ctxAppKey} cmd[${i}]: skeleton="${ready.canonical[i]}" vs normal="${fresh[i]}"`;
          break;
        }
      }
      if (!diff && fresh.length !== ready.canonical.length) {
        diff = `appKey=${ctxAppKey}: stream length skeleton=${ready.canonical.length} vs normal=${fresh.length}`;
      }
    }
    if (diff) {
      stats.verifyDiffs++;
      if (diffDiagnostics.length < 32) diffDiagnostics.push(diff);
      console.warn(`[step-tape] VERIFY DIFF: ${diff}`);
      strictThrow(`verify diff — ${diff}`);
    }
  }

  candidate = {
    appKey: ctxAppKey,
    fp,
    planNodes,
    loweredPlan,
    bufferArena,
    commands,
    epoch: ctxEpoch,
    stepScopedCleanup: ctxStepScopedCleanup,
  };
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
  const cand = candidate;
  candidate = null;
  if (!elig || !cand) return;
  if (!elig.fps.includes(cand.fp)) return;
  if (readySkeletons.has(cand.appKey)) return; // already ready — keep the first

  const tape = stGetTapes().get(elig.bucketKey);
  if (!tape) return;

  // Upload nodes: the tape's "upload" slots carry ids "w:<fpHex>:<pos>" — pos
  // indexes the SAME planNodes array the executor captured this step.
  const uploads: UploadNode[] = [];
  const fpHex = (cand.fp >>> 0).toString(16);
  for (const s of tape.slots) {
    if (s.source !== "upload") continue;
    const m = /^w:([0-9a-f]+):(\d+)$/.exec(s.id);
    if (!m || m[1] !== fpHex) continue;
    const pos = Number(m[2]);
    const node = cand.planNodes[pos];
    if (!node || node.op !== "tensorFromArray") return; // shape drift → abort
    uploads.push({ node, shapeKey: shapeKeyOf(node.shape), shape: node.shape });
  }
  // Ambiguous shapes ⇒ the driver can't key payloads unambiguously; refuse.
  const seenShapes = new Set<string>();
  for (const u of uploads) {
    if (seenShapes.has(u.shapeKey)) return;
    seenShapes.add(u.shapeKey);
  }

  const scalars: ScalarSlot[] = [];
  for (const s of elig.scalarSlots) {
    const node = cand.planNodes[s.pos];
    const ref = node?.inputs[s.inputIndex];
    if (!node || !ref || ref.kind !== "scalar") return;
    scalars.push({ node, inputIndex: s.inputIndex, recorded: ref.value });
  }

  readySkeletons.set(cand.appKey, {
    appKey: cand.appKey,
    fp: cand.fp,
    planNodes: cand.planNodes,
    loweredPlan: cand.loweredPlan,
    bufferArena: cand.bufferArena,
    uploads,
    scalars,
    epoch: cand.epoch,
    stepScopedCleanup: cand.stepScopedCleanup,
    canonical: canonicalizeStream(cand.commands),
    lastNode: cand.planNodes[cand.planNodes.length - 1],
  });
  stats.promotions++;
}

// ---------------------------------------------------------------------------
// Replay (§2.3)
// ---------------------------------------------------------------------------

export interface ReplayResult {
  hit: boolean;
  resultHandle?: StorageHandle;
  shape?: number[];
  dtype?: DType;
  device?: DeviceKind;
}

/**
 * Attempt to replay the current step's tape for `ctxAppKey`. `uploads` supplies
 * the fresh per-token payloads keyed by shape (unique per bucket). Returns
 * hit=false on ANY guard miss (the driver then runs the normal path).
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

  // Guard 4: plan validity.
  if (!sk.loweredPlan.compiledPlan?.valid) {
    readySkeletons.delete(ctxAppKey);
    stats.missValidity++;
    stats.invalidations++;
    strictThrow(`plan invalidated for appKey=${ctxAppKey}`);
    return { hit: false };
  }
  // Guard 5: boundary REGIME. The absolute epoch counter advances several
  // times per step (stepBoundary + pool flush + fence — src/core/epoch.ts), so
  // it is NOT an equality target; the meaningful invariant is the boundary
  // regime (stepScopedCleanup). The one discontinuous epoch bump that matters —
  // a plannerReset / engine-instance boundary — is caught by guard 4 (the
  // plannerGen check inside executeCompiledPlan invalidates the compiled plan).
  if (sk.stepScopedCleanup !== ctxStepScopedCleanup) {
    stats.missEpoch++;
    strictThrow(
      `regime drift appKey=${ctxAppKey} (stepScopedCleanup ${sk.stepScopedCleanup}->${ctxStepScopedCleanup})`,
    );
    return { hit: false };
  }
  // Guard 3: scalar coverage (value-level, the frozen-α defense).
  if (ctxScalars.length !== sk.scalars.length) {
    stats.missScalar++;
    return { hit: false };
  }
  for (let i = 0; i < sk.scalars.length; i++) {
    if (!Object.is(ctxScalars[i], sk.scalars[i].recorded)) {
      stats.missScalar++;
      strictThrow(
        `scalar slot ${i} changed ${sk.scalars[i].recorded} -> ${ctxScalars[i]} (frozen-α guard) appKey=${ctxAppKey}`,
      );
      return { hit: false };
    }
  }

  // Re-dress: match fresh payloads to skeleton upload nodes by shape.
  const byShape = new Map<string, Float32Array>();
  for (const u of uploads) byShape.set(shapeKeyOf(u.shape), u.values);
  for (const u of sk.uploads) {
    const values = byShape.get(u.shapeKey);
    if (!values || values.length !== u.node.shape.reduce((a, b) => a * b, 1)) {
      stats.missShape++;
      strictThrow(`upload shape ${u.shapeKey} missing/mismatched appKey=${ctxAppKey}`);
      return { hit: false };
    }
    const p = u.node.payload as { values?: Float32Array; dtype?: DType } | undefined;
    if (p && "values" in p) {
      p.values = values;
    } else {
      u.node.payload = { values, dtype: "f32" };
    }
  }

  // Reset per-step node state so the replay behaves as a fresh graph: results
  // (primary + side outputs) and execution flags must be undefined — otherwise
  // external-slot resolution and the harvest read stale storages from the prior
  // replay (whose buffers were swept at the last markStep).
  for (const n of sk.planNodes) {
    n.results = undefined;
    n._executed = undefined;
    n._inputsRetained = undefined;
  }

  await executeLoweredPlan(
    { nodes: sk.planNodes },
    sk.planNodes,
    sk.loweredPlan,
    backend,
    { bufferArena: sk.bufferArena },
  );

  // If executeLoweredPlan fell back to the lowered path (compiled plan dropped
  // mid-step by a staleness check), the skeleton is stale — drop it so the next
  // step re-records. The result is still correct (lowered path ran), so this
  // step is a valid hit.
  if (!sk.loweredPlan.compiledPlan?.valid) {
    readySkeletons.delete(ctxAppKey);
    stats.invalidations++;
  }

  const result = sk.lastNode.result;
  if (!result) {
    // Should never happen for a valid decode plan; degrade to normal path.
    readySkeletons.delete(ctxAppKey);
    stats.missValidity++;
    strictThrow(`replay produced no result for appKey=${ctxAppKey}`);
    return { hit: false };
  }
  stats.hits++;
  stMarkReplayStep(); // the upcoming readback + markStep are invisible to the recorder
  return {
    hit: true,
    resultHandle: result,
    shape: sk.lastNode.shape,
    dtype: sk.lastNode.dtype,
    device: sk.lastNode.device,
  };
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

export function stReplayReset(): void {
  readySkeletons.clear();
  candidate = null;
  ctxAppKey = null;
  ctxScalars = [];
  for (const k of Object.keys(stats)) (stats as Record<string, number>)[k] = 0;
  diffDiagnostics.length = 0;
}
