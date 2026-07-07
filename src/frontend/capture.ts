/**
 * Phase 2a `capture()` — user-declared staging for decode/inference-shaped loop
 * bodies, built ON the phase-1 step-tape (docs/staged-execution-phase2a.md).
 *
 * A CapturedFn traces its fn normally on early calls (the phase-1 recorder
 * observes the plan and derives eligibility); once a replayable skeleton exists
 * and every guard passes, it replays via the phase-1 tape and the fn's heavy
 * body never executes past its per-step uploads.
 *
 * ## THE ARG-BOUNDARY CONTRACT (value coverage)
 *
 * Everything that varies must cross the ARGUMENT LIST — coverage is derived
 * from the args, observable at call time without executing the body:
 *
 *  - TENSOR arguments are dynamic slots (the WARM knob): their fresh VALUES are
 *    re-dressed onto the skeleton every call (a caller-built `tensorFromArray`
 *    arg) or read live from their stable buffer by the replayed plan (a
 *    persistent tensor arg updated in place). Change the data every call at
 *    full replay speed — zero misses. Caller-built upload args are DONATED to
 *    the call (consumed; disposed on a replay hit) — build them fresh per
 *    call and do not reuse them afterwards.
 *  - PLAIN-VALUE arguments are hashed into the bucket key (the COLD-per-value
 *    knob): a changed value is a COUNTED miss + re-record — correct, one
 *    re-trace per new value. For occasional config (a sampling k, a debug
 *    flag), not per-step data.
 *  - CLOSURE-CAPTURED VALUES ARE FROZEN AT RECORD TIME (jax.jit semantics).
 *    Nothing re-reads a JS closure on a replay hit; a closure variable changed
 *    after recording silently keeps its recorded value until the next
 *    re-trace. PASS ANYTHING THAT VARIES AS AN ARGUMENT — a tensor for
 *    scrubbed knobs (steering α), a plain value for occasional config. The
 *    freezing is tested (test/capture.spec.ts "closure values are FROZEN") and
 *    `TORCHLETTE_TAPE_VERIFY=N` is the paranoia backstop that byte-diffs
 *    replays against fresh builds.
 *
 * The body runs EXACTLY ONCE per call, always for real (no probe, no re-run):
 * on a hit it is short-circuited by a sentinel thrown at its LAST internal
 * upload — before any layer compute or in-place state op — inside an aborting
 * scope that reclaims the partial graph. Phase-2a finding
 * (docs/staged-execution-phase1.md §9): never execute a captured body PAST
 * side-effecting ops just to observe coverage — in-place version commits
 * happen eagerly at graph build and no scope abort can roll them back.
 *
 * Single-engine by phase-1 scope. Zero new env flags: capture rides
 * TORCHLETTE_STEP_TAPE (flag off ⇒ transparent pass-through).
 */

import { STEP_TAPE_VERIFY_N } from "../core/step-tape";
import type { Torchlette } from "./torchlette";
import type { Tensor } from "./tensor";

/** Sentinel thrown by the upload interceptor to short-circuit fn on a HIT. */
class CaptureShortCircuit {
  constructor(
    readonly uploads: Array<{ shape: number[]; values: Float32Array }>,
  ) {}
}

export interface CaptureOptions {
  /**
   * Optional STRUCTURAL discriminator folded into the derived appKey — for
   * structure the argument surface can't express (KV bucket length, a cache-
   * instance nonce). NOT for per-step data (tensor args) and NOT a place to
   * smuggle closure values (they belong in the args).
   */
  key?: (...args: unknown[]) => string;
  /** Staging ring depth (output validity window, §4). Default 3. */
  ringDepth?: number;
}

export interface CaptureStats {
  calls: number;
  hits: number;
  traces: number;
  /** Calls that missed on a bucket key while this fn already had warm buckets
   *  — the cold-knob counter (a plain-value arg or opts.key changed). */
  coldMisses: number;
  invalidations: number;
  ready: boolean;
}

let nextCaptureId = 0;

/** A staging-ring slot: the materialized output(s) of one captured call. */
interface RingEntry {
  step: number;
  outputs: Tensor[];
}

export class CapturedFn {
  private readonly id = nextCaptureId++;
  private readonly keyFn?: (...args: unknown[]) => string;
  private readonly ringDepth: number;

  /** Per-bucket count of the fn's INTERNAL uploads (tensorFromArray calls made
   *  inside the body), learned on the first trace of each bucket. */
  private expectedUploads = new Map<string, number>();

  private ring: RingEntry[] = [];
  private stepCounter = 0;

  private stats: CaptureStats = {
    calls: 0,
    hits: 0,
    traces: 0,
    coldMisses: 0,
    invalidations: 0,
    ready: false,
  };

  constructor(
    private readonly api: Torchlette,
    private readonly fn: (...args: never[]) => Tensor | Tensor[],
    opts?: CaptureOptions,
  ) {
    this.keyFn = opts?.key;
    this.ringDepth = Math.max(1, opts?.ringDepth ?? 3);
  }

  /** Derived appKey — the arg-boundary contract made concrete: tensor args
   *  contribute STRUCTURE (shape/dtype/device; their values are slot data),
   *  plain-value args contribute their VALUE (the cold knob), and opts.key
   *  contributes declared structure. */
  private appKey(args: unknown[]): string {
    const argSig = args
      .map((a) => {
        const t = a as Tensor;
        if (t && typeof t === "object" && Array.isArray(t.shape)) {
          return `t[${t.shape.join(",")}:${t.dtype}:${t.device}]`;
        }
        return `v=${stableValueKey(a)}`;
      })
      .join(",");
    const k = this.keyFn ? this.keyFn(...args) : "";
    return `capture:${this.id}:${argSig}:${k}`;
  }

  /** §6 cold knob: drop every skeleton so the next call re-traces. */
  invalidate(): void {
    for (const key of this.expectedUploads.keys()) {
      this.api._invalidateCapture(key);
    }
    this.expectedUploads.clear();
    this.stats.invalidations++;
    this.stats.ready = false;
  }

  stats_(): CaptureStats {
    return {
      ...this.stats,
      ready: this.expectedUploads.size > 0 && this.stats.hits > 0,
    };
  }

  /** Run the body ONCE under the upload interceptor. When `shortCircuit` is
   *  set, the sentinel is thrown at the `expectedUploads`-th internal upload
   *  (the proven-safe point: every decode upload precedes any layer/in-place
   *  op); the throw path reclaims the partial graph by disposing the tracked
   *  wrappers. A body that completes anyway (structural drift: fewer internal
   *  uploads than the recorded bucket expects) returns its real result — a
   *  valid re-record. */
  private runIntercepted(
    body: () => Tensor | Tensor[],
    expectedUploads: number,
    shortCircuit: boolean,
  ): {
    result: Tensor | Tensor[] | null;
    uploads: Array<{ shape: number[]; values: Float32Array }>;
    shortCircuited: boolean;
  } {
    const uploads: Array<{ shape: number[]; values: Float32Array }> = [];
    // Short-circuit path: track every Tensor the partial body creates so the
    // throw can reclaim exactly them — orphan pending nodes would otherwise be
    // force-executed by the next markStep (redundant GPU work + fingerprint
    // pollution of the following full trace). Direct disposal of the tracked
    // wrappers is much cheaper per token than a reclamation-scope open/abort
    // on this hot path. If the body COMPLETES instead (structural drift),
    // nothing is disposed — the real result stands (the correct re-record; fn
    // ran exactly once, so any in-fn state applied exactly once).
    const created: Tensor[] | null = shortCircuit ? [] : null;
    const restore = this.api._installCaptureInterceptor({
      onUpload: (shape, values) => {
        uploads.push({ shape: shape.slice(), values });
        if (shortCircuit && uploads.length >= expectedUploads) {
          throw new CaptureShortCircuit(uploads);
        }
      },
      onWrap: created ? (t) => created.push(t) : undefined,
    });
    let result: Tensor | Tensor[] | null = null;
    try {
      result = body();
    } catch (e) {
      if (!(e instanceof CaptureShortCircuit)) throw e;
      restore(); // stop tracking before the disposal churn
      if (created) for (const t of created) t.dispose();
      return { result: null, uploads: e.uploads, shortCircuited: true };
    } finally {
      restore();
    }
    return { result, uploads, shortCircuited: false };
  }

  async call(args: unknown[]): Promise<Tensor | Tensor[]> {
    this.stats.calls++;
    // Flag off: capture is a transparent pass-through (just run fn).
    if (!this.api._tapeActive()) {
      return this.fn(...(args as never[]));
    }
    const appKey = this.appKey(args);
    this.api._setCaptureTapeContext(appKey, []);

    const known = this.expectedUploads.get(appKey);
    // TAPE_VERIFY=N (§2.4 guard 6, inherited): every Nth call runs the NORMAL
    // path even when a skeleton is ready — the phase-1 executor seam then
    // byte-compares the skeleton it WOULD have replayed against the fresh
    // build (stCaptureCompiledStep's fp + command-stream diff). Shadow mode
    // (N=1) verifies every captured step.
    const doVerify =
      STEP_TAPE_VERIFY_N > 0 && this.stats.calls % STEP_TAPE_VERIFY_N === 0;
    const ready =
      !doVerify && known !== undefined && this.api._tapeReadyFor(appKey);

    if (!ready) {
      // Cold-knob accounting: this bucket is cold while the fn has warm ones.
      // (Verify-forced traces are deliberate, not misses.)
      if (this.stats.hits > 0 && !doVerify) this.stats.coldMisses++;
      const { result, uploads } = this.runIntercepted(
        () => this.fn(...(args as never[])),
        Number.MAX_SAFE_INTEGER,
        false,
      );
      return this.finishTrace(appKey, result, uploads);
    }

    // Fresh tensor-arg values: caller-built pending tensorFromArray args are
    // upload slots (values read synchronously from their un-forced payload);
    // persistent tensor args are external inputs whose STABLE buffer the
    // replayed plan reads live (the warm in-place knob) — nothing to dress.
    // On a HIT the upload args are DONATED (disposed): the replay consumed
    // their values, and the never-consumed pending node would otherwise be
    // force-executed as a wasted mini-plan at every markStep.
    const { uploads: argUploads, donatable } = this.api._captureArgUploads(args);
    const donate = () => {
      for (const t of donatable) t.dispose();
    };

    if (known === 0) {
      // Arg-only fn (all dynamic data enters via args): nothing inside the
      // body to collect — replay directly; the body never runs on a hit.
      const out = await this.api._captureReplay(argUploads);
      if (out !== null) {
        this.stats.hits++;
        donate();
        return this.pushRing(Array.isArray(out) ? out : [out]);
      }
      const re = this.runIntercepted(
        () => this.fn(...(args as never[])),
        Number.MAX_SAFE_INTEGER,
        false,
      );
      return this.finishTrace(appKey, re.result, re.uploads);
    }

    // Body runs EXACTLY ONCE: short-circuit at its last internal upload.
    const { result, uploads, shortCircuited } = this.runIntercepted(
      () => this.fn(...(args as never[])),
      known,
      true,
    );

    if (shortCircuited) {
      const out = await this.api._captureReplay(argUploads.concat(uploads));
      if (out !== null) {
        this.stats.hits++;
        donate();
        return this.pushRing(Array.isArray(out) ? out : [out]);
      }
      // Phase-1 guard declined AFTER the short-circuit — only reachable if the
      // compiled plan invalidated between the validity pre-check (_tapeReadyFor
      // → stTapeReplayValid) and the replay (a mid-call plan eviction). Drop
      // the skeleton and re-run fn for a correct result. NOTE: in-fn JS state
      // (a KV cache-length advance) already applied once in the aborted run,
      // so the re-run double-applies it for THIS one token before the next
      // step re-records — a bounded, plan-invalidation-only transient.
      this.api._invalidateCapture(appKey);
      this.expectedUploads.delete(appKey);
      const re = this.runIntercepted(
        () => this.fn(...(args as never[])),
        Number.MAX_SAFE_INTEGER,
        false,
      );
      return this.finishTrace(appKey, re.result, re.uploads);
    }

    // Structural drift (body completed under short-circuit): real re-record.
    return this.finishTrace(appKey, result, uploads);
  }

  /** Record the bucket's upload count and return the real result via the ring. */
  private finishTrace(
    appKey: string,
    result: Tensor | Tensor[] | null,
    uploads: Array<{ shape: number[]; values: Float32Array }>,
  ): Tensor | Tensor[] {
    this.stats.traces++;
    this.expectedUploads.set(appKey, uploads.length);
    if (result === null) {
      throw new Error("capture: fn produced no result on trace");
    }
    return this.pushRing(Array.isArray(result) ? result : [result]);
  }

  /** Output staging ring (§4): materialized handles valid for `ringDepth`
   *  subsequent captured calls; reading past the window is a LOUD error. */
  private pushRing(outputs: Tensor[]): Tensor | Tensor[] {
    const step = this.stepCounter++;
    const cutoff = step - this.ringDepth;
    while (this.ring.length && this.ring[0].step <= cutoff) {
      const old = this.ring.shift()!;
      for (const t of old.outputs) {
        this.api._markCaptureExpired(t, old.step, step, this.ringDepth);
      }
    }
    this.ring.push({ step, outputs });
    return outputs.length === 1 ? outputs[0] : outputs;
  }
}

/** Stable short key for a plain argument value (the cold knob). Objects/arrays
 *  key by JSON; a hash keeps bucket keys bounded. */
function stableValueKey(v: unknown): string {
  const s =
    typeof v === "number" || typeof v === "boolean" || v == null
      ? String(v)
      : typeof v === "string"
        ? v
        : JSON.stringify(v);
  if (s.length <= 24) return s;
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return `#${(h >>> 0).toString(16)}`;
}

export { CaptureShortCircuit };
