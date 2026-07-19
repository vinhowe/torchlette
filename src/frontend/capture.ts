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
  /**
   * Staging ring depth K.
   *  - Decode (default 3): output-validity window — a handle is valid for K
   *    subsequent captured calls; reading past it is a LOUD error.
   *  - Training (default 2): the RUNAHEAD depth (inc-3, docs/staged-execution-
   *    phase2b.md §2). The driver builds+submits up to K steps ahead while the
   *    ring holds K deferred step-boundary fences; the (K+1)-th call blocks on
   *    the oldest (backpressure). G0(b): K=2 saturates the GPU-bound floor —
   *    K>2 buys zero throughput, only K× in-flight memory. No pressure-reactive
   *    automation; K is this explicit knob.
   */
  ringDepth?: number;
  /**
   * TRAINING mode (2b): the body is a whole training step (forward + backward +
   * optimizer) and MAY be async (`await loss.backward()`). Differences from the
   * decode default: the body is NEVER short-circuited — on a MISS it runs to
   * completion (advancing optimizer/param state exactly once); on a HIT it is
   * NOT run at all (the recorded multi-plan sequence replays, advancing state
   * via its in-place ops). The batch enters as tensor ARGS (warm slots); the
   * returned loss maps to an output node harvested from the replayed plans.
   * The hit path queues the implied step boundary the un-run opt.step() would
   * have. See docs/staged-execution-phase2b.md §1/surface-4.
   */
  training?: boolean;
  /**
   * RUNAHEAD (inc-3): the ring OWNS the step boundary. The driver must NOT call
   * markStep per step and must `await step.drain()` at the end of the loop. The
   * ring defers each step's boundary fence+sweep K deep (backpressure at the
   * top of the (K+1)-th call), so CPU builds+submits step N+1 while GPU drains
   * step N — the G0(b) ~30% training-wall win. Only meaningful with
   * `training: true`. When OFF (default), the ring is the 2a output-validity
   * window and the driver owns markStep (the serial 2a/2b path, bit-identical).
   */
  runahead?: boolean;
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

/** A staging-ring slot: the materialized output(s) of one captured call.
 *  `settle` (training/runahead only) is the DEFERRED step-boundary fence+sweep
 *  this step's `markStep` would have run — held in the ring so CPU can build
 *  step N+1 while GPU drains step N, and awaited on backpressure / drain. Decode
 *  entries have no settle (2a's pure output-validity window). */
interface RingEntry {
  step: number;
  outputs: Tensor[];
  settle?: () => Promise<void>;
  /** True once `settle` has run (so drain/expiry never double-fence). */
  settled?: boolean;
}

export class CapturedFn {
  private readonly id = nextCaptureId++;
  private readonly keyFn?: (...args: unknown[]) => string;
  private readonly ringDepth: number;
  private readonly training: boolean;
  private readonly runahead: boolean;

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
    private readonly fn: (
      ...args: never[]
    ) => Tensor | Tensor[] | Promise<Tensor | Tensor[]>,
    opts?: CaptureOptions,
  ) {
    this.keyFn = opts?.key;
    this.training = opts?.training === true;
    this.runahead = this.training && opts?.runahead === true;
    // Ring depth is the RUNAHEAD knob for training (G0b: K=2 saturates the
    // GPU-bound floor — one step in flight hides all overlappable CPU behind
    // the GPU fence; K>2 buys zero throughput, only memory). For decode it is
    // the 2a output-validity window (default 3). `ringDepth` overrides either.
    this.ringDepth = Math.max(1, opts?.ringDepth ?? (this.training ? 2 : 3));
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
  private async runIntercepted(
    body: () => Tensor | Tensor[] | Promise<Tensor | Tensor[]>,
    expectedUploads: number,
    shortCircuit: boolean,
  ): Promise<{
    result: Tensor | Tensor[] | null;
    uploads: Array<{ shape: number[]; values: Float32Array }>;
    shortCircuited: boolean;
  }> {
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
      result = await body();
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

    if (this.training) return this.callTraining(appKey, args, ready);

    if (!ready) {
      // Cold-knob accounting: this bucket is cold while the fn has warm ones.
      // (Verify-forced traces are deliberate, not misses.)
      if (this.stats.hits > 0 && !doVerify) this.stats.coldMisses++;
      const { result, uploads } = await this.runIntercepted(
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
        return this.pushRing(out);
      }
      const re = await this.runIntercepted(
        () => this.fn(...(args as never[])),
        Number.MAX_SAFE_INTEGER,
        false,
      );
      return this.finishTrace(appKey, re.result, re.uploads);
    }

    // Body runs EXACTLY ONCE: short-circuit at its last internal upload.
    const { result, uploads, shortCircuited } = await this.runIntercepted(
      () => this.fn(...(args as never[])),
      known,
      true,
    );

    if (shortCircuited) {
      const out = await this.api._captureReplay(argUploads.concat(uploads));
      if (out !== null) {
        this.stats.hits++;
        donate();
        return this.pushRing(out);
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
      const re = await this.runIntercepted(
        () => this.fn(...(args as never[])),
        Number.MAX_SAFE_INTEGER,
        false,
      );
      return this.finishTrace(appKey, re.result, re.uploads);
    }

    // Structural drift (body completed under short-circuit): real re-record.
    return this.finishTrace(appKey, result, uploads);
  }

  /**
   * TRAINING whole-step capture (2b). The body runs a full step (forward +
   * backward + optimizer) and is either replayed WITHOUT running (hit) or run
   * to completion (miss/trace) — never short-circuited (surface 4: the body's
   * optimizer state must advance exactly once, and the replay advances it via
   * the recorded in-place ops).
   */
  private async callTraining(
    appKey: string,
    args: unknown[],
    ready: boolean,
  ): Promise<Tensor | Tensor[]> {
    // RUNAHEAD ring (inc-3): engaged only when the driver RELINQUISHES the step
    // boundary to the ring — i.e. it does NOT call markStep per step and instead
    // calls `drain()` at the end. `runahead` (opt-in on the callable) turns the
    // ring's deferred `settle` on. When OFF (the serial 2a/2b driver that owns
    // markStep itself), the ring is the pure 2a output-validity window — the
    // ring never runs a settle, so there is no double-boundary. This keeps the
    // existing serial 2b path bit-identical while the runahead path defers.
    if (this.runahead) {
      // BACKPRESSURE AT THE TOP (§2b (b)): before submitting THIS step, if the
      // ring is full, fence+sweep the OLDEST in-flight step. Mirrors the real
      // driver's `scaler.resolveDeferred()→markStep()` at the top of a step:
      // the oldest boundary must complete BEFORE this step's work is submitted,
      // or the sweep's destroys race this step's in-flight buffers ("used in
      // submit while destroyed"). Keeps ≤ K steps un-fenced.
      await this.backpressure();
    }

    // Batch (x, y) are tensor ARGS → warm upload slots. Persistent tensor args
    // (params) ride their stable buffers live. Donated on a hit.
    const { uploads: argUploads, donatable } = this.api._captureArgUploads(args);

    if (ready) {
      const out = await this.api._captureReplay(argUploads);
      if (out !== null) {
        // HIT: the body never ran. Advance the implied step boundary the un-run
        // opt.step() would have queued (surface 4 — superseded by the driver's
        // markStep in the serial path, or by the ring's settle under runahead).
        this.stats.hits++;
        for (const t of donatable) t.dispose();
        if (this.runahead) {
          // The ring OWNS this step's boundary: a deferred commit the ring runs
          // K steps later (backpressure/drain) so the driver never awaits a
          // per-step fence. Commit FIRST (it takes the fresh step snapshot),
          // THEN PIN the harvested output(s) INTO that new snapshot (§2 "the
          // ring PINS each in-flight step's output buffers until its fence") —
          // persisting before the commit would adopt into the snapshot the
          // commit immediately replaces, so the loss buffer would still be swept
          // and the driver's K-steps-later readback reads garbage.
          const settle = await this.api._deferBoundaryCommit();
          for (const t of out) {
            this.api.registerState(t);
            // Blocker #1: stage each scalar output into a POOL-EXCLUDED
            // readback buffer NOW (queue order — before any newer step can
            // rebind the live planner slot). The K-behind read then returns
            // this step's value and waits only for this step's GPU.
            t._stagedScalarRead = await this.api._startRingScalarReadback(t);
          }
          return this.pushRing(out, settle);
        }
        this.api.queueStepBoundary();
        return this.pushRing(out);
      }
      // Guard declined post-readiness (mid-call plan eviction): drop + re-record.
      this.api._invalidateCapture(appKey);
      this.expectedUploads.delete(appKey);
    } else if (this.stats.hits > 0) {
      this.stats.coldMisses++;
    }

    // Declare the tensor ARG nodes (batch x/y) so the promoted skeleton marks
    // only those upload slots warm (surface 4). Declared before the body so the
    // arg nodes are captured; consumed at the promote boundary.
    this.api._declareCaptureArgNodes(args);
    // Mark the BODY-BEGIN boundary: plans captured before this point this step
    // are driver-level pre-body work (scheduler lr writes) — recorded for
    // structure, never replayed (the driver re-executes them for real).
    this.api._markCaptureBodyBegin();
    // TRACE / MISS: run the whole body for real (state advances exactly once).
    // Under TORCHLETTE_WHOLE_STEP the body runs inside the whole-step trace
    // scope so its backward defers the grad-write force to the ring's boundary
    // commit — forward+backward+optimizer become ONE forced plan (P1,
    // docs/step-function-compiler-design.md §3.1). No-op when the flag is off
    // (the scope only gates deferral). A HIT never enters here (body un-run).
    const { result, uploads } = await this.runIntercepted(
      async () => {
        this.api._enterWholeStep();
        try {
          return await this.fn(...(args as never[]));
        } finally {
          this.api._exitWholeStep();
        }
      },
      Number.MAX_SAFE_INTEGER,
      false,
    );
    // Declare the returned loss node(s) so a promoted skeleton knows which plan
    // node to harvest on a hit (surface 3).
    if (result !== null) {
      this.api._declareCaptureOutputs(Array.isArray(result) ? result : [result]);
    }
    // Under runahead the body ran for real; open the SAME deferred boundary so a
    // miss/trace entry pipelines identically to a hit (K is a pure knob). Serial:
    // no settle — the driver owns markStep (2a/2b baseline, bit-identical).
    // Commit FIRST (fresh snapshot), THEN pin the output into it (see hit path).
    const settle = this.runahead
      ? await this.api._deferBoundaryCommit()
      : undefined;
    if (this.runahead && result !== null) {
      for (const t of Array.isArray(result) ? result : [result]) {
        this.api.registerState(t);
        // Blocker #1 (see hit path): stage the scalar output pool-excluded.
        t._stagedScalarRead = await this.api._startRingScalarReadback(t);
      }
    }
    return this.finishTrace(appKey, result, uploads, settle);
  }

  /** Record the bucket's upload count and return the real result via the ring.
   *  On the training path a `settle` (the deferred step-boundary markStep) is
   *  supplied so a miss/trace entry pipelines identically to a hit — the driver
   *  never awaits a per-step fence. */
  private finishTrace(
    appKey: string,
    result: Tensor | Tensor[] | null,
    uploads: Array<{ shape: number[]; values: Float32Array }>,
    settle?: () => Promise<void>,
  ): Tensor | Tensor[] {
    this.stats.traces++;
    this.expectedUploads.set(appKey, uploads.length);
    if (result === null) {
      throw new Error("capture: fn produced no result on trace");
    }
    return this.pushRing(Array.isArray(result) ? result : [result], settle);
  }

  /**
   * BACKPRESSURE (training/runahead only): before THIS captured call submits,
   * if the ring already holds ≥ K in-flight steps, fence+sweep the OLDEST
   * (run its deferred `settle` = gen-scoped boundary commit) and expire+shift
   * it. Called at the TOP of a captured call (§2b (b)), so the oldest step's
   * boundary completes BEFORE this step's work is submitted — no sweep-vs-submit
   * race. Decode entries carry no settle and are never held here (their window
   * is enforced at push).
   */
  private async backpressure(): Promise<void> {
    while (this.ring.length >= this.ringDepth) {
      const old = this.ring.shift()!;
      if (old.settle && !old.settled) {
        old.settled = true;
        await old.settle();
      }
      for (const t of old.outputs) {
        // Reclaim a never-read staging buffer (cached finish is idempotent —
        // a no-op if the driver already read it; the map resolves promptly
        // because the settle above fenced this step). Fire-and-forget: the
        // buffer's destroy rides deferredDestroy either way.
        if (t._stagedScalarRead) void t._stagedScalarRead().catch(() => {});
        this.api._markCaptureExpired(t, old.step, this.stepCounter, this.ringDepth);
      }
    }
  }

  /**
   * Output staging ring. Two regimes, ONE structure:
   *
   * - **Decode (2a):** materialized handles valid for `ringDepth` subsequent
   *   captured calls; reading past the window is a LOUD error. No `settle` —
   *   the decode driver owns its markStep; the ring only bounds validity, and
   *   the expiry is enforced HERE (at push, no backpressure fence).
   * - **Training runahead (inc-3):** `settle` is the DEFERRED step-boundary
   *   fence+sweep, held in the entry. The fence gate runs at the TOP of the NEXT
   *   over-full call (`backpressure()`), not here — so CPU can build+submit up
   *   to K steps ahead of the GPU (G0b's runahead).
   */
  private pushRing(
    outputs: Tensor[],
    settle?: () => Promise<void>,
  ): Tensor | Tensor[] {
    const step = this.stepCounter++;
    if (!settle) {
      // Decode window: expire (no fence) any entry older than the K-window.
      const cutoff = step - this.ringDepth;
      while (this.ring.length && this.ring[0].step <= cutoff) {
        const old = this.ring.shift()!;
        for (const t of old.outputs) {
          this.api._markCaptureExpired(t, old.step, step, this.ringDepth);
        }
      }
    }
    this.ring.push({ step, outputs, settle });
    return outputs.length === 1 ? outputs[0] : outputs;
  }

  /**
   * Drain the ring (user stops training / awaits a handle / process teardown,
   * §2b design (d)): fence every in-flight step IN ORDER (oldest-first), running
   * each deferred `settle` so its boundary sweep completes past its buffers' last
   * GPU use, then clear the ring. A captured step is atomic (its whole plan
   * sequence submitted, or the fallback ran), so no partial-step state exists;
   * the only in-flight thing is submitted GPU work + unread ring outputs.
   * Draining = awaiting every un-fenced entry's settle. Idempotent.
   */
  async drain(): Promise<void> {
    let settled = false;
    while (this.ring.length) {
      const old = this.ring.shift()!;
      if (old.settle && !old.settled) {
        old.settled = true;
        await old.settle();
        settled = true;
      }
    }
    // Full quiescent point (runahead only): nothing is in flight after the
    // settles, so the pool bookkeeping the per-settle isolated fences skip
    // (#84-safety) is legal here — fence everything, promote pendingRelease,
    // fire pending destroys. Idempotent (a second drain settles nothing).
    if (settled && this.runahead) await this.api._ringQuiesce();
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
