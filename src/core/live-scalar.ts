/**
 * LIVE SCALAR SLOT — the single primitive for per-step optimizer scalars
 * (the LR scheduler's lr, the GradScaler's scale/invScale) delivered to a
 * compiled/replayed training step as DATA.
 *
 * It exposes the per-step-value delivery that batch tensor ARGS get (their
 * fresh `tensorFromArray` value re-dressed onto the recorded plan each replay)
 * to NON-argument writers — internal optimizer state flows like an argument
 * (jax) — WITHOUT adding a fourth mechanism: it rides the step-tape's existing
 * in-plan scalar-write re-dress seam. The design was falsification-checked: the
 * naive "arg channel = a fresh-buffer upload NODE" reading FAILED — a large
 * plan's high-fan-out readers (the packed adam) bind ONE record-time buffer
 * that a separate TAG_WRITE stableBuf never feeds (measured on the 124M model).
 * So the write is an IN-PLACE scatter into a fixed buffer (below).
 *
 * A `LiveScalar` owns ONE persistent f32[1] tensor with a STABLE physical
 * buffer (clause 2: FIXED BUFFER — created once, written IN PLACE, so a
 * compiled replay's consumers bind ONE buffer at record time and read the live
 * value from it every replay; a fresh-buffer upload silently corrupts a large
 * plan's high-fan-out readers, measured on the 124M model). `set(v)` writes `v`
 * into that buffer via an IN-PLACE, GRAPH-ORDERED scatter (clause 1: the write
 * is a plan node ordered against the consumers' reads, NOT a raw out-of-graph
 * queue.writeBuffer — the item-1 uncaptured-divergence cure) and notes the host
 * value here. A compiled replay re-executes the recorded scatter's
 * `tensorFromArray` source from the CURRENT noted value (clause 3: LIVE REPLAY
 * READS — never record-time bytes; the step-tape live-scalar re-dress).
 *
 * This SUBSUMES the ad-hoc `copy_(lrT, tensorFromArray([lr]))` + `noteScalarSlot`
 * delivery that was open-coded in each optimizer (Adam.setLR / setGroupLR): the
 * ONE primitive owns the fixed-buffer tensor, the in-place graph-ordered write
 * (`RuntimeEngine.setScalarInPlace`), and the host-value note. The step-tape's
 * existing scalar-write re-dress (step-tape-replay.ts `scalarDresses`, sourced
 * from `scalar-slots.ts`) carries the value through replays unchanged — the
 * primitive rides that proven seam rather than adding a parallel one.
 */

import type { Tensor } from "../frontend/tensor";
import type { Torchlette } from "../frontend/torchlette";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import { noteScalarSlotValue } from "./scalar-slots";

/**
 * A per-step scalar (or short vector) with a fixed-identity persistent f32[n]
 * tensor, delivered through the graph-ordered in-plan upload channel. Consumers
 * read `.tensor` as an ordinary graph input; the driver calls `.set(v)` each
 * step. n=1 is the LR scheduler's lr / the GradScaler's scale; n=2 is Adam's
 * host-computed bias-correction bc=[bc1,bc2] (fork C) — one primitive, the
 * length is the constructor's.
 */
export class LiveScalar {
  /** The persistent, buffer-stable f32[n] tensor consumers read. */
  readonly tensor: Tensor;
  private _value: number | number[];
  private readonly _len: number;
  // Pin the write chain's short-lived storages across the RUNAHEAD window
  // (the setLR-under-ringK2 STRICT transient that kept the scheduler NOSCHED):
  //  - the scatter SOURCE (setScalarInPlace returns it TRACKED): ownerless it
  //    is rc=0 once its plan claim drops, and it can execute in an EARLY force
  //    while its scatter rides the DEFERRED boundary commit — reachability
  //    destroys it in between and the deferred scatter reads RECLAIMED storage;
  //  - the PRE-WRITE dst handle: consumers queued behind the deferred boundary
  //    (the ring's adamStep) hold materialized refs to it, but the write's
  //    `_updateLazyRef` releases the owner claim immediately — a sweep between
  //    the driver write and the deferred force destroys it under the readers.
  // A short FIFO spans the ring's in-flight window (K≤2, two pins per set,
  // plus slack); drained entries are disposed back to reachability.
  private static readonly PIN_KEEP = 8;
  private readonly _pinRing: RuntimeTensor[] = [];

  constructor(
    private readonly api: Torchlette,
    initial: number | number[],
    device: "cpu" | "webgpu",
  ) {
    // A fixed-buffer persistent f32[n]. registerState() declares it REG state so
    // it survives boundaries (model-level state). Its physical buffer is created
    // once and written IN PLACE for the primitive's life (clause 2 — FIXED
    // BUFFER); the value is delivered via setScalarInPlace. n=1 uses full() (one
    // fillValue, no template-fingerprint thrash); n>1 seeds distinct lanes via
    // tensorFromArray, then rides the same in-place delivery.
    if (typeof initial === "number") {
      this._len = 1;
      this.tensor = api.registerState(api.full([1], initial, { device }));
    } else {
      this._len = initial.length;
      this.tensor = api.registerState(
        api.tensorFromArray(initial, [initial.length], { device }),
      );
    }
    this._value = initial;
    noteScalarSlotValue(this.tensor._unwrap(), initial);
  }

  get value(): number | number[] {
    return this._value;
  }

  /**
   * Deliver `v` as this step's value: an in-place, graph-ordered scatter into
   * the fixed buffer (clause 1) + a registry note for the replay re-dress
   * (clause 3). No new buffer is minted; consumers read the same fixed buffer.
   */
  set(v: number | number[]): void {
    if (typeof v !== "number" && v.length !== this._len)
      throw new Error(
        `LiveScalar.set: length ${v.length} != tensor length ${this._len}`,
      );
    this._value = v;
    const rt = this.api._runtime();
    const inner = this.tensor._unwrap();
    // Pin the PRE-WRITE handle before the write moves the owner claim off it
    // (see `_pinRing` — deferred consumers still read it under runahead).
    const prev = inner.lazyRef;
    if (prev.kind === "materialized") {
      // SIDECAR pin (task #74): shares the persistent scale storage to keep it
      // alive across the runahead window via its own rc + the `_sidecarShare`
      // keep flag the derived classifier reads (task #70 D2 — no owner slot to
      // steal anymore; the pin is a plain owner-SET member with a keep signal).
      this._pin(
        rt.createSidecarFromStorageHandle(
          prev.storage,
          inner.shape,
          inner.device,
          inner.dtype,
        ),
      );
    }
    // Pin the tracked source so a scatter whose execution rides a DEFERRED
    // boundary still finds it alive.
    this._pin(rt.setScalarInPlace(inner, v));
    // Note the host value for the step-tape re-dress (single source at the
    // seam): the recorded in-plan scatter re-executes its tensorFromArray
    // source from THIS value each replay, keyed by the owning tensor identity.
    noteScalarSlotValue(inner, v);
  }

  private _pin(t: RuntimeTensor): void {
    this._pinRing.push(t);
    if (this._pinRing.length > LiveScalar.PIN_KEEP) {
      this._pinRing.shift()?.dispose();
    }
  }
}
