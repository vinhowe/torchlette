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
import { noteScalarSlotValue } from "./scalar-slots";

/**
 * A per-step scalar with a fixed-identity persistent f32[1] tensor, delivered
 * through the graph-ordered in-plan upload channel. Consumers read `.tensor`
 * as an ordinary graph input; the driver calls `.set(v)` each step.
 */
export class LiveScalar {
  /** The persistent, buffer-stable f32[1] tensor consumers read. */
  readonly tensor: Tensor;
  private _value: number;

  constructor(
    private readonly api: Torchlette,
    initial: number,
    device: "cpu" | "webgpu",
  ) {
    // full(): a fixed-buffer persistent f32[1]. persist() adopts it into the
    // step snapshot so it survives boundaries (model-level state). Its physical
    // buffer is created once and written IN PLACE for the scalar's life (clause
    // 2 — FIXED BUFFER); the value is delivered via setScalarInPlace, so the
    // fillValue never varies (no template-fingerprint thrash from full()).
    this.tensor = api.persist(api.full([1], initial, { device }));
    this._value = initial;
    noteScalarSlotValue(this.tensor._unwrap(), initial);
  }

  get value(): number {
    return this._value;
  }

  /**
   * Deliver `v` as this step's value: an in-place, graph-ordered scatter into
   * the fixed buffer (clause 1) + a registry note for the replay re-dress
   * (clause 3). No new buffer is minted; consumers read the same fixed buffer.
   */
  set(v: number): void {
    this._value = v;
    this.api._runtime().setScalarInPlace(this.tensor._unwrap(), v);
    // Note the host value for the step-tape re-dress (single source at the
    // seam): the recorded in-plan scatter re-executes its tensorFromArray
    // source from THIS value each replay, keyed by the owning tensor identity.
    noteScalarSlotValue(this.tensor._unwrap(), v);
  }
}
