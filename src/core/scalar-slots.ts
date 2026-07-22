/**
 * Scalar-slot value registry — the authoritative HOST source for per-step
 * optimizer scalars delivered to replayed step-tapes.
 *
 * A driver-level scalar write (the LR scheduler's `setLR` → `copy_(lrTensor,
 * tensorFromArray([lr]))`) reaches a replayed training tape as DATA: the
 * skeleton re-dresses the recorded write node's payload each replay. Deriving
 * that value by walking the owner's pending lazy chain is fragile — when the
 * write happens to be FORCED before the replay begins (mixed markStep
 * regimes), the chain is gone and a chain-derived dress goes stale by one
 * step (measured: sporadic 3e-3..0.07 divergence). Single source of truth at
 * the seam instead: the SETTER that changes the value also notes it here, and
 * the replay reads it back by the owning tensor's identity.
 *
 * Keyed weakly by the persistent RuntimeTensor wrapper — the same object the
 * step-tape's owner snapshot resolves (storage-tracker backref), so the two
 * sides agree by construction.
 */

// A slot value is a single f32 (the LR scheduler's lr, the GradScaler's scale)
// OR a short f32 vector (Adam's `[2]` bias-correction bc=[bc1,bc2], fork C) —
// the SAME live-delivery seam carries both; the length is the owner tensor's.
type SlotValue = number | readonly number[];
const slotValues = new WeakMap<object, SlotValue>();

/** Record `value` as the current host value(s) of the persistent scalar/vector
 *  tensor `owner` (called by the write's author, e.g. Adam.setLR / bc delivery). */
export function noteScalarSlotValue(owner: object, value: SlotValue): void {
  slotValues.set(owner, value);
}

/** The current host value(s) for `owner`, if its author notes values. A scalar
 *  owner returns a `number`; a live-vector owner (Adam bc) returns `number[]`. */
export function getScalarSlotValue(owner: object): SlotValue | undefined {
  return slotValues.get(owner);
}
