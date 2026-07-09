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

const slotValues = new WeakMap<object, number>();

/** Record `value` as the current host value of the persistent scalar tensor
 *  `owner` (called by the write's author, e.g. Adam.setLR). */
export function noteScalarSlotValue(owner: object, value: number): void {
  slotValues.set(owner, value);
}

/** The current host value for `owner`, if its author notes values. */
export function getScalarSlotValue(owner: object): number | undefined {
  return slotValues.get(owner);
}
