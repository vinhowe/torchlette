/**
 * crossPlanEdges — REDUCED TO INERT STUBS (P4b-R R3.3).
 *
 * The derived cross-plan edge set was populated ONLY by the step-tape recorder's
 * witness path (`publishCrossPlanEdges`, called from the deleted step-tape.ts at
 * K_w=2 witness eligibility). With the tape deleted the store is always empty, so
 * the two queries below return their constant tape-off values — EXACTLY what they
 * returned before (an unwitnessed producer always yielded an empty keep-set and
 * `false` for "has other consumer").
 *
 * These two functions survive because their consumers are KEPT and NOT part of
 * the tape subsystem:
 *   - `crossPlanEdgeKeepSet` — read by `observed-liveness.ts` (byte-untouched,
 *     decode-live harvest prune): with an empty edge store the harvest keeps the
 *     `needed ∪ keepAlways` set, the tape-off behaviour.
 *   - `crossPlanEdgeHasOtherConsumer` — read by `executor.ts` (the overlay-release
 *     UAF guard): with no witnessed edges it declines to find another consumer,
 *     the tape-off behaviour (there was never a witnessed edge without the tape).
 *
 * The store, the witness writer (`publishCrossPlanEdges`), the projection
 * (`crossPlanEdges`), and the telemetry (`crossPlanEdgeStats` /
 * `getAllCrossPlanEdgeKeepSets` / `crossPlanEdgePerProducer` / `resetCrossPlanEdges`)
 * were all deleted with the tape — nothing outside the deleted tape/instrument
 * tools consumed them.
 */

/**
 * THE HARVEST QUERY: the keep-set for a producer. Always empty now — there is no
 * witnessed cross-plan edge store to project (the tape recorder that populated it
 * is deleted). The consuming harvest prune then keeps `needed ∪ keepAlways`, the
 * exact tape-off behaviour.
 */
export function crossPlanEdgeKeepSet(_producerFp: number): Set<string> {
  return new Set<string>();
}

/**
 * THE OVERLAY-RELEASE QUERY: does this producer value pair have a witnessed
 * cross-plan consumer edge to a template OTHER than `excludeFp`? Always false now
 * — with no witnessed edges the overlay-release guard never finds another
 * consumer, the exact tape-off behaviour.
 */
export function crossPlanEdgeHasOtherConsumer(
  _producerFp: number,
  _pairKey: string,
  _excludeFp: number,
): boolean {
  return false;
}
