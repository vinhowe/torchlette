/**
 * crossPlanEdges — THE ONE DERIVED CROSS-PLAN EDGE SET (campaign phase D1,
 * docs/step-data-dependence-design.md §2 / §4).
 *
 * WHAT THIS IS. Element B of the step's data-dependence: by witness time the
 * step object knows the ordered plan sequence; the cross-plan edge set —
 * producer plan → consumer plan, per VALUE — is DERIVABLE from the witnessed
 * streams (§4.1). This module is that ONE derived object. It holds, per PRODUCER
 * value identity `(producer plan fp, ni:oi)`, the set of consumer edges
 * `(consumer plan fp, position)` the witness physically observed reading that
 * value across a plan boundary (§4.1 "autograd cross-plan reads" — the ~47
 * measured `forwardToForce` edges, `[E-1]`).
 *
 * SINGLE SOURCE — ONE PRODUCER, CONSUMERS QUERY (ruling 1, §4.2). The ONLY
 * writer is the step-tape recorder's witness-eligibility path (`step-tape.ts`
 * `reconcileWitnessReads`), which publishes at the SAME K_w=2 moment it
 * publishes the per-producer witnessed harvest set — so the edge set's
 * per-producer keep-set PROJECTION is byte-EQUAL to that shadow oracle by
 * construction (the D1 null gate). The harvest, the planner's result retention,
 * and the observed-liveness predicates become CALLERS of this one query (§4.2).
 *
 * TEMPLATE-GENERATION KEYING (D0 refinement — the campaign's ground truth). A
 * producer's cross-plan read set GROWS as the main plan settles across template
 * transitions (measured: producer `0xa442983c` 97→123→125 pairs as the reader
 * template transitioned `0x991b475f → 0xaded5357 + 0x77663cc5`). The CONSUMER
 * PLAN FP is the generation discriminator (a reader-template transition writes
 * edges under a NEW `consumerFp`), and each edge carries a `lastStep` watermark
 * so a later phase (D2 "derive-through") can drop edges whose consumer fp has
 * fallen out of the current generation. D1 KEEPS the full monotone union
 * (assumption 4 — over-keep is memory, under-keep is a UAF), so the projection
 * matches the monotone-growing shadow oracle exactly; the generation metadata is
 * recorded, not yet used to prune.
 *
 * NULL ON NON-WITNESS PATHS (reify discipline). Nothing is written until a
 * producer reaches K_w=2 witness eligibility; `crossPlanEdges()` returns `null`
 * with no witnessed producer, and every query returns empty. The module owns no
 * GPU memory, references only fingerprints + `ni:oi` coordinates, and adds NO
 * env flag (it rides `TORCHLETTE_STEP_TAPE` with the campaign).
 *
 * A leaf core module: it imports nothing from `step-tape`/executor (both import
 * IT), so there is no cycle.
 */

/** A cross-plan CONSUMER edge of one producer value: which consumer plan read it
 *  (the generation discriminator) and where in that plan's cross-plan read
 *  stream, with the observation watermarks the D2 derive-through will consult. */
export interface CrossPlanEdge {
  /** The consumer plan's execution fingerprint (the template-generation key). */
  readonly consumerFp: number;
  /** Read ordinals within the consumer plan's cross-plan read stream (the
   *  "position" of the schema; sorted, deduped). Multiple ⇒ the consumer read
   *  the value at more than one node position. */
  readonly positions: readonly number[];
  /** Recorder step ordinal at which this edge was FIRST witnessed. */
  readonly firstStep: number;
  /** Recorder step ordinal at which this edge was LAST witnessed (the generation
   *  watermark — D2 drops edges whose consumer has fallen out of the current
   *  generation; unused for pruning in D1). */
  readonly lastStep: number;
}

/** The derived edge set: producer plan fp → value pair "ni:oi" → its consumer
 *  edges. */
export interface CrossPlanEdgeSet {
  /** producerFp → ("ni:oi" → edges). */
  readonly producers: ReadonlyMap<
    number,
    ReadonlyMap<string, readonly CrossPlanEdge[]>
  >;
}

/** Mutable edge accumulator (module-internal; the published object exposes the
 *  read-only projection above). */
interface EdgeAccum {
  positions: Set<number>;
  firstStep: number;
  lastStep: number;
}

/** The store: producerFp → pairKey("ni:oi") → consumerFp → EdgeAccum. Written
 *  ONLY by `publishCrossPlanEdges` (the witness); read by the harvest + the
 *  shadow-diff instrument. */
const store = new Map<number, Map<string, Map<number, EdgeAccum>>>();

/** The producer-edge payload the witness hands to `publishCrossPlanEdges`:
 *  pairKey → (consumerFp → observed positions + watermarks). */
export type ProducerEdgePayload = ReadonlyMap<
  string,
  ReadonlyMap<
    number,
    { positions: ReadonlySet<number>; firstStep: number; lastStep: number }
  >
>;

/**
 * Publish (REPLACE) the derived cross-plan edges for ONE producer template.
 * Called by the step-tape recorder at witness eligibility (K_w=2), for exactly
 * the pair set it publishes to the per-producer harvest oracle — so
 * `crossPlanEdgeKeepSet(producerFp)` equals that oracle's set by construction.
 * A fresh publication REPLACES the producer's prior edges (re-witnessing under a
 * changed declaration re-establishes them), mirroring `setWitnessedHarvest`.
 */
export function publishCrossPlanEdges(
  producerFp: number,
  edges: ProducerEdgePayload,
): void {
  if (edges.size === 0) {
    store.delete(producerFp);
    return;
  }
  const byPair = new Map<string, Map<number, EdgeAccum>>();
  for (const [pairKey, byConsumer] of edges) {
    const consumers = new Map<number, EdgeAccum>();
    for (const [consumerFp, e] of byConsumer) {
      consumers.set(consumerFp, {
        positions: new Set(e.positions),
        firstStep: e.firstStep,
        lastStep: e.lastStep,
      });
    }
    byPair.set(pairKey, consumers);
  }
  store.set(producerFp, byPair);
}

/**
 * THE HARVEST QUERY (§4.2): the keep-set for a producer = the set of value pairs
 * "ni:oi" that have AT LEAST ONE cross-plan consumer edge. This is the
 * projection observed-liveness consults; it EQUALS the per-producer witnessed
 * harvest set (the D1 shadow-diff invariant). Empty when the producer is not
 * (yet) witnessed.
 */
export function crossPlanEdgeKeepSet(producerFp: number): Set<string> {
  const out = new Set<string>();
  const byPair = store.get(producerFp);
  if (!byPair) return out;
  for (const [pairKey, consumers] of byPair) {
    if (consumers.size > 0) out.add(pairKey);
  }
  return out;
}

/**
 * THE OVERLAY-RELEASE QUERY (§5): does this producer VALUE pair have a witnessed
 * cross-plan consumer edge to a template OTHER than `excludeFp`? The stage-3 B
 * overlay-release consults this to DECLINE a claim whose value a lowered consumer
 * still re-reads — the derived edge set superseding the `graphHeldAt` heuristic
 * (a graphHeld=false forwardToForce activation with a witnessed later reader).
 * A cheap targeted probe (no projection allocation), the single-source seam the
 * claim reads instead of independently reconstructing liveness.
 */
export function crossPlanEdgeHasOtherConsumer(
  producerFp: number,
  pairKey: string,
  excludeFp: number,
): boolean {
  const consumers = store.get(producerFp)?.get(pairKey);
  if (!consumers) return false;
  for (const consumerFp of consumers.keys()) {
    if (consumerFp !== excludeFp) return true;
  }
  return false;
}

/**
 * The derived edge set — the object the doc's schema names. Returns `null` with
 * no witnessed producer (the reify-discipline null on non-witness paths). A
 * read-only PROJECTION of the store.
 */
export function crossPlanEdges(): CrossPlanEdgeSet | null {
  if (store.size === 0) return null;
  const producers = new Map<number, Map<string, CrossPlanEdge[]>>();
  for (const [producerFp, byPair] of store) {
    const pairs = new Map<string, CrossPlanEdge[]>();
    for (const [pairKey, byConsumer] of byPair) {
      const edges: CrossPlanEdge[] = [];
      for (const [consumerFp, e] of byConsumer) {
        edges.push({
          consumerFp,
          positions: [...e.positions].sort((a, b) => a - b),
          firstStep: e.firstStep,
          lastStep: e.lastStep,
        });
      }
      edges.sort((a, b) => a.consumerFp - b.consumerFp);
      pairs.set(pairKey, edges);
    }
    producers.set(producerFp, pairs);
  }
  return { producers };
}

/**
 * SHADOW-DIFF SUPPORT: every witnessed producer's keep-set projection (the
 * per-producer harvest oracle comparison target). producerFp → sorted "ni:oi"
 * pairs.
 */
export function getAllCrossPlanEdgeKeepSets(): Map<number, string[]> {
  const out = new Map<number, string[]>();
  for (const [producerFp, byPair] of store) {
    const s = new Set<string>();
    for (const [pairKey, consumers] of byPair) {
      if (consumers.size > 0) s.add(pairKey);
    }
    out.set(producerFp, [...s].sort());
  }
  return out;
}

/** Telemetry: sizes of the derived object (for the D1 gate's edge-set report). */
export function crossPlanEdgeStats(): {
  producers: number;
  pairs: number;
  edges: number;
  positions: number;
} {
  let pairs = 0;
  let edges = 0;
  let positions = 0;
  for (const byPair of store.values()) {
    for (const consumers of byPair.values()) {
      pairs++;
      for (const e of consumers.values()) {
        edges++;
        positions += e.positions.size;
      }
    }
  }
  return {
    producers: store.size,
    pairs,
    edges,
    positions,
  };
}

/** Per-producer edge-set sizes (for the D1 gate: "the edge-set sizes measured
 *  per cell"). producerFp → { pairs, edges, consumers }. */
export function crossPlanEdgePerProducer(): Map<
  number,
  { pairs: number; edges: number; consumers: number }
> {
  const out = new Map<
    number,
    { pairs: number; edges: number; consumers: number }
  >();
  for (const [producerFp, byPair] of store) {
    const rec = { pairs: 0, edges: 0, consumers: 0 };
    const consumerSet = new Set<number>();
    for (const consumers of byPair.values()) {
      rec.pairs++;
      for (const cfp of consumers.keys()) {
        rec.edges++;
        consumerSet.add(cfp);
      }
    }
    rec.consumers = consumerSet.size;
    out.set(producerFp, rec);
  }
  return out;
}

/** Clear all derived edges (test/telemetry reset; mirrors `stResetAll`). */
export function resetCrossPlanEdges(): void {
  store.clear();
}
