/**
 * Task #97 — LIVENESS UNIFICATION: the derived oracle at the overlay-release
 * seam.
 *
 * Stage-3 B (observed-liveness) may hand a producer's registry entry to a
 * consumer's temps (overlay-release) when its EMPIRICAL last-reader observation
 * concludes that consumer is the producer's final reader this step. That
 * observation is fed ONLY by the compiled external-slot seam (observeConsumed);
 * a consumer that reads the producer LOWERED — the canonical case is the
 * BACKWARD pass re-reading a saved-for-backward forward activation, resolved
 * through getInputStorage — is structurally invisible to it. The empirical
 * "last reader" can therefore be WRONG, and the overlay clobbers a value
 * backward still needs. On the recorded build the second (compiled) consumer
 * hit the compiled-only guardMiss recovery net; delete the recorded build and
 * the second consumer falls to lowered → a hard `[lifetime] reading
 * step-globally RELEASED (stage-3 B)` throw.
 *
 * The fix DERIVES the cross-plan liveness stage-3 B was guessing: a producer
 * whose value is GRAPH-HELD (saved for backward — ∃ live `_graphRetained`
 * clone, G(s)>0) WILL be re-read, so its entry is never overlay-released. This
 * gate proves the single-source derived query `storageTracker.graphHeldAt`
 * distinguishes the graph-held (protect) from the boundary-dead (releasable)
 * class — the exact predicate the claim seam consults.
 *
 * FAILING-FIRST: pre-fix `graphHeldAt` does not exist; the whole-workload proof
 * is `test/compiled-plan-parity.spec.ts` + the observed-liveness gate-2
 * trajectory running WITHOUT the recorded build (the harvest deletion re-applied
 * in stage 3) — this unit gate pins the derivation the claim seam relies on.
 */
import { afterEach, describe, expect, it } from "vitest";
import { rcRetain } from "../src/graph/refcount";
import { storageTracker } from "../src/graph/storage-tracker";
import type { StorageHandle } from "../src/graph/types";

let nextId = 900_000; // high base to avoid colliding with any real storage ids

function makeStorage(baseStorageId?: number): StorageHandle {
  const id = nextId++;
  const sh = {
    id,
    device: "cpu",
    backendTensor: { shape: [4], buffer: {} },
    baseStorageId,
  } as unknown as StorageHandle;
  storageTracker.register(sh);
  rcRetain(id, "test.graphHeld");
  return sh;
}

/** A synthetic wrapper carrying the `_graphRetained` flag — exactly what
 *  `_cloneForRetention` produces for a saved-for-backward value. */
function graphRetainedOwner(): object {
  return { _graphRetained: true };
}
function plainOwner(): object {
  return { _graphRetained: false };
}

describe("task #97 — derived overlay-release oracle (graphHeldAt)", () => {
  const owners: object[] = []; // keep strong refs so the WeakRef owner set holds

  afterEach(() => {
    owners.length = 0;
  });

  it("a storage with a live _graphRetained owner is graph-held (protect)", () => {
    const sh = makeStorage();
    const o = graphRetainedOwner();
    owners.push(o);
    storageTracker.trackTensor(sh.id, o);
    expect(storageTracker.graphHeldAt(sh.id)).toBe(true);
  });

  it("a storage with only a plain owner is NOT graph-held (boundary-dead — releasable)", () => {
    const sh = makeStorage();
    const o = plainOwner();
    owners.push(o);
    storageTracker.trackTensor(sh.id, o);
    expect(storageTracker.graphHeldAt(sh.id)).toBe(false);
  });

  it("a VIEW whose flattened base is graph-held is itself graph-held (backward reads through the view)", () => {
    const base = makeStorage();
    const bo = graphRetainedOwner();
    owners.push(bo);
    storageTracker.trackTensor(base.id, bo);

    // A depth-2 view chain over the graph-held base. Its own owners are plain,
    // but backward reads the base value through it → must be protected.
    const mid = makeStorage(base.id);
    const view = makeStorage(mid.id);
    const vo = plainOwner();
    owners.push(vo);
    storageTracker.trackTensor(view.id, vo);

    expect(storageTracker.graphHeldAt(view.id)).toBe(true);
  });

  it("a view over a NON-graph-held base is not graph-held", () => {
    const base = makeStorage();
    const bo = plainOwner();
    owners.push(bo);
    storageTracker.trackTensor(base.id, bo);
    const view = makeStorage(base.id);
    const vo = plainOwner();
    owners.push(vo);
    storageTracker.trackTensor(view.id, vo);
    expect(storageTracker.graphHeldAt(view.id)).toBe(false);
  });

  it("an unknown storage id is not graph-held (no owners, no base)", () => {
    expect(storageTracker.graphHeldAt(123_456_789)).toBe(false);
  });
});
