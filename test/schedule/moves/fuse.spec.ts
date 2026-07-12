/**
 * fuseGesture DRIVER unit gate (P2 wave B deliverable 3).
 *
 * The S3 composite transaction at the schedule/semantic-region altitude: binary,
 * validate-interior → drive membership through the re-record channel → attach
 * state → ONE FuseProvenance → rollback on realization failure; defer realization
 * to the final state of a chained transaction (fuse ×2).
 */

import { describe, expect, it } from "vitest";
import { naiveAttentionComposition } from "../../../src/schedule/attention-skeleton";
import type { FuseGesture } from "../../../src/schedule/moves/fuse";
import {
  composeInterior,
  fuseChain,
  fuseGesture,
  makeInMemoryChannel,
  mintRegion,
} from "../../../src/schedule/moves/fuse";

function naiveGesture(): FuseGesture {
  const c = naiveAttentionComposition(64);
  return {
    a: c.qkT.region,
    b: c.softmax.region,
    aState: c.qkT.state,
    bState: c.softmax.state,
    islandFlow: c.islandFlow,
  };
}

describe("fuseGesture — the composite transaction", () => {
  it("commits a legal binary fuse: mints region', attaches, ONE provenance", () => {
    const g = naiveGesture();
    const outcome = fuseGesture(g);
    expect(outcome.kind).toBe("committed");
    if (outcome.kind !== "committed") throw new Error("unreachable");
    // region' is minted from a+b (R8 — the partition mints it).
    expect(outcome.result.region).toBe(mintRegion(g.a, g.b));
    expect(outcome.result.state.region).toBe(outcome.result.region);
    // ONE provenance carrying BOTH hashes.
    expect(outcome.result.provenance.boundaryHash).toContain("boundary");
    expect(outcome.result.provenance.semanticHash).toMatch(/^[0-9a-f]{32}$/);
    expect(outcome.result.provenance.compilationHash).toMatch(/^[0-9a-f]{32}$/);
    // The inverse payload carries the split cut's prior regions (undo = split).
    expect(outcome.result.provenance.inverse.priorRegions).toEqual([g.a, g.b]);
  });

  it("refuses at MERGE when the union is not convex (no island-flow edge)", () => {
    const g = naiveGesture();
    // strip the island-flow edge between a and b → non-convex.
    const noFlow: FuseGesture = { ...g, islandFlow: [] };
    const outcome = fuseGesture(noFlow);
    expect(outcome.kind).toBe("refused");
    if (outcome.kind === "refused") {
      expect(outcome.stage).toBe("merge");
      expect(outcome.code).toBe("MERGE_REFUSED");
    }
  });

  it("refuses at VALIDATE-INTERIOR when the proposed interior has no store", () => {
    const g = naiveGesture();
    const emptyInterior = { ...g.aState.semantic, stores: [], bodies: [] };
    const outcome = fuseGesture({ ...g, proposedInterior: emptyInterior });
    expect(outcome.kind).toBe("refused");
    if (outcome.kind === "refused") {
      expect(outcome.stage).toBe("validate-interior");
      expect(outcome.code).toBe("INTERIOR_ILLEGAL");
    }
  });

  it("ROLLS BACK on realization failure (channel decision discarded)", () => {
    const g = naiveGesture();
    const channel = makeInMemoryChannel();
    const outcome = fuseGesture(g, {
      channel,
      realize: () => {
        throw new Error("capability absent");
      },
    });
    expect(outcome.kind).toBe("refused");
    if (outcome.kind === "refused") {
      expect(outcome.stage).toBe("realize");
      expect(outcome.code).toBe("REALIZATION_REFUSED");
    }
    // The membership decision was rolled back — nothing left requested.
    expect(channel.requested.size).toBe(0);
  });

  it("a committed fuse leaves the requested membership decision in the channel", () => {
    const g = naiveGesture();
    const channel = makeInMemoryChannel();
    fuseGesture(g, { channel });
    expect(channel.requested.size).toBe(1);
  });
});

describe("fuseChain — 3→1 as fuse ×2, realization deferred to the final state", () => {
  it("commits both merges; realizes ONCE at the final state", () => {
    const c = naiveAttentionComposition(64);
    const realizedStates: string[] = [];
    const outcome = fuseChain(
      [
        { id: c.qkT.region, state: c.qkT.state },
        { id: c.softmax.region, state: c.softmax.state },
        { id: c.pv.region, state: c.pv.state },
      ],
      c.islandFlow,
      { realize: (s) => realizedStates.push(s.region as unknown as string) },
    );
    expect(outcome.kind).toBe("committed");
    if (outcome.kind !== "committed") throw new Error("unreachable");
    // Two binary merges (3→1 is fuse ×2, §5 ruling 1).
    expect(outcome.commits).toHaveLength(2);
    // Realized ONCE, at the FINAL state (§5 ruling 2 — deferred realization).
    expect(realizedStates).toHaveLength(1);
    expect(realizedStates[0]).toBe(outcome.final.region as unknown as string);
  });

  it("rolls back the whole chain if the final realization fails", () => {
    const c = naiveAttentionComposition(64);
    const channel = makeInMemoryChannel();
    const outcome = fuseChain(
      [
        { id: c.qkT.region, state: c.qkT.state },
        { id: c.softmax.region, state: c.softmax.state },
        { id: c.pv.region, state: c.pv.state },
      ],
      c.islandFlow,
      {
        channel,
        realize: () => {
          throw new Error("final realize refused");
        },
      },
    );
    expect(outcome.kind).toBe("refused");
    // Every committed merge's region rolled back.
    expect(channel.requested.size).toBe(0);
  });
});

describe("composeInterior — the default fused interior", () => {
  it("unions bodies, keeps the consumer's store, and records the intermediate as no-mat", () => {
    const c = naiveAttentionComposition(64);
    const interior = composeInterior(
      c.qkT.state.semantic,
      c.softmax.state.semantic,
    );
    // Bodies of both regions are present.
    expect(interior.bodies.length).toBe(
      c.qkT.state.semantic.bodies.length +
        c.softmax.state.semantic.bodies.length,
    );
    // The intermediate (qkT's store target) is now a no-materialization edge.
    expect(interior.noMaterialization.length).toBeGreaterThan(
      c.qkT.state.semantic.noMaterialization.length +
        c.softmax.state.semantic.noMaterialization.length,
    );
  });
});
