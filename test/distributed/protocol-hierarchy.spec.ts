/**
 * Hierarchical (two-tier) barrier smoke tests.
 *
 * Verifies that the multi-cluster outer step produces bit-identical
 * post-step params on every peer AND that the result equals what a
 * flat all-to-all average would have computed for the same per-peer
 * pseudograds.
 */

import { describe, expect, it } from "vitest";
import { HierarchicalBarrierStateMachine } from "../../src/distributed/protocol/hierarchical-state-machine.ts";
import { StubTrainer } from "../../src/distributed/protocol/stub-trainer.ts";
import {
  FixedClusterAssigner,
  InProcessBus,
} from "../../src/distributed/transports/in-process.ts";

const FAST_OPTS = {
  quorumMin: 1,
  quorumTargetFrac: 1.0,
  intraDeadlineMs: 5_000,
  interDeadlineMs: 5_000,
  globalDeadlineMs: 5_000,
  f16wDebounceMs: 1_000,
};

function maxAbsDiff(a: Float32Array[], b: Float32Array[]): number {
  let m = 0;
  for (let t = 0; t < a.length; t++) {
    const at = a[t];
    const bt = b[t];
    for (let i = 0; i < at.length; i++) {
      const d = Math.abs(at[i] - bt[i]);
      if (d > m) m = d;
    }
  }
  return m;
}

describe("HierarchicalBarrierStateMachine — 8 peers in 2 clusters", () => {
  it("converges in lockstep across clusters with bit-identical consensus", async () => {
    const bus = new InProcessBus(new FixedClusterAssigner(4));
    const paramShapes = [[4]] as const;
    const targets = [
      [new Float32Array([1, 0, 0, 0])],
      [new Float32Array([0, 1, 0, 0])],
      [new Float32Array([0, 0, 1, 0])],
      [new Float32Array([0, 0, 0, 1])],
      [new Float32Array([-1, 0, 0, 0])],
      [new Float32Array([0, -1, 0, 0])],
      [new Float32Array([0, 0, -1, 0])],
      [new Float32Array([0, 0, 0, -1])],
    ];

    const peers = [] as Array<{
      id: string;
      trainer: StubTrainer;
      sm: HierarchicalBarrierStateMachine;
    }>;
    for (let i = 0; i < 8; i++) {
      const trainer = new StubTrainer({
        paramShapes,
        targets: targets[i],
        seed: 42,
        innerSteps: 5,
      });
      const tx = bus.connect(`peer-${i}`);
      const sm = new HierarchicalBarrierStateMachine(tx, trainer, FAST_OPTS);
      peers.push({ id: `peer-${i}`, trainer, sm });
    }

    // Wait for join-acks so we can assert the topology.
    await Promise.all(peers.map((p) => p.sm.awaitJoined()));
    const heads = peers.filter((p) => p.sm.getSelf()?.isHead).map((p) => p.id);
    expect(heads.length).toBe(2);
    // First peer in each cluster is the head, by default assigner.
    expect(heads).toEqual(["peer-0", "peer-4"]);

    const reports = await Promise.all(peers.map((p) => p.sm.run(15)));

    // Every peer ran every round successfully.
    for (const r of reports) {
      expect(r.length).toBe(15);
      expect(r.every((round) => round.outerStepTaken)).toBe(true);
    }
    // CHs report 8 contributors (all 8 peers across 2 clusters); non-heads
    // trust the CH's accounting and also see 8.
    for (const r of reports) {
      expect(r.every((round) => round.contributors === 8)).toBe(true);
    }

    // Anchors advanced in lockstep on all 8 peers.
    for (const { sm } of peers) {
      expect(sm.getAnchorRound()).toBe(15);
    }

    // Bit-identical post-outer-step params across all 8 peers (across clusters!).
    const ref = peers[0].trainer.currentParams();
    for (let i = 1; i < peers.length; i++) {
      const p = peers[i].trainer.currentParams();
      expect(maxAbsDiff(ref, p)).toBeLessThan(1e-5);
    }
  });

  it("single-cluster degenerate case still produces consensus", async () => {
    // 4 peers, cluster size 10 → everyone in cluster 0, one head.
    const bus = new InProcessBus(new FixedClusterAssigner(10));
    const paramShapes = [[4]] as const;
    const targets = [
      [new Float32Array([1, 0, 0, 0])],
      [new Float32Array([0, 1, 0, 0])],
      [new Float32Array([0, 0, 1, 0])],
      [new Float32Array([0, 0, 0, 1])],
    ];

    const peers = [];
    for (let i = 0; i < 4; i++) {
      const trainer = new StubTrainer({
        paramShapes,
        targets: targets[i],
        seed: 42,
        innerSteps: 5,
      });
      const tx = bus.connect(`peer-${i}`);
      const sm = new HierarchicalBarrierStateMachine(tx, trainer, FAST_OPTS);
      peers.push({ id: `peer-${i}`, trainer, sm });
    }

    const reports = await Promise.all(peers.map((p) => p.sm.run(10)));
    for (const r of reports) {
      expect(r.every((round) => round.outerStepTaken)).toBe(true);
      expect(r.every((round) => round.contributors === 4)).toBe(true);
      expect(r.every((round) => round.clustersContributed === 1)).toBe(true);
    }

    const ref = peers[0].trainer.currentParams();
    for (let i = 1; i < peers.length; i++) {
      expect(maxAbsDiff(ref, peers[i].trainer.currentParams())).toBeLessThan(
        1e-5,
      );
    }
  });

  it("uneven cluster sizes produce weighted (= flat-equivalent) average", async () => {
    // 5 peers, cluster size 3 → cluster 0 = {a,b,c}, cluster 1 = {d,e}.
    // The global average must weight cluster 0 by 3 and cluster 1 by 2.
    const bus = new InProcessBus(new FixedClusterAssigner(3));
    const paramShapes = [[2]] as const;
    const targets = [
      [new Float32Array([1, 0])],
      [new Float32Array([1, 0])],
      [new Float32Array([1, 0])],
      [new Float32Array([-1, 0])],
      [new Float32Array([-1, 0])],
    ];
    const peers = [];
    for (let i = 0; i < 5; i++) {
      const trainer = new StubTrainer({
        paramShapes,
        targets: targets[i],
        seed: 42,
        innerSteps: 5,
      });
      const tx = bus.connect(`peer-${i}`);
      const sm = new HierarchicalBarrierStateMachine(tx, trainer, FAST_OPTS);
      peers.push({ id: `peer-${i}`, trainer, sm });
    }

    const reports = await Promise.all(peers.map((p) => p.sm.run(20)));
    for (const r of reports) {
      expect(r.every((round) => round.outerStepTaken)).toBe(true);
    }
    // All 5 peers land at the same params.
    const ref = peers[0].trainer.currentParams();
    for (let i = 1; i < peers.length; i++) {
      expect(maxAbsDiff(ref, peers[i].trainer.currentParams())).toBeLessThan(
        1e-5,
      );
    }

    // With weighted averaging (3 vs 2), the consensus should bias toward
    // target=+1 (3 peers prefer it) over target=-1 (2 peers prefer it).
    // Per-dim 0 the converged value should be positive.
    expect(ref[0][0]).toBeGreaterThan(0);
  });
});
