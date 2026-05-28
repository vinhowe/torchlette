/**
 * Flat barrier protocol smoke tests.
 *
 * Run N stub peers through the in-process bus and assert:
 *   - All peers' anchors advance in lockstep (one outer step per round).
 *   - Final params are identical across all peers (consensus preserved).
 *   - Total system loss descends over the run.
 */

import { describe, expect, it } from "vitest";
import { FlatBarrierStateMachine } from "../../src/distributed/protocol/state-machine.ts";
import { StubTrainer } from "../../src/distributed/protocol/stub-trainer.ts";
import {
  FixedClusterAssigner,
  InProcessBus,
} from "../../src/distributed/transports/in-process.ts";

const FAST_OPTS = {
  quorumMin: 2,
  quorumTargetFrac: 1.0,
  matchmakingDeadlineMs: 5_000,
  gradWaitMs: 5_000,
  f16wDebounceMs: 1_000,
  pollIntervalMs: 1,
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

describe("FlatBarrierStateMachine — 2 peers, flat, happy path", () => {
  it("converges in lockstep with shared init + shared targets", async () => {
    const bus = new InProcessBus(new FixedClusterAssigner(Infinity));
    const paramShapes = [[8], [16]] as const;
    // Shared targets across peers → identical pseudograds → trivially same
    // post-outer-step params, but exercises the full barrier code path.
    const targets = [
      new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]),
      new Float32Array(16).fill(0.5),
    ];

    const trainerA = new StubTrainer({
      paramShapes,
      targets,
      seed: 42,
      innerSteps: 5,
    });
    const trainerB = new StubTrainer({
      paramShapes,
      targets,
      seed: 42,
      innerSteps: 5,
    });

    const txA = bus.connect("peer-a");
    const txB = bus.connect("peer-b");

    const smA = new FlatBarrierStateMachine(txA, trainerA, FAST_OPTS);
    const smB = new FlatBarrierStateMachine(txB, trainerB, FAST_OPTS);

    const initialLoss =
      (trainerA.lossToTarget() + trainerB.lossToTarget()) / 2;

    const [reportsA, reportsB] = await Promise.all([
      smA.run(20),
      smB.run(20),
    ]);

    // Both peers ran the same number of rounds.
    expect(reportsA.length).toBe(20);
    expect(reportsB.length).toBe(20);

    // Every round took an outer step on both peers — no solo reverts.
    expect(reportsA.every((r) => r.outerStepTaken)).toBe(true);
    expect(reportsB.every((r) => r.outerStepTaken)).toBe(true);
    expect(reportsA.every((r) => r.contributors === 2)).toBe(true);
    expect(reportsB.every((r) => r.contributors === 2)).toBe(true);

    // Anchors advanced once per round on both peers.
    expect(smA.getAnchorRound()).toBe(20);
    expect(smB.getAnchorRound()).toBe(20);

    // Final params are bit-identical between peers — consensus preserved.
    const pA = trainerA.currentParams();
    const pB = trainerB.currentParams();
    expect(maxAbsDiff(pA, pB)).toBeLessThan(1e-6);

    // Loss descended toward the (shared) target.
    const finalLoss =
      (trainerA.lossToTarget() + trainerB.lossToTarget()) / 2;
    expect(finalLoss).toBeLessThan(initialLoss * 0.5);
  });

  it("preserves consensus with divergent per-peer targets", async () => {
    // Each peer pulls toward its own target → pseudograds differ each
    // round, the barrier averages them, both peers end at the same
    // consensus params (the mean direction).
    const bus = new InProcessBus(new FixedClusterAssigner(Infinity));
    const paramShapes = [[4]] as const;
    const targetA = [new Float32Array([1, 1, 1, 1])];
    const targetB = [new Float32Array([-1, -1, -1, -1])];

    const trainerA = new StubTrainer({
      paramShapes,
      targets: targetA,
      seed: 42,
      innerSteps: 5,
    });
    const trainerB = new StubTrainer({
      paramShapes,
      targets: targetB,
      seed: 42,
      innerSteps: 5,
    });

    const txA = bus.connect("peer-a");
    const txB = bus.connect("peer-b");

    const smA = new FlatBarrierStateMachine(txA, trainerA, FAST_OPTS);
    const smB = new FlatBarrierStateMachine(txB, trainerB, FAST_OPTS);

    await Promise.all([smA.run(30), smB.run(30)]);

    // Bit-identical post-outer-step params on both peers.
    const pA = trainerA.currentParams();
    const pB = trainerB.currentParams();
    expect(maxAbsDiff(pA, pB)).toBeLessThan(1e-6);

    // Anchors advanced in lockstep.
    expect(smA.getAnchorRound()).toBe(30);
    expect(smB.getAnchorRound()).toBe(30);

    // Consensus pulls toward the average of the targets, which is zero.
    // Final params should be much closer to 0 than to either target.
    const meanAbs = (a: Float32Array[]) => {
      let s = 0;
      let n = 0;
      for (const t of a) {
        for (const v of t) {
          s += Math.abs(v);
          n++;
        }
      }
      return s / n;
    };
    expect(meanAbs(pA)).toBeLessThan(0.3);
  });
});

describe("FlatBarrierStateMachine — 4 peers, flat, happy path", () => {
  it("converges with 4-way averaging", async () => {
    // For bit-identical consensus, all peers must start from the same
    // anchor — i.e., same seeded init. Per-peer divergence comes from
    // different targets (simulating different data shards), which gives
    // each peer a distinct pseudograd while preserving anchor consistency.
    const bus = new InProcessBus(new FixedClusterAssigner(Infinity));
    const paramShapes = [[4]] as const;
    const targets = [
      [new Float32Array([1, 0, 0, 0])],
      [new Float32Array([0, 1, 0, 0])],
      [new Float32Array([0, 0, 1, 0])],
      [new Float32Array([0, 0, 0, 1])],
    ];

    const peers: Array<{
      trainer: StubTrainer;
      sm: FlatBarrierStateMachine;
      id: string;
    }> = [];
    for (let i = 0; i < 4; i++) {
      const trainer = new StubTrainer({
        paramShapes,
        targets: targets[i],
        seed: 42, // shared seed → identical anchor → bit-identical consensus
        innerSteps: 5,
      });
      const tx = bus.connect(`peer-${i}`);
      const sm = new FlatBarrierStateMachine(tx, trainer, {
        ...FAST_OPTS,
        quorumMin: 4,
      });
      peers.push({ trainer, sm, id: `peer-${i}` });
    }

    const reports = await Promise.all(peers.map((p) => p.sm.run(15)));

    // Every peer took an outer step every round, with 4 contributors.
    for (const r of reports) {
      expect(r.length).toBe(15);
      expect(r.every((round) => round.outerStepTaken)).toBe(true);
      expect(r.every((round) => round.contributors === 4)).toBe(true);
    }

    // All four peers landed on the same params.
    const refParams = peers[0].trainer.currentParams();
    for (let i = 1; i < peers.length; i++) {
      const p = peers[i].trainer.currentParams();
      expect(maxAbsDiff(refParams, p)).toBeLessThan(1e-6);
    }

    // Each anchor at 15.
    for (const { sm } of peers) {
      expect(sm.getAnchorRound()).toBe(15);
    }
  });
});
