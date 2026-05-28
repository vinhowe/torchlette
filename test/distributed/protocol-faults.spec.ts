/**
 * Fault scenarios for the barrier protocol.
 *
 * Verifies the protocol degrades gracefully under common distributed-
 * system faults: a peer crashing mid-run, a cluster head leaving, a
 * straggler joining late.
 *
 * The in-process bus broadcasts peer-list updates on every membership
 * change and auto-promotes a new cluster head when one disconnects.
 * Tests check that the state machine's per-round quorum snapshot
 * adapts so the remaining peers continue training.
 */

import { describe, expect, it } from "vitest";
import { HierarchicalBarrierStateMachine } from "../../src/distributed/protocol/hierarchical-state-machine.ts";
import { StubTrainer } from "../../src/distributed/protocol/stub-trainer.ts";
import {
  FixedClusterAssigner,
  InProcessBus,
} from "../../src/distributed/transports/in-process.ts";
import type { Transport } from "../../src/distributed/protocol/transport.ts";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const DEFAULT_OPTS = {
  quorumMin: 1,
  quorumTargetFrac: 1.0,
  intraDeadlineMs: 2_000,
  interDeadlineMs: 2_000,
  globalDeadlineMs: 2_000,
  f16wDebounceMs: 500,
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

describe("Fault: single cluster member drops mid-run", () => {
  it("remaining peers continue with reduced quorum", async () => {
    // 4 peers in one cluster. peer-3 stops after round 3 and tears down;
    // the surviving peers must continue advancing to round 15 with
    // adapted quorum, then agree on final params.
    const bus = new InProcessBus(new FixedClusterAssigner(10));
    const paramShapes = [[2]] as const;
    const targets = [
      [new Float32Array([1, 0])],
      [new Float32Array([0, 1])],
      [new Float32Array([-1, 0])],
      [new Float32Array([0, -1])],
    ];
    const peers = [] as Array<{
      id: string;
      trainer: StubTrainer;
      sm: HierarchicalBarrierStateMachine;
      tx: Transport;
    }>;
    for (let i = 0; i < 4; i++) {
      const trainer = new StubTrainer({
        paramShapes,
        targets: targets[i],
        seed: 42,
        innerSteps: 3,
      });
      const tx = bus.connect(`peer-${i}`);
      const sm = new HierarchicalBarrierStateMachine(tx, trainer, {
        ...DEFAULT_OPTS,
        quorumTargetFrac: 0.7,
      });
      peers.push({ id: `peer-${i}`, trainer, sm, tx });
    }

    const reports = await Promise.all([
      peers[0].sm.run(15),
      peers[1].sm.run(15),
      peers[2].sm.run(15),
      peers[3].sm.run(3).then((r) => {
        peers[3].sm.dispose();
        peers[3].tx.close();
        return r;
      }),
    ]);

    // Peers 0..2 reached the end despite peer-3 leaving.
    for (let i = 0; i < 3; i++) {
      expect(peers[i].sm.getAnchorRound()).toBe(15);
    }
    expect(peers[3].sm.getAnchorRound()).toBe(3);

    // Surviving peers agree on params.
    const ref = peers[0].trainer.currentParams();
    for (let i = 1; i < 3; i++) {
      expect(maxAbsDiff(ref, peers[i].trainer.currentParams())).toBeLessThan(
        1e-5,
      );
    }

    // Surviving peers each took outer steps in the post-dropout rounds.
    for (let i = 0; i < 3; i++) {
      const postDropRounds = reports[i].filter(
        (r) => r.round >= 3 && r.outerStepTaken,
      );
      expect(postDropRounds.length).toBeGreaterThan(0);
    }
  });
});

describe("Fault: cluster head dies", () => {
  it("bus promotes a new head; surviving peers complete the run", async () => {
    // 4 peers in one cluster. Head (peer-0) dies at round 3. The bus
    // promotes the next survivor (peer-1) and broadcasts the new
    // peer-list. The remaining 3 peers continue with peer-1 as head.
    const bus = new InProcessBus(new FixedClusterAssigner(10));
    const paramShapes = [[2]] as const;
    const peers = [] as Array<{
      id: string;
      trainer: StubTrainer;
      sm: HierarchicalBarrierStateMachine;
      tx: Transport;
    }>;
    for (let i = 0; i < 4; i++) {
      const trainer = new StubTrainer({
        paramShapes,
        targets: [new Float32Array([i, -i])],
        seed: 42,
        innerSteps: 3,
      });
      const tx = bus.connect(`peer-${i}`);
      const sm = new HierarchicalBarrierStateMachine(tx, trainer, {
        ...DEFAULT_OPTS,
        quorumTargetFrac: 0.6,
      });
      peers.push({ id: `peer-${i}`, trainer, sm, tx });
    }
    await Promise.all(peers.map((p) => p.sm.awaitJoined()));
    // peer-0 starts as head; peer-1 is not.
    expect(peers[0].sm.getSelf()?.isHead).toBe(true);
    expect(peers[1].sm.getSelf()?.isHead).toBe(false);

    const dropAt = 3;
    const reports = await Promise.all([
      peers[0].sm.run(dropAt).then((r) => {
        peers[0].sm.dispose();
        peers[0].tx.close();
        return r;
      }),
      peers[1].sm.run(15),
      peers[2].sm.run(15),
      peers[3].sm.run(15),
    ]);

    // Surviving peers reached the end.
    for (let i = 1; i < 4; i++) {
      expect(peers[i].sm.getAnchorRound()).toBe(15);
    }
    // Head moved.
    expect(peers[1].sm.getSelf()?.isHead).toBe(true);

    // Survivors agree.
    const ref = peers[1].trainer.currentParams();
    for (let i = 2; i < 4; i++) {
      expect(maxAbsDiff(ref, peers[i].trainer.currentParams())).toBeLessThan(
        1e-5,
      );
    }

    // Each survivor took outer steps after the drop.
    for (let i = 1; i < 4; i++) {
      const postDropRounds = reports[i].filter(
        (r) => r.round >= dropAt && r.outerStepTaken,
      );
      expect(postDropRounds.length).toBeGreaterThan(0);
    }
  });
});
