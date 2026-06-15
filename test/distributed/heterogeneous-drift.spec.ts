/**
 * Heterogeneous-speed stability (task #48).
 *
 * Live browser+node demo (2026-06-15) showed `contributors` oscillating
 * 2→1→1→2 with full-params f16w re-sync thrash: a fast peer (A100) and a
 * slow peer (browser over the internet) repeatedly drift out of lockstep,
 * then re-sync. Mechanism under investigation:
 *
 *   - Consensus is gated on EXACT anchorRound equality (onMessage drops any
 *     ready/grad whose anchor != ours). The head increments its anchor the
 *     moment it applies the outer step — before the non-head has confirmed
 *     receipt of the global-aggregate. If anything delays that confirmation
 *     past a deadline, anchors diverge by 1 and EVERY subsequent message is
 *     dropped on the mismatch check until a full f16w resync realigns them.
 *   - currentRound increments every round, including failed/solo rounds, so a
 *     fast peer spinning solo can outrun a slow peer's round counter.
 *
 * This reproduces the drift deterministically with an in-process bus + link
 * latency on the slow peer, and asserts the FIXED behavior: once both peers
 * are present, every round is a clean 2-contributor aggregation (no thrash).
 */

import { describe, expect, it } from "vitest";
import {
  HierarchicalBarrierStateMachine,
  type RoundReport,
} from "../../src/distributed/protocol/hierarchical-state-machine.ts";
import { StubTrainer } from "../../src/distributed/protocol/stub-trainer.ts";
import {
  FixedClusterAssigner,
  InProcessBus,
} from "../../src/distributed/transports/in-process.ts";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe("Heterogeneous-speed 2-peer stability (task #48)", () => {
  it("a slow late-joining peer stays in lockstep (no anchor-drift thrash)", async () => {
    const bus = new InProcessBus(new FixedClusterAssigner(Infinity)); // 1 cluster
    const paramShapes = [[4]] as const;
    // Shared target so convergence is monotone and easy to read.
    const target = [new Float32Array([1, -1, 0.5, -0.5])];

    // Deadlines generous relative to link latency (≈25×), matching the live
    // ratio (300s deadline vs ~seconds of latency). A peer recovering a
    // dropped global must not cause its partner to give up the round.
    const OPTS = {
      quorumMin: 2,
      quorumTargetFrac: 1.0,
      intraDeadlineMs: 3_000,
      interDeadlineMs: 3_000,
      globalDeadlineMs: 3_000,
      f16wDebounceMs: 200,
      globalRetransmitMs: 400,
    };

    // Fast peer (head) connects first and starts spinning solo rounds.
    const fastTrainer = new StubTrainer({
      paramShapes,
      targets: target,
      seed: 1,
      innerSteps: 3,
    });
    const fastTx = bus.connect("fast");
    const fastSm = new HierarchicalBarrierStateMachine(fastTx, fastTrainer, OPTS);
    const fastReports: RoundReport[] = [];
    fastSm.onReport = (r) => fastReports.push(r);

    // Slow peer: ~120ms link latency each way, joins ~600ms late. Plus a
    // single dropped global-aggregate mid-run — models a message lost on a
    // transient reconnect (exactly what the live websocket blip did). This
    // is the canonical anchor-divergence trigger: the head commits its outer
    // step (anchor++) but the non-head never sees the global and reverts.
    const SLOW = "slow";
    let droppedGlobal = false;
    bus.faultHook = ({ from, to, message }) => {
      const slowLink = from === SLOW || to === SLOW;
      if (
        to === SLOW &&
        message.type === "grad" &&
        message.kind === "global-aggregate" &&
        message.round >= 4 &&
        !droppedGlobal
      ) {
        droppedGlobal = true;
        return { drop: true };
      }
      return slowLink ? { delayMs: 120 } : undefined;
    };

    const fastRun = fastSm.run(18);

    await sleep(600);
    const slowTrainer = new StubTrainer({
      paramShapes,
      targets: target,
      seed: 2,
      innerSteps: 3,
    });
    const slowTx = bus.connect(SLOW);
    const slowSm = new HierarchicalBarrierStateMachine(slowTx, slowTrainer, OPTS);
    const slowReports: RoundReport[] = [];
    slowSm.onReport = (r) => slowReports.push(r);
    const slowRun = slowSm.run(18);

    const [fast, slow] = await Promise.all([fastRun, slowRun]);

    const fmt = (rs: RoundReport[]) =>
      rs
        .map(
          (r) =>
            `r${r.round}/a${r.anchorAfter}/c${r.contributors}${r.outerStepTaken ? "*" : ""}${r.f16wApplied ? "F" : ""}`,
        )
        .join(" ");
    // eslint-disable-next-line no-console
    console.log("FAST:", fmt(fast));
    // eslint-disable-next-line no-console
    console.log("SLOW:", fmt(slow));

    // The dropped global lands on the round at anchor 4 (report.round is the
    // anchor — the barrier's matching coordinate). PRE-FIX: the slow peer
    // reverts (c1), drifts an anchor, then burns a round + a full-params f16w
    // to recover. POST-FIX: it nudges the head to resend the global and
    // completes the SAME round's joint outer step — no drift, no f16w.
    const dropRound = slow.find((r) => r.round === 4);
    expect(dropRound?.outerStepTaken).toBe(true);
    expect(dropRound?.contributors).toBe(2);

    // No anchor-drift thrash: recovery is the cheap resend, NEVER a
    // full-params f16w resync — on either peer.
    expect(slow.some((r) => r.f16wApplied)).toBe(false);
    expect(fast.some((r) => r.f16wApplied)).toBe(false);

    // Full lockstep: every aggregating round is a clean 2-contributor outer
    // step. The ONLY non-outer rounds allowed are at the boundaries — FAST's
    // first round (before the slow peer has synced) and SLOW's last (after
    // FAST hit maxRounds and exited). At most one per peer; none in between.
    for (const rs of [fast, slow]) {
      const solo = rs.filter((r) => !r.outerStepTaken);
      expect(solo.length).toBeLessThanOrEqual(1);
      for (const r of solo) {
        expect(r === rs[0] || r === rs[rs.length - 1]).toBe(true);
      }
    }

    // Strongest consensus check: both peers applied the identical sequence of
    // global aggregates, so their params are bit-for-bit equal at the end.
    expect(maxAbsDiff(fastTrainer.currentParams(), slowTrainer.currentParams())).toBe(0);

    fastSm.dispose();
    slowSm.dispose();
    fastTx.close();
    slowTx.close();
  }, 30_000);

  it("a late joiner recovers when its first f16w-request is lost", async () => {
    const bus = new InProcessBus(new FixedClusterAssigner(Infinity));
    const paramShapes = [[4]] as const;
    const target = [new Float32Array([0.5, 0.5, -0.5, -0.5])];
    const OPTS = {
      quorumMin: 2,
      quorumTargetFrac: 1.0,
      intraDeadlineMs: 2_000,
      interDeadlineMs: 2_000,
      globalDeadlineMs: 2_000,
      f16wDebounceMs: 200,
      globalRetransmitMs: 400,
    };

    const headTrainer = new StubTrainer({ paramShapes, targets: target, seed: 7, innerSteps: 2 });
    const headTx = bus.connect("head");
    const headSm = new HierarchicalBarrierStateMachine(headTx, headTrainer, OPTS);
    const headRun = headSm.run(6);

    // Drop the joiner's FIRST f16w-request so the single-shot path would have
    // stranded it; the retry must recover the sync.
    const JOINER = "joiner";
    let droppedReq = false;
    bus.faultHook = ({ from, message }) => {
      if (from === JOINER && message.type === "f16w-request" && !droppedReq) {
        droppedReq = true;
        return { drop: true };
      }
      return undefined;
    };

    await sleep(300);
    const joinTrainer = new StubTrainer({ paramShapes, targets: target, seed: 9, innerSteps: 2 });
    const joinTx = bus.connect(JOINER);
    const joinSm = new HierarchicalBarrierStateMachine(joinTx, joinTrainer, OPTS);
    const joinReports: RoundReport[] = [];
    joinSm.onReport = (r) => joinReports.push(r);
    const joinRun = joinSm.run(6);

    const [, join] = await Promise.all([headRun, joinRun]);

    // The dropped first request must not strand the joiner: it syncs (the
    // retry gets answered) and goes on to take joint outer steps.
    expect(droppedReq).toBe(true);
    expect(join.some((r) => r.outerStepTaken && r.contributors === 2)).toBe(true);
    // And the two peers end bit-identical (true consensus, not just "ran").
    expect(maxAbsDiff(headTrainer.currentParams(), joinTrainer.currentParams())).toBe(0);

    headSm.dispose();
    joinSm.dispose();
    headTx.close();
    joinTx.close();
  }, 30_000);
});

function maxAbsDiff(a: Float32Array[], b: Float32Array[]): number {
  let m = 0;
  for (let t = 0; t < a.length; t++) {
    for (let i = 0; i < a[t].length; i++) {
      const d = Math.abs(a[t][i] - b[t][i]);
      if (d > m) m = d;
    }
  }
  return m;
}
