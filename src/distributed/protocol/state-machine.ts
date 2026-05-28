/**
 * Flat barrier state machine for the DiLoCo outer step.
 *
 * Per-round lifecycle:
 *   1. If pending F16W advances our anchor, apply it and skip to next round.
 *   2. INNER:        trainer.innerSteps(round)
 *   3. ANNOUNCE:     broadcast ROUND_READY{round, anchor, clusterId}
 *                    add self to readySet[round]
 *   4. AWAIT_QUORUM: wait until enough peers (from the round-start snapshot)
 *                    have sent ROUND_READY for the same anchor, or
 *                    matchmakingDeadline expires.
 *   5. EXCHANGE:     broadcast GRAD with our pseudograd; receive others'.
 *   6. AWAIT_GRADS:  wait until grads from the quorum set arrive, or
 *                    gradWaitMs expires.
 *   7. APPLY:        average grads + trainer.applyOuterStep(avg); anchor += 1.
 *
 * On any wait timeout that drops us below quorumMin, we revert to anchor —
 * the local inner-step drift would otherwise diverge from consensus.
 *
 * Anchor mismatches between peers are detected at ROUND_READY (peer-ahead
 * triggers an F16W request; peer-behind is ignored). The barrier guarantees
 * that within a single round, all participants share the same anchor, so
 * the resulting outer step is consensus-deterministic.
 *
 * This module is FLAT (single-tier) by design — every peer aggregates
 * its own copy of the average. Hierarchical aggregation is layered on
 * top in a future task.
 */

import type {
  AnchorRound,
  ClusterId,
  GradMessage,
  PeerId,
  PeerInfo,
  ProtocolMessage,
  RoundNumber,
} from "./messages.ts";
import type { Trainer } from "./trainer.ts";
import type { Transport } from "./transport.ts";

export interface BarrierStateMachineOptions {
  /** Hard floor: never apply an outer step with fewer peers than this. */
  quorumMin: number;
  /** Target fraction of round-start swarm size. Higher = stricter sync. */
  quorumTargetFrac: number;
  /** Deadline (ms) for the AWAIT_QUORUM phase. */
  matchmakingDeadlineMs: number;
  /** Deadline (ms) for the AWAIT_GRADS phase after quorum committed. */
  gradWaitMs: number;
  /**
   * Minimum interval between F16W *sends* of the same anchor — drops
   * redundant retransmissions when multiple peers request resync to the
   * same checkpoint within a short window.
   */
  f16wDebounceMs: number;
  /** Poll interval (ms) for the wait loops. Lower = snappier, more CPU. */
  pollIntervalMs?: number;
}

export const defaultOptions: BarrierStateMachineOptions = {
  quorumMin: 2,
  quorumTargetFrac: 1.0,
  matchmakingDeadlineMs: 60_000,
  gradWaitMs: 60_000,
  f16wDebounceMs: 10_000,
  pollIntervalMs: 20,
};

interface PendingF16W {
  sourceAnchor: AnchorRound;
  sourceCurrentRound: RoundNumber;
  params: Float32Array[];
}

export interface RoundReport {
  round: RoundNumber;
  anchorAfter: AnchorRound;
  outerStepTaken: boolean;
  contributors: number;
  f16wApplied: boolean;
  loss?: number;
}

export class FlatBarrierStateMachine {
  // ─── Identity / membership ───────────────────────────────────────────
  private self: PeerInfo | null = null;
  private readonly peers = new Map<PeerId, PeerInfo>();

  // ─── Logical clocks ──────────────────────────────────────────────────
  private currentRound: RoundNumber = 0;
  private anchorRound: AnchorRound = 0;

  // ─── Per-round buffers ───────────────────────────────────────────────
  private readonly readySet = new Map<RoundNumber, Set<PeerId>>();
  private readonly gradSet = new Map<
    RoundNumber,
    Map<PeerId, GradMessage>
  >();

  // ─── F16W ────────────────────────────────────────────────────────────
  private pendingF16W: PendingF16W | null = null;
  private lastF16WSentMs = 0;
  private lastF16WSentAnchor: AnchorRound = -1;
  private lastF16WRequestedAtAnchor: AnchorRound = -1;

  // ─── Control ─────────────────────────────────────────────────────────
  private running = false;
  private joined: Promise<void>;
  private readonly transport: Transport;
  private readonly trainer: Trainer;
  private readonly opts: BarrierStateMachineOptions;
  private readonly unsubscribe: () => void;
  /** Waiters parked on the next state-change signal. */
  private waiters: Array<() => void> = [];

  constructor(
    transport: Transport,
    trainer: Trainer,
    opts: Partial<BarrierStateMachineOptions> = {},
  ) {
    this.transport = transport;
    this.trainer = trainer;
    this.opts = { ...defaultOptions, ...opts };

    let resolveJoin: () => void = () => {};
    this.joined = new Promise<void>((r) => {
      resolveJoin = r;
    });
    this.unsubscribe = this.transport.onReceive((msg) =>
      this.onMessage(msg, resolveJoin),
    );
  }

  /** Wait until the coordinator's join-ack arrives. */
  async awaitJoined(): Promise<void> {
    return this.joined;
  }

  /**
   * Run up to `maxRounds` rounds. Resolves when the loop exits — either
   * because the round count is met or stop() was called.
   */
  async run(maxRounds: number): Promise<RoundReport[]> {
    await this.joined;
    await this.trainer.setAnchor();
    this.running = true;
    const reports: RoundReport[] = [];
    while (this.running && this.currentRound < maxRounds) {
      const report = await this.runRound();
      reports.push(report);
      this.currentRound++;
    }
    return reports;
  }

  stop(): void {
    this.running = false;
    this.signal();
  }

  /** Tear down message subscription; does not close the transport. */
  dispose(): void {
    this.running = false;
    this.signal();
    this.unsubscribe();
  }

  // ─── Inspection (for tests) ──────────────────────────────────────────
  getAnchorRound(): AnchorRound {
    return this.anchorRound;
  }

  getCurrentRound(): RoundNumber {
    return this.currentRound;
  }

  getSelf(): PeerInfo | null {
    return this.self;
  }

  // ─── Round body ──────────────────────────────────────────────────────
  private async runRound(): Promise<RoundReport> {
    const round = this.currentRound;

    // Step 1: apply pending F16W if it strictly advances our anchor.
    if (await this.maybeApplyPendingF16W()) {
      this.cleanupRound(round);
      return {
        round,
        anchorAfter: this.anchorRound,
        outerStepTaken: false,
        contributors: 1,
        f16wApplied: true,
      };
    }

    // Step 2: inner training.
    await this.trainer.innerSteps(round);

    // Step 3: snapshot the swarm view at round start. Quorum is computed
    // against this set so late joiners don't inflate the denominator.
    const expectedPeers = new Set(this.peers.keys());
    if (this.self) expectedPeers.add(this.self.peerId);

    // Step 4: announce. Mark self ready locally so we count toward quorum.
    this.sendReady(round);
    this.markReady(round, this.self?.peerId ?? "");

    // Step 5: wait for quorum.
    const quorumGroup = await this.awaitQuorum(round, expectedPeers);
    if (quorumGroup === null) {
      await this.trainer.revertToAnchor();
      this.cleanupRound(round);
      return {
        round,
        anchorAfter: this.anchorRound,
        outerStepTaken: false,
        contributors: this.readySet.get(round)?.size ?? 1,
        f16wApplied: false,
      };
    }

    // Step 6: send grad to everyone in quorum (broadcast — peer routing
    // will deliver to whichever quorum members are still connected). We
    // also store our own grad locally so step 8 can include it.
    const myGrad = await this.trainer.pseudograd();
    this.sendGrad(round, myGrad);
    this.storeOwnGrad(round, myGrad);

    // Step 7: wait for grads from quorum members.
    const grads = await this.awaitGrads(round, quorumGroup);
    if (grads === null) {
      await this.trainer.revertToAnchor();
      this.cleanupRound(round);
      return {
        round,
        anchorAfter: this.anchorRound,
        outerStepTaken: false,
        contributors: this.gradSet.get(round)?.size ?? 1,
        f16wApplied: false,
      };
    }

    // Step 8: average + apply.
    const avg = averageGrads(grads);
    await this.trainer.applyOuterStep(avg);
    this.anchorRound += 1;
    const contributors = grads.length;
    this.cleanupRound(round);
    return {
      round,
      anchorAfter: this.anchorRound,
      outerStepTaken: true,
      contributors,
      f16wApplied: false,
    };
  }

  // ─── Phase helpers ───────────────────────────────────────────────────
  private async maybeApplyPendingF16W(): Promise<boolean> {
    const blob = this.pendingF16W;
    if (!blob || blob.sourceAnchor <= this.anchorRound) {
      this.pendingF16W = null;
      return false;
    }
    this.pendingF16W = null;
    await this.trainer.applyF16W(blob.params);
    await this.trainer.resetOptimState();
    this.anchorRound = blob.sourceAnchor;
    this.currentRound = blob.sourceCurrentRound;
    return true;
  }

  private sendReady(round: RoundNumber): void {
    if (!this.self) return;
    this.transport.send(
      { kind: "broadcast" },
      {
        type: "round-ready",
        peerId: this.self.peerId,
        round,
        anchor: this.anchorRound,
        clusterId: this.self.clusterId,
      },
    );
  }

  private sendGrad(round: RoundNumber, grad: Float32Array[]): void {
    if (!this.self) return;
    this.transport.send(
      { kind: "broadcast" },
      {
        type: "grad",
        fromPeerId: this.self.peerId,
        round,
        anchor: this.anchorRound,
        kind: "peer-grad",
        peerCount: 1,
        payload: grad,
      },
    );
  }

  private async awaitQuorum(
    round: RoundNumber,
    expectedPeers: Set<PeerId>,
  ): Promise<Set<PeerId> | null> {
    const expected = expectedPeers.size;
    const target = Math.ceil(expected * this.opts.quorumTargetFrac);
    const needed = Math.max(this.opts.quorumMin, target);
    const deadline = Date.now() + this.opts.matchmakingDeadlineMs;

    while (Date.now() < deadline) {
      if (!this.running) return null;
      // Forward-jumping F16W aborts the wait — let next round apply it.
      if (
        this.pendingF16W &&
        this.pendingF16W.sourceAnchor > this.anchorRound
      ) {
        return null;
      }
      const ready = this.readySet.get(round);
      if (ready) {
        const valid = intersect(ready, expectedPeers);
        if (valid.size >= needed) return valid;
      }
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await this.waitForSignal(remaining);
    }
    // Deadline expired — accept what we have if it clears quorumMin.
    const ready = this.readySet.get(round);
    if (!ready) return null;
    const valid = intersect(ready, expectedPeers);
    return valid.size >= this.opts.quorumMin ? valid : null;
  }

  private async awaitGrads(
    round: RoundNumber,
    quorumGroup: Set<PeerId>,
  ): Promise<Float32Array[][] | null> {
    const deadline = Date.now() + this.opts.gradWaitMs;

    while (Date.now() < deadline) {
      if (!this.running) return null;
      if (
        this.pendingF16W &&
        this.pendingF16W.sourceAnchor > this.anchorRound
      ) {
        return null;
      }
      const grads = this.gradSet.get(round);
      if (grads && grads.size >= quorumGroup.size) {
        // We have all expected grads.
        const list: Float32Array[][] = [];
        for (const id of quorumGroup) {
          const m = grads.get(id);
          if (m) list.push(m.payload);
        }
        return list;
      }
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await this.waitForSignal(remaining);
    }
    // Deadline expired.
    const grads = this.gradSet.get(round);
    if (!grads || grads.size < this.opts.quorumMin) return null;
    const list: Float32Array[][] = [];
    for (const id of quorumGroup) {
      const m = grads.get(id);
      if (m) list.push(m.payload);
    }
    return list.length >= this.opts.quorumMin ? list : null;
  }

  /**
   * Wait for any state-change signal (a peer's READY, GRAD, or F16W arrival)
   * or the timeout, whichever comes first. Replaces polling; rounds are
   * gated only by message latency, not by poll cadence.
   */
  private waitForSignal(timeoutMs: number): Promise<void> {
    return new Promise<void>((resolve) => {
      let fired = false;
      const fire = () => {
        if (fired) return;
        fired = true;
        clearTimeout(timer);
        resolve();
      };
      const timer = setTimeout(fire, timeoutMs);
      this.waiters.push(fire);
    });
  }

  /** Wake every parked waiter; call after any state mutation that could
   *  satisfy a quorum-or-grads condition. */
  private signal(): void {
    if (this.waiters.length === 0) return;
    const toFire = this.waiters;
    this.waiters = [];
    for (const w of toFire) w();
  }

  private cleanupRound(round: RoundNumber): void {
    this.readySet.delete(round);
    this.gradSet.delete(round);
  }

  // ─── Bookkeeping ─────────────────────────────────────────────────────
  private markReady(round: RoundNumber, peerId: PeerId): void {
    if (!peerId) return;
    let set = this.readySet.get(round);
    if (!set) {
      set = new Set();
      this.readySet.set(round, set);
    }
    const before = set.size;
    set.add(peerId);
    if (set.size !== before) this.signal();
  }

  private storeOwnGrad(round: RoundNumber, grad: Float32Array[]): void {
    if (!this.self) return;
    let m = this.gradSet.get(round);
    if (!m) {
      m = new Map();
      this.gradSet.set(round, m);
    }
    m.set(this.self.peerId, {
      type: "grad",
      fromPeerId: this.self.peerId,
      round,
      anchor: this.anchorRound,
      kind: "peer-grad",
      peerCount: 1,
      payload: grad,
    });
    this.signal();
  }

  // ─── Message handling ────────────────────────────────────────────────
  private onMessage(
    msg: ProtocolMessage,
    resolveJoin: () => void,
  ): void {
    switch (msg.type) {
      case "join-ack": {
        this.self = {
          peerId: msg.peerId,
          clusterId: msg.clusterId,
          isHead: msg.isHead,
        };
        this.peers.clear();
        for (const p of msg.peers) this.peers.set(p.peerId, p);
        resolveJoin();
        return;
      }
      case "peer-list": {
        this.peers.clear();
        for (const p of msg.peers) this.peers.set(p.peerId, p);
        return;
      }
      case "round-ready": {
        if (msg.anchor === this.anchorRound) {
          this.markReady(msg.round, msg.peerId);
        } else if (msg.anchor > this.anchorRound) {
          this.requestF16WIfNeeded(msg.anchor);
        }
        // anchor < ours: peer is behind, ignore.
        return;
      }
      case "grad": {
        if (msg.anchor !== this.anchorRound) {
          if (msg.anchor > this.anchorRound) {
            this.requestF16WIfNeeded(msg.anchor);
          }
          return;
        }
        let m = this.gradSet.get(msg.round);
        if (!m) {
          m = new Map();
          this.gradSet.set(msg.round, m);
        }
        m.set(msg.fromPeerId, msg);
        this.signal();
        return;
      }
      case "f16w-request": {
        this.handleF16WRequest();
        return;
      }
      case "f16w": {
        if (msg.sourceAnchor > this.anchorRound) {
          this.pendingF16W = {
            sourceAnchor: msg.sourceAnchor,
            sourceCurrentRound: msg.sourceCurrentRound,
            params: msg.params,
          };
          this.signal();
        }
        return;
      }
      case "join":
      case "leave":
        return;
    }
  }

  private requestF16WIfNeeded(peerAnchor: AnchorRound): void {
    if (!this.self) return;
    // Avoid spamming the relay: don't re-request the same anchor we
    // already have a request outstanding for.
    if (this.lastF16WRequestedAtAnchor >= peerAnchor) return;
    this.lastF16WRequestedAtAnchor = peerAnchor;
    this.transport.send(
      { kind: "broadcast" },
      {
        type: "f16w-request",
        peerId: self_peerId(this.self),
        atLeastAnchor: peerAnchor,
      },
    );
  }

  private async handleF16WRequest(): Promise<void> {
    if (!this.self) return;
    const now = Date.now();
    if (
      this.lastF16WSentAnchor === this.anchorRound &&
      now - this.lastF16WSentMs < this.opts.f16wDebounceMs
    ) {
      return;
    }
    // Reserve the debounce window BEFORE the async read so concurrent
    // requests during the read don't all kick off duplicate snapshots.
    this.lastF16WSentMs = now;
    this.lastF16WSentAnchor = this.anchorRound;
    const params = await this.trainer.snapshotAnchor();
    this.transport.send(
      { kind: "broadcast" },
      {
        type: "f16w",
        fromPeerId: this.self.peerId,
        sourceAnchor: this.anchorRound,
        sourceCurrentRound: this.currentRound,
        params,
      },
    );
  }
}

// ─── Helpers ───────────────────────────────────────────────────────────
function self_peerId(self: PeerInfo): PeerId {
  return self.peerId;
}

function intersect<T>(a: Set<T>, b: Set<T>): Set<T> {
  const out = new Set<T>();
  for (const x of a) if (b.has(x)) out.add(x);
  return out;
}

function averageGrads(grads: Float32Array[][]): Float32Array[] {
  if (grads.length === 0) throw new Error("averageGrads: empty input");
  const n = grads.length;
  const numTensors = grads[0].length;
  const out: Float32Array[] = [];
  for (let t = 0; t < numTensors; t++) {
    const len = grads[0][t].length;
    const acc = new Float32Array(len);
    for (let g = 0; g < n; g++) {
      const src = grads[g][t];
      if (src.length !== len) {
        throw new Error(
          `averageGrads: tensor ${t} length mismatch ${src.length} vs ${len}`,
        );
      }
      for (let i = 0; i < len; i++) acc[i] += src[i];
    }
    const inv = 1 / n;
    for (let i = 0; i < len; i++) acc[i] *= inv;
    out.push(acc);
  }
  return out;
}

// Re-export some types so callers don't need to reach into ./messages.
export type { ClusterId };
