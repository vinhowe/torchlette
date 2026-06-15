/**
 * Hierarchical (two-tier) barrier state machine.
 *
 * Cluster topology — each peer belongs to a cluster (from the JoinAck) and
 * one peer per cluster is designated the cluster head (CH). Per round, the
 * data flow is:
 *
 *   [non-head members] -- peer-grad --> [CH]
 *                                         | aggregate within cluster
 *                                         v
 *                              [CH] -- cluster-aggregate --> [other CHs]
 *                                                                |
 *                                              all CHs combine cluster
 *                                              aggregates into a global
 *                                              average (weighted by the
 *                                              peerCount each cluster
 *                                              contributed)
 *                                                                |
 *                                                                v
 *                                           [CH] -- global-aggregate -->
 *                                                                  [cluster
 *                                                                   members]
 *
 * Bandwidth on the relay:
 *   - Flat broadcast: every peer broadcasts to every other → N×(N-1) grads
 *     on the wire per round.
 *   - Hierarchical with cluster size K: each peer sends 1 peer-grad to its
 *     CH (N), each CH sends 1 cluster-aggregate to other CHs ((N/K)×(N/K-1)),
 *     each CH broadcasts 1 global-aggregate to its cluster (N) → ~2N+
 *     small overhead total. Much better than flat for large N.
 *
 * The mathematical result is bit-identical to a flat all-to-all average
 * AS LONG AS the cross-cluster averaging is properly weighted by each
 * cluster's contributing peer count.
 *
 * Single-cluster degenerate case: there are no "other CHs," so the global
 * aggregate IS the cluster aggregate. The protocol still has the
 * peer→CH→peer indirection in this case, which is a slight overhead vs
 * the flat protocol; we keep the indirection so the code path is uniform.
 */

import type {
  AnchorRound,
  ClusterId,
  GradMessage,
  PeerId,
  PeerInfo,
  ProtocolMessage,
  RoundNumber,
  RoundReadyMessage,
} from "./messages.ts";
import type { Trainer } from "./trainer.ts";
import type { Transport } from "./transport.ts";

export interface HierarchicalOptions {
  /** Hard floor on cluster-quorum membership at intra-cluster step. */
  quorumMin: number;
  /** Target fraction of cluster size for intra-quorum. */
  quorumTargetFrac: number;
  /** Deadline (ms) for waiting on cluster members' readys + grads. */
  intraDeadlineMs: number;
  /** Deadline (ms) for waiting on other CHs' cluster aggregates. */
  interDeadlineMs: number;
  /** Deadline (ms) for waiting for the global aggregate (non-heads). */
  globalDeadlineMs: number;
  f16wDebounceMs: number;
  /**
   * How long a non-head waits for the global-aggregate before nudging its
   * head to resend it (and re-nudging each interval). Capped internally to
   * globalDeadlineMs/2 so it always fires at least once before the deadline.
   * This is the drift-free recovery for a single lost global — far cheaper
   * than reverting and pulling a full-params f16w.
   */
  globalRetransmitMs: number;
}

export const defaultHierarchicalOptions: HierarchicalOptions = {
  quorumMin: 1,
  quorumTargetFrac: 1.0,
  intraDeadlineMs: 60_000,
  interDeadlineMs: 60_000,
  globalDeadlineMs: 60_000,
  f16wDebounceMs: 10_000,
  globalRetransmitMs: 2_000,
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
  /** Total peer-grads folded into this round's outer step (across all clusters). */
  contributors: number;
  /** Clusters that contributed an aggregate. */
  clustersContributed: number;
  f16wApplied: boolean;
  /** Avg inner-step loss for this round (NaN if no inner steps ran). */
  innerLoss: number;
}

export class HierarchicalBarrierStateMachine {
  private self: PeerInfo | null = null;
  private readonly peers = new Map<PeerId, PeerInfo>();

  private currentRound: RoundNumber = 0;
  private anchorRound: AnchorRound = 0;
  private lastInnerLoss = NaN;

  // Per-round buffers
  private readonly readySet = new Map<RoundNumber, Set<PeerId>>();
  /** Grads received from any source, keyed by round and sender peer-id. */
  private readonly intraGrads = new Map<
    RoundNumber,
    Map<PeerId, GradMessage>
  >();
  /** Cluster-aggregates received from other CHs, keyed by round + sender. */
  private readonly interAggs = new Map<
    RoundNumber,
    Map<PeerId, GradMessage>
  >();
  /** Global-aggregate received from our CH, keyed by round. */
  private readonly globalAgg = new Map<RoundNumber, GradMessage>();
  /**
   * Heads only: the last few global-aggregates this head broadcast, keyed by
   * round, so it can answer a `resend-global` request from a member that
   * missed the broadcast. `anchor` is the round's anchor (pre-commit), which
   * is what the requesting member still expects. Bounded to a handful of
   * entries (insert-capped) — each is params-sized, but a member's resend
   * request always arrives within a round or two of the broadcast.
   */
  private readonly recentGlobals = new Map<
    RoundNumber,
    { anchor: AnchorRound; peerCount: number; payload: Float32Array[] }
  >();
  private static readonly RECENT_GLOBALS_KEEP = 3;
  /**
   * Ready/grad messages that arrived exactly one anchor ahead of us (the
   * sender committed an outer step we haven't finished yet). Replayed the
   * instant we advance our anchor. Bounded so a misbehaving peer can't grow
   * it without limit.
   */
  private earlyMessages: Array<RoundReadyMessage | GradMessage> = [];
  private static readonly EARLY_BUFFER_MAX = 512;

  private pendingF16W: PendingF16W | null = null;
  private lastF16WSentMs = 0;
  private lastF16WSentAnchor: AnchorRound = -1;
  private lastF16WRequestedAtAnchor: AnchorRound = -1;
  /** Set from the join-ack: true when other peers were present at join.
   *  run() proactively requests F16W before training when this is set,
   *  so a late joiner doesn't waste rounds discovering the mismatch via
   *  anchor-tagged grads. */
  private needsSync = false;

  private running = false;
  /** Optional per-round callback for streaming progress (e.g. STATS lines). */
  onReport: ((r: RoundReport) => void) | undefined = undefined;
  private joined: Promise<void>;
  private waiters: Array<() => void> = [];
  private readonly transport: Transport;
  private readonly trainer: Trainer;
  private readonly opts: HierarchicalOptions;
  private readonly unsubscribe: () => void;

  constructor(
    transport: Transport,
    trainer: Trainer,
    opts: Partial<HierarchicalOptions> = {},
  ) {
    this.transport = transport;
    this.trainer = trainer;
    this.opts = { ...defaultHierarchicalOptions, ...opts };

    let resolveJoin: () => void = () => {};
    this.joined = new Promise<void>((r) => {
      resolveJoin = r;
    });
    this.unsubscribe = this.transport.onReceive((msg) =>
      this.onMessage(msg, resolveJoin),
    );
  }

  awaitJoined(): Promise<void> {
    return this.joined;
  }

  async run(maxRounds: number): Promise<RoundReport[]> {
    await this.joined;
    // Late joiner: pull current consensus from a peer BEFORE establishing
    // our own random-init anchor. Otherwise the first peer-grad exchange
    // burns a round catching us up via the anchor-mismatch path.
    if (this.needsSync) {
      const synced = await this.fetchInitialF16W();
      if (!synced) {
        // Sync timed out — fall back to own init. The first peer-grad
        // exchange will detect the mismatch and recover via the regular
        // F16W path.
        await this.trainer.setAnchor();
      }
    } else {
      await this.trainer.setAnchor();
    }
    this.running = true;
    const reports: RoundReport[] = [];
    while (this.running && this.currentRound < maxRounds) {
      const report = await this.runRound();
      reports.push(report);
      this.onReport?.(report);
      this.currentRound++;
    }
    return reports;
  }

  /**
   * Proactive initial sync. Broadcasts an f16w-request and waits for the
   * first F16W to arrive, then applies it (sets params + anchor +
   * currentRound). Returns true if synced, false on timeout.
   */
  private async fetchInitialF16W(): Promise<boolean> {
    if (!this.self) return false;
    const sendRequest = () => {
      // Bypass the "have we already requested at this anchor" debounce —
      // we're at anchor=0 with random init, so 0 is what we'd send anyway,
      // but we want this to fire unconditionally on cold join.
      this.lastF16WRequestedAtAnchor = -1;
      this.transport.send(
        { kind: "broadcast" },
        { type: "f16w-request", peerId: this.self!.peerId, atLeastAnchor: 0 },
      );
    };
    sendRequest();
    const deadline = Date.now() + this.opts.intraDeadlineMs;
    // Re-broadcast the request periodically: a single lost request (or lost
    // response — e.g. across a transient reconnect) must NOT strand the late
    // joiner for the whole deadline. The interval clears the head's f16w
    // debounce so a retry actually triggers a fresh snapshot+send.
    const retransmit = Math.max(
      this.opts.f16wDebounceMs + 1_000,
      Math.min(this.opts.globalRetransmitMs, Math.floor(this.opts.intraDeadlineMs / 2)),
    );
    let nextResendAt = Date.now() + retransmit;
    // NB: this runs BEFORE run() sets this.running = true, so do NOT gate the
    // loop on this.running here (that would bail on the first iteration and
    // skip the sync entirely — caught by the strict params-equality gate).
    while (Date.now() < deadline) {
      if (this.pendingF16W) {
        const blob = this.pendingF16W;
        this.pendingF16W = null;
        await this.trainer.applyF16W(blob.params);
        await this.trainer.resetOptimState();
        this.anchorRound = blob.sourceAnchor;
        this.currentRound = blob.sourceCurrentRound;
        this.needsSync = false;
        this.replayEarly();
        return true;
      }
      const now = Date.now();
      if (now >= nextResendAt) {
        // Suppress the re-request while the f16w is actively streaming in
        // (chunks arriving). Re-requesting mid-transfer makes the head fire a
        // fresh full f16w, piling up gigabytes in the relay's send buffer to a
        // slow peer and congesting the link so the first transfer never
        // finishes. Only re-request after a genuine stall (no chunk for a
        // whole interval) — a lost request/response, not a slow-but-progressing
        // transfer.
        const sinceChunk = this.transport.msSinceLastChunk?.() ?? Infinity;
        if (sinceChunk >= retransmit) sendRequest();
        nextResendAt = now + retransmit;
      }
      const remaining = Math.min(deadline - now, nextResendAt - now);
      if (remaining <= 0) continue;
      await this.waitForSignal(remaining);
    }
    this.needsSync = false;
    return false;
  }

  stop(): void {
    this.running = false;
    this.signal();
  }

  dispose(): void {
    this.running = false;
    this.signal();
    this.unsubscribe();
  }

  // Inspection (for tests)
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
    // The barrier's matching coordinate is the ANCHOR, not the local loop
    // counter. Two peers at the same anchor are at the same point in weight
    // space and MUST aggregate — even if their currentRound counters have
    // drifted (which they do: each peer advances currentRound on every loop
    // iteration including solo/failed rounds, at its own wall-clock rate). Key
    // readys/grads/globals by anchor so same-anchor peers always align;
    // currentRound stays a purely local iteration counter (loop bound + logs).
    const round = this.anchorRound;

    if (await this.maybeApplyPendingF16W()) {
      this.cleanupRound(round);
      return {
        round,
        anchorAfter: this.anchorRound,
        outerStepTaken: false,
        contributors: 1,
        clustersContributed: 1,
        f16wApplied: true,
        innerLoss: NaN,
      };
    }

    const innerLoss = await this.trainer.innerSteps(round);
    this.lastInnerLoss = innerLoss;

    const self = this.self;
    if (!self) {
      return this.failRound(round, "no self");
    }

    // Snapshot cluster membership at round start.
    const clusterMembers = new Set<PeerId>();
    for (const p of this.peers.values()) {
      if (p.clusterId === self.clusterId) clusterMembers.add(p.peerId);
    }
    clusterMembers.add(self.peerId);
    const clusterHead = this.findClusterHead(self.clusterId);
    const clusterSize = clusterMembers.size;
    const otherHeads = new Set<PeerId>();
    for (const p of this.peers.values()) {
      if (p.isHead && p.peerId !== self.peerId) otherHeads.add(p.peerId);
    }

    // Phase A: announce.
    this.sendReady(round);
    this.markReady(round, self.peerId);

    // Phase B: await intra quorum (peers in our cluster ready).
    const intraGroup = await this.awaitIntraQuorum(round, clusterMembers);
    if (intraGroup === null) {
      await this.trainer.revertToAnchor();
      this.cleanupRound(round);
      return this.report(round, false, 1, 1, false);
    }

    // Phase C: produce and route peer-grad.
    const myGrad = await this.trainer.pseudograd();
    if (self.isHead) {
      // CH stores own grad locally for intra aggregation.
      this.storeIntraGrad(round, self.peerId, {
        type: "grad",
        fromPeerId: self.peerId,
        round,
        anchor: this.anchorRound,
        kind: "peer-grad",
        peerCount: 1,
        payload: myGrad,
      });
    } else {
      if (!clusterHead) {
        // Cluster has no head — degenerate. Treat as solo.
        await this.trainer.revertToAnchor();
        this.cleanupRound(round);
        return this.report(round, false, 1, 1, false);
      }
      this.transport.send(
        { kind: "peer", peerId: clusterHead },
        {
          type: "grad",
          fromPeerId: self.peerId,
          round,
          anchor: this.anchorRound,
          kind: "peer-grad",
          peerCount: 1,
          payload: myGrad,
        },
      );
    }

    let globalAggregate: Float32Array[] | null = null;
    let totalContributors = 0;
    let clustersContributed = 0;

    if (self.isHead) {
      // ── Head path ──
      // Phase D: wait for cluster members' peer-grads.
      const intraResult = await this.awaitIntraGrads(round, intraGroup);
      if (!intraResult) {
        await this.trainer.revertToAnchor();
        this.cleanupRound(round);
        return this.report(round, false, 1, 1, false);
      }
      const clusterAggregate = weightedAverage(intraResult.grads);
      const clusterPeerCount = intraResult.totalPeerCount;

      // Phase E: send cluster aggregate to other heads.
      if (otherHeads.size > 0) {
        this.transport.send(
          { kind: "heads" },
          {
            type: "grad",
            fromPeerId: self.peerId,
            round,
            anchor: this.anchorRound,
            kind: "cluster-aggregate",
            peerCount: clusterPeerCount,
            payload: clusterAggregate,
          },
        );
      }
      // Store own cluster aggregate for inter-cluster averaging.
      this.storeInterAgg(round, self.peerId, {
        type: "grad",
        fromPeerId: self.peerId,
        round,
        anchor: this.anchorRound,
        kind: "cluster-aggregate",
        peerCount: clusterPeerCount,
        payload: clusterAggregate,
      });

      // Phase F: wait for other heads' cluster aggregates.
      const interResult = await this.awaitInterAggs(round, otherHeads);
      if (!interResult) {
        // No inter aggregates received → single-cluster mode, treat cluster
        // aggregate as global.
        if (otherHeads.size === 0) {
          globalAggregate = clusterAggregate;
          totalContributors = clusterPeerCount;
          clustersContributed = 1;
        } else {
          await this.trainer.revertToAnchor();
          this.cleanupRound(round);
          return this.report(round, false, 1, 1, false);
        }
      } else {
        globalAggregate = weightedAverage(interResult.grads);
        totalContributors = interResult.totalPeerCount;
        clustersContributed = interResult.grads.length;
      }

      // Phase G: broadcast global to cluster members. Retain it (keyed by
      // round, tagged with the pre-commit anchor a member still expects) so a
      // member that missed the broadcast can cheaply re-request it instead of
      // drifting an anchor and pulling a full-params f16w.
      this.retainGlobal(round, this.anchorRound, totalContributors, globalAggregate);
      this.transport.send(
        { kind: "cluster", clusterId: self.clusterId },
        {
          type: "grad",
          fromPeerId: self.peerId,
          round,
          anchor: this.anchorRound,
          kind: "global-aggregate",
          peerCount: totalContributors,
          payload: globalAggregate,
        },
      );
    } else {
      // ── Non-head path ──
      // Phase F': wait for global from CH.
      const global = await this.awaitGlobal(round);
      if (!global) {
        await this.trainer.revertToAnchor();
        this.cleanupRound(round);
        return this.report(round, false, 1, 1, false);
      }
      globalAggregate = global.payload;
      totalContributors = global.peerCount;
      // We don't know precisely how many clusters; trust CH's accounting.
      clustersContributed = 1;
    }

    // Phase H: apply outer step.
    await this.trainer.applyOuterStep(globalAggregate);
    this.anchorRound += 1;
    this.replayEarly();
    this.cleanupRound(round);
    return this.report(round, true, totalContributors, clustersContributed, false);
  }

  private report(
    round: RoundNumber,
    outerStepTaken: boolean,
    contributors: number,
    clustersContributed: number,
    f16wApplied: boolean,
  ): RoundReport {
    return {
      round,
      anchorAfter: this.anchorRound,
      outerStepTaken,
      contributors,
      clustersContributed,
      f16wApplied,
      innerLoss: this.lastInnerLoss,
    };
  }

  private failRound(round: RoundNumber, _reason: string): RoundReport {
    this.cleanupRound(round);
    return this.report(round, false, 1, 1, false);
  }

  // ─── Waits (event-driven) ────────────────────────────────────────────
  private async awaitIntraQuorum(
    round: RoundNumber,
    clusterMembers: Set<PeerId>,
  ): Promise<Set<PeerId> | null> {
    const deadline = Date.now() + this.opts.intraDeadlineMs;

    while (Date.now() < deadline) {
      if (!this.running) return null;
      if (this.pendingF16WAdvances()) return null;
      const ready = this.readySet.get(round);
      // Only count cluster members who are STILL in the swarm — a peer
      // who left mid-round should not gate the quorum forever.
      const alive = aliveSet(clusterMembers, this.peers, this.self);
      const expected = alive.size;
      const target = Math.ceil(expected * this.opts.quorumTargetFrac);
      const needed = Math.max(this.opts.quorumMin, target);
      if (ready) {
        const valid = intersect(ready, alive);
        if (valid.size >= needed) return valid;
      }
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await this.waitForSignal(remaining);
    }
    const ready = this.readySet.get(round);
    if (!ready) return null;
    const alive = aliveSet(clusterMembers, this.peers, this.self);
    const valid = intersect(ready, alive);
    return valid.size >= this.opts.quorumMin ? valid : null;
  }

  private async awaitIntraGrads(
    round: RoundNumber,
    intraGroup: Set<PeerId>,
  ): Promise<{
    grads: { payload: Float32Array[]; peerCount: number }[];
    totalPeerCount: number;
  } | null> {
    const deadline = Date.now() + this.opts.intraDeadlineMs;
    while (Date.now() < deadline) {
      if (!this.running) return null;
      if (this.pendingF16WAdvances()) return null;
      const grads = this.intraGrads.get(round);
      // Departed peers can't fulfil their grad obligation; require only the
      // still-alive members of the original quorum.
      const stillExpected = aliveSet(intraGroup, this.peers, this.self);
      if (grads) {
        const received = new Set<PeerId>(grads.keys());
        const valid = intersect(received, stillExpected);
        if (valid.size >= stillExpected.size) {
          return collectGrads(grads, stillExpected);
        }
      }
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await this.waitForSignal(remaining);
    }
    const grads = this.intraGrads.get(round);
    if (!grads) return null;
    const alive = aliveSet(intraGroup, this.peers, this.self);
    const valid = intersect(new Set(grads.keys()), alive);
    return valid.size >= this.opts.quorumMin
      ? collectGrads(grads, valid)
      : null;
  }

  private async awaitInterAggs(
    round: RoundNumber,
    otherHeads: Set<PeerId>,
  ): Promise<{
    grads: { payload: Float32Array[]; peerCount: number }[];
    totalPeerCount: number;
  } | null> {
    if (otherHeads.size === 0) {
      // Single-cluster — there are no other heads to await.
      const ownAggMap = this.interAggs.get(round);
      if (!ownAggMap) return null;
      // The caller (head path) handles this branch specially.
      return null;
    }
    const deadline = Date.now() + this.opts.interDeadlineMs;
    while (Date.now() < deadline) {
      if (!this.running) return null;
      if (this.pendingF16WAdvances()) return null;
      const aggs = this.interAggs.get(round);
      // Re-evaluate which heads are still around — a head that departed
      // mid-round shouldn't gate consensus across the surviving clusters.
      const expected = new Set<PeerId>(otherHeads);
      if (this.self) expected.add(this.self.peerId);
      const stillExpected = aliveSet(expected, this.peers, this.self);
      if (aggs) {
        const received = new Set<PeerId>(aggs.keys());
        const valid = intersect(received, stillExpected);
        if (valid.size >= stillExpected.size) {
          return collectGrads(aggs, stillExpected);
        }
      }
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await this.waitForSignal(remaining);
    }
    const aggs = this.interAggs.get(round);
    if (!aggs) return null;
    const expected = new Set<PeerId>(otherHeads);
    if (this.self) expected.add(this.self.peerId);
    const alive = aliveSet(expected, this.peers, this.self);
    const valid = intersect(new Set(aggs.keys()), alive);
    return valid.size >= 2 ? collectGrads(aggs, valid) : null;
  }

  private async awaitGlobal(
    round: RoundNumber,
  ): Promise<GradMessage | null> {
    const deadline = Date.now() + this.opts.globalDeadlineMs;
    // Nudge the head to resend if the global doesn't show up promptly. Capped
    // to half the deadline so at least one resend fires before we give up and
    // fall back to the (expensive) revert + f16w path.
    const retransmit = Math.max(
      200,
      Math.min(this.opts.globalRetransmitMs, Math.floor(this.opts.globalDeadlineMs / 2)),
    );
    const head = this.self ? this.findClusterHead(this.self.clusterId) : null;
    // Bounded resends: if the head can't answer after several nudges it has
    // likely left or evicted the entry — escalate to a full f16w and bail.
    const MAX_RESENDS = 4;
    let resends = 0;
    let nextResendAt = Date.now() + retransmit;
    while (Date.now() < deadline) {
      if (!this.running) return null;
      // A pending f16w that advances us strictly past this round wins (genuine
      // far-behind resync). A +1 skew does NOT set pendingF16W anymore (see
      // requestF16WIfNeeded) so it can't preempt the cheap resend below.
      if (this.pendingF16WAdvances()) return null;
      const g = this.globalAgg.get(round);
      if (g) return g;
      const now = Date.now();
      if (head && head !== this.self?.peerId && now >= nextResendAt) {
        if (resends >= MAX_RESENDS) {
          // Resends unanswered — escalate to a full-params f16w resync.
          this.lastF16WRequestedAtAnchor = -1;
          this.transport.send(
            { kind: "broadcast" },
            {
              type: "f16w-request",
              peerId: this.self!.peerId,
              atLeastAnchor: this.anchorRound + 1,
            },
          );
          return null;
        }
        this.transport.send(
          { kind: "peer", peerId: head },
          {
            type: "resend-global",
            fromPeerId: this.self!.peerId,
            round,
            anchor: this.anchorRound,
          },
        );
        resends += 1;
        nextResendAt = now + retransmit;
      }
      const remaining = Math.min(deadline - now, nextResendAt - now);
      if (remaining <= 0) continue;
      await this.waitForSignal(remaining);
    }
    return this.globalAgg.get(round) ?? null;
  }

  private pendingF16WAdvances(): boolean {
    return (
      this.pendingF16W !== null &&
      this.pendingF16W.sourceAnchor > this.anchorRound
    );
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
    this.replayEarly();
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

  private findClusterHead(clusterId: ClusterId): PeerId | null {
    for (const p of this.peers.values()) {
      if (p.clusterId === clusterId && p.isHead) return p.peerId;
    }
    return null;
  }

  // ─── Bookkeeping with signal() ───────────────────────────────────────
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

  private storeIntraGrad(
    round: RoundNumber,
    fromPeerId: PeerId,
    msg: GradMessage,
  ): void {
    let m = this.intraGrads.get(round);
    if (!m) {
      m = new Map();
      this.intraGrads.set(round, m);
    }
    m.set(fromPeerId, msg);
    this.signal();
  }

  private storeInterAgg(
    round: RoundNumber,
    fromPeerId: PeerId,
    msg: GradMessage,
  ): void {
    let m = this.interAggs.get(round);
    if (!m) {
      m = new Map();
      this.interAggs.set(round, m);
    }
    m.set(fromPeerId, msg);
    this.signal();
  }

  private storeGlobal(round: RoundNumber, msg: GradMessage): void {
    this.globalAgg.set(round, msg);
    this.signal();
  }

  /** Dispatch a ready/grad whose anchor matches ours into its buffer. */
  private applyReadyOrGrad(msg: RoundReadyMessage | GradMessage): void {
    if (msg.type === "round-ready") {
      this.markReady(msg.round, msg.peerId);
      return;
    }
    switch (msg.kind) {
      case "peer-grad":
        this.storeIntraGrad(msg.round, msg.fromPeerId, msg);
        return;
      case "cluster-aggregate":
        this.storeInterAgg(msg.round, msg.fromPeerId, msg);
        return;
      case "global-aggregate":
        this.storeGlobal(msg.round, msg);
        return;
    }
  }

  private bufferEarly(msg: RoundReadyMessage | GradMessage): void {
    if (this.earlyMessages.length >= HierarchicalBarrierStateMachine.EARLY_BUFFER_MAX) {
      this.earlyMessages.shift();
    }
    this.earlyMessages.push(msg);
  }

  /**
   * Called after any anchor advance (outer step or f16w apply). Replays
   * buffered messages that now match our anchor, keeps those still ahead, and
   * drops those now stale.
   */
  private replayEarly(): void {
    if (this.earlyMessages.length === 0) return;
    const nowMatching: Array<RoundReadyMessage | GradMessage> = [];
    const stillAhead: Array<RoundReadyMessage | GradMessage> = [];
    for (const m of this.earlyMessages) {
      if (m.anchor === this.anchorRound) nowMatching.push(m);
      else if (m.anchor > this.anchorRound) stillAhead.push(m);
      // else: stale (anchor < ours) — drop.
    }
    this.earlyMessages = stillAhead;
    for (const m of nowMatching) this.applyReadyOrGrad(m);
  }

  private cleanupRound(round: RoundNumber): void {
    // Only clear once the anchor has advanced past this round. A FAILED round
    // stays at the same anchor and retries; its readys/grads are still valid
    // for that retry. Clearing them on every failure forced both peers to
    // re-overlap their attempts in wall-clock to re-exchange readys (cleared
    // between attempts), which they routinely missed → permanent solo spin at
    // the same anchor. Keying by anchor + retaining across retries lets the
    // exchange accumulate until quorum forms. Success/f16w advance the anchor
    // first (round < anchorRound here), so the now-stale buffers are cleared.
    if (round >= this.anchorRound) return;
    this.readySet.delete(round);
    this.intraGrads.delete(round);
    this.interAggs.delete(round);
    this.globalAgg.delete(round);
  }

  // ─── Wait primitive ──────────────────────────────────────────────────
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

  private signal(): void {
    if (this.waiters.length === 0) return;
    const toFire = this.waiters;
    this.waiters = [];
    for (const w of toFire) w();
  }

  // ─── Message handling ────────────────────────────────────────────────
  private onMessage(msg: ProtocolMessage, resolveJoin: () => void): void {
    switch (msg.type) {
      case "join-ack": {
        this.self = {
          peerId: msg.peerId,
          clusterId: msg.clusterId,
          isHead: msg.isHead,
        };
        this.peers.clear();
        for (const p of msg.peers) this.peers.set(p.peerId, p);
        this.needsSync = msg.needsSync;
        resolveJoin();
        return;
      }
      case "peer-list": {
        this.peers.clear();
        for (const p of msg.peers) this.peers.set(p.peerId, p);
        // Our own role may have changed (e.g., we got promoted to head when
        // the previous head left). Track it so the next round runs the
        // correct path.
        if (this.self) {
          const me = this.peers.get(this.self.peerId);
          if (me) this.self = me;
        }
        // Departed peers may have been gating a current wait; wake waiters
        // so they re-evaluate against the new swarm view.
        this.signal();
        return;
      }
      case "round-ready":
      case "grad": {
        if (msg.anchor === this.anchorRound) {
          this.applyReadyOrGrad(msg);
        } else if (msg.anchor === this.anchorRound + 1) {
          // Arrived one anchor early: the sender already committed the outer
          // step we're still finishing. Buffer and replay it the instant we
          // advance, instead of dropping it (which would strand the sender's
          // next round — they don't retransmit readys). This is the symmetric
          // partner to the global-resend recovery; together they keep peers
          // in lockstep across a single lost/late message without a full
          // f16w resync.
          this.bufferEarly(msg);
        } else if (msg.anchor > this.anchorRound) {
          this.requestF16WIfNeeded(msg.anchor);
        }
        return;
      }
      case "resend-global":
        this.handleResendGlobal(msg.fromPeerId, msg.round);
        return;
      case "f16w-request":
        this.handleF16WRequest();
        return;
      case "f16w":
        // Mid-run: require strict advance to avoid replaying stale blobs
        // that arrive after we've already moved past them. Initial sync:
        // accept >= because both peers may be at anchor 0 and we want the
        // late joiner to adopt the first peer's params.
        if (
          msg.sourceAnchor > this.anchorRound ||
          (this.needsSync && msg.sourceAnchor >= this.anchorRound)
        ) {
          this.pendingF16W = {
            sourceAnchor: msg.sourceAnchor,
            sourceCurrentRound: msg.sourceCurrentRound,
            params: msg.params,
          };
          this.signal();
        }
        return;
      case "join":
      case "leave":
        return;
    }
  }

  /** Head-side: retain a broadcast global so we can resend it on request. */
  private retainGlobal(
    round: RoundNumber,
    anchor: AnchorRound,
    peerCount: number,
    payload: Float32Array[],
  ): void {
    this.recentGlobals.set(round, { anchor, peerCount, payload });
    // Insert-cap: drop the oldest (lowest round) beyond the keep window.
    while (this.recentGlobals.size > HierarchicalBarrierStateMachine.RECENT_GLOBALS_KEEP) {
      let oldest = Infinity;
      for (const r of this.recentGlobals.keys()) if (r < oldest) oldest = r;
      this.recentGlobals.delete(oldest);
    }
  }

  /**
   * Head-side: a cluster member missed our global broadcast for `round` and
   * asked us to resend it. If we still hold it, re-send to just that peer
   * (tagged with the round's pre-commit anchor, which the member still
   * expects). If we've advanced past it and dropped the entry, this is a
   * no-op — the member falls back to the f16w path.
   */
  private handleResendGlobal(toPeerId: PeerId, round: RoundNumber): void {
    if (!this.self) return;
    const entry = this.recentGlobals.get(round);
    if (!entry) return;
    this.transport.send(
      { kind: "peer", peerId: toPeerId },
      {
        type: "grad",
        fromPeerId: this.self.peerId,
        round,
        anchor: entry.anchor,
        kind: "global-aggregate",
        peerCount: entry.peerCount,
        payload: entry.payload,
      },
    );
  }

  private requestF16WIfNeeded(peerAnchor: AnchorRound): void {
    if (!this.self) return;
    // A +1 anchor skew means we missed exactly one outer step's global — the
    // cheap, drift-free recovery is the global-resend path (awaitGlobal nudges
    // the head to resend the round's global so we complete the SAME round). A
    // full-params f16w here would be the expensive thrash we're eliminating.
    // Only escalate to f16w when we're >=2 behind: too far for one round's
    // global to bridge. (awaitGlobal still escalates to f16w on its own if the
    // resend goes unanswered — the head left or dropped the entry.)
    if (peerAnchor <= this.anchorRound + 1) return;
    if (this.lastF16WRequestedAtAnchor >= peerAnchor) return;
    this.lastF16WRequestedAtAnchor = peerAnchor;
    this.transport.send(
      { kind: "broadcast" },
      {
        type: "f16w-request",
        peerId: this.self.peerId,
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
function intersect<T>(a: Set<T>, b: Set<T>): Set<T> {
  const out = new Set<T>();
  for (const x of a) if (b.has(x)) out.add(x);
  return out;
}

/**
 * Filter `members` down to the subset that's still in the current swarm
 * view. Self is always considered alive (we don't remove ourselves from
 * `this.peers` on departure events).
 */
function aliveSet(
  members: Set<PeerId>,
  peers: Map<PeerId, PeerInfo>,
  self: PeerInfo | null,
): Set<PeerId> {
  const out = new Set<PeerId>();
  for (const id of members) {
    if (peers.has(id) || id === self?.peerId) out.add(id);
  }
  return out;
}

function collectGrads(
  map: Map<PeerId, GradMessage>,
  ids: Set<PeerId>,
): {
  grads: { payload: Float32Array[]; peerCount: number }[];
  totalPeerCount: number;
} {
  const grads: { payload: Float32Array[]; peerCount: number }[] = [];
  let total = 0;
  for (const id of ids) {
    const m = map.get(id);
    if (!m) continue;
    grads.push({ payload: m.payload, peerCount: m.peerCount });
    total += m.peerCount;
  }
  return { grads, totalPeerCount: total };
}

/**
 * Compute weighted average across grads, weights = peerCount.
 *
 * For peer-grads (peerCount=1 each), this is a straight mean.
 * For cluster-aggregates, weight each cluster by the number of original
 * peer-grads folded into its aggregate so the result equals what a flat
 * N-way all-to-all average would have produced.
 */
function weightedAverage(
  grads: { payload: Float32Array[]; peerCount: number }[],
): Float32Array[] {
  if (grads.length === 0) throw new Error("weightedAverage: empty input");
  let totalWeight = 0;
  for (const g of grads) totalWeight += g.peerCount;
  if (totalWeight === 0) {
    throw new Error("weightedAverage: total weight is zero");
  }
  const numTensors = grads[0].payload.length;
  const out: Float32Array[] = [];
  for (let t = 0; t < numTensors; t++) {
    const len = grads[0].payload[t].length;
    const acc = new Float32Array(len);
    for (const g of grads) {
      const src = g.payload[t];
      if (src.length !== len) {
        throw new Error(
          `weightedAverage: tensor ${t} length mismatch ${src.length} vs ${len}`,
        );
      }
      const w = g.peerCount;
      for (let i = 0; i < len; i++) acc[i] += w * src[i];
    }
    const inv = 1 / totalWeight;
    for (let i = 0; i < len; i++) acc[i] *= inv;
    out.push(acc);
  }
  return out;
}
