/**
 * In-process Transport bus for tests.
 *
 * Spawns N peers in a single Node process, all communicating through a
 * shared in-memory message bus. The bus also plays the role of the
 * production relay's coordinator: it assigns each connecting peer to a
 * cluster, designates cluster heads, and broadcasts peer-list updates on
 * membership changes.
 *
 * Routing rules mirror what the production WebRTC relay will do, so a
 * protocol test that passes against this bus tells you something true
 * about how the same protocol will behave against the relay.
 *
 * Delivery is via queueMicrotask: deterministic ordering (sender's queue
 * order is preserved per receiver), but always asynchronous (no synchronous
 * loop back into the sender's stack). That matches how a real network
 * delivers — never on the sender's call stack — without introducing the
 * scheduler nondeterminism that setTimeout would.
 */

import type {
  ClusterId,
  JoinAckMessage,
  PeerId,
  PeerInfo,
  PeerListUpdateMessage,
  ProtocolMessage,
  SendTarget,
} from "../protocol/messages.ts";
import type { Transport } from "../protocol/transport.ts";

export interface ClusterAssigner {
  /**
   * Decide which cluster a newly-joining peer belongs to and whether they
   * are the head of that cluster. Called with the swarm view BEFORE the
   * new peer is added.
   */
  assign(
    peerId: PeerId,
    currentPeers: readonly PeerInfo[],
  ): { clusterId: ClusterId; isHead: boolean };
}

/**
 * Default assigner: fill clusters in order up to `clusterSize`, then start a
 * new cluster. First peer in each cluster is its head; subsequent peers
 * inherit head-ness when the previous head leaves (handled by the bus).
 *
 * Use `clusterSize: Infinity` for flat mode (one cluster, first peer is
 * head, hierarchical phases collapse to flat).
 */
export class FixedClusterAssigner implements ClusterAssigner {
  constructor(private readonly clusterSize: number) {}

  assign(
    _peerId: PeerId,
    currentPeers: readonly PeerInfo[],
  ): { clusterId: ClusterId; isHead: boolean } {
    const counts = new Map<ClusterId, number>();
    let maxClusterId = -1;
    for (const p of currentPeers) {
      counts.set(p.clusterId, (counts.get(p.clusterId) ?? 0) + 1);
      if (p.clusterId > maxClusterId) maxClusterId = p.clusterId;
    }
    // Find smallest non-full cluster
    let clusterId = -1;
    let smallestCount = Number.POSITIVE_INFINITY;
    for (const [cid, count] of counts) {
      if (count < this.clusterSize && count < smallestCount) {
        clusterId = cid;
        smallestCount = count;
      }
    }
    if (clusterId === -1) {
      clusterId = maxClusterId + 1; // start a fresh cluster (>= 0)
    }
    const existingHead = currentPeers.some(
      (p) => p.clusterId === clusterId && p.isHead,
    );
    return { clusterId, isHead: !existingHead };
  }
}

interface BusPeer {
  info: PeerInfo;
  handlers: ((message: ProtocolMessage) => void)[];
  closed: boolean;
}

/**
 * Test-only fault injector. Consulted for every routed message (NOT for the
 * bus-originated join-ack / peer-list, which model the coordinator, not the
 * peer link). Return `drop` to model a lost message, `delayMs` to model link
 * latency (a slow peer). Undefined = current instant queueMicrotask delivery.
 */
export type FaultHook = (ctx: {
  from: PeerId;
  to: PeerId;
  message: ProtocolMessage;
}) => { drop?: boolean; delayMs?: number } | void;

export class InProcessBus {
  private readonly peers = new Map<PeerId, BusPeer>();

  /** Test-only: see {@link FaultHook}. Default undefined = no faults. */
  faultHook: FaultHook | undefined = undefined;

  constructor(private readonly assigner: ClusterAssigner) {}

  /**
   * Connect a peer. Returns a Transport bound to that peer-id; the
   * connecting peer receives a join-ack and existing peers see a
   * peer-list update, both delivered asynchronously.
   */
  connect(peerId: PeerId): Transport {
    if (this.peers.has(peerId)) {
      throw new Error(`InProcessBus: duplicate peer ${peerId}`);
    }
    const currentInfo = Array.from(this.peers.values()).map((p) => p.info);
    const { clusterId, isHead } = this.assigner.assign(peerId, currentInfo);
    const needsSync = currentInfo.length > 0;
    const info: PeerInfo = { peerId, clusterId, isHead };
    const peer: BusPeer = { info, handlers: [], closed: false };
    this.peers.set(peerId, peer);

    const transport: Transport = {
      peerId,
      send: (target, message) => this.route(peerId, target, message),
      onReceive: (handler) => {
        peer.handlers.push(handler);
        return () => {
          const i = peer.handlers.indexOf(handler);
          if (i >= 0) peer.handlers.splice(i, 1);
        };
      },
      close: () => this.disconnect(peerId),
    };

    // Deliver join-ack to the new peer, then broadcast peer-list update to
    // everyone else. Both async so the caller's `connect()` returns before
    // any handler fires (mirrors network behavior).
    queueMicrotask(() => {
      if (peer.closed) return;
      const allPeers = this.snapshotPeers();
      const ack: JoinAckMessage = {
        type: "join-ack",
        peerId,
        clusterId,
        isHead,
        peers: allPeers,
        needsSync,
      };
      this.deliverTo(peer, ack);

      const update: PeerListUpdateMessage = {
        type: "peer-list",
        peers: allPeers,
        added: [peerId],
        removed: [],
      };
      for (const other of this.peers.values()) {
        if (other.info.peerId !== peerId) this.deliverTo(other, update);
      }
    });

    return transport;
  }

  private disconnect(peerId: PeerId): void {
    const peer = this.peers.get(peerId);
    if (!peer || peer.closed) return;
    peer.closed = true;
    this.peers.delete(peerId);

    // If we just disconnected a cluster head, promote the next surviving
    // peer in that cluster (lowest peer-id) so the cluster keeps a head.
    if (peer.info.isHead) {
      const remaining = Array.from(this.peers.values())
        .filter((p) => p.info.clusterId === peer.info.clusterId)
        .sort((a, b) => a.info.peerId.localeCompare(b.info.peerId));
      if (remaining.length > 0) {
        remaining[0].info = { ...remaining[0].info, isHead: true };
      }
    }

    const allPeers = this.snapshotPeers();
    const update: PeerListUpdateMessage = {
      type: "peer-list",
      peers: allPeers,
      added: [],
      removed: [peerId],
    };
    queueMicrotask(() => {
      for (const other of this.peers.values()) this.deliverTo(other, update);
    });
  }

  private route(
    fromPeerId: PeerId,
    target: SendTarget,
    message: ProtocolMessage,
  ): void {
    const sender = this.peers.get(fromPeerId);
    if (!sender || sender.closed) return;

    const targets: BusPeer[] = (() => {
      switch (target.kind) {
        case "broadcast":
          return Array.from(this.peers.values()).filter(
            (p) => p.info.peerId !== fromPeerId,
          );
        case "peer": {
          const p = this.peers.get(target.peerId);
          return p && !p.closed ? [p] : [];
        }
        case "cluster":
          return Array.from(this.peers.values()).filter(
            (p) =>
              p.info.clusterId === target.clusterId &&
              p.info.peerId !== fromPeerId,
          );
        case "heads":
          return Array.from(this.peers.values()).filter(
            (p) => p.info.isHead && p.info.peerId !== fromPeerId,
          );
      }
    })();

    const hook = this.faultHook;
    if (!hook) {
      queueMicrotask(() => {
        for (const t of targets) this.deliverTo(t, message);
      });
      return;
    }
    // Per-target fault evaluation so a slow/lossy link to one peer doesn't
    // affect delivery to others (matches a real per-connection relay).
    for (const t of targets) {
      const verdict = hook({ from: fromPeerId, to: t.info.peerId, message }) || {};
      if (verdict.drop) continue;
      const deliver = () => this.deliverTo(t, message);
      if (verdict.delayMs && verdict.delayMs > 0) {
        setTimeout(deliver, verdict.delayMs);
      } else {
        queueMicrotask(deliver);
      }
    }
  }

  private deliverTo(peer: BusPeer, message: ProtocolMessage): void {
    if (peer.closed) return;
    // Iterate over a snapshot — handlers may unsubscribe themselves.
    for (const h of peer.handlers.slice()) h(message);
  }

  private snapshotPeers(): PeerInfo[] {
    return Array.from(this.peers.values()).map((p) => ({ ...p.info }));
  }

  /** Test helper: current swarm view. */
  peerView(): readonly PeerInfo[] {
    return this.snapshotPeers();
  }
}
