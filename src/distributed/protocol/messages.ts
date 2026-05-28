/**
 * Wire types for the DiLoCo barrier protocol.
 *
 * Two categories:
 *   - Control messages (JSON-serializable): membership, ready-signals, F16W
 *     requests, cluster assignment. Small, structural.
 *   - Payload-heavy messages: GRAD and F16W carry Float32Array[] payloads
 *     that should not be JSON-encoded in production transports (browsers
 *     and the relay use a binary framing layer; the in-process transport
 *     can pass them by reference).
 *
 * Routing is NOT part of the message — see Transport.send(target, message).
 * Keeping content and routing orthogonal lets the in-process bus mimic the
 * production relay's routing rules without changing the protocol shape.
 */

export type PeerId = string;

/** Cluster index in hierarchical mode. Always 0 in flat mode. */
export type ClusterId = number;

/**
 * Outer-step counter. Increments by 1 each time a peer successfully takes
 * an outer step. Two peers with the same anchor agree their pseudograds
 * are displacements from the same point in weight space.
 */
export type AnchorRound = number;

/**
 * Local loop counter. Increments by 1 every loop iteration (whether or
 * not an outer step happened). Used for logging and cross-peer alignment
 * after F16W resync.
 */
export type RoundNumber = number;

// ─── Membership ──────────────────────────────────────────────────────────

export interface PeerInfo {
  peerId: PeerId;
  clusterId: ClusterId;
  isHead: boolean;
}

export interface JoinMessage {
  type: "join";
  peerId: PeerId;
}

/**
 * Sent by the coordinator (relay) to a joining peer with their cluster
 * assignment and the current swarm view. Receipt of this message is when
 * the peer transitions from "connecting" to "training".
 */
export interface JoinAckMessage {
  type: "join-ack";
  peerId: PeerId;
  clusterId: ClusterId;
  isHead: boolean;
  peers: PeerInfo[];
}

/**
 * Coordinator broadcasts on every membership change. Receivers reconcile
 * their local PeerInfo cache against `peers`; `added`/`removed` are hints
 * for efficient diffing but the full `peers` set is authoritative.
 */
export interface PeerListUpdateMessage {
  type: "peer-list";
  peers: PeerInfo[];
  added: PeerId[];
  removed: PeerId[];
}

export interface LeaveMessage {
  type: "leave";
  peerId: PeerId;
}

// ─── Barrier protocol ────────────────────────────────────────────────────

/**
 * Peer announces it has finished its K inner steps for `round` and is
 * ready to enter the outer-step barrier. Anchor lets receivers detect
 * cross-anchor mismatches before any gradient bytes flow.
 */
export interface RoundReadyMessage {
  type: "round-ready";
  peerId: PeerId;
  round: RoundNumber;
  anchor: AnchorRound;
  clusterId: ClusterId;
}

// ─── Payload-heavy messages ──────────────────────────────────────────────

/**
 * Pseudogradient (or aggregated pseudogradient) exchange.
 *
 * In flat mode, every peer broadcasts kind="peer-grad" to every other.
 *
 * In hierarchical mode:
 *   - cluster members send kind="peer-grad" to their cluster head;
 *   - cluster head aggregates and emits kind="cluster-aggregate" to other
 *     cluster heads;
 *   - cluster heads compute the global average and emit
 *     kind="global-aggregate" back to their cluster members.
 *
 * `peerCount` is the number of original peer pseudograds folded into
 * `payload`. Receivers use it to weight the aggregate correctly when
 * combining at the next level.
 */
export interface GradMessage {
  type: "grad";
  fromPeerId: PeerId;
  round: RoundNumber;
  anchor: AnchorRound;
  kind: "peer-grad" | "cluster-aggregate" | "global-aggregate";
  peerCount: number;
  payload: Float32Array[];
}

/**
 * Lagging peer requests full param resync from anyone whose anchor is
 * already `atLeastAnchor` or higher. The relay forwards to a peer at or
 * past that anchor (preferring our cluster head when available).
 */
export interface F16WRequestMessage {
  type: "f16w-request";
  peerId: PeerId;
  atLeastAnchor: AnchorRound;
}

/**
 * Full f16-encoded parameter dump used for recovery after long disconnects
 * or for late-joiners. The receiver applies these params, sets its anchor
 * to `sourceAnchor`, advances currentRound to `sourceCurrentRound`, and
 * resets inner+outer optimizer state.
 */
export interface F16WMessage {
  type: "f16w";
  fromPeerId: PeerId;
  sourceAnchor: AnchorRound;
  sourceCurrentRound: RoundNumber;
  params: Float32Array[];
}

// ─── Union ───────────────────────────────────────────────────────────────

export type ProtocolMessage =
  | JoinMessage
  | JoinAckMessage
  | PeerListUpdateMessage
  | LeaveMessage
  | RoundReadyMessage
  | GradMessage
  | F16WRequestMessage
  | F16WMessage;

/**
 * Routing target for Transport.send. Orthogonal to message content.
 *
 *   - broadcast: deliver to all currently-known peers
 *   - peer:      deliver to a single peer-id
 *   - cluster:   deliver to all peers in a cluster (used by CH broadcasts)
 *   - heads:     deliver to all cluster heads (used by inter-cluster
 *                cluster-aggregate exchange)
 */
export type SendTarget =
  | { kind: "broadcast" }
  | { kind: "peer"; peerId: PeerId }
  | { kind: "cluster"; clusterId: ClusterId }
  | { kind: "heads" };
