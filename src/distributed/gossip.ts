/**
 * Gossip Network for DiLoCo Pseudo-Gradient Exchange
 *
 * Each peer connects to the PeerJS cloud signaling server and maintains
 * WebRTC data channels with a handful of other peers. When a peer
 * finishes an inner training loop, it sends E3M0-compressed pseudo-
 * gradients to its connected peers and incorporates received ones.
 *
 * No coordinator, no rounds, no barriers. Peers operate independently
 * at their own speed. Fast peers exchange more frequently. Slow peers
 * contribute less often but still pull the swarm toward their data.
 *
 * Topology: each peer targets K connections. When a peer joins, it
 * connects to K random existing peers. Connections churn naturally
 * (peers leave, new ones join). The swarm is self-healing.
 */

import { e3m0CompressedSize, e3m0Dequantize, e3m0Quantize } from "./e3m0";

// ============================================================================
// Types
// ============================================================================

/** A compressed pseudo-gradient ready for transmission. */
export interface CompressedPseudoGrad {
  /** Parameter index (which parameter this gradient is for) */
  paramIndex: number;
  /** Packed 4-bit E3M0 codes */
  codes: Uint32Array;
  /** Per-block max exponents */
  scales: Uint8Array;
  /** Number of original f32 values */
  numValues: number;
}

/** A complete sync message: all parameters' pseudo-gradients. */
export interface SyncMessage {
  /** Sender's peer ID */
  senderId: string;
  /** Outer step number (so receiver can detect stale messages) */
  outerStep: number;
  /** Compressed pseudo-gradients for each parameter */
  grads: CompressedPseudoGrad[];
}

/** Callback when pseudo-gradients are received from a peer. */
export type OnReceiveCallback = (
  pseudoGrads: Float32Array[],
  senderId: string,
  outerStep: number,
) => void;

export interface GossipConfig {
  /** Target number of peer connections (default: 5) */
  targetPeers?: number;
  /** Explicit PeerJS ID (default: random). Set for server agents so browsers can connect by known ID. */
  peerId?: string;
  /** PeerJS IDs to connect to on startup (skip discovery). */
  connectTo?: string[];
  /** Custom PeerJS server config. Default: PeerJS cloud. */
  peerServer?: { host: string; port: number; path: string; secure?: boolean };
  /** ICE configuration (STUN/TURN servers). */
  iceConfig?: {
    iceServers: Array<{ urls: string; username?: string; credential?: string }>;
  };
  /** Callback when pseudo-gradients are received */
  onReceive: OnReceiveCallback;
  /** Callback when peer count changes */
  onPeerCountChange?: (count: number) => void;
  /** Callback for status messages */
  onStatus?: (msg: string) => void;
}

// ============================================================================
// Serialization
// ============================================================================

/** Serialize a SyncMessage to an ArrayBuffer for WebRTC transmission. */
function serializeSyncMessage(msg: SyncMessage): ArrayBuffer {
  // Header: outerStep (u32) + numParams (u32) + senderId length (u16) + senderId (utf8)
  // Per param: paramIndex (u32) + numValues (u32) + codesLen (u32) + codes + scalesLen (u32) + scales
  const senderBytes = new TextEncoder().encode(msg.senderId);
  let totalSize = 4 + 4 + 2 + senderBytes.length; // header
  for (const g of msg.grads) {
    totalSize += 4 + 4 + 4 + g.codes.byteLength + 4 + g.scales.byteLength;
  }

  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);
  let offset = 0;

  view.setUint32(offset, msg.outerStep, true);
  offset += 4;
  view.setUint32(offset, msg.grads.length, true);
  offset += 4;
  view.setUint16(offset, senderBytes.length, true);
  offset += 2;
  u8.set(senderBytes, offset);
  offset += senderBytes.length;

  for (const g of msg.grads) {
    view.setUint32(offset, g.paramIndex, true);
    offset += 4;
    view.setUint32(offset, g.numValues, true);
    offset += 4;
    view.setUint32(offset, g.codes.byteLength, true);
    offset += 4;
    u8.set(
      new Uint8Array(g.codes.buffer, g.codes.byteOffset, g.codes.byteLength),
      offset,
    );
    offset += g.codes.byteLength;
    view.setUint32(offset, g.scales.byteLength, true);
    offset += 4;
    u8.set(g.scales, offset);
    offset += g.scales.byteLength;
  }

  return buf;
}

/** Deserialize a SyncMessage from an ArrayBuffer. */
function deserializeSyncMessage(buf: ArrayBuffer): SyncMessage {
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);
  let offset = 0;

  const outerStep = view.getUint32(offset, true);
  offset += 4;
  const numParams = view.getUint32(offset, true);
  offset += 4;
  const senderIdLen = view.getUint16(offset, true);
  offset += 2;
  const senderId = new TextDecoder().decode(
    u8.slice(offset, offset + senderIdLen),
  );
  offset += senderIdLen;

  const grads: CompressedPseudoGrad[] = [];
  for (let i = 0; i < numParams; i++) {
    const paramIndex = view.getUint32(offset, true);
    offset += 4;
    const numValues = view.getUint32(offset, true);
    offset += 4;
    const codesLen = view.getUint32(offset, true);
    offset += 4;
    const codes = new Uint32Array(buf.slice(offset, offset + codesLen));
    offset += codesLen;
    const scalesLen = view.getUint32(offset, true);
    offset += 4;
    const scales = new Uint8Array(buf.slice(offset, offset + scalesLen));
    offset += scalesLen;
    grads.push({ paramIndex, codes, scales, numValues });
  }

  return { senderId, outerStep, grads };
}

// ============================================================================
// Gossip Network
// ============================================================================

export class GossipNetwork {
  private peer: any = null; // PeerJS Peer instance
  private connections = new Map<string, any>(); // peerId → DataConnection
  private readonly targetPeers: number;
  private readonly onReceive: OnReceiveCallback;
  private readonly onPeerCountChange?: (count: number) => void;
  private readonly onStatus?: (msg: string) => void;
  private myId = "";
  private destroyed = false;

  private readonly explicitPeerId?: string;
  private readonly connectTo?: string[];
  private readonly peerServerConfig?: GossipConfig["peerServer"];
  private readonly iceConfig?: GossipConfig["iceConfig"];

  constructor(config: GossipConfig) {
    this.targetPeers = config.targetPeers ?? 5;
    this.explicitPeerId = config.peerId;
    this.connectTo = config.connectTo;
    this.peerServerConfig = config.peerServer;
    this.iceConfig = config.iceConfig;
    this.onReceive = config.onReceive;
    this.onPeerCountChange = config.onPeerCountChange;
    this.onStatus = config.onStatus;
  }

  /** Join the gossip network. Returns this peer's ID. */
  async join(): Promise<string> {
    // Polyfill WebRTC for Node.js if not in browser
    if (typeof globalThis.RTCPeerConnection === "undefined") {
      try {
        const wrtc = await import("@roamhq/wrtc");
        (globalThis as any).RTCPeerConnection =
          (wrtc as any).default?.RTCPeerConnection ?? wrtc.RTCPeerConnection;
        (globalThis as any).RTCSessionDescription =
          (wrtc as any).default?.RTCSessionDescription ??
          wrtc.RTCSessionDescription;
        (globalThis as any).RTCIceCandidate =
          (wrtc as any).default?.RTCIceCandidate ?? wrtc.RTCIceCandidate;
      } catch (e) {
        this.onStatus?.(`WebRTC polyfill failed: ${e}`);
      }
    }
    const { Peer } = await import("peerjs");

    return new Promise((resolve, reject) => {
      // Generate a deterministic-ish prefix so peers can find each other
      // by listing peers with this prefix on the PeerJS server.
      const prefix = "torchlette-diloco-";
      const id =
        this.explicitPeerId ?? prefix + Math.random().toString(36).slice(2, 10);

      this.peer = new Peer(id, {
        debug: 0, // silent
        ...(this.peerServerConfig ?? {}),
        ...(this.iceConfig ? { config: this.iceConfig } : {}),
      });

      this.peer.on("open", (myId: string) => {
        this.myId = myId;
        this.onStatus?.(`Joined as ${myId}`);
        // Connect to explicit peers first, then discover
        if (this.connectTo) {
          for (const peerId of this.connectTo) {
            if (peerId === myId) continue;
            const conn = this.peer.connect(peerId, { reliable: true });
            this.setupConnection(conn);
          }
        }
        this.discoverPeers();
        resolve(myId);
      });

      this.peer.on("connection", (conn: any) => {
        this.setupConnection(conn);
      });

      this.peer.on("error", (err: any) => {
        this.onStatus?.(`PeerJS error: ${err.type}`);
        if (!this.myId) reject(err);
      });
    });
  }

  /** Send compressed pseudo-gradients to all connected peers. */
  broadcastPseudoGrads(pseudoGrads: Float32Array[], outerStep: number): void {
    // Compress each parameter's pseudo-gradient
    const grads: CompressedPseudoGrad[] = [];
    for (let i = 0; i < pseudoGrads.length; i++) {
      const pg = pseudoGrads[i];
      // Pad to multiple of 8 for E3M0
      const padded = new Float32Array(Math.ceil(pg.length / 8) * 8);
      padded.set(pg);
      const { codes, scales } = e3m0Quantize(padded);
      grads.push({ paramIndex: i, codes, scales, numValues: pg.length });
    }

    const msg: SyncMessage = {
      senderId: this.myId,
      outerStep,
      grads,
    };

    const buf = serializeSyncMessage(msg);
    const sizeMB = (buf.byteLength / (1024 * 1024)).toFixed(1);
    this.onStatus?.(
      `Broadcasting ${sizeMB}MB to ${this.connections.size} peers`,
    );

    for (const [, conn] of this.connections) {
      if (conn.open) {
        conn.send(buf);
      }
    }
  }

  /** Number of active peer connections. */
  get peerCount(): number {
    return this.connections.size;
  }

  /** Get all connected peer IDs. */
  get connectedPeers(): string[] {
    return Array.from(this.connections.keys());
  }

  /** Leave the network. */
  destroy(): void {
    this.destroyed = true;
    for (const [, conn] of this.connections) {
      conn.close();
    }
    this.connections.clear();
    this.peer?.destroy();
    this.peer = null;
  }

  // --------------------------------------------------------------------------
  // Internal
  // --------------------------------------------------------------------------

  private setupConnection(conn: any): void {
    const peerId = conn.peer;

    conn.on("open", () => {
      this.connections.set(peerId, conn);
      this.onStatus?.(
        `Connected to ${peerId} (${this.connections.size} peers)`,
      );
      this.onPeerCountChange?.(this.connections.size);
    });

    conn.on("data", (data: ArrayBuffer) => {
      try {
        const msg = deserializeSyncMessage(data);
        // Decompress pseudo-gradients
        const pseudoGrads: Float32Array[] = [];
        for (const g of msg.grads) {
          const restored = e3m0Dequantize(g.codes, g.scales, g.numValues);
          pseudoGrads.push(restored.slice(0, g.numValues));
        }
        this.onReceive(pseudoGrads, msg.senderId, msg.outerStep);
      } catch (e) {
        this.onStatus?.(`Failed to parse message from ${peerId}: ${e}`);
      }
    });

    conn.on("close", () => {
      this.connections.delete(peerId);
      this.onStatus?.(
        `Disconnected from ${peerId} (${this.connections.size} peers)`,
      );
      this.onPeerCountChange?.(this.connections.size);
      // Try to replace lost connection
      if (!this.destroyed) this.discoverPeers();
    });

    conn.on("error", (err: any) => {
      this.onStatus?.(`Connection error with ${peerId}: ${err}`);
      this.connections.delete(peerId);
      this.onPeerCountChange?.(this.connections.size);
    });
  }

  private async discoverPeers(): Promise<void> {
    if (this.destroyed || !this.peer) return;
    if (this.connections.size >= this.targetPeers) return;

    try {
      // PeerJS cloud server supports listing peers
      // This is a simple discovery mechanism — in production you'd use
      // a dedicated tracker or DHT.
      const peers: string[] = await new Promise((resolve, reject) => {
        this.peer.listAllPeers((list: string[]) => {
          resolve(list);
        });
        // Timeout after 5s
        setTimeout(() => resolve([]), 5000);
      });

      // Filter to our prefix, exclude self, exclude already connected
      const candidates = peers.filter(
        (id) =>
          id.startsWith("torchlette-diloco-") &&
          id !== this.myId &&
          !this.connections.has(id),
      );

      // Connect to random candidates up to target
      const needed = this.targetPeers - this.connections.size;
      const shuffled = candidates.sort(() => Math.random() - 0.5);
      const toConnect = shuffled.slice(0, needed);

      for (const peerId of toConnect) {
        if (this.connections.has(peerId)) continue;
        const conn = this.peer.connect(peerId, { reliable: true });
        this.setupConnection(conn);
      }

      if (toConnect.length > 0) {
        this.onStatus?.(`Connecting to ${toConnect.length} new peers...`);
      }
    } catch (e) {
      this.onStatus?.(`Peer discovery failed: ${e}`);
    }

    // Periodically rediscover if we don't have enough peers
    if (!this.destroyed && this.connections.size < this.targetPeers) {
      setTimeout(() => this.discoverPeers(), 10000);
    }
  }
}
