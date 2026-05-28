/**
 * WebSocket Transport for the v2 DiLoCo relay server.
 *
 * Wire format (see server/diloco-server-v2.cjs):
 *   - Text frames: JSON {from, target, msg} for small protocol messages.
 *   - Binary frames: [u32 envelope_len LE][envelope JSON utf-8][payload]
 *     for messages carrying Float32Array[] (grad, F16W). The envelope
 *     mirrors the text JSON plus tensorShapes for payload reconstruction.
 *
 * The server's `registered` reply is translated to our JoinAck shape
 * before delivery so the state machine sees the same protocol message
 * it sees against InProcessBus.
 */

import WebSocket from "ws";
import type {
  GradMessage,
  JoinAckMessage,
  PeerId,
  PeerInfo,
  PeerListUpdateMessage,
  ProtocolMessage,
  SendTarget,
} from "../protocol/messages.ts";
import type { Transport } from "../protocol/transport.ts";

export interface WebSocketRelayTransportOptions {
  serverUrl: string;
  peerId: PeerId;
  model?: string;
  /** Override max websocket payload size (bytes). Default 500 MB. */
  maxPayload?: number;
  /** Logger; defaults to console.error. */
  log?: (msg: string) => void;
}

interface Envelope {
  from: PeerId;
  target: SendTarget;
  msg: ProtocolMessage;
  /** Per-tensor shapes for binary-frame payload reconstruction. */
  tensorShapes?: number[][];
}

const FOUR = 4;

function concatFloat32Buffers(arrays: Float32Array[]): Buffer {
  let total = 0;
  for (const a of arrays) total += a.byteLength;
  const out = Buffer.alloc(total);
  let pos = 0;
  for (const a of arrays) {
    Buffer.from(a.buffer, a.byteOffset, a.byteLength).copy(out, pos);
    pos += a.byteLength;
  }
  return out;
}

function splitFloat32Buffers(
  payload: Buffer,
  shapes: number[][],
): Float32Array[] {
  const out: Float32Array[] = [];
  let pos = 0;
  for (const shape of shapes) {
    const n = shape.reduce((a, b) => a * b, 1);
    const bytes = n * 4;
    // Copy into a fresh allocation so the underlying Node buffer can be
    // garbage-collected independently of the long-lived param tensors.
    const slice = payload.subarray(pos, pos + bytes);
    const arr = new Float32Array(n);
    Buffer.from(arr.buffer).set(slice);
    out.push(arr);
    pos += bytes;
  }
  return out;
}

function takeTensorPayload(
  msg: ProtocolMessage,
): { tensors: Float32Array[]; stripped: ProtocolMessage } | null {
  if (msg.type === "grad") {
    const { payload, ...rest } = msg as GradMessage;
    // We must not lose the payload key in the envelope shape — use a tagged
    // placeholder so the receiver's reconstructor knows to fill it back.
    return {
      tensors: payload,
      stripped: { ...rest, payload: [] } as GradMessage,
    };
  }
  if (msg.type === "f16w") {
    const { params, ...rest } = msg;
    return {
      tensors: params,
      stripped: { ...rest, params: [] },
    };
  }
  return null;
}

function restoreTensorPayload(
  msg: ProtocolMessage,
  tensors: Float32Array[],
): ProtocolMessage {
  if (msg.type === "grad") {
    return { ...(msg as GradMessage), payload: tensors };
  }
  if (msg.type === "f16w") {
    return { ...msg, params: tensors };
  }
  return msg;
}

export class WebSocketRelayTransport implements Transport {
  readonly peerId: PeerId;
  private readonly serverUrl: string;
  private readonly model: string;
  private readonly log: (msg: string) => void;
  private readonly maxPayload: number;

  private ws: WebSocket | null = null;
  private handlers: Array<(message: ProtocolMessage) => void> = [];
  private pending: ProtocolMessage[] = [];
  private hasReplayedToFirstSubscriber = false;
  private closed = false;

  private constructor(opts: WebSocketRelayTransportOptions) {
    this.peerId = opts.peerId;
    this.serverUrl = opts.serverUrl;
    this.model = opts.model ?? "gpt2";
    this.log = opts.log ?? ((m) => console.error(`[transport] ${m}`));
    this.maxPayload = opts.maxPayload ?? 500 * 1024 * 1024;
  }

  /**
   * Connect, register, return a ready Transport. The server's `registered`
   * reply is translated to a JoinAck and queued for the first subscriber.
   */
  static async create(
    opts: WebSocketRelayTransportOptions,
  ): Promise<WebSocketRelayTransport> {
    const t = new WebSocketRelayTransport(opts);
    await t.connectAndRegister();
    return t;
  }

  private async connectAndRegister(): Promise<void> {
    const ws = new WebSocket(this.serverUrl, { maxPayload: this.maxPayload });
    this.ws = ws;

    await new Promise<void>((resolve, reject) => {
      const onError = (e: Error) => {
        reject(e);
      };
      ws.once("error", onError);
      ws.once("open", () => {
        ws.off("error", onError);
        resolve();
      });
      setTimeout(() => reject(new Error("WebSocket open timeout")), 15_000);
    });

    // Wait for the registered reply BEFORE returning. Buffer everything
    // else that arrives in the meantime — it goes to onReceive once a
    // subscriber attaches.
    const registered = await new Promise<JoinAckMessage>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error("Registration timeout")),
        15_000,
      );
      const onMessage = (raw: WebSocket.RawData, isBinary?: boolean) => {
        const parsed = this.parseFrame(raw, isBinary === true);
        if (!parsed) return;
        if (
          parsed.kind === "text" &&
          (parsed.value as { type?: string }).type === "registered"
        ) {
          clearTimeout(timer);
          ws.off("message", onMessage);
          // Translate "registered" → JoinAck.
          const r = parsed.value as {
            peerId: string;
            clusterId: number;
            isHead: boolean;
            peers: PeerInfo[];
          };
          const ack: JoinAckMessage = {
            type: "join-ack",
            peerId: r.peerId,
            clusterId: r.clusterId,
            isHead: r.isHead,
            peers: r.peers,
          };
          resolve(ack);
        } else if (parsed.kind === "protocol") {
          // Anything else (peer-list, grads, etc.) arriving pre-subscription
          // gets queued.
          this.pending.push(parsed.value);
        }
      };
      ws.on("message", onMessage);
      ws.send(
        JSON.stringify({
          type: "register",
          peerId: this.peerId,
          model: this.model,
        }),
      );
    });

    // Hand-off: once registration is done, wire the steady-state handler
    // and queue the JoinAck so the first subscriber sees it.
    this.pending.unshift(registered);
    this.ws.on("message", (raw, isBinary) =>
      this.onWsMessage(raw, isBinary === true),
    );
    this.ws.on("close", () => {
      this.closed = true;
      this.log("websocket closed");
    });
  }

  private onWsMessage(raw: WebSocket.RawData, isBinary: boolean): void {
    const parsed = this.parseFrame(raw, isBinary);
    if (!parsed || parsed.kind !== "protocol") return;
    this.deliver(parsed.value);
  }

  /**
   * Parse an inbound frame. `isBinary` is the ws-library opcode flag — the
   * payload arrives as Buffer either way, so this is the only reliable
   * way to tell text from binary frames.
   */
  private parseFrame(
    raw: WebSocket.RawData,
    isBinary: boolean,
  ):
    | { kind: "text"; value: unknown }
    | { kind: "protocol"; value: ProtocolMessage }
    | null {
    if (isBinary && Buffer.isBuffer(raw)) {
      if (raw.length < FOUR) return null;
      const envLen = raw.readUInt32LE(0);
      if (envLen <= 0 || envLen > raw.length - FOUR) return null;
      let envelope: Envelope | null = null;
      try {
        envelope = JSON.parse(raw.toString("utf8", FOUR, FOUR + envLen));
      } catch {
        return null;
      }
      if (!envelope) return null;
      const payload = raw.subarray(FOUR + envLen);
      const tensors = envelope.tensorShapes
        ? splitFloat32Buffers(payload, envelope.tensorShapes)
        : [];
      const restored = restoreTensorPayload(envelope.msg, tensors);
      return { kind: "protocol", value: restored };
    }

    // Text frame.
    let text: unknown;
    try {
      text = JSON.parse(raw.toString());
    } catch {
      return null;
    }
    if (!text || typeof text !== "object") return null;
    const obj = text as Record<string, unknown>;
    if (obj.type === "registered" || obj.type === "pong") {
      return { kind: "text", value: obj };
    }
    // Server-originated control messages (peer-list) come through bare —
    // i.e., as the protocol message itself, NOT wrapped in {from,target,msg}.
    if (obj.type === "peer-list") {
      return { kind: "protocol", value: obj as PeerListUpdateMessage };
    }
    // Protocol envelope: {from, target, msg}.
    if (obj.target && obj.msg) {
      return { kind: "protocol", value: obj.msg as ProtocolMessage };
    }
    return null;
  }

  send(target: SendTarget, message: ProtocolMessage): void {
    if (this.closed || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    const taken = takeTensorPayload(message);
    if (taken) {
      const envelope: Envelope = {
        from: this.peerId,
        target,
        msg: taken.stripped,
        tensorShapes: taken.tensors.map((a) => [a.length]),
      };
      const envBytes = Buffer.from(JSON.stringify(envelope), "utf8");
      const lenBuf = Buffer.alloc(FOUR);
      lenBuf.writeUInt32LE(envBytes.length, 0);
      const payloadBuf = concatFloat32Buffers(taken.tensors);
      this.ws.send(Buffer.concat([lenBuf, envBytes, payloadBuf]));
    } else {
      const envelope: Envelope = {
        from: this.peerId,
        target,
        msg: message,
      };
      this.ws.send(JSON.stringify(envelope));
    }
  }

  onReceive(handler: (message: ProtocolMessage) => void): () => void {
    this.handlers.push(handler);
    if (!this.hasReplayedToFirstSubscriber) {
      this.hasReplayedToFirstSubscriber = true;
      // Replay any messages buffered before subscription (notably JoinAck).
      const pending = this.pending;
      this.pending = [];
      // Use queueMicrotask so the caller's constructor returns before any
      // handler invocations — mirrors InProcessBus's join-ack delivery.
      queueMicrotask(() => {
        for (const m of pending) handler(m);
      });
    }
    return () => {
      const i = this.handlers.indexOf(handler);
      if (i >= 0) this.handlers.splice(i, 1);
    };
  }

  private deliver(msg: ProtocolMessage): void {
    if (this.handlers.length === 0) {
      this.pending.push(msg);
      return;
    }
    for (const h of this.handlers.slice()) h(msg);
  }

  close(): void {
    this.closed = true;
    if (this.ws) {
      try {
        this.ws.close();
      } catch {}
      this.ws = null;
    }
  }
}
