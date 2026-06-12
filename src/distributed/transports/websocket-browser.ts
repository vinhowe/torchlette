/**
 * Browser-side WebSocket Transport.
 *
 * Mirrors WebSocketRelayTransport (Node) but uses the browser's global
 * WebSocket, ArrayBuffer-based binary frames, and addEventListener-style
 * event handling. Wire format is identical — same envelope JSON, same
 * length-prefixed binary layout — so a browser peer is indistinguishable
 * from a Node peer to the relay.
 */

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

export interface WebSocketBrowserTransportOptions {
  serverUrl: string;
  peerId: PeerId;
  model?: string;
  log?: (msg: string) => void;
}

interface Envelope {
  from: PeerId;
  target: SendTarget;
  msg: ProtocolMessage;
  tensorShapes?: number[][];
  /** Payload element encoding. Absent = f32 (back-compat). */
  wireDtype?: import("../wire-codec").WireDtype;
}

const FOUR = 4;

function takeTensorPayload(
  msg: ProtocolMessage,
): { tensors: Float32Array[]; stripped: ProtocolMessage } | null {
  if (msg.type === "grad") {
    const { payload, ...rest } = msg as GradMessage;
    return {
      tensors: payload,
      stripped: { ...rest, payload: [] } as GradMessage,
    };
  }
  if (msg.type === "f16w") {
    const { params, ...rest } = msg;
    return { tensors: params, stripped: { ...rest, params: [] } };
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

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

export class WebSocketBrowserTransport implements Transport {
  readonly peerId: PeerId;
  private readonly serverUrl: string;
  private readonly model: string;
  private readonly log: (msg: string) => void;

  private ws: WebSocket | null = null;
  private handlers: Array<(message: ProtocolMessage) => void> = [];
  private pending: ProtocolMessage[] = [];
  private hasReplayedToFirstSubscriber = false;
  private closed = false;
  private reconnecting = false;
  private reconnectDelayMs = 1_000;
  private readonly reconnectDelayMaxMs = 30_000;

  private constructor(opts: WebSocketBrowserTransportOptions) {
    this.peerId = opts.peerId;
    this.serverUrl = opts.serverUrl;
    this.model = opts.model ?? "gpt2";
    this.log = opts.log ?? ((m) => console.warn(`[transport] ${m}`));
  }

  static async create(
    opts: WebSocketBrowserTransportOptions,
  ): Promise<WebSocketBrowserTransport> {
    const t = new WebSocketBrowserTransport(opts);
    await t.connectAndRegister();
    return t;
  }

  private async connectAndRegister(): Promise<void> {
    const ws = new WebSocket(this.serverUrl);
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    await new Promise<void>((resolve, reject) => {
      const onError = (e: Event) => {
        ws.removeEventListener("error", onError);
        reject(new Error(`websocket open failed: ${(e as ErrorEvent).message ?? "unknown"}`));
      };
      ws.addEventListener("error", onError, { once: true });
      ws.addEventListener("open", () => {
        ws.removeEventListener("error", onError);
        resolve();
      }, { once: true });
      setTimeout(() => reject(new Error("WebSocket open timeout")), 15_000);
    });

    const registered = await new Promise<JoinAckMessage>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error("Registration timeout")),
        15_000,
      );
      const onMessage = (event: MessageEvent) => {
        const parsed = this.parseFrame(event.data);
        if (!parsed) return;
        if (
          parsed.kind === "text" &&
          (parsed.value as { type?: string }).type === "registered"
        ) {
          clearTimeout(timer);
          ws.removeEventListener("message", onMessage);
          const r = parsed.value as {
            peerId: string;
            clusterId: number;
            isHead: boolean;
            peers: PeerInfo[];
            needsSync?: boolean;
          };
          resolve({
            type: "join-ack",
            peerId: r.peerId,
            clusterId: r.clusterId,
            isHead: r.isHead,
            peers: r.peers,
            needsSync: r.needsSync === true,
          });
        } else if (parsed.kind === "protocol") {
          this.pending.push(parsed.value);
        }
      };
      ws.addEventListener("message", onMessage);
      ws.send(
        JSON.stringify({
          type: "register",
          peerId: this.peerId,
          model: this.model,
        }),
      );
    });

    this.pending.unshift(registered);
    this.attachSteadyStateHandlers(ws);
    this.reconnectDelayMs = 1_000;
  }

  private attachSteadyStateHandlers(ws: WebSocket): void {
    ws.addEventListener("message", (event) => {
      const parsed = this.parseFrame(event.data);
      if (!parsed || parsed.kind !== "protocol") return;
      this.deliver(parsed.value);
    });
    ws.addEventListener("close", () => {
      if (this.closed) return;
      if (this.ws !== ws) return;
      this.ws = null;
      this.log("websocket closed unexpectedly; scheduling reconnect");
      this.scheduleReconnect();
    });
    ws.addEventListener("error", (e) => {
      const err = e as ErrorEvent;
      this.log(`websocket error: ${err.message ?? "unknown"}`);
    });
  }

  private scheduleReconnect(): void {
    if (this.closed || this.reconnecting) return;
    this.reconnecting = true;
    const delay = this.reconnectDelayMs;
    this.reconnectDelayMs = Math.min(
      this.reconnectDelayMaxMs,
      this.reconnectDelayMs * 2,
    );
    setTimeout(() => {
      if (this.closed) {
        this.reconnecting = false;
        return;
      }
      this.attemptReconnect().finally(() => {
        this.reconnecting = false;
      });
    }, delay);
  }

  private async attemptReconnect(): Promise<void> {
    this.log(`reconnect attempt (peerId=${this.peerId})`);
    const ws = new WebSocket(this.serverUrl);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    try {
      await new Promise<void>((resolve, reject) => {
        const onError = () => reject(new Error("reconnect open failed"));
        ws.addEventListener("error", onError, { once: true });
        ws.addEventListener("open", () => {
          ws.removeEventListener("error", onError);
          resolve();
        }, { once: true });
        setTimeout(
          () => reject(new Error("reconnect open timeout")),
          15_000,
        );
      });

      const reregistered = await new Promise<JoinAckMessage>(
        (resolve, reject) => {
          const timer = setTimeout(
            () => reject(new Error("reconnect registration timeout")),
            15_000,
          );
          const onMessage = (event: MessageEvent) => {
            const parsed = this.parseFrame(event.data);
            if (!parsed) return;
            if (
              parsed.kind === "text" &&
              (parsed.value as { type?: string }).type === "registered"
            ) {
              clearTimeout(timer);
              ws.removeEventListener("message", onMessage);
              const r = parsed.value as {
                peerId: string;
                clusterId: number;
                isHead: boolean;
                peers: PeerInfo[];
                needsSync?: boolean;
              };
              resolve({
                type: "join-ack",
                peerId: r.peerId,
                clusterId: r.clusterId,
                isHead: r.isHead,
                peers: r.peers,
                needsSync: r.needsSync === true,
              });
            } else if (parsed.kind === "protocol") {
              this.deliver(parsed.value);
            }
          };
          ws.addEventListener("message", onMessage);
          ws.send(
            JSON.stringify({
              type: "register",
              peerId: this.peerId,
              model: this.model,
            }),
          );
        },
      );

      const update: PeerListUpdateMessage = {
        type: "peer-list",
        peers: reregistered.peers,
        added: [],
        removed: [],
      };
      this.deliver(update);

      this.attachSteadyStateHandlers(ws);
      this.reconnectDelayMs = 1_000;
      this.log(`reconnect succeeded (cluster=${reregistered.clusterId}${reregistered.isHead ? "/head" : ""})`);
    } catch (e) {
      this.log(`reconnect failed: ${(e as Error).message}`);
      try {
        ws.close();
      } catch {}
      if (this.ws === ws) this.ws = null;
      this.scheduleReconnect();
    }
  }

  /**
   * Parse `event.data` — either string (text frame) or ArrayBuffer
   * (binary envelope frame). Browser WebSocket with binaryType
   * "arraybuffer" delivers ArrayBuffer for binary frames; setting
   * binaryType is critical (default is "blob" which would force an
   * async read).
   */
  private parseFrame(
    data: unknown,
  ):
    | { kind: "text"; value: unknown }
    | { kind: "protocol"; value: ProtocolMessage }
    | null {
    if (data instanceof ArrayBuffer) {
      const buf = data;
      if (buf.byteLength < FOUR) return null;
      const view = new DataView(buf);
      const envLen = view.getUint32(0, true);
      if (envLen <= 0 || envLen > buf.byteLength - FOUR) return null;
      let envelope: Envelope | null = null;
      try {
        envelope = JSON.parse(
          textDecoder.decode(new Uint8Array(buf, FOUR, envLen)),
        );
      } catch {
        return null;
      }
      if (!envelope) return null;
      const payload = new Uint8Array(buf, FOUR + envLen);
      const tensors = envelope.tensorShapes
        ? decodeTensors(payload, envelope.tensorShapes, envelope.wireDtype ?? "f32")
        : [];
      return {
        kind: "protocol",
        value: restoreTensorPayload(envelope.msg, tensors),
      };
    }

    if (typeof data !== "string") return null;
    let parsed: unknown;
    try {
      parsed = JSON.parse(data);
    } catch {
      return null;
    }
    if (!parsed || typeof parsed !== "object") return null;
    const obj = parsed as Record<string, unknown>;
    if (obj.type === "registered" || obj.type === "pong") {
      return { kind: "text", value: obj };
    }
    if (obj.type === "peer-list") {
      return { kind: "protocol", value: obj as PeerListUpdateMessage };
    }
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
      const wireDtype = defaultWireDtype();
      const envelope: Envelope = {
        from: this.peerId,
        target,
        msg: taken.stripped,
        tensorShapes: taken.tensors.map((a) => [a.length]),
        wireDtype,
      };
      const envBytes = textEncoder.encode(JSON.stringify(envelope));
      const payload = encodeTensors(taken.tensors, wireDtype);
      const out = new Uint8Array(FOUR + envBytes.byteLength + payload.byteLength);
      new DataView(out.buffer).setUint32(0, envBytes.byteLength, true);
      out.set(envBytes, FOUR);
      out.set(payload, FOUR + envBytes.byteLength);
      this.ws.send(out.buffer);
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
      const pending = this.pending;
      this.pending = [];
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
