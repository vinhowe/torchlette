/**
 * WebSocket RPC client. Browser and Node compatible (uses global WebSocket).
 *
 * One connection per client. Control messages are JSON text frames; bulk
 * tensor data (upload, download-response) rides as binary frames carrying
 * [id(4)][dtype(1)][rank(1)][pad(2)][shape[rank]*4][payload].
 */

import {
  type BinaryFrame,
  decodeBinaryFrame,
  encodeBinaryFrame,
  valuesToTypedArray,
} from "../../../src/remote/binary-frame.ts";
import type { Transport } from "../../../src/remote/client-engine.ts";
import type {
  DownloadParams,
  DownloadResult,
  ExecuteParams,
  ExecuteResult,
  HelloResult,
  ReadScalarParams,
  ReadScalarResult,
  ReleaseParams,
  ReleaseResult,
  RpcResponse,
  UploadParams,
  UploadResult,
} from "../../../src/remote/rpc.ts";

type TextResolver = (r: RpcResponse) => void;
type BinaryResolver = (f: BinaryFrame) => void;

export interface ConnectOptions {
  url: string;
  onLog?: (msg: string) => void;
}

export class RpcClient implements Transport {
  private ws: WebSocket | null = null;
  private nextId = 1;
  private readonly pendingText = new Map<number, TextResolver>();
  private readonly pendingBinary = new Map<number, BinaryResolver>();
  private readonly log: (msg: string) => void;
  private _sessionId = "";
  private _onClose: (() => void) | null = null;

  constructor(private readonly opts: ConnectOptions) {
    this.log = opts.onLog ?? (() => {});
  }

  get sessionId(): string {
    return this._sessionId;
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.opts.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      let helloed = false;
      ws.addEventListener("open", () => {
        this.log(`[rpc] connected to ${this.opts.url}`);
      });
      ws.addEventListener("message", (ev) => {
        if (typeof ev.data === "string") {
          // Text frame: JSON control message.
          const msg = JSON.parse(ev.data) as
            | RpcResponse
            | { id: 0; result: HelloResult };
          if (!helloed && msg.id === 0 && "result" in msg) {
            this._sessionId = (msg.result as HelloResult).sessionId;
            this.log(`[rpc] session ${this._sessionId}`);
            helloed = true;
            resolve();
            return;
          }
          const textResolver = this.pendingText.get(msg.id);
          if (textResolver) {
            this.pendingText.delete(msg.id);
            textResolver(msg as RpcResponse);
            return;
          }
          // Text error for a pending-binary request
          const binaryResolver = this.pendingBinary.get(msg.id);
          if (binaryResolver && "error" in msg) {
            this.pendingBinary.delete(msg.id);
            throw new Error(`[rpc] ${msg.error.message}`);
          }
          return;
        }
        // Binary frame: download response or similar.
        const buffer =
          ev.data instanceof ArrayBuffer
            ? ev.data
            : (ev.data as Uint8Array).buffer;
        const frame = decodeBinaryFrame(buffer);
        const resolver = this.pendingBinary.get(frame.id);
        if (resolver) {
          this.pendingBinary.delete(frame.id);
          resolver(frame);
        }
      });
      ws.addEventListener("error", (e) => {
        this.log(`[rpc] error: ${String(e)}`);
        if (!helloed) reject(new Error("WebSocket error"));
      });
      ws.addEventListener("close", () => {
        this.log(`[rpc] closed`);
        this._onClose?.();
      });
    });
  }

  onClose(cb: () => void): void {
    this._onClose = cb;
  }

  close(): void {
    this.ws?.close();
  }

  /** Send a JSON RPC, await a JSON response. */
  private async rpcText<T>(method: string, params: unknown): Promise<T> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      throw new Error("Not connected");
    }
    const id = this.nextId++;
    const p = new Promise<RpcResponse>((resolve) => {
      this.pendingText.set(id, resolve);
    });
    ws.send(JSON.stringify({ id, method, params }));
    const resp = await p;
    if ("error" in resp) throw new Error(`[${method}] ${resp.error.message}`);
    return resp.result as T;
  }

  execute(params: ExecuteParams): Promise<ExecuteResult> {
    return this.rpcText<ExecuteResult>("execute", params);
  }
  readScalar(params: ReadScalarParams): Promise<ReadScalarResult> {
    return this.rpcText<ReadScalarResult>("readScalar", params);
  }
  release(params: ReleaseParams): Promise<ReleaseResult> {
    return this.rpcText<ReleaseResult>("release", params);
  }

  /** Upload: single binary frame from client, text response from server. */
  async upload(params: UploadParams): Promise<UploadResult> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      throw new Error("Not connected");
    }
    const id = this.nextId++;
    const values = valuesToTypedArray(params.values, params.dtype);
    const frame = encodeBinaryFrame({
      id,
      dtype: params.dtype,
      shape: params.shape,
      values,
    });
    const p = new Promise<RpcResponse>((resolve) => {
      this.pendingText.set(id, resolve);
    });
    ws.send(frame);
    const resp = await p;
    if ("error" in resp) throw new Error(`[upload] ${resp.error.message}`);
    return resp.result as UploadResult;
  }

  /** Download: text request from client, binary frame response from server. */
  async download(params: DownloadParams): Promise<DownloadResult> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      throw new Error("Not connected");
    }
    const id = this.nextId++;
    const p = new Promise<BinaryFrame>((resolve) => {
      this.pendingBinary.set(id, resolve);
    });
    ws.send(JSON.stringify({ id, method: "download", params }));
    const frame = await p;
    // The Transport interface returns number[] for back-compat with the
    // client engine's runtime.cpu() contract. One allocation at the boundary.
    return { values: Array.from(frame.values) };
  }
}
