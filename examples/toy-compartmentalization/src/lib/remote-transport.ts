/**
 * WebSocket RPC client for remote training. Thin wrapper copied from the
 * remote-training-demo, re-pathed to use the vite alias imports.
 */

import {
  type BinaryFrame,
  decodeBinaryFrame,
  encodeBinaryFrame,
  valuesToTypedArray,
} from "../../../../src/remote/binary-frame";
import type { Transport } from "../../../../src/remote/client-engine";
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
} from "../../../../src/remote/rpc";

type TextResolver = (r: RpcResponse) => void;
type BinaryResolver = (f: BinaryFrame) => void;

export class RpcClient implements Transport {
  private ws: WebSocket | null = null;
  private nextId = 1;
  private readonly pendingText = new Map<number, TextResolver>();
  private readonly pendingBinary = new Map<number, BinaryResolver>();
  private readonly log: (msg: string) => void;
  private _sessionId = "";
  private _onClose: (() => void) | null = null;

  constructor(private readonly url: string, onLog?: (msg: string) => void) {
    this.log = onLog ?? (() => {});
  }

  get sessionId(): string { return this._sessionId; }
  get connected(): boolean { return this.ws?.readyState === WebSocket.OPEN; }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      let helloed = false;
      ws.addEventListener("open", () => { this.log(`[rpc] connected to ${this.url}`); });
      ws.addEventListener("message", (ev) => {
        if (typeof ev.data === "string") {
          const msg = JSON.parse(ev.data) as RpcResponse | { id: 0; result: HelloResult };
          if (!helloed && msg.id === 0 && "result" in msg) {
            this._sessionId = (msg.result as HelloResult).sessionId;
            this.log(`[rpc] session ${this._sessionId}`);
            helloed = true;
            resolve();
            return;
          }
          const textResolver = this.pendingText.get(msg.id);
          if (textResolver) { this.pendingText.delete(msg.id); textResolver(msg as RpcResponse); return; }
          const binaryResolver = this.pendingBinary.get(msg.id);
          if (binaryResolver && "error" in msg) {
            this.pendingBinary.delete(msg.id);
            throw new Error(`[rpc] ${(msg as any).error.message}`);
          }
          return;
        }
        const buffer = ev.data instanceof ArrayBuffer ? ev.data : (ev.data as Uint8Array).buffer;
        const frame = decodeBinaryFrame(buffer);
        const resolver = this.pendingBinary.get(frame.id);
        if (resolver) { this.pendingBinary.delete(frame.id); resolver(frame); }
      });
      ws.addEventListener("error", (e) => {
        this.log(`[rpc] error: ${String(e)}`);
        if (!helloed) reject(new Error("WebSocket error"));
      });
      ws.addEventListener("close", () => { this.log(`[rpc] closed`); this._onClose?.(); });
    });
  }

  onClose(cb: () => void): void { this._onClose = cb; }
  close(): void { this.ws?.close(); }

  private async rpcText<T>(method: string, params: unknown): Promise<T> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) throw new Error("Not connected");
    const id = this.nextId++;
    const p = new Promise<RpcResponse>((resolve) => { this.pendingText.set(id, resolve); });
    ws.send(JSON.stringify({ id, method, params }));
    const resp = await p;
    if ("error" in resp) throw new Error(`[${method}] ${(resp as any).error.message}`);
    return resp.result as T;
  }

  execute(params: ExecuteParams): Promise<ExecuteResult> { return this.rpcText<ExecuteResult>("execute", params); }
  readScalar(params: ReadScalarParams): Promise<ReadScalarResult> { return this.rpcText<ReadScalarResult>("readScalar", params); }
  release(params: ReleaseParams): Promise<ReleaseResult> { return this.rpcText<ReleaseResult>("release", params); }

  async upload(params: UploadParams): Promise<UploadResult> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) throw new Error("Not connected");
    const id = this.nextId++;
    const values = valuesToTypedArray(params.values, params.dtype);
    const frame = encodeBinaryFrame({ id, dtype: params.dtype, shape: params.shape, values });
    const p = new Promise<RpcResponse>((resolve) => { this.pendingText.set(id, resolve); });
    ws.send(frame);
    const resp = await p;
    if ("error" in resp) throw new Error(`[upload] ${(resp as any).error.message}`);
    return resp.result as UploadResult;
  }

  /** Non-standard: query server GPU memory stats. */
  async stats(): Promise<{ handles: number; gpu: { currentMB: number; peakMB: number } | null }> {
    return this.rpcText('stats', {});
  }

  async download(params: DownloadParams): Promise<DownloadResult> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) throw new Error("Not connected");
    const id = this.nextId++;
    const p = new Promise<BinaryFrame>((resolve) => { this.pendingBinary.set(id, resolve); });
    ws.send(JSON.stringify({ id, method: "download", params }));
    const frame = await p;
    return { values: Array.from(frame.values) };
  }
}
