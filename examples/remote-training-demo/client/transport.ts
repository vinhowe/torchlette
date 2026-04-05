/**
 * WebSocket RPC client. Browser and Node compatible (uses global WebSocket).
 *
 * One connection per client. Messages are newline-free JSON. Each RPC sends
 * a request with a unique id and awaits a response carrying the same id.
 */

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

type Resolver = (r: RpcResponse) => void;

export interface ConnectOptions {
  url: string;
  onLog?: (msg: string) => void;
}

export class RpcClient {
  private ws: WebSocket | null = null;
  private nextId = 1;
  private readonly pending = new Map<number, Resolver>();
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
      this.ws = ws;
      let helloed = false;
      ws.addEventListener("open", () => {
        this.log(`[rpc] connected to ${this.opts.url}`);
      });
      ws.addEventListener("message", (ev) => {
        const msg = JSON.parse(ev.data as string) as
          | RpcResponse
          | { id: 0; result: HelloResult };
        // Hello is id=0 with a result.
        if (!helloed && msg.id === 0 && "result" in msg) {
          this._sessionId = (msg.result as HelloResult).sessionId;
          this.log(`[rpc] session ${this._sessionId}`);
          helloed = true;
          resolve();
          return;
        }
        const resolver = this.pending.get(msg.id);
        if (resolver) {
          this.pending.delete(msg.id);
          resolver(msg as RpcResponse);
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

  async rpc<T>(method: string, params: unknown): Promise<T> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("Not connected");
    }
    const id = this.nextId++;
    const req = { id, method, params };
    const p = new Promise<RpcResponse>((resolve) => {
      this.pending.set(id, resolve);
    });
    this.ws.send(JSON.stringify(req));
    const resp = await p;
    if ("error" in resp) {
      throw new Error(`[${method}] ${resp.error.message}`);
    }
    return resp.result as T;
  }

  // Typed convenience wrappers

  execute(params: ExecuteParams): Promise<ExecuteResult> {
    return this.rpc<ExecuteResult>("execute", params);
  }
  upload(params: UploadParams): Promise<UploadResult> {
    return this.rpc<UploadResult>("upload", params);
  }
  download(params: DownloadParams): Promise<DownloadResult> {
    return this.rpc<DownloadResult>("download", params);
  }
  readScalar(params: ReadScalarParams): Promise<ReadScalarResult> {
    return this.rpc<ReadScalarResult>("readScalar", params);
  }
  release(params: ReleaseParams): Promise<ReleaseResult> {
    return this.rpc<ReleaseResult>("release", params);
  }
}
