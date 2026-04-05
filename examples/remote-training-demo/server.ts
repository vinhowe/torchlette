/**
 * Remote-training demo server.
 *
 * Serves the demo HTML/JS as static files and accepts WebSocket connections
 * that speak the remote-training RPC protocol. Each connection is its own
 * session: independent handle registry, independent lifecycle.
 *
 * Run: npx tsx examples/remote-training-demo/server.ts [--port 9876]
 */

import { randomUUID } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { extname, resolve } from "node:path";
import { WebSocketServer, type WebSocket } from "ws";

import { cpuBackend } from "../../src/backend/cpu/index.js";
import { registerBackend } from "../../src/backend/registry.js";
import { executePlanSequential } from "../../src/executor/sequential.js";
import { createStorageHandle, wrapResultAsStorage } from "../../src/graph/node-factory.js";
import { storageTracker } from "../../src/graph/storage-tracker.js";
import type { StorageHandle } from "../../src/graph/types.js";
import { deserializePlan } from "../../src/remote/serialize.js";
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
  RpcError,
  RpcRequest,
  RpcResponse,
  UploadParams,
  UploadResult,
} from "../../src/remote/rpc.js";
import type { HandleRef, NodeIdx } from "../../src/remote/wire.js";

registerBackend(cpuBackend);

// ============================================================================
// Session state
// ============================================================================

class Session {
  readonly id: string = randomUUID();
  /** handle → StorageHandle (server-side) */
  private readonly handles = new Map<HandleRef, StorageHandle>();
  private nextHandle = 1;

  allocHandle(storage: StorageHandle): HandleRef {
    const h = `h${this.nextHandle++}`;
    this.handles.set(h, storage);
    return h;
  }

  resolve(handle: HandleRef): StorageHandle {
    const s = this.handles.get(handle);
    if (!s) throw new Error(`Unknown handle: ${handle}`);
    return s;
  }

  release(handles: HandleRef[]): number {
    let n = 0;
    for (const h of handles) {
      if (this.handles.delete(h)) n++;
    }
    return n;
  }

  handleCount(): number {
    return this.handles.size;
  }
}

// ============================================================================
// RPC handlers (all async)
// ============================================================================

async function executeHandler(
  session: Session,
  params: ExecuteParams,
): Promise<ExecuteResult> {
  const plan = deserializePlan(params.plan, {
    resolveHandle: (h) => session.resolve(h),
  });
  await executePlanSequential(plan, cpuBackend);

  const outputs: Record<NodeIdx, HandleRef> = {};
  const outputSet = params.plan.outputNodes
    ? new Set(params.plan.outputNodes)
    : null;

  for (let i = 0; i < plan.nodes.length; i++) {
    const node = plan.nodes[i];
    if (!node.result) continue;
    if (outputSet === null || outputSet.has(i)) {
      outputs[i] = session.allocHandle(node.result);
    }
  }
  return { outputs };
}

function uploadHandler(session: Session, params: UploadParams): UploadResult {
  const backendTensor = cpuBackend.ops.tensorFromArray(params.values, params.shape);
  const storage = createStorageHandle("cpu", backendTensor);
  return { handle: session.allocHandle(storage) };
}

async function downloadHandler(
  session: Session,
  params: DownloadParams,
): Promise<DownloadResult> {
  const storage = session.resolve(params.handle);
  const values = await cpuBackend.ops.read(storage.backendTensor);
  return { values };
}

async function readScalarHandler(
  session: Session,
  params: ReadScalarParams,
): Promise<ReadScalarResult> {
  const storage = session.resolve(params.handle);
  const values = await cpuBackend.ops.read(storage.backendTensor);
  if (values.length !== 1) {
    throw new Error(
      `readScalar: tensor has ${values.length} elements, expected 1`,
    );
  }
  return { value: values[0] };
}

function releaseHandler(session: Session, params: ReleaseParams): ReleaseResult {
  return { releasedCount: session.release(params.handles) };
}

async function dispatch(
  session: Session,
  req: RpcRequest,
): Promise<RpcResponse> {
  try {
    let result: unknown;
    switch (req.method) {
      case "execute":
        result = await executeHandler(session, req.params as ExecuteParams);
        break;
      case "upload":
        result = uploadHandler(session, req.params as UploadParams);
        break;
      case "download":
        result = await downloadHandler(session, req.params as DownloadParams);
        break;
      case "readScalar":
        result = await readScalarHandler(session, req.params as ReadScalarParams);
        break;
      case "release":
        result = releaseHandler(session, req.params as ReleaseParams);
        break;
      default:
        throw new Error(`Unknown method: ${req.method}`);
    }
    return { id: req.id, result };
  } catch (err) {
    const e = err as Error;
    return {
      id: req.id,
      error: { message: e.message, stack: e.stack },
    } satisfies RpcError;
  }
}

// ============================================================================
// WebSocket server
// ============================================================================

function attachWebSocketHandlers(ws: WebSocket): void {
  const session = new Session();
  console.log(`[ws] session opened: ${session.id}`);

  const hello: { id: number; result: HelloResult } = {
    id: 0,
    result: { sessionId: session.id, protocolVersion: 1 },
  };
  ws.send(JSON.stringify(hello));

  ws.on("message", async (data) => {
    let req: RpcRequest;
    try {
      req = JSON.parse(data.toString()) as RpcRequest;
    } catch (e) {
      console.error("[ws] bad JSON:", e);
      return;
    }
    const response = await dispatch(session, req);
    ws.send(JSON.stringify(response));
  });

  ws.on("close", () => {
    console.log(
      `[ws] session closed: ${session.id} (${session.handleCount()} handles leaked)`,
    );
    // Session dies with the socket; handles become unreachable and GC'd
    // eventually via storageTracker's reachability sweep.
  });

  ws.on("error", (e) => console.error(`[ws] error on ${session.id}:`, e));
}

// ============================================================================
// HTTP static file server
// ============================================================================

const CONTENT_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

const STATIC_ROOT = resolve(
  new URL("./client", import.meta.url).pathname,
);

function handleHttp(req: IncomingMessage, res: ServerResponse): void {
  const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
  let pathname = url.pathname;
  if (pathname === "/") pathname = "/index.html";

  const filePath = resolve(STATIC_ROOT, `.${pathname}`);
  if (!filePath.startsWith(STATIC_ROOT)) {
    res.writeHead(403).end("Forbidden");
    return;
  }

  try {
    const st = statSync(filePath);
    if (!st.isFile()) {
      res.writeHead(404).end("Not found");
      return;
    }
    const body = readFileSync(filePath);
    const type = CONTENT_TYPES[extname(filePath)] ?? "application/octet-stream";
    res.writeHead(200, { "Content-Type": type }).end(body);
  } catch {
    res.writeHead(404).end("Not found");
  }
}

// ============================================================================
// Entry point
// ============================================================================

function parseArgs(): { port: number } {
  let port = 9876;
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--port") port = Number(args[++i]);
  }
  return { port };
}

function main(): void {
  const { port } = parseArgs();

  const server = createServer(handleHttp);
  const wss = new WebSocketServer({ server, path: "/ws" });
  wss.on("connection", (ws) => attachWebSocketHandlers(ws));

  server.listen(port, () => {
    console.log(`[server] http://localhost:${port}/ (WebSocket at /ws)`);
    console.log(`[server] static root: ${STATIC_ROOT}`);
  });
}

main();
