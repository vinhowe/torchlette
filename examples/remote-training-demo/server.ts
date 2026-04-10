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
import type { Backend, DType } from "../../src/backend/types.js";
import { executePlanOptimized } from "../../src/executor/executor.js";
import { createStorageHandle } from "../../src/graph/node-factory.js";
import type { StorageHandle } from "../../src/graph/types.js";
import {
  decodeBinaryFrame,
  encodeBinaryFrame,
  valuesToTypedArray,
} from "../../src/remote/binary-frame.js";
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

/** Active compute backend — set during startup by initBackend(). */
let backend: Backend = cpuBackend;
let gpuMemoryStatsFn: (() => { currentBytes: number; peakBytes: number }) | null = null;
registerBackend(cpuBackend);

async function initBackend(preferWebGPU: boolean): Promise<string> {
  if (!preferWebGPU) return "cpu";
  try {
    const { initWebGPU, webgpuBackend } = await import(
      "../../src/backend/webgpu/index.js"
    );
    const ok = await initWebGPU();
    if (!ok) {
      console.log("[server] WebGPU init failed, falling back to CPU");
      return "cpu";
    }
    registerBackend(webgpuBackend);
    backend = webgpuBackend;
    const { getGPUMemoryStats } = await import("../../src/backend/webgpu/memory-tracker.js");
    gpuMemoryStatsFn = getGPUMemoryStats;
    return "webgpu";
  } catch (e) {
    console.log(`[server] WebGPU not available (${(e as Error).message}), using CPU`);
    return "cpu";
  }
}

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
      const storage = this.handles.get(h);
      if (storage) {
        this.handles.delete(h);
        // Destroy the GPU buffer to reclaim VRAM. Without this, every
        // released handle leaks its buffer and the GPU fills up.
        storage.backendTensor.destroy?.();
        n++;
      }
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
  const t0 = performance.now();
  const plan = deserializePlan(params.plan, {
    resolveHandle: (h) => session.resolve(h),
  });
  const tDeser = performance.now();
  // Rewrite device labels — client uses "cpu", server uses its backend.
  const serverDevice = backend.name as "cpu" | "webgpu";
  for (const node of plan.nodes) {
    node.device = serverDevice;
  }

  // Optimized execution: fusion + arena + compiled plan replay.
  // The plan carries outputIndices (set by the client) so the executor
  // scopes its all-results pass to only client-needed nodes.
  await executePlanOptimized(plan, backend);
  const tExec = performance.now();

  // Allocate handles only for output nodes (plan.outputIndices).
  // Non-output nodes are internal to the plan — their GPU buffers are
  // managed by the arena and reused across steps.
  const outputSet = plan.outputIndices ? new Set(plan.outputIndices) : null;
  const outputs: Record<NodeIdx, HandleRef> = {};
  const sideOutputs: Record<string, HandleRef> = {};

  for (let i = 0; i < plan.nodes.length; i++) {
    if (outputSet && !outputSet.has(i)) continue;
    const node = plan.nodes[i];
    if (!node.result) continue;
    outputs[i] = session.allocHandle(node.result);
    if (node.results) {
      for (let j = 0; j < node.results.length; j++) {
        const r = node.results[j];
        if (r) sideOutputs[`${i}:${j}`] = session.allocHandle(r);
      }
    }
  }
  const tMarshal = performance.now();
  const nSide = Object.keys(sideOutputs).length;
  if (plan.nodes.length > 100) {
    console.log(`[exec] ${plan.nodes.length} nodes: deser=${(tDeser-t0).toFixed(0)}ms exec=${(tExec-tDeser).toFixed(0)}ms marshal=${(tMarshal-tExec).toFixed(0)}ms total=${(tMarshal-t0).toFixed(0)}ms handles=${Object.keys(outputs).length}+${nSide} side`);
  }
  return nSide > 0 ? { outputs, sideOutputs } : { outputs };
}

function uploadRaw(
  session: Session,
  values: number[] | Float32Array | Int32Array | Uint32Array | Uint8Array,
  shape: number[],
): UploadResult {
  const nums = Array.isArray(values) ? values : Array.from(values);
  const backendTensor = backend.ops.tensorFromArray(nums, shape);
  const storage = createStorageHandle(backend.name as "cpu" | "webgpu", backendTensor);
  return { handle: session.allocHandle(storage) };
}

function uploadHandler(session: Session, params: UploadParams): UploadResult {
  return uploadRaw(session, params.values, params.shape);
}

async function downloadHandler(
  session: Session,
  params: DownloadParams,
): Promise<DownloadResult> {
  const storage = session.resolve(params.handle);
  const values = await backend.ops.read(storage.backendTensor);
  return { values };
}

/** Read a handle's data as a binary frame for the wire. */
async function downloadBinary(
  session: Session,
  id: number,
  handle: string,
): Promise<ArrayBuffer> {
  const storage = session.resolve(handle);
  const values = await backend.ops.read(storage.backendTensor);
  // All current backends read as number[]; dtype comes from the tensor.
  const dtype = ((storage.backendTensor as { dtype?: DType }).dtype ??
    "f32") as DType;
  const typed = valuesToTypedArray(values, dtype);
  const shape = (storage.backendTensor as { shape: number[] }).shape;
  return encodeBinaryFrame({ id, dtype, shape, values: typed });
}

async function readScalarHandler(
  session: Session,
  params: ReadScalarParams,
): Promise<ReadScalarResult> {
  const storage = session.resolve(params.handle);
  const values = await backend.ops.read(storage.backendTensor);
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

function statsHandler(session: Session): Record<string, unknown> {
  let gpu: Record<string, number> | null = null;
  if (gpuMemoryStatsFn) {
    const mem = gpuMemoryStatsFn();
    gpu = {
      currentBytes: mem.currentBytes,
      peakBytes: mem.peakBytes,
      currentMB: Math.round(mem.currentBytes / (1024 * 1024)),
      peakMB: Math.round(mem.peakBytes / (1024 * 1024)),
    };
  }
  return { handles: session.handleCount(), gpu };
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
      case "stats":
        result = statsHandler(session);
        break;
      default:
        throw new Error(`Unknown method: ${req.method}`);
    }
    return { id: req.id, result };
  } catch (err) {
    const e = err as Error;
    console.error(`[rpc] ${req.method} failed: ${e.message}`);
    if (e.stack) console.error(e.stack.split("\n").slice(0, 5).join("\n"));
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

  ws.on("message", async (data, isBinary) => {
    if (isBinary) {
      // Binary frame = upload.
      const bytes = data as Buffer;
      const buf = bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      );
      try {
        const frame = decodeBinaryFrame(buf);
        const result = uploadRaw(session, frame.values, frame.shape);
        ws.send(JSON.stringify({ id: frame.id, result }));
      } catch (err) {
        const e = err as Error;
        console.error("[ws] binary upload failed:", e.message);
        // No id to reply to; log and move on.
      }
      return;
    }
    // Text frame.
    let req: RpcRequest;
    try {
      req = JSON.parse(data.toString()) as RpcRequest;
    } catch (e) {
      console.error("[ws] bad JSON:", e);
      return;
    }
    // Download: respond with a binary frame carrying the bytes.
    if (req.method === "download") {
      try {
        const params = req.params as { handle: string };
        const frame = await downloadBinary(session, req.id, params.handle);
        ws.send(frame, { binary: true });
      } catch (err) {
        const e = err as Error;
        console.error(`[rpc] download failed: ${e.message}`);
        ws.send(
          JSON.stringify({
            id: req.id,
            error: { message: e.message, stack: e.stack },
          }),
        );
      }
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

function parseArgs(): { port: number; cpu: boolean } {
  let port = 9876;
  let cpu = false;
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--port") port = Number(args[++i]);
    else if (args[i] === "--cpu") cpu = true;
  }
  return { port, cpu };
}

async function main(): Promise<void> {
  const { port, cpu } = parseArgs();
  const backendName = await initBackend(!cpu);
  console.log(`[server] backend: ${backendName}`);

  const server = createServer(handleHttp);
  const wss = new WebSocketServer({
    server,
    path: "/ws",
    perMessageDeflate: true, // Compress JSON plan payloads (~60-80% smaller)
  });
  wss.on("connection", (ws) => attachWebSocketHandlers(ws));

  server.listen(port, () => {
    console.log(`[server] http://localhost:${port}/ (WebSocket at /ws)`);
    console.log(`[server] static root: ${STATIC_ROOT}`);
  });

  // Dawn (used by WebGPU backend on Node) holds background threads that
  // prevent clean exit — force termination on SIGTERM/SIGINT.
  const shutdown = (sig: string) => {
    console.log(`[server] ${sig} — shutting down`);
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 500).unref();
  };
  process.on("SIGTERM", () => shutdown("SIGTERM"));
  process.on("SIGINT", () => shutdown("SIGINT"));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
