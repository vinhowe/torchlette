/**
 * Remote-training demo server.
 *
 * Serves the demo HTML/JS as static files and accepts WebSocket connections
 * that speak the remote-training RPC protocol. Each connection is its own
 * session: independent handle registry, independent lifecycle.
 *
 * Run: npx tsx examples/remote-training-demo/server.ts [--port 9882]
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
/**
 * Recycle process-global executor caches (arenas, bind groups, pool buffers,
 * f16 weight cache) — invoked when the last active session disconnects so a
 * fresh client doesn't reuse a server that has accumulated GPU resources from
 * many prior sessions. Set during initBackend() once WebGPU is up; null on CPU.
 */
let recycleExecutorState: (() => void) | null = null;
/** Pool stats accessor — set after WebGPU init so statsHandler can report. */
let getBufferPoolStats: (() => Record<string, number>) | null = null;
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

    // Resolve all the reset functions once so the close-handler can run them
    // synchronously without doing dynamic imports on every disconnect.
    //
    // The first version of this recycle missed several leak sources and let
    // GPU memory creep up across sessions (~20 MiB/session); each new
    // session ran 3-4× slower than the first because Dawn was operating
    // under increasing memory pressure. The fixed version covers all the
    // process-global GPU buffer caches:
    //
    //   1. evictAllArenas(force=true) — destroy ALL arena GPUBuffers,
    //      ignoring the in-session "live tensor" safety check (which
    //      uses stale state between sessions and silently leaks).
    //   2. clearBindGroupCache — drop bind group + sequence cache state.
    //   3. drainParamsBufferPools — destroy uniform buffers pooled by
    //      kernel dispatch (createParamsBuffer in bind-group-cache.ts).
    //      The clearBindGroupCache call alone leaves these alive.
    //   4. f16WeightCache: iterate values + destroy each, THEN clear the
    //      Map. Just calling .clear() drops the Map entries but leaks the
    //      f16 GPUBuffers.
    //   5. runTeardownCallbacks — drains per-kernel module-local caches
    //      registered via onTeardown (adam-kernel, attention-kernel,
    //      cross-entropy-kernel, fusion-dispatch, layernorm-kernel,
    //      rmsnorm-kernel, rope-kernel, row-program-dispatch,
    //      unscale-kernel, matmul/dispatch). Each holds its own
    //      Map<*, GPUBuffer> of config buffers that would otherwise
    //      accumulate.
    //   6. flush + destroy + evict the buffer pool itself.
    const { evictAllArenas } = await import("../../src/executor/executor.js");
    const { clearBindGroupCache } = await import(
      "../../src/backend/webgpu/bind-group-cache.js"
    );
    const { flushBufferPool, destroyPendingGPUBuffers, evictAllPoolBuffers } =
      await import("../../src/backend/webgpu/buffer-pool.js");
    const { f16WeightCache } = await import(
      "../../src/backend/webgpu/gpu-context.js"
    );
    const { drainParamsBufferPools, runTeardownCallbacks } = await import(
      "../../src/backend/webgpu/webgpu-state.js"
    );
    const { bufferPool } = await import("../../src/backend/webgpu/buffer-pool.js");
    getBufferPoolStats = (): Record<string, number> => {
      const s = bufferPool.stats();
      return {
        pooledBuffers: s.pooledBuffers,
        pooledMB: Math.round(s.pooledBytes / (1024 * 1024)),
        pendingRelease: s.pendingRelease,
        pendingReleaseMB: Math.round(s.pendingReleaseBytes / (1024 * 1024)),
        pendingDestroy: s.pendingDestroy,
        reuseCount: s.reuseCount,
        allocCount: s.allocCount,
        liveCount: s.liveCount,
      };
    };
    recycleExecutorState = (): void => {
      // 1. Arena buffers (force=true: ignore stale liveness state).
      evictAllArenas(true);
      // 2. Bind-group / params-sequence cache references.
      clearBindGroupCache();
      // 3. Pooled uniform buffers — must explicitly destroy each.
      drainParamsBufferPools();
      // 4. f16 dual-write cache: destroy each buffer, then clear map.
      for (const buf of f16WeightCache.values()) {
        try { buf.destroy(); } catch { /* already destroyed */ }
      }
      f16WeightCache.clear();
      // 5. Per-kernel module-local config caches.
      runTeardownCallbacks();
      // 6. The pool itself.
      flushBufferPool();
      destroyPendingGPUBuffers();
      evictAllPoolBuffers();
    };

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

  /**
   * Destroy every backend tensor still held by this session and clear the
   * handle map. Call from `ws.on('close')` so a disconnecting client doesn't
   * leak its working set on the GPU. Without this every dropped session
   * permanently consumes VRAM until the server process restarts.
   */
  releaseAll(): number {
    let n = 0;
    for (const storage of this.handles.values()) {
      storage.backendTensor.destroy?.();
      n++;
    }
    this.handles.clear();
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
  // Process piggybacked releases BEFORE deserializing the plan. The plan
  // can't reference any handle in `releases` — markStep computes the
  // release set from handles NOT in the live lazy graph, and the next
  // execute is built from that same lazy graph.
  if (params.releases && params.releases.length > 0) {
    session.release(params.releases);
  }
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
    // Side outputs start at j=1: results[0] is the same StorageHandle as
    // node.result and is already allocated above as outputs[i]. After the
    // LazyIRNode class refactor that made `result` a derived view of
    // `results[0]`, every node has results.length >= 1, so iterating from
    // j=0 here would allocate a duplicate handle for every single-output
    // node — doubling the per-step bookkeeping cost.
    if (node.results && node.results.length > 1) {
      for (let j = 1; j < node.results.length; j++) {
        const r = node.results[j];
        if (r) sideOutputs[`${i}:${j}`] = session.allocHandle(r);
      }
    }
  }

  // Destroy intermediate node.result storages — those NOT in outputSet and
  // not bound to any session handle. These are step-scoped temporaries that
  // the server has no further use for. Without this cleanup the executor's
  // storageTracker accumulates ~150 storages per training step, which keeps
  // their underlying buffers' liveCount > 0 and pins them in the buffer
  // pool's pendingRelease queue forever — the actual root cause of the
  // batch=1024 OOM and the steady cross-step memory growth observed in
  // both Node and the browser.
  //
  // Arena buffers short-circuit inside backendTensor.destroy() (checked via
  // arenaBufferSet) so they stay in the arena and get reused next step;
  // non-arena buffers go through the pool release chain.
  for (let i = 0; i < plan.nodes.length; i++) {
    if (outputSet?.has(i)) continue;
    const node = plan.nodes[i];
    if (node.result?.backendTensor?.destroy) {
      node.result.backendTensor.destroy();
    }
    if (node.results && node.results.length > 1) {
      for (let j = 1; j < node.results.length; j++) {
        const r = node.results[j];
        if (r?.backendTensor?.destroy) r.backendTensor.destroy();
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
  const v = values[0];
  if (!Number.isFinite(v)) {
    console.log(
      `[readScalar] non-finite value=${v} handle=${params.handle} session=${session.id.slice(0, 8)}`,
    );
  }
  return { value: v };
}

function releaseHandler(session: Session, params: ReleaseParams): ReleaseResult {
  const released = session.release(params.handles);
  // Log so we can verify the client is actually releasing during training.
  // (Logged unconditionally because release calls are infrequent — once per
  //  markStep — but the count tells us whether the loop is healthy.)
  console.log(
    `[release] session=${session.id.slice(0, 8)} released=${released}/${params.handles.length} held=${session.handleCount()}`,
  );
  return { releasedCount: released };
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
  // Pool / arena stats so the client can see what the tracker undercounts
  // (gpuMemoryTracker doesn't include pool-held bytes; pool stats let
  // a debug client compute real physical-usage = tracker + pool).
  const pool = getBufferPoolStats ? getBufferPoolStats() : null;
  return { handles: session.handleCount(), gpu, pool };
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

/**
 * Number of currently-connected WebSocket sessions. The executor state recycle
 * (arenas, bind groups, pool buffers) only runs when this drops to 0 — running
 * it while another session is mid-plan would invalidate the buffers it's
 * dispatching against. Single-tenant by intent for the demo, but the gate
 * keeps things sane if a second tab connects briefly.
 */
let activeSessionCount = 0;

function attachWebSocketHandlers(ws: WebSocket): void {
  const session = new Session();
  activeSessionCount++;
  console.log(`[ws] session opened: ${session.id} (active=${activeSessionCount})`);

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
    // Destroy every backend tensor before the session goes out of scope.
    // The JS handle map is GC'd by Node, but Dawn-owned GPU buffers are
    // not — they leak permanently unless we destroy them explicitly here.
    const heldBefore = session.handleCount();
    const destroyed = session.releaseAll();
    activeSessionCount--;
    console.log(
      `[ws] session closed: ${session.id} (held=${heldBefore} destroyed=${destroyed} active=${activeSessionCount})`,
    );

    // When no clients remain, recycle process-global executor caches that
    // would otherwise grow unbounded across sessions: per-template arenas
    // (compiled plans + their GPU buffers), the bind group cache, the f16
    // weight cache, and pooled GPUBuffers waiting for fence completion.
    // Without this, a long-running server eventually exhausts GPU memory
    // and starts serving Vulkan validation errors instead of plans.
    if (activeSessionCount === 0 && recycleExecutorState) {
      const beforeMB = gpuMemoryStatsFn
        ? Math.round(gpuMemoryStatsFn().currentBytes / (1024 * 1024))
        : null;
      recycleExecutorState();
      const afterMB = gpuMemoryStatsFn
        ? Math.round(gpuMemoryStatsFn().currentBytes / (1024 * 1024))
        : null;
      const memNote =
        beforeMB !== null && afterMB !== null
          ? ` gpuMB=${beforeMB}→${afterMB}`
          : "";
      console.log(`[ws] executor caches recycled${memNote}`);
    }
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
  let port = 9882;
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
  // prevent clean exit — force termination on SIGTERM/SIGINT, but FIRST
  // explicitly destroy the WebGPU device so the Vulkan device handle
  // (and its ~few hundred MiB of GPU resident memory) doesn't leak.
  // Without this, every SIGKILLed-or-SIGTERMed server leaves a zombie
  // GPU context that nvidia-smi reports as `[Not Found]` PID, eventually
  // forcing a host reboot to reclaim. SIGKILL still leaks (no chance to
  // run handlers), so callers should prefer SIGTERM.
  let shuttingDown = false;
  const shutdown = async (sig: string) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`[server] ${sig} — shutting down`);
    // Stop accepting new connections.
    server.close();
    // Tear down WebGPU before exiting so Dawn can call vkDestroyDevice.
    if (backend.name === "webgpu") {
      try {
        const { destroyWebGPU } = await import("../../src/backend/webgpu/index.js");
        destroyWebGPU();
        console.log("[server] WebGPU device destroyed");
      } catch (e) {
        console.error(`[server] destroyWebGPU failed: ${(e as Error).message}`);
      }
    }
    process.exit(0);
  };
  process.on("SIGTERM", () => {
    shutdown("SIGTERM").catch(() => process.exit(1));
  });
  process.on("SIGINT", () => {
    shutdown("SIGINT").catch(() => process.exit(1));
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
