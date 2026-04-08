/**
 * End-to-end remote training test: Node.js client → WebSocket → real server.
 *
 * This is the Node.js equivalent of the browser demo. It connects to the
 * actual server process via WebSocket, sends serialized plans, receives
 * handles, creates stubs — exactly like remote-hooks.ts in the browser.
 *
 * Usage:
 *   1. Start server: npx tsx examples/remote-training-demo/server.ts --port 9882
 *   2. Run this:     npx tsx tools/test-remote-e2e.ts
 */
import WebSocket from "ws";
import { Torchlette, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import {
  generateBatchWithCompartments,
  setTransitionMatrices,
  VOCAB_SIZE_DATA,
} from "../examples/toy-compartmentalization/src/lib/data";
import type { ExecutionPlan, LazyIRNode } from "../src/graph/types";
import type { BackendTensor, DType } from "../src/backend/types";
import { createStorageHandle } from "../src/graph/node-factory";
import { serializePlan } from "../src/remote/serialize";
import { getPendingNodeIds } from "../src/runtime/tensor";
import type { HandleRef } from "../src/remote/wire";
import type {
  ExecuteParams,
  ExecuteResult,
  RpcRequest,
  RpcResponse,
  HelloResult,
} from "../src/remote/rpc";

// ============================================================================
// Node.js WebSocket RPC client (mirrors remote-transport.ts for browser)
// ============================================================================

class NodeRpcClient {
  private ws!: WebSocket;
  private nextId = 1;
  private pending = new Map<number, (r: any) => void>();
  sessionId = "";

  async connect(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(url);
      let helloed = false;
      this.ws.on("open", () => console.log("[rpc] connected"));
      this.ws.on("message", (data: Buffer | string, isBinary: boolean) => {
        if (!isBinary) {
          const msg = JSON.parse(data.toString());
          if (!helloed && msg.id === 0) {
            this.sessionId = msg.result.sessionId;
            console.log(`[rpc] session ${this.sessionId}`);
            helloed = true;
            resolve();
            return;
          }
          const resolver = this.pending.get(msg.id);
          if (resolver) {
            this.pending.delete(msg.id);
            if (msg.error) resolver(Promise.reject(new Error(msg.error.message)));
            else resolver(msg.result);
          }
        }
      });
      this.ws.on("error", reject);
    });
  }

  async execute(params: ExecuteParams): Promise<ExecuteResult> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, (r) => {
        if (r instanceof Promise) r.catch(reject);
        else resolve(r);
      });
      this.ws.send(JSON.stringify({ id, method: "execute", params }));
    });
  }

  close() {
    this.ws.close();
  }
}

// ============================================================================
// Remote hooks (mirrors remote-hooks.ts exactly)
// ============================================================================

function createRemoteHooks(transport: NodeRpcClient) {
  const handleMap = new Map<number, HandleRef>();

  const executionHook = async (plan: ExecutionPlan) => {
    if (plan.nodes.length === 0) return;

    // Compute outputNodes (same as remote-hooks.ts)
    const pendingIds = getPendingNodeIds();
    const outputNodes: LazyIRNode[] = [];
    for (const node of plan.nodes) {
      if (pendingIds.has(node.id)) outputNodes.push(node);
    }

    const wire = serializePlan(plan, {
      resolveHandle: (storageId: number) => {
        const h = handleMap.get(storageId);
        if (!h) throw new Error(`No HandleRef for storage id=${storageId}`);
        return h;
      },
      outputNodes,
    });

    const result = await transport.execute({ plan: wire });

    // Create stubs (same as remote-hooks.ts)
    for (const [idxStr, handleRef] of Object.entries(result.outputs)) {
      const idx = Number(idxStr);
      const node = plan.nodes[idx];
      const stub: BackendTensor = {
        shape: [...node.shape],
        dtype: node.dtype,
        ownsBuffer: true,
        toArray() { throw new Error("Remote stub"); },
        destroy() {},
      };
      const storage = createStorageHandle(node.device, stub);
      handleMap.set(storage.id, handleRef);
      node.result = storage;
    }
    if (result.sideOutputs) {
      for (const [key, handleRef] of Object.entries(result.sideOutputs)) {
        const [idxStr, outIdxStr] = key.split(":");
        const node = plan.nodes[Number(idxStr)];
        const outIdx = Number(outIdxStr);
        const stub: BackendTensor = {
          shape: [...node.shape],
          dtype: node.dtype,
          ownsBuffer: true,
          toArray() { throw new Error("Remote stub"); },
          destroy() {},
        };
        const storage = createStorageHandle(node.device, stub);
        handleMap.set(storage.id, handleRef);
        if (!node.results) node.results = [];
        node.results[outIdx] = storage;
      }
    }
  };

  // readHook: stubs can't be read locally, just return empty
  const readHook = async (bt: BackendTensor) => {
    if ("_download" in bt) return (bt as any)._download();
    return bt.toArray();
  };

  return { executionHook, readHook, handleMap };
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const SERVER_URL = process.env.SERVER_URL ?? "ws://localhost:9882/ws";
  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA * 1 + 1, S = 10, B = 128; // c=1

  // Connect to server
  const transport = new NodeRpcClient();
  await transport.connect(SERVER_URL);

  const { executionHook, readHook } = createRemoteHooks(transport);

  // Create Torchlette with CPU backend (client doesn't compute)
  const api = new Torchlette("cpu", {
    executionHook,
    readHook,
  });
  api.manualSeed(42);
  const m = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });
  const o = new Adam(m.parameters(), { lr: 1e-3 });

  const planSizes: number[][] = [];
  let hookCalls = 0;
  const origHook = executionHook;
  const trackedHook = async (plan: ExecutionPlan) => {
    hookCalls++;
    if (planSizes.length > 0) {
      planSizes[planSizes.length - 1].push(plan.nodes.length);
    }
    return origHook(plan);
  };
  // Patch the hook to track plan sizes
  (api as any).runtime._executionHook = trackedHook;

  const STEPS = 8;
  for (let step = 0; step < STEPS; step++) {
    planSizes.push([]);
    const t0 = performance.now();
    await api.beginStep();
    const b = generateBatchWithCompartments({ seqLen: S, batchSize: B }, 2);
    const t = api.tensorFromArray(b.tokens, [B, S], { dtype: "i32" });
    const g = api.tensorFromArray(b.targets as number[], [B * (S - 1)], { dtype: "i32" });
    const l = api.tidy(() => {
      const f = m.forward(t);
      const lg = f.logits.narrow(1, 0, S - 1).contiguous().reshape([B * (S - 1), V]);
      const x = crossEntropy(api, lg, g); api.keep(x); return x;
    });
    t.dispose(); g.dispose();

    // Browser only calls item() every LOG_INTERVAL=10 steps.
    // Skip it most steps to match the browser's plan structure.
    const LOG_INTERVAL = 10;
    if (step % LOG_INTERVAL === 0) {
      try {
        const v = await l.item();
        console.log(`step ${step}: loss=${typeof v === "number" ? v.toFixed(4) : v}, ${(performance.now() - t0).toFixed(0)}ms`);
      } catch (e) {
        console.log(`step ${step}: item() failed: ${(e as Error).message}, ${(performance.now() - t0).toFixed(0)}ms`);
      }
    }

    await l.backward(); l.dispose();
    o.step(); o.zeroGrad();
    api.endStep();
  }

  // Report
  console.log("\nPlan sizes per step:");
  for (let i = 0; i < planSizes.length; i++) {
    const total = planSizes[i].reduce((a, b) => a + b, 0);
    console.log(`  step ${i}: ${planSizes[i].join(" + ")} = ${total}`);
  }
  const step1Total = planSizes[1]?.reduce((a, b) => a + b, 0) ?? 0;
  const lastTotal = planSizes[STEPS - 1]?.reduce((a, b) => a + b, 0) ?? 0;
  const growth = step1Total > 0 ? lastTotal / step1Total : 0;
  console.log(`\nGrowth factor: ${growth.toFixed(2)}x`);
  if (growth > 1.15) {
    console.error(`FAIL: Plans growing! ${step1Total} → ${lastTotal}`);
  } else {
    console.log("PASS: Plan sizes stable");
  }

  transport.close();
  process.exit(growth > 1.15 ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
