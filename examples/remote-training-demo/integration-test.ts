/**
 * Integration test: drives the remote server end-to-end from a Node client
 * using RemoteRuntimeEngine. Runs the same toy 2-layer MLP as
 * src/remote/toy-training-test.ts, but every plan travels over WebSocket.
 *
 * Launches the server as a subprocess, runs the training loop against it,
 * verifies loss decreases, kills the server.
 *
 * Run: npx tsx examples/remote-training-demo/integration-test.ts
 */

import { spawn, type ChildProcess } from "node:child_process";
import { createRemoteEngine } from "../../src/remote/client-engine.ts";
import { RpcClient } from "./client/transport.ts";

// Node 22+ has WebSocket globally.

async function waitForServer(url: string, maxRetries = 20): Promise<void> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const ws = new WebSocket(url);
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("timeout")), 200);
        ws.addEventListener("open", () => {
          clearTimeout(timer);
          ws.close();
          resolve();
        });
        ws.addEventListener("error", (e) => {
          clearTimeout(timer);
          reject(e);
        });
      });
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 100));
    }
  }
  throw new Error(`Server did not start at ${url}`);
}

function startServer(port: number): ChildProcess {
  const proc = spawn(
    "npx",
    ["tsx", "examples/remote-training-demo/server.ts", "--port", String(port)],
    { stdio: ["ignore", "inherit", "inherit"] },
  );
  return proc;
}

// ============================================================================
// Tiny MLP training via RemoteRuntimeEngine
// ============================================================================

const XOR_INPUTS = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const XOR_TARGETS = [0, 1, 1, 0];

function makePrng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return ((s >>> 0) / 0x100000000) * 2 - 1;
  };
}

async function runRemoteTraining(url: string): Promise<{
  losses: number[];
  stats: ReturnType<typeof createRemoteEngine>["stats"];
}> {
  const rpc = new RpcClient({
    url,
    onLog: (m) => console.log(m),
  });
  await rpc.connect();

  const { torch: api, handles, stats } = createRemoteEngine(rpc);

  const hidden = 4;
  const steps = 20;
  const lr = 0.1;
  const opts = { device: "cpu" as const };

  const rng = makePrng(42);
  const draw = (n: number) => Array.from({ length: n }, () => rng() * 0.5);

  const W1 = api
    .tensorFromArray(draw(2 * hidden), [2, hidden], opts)
    .requires_grad_(true);
  const b1 = api
    .tensorFromArray(new Array(hidden).fill(0), [hidden], opts)
    .requires_grad_(true);
  const W2 = api
    .tensorFromArray(draw(hidden), [hidden, 1], opts)
    .requires_grad_(true);
  const b2 = api
    .tensorFromArray([0], [1], opts)
    .requires_grad_(true);

  const X = api.tensorFromArray(XOR_INPUTS.flat(), [4, 2], opts);
  const T = api.tensorFromArray(XOR_TARGETS, [4, 1], opts);

  const losses: number[] = [];
  for (let step = 0; step < steps; step++) {
    const h = api.relu(api.add(api.matmul(X, W1), b1));
    const y = api.add(api.matmul(h, W2), b2);
    const diff = api.sub(y, T);
    const sq = api.mul(diff, diff);
    const loss = sq.mean();
    if (typeof loss === "number") throw new Error("loss should be tensor");

    const lossVal = await loss.item();
    losses.push(lossVal);

    await loss.backward();

    for (const p of [W1, b1, W2, b2]) {
      if (!p.grad) throw new Error("missing grad");
      api.noGrad(() => {
        const updated = api.sub(p, api.mul(p.grad!, lr));
        p.copy_(updated);
      });
      p.zeroGrad();
    }

    if (step % 5 === 0 || step === steps - 1) {
      console.log(
        `step ${String(step).padStart(2)}  loss=${lossVal.toFixed(6)}  handles=${handles.size()}`,
      );
    }
  }

  rpc.close();
  return { losses, stats };
}

// ============================================================================
// Runner
// ============================================================================

async function main(): Promise<void> {
  const port = 9880;
  const url = `ws://localhost:${port}/ws`;

  console.log(`[test] starting server on :${port}...`);
  const server = startServer(port);

  let exitCode = 1;
  try {
    await waitForServer(url);
    console.log(`[test] server ready, running training...`);
    const { losses, stats } = await runRemoteTraining(url);

    const first = losses[0];
    const last = losses[losses.length - 1];
    const decreased = last < first * 0.95; // at least 5% decrease

    console.log();
    console.log(`losses first=${first.toFixed(4)} last=${last.toFixed(4)}`);
    console.log(
      `stats: executes=${stats.executes} nodes=${stats.nodesShipped} ` +
        `downloads=${stats.downloads} reads=${stats.scalarReads} ` +
        `up=${(stats.bytesUp / 1024).toFixed(1)}KB ` +
        `down=${(stats.bytesDown / 1024).toFixed(1)}KB`,
    );

    if (decreased) {
      console.log("PASS  training converged through remote wire");
      exitCode = 0;
    } else {
      console.log(`FAIL  loss did not decrease sufficiently`);
    }
  } catch (e) {
    console.error("[test] error:", e);
  } finally {
    server.kill("SIGTERM");
  }
  process.exit(exitCode);
}

main();
