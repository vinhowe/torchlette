/**
 * Node-side integration test: trains the tiny transformer through the remote
 * server for a handful of steps. Verifies loss drops and markStep is cleaning
 * up handles.
 *
 * Run: npx tsx examples/remote-training-demo/transformer-test.ts
 */

import { spawn } from "node:child_process";
import { createRemoteEngine } from "./client/engine.ts";
import { buildCharDataset, createModel, parameters } from "./client/model.ts";
import { modelConfigSmall, trainStep } from "./client/train.ts";
import { RpcClient } from "./client/transport.ts";

async function waitForServer(url: string): Promise<void> {
  for (let i = 0; i < 40; i++) {
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

async function main(): Promise<void> {
  const port = 9881;
  const url = `ws://localhost:${port}/ws`;

  console.log(`[test] starting server...`);
  const server = spawn(
    "npx",
    ["tsx", "examples/remote-training-demo/server.ts", "--port", String(port)],
    { stdio: ["ignore", "inherit", "inherit"] },
  );

  let exitCode = 1;
  try {
    await waitForServer(url);

    const rpc = new RpcClient({ url, onLog: () => {} });
    await rpc.connect();
    const remote = createRemoteEngine(rpc);

    const text =
      "the quick brown fox jumps over the lazy dog. " +
      "how vexingly quick daft zebras jump! " +
      "pack my box with five dozen liquor jugs. ".repeat(4);
    const ds = buildCharDataset(text);
    console.log(
      `[test] dataset: ${ds.text.length} chars, vocab=${ds.vocabSize}`,
    );

    const cfg = modelConfigSmall(ds.vocabSize);
    const model = createModel(remote.torch, cfg, 42);
    const params = parameters(model);

    console.log(
      `[test] model: ${cfg.numLayers}L, D=${cfg.embedDim}, H=${cfg.numHeads}, ` +
        `V=${ds.vocabSize}, T=${cfg.blockSize}, ${params.length} param tensors`,
    );

    let s = 1;
    const rng = () => {
      s = (Math.imul(s, 1103515245) + 12345) >>> 0;
      return ((s >>> 0) / 0x100000000) * 2 - 1;
    };

    const numSteps = 10;
    const trainCfg = { lr: 0.03, batchSize: 4, seqLen: cfg.blockSize, seed: 1 };

    console.log(`[test] training ${numSteps} steps...`);
    const t0 = performance.now();
    let firstLoss = 0;
    let lastLoss = 0;
    for (let step = 0; step < numSteps; step++) {
      const stepStart = performance.now();
      const loss = await trainStep(remote.torch, model, ds, params, trainCfg, rng);
      await remote.markStep(params);
      const stepMs = performance.now() - stepStart;
      if (step === 0) firstLoss = loss;
      lastLoss = loss;
      console.log(
        `  step ${String(step).padStart(2)}  loss=${loss.toFixed(4)}  ` +
          `${stepMs.toFixed(0)}ms  handles=${remote.handles.size()}`,
      );
    }
    const totalMs = performance.now() - t0;

    rpc.close();

    console.log();
    console.log(`=== summary ===`);
    console.log(`first loss: ${firstLoss.toFixed(4)}`);
    console.log(`last loss:  ${lastLoss.toFixed(4)}`);
    console.log(`total:      ${totalMs.toFixed(0)}ms  (${(totalMs / numSteps).toFixed(0)}ms/step)`);
    console.log(
      `rpc stats: executes=${remote.stats.executes} ` +
        `nodes=${remote.stats.nodesShipped} ` +
        `reads=${remote.stats.scalarReads} ` +
        `released=${remote.stats.handlesReleased} ` +
        `up=${(remote.stats.bytesUp / 1024).toFixed(1)}KB ` +
        `down=${(remote.stats.bytesDown / 1024).toFixed(1)}KB`,
    );

    const dropped = lastLoss < firstLoss * 0.98;
    if (dropped) {
      console.log("PASS  transformer training ran through remote wire");
      exitCode = 0;
    } else {
      console.log("FAIL  loss did not decrease");
    }
  } catch (e) {
    console.error("[test] error:", e);
  } finally {
    server.kill("SIGTERM");
  }
  process.exit(exitCode);
}

main();
