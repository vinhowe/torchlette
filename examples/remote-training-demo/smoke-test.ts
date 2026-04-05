/**
 * Smoke test for the remote server: connects via WebSocket, uploads two
 * tensors, executes a plan that adds them, downloads the result.
 *
 * Run: npx tsx examples/remote-training-demo/smoke-test.ts [--port 9877]
 */

import { WebSocket } from "ws";
import type {
  DownloadParams,
  DownloadResult,
  ExecuteParams,
  ExecuteResult,
  RpcResponse,
  UploadParams,
  UploadResult,
} from "../../src/remote/rpc.js";
import type { SerializedPlan } from "../../src/remote/wire.js";

async function main(): Promise<void> {
  const portArg = process.argv.indexOf("--port");
  const port = portArg >= 0 ? Number(process.argv[portArg + 1]) : 9877;

  const ws = new WebSocket(`ws://localhost:${port}/ws`);
  await new Promise<void>((resolve, reject) => {
    ws.on("open", () => resolve());
    ws.on("error", reject);
  });

  let nextId = 1;
  const pending = new Map<number, (r: RpcResponse) => void>();
  ws.on("message", (data) => {
    const msg = JSON.parse(data.toString()) as RpcResponse;
    const resolver = pending.get(msg.id);
    if (resolver) {
      pending.delete(msg.id);
      resolver(msg);
    }
  });

  async function rpc<T>(method: string, params: unknown): Promise<T> {
    const id = nextId++;
    const req = { id, method, params };
    const p = new Promise<RpcResponse>((resolve) => {
      pending.set(id, resolve);
    });
    ws.send(JSON.stringify(req));
    const resp = await p;
    if ("error" in resp) throw new Error(resp.error.message);
    return resp.result as T;
  }

  // Upload two tensors.
  const a = await rpc<UploadResult>("upload", {
    values: [1, 2, 3, 4],
    shape: [2, 2],
    dtype: "f32",
  } satisfies UploadParams);
  const b = await rpc<UploadResult>("upload", {
    values: [10, 20, 30, 40],
    shape: [2, 2],
    dtype: "f32",
  } satisfies UploadParams);
  console.log(`uploaded handles: a=${a.handle} b=${b.handle}`);

  // Execute a plan: c = a + b
  const plan: SerializedPlan = {
    version: 1,
    nodes: [
      {
        idx: 0,
        op: "add",
        inputs: [
          { kind: "materialized", handle: a.handle },
          { kind: "materialized", handle: b.handle },
        ],
        shape: [2, 2],
        dtype: "f32",
        device: "cpu",
      },
    ],
    externalHandles: [a.handle, b.handle],
    outputNodes: [0],
  };
  const exec = await rpc<ExecuteResult>("execute", { plan } satisfies ExecuteParams);
  console.log(`executed, outputs=`, exec.outputs);

  // Download result.
  const dl = await rpc<DownloadResult>("download", {
    handle: exec.outputs[0],
  } satisfies DownloadParams);
  console.log(`result values:`, dl.values);

  const expected = [11, 22, 33, 44];
  const ok = expected.every((v, i) => dl.values[i] === v);
  console.log(ok ? "PASS" : "FAIL");

  ws.close();
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
