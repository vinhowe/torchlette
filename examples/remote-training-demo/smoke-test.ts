/**
 * Smoke test for the remote server: uploads two tensors, executes a plan
 * that adds them, downloads the result. Uses the same WebSocket RpcClient
 * as the browser, so it exercises binary frame encoding end-to-end.
 *
 * Run: npx tsx examples/remote-training-demo/smoke-test.ts [--port 9877]
 */

import { RpcClient } from "./client/transport.ts";
import type { SerializedPlan } from "../../src/remote/wire.ts";

async function main(): Promise<void> {
  const portArg = process.argv.indexOf("--port");
  const port = portArg >= 0 ? Number(process.argv[portArg + 1]) : 9877;

  const rpc = new RpcClient({ url: `ws://localhost:${port}/ws` });
  await rpc.connect();

  // Upload two tensors (binary frames on the wire).
  const a = await rpc.upload({
    values: [1, 2, 3, 4],
    shape: [2, 2],
    dtype: "f32",
  });
  const b = await rpc.upload({
    values: [10, 20, 30, 40],
    shape: [2, 2],
    dtype: "f32",
  });
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
  const exec = await rpc.execute({ plan });
  console.log(`executed, outputs=`, exec.outputs);

  // Download result (binary frame on the wire).
  const dl = await rpc.download({ handle: exec.outputs[0] });
  console.log(`result values:`, dl.values);

  const expected = [11, 22, 33, 44];
  const ok = expected.every((v, i) => dl.values[i] === v);
  console.log(ok ? "PASS" : "FAIL");

  rpc.close();
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
