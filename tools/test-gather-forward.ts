/**
 * Quick check that gather forward still works.
 */
import { Torchlette, initWebGPU } from "../src/index";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");
  const weight = api.tensorFromArray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], [5, 3]);
  const tokens = api.tensorFromArray([0, 2, 4, 1], [4]);
  const emb = api.embedding(weight, tokens);
  console.log("embedding:", Array.from(await emb.cpu()));
  console.log("expected: [0,0,0, 2,2,2, 4,4,4, 1,1,1]");
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
