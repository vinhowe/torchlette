/**
 * Test scatterAdd with 2D tensors, dim=0.
 */
import { Torchlette, initWebGPU } from "../src/index";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  // Simulate what embedding backward does:
  // out [3,2] target, contig index [3,2], grad_out [3,2]
  const out = api.zeros([3, 2]);
  const index = api.tensorFromArray([0, 0, 2, 2, 0, 0], [3, 2]);
  const src = api.tensorFromArray([1, 1, 1, 1, 1, 1], [3, 2]);
  const result = api.scatterAdd(out, index, src, { dim: 0 });
  console.log("2D scatter-add:");
  console.log("  got:     ", Array.from(await result.cpu()));
  console.log("  expected: [2,2, 0,0, 1,1]");
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
