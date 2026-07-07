/**
 * Test runtime.scatterAdd directly (the path that api.gather backward uses).
 */
import { Torchlette, initWebGPU } from "../src/index";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");
  const rt = (api as any)._runtime();

  const zeros = rt.zeros([3, 2]);
  const index = rt.tensorFromArray([0, 0, 2, 2, 0, 0], [3, 2], "webgpu");
  const src = rt.tensorFromArray([1, 1, 1, 1, 1, 1], [3, 2], "webgpu");
  const result = rt.scatterAdd(zeros, index, src, { dim: 0 });
  console.log("runtime.scatterAdd result:");
  console.log("  got:     ", Array.from(await rt.toHost(result, "webgpu")));
  console.log("  expected: [2,2, 0,0, 1,1]");
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
