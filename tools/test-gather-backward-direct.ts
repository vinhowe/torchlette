/**
 * Test gather backward directly (no embedding wrapper) on WebGPU.
 */
import { Torchlette, initWebGPU } from "../src/index";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  const weight = api.tensorFromArray([0, 0, 0, 0, 0, 0], [3, 2], {
    requiresGrad: true,
  });
  // Index [[0,0], [2,2], [0,0]] directly — shape [3,2]
  const index = api.tensorFromArray([0, 0, 2, 2, 0, 0], [3, 2]);
  const gathered = api.gather(weight, index, { dim: 0 });
  const loss = gathered.sum();
  await loss.backward();
  console.log("direct-index gather:");
  console.log("  got:     ", Array.from(await weight.grad!.cpu()));
  console.log("  expected:", [2, 2, 0, 0, 1, 1]);
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
