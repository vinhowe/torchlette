/**
 * Mirror the CPU frontend.spec.ts embedding backward test on WebGPU.
 */
import { Torchlette, initWebGPU } from "../src/index";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  const weight = api.tensorFromArray([0, 0, 0, 0, 0, 0], [3, 2], {
    requiresGrad: true,
  });
  const indices = api.tensorFromArray([0, 2, 0], [3]);
  const result = api.embedding(weight, indices);
  const loss = result.sum();
  await loss.backward();

  const g = Array.from(await weight.grad!.cpu());
  console.log("got:      ", g);
  console.log("expected: ", [2, 2, 0, 0, 1, 1]);
  console.log(g.join(",") === [2, 2, 0, 0, 1, 1].join(",") ? "PASS" : "FAIL");
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
