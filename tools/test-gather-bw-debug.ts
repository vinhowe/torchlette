import { Torchlette, initWebGPU } from "../src/index";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  // Do what gather backward does, explicitly
  const index = api.tensorFromArray([0, 0, 2, 2, 0, 0], [3, 2]);
  const grad = api.tensorFromArray([1, 1, 1, 1, 1, 1], [3, 2]);
  const z = api.zeros([3, 2]);
  const result = api.scatterAdd(z, index, grad, { dim: 0 });
  console.log("manual scatterAdd (frontend):");
  console.log("  got:     ", Array.from(await result.cpu()));
  console.log("  expected: [2,2, 0,0, 1,1]");

  // Now via runtime directly
  const rt = (api as any)._runtime();
  const z2 = rt.zeros([3, 2]);
  const result2 = rt.scatterAdd(z2._unwrap ? z2._unwrap() : z2, index._unwrap(), grad._unwrap(), { dim: 0 });
  // Need to materialize from runtime — use the api.tidy wrapper or force
  const { Tensor: FrontendTensor } = await import("../src/frontend/tensor");
  // Hack: create a new wrapper
  console.log("\nruntime.scatterAdd output shape:", result2.shape);
  console.log("  trying to read via cpu...");
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
