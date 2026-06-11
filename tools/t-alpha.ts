import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
async function main() {
  if (!(await initWebGPU())) process.exit(1);
  for (const dev of ["webgpu", "cpu"] as const) {
    const api = new Torchlette(dev);
    const rt = api._runtime();
    const a = rt.tensorFromArray([10, 20], [2], dev);
    const b = rt.tensorFromArray([1, 2], [2], dev);
    const out = rt.sub(a, b, { alpha: 0.5 });
    console.log(dev, Array.from(await rt.cpu(out)), "want [9.5, 19]");
  }
  process.exit(0);
}
main().catch((e) => { console.error(e); process.exit(1); });
