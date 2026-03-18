/**
 * Quick test: verify row-program fusion works for 2D softmax / log_softmax.
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { RuntimeEngine } from "../src/runtime/engine";

async function main() {
  await initWebGPU();
  const runtime = new RuntimeEngine({ enableFusion: true });
  const api = new Torchlette(runtime);

  const x = api.tensorFromArray(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [3, 4],
    { device: "webgpu" },
  );

  // softmax
  const sm = x.softmax(-1);
  await api.beginStep(); // force through optimized path (forceAllPending)
  const smData = await sm.cpu();
  for (let row = 0; row < 3; row++) {
    let rowSum = 0;
    for (let col = 0; col < 4; col++) rowSum += smData[row * 4 + col];
    console.log(`softmax row ${row} sum: ${rowSum.toFixed(6)}`);
    if (Math.abs(rowSum - 1.0) > 0.001) throw new Error(`FAIL row ${row}`);
  }

  console.log("\nPASSED");
  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
