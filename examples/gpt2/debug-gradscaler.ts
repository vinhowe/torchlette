/**
 * Debug GradScaler NaN detection
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GradScaler, Adam } from "../../src/optim";

async function main() {
  console.log("=== GradScaler Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const scaler = new GradScaler(api, {
    initScale: 4.0,
    growthFactor: 2.0,
    backoffFactor: 0.5,
    growthInterval: 3,
  });

  const param = api.tensorFromArray([1.0], [1], {
    device: "webgpu",
    requiresGrad: true,
  });
  const optimizer = new Adam([param], { lr: 0.01 }, api);

  // Test 1: Check what log(0) produces
  console.log("Test 1: Check log(0)");
  const zero = api.tensorFromArray([0.0], [], { device: "webgpu" });
  const logZero = api.log(zero);
  const logZeroVal = await logZero.cpu();
  console.log(`  log(0) = ${logZeroVal[0]}`);
  console.log(`  isFinite: ${Number.isFinite(logZeroVal[0])}`);

  // Test 2: Do 2 successful steps
  console.log("\nTest 2: Do 2 successful steps");
  for (let i = 0; i < 2; i++) {
    const loss = api.sum(param);
    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();

    console.log(`  Step ${i + 1} before unscale:`);
    console.log(`    param.grad = ${(await param.grad!.cpu())[0]}`);

    scaler.unscale_(optimizer);

    console.log(`  Step ${i + 1} after unscale:`);
    console.log(`    param.grad = ${(await param.grad!.cpu())[0]}`);

    scaler.step(optimizer);
    scaler.update();
    optimizer.zeroGrad();
    await scaler.resolveDeferred();

    console.log(`    foundInf = ${scaler.foundInf}`);
    console.log(`  Scale after update: ${scaler.getScale()}`);
  }

  console.log(`\nScale after 2 successful steps: ${scaler.getScale()}`);

  // Test 3: Trigger inf gradient
  console.log("\nTest 3: Trigger inf gradient");

  // Reset param grad
  param.zeroGrad();

  const zero2 = api.tensorFromArray([0.0], [], { device: "webgpu" });
  const logZero2 = api.log(zero2);
  const loss = api.mul(api.sum(param), logZero2);

  console.log("  Before backward:");
  const lossVal = await loss.cpu();
  console.log(`    loss value = ${lossVal[0]}`);

  await loss.backward();

  console.log("  After backward:");
  const gradVal = await param.grad!.cpu();
  console.log(`    param.grad = ${gradVal[0]}`);
  console.log(`    isFinite(grad): ${Number.isFinite(gradVal[0])}`);

  scaler.unscale_(optimizer);

  console.log("  After unscale_:");
  const gradAfterUnscale = await param.grad!.cpu();
  console.log(`    param.grad = ${gradAfterUnscale[0]}`);

  const stepped = scaler.step(optimizer);
  console.log(`  step() returned: ${stepped}`);

  scaler.update();
  await scaler.resolveDeferred();

  console.log(`    foundInf = ${scaler.foundInf}`);
  console.log(`\n  Final scale: ${scaler.getScale()}`);
  console.log(`  Expected: 2.0 (4.0 * 0.5)`);
  console.log(`  ${scaler.getScale() === 2.0 ? "PASS" : "FAIL"}`);

  process.exit(scaler.getScale() === 2.0 ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
