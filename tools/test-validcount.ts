/**
 * Test validCount computation: count of targets != -1
 */
import { Torchlette, initWebGPU } from "../src/index";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  // Same pattern as bio: mix of -1 and valid targets
  const tgtArr = new Float32Array(20);
  tgtArr[0] = 3; tgtArr[1] = -1; tgtArr[2] = 5; tgtArr[3] = -1; tgtArr[4] = 7;
  tgtArr[5] = 2; tgtArr[6] = -1; tgtArr[7] = 4; tgtArr[8] = -1; tgtArr[9] = 0;
  tgtArr[10] = -1; tgtArr[11] = -1; tgtArr[12] = 1; tgtArr[13] = -1; tgtArr[14] = 8;
  tgtArr[15] = 9; tgtArr[16] = -1; tgtArr[17] = -1; tgtArr[18] = -1; tgtArr[19] = 6;
  // 10 valid, 10 ignored

  const targets = api.tensorFromArray(tgtArr, [20]);
  const ignoreT = api.full([20], -1, { device: "webgpu", dtype: "f32" });
  const mask = api.ne(targets, ignoreT);
  const validCount = api.mul(mask, 1.0).sum();
  const vc = await validCount.item();
  console.log(`validCount: ${vc} (expected 10)`);

  // Also check mask values
  const maskF32 = api.mul(mask, 1.0);
  const maskData = await maskF32.cpu();
  console.log("mask:", Array.from(maskData));

  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
