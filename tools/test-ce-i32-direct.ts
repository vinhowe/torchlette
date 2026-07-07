/**
 * Isolate the bug: direct i32 targets + ignoreIndex = -1 to fused CE.
 * Verify what the kernel actually reads.
 */
import { Torchlette, initWebGPU } from "../src/index";
import { crossEntropy } from "../src/nn/functional";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  const N = 4, V = 10;
  const logitsData = new Float32Array(N * V);
  for (let i = 0; i < logitsData.length; i++) logitsData[i] = Math.sin(i * 0.5);
  // Mix with -1 as ignore
  const targets = [3, -1, 0, -1];
  const logits = api.tensorFromArray(logitsData, [N, V]);

  // Direct i32 via tensorFromArray
  const tgtI32 = api.tensorFromArray(targets, [N], { dtype: "i32" });
  console.log("tgtI32.dtype:", tgtI32.dtype);
  const i32Data = await tgtI32.cpu();
  console.log("tgtI32 readback:", Array.from(i32Data));
  // Check raw bytes to confirm negative encoding
  const i32Bytes = new Uint32Array((i32Data as any).buffer || new Int32Array(i32Data).buffer);
  console.log("tgtI32 as u32:", Array.from(i32Bytes).map(x => x.toString(16)));

  // Test with NON-negative targets first (no ignore needed)
  const posTargets = [3, 0, 5, 2];
  const tgtPosI32 = api.tensorFromArray(posTargets, [N], { dtype: "i32" });
  const tgtPosF32 = api.tensorFromArray(posTargets, [N]);
  const lossPosI32 = crossEntropy(api, logits, tgtPosI32);
  const lossPosF32 = crossEntropy(api, logits, tgtPosF32);
  console.log(`GPU i32 (no ignore) loss: ${(await lossPosI32.item()).toFixed(4)}`);
  console.log(`GPU f32 (no ignore) loss: ${(await lossPosF32.item()).toFixed(4)}`);

  // Now try with ignoreIndex
  const tgtF32 = api.tensorFromArray(targets, [N]);
  const lossF32 = crossEntropy(api, logits, tgtF32, { ignoreIndex: -1 });
  const lF32 = await lossF32.item();
  console.log(`GPU f32 (ignore=-1) loss: ${lF32.toFixed(4)}`);

  const lossI32 = crossEntropy(api, logits, tgtI32, { ignoreIndex: -1 });
  const lI32 = await lossI32.item();
  console.log(`GPU i32 (ignore=-1) loss: ${lI32.toFixed(4)}`);

  // Try positive ignoreIndex = 99 (not in targets)
  const lossPos99 = crossEntropy(api, logits, tgtPosI32, { ignoreIndex: 99 });
  console.log(`GPU i32 (ignore=99) loss: ${(await lossPos99.item()).toFixed(4)}`);

  // Try positive ignoreIndex = 0 (in targets)
  const lossPos0 = crossEntropy(api, logits, tgtPosI32, { ignoreIndex: 0 });
  console.log(`GPU i32 (ignore=0) loss: ${(await lossPos0.item()).toFixed(4)}`);

  // Test api.ne / api.full with i32
  const ignoreT = api.full([N], -1, { device: "webgpu", dtype: "i32" });
  console.log("ignoreT dtype:", ignoreT.dtype, "values:", Array.from(await ignoreT.cpu()));
  const mask = api.ne(tgtI32, ignoreT);
  console.log("mask dtype:", mask.dtype, "values:", Array.from(await mask.cpu()));
  const maskMul = api.mul(mask, 1.0);
  const validCount = maskMul.sum();
  console.log("validCount:", await validCount.item(), "(expected 2)");

  // Try: cast i32→f32→i32 to force a fresh buffer
  const tgtF32FromI32 = api.toDtype(tgtI32, "f32");
  const tgtI32Fresh = api.toDtype(tgtF32FromI32, "i32");
  console.log("fresh i32 readback:", Array.from(await tgtI32Fresh.cpu()));
  const lossFresh = crossEntropy(api, logits, tgtI32Fresh, { ignoreIndex: -1 });
  const lFresh = await lossFresh.item();
  console.log(`GPU i32 (via cast roundtrip) loss: ${lFresh.toFixed(4)}`);

  // Check what tgtF32 looks like AFTER cast to i32 (what the kernel actually reads)
  const tgtI32_fromCast = api.toDtype(tgtF32, "i32");
  const castData = await tgtI32_fromCast.cpu();
  console.log("tgtF32 after cast to i32:", Array.from(castData));

  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
