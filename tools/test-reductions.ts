import { initWebGPU, webgpuBackend } from "../src/backend/webgpu/index";

async function main() {
  await initWebGPU();
  const ops = webgpuBackend.ops;

  // Test 1: simple full sum
  const t1 = ops.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], "f32");
  const s1 = ops.sum(t1);
  const r1 = await ops.read(s1);
  console.log("full sum [1..6]:", r1, "(expected 21)");

  // Test 2: sum dim=1
  const s2 = ops.sum(t1, { dim: 1 });
  const r2 = await ops.read(s2);
  console.log("sum dim=1 [2,3]:", r2, "(expected [6, 15])");

  // Test 3: sum dim=0
  const s3 = ops.sum(t1, { dim: 0 });
  const r3 = await ops.read(s3);
  console.log("sum dim=0 [2,3]:", r3, "(expected [5, 7, 9])");

  // Test 4: mean
  const m1 = ops.mean(t1);
  const r4 = await ops.read(m1);
  console.log("mean [1..6]:", r4, "(expected 3.5)");

  // Test 5: max full
  const mx1 = ops.max(t1);
  const r5 = await ops.read(mx1);
  console.log("max [1..6]:", r5, "(expected 6)");

  // Test 6: max dim=1
  const mx2 = ops.max(t1, { dim: 1 });
  const r6 = await ops.read(mx2);
  console.log("max dim=1 [2,3]:", r6, "(expected [3, 6])");

  // Test 7: sum of ones (like expand backward)
  const t2 = ops.tensorFromArray([1, 1, 1, 1, 1, 1], [2, 3], "f32");
  const s7 = ops.sum(t2, { dim: 1 });
  const r7 = await ops.read(s7);
  console.log("sum ones dim=1 [2,3]:", r7, "(expected [3, 3])");

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
