/**
 * Minimal test for compute shader timestamp readback on V100/Dawn.
 * Tests whether using a compute shader to copy from the resolve buffer
 * (instead of copyBufferToBuffer) avoids the mapAsync deadlock.
 */

async function main() {
  const mod = await import("webgpu");
  const gpu = mod.create([]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    console.error("No adapter");
    process.exit(1);
  }

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
  } catch (e: any) {
    console.error("Device creation failed:", e.message);
    process.exit(1);
  }
  console.log("Device created with timestamp-query");

  const QUERY_RESOLVE = 0x0200;
  const COPY_SRC = 0x0004;
  const COPY_DST = 0x0008;
  const MAP_READ = 0x0001;
  const STORAGE = 0x0080;

  // Create query set (simulate ~100 compute passes = 200 timestamps)
  const NUM_PASSES = 100;
  const querySet = device.createQuerySet({
    type: "timestamp" as GPUQueryType,
    count: NUM_PASSES * 2,
  });

  // Resolve buffer with STORAGE for compute shader readback
  const resolveBuffer = device.createBuffer({
    size: NUM_PASSES * 2 * 8,
    usage: QUERY_RESOLVE | COPY_SRC | STORAGE,
  });

  // Create a dummy storage buffer for compute passes
  const dummyBuf = device.createBuffer({
    size: 1024 * 1024, // 1MB
    usage: STORAGE,
  });

  // Create a simple compute pipeline for the dummy work
  const workShader = device.createShaderModule({
    code: `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx < arrayLength(&data)) {
    data[idx] = data[idx] * 1.001 + 0.001;
  }
}
`,
  });
  const workPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: workShader, entryPoint: "main" },
  });
  const workBindGroup = device.createBindGroup({
    layout: workPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: dummyBuf } }],
  });

  // Create compute shader copy pipeline (the thing we're testing)
  const copyShader = device.createShaderModule({
    code: `
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx < arrayLength(&src)) {
    dst[idx] = src[idx];
  }
}
`,
  });
  const copyBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: 0x4, buffer: { type: "read-only-storage" as GPUBufferBindingType } },
      { binding: 1, visibility: 0x4, buffer: { type: "storage" as GPUBufferBindingType } },
    ],
  });
  const copyPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [copyBindGroupLayout],
  });
  const copyPipeline = device.createComputePipeline({
    layout: copyPipelineLayout,
    compute: { module: copyShader, entryPoint: "main" },
  });

  console.log(`Running ${NUM_PASSES} compute passes with timestamps...`);

  // Run compute passes with timestamp writes
  const encoder = device.createCommandEncoder();
  for (let i = 0; i < NUM_PASSES; i++) {
    const pass = encoder.beginComputePass({
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: i * 2,
        endOfPassWriteIndex: i * 2 + 1,
      },
    });
    pass.setPipeline(workPipeline);
    pass.setBindGroup(0, workBindGroup);
    pass.dispatchWorkgroups(Math.ceil(1024 * 1024 / 4 / 256));
    pass.end();
  }

  // Resolve timestamps
  encoder.resolveQuerySet(querySet, 0, NUM_PASSES * 2, resolveBuffer, 0);
  device.queue.submit([encoder.finish()]);
  console.log("Submitted compute + resolveQuerySet");

  // Fence via writeBuffer+mapAsync (known to work on V100/Dawn)
  const fenceBuf = device.createBuffer({ size: 4, usage: COPY_DST | MAP_READ });
  device.queue.writeBuffer(fenceBuf, 0, new Uint8Array([1, 2, 3, 4]));
  await fenceBuf.mapAsync(MAP_READ);
  fenceBuf.unmap();
  fenceBuf.destroy();
  console.log("Fence drained — GPU work complete");

  // === Test 1: Direct copyBufferToBuffer (expected to deadlock) ===
  console.log("\n--- Test 1: Direct copyBufferToBuffer from resolve buffer ---");
  const byteSize = NUM_PASSES * 2 * 8;
  const directStaging = device.createBuffer({ size: byteSize, usage: MAP_READ | COPY_DST });
  const directEncoder = device.createCommandEncoder();
  directEncoder.copyBufferToBuffer(resolveBuffer, 0, directStaging, 0, byteSize);
  device.queue.submit([directEncoder.finish()]);

  const directFence = device.createBuffer({ size: 4, usage: COPY_DST | MAP_READ });
  device.queue.writeBuffer(directFence, 0, new Uint8Array([1, 2, 3, 4]));

  let directOk = false;
  try {
    const result = await Promise.race([
      directStaging.mapAsync(MAP_READ).then(() => true),
      new Promise<false>((r) => setTimeout(() => r(false), 3000)),
    ]);
    directOk = result;
  } catch {}
  if (directOk) {
    const data = new BigInt64Array(directStaging.getMappedRange());
    console.log(`  SUCCESS! First timestamp: ${data[0]}, Last: ${data[NUM_PASSES * 2 - 1]}`);
    directStaging.unmap();
  } else {
    console.log("  DEADLOCKED (as expected on V100/Dawn) — timed out after 3s");
  }
  directStaging.destroy();
  directFence.destroy();

  // === Test 2: Compute shader copy (our fix) ===
  console.log("\n--- Test 2: Compute shader copy from resolve buffer ---");
  const intermediate = device.createBuffer({ size: byteSize, usage: STORAGE | COPY_SRC });
  const staging = device.createBuffer({ size: byteSize, usage: MAP_READ | COPY_DST });

  const copyBindGroup = device.createBindGroup({
    layout: copyBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: resolveBuffer, size: byteSize } },
      { binding: 1, resource: { buffer: intermediate, size: byteSize } },
    ],
  });

  const numU32 = byteSize / 4;
  const copyEncoder = device.createCommandEncoder();
  const copyPass = copyEncoder.beginComputePass();
  copyPass.setPipeline(copyPipeline);
  copyPass.setBindGroup(0, copyBindGroup);
  copyPass.dispatchWorkgroups(Math.ceil(numU32 / 256));
  copyPass.end();
  copyEncoder.copyBufferToBuffer(intermediate, 0, staging, 0, byteSize);
  device.queue.submit([copyEncoder.finish()]);

  // Fence
  const fence2 = device.createBuffer({ size: 4, usage: COPY_DST | MAP_READ });
  device.queue.writeBuffer(fence2, 0, new Uint8Array([1, 2, 3, 4]));

  let shaderOk = false;
  try {
    // First fence
    const fenceOk = await Promise.race([
      fence2.mapAsync(MAP_READ).then(() => true),
      new Promise<false>((r) => setTimeout(() => r(false), 5000)),
    ]);
    fence2.unmap();
    fence2.destroy();
    if (!fenceOk) {
      console.log("  FENCE DEADLOCKED — unexpected!");
    } else {
      // Map staging
      const result = await Promise.race([
        staging.mapAsync(MAP_READ).then(() => true),
        new Promise<false>((r) => setTimeout(() => r(false), 5000)),
      ]);
      shaderOk = result;
    }
  } catch (e) {
    console.log("  ERROR:", e);
  }

  if (shaderOk) {
    const data = new BigInt64Array(staging.getMappedRange());
    let validCount = 0;
    let totalNs = 0n;
    for (let i = 0; i < NUM_PASSES; i++) {
      const start = data[i * 2];
      const end = data[i * 2 + 1];
      if (start > 0n && end > 0n && end > start) {
        validCount++;
        totalNs += end - start;
      }
    }
    console.log(`  SUCCESS! ${validCount}/${NUM_PASSES} valid timestamps`);
    console.log(`  Total GPU time: ${Number(totalNs) / 1_000_000}ms`);
    console.log(`  Avg per pass: ${Number(totalNs) / validCount / 1_000}µs`);
    staging.unmap();
  } else {
    console.log("  DEADLOCKED — compute shader approach also fails");
  }

  intermediate.destroy();
  staging.destroy();
  dummyBuf.destroy();
  resolveBuffer.destroy();
  querySet.destroy();
  device.destroy();
  console.log("\nDone.");
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
