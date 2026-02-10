/**
 * Test GPU buffer pool reuse with compute ops.
 *
 * This test creates tensors, performs compute ops, disposes the outputs,
 * and verifies that the pool is actually reusing GPU buffers.
 */
import { initWebGPU, getBufferPoolStats, getGPUMemoryStats, clearBufferPool, syncWebGPU } from "../../src/backend/webgpu";
import { webgpuBackend } from "../../src/backend/webgpu";

async function main() {
  console.log("Initializing WebGPU...");
  const success = await initWebGPU();
  if (!success) {
    console.error("Failed to initialize WebGPU");
    process.exit(1);
  }

  // Clear pool to start fresh
  clearBufferPool();

  console.log("\n=== Phase 1: Create compute outputs and dispose them ===");
  console.log("Initial pool stats:", getBufferPoolStats());

  // Create input tensors (these use mappedAtCreation, won't be pooled)
  const a = webgpuBackend.ops.tensorFromArray(
    Array.from({ length: 1024 }, (_, i) => i),
    [32, 32]
  );
  const b = webgpuBackend.ops.tensorFromArray(
    Array.from({ length: 1024 }, (_, i) => i * 2),
    [32, 32]
  );

  // Perform compute ops - outputs WILL be pooled
  const outputs = [];
  for (let i = 0; i < 5; i++) {
    const c = webgpuBackend.ops.add(a, b);
    outputs.push(c);
  }

  console.log(`Created ${outputs.length} compute outputs`);
  console.log("Pool stats after compute:", getBufferPoolStats());

  // Dispose outputs - they should go to the pending queue
  for (const output of outputs) {
    output.destroy();
  }

  console.log("Pool stats after dispose:", getBufferPoolStats());

  // Sync to allow fence to complete
  await syncWebGPU();

  // Give the fence promise time to resolve
  await new Promise(r => setTimeout(r, 50));

  console.log("Pool stats after sync:", getBufferPoolStats());

  console.log("\n=== Phase 2: Create MORE compute outputs (should reuse from pool) ===");

  const poolBefore = getBufferPoolStats();
  console.log("Pool before phase 2:", poolBefore);

  const outputs2 = [];
  for (let i = 0; i < 5; i++) {
    const c = webgpuBackend.ops.add(a, b);
    outputs2.push(c);
  }

  const poolAfter = getBufferPoolStats();
  console.log("Pool after phase 2:", poolAfter);

  // Verify reuse
  const reusedInPhase2 = poolAfter.reuseCount - poolBefore.reuseCount;
  console.log(`\nReused ${reusedInPhase2} buffers in phase 2`);

  if (reusedInPhase2 > 0) {
    console.log("\n✓ Buffer reuse is working!");
    console.log(`  Reuse rate: ${(poolAfter.reuseRate * 100).toFixed(1)}%`);
  } else {
    console.log("\n✗ No buffer reuse detected");
  }

  // Cleanup
  for (const output of outputs2) {
    output.destroy();
  }
  a.destroy();
  b.destroy();

  console.log("\nFinal stats:");
  console.log("  Pool:", getBufferPoolStats());
  console.log("  GPU memory:", getGPUMemoryStats());

  process.exit(0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
