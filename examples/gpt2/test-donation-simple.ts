/**
 * Simple test to verify memory donation works in executeWithMemoryPlanning.
 *
 * Uses RuntimeEngine directly (not Torchlette frontend) to ensure
 * memory planning is active during force().
 */
import { initWebGPU, getBufferPoolStats, clearBufferPool } from "../../src/backend/webgpu";
import { RuntimeEngine } from "../../src/runtime/engine";
import { getMemoryPlannerStats } from "../../src/engine/memory-planned-executor";

async function main() {
  console.log("Initializing WebGPU...");
  const success = await initWebGPU();
  if (!success) {
    console.error("Failed to initialize WebGPU");
    process.exit(1);
  }

  clearBufferPool();

  // Create RuntimeEngine directly with memory planning enabled
  // This ensures memory planning is active during force()
  const engine = new RuntimeEngine("webgpu", {
    enableMemoryPlanning: true,
    enableDonation: true,
    trackStats: true,
    enableFusion: false,
  });

  console.log("\n=== Testing memory donation in executeWithMemoryPlanning ===");

  // Create a simple computation: a + b -> c, then relu(c) -> d
  // a and b are dead after add, so their buffers can be donated to relu output
  const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2], "webgpu");
  const b = engine.tensorFromArray([5, 6, 7, 8], [2, 2], "webgpu");
  const c = engine.add(a, b);  // c = a + b = [6, 8, 10, 12]
  const d = engine.relu(c);    // d = relu(c) - a or b's buffer can be donated here

  // Force execution using engine.cpu()
  const result = await engine.cpu(d);
  console.log("Result of relu(a+b):", result);  // Should be [6, 8, 10, 12]

  // Get memory stats from engine
  const memStats = engine.getLastMemoryStats();
  console.log("\nMemory Planning Stats:");
  if (memStats) {
    console.log(`  - Total nodes: ${memStats.totalNodes}`);
    console.log(`  - Donation count: ${memStats.donationCount}`);
    if (memStats.donationCount > 0) {
      console.log("\n✓ Memory donation is working!");
    } else {
      console.log("\n✗ No donations occurred");
      process.exit(1);
    }
  } else {
    console.log("  No memory stats available");
    process.exit(1);
  }

  // Cleanup
  a.dispose();
  b.dispose();
  c.dispose();
  d.dispose();

  console.log("\n✓ Test complete");
  process.exit(0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
