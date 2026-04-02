/**
 * DiLoCo Coordinator
 *
 * Spawns N agent processes on separate GPUs using MESA_VK_DEVICE_SELECT.
 * Agents sync via shared filesystem.
 *
 * Usage:
 *   AGENTS=8 ROUNDS=10 STEPS=100 npx tsx tools/diloco-coordinator.ts
 */

import { execSync, spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";

const NUM_AGENTS = parseInt(process.env.AGENTS ?? "8", 10);
const ROUNDS = parseInt(process.env.ROUNDS ?? "10", 10);
const STEPS = parseInt(process.env.STEPS ?? "50", 10);
const SEED = parseInt(process.env.SEED ?? "42", 10);
const LR = process.env.LR ?? "1e-4";
const OUTER_LR = process.env.OUTER_LR ?? "0.7";
const OUTER_MU = process.env.OUTER_MU ?? "0.9";
const SEQ_LEN = process.env.SEQ_LEN ?? "512";
const BATCH_SIZE = process.env.BATCH_SIZE ?? "4";
const ACCUM_STEPS = process.env.ACCUM_STEPS ?? "2";
const SYNC_DIR = `/tmp/diloco-${Date.now()}`;

/** Get PCI bus addresses of all GPUs via Mesa Vulkan layer. */
function getGPUAddresses(): string[] {
  try {
    const output = execSync(
      "MESA_VK_DEVICE_SELECT=list npx tsx -e 'import(\"webgpu\").then(m => m.create([]).requestAdapter())' 2>&1",
      { timeout: 30000 },
    ).toString();
    const addresses: string[] = [];
    for (const line of output.split("\n")) {
      const match = line.match(/GPU \d+:.*discrete GPU ([\da-f:.]+)/);
      if (match) addresses.push(match[1]);
    }
    return addresses;
  } catch {
    return [];
  }
}

/** Find GPUs with enough free memory (>20GB). */
function findFreeGPUs(): number[] {
  try {
    const output = execSync(
      "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader",
    ).toString();
    const free: number[] = [];
    for (const line of output.trim().split("\n")) {
      const [idx, mem] = line.split(",").map((s) => s.trim());
      if (parseInt(mem) > 20000) free.push(parseInt(idx));
    }
    return free;
  } catch {
    return [];
  }
}

async function main() {
  console.log("=== DiLoCo Coordinator ===");
  console.log(
    `Agents: ${NUM_AGENTS}, Rounds: ${ROUNDS}, Steps: ${STEPS}, Seed: ${SEED}`,
  );

  const gpuAddresses = getGPUAddresses();
  const freeGPUs = findFreeGPUs();
  console.log(
    `Available GPUs: ${gpuAddresses.length}, Free (>20GB): ${freeGPUs.length}`,
  );

  if (freeGPUs.length < NUM_AGENTS) {
    console.error(
      `Not enough free GPUs: need ${NUM_AGENTS}, have ${freeGPUs.length}`,
    );
    process.exit(1);
  }

  // Select GPUs for agents
  const selectedGPUs = freeGPUs.slice(0, NUM_AGENTS);
  console.log(`Using GPUs: ${selectedGPUs.join(", ")}`);
  console.log(`Sync dir: ${SYNC_DIR}\n`);

  fs.mkdirSync(SYNC_DIR, { recursive: true });

  // Spawn agents — one per GPU using Vulkan device filter shim
  const vkShimPath = path.join(process.cwd(), "tools", "vk-shim");
  const hasVkShim = fs.existsSync(path.join(vkShimPath, "libvulkan.so.1"));
  if (!hasVkShim) {
    console.warn(
      "WARNING: Vulkan shim not found at tools/vk-shim/libvulkan.so.1",
    );
    console.warn(
      "Multi-GPU may not work. Build with: cd tools && gcc -shared -fPIC -o vk-shim/libvulkan.so.1 vk-device-filter.c -ldl",
    );
  }

  const procs = [];
  for (let i = 0; i < NUM_AGENTS; i++) {
    const gpuIdx = selectedGPUs[i];
    const ldPath = hasVkShim
      ? `${vkShimPath}:${process.env.LD_LIBRARY_PATH ?? ""}`
      : (process.env.LD_LIBRARY_PATH ?? "");
    const proc = spawn("npx", ["tsx", "tools/diloco-agent.ts"], {
      env: {
        ...process.env,
        SEED: String(SEED),
        STEPS: String(STEPS),
        ROUNDS: String(ROUNDS),
        LR,
        OUTER_LR,
        OUTER_MU,
        SEQ_LEN,
        BATCH_SIZE,
        ACCUM_STEPS,
        AGENT_ID: String(i),
        NUM_AGENTS: String(NUM_AGENTS),
        SYNC_DIR,
        TORCHLETTE_LIVENESS_RELEASE: "1",
        TORCHLETTE_POOL_BUDGET_MB: "2000",
        // Pin to specific GPU via Vulkan device filter shim
        VULKAN_DEVICE_INDEX: String(gpuIdx),
        LD_LIBRARY_PATH: ldPath,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    // Prefix agent output
    proc.stdout?.on("data", (d: Buffer) =>
      process.stdout.write(`[agent-${i}] ${d}`),
    );
    proc.stderr?.on("data", (d: Buffer) => process.stderr.write(`${d}`));
    procs.push(proc);
    console.log(`  Agent ${i} → GPU ${gpuIdx}`);
  }

  // Wait for all to exit
  const codes = await Promise.all(
    procs.map(
      (proc, i) =>
        new Promise<number>((resolve) => {
          proc.on("exit", (code) => {
            console.log(`Agent ${i} exited (code ${code ?? "?"})`);
            resolve(code ?? 1);
          });
        }),
    ),
  );

  // Cleanup
  try {
    // Don't clean up — keep checkpoint for evaluation
    console.log(`\nCheckpoint dir: ${SYNC_DIR}/checkpoint`);
  } catch {}

  const allOk = codes.every((c) => c === 0);
  console.log(`\n=== ${allOk ? "SUCCESS" : "SOME FAILURES"} ===`);
  process.exit(allOk ? 0 : 1);
}

main().catch((e) => {
  console.error("Coordinator error:", e);
  process.exit(1);
});
