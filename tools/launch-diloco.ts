/**
 * DiLoCo Multi-GPU Launcher
 *
 * Launches N diloco-agent processes, each on a separate GPU.
 * Uses a Vulkan device filter (LD_LIBRARY_PATH shim) to direct Dawn
 * to the correct GPU, since Dawn ignores CUDA_VISIBLE_DEVICES.
 *
 * Usage:
 *   npx tsx tools/launch-diloco.ts [--agents N] [--gpus 4,5,6,7] [-- agent args...]
 *
 * Examples:
 *   npx tsx tools/launch-diloco.ts --agents 4 --gpus 4,5,6,7
 *   npx tsx tools/launch-diloco.ts --agents 8 --gpus 4,5,6,7,8,9,10,11 -- STEPS=50 ROUNDS=5
 *   npx tsx tools/launch-diloco.ts --agents 2  # auto-selects free GPUs
 */
import { execSync, spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";

const PROJECT_DIR = path.resolve(import.meta.dirname, "..");
const VK_SHIM_DIR = path.join(PROJECT_DIR, "tools", "vk-shim");
const VK_SHIM_LIB = path.join(VK_SHIM_DIR, "libvulkan.so.1");
const AGENT_SCRIPT = path.join(PROJECT_DIR, "tools", "diloco-agent.ts");

function parseArgs(): {
  numAgents: number;
  gpuIds: number[];
  agentEnv: Record<string, string>;
} {
  const args = process.argv.slice(2);
  let numAgents = 4;
  let gpuIds: number[] = [];
  const agentEnv: Record<string, string> = {};

  let i = 0;
  while (i < args.length) {
    if (args[i] === "--agents" && i + 1 < args.length) {
      numAgents = parseInt(args[++i], 10);
    } else if (args[i] === "--gpus" && i + 1 < args.length) {
      gpuIds = args[++i].split(",").map((s) => parseInt(s.trim(), 10));
    } else if (args[i] === "--") {
      // Remaining args are KEY=VALUE pairs for agent env
      for (let j = i + 1; j < args.length; j++) {
        const [k, ...v] = args[j].split("=");
        agentEnv[k] = v.join("=");
      }
      break;
    }
    i++;
  }

  return { numAgents, gpuIds, agentEnv };
}

function findFreeGPUs(count: number): number[] {
  const output = execSync(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader",
  )
    .toString()
    .trim();

  const freeGPUs: number[] = [];
  for (const line of output.split("\n")) {
    const [idxStr, memStr] = line.split(",").map((s) => s.trim());
    const idx = parseInt(idxStr, 10);
    const memMiB = parseInt(memStr, 10);
    // Consider a GPU "free" if it has < 100 MiB used
    if (memMiB < 100) freeGPUs.push(idx);
  }

  if (freeGPUs.length < count) {
    console.error(
      `ERROR: Need ${count} free GPUs but only found ${freeGPUs.length}: [${freeGPUs.join(",")}]`,
    );
    console.error("Use --gpus to specify GPU indices explicitly.");
    process.exit(1);
  }

  return freeGPUs.slice(0, count);
}

function ensureVkShim(): void {
  if (fs.existsSync(VK_SHIM_LIB)) return;

  const srcFile = path.join(PROJECT_DIR, "tools", "vk-device-filter.c");
  if (!fs.existsSync(srcFile)) {
    console.error(`ERROR: Vulkan device filter source not found: ${srcFile}`);
    process.exit(1);
  }

  console.log("Building Vulkan device filter shim...");
  fs.mkdirSync(VK_SHIM_DIR, { recursive: true });
  execSync(`gcc -shared -fPIC -o ${VK_SHIM_LIB} ${srcFile} -ldl`, {
    stdio: "inherit",
  });
  console.log(`Built: ${VK_SHIM_LIB}\n`);
}

async function main() {
  const { numAgents, gpuIds: requestedGPUs, agentEnv } = parseArgs();

  ensureVkShim();

  const gpuIds =
    requestedGPUs.length >= numAgents
      ? requestedGPUs.slice(0, numAgents)
      : findFreeGPUs(numAgents);

  const syncDir = agentEnv.SYNC_DIR ?? `/tmp/diloco-${Date.now()}`;
  fs.mkdirSync(syncDir, { recursive: true });

  console.log(`=== DiLoCo Multi-GPU Launch ===`);
  console.log(`Agents:   ${numAgents}`);
  console.log(`GPUs:     [${gpuIds.join(", ")}]`);
  console.log(`Sync dir: ${syncDir}`);
  console.log(`Agent env: ${JSON.stringify(agentEnv)}\n`);

  // Show GPU status
  console.log("GPU memory before launch:");
  const memBefore = execSync(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader",
  )
    .toString()
    .trim();
  for (const line of memBefore.split("\n")) {
    const [idx] = line.split(",").map((s) => s.trim());
    if (gpuIds.includes(parseInt(idx, 10))) {
      console.log(`  GPU ${line.trim()}`);
    }
  }
  console.log();

  // Launch agents
  const children: ReturnType<typeof spawn>[] = [];
  const results: Promise<{
    agentId: number;
    gpuId: number;
    code: number;
    lastLines: string[];
  }>[] = [];

  for (let i = 0; i < numAgents; i++) {
    const gpuId = gpuIds[i];
    const child = spawn("npx", ["tsx", AGENT_SCRIPT], {
      cwd: PROJECT_DIR,
      env: {
        ...process.env,
        AGENT_ID: String(i),
        NUM_AGENTS: String(numAgents),
        VULKAN_DEVICE_INDEX: String(gpuId),
        LD_LIBRARY_PATH: `${VK_SHIM_DIR}:${process.env.LD_LIBRARY_PATH ?? ""}`,
        SYNC_DIR: syncDir,
        ...agentEnv,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    children.push(child);

    // Pipe stderr (where logs go) to parent stderr with prefix
    child.stderr.on("data", (d: Buffer) => {
      process.stderr.write(d);
    });

    results.push(
      new Promise((resolve) => {
        const stderrLines: string[] = [];
        child.stderr.on("data", (d: Buffer) => {
          for (const line of d.toString().split("\n")) {
            if (line.trim()) stderrLines.push(line);
            // Keep only last 20 lines
            if (stderrLines.length > 20) stderrLines.shift();
          }
        });
        child.on("close", (code) =>
          resolve({
            agentId: i,
            gpuId,
            code: code ?? -1,
            lastLines: stderrLines.slice(-5),
          }),
        );
      }),
    );
  }

  // Handle Ctrl+C gracefully
  process.on("SIGINT", () => {
    console.log("\nStopping agents...");
    for (const child of children) child.kill("SIGTERM");
    setTimeout(() => {
      for (const child of children) child.kill("SIGKILL");
      process.exit(1);
    }, 5000);
  });

  const allResults = await Promise.all(results);

  console.log("\n=== Summary ===");
  for (const r of allResults) {
    const status = r.code === 0 ? "OK" : `FAILED (exit ${r.code})`;
    console.log(`Agent ${r.agentId} (GPU ${r.gpuId}): ${status}`);
  }

  const failed = allResults.filter((r) => r.code !== 0);
  if (failed.length > 0) {
    console.error(`\n${failed.length} agent(s) failed.`);
    process.exit(1);
  }

  console.log("\nAll agents completed successfully.");
  process.exit(0);
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(1);
});
