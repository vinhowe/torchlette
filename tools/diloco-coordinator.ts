/**
 * DiLoCo Coordinator
 *
 * Spawns N agent processes that sync via shared filesystem.
 * Each agent writes E3M0-compressed pseudo-gradients to files,
 * reads all peers' files, averages, and applies the outer update.
 * The coordinator just launches processes and waits.
 *
 * Usage:
 *   npx tsx tools/diloco-coordinator.ts
 *
 * Environment:
 *   AGENTS=2     Number of agents to spawn
 *   ROUNDS=5     DiLoCo sync rounds
 *   STEPS=50     Inner training steps per round
 *   SEED=42      Global RNG seed
 */

import { spawn } from "node:child_process";
import * as fs from "node:fs";

const NUM_AGENTS = parseInt(process.env.AGENTS ?? "2", 10);
const ROUNDS = parseInt(process.env.ROUNDS ?? "2", 10);
const STEPS = parseInt(process.env.STEPS ?? "5", 10);
const SEED = parseInt(process.env.SEED ?? "42", 10);
const SYNC_DIR = `/tmp/diloco-${Date.now()}`;

async function main() {
  console.log("=== DiLoCo Coordinator ===");
  console.log(
    `Agents: ${NUM_AGENTS}, Rounds: ${ROUNDS}, Steps: ${STEPS}, Seed: ${SEED}`,
  );
  console.log(`Sync dir: ${SYNC_DIR}\n`);

  // Clean sync directory
  fs.mkdirSync(SYNC_DIR, { recursive: true });

  // Spawn agents
  const procs = [];
  for (let i = 0; i < NUM_AGENTS; i++) {
    const proc = spawn("npx", ["tsx", "tools/diloco-agent.ts"], {
      env: {
        ...process.env,
        SEED: String(SEED),
        STEPS: String(STEPS),
        ROUNDS: String(ROUNDS),
        AGENT_ID: String(i),
        NUM_AGENTS: String(NUM_AGENTS),
        SYNC_DIR,
      },
      stdio: ["ignore", "ignore", "inherit"],
    });
    procs.push(proc);
  }

  // Wait for all to exit
  await Promise.all(
    procs.map(
      (proc) =>
        new Promise<number>((resolve) => {
          proc.on("exit", (code) => resolve(code ?? 1));
        }),
    ),
  );

  // Cleanup sync files
  try {
    fs.rmSync(SYNC_DIR, { recursive: true });
  } catch {}

  console.log("\n=== All agents finished ===");
}

main().catch((e) => {
  console.error("Coordinator error:", e);
  process.exit(1);
});
