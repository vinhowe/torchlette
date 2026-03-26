/**
 * DiLoCo Coordinator
 *
 * Spawns N agent processes and routes pseudo-gradients between them.
 * Each agent trains independently and sends pseudo-gradients via stdout.
 * The coordinator averages them and sends the result back via stdin.
 *
 * Usage:
 *   npx tsx tools/diloco-coordinator.ts [--agents 2] [--rounds 5] [--steps 50]
 */

import { type ChildProcess, spawn } from "node:child_process";
import * as readline from "node:readline";

const NUM_AGENTS = parseInt(process.env.AGENTS ?? "2", 10);
const ROUNDS = parseInt(process.env.ROUNDS ?? "5", 10);
const STEPS = parseInt(process.env.STEPS ?? "50", 10);
const SEED = parseInt(process.env.SEED ?? "42", 10);

interface AgentState {
  proc: ChildProcess;
  id: string;
  rl: readline.Interface;
  ready: boolean;
}

async function main() {
  console.log(`=== DiLoCo Coordinator ===`);
  console.log(
    `Agents: ${NUM_AGENTS}, Rounds: ${ROUNDS}, Steps: ${STEPS}, Seed: ${SEED}\n`,
  );

  const agents: AgentState[] = [];

  // Spawn agents
  for (let i = 0; i < NUM_AGENTS; i++) {
    const proc = spawn("npx", ["tsx", "tools/diloco-agent.ts"], {
      env: {
        ...process.env,
        SEED: String(SEED),
        STEPS: String(STEPS),
        ROUNDS: String(ROUNDS),
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    const rl = readline.createInterface({ input: proc.stdout! });
    const id = `agent-${i}`;

    // Forward agent stderr to our stderr with prefix
    proc.stderr!.on("data", (chunk: Buffer) => {
      process.stderr.write(chunk);
    });

    agents.push({ proc, id, rl, ready: false });
  }

  // Wait for all agents to signal ready
  console.log("Waiting for agents to initialize...");
  await Promise.all(
    agents.map(
      (agent) =>
        new Promise<void>((resolve) => {
          agent.rl.on("line", (line) => {
            try {
              const msg = JSON.parse(line);
              if (msg.type === "ready") {
                agent.ready = true;
                resolve();
              }
            } catch {}
          });
        }),
    ),
  );
  console.log("All agents ready.\n");

  // Send start signal
  for (let i = 0; i < agents.length; i++) {
    const startMsg = JSON.stringify({
      type: "start",
      agentId: agents[i].id,
      numAgents: NUM_AGENTS,
    });
    agents[i].proc.stdin!.write(startMsg + "\n");
  }

  // Main coordination loop: collect pseudo-gradients, average, distribute
  for (let round = 0; round < ROUNDS; round++) {
    // Collect pseudo-gradients from all agents for this round
    const allGrads = await Promise.all(
      agents.map(
        (agent) =>
          new Promise<{ agentId: string; data: number[][] }>((resolve) => {
            const handler = (line: string) => {
              try {
                const msg = JSON.parse(line);
                if (msg.type === "pseudograd" && msg.round === round) {
                  agent.rl.removeListener("line", handler);
                  resolve({ agentId: msg.agentId, data: msg.data });
                }
              } catch {}
            };
            agent.rl.on("line", handler);
          }),
      ),
    );

    // Average pseudo-gradients
    const numParams = allGrads[0].data.length;
    const averaged: number[][] = [];
    for (let p = 0; p < numParams; p++) {
      const size = allGrads[0].data[p].length;
      const avg = new Array(size).fill(0);
      for (const ag of allGrads) {
        for (let j = 0; j < size; j++) {
          avg[j] += ag.data[p][j] / NUM_AGENTS;
        }
      }
      averaged.push(avg);
    }

    // Send averaged result back to all agents
    for (const agent of agents) {
      const msg = JSON.stringify({ type: "averaged", round, data: averaged });
      agent.proc.stdin!.write(msg + "\n");
    }

    console.log(`Round ${round}: all agents synced`);
  }

  // Wait for agents to exit
  console.log("\nWaiting for agents to finish...");
  await Promise.all(
    agents.map(
      (agent) =>
        new Promise<void>((resolve) => {
          agent.proc.on("exit", () => resolve());
        }),
    ),
  );

  console.log("\n=== All agents finished ===");
}

main().catch((e) => {
  console.error("Coordinator error:", e);
  process.exit(1);
});
