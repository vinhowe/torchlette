/**
 * DiLoCo Node Agent
 *
 * A headless training worker that participates in distributed pretraining
 * via the gossip network. Runs on Node.js/Dawn (V100, A100, etc.).
 *
 * Usage:
 *   npx tsx tools/diloco-agent.ts [--seed 42] [--steps 500] [--rounds 10]
 *
 * Multiple agents on the same or different machines find each other via
 * the PeerJS cloud signaling server and exchange pseudo-gradients over
 * WebRTC data channels.
 */

import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import {
  e3m0Dequantize,
  e3m0Quantize,
  NesterovOuterOptimizer,
} from "../src/distributed";
import { Torchlette } from "../src/frontend/torchlette";
import { normal_ } from "../src/nn/init";
import { Adam } from "../src/optim";

// ============================================================================
// Config
// ============================================================================

const SEED = parseInt(process.env.SEED ?? "42", 10);
const INNER_STEPS = parseInt(process.env.STEPS ?? "50", 10);
const OUTER_ROUNDS = parseInt(process.env.ROUNDS ?? "10", 10);
const LR = parseFloat(process.env.LR ?? "1e-3");
const OUTER_LR = parseFloat(process.env.OUTER_LR ?? "1.0");
const OUTER_MU = parseFloat(process.env.OUTER_MU ?? "0.0");
const SEQ_LEN = 32;

// Tiny GPT-2 for testing coordination (not real pretraining)
const TINY_CONFIG = {
  vocabSize: 256,
  blockSize: 64,
  numLayers: 2,
  numHeads: 2,
  embedDim: 64,
  dropoutRate: 0,
};

// ============================================================================
// Gossip (simplified for Node — no PeerJS, use stdin/stdout IPC)
// ============================================================================

/**
 * For local testing, agents communicate via a coordinator process that
 * pipes messages between them. Each agent reads JSON lines from stdin
 * and writes JSON lines to stdout.
 *
 * Protocol:
 *   Agent → Coordinator: { type: "ready", agentId: string }
 *   Agent → Coordinator: { type: "pseudograd", agentId: string, round: number, data: number[][] }
 *   Coordinator → Agent: { type: "averaged", round: number, data: number[][] }
 *   Coordinator → Agent: { type: "start", agentId: string, numAgents: number }
 */

let agentId = `agent-${process.pid}`;
const pendingAverages = new Map<number, (grads: Float32Array[]) => void>();

function sendMessage(msg: object): void {
  process.stdout.write(JSON.stringify(msg) + "\n");
}

function setupIPC(): void {
  let buf = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk: string) => {
    buf += chunk;
    const lines = buf.split("\n");
    buf = lines.pop()!;
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line);
        handleMessage(msg);
      } catch {}
    }
  });
}

function handleMessage(msg: any): void {
  if (msg.type === "start") {
    agentId = msg.agentId;
    console.error(`[${agentId}] Started, ${msg.numAgents} agents in swarm`);
  } else if (msg.type === "averaged") {
    const resolve = pendingAverages.get(msg.round);
    if (resolve) {
      const grads = msg.data.map((arr: number[]) => new Float32Array(arr));
      resolve(grads);
      pendingAverages.delete(msg.round);
    }
  }
}

/** Send pseudo-gradients and wait for averaged result. */
function exchangePseudoGrads(
  pseudoGrads: Float32Array[],
  round: number,
): Promise<Float32Array[]> {
  return new Promise((resolve) => {
    pendingAverages.set(round, resolve);
    sendMessage({
      type: "pseudograd",
      agentId,
      round,
      data: pseudoGrads.map((pg) => Array.from(pg)),
    });
  });
}

// ============================================================================
// Training
// ============================================================================

/** Simple repeating text data (byte-level). */
function generateData(seed: number): number[] {
  const texts = [
    "the cat sat on the mat. the dog ran in the park. ",
    "once upon a time in a land far away there lived a king. ",
    "the quick brown fox jumps over the lazy dog again and again. ",
    "to be or not to be that is the question. all the world is a stage. ",
  ];
  // Each agent gets a different text based on seed
  const text = texts[seed % texts.length];
  const data: number[] = [];
  for (let i = 0; i < 2000; i++) {
    data.push(text.charCodeAt(i % text.length));
  }
  return data;
}

async function trainStep(
  api: Torchlette,
  model: any,
  optimizer: Adam,
  tokens: number[],
  offset: number,
): Promise<number> {
  const start = offset % (tokens.length - SEQ_LEN - 1);
  const inputData = tokens.slice(start, start + SEQ_LEN);
  const targetData = tokens.slice(start + 1, start + SEQ_LEN + 1);

  await api.beginStep();
  const input = api.tensorFromArray(inputData, [1, SEQ_LEN], {
    device: "webgpu",
  });
  const target = api.tensorFromArray(targetData, [1, SEQ_LEN], {
    device: "webgpu",
  });

  const loss = api.tidy(() => {
    const { loss: l } = model.forwardWithLoss(input, target);
    api.keep(l);
    return l;
  });

  const lossVal = await loss.item();
  await loss.backward();
  optimizer.step();
  optimizer.zeroGrad();

  input.dispose();
  target.dispose();
  api.endStep();
  await api.markStep();

  return lossVal;
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  setupIPC();

  const ok = await initWebGPU();
  if (!ok) {
    console.error("WebGPU not available");
    process.exit(1);
  }

  const api = new Torchlette("webgpu", { enableFusion: true });

  // All agents use the same seed for identical starting weights
  api.manualSeed(SEED);

  // Create tiny model
  const { GPT2WithLoRA } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora"
  );
  const loraConfig = { rank: 4, alpha: 4 };
  const model = new GPT2WithLoRA(api, TINY_CONFIG, loraConfig, "webgpu");
  model.train(true);
  const params = model.getLoRAParameters();

  // Each agent gets different training data (use PID for uniqueness before ID assignment)
  const tokens = generateData(process.pid);

  const outerOpt = new NesterovOuterOptimizer(api, {
    lr: OUTER_LR,
    momentum: OUTER_MU,
  });

  // Signal ready
  sendMessage({ type: "ready", agentId });

  // Wait for start signal
  await new Promise<void>((resolve) => {
    const check = () => {
      if (agentId.startsWith("agent-") && agentId !== `agent-${process.pid}`) {
        // Got reassigned by coordinator
      }
      // Check periodically — start signal comes via handleMessage
      const origHandler = handleMessage;
      const wrapped = (msg: any) => {
        origHandler(msg);
        if (msg.type === "start") resolve();
      };
      // Replace handler temporarily
      (globalThis as any).__handleMsg = wrapped;
    };
    // Actually, just wait for stdin data
    const onData = () => {
      // Start signal will be handled by handleMessage which was already set up
      // Just resolve after a short delay to let the message process
      setTimeout(resolve, 100);
    };
    process.stdin.once("data", onData);
  });

  console.error(
    `[${agentId}] Training: ${OUTER_ROUNDS} rounds × ${INNER_STEPS} steps`,
  );

  for (let round = 0; round < OUTER_ROUNDS; round++) {
    // Snapshot global params
    const globalSnapshot: Float32Array[] = [];
    for (const p of params) {
      globalSnapshot.push(new Float32Array(await p.cpu()));
    }

    // Inner training loop
    const innerOpt = new Adam(params, { lr: LR }, api);
    const losses: number[] = [];
    for (let step = 0; step < INNER_STEPS; step++) {
      const offset = (round * INNER_STEPS + step) * SEQ_LEN;
      const l = await trainStep(api, model, innerOpt, tokens, offset);
      losses.push(l);
    }
    const avgLoss = losses.reduce((a, b) => a + b, 0) / losses.length;

    // Compute pseudo-gradients
    const pseudoGrads: Float32Array[] = [];
    for (let i = 0; i < params.length; i++) {
      const localData = await params[i].cpu();
      const delta = new Float32Array(localData.length);
      for (let j = 0; j < delta.length; j++) {
        delta[j] = localData[j] - globalSnapshot[i][j];
      }
      pseudoGrads.push(delta);
    }

    // Exchange with other agents and get averaged result
    const avgPseudoGrads = await exchangePseudoGrads(pseudoGrads, round);

    // Restore to snapshot + apply outer update
    await api.beginStep();
    for (let i = 0; i < params.length; i++) {
      const updated = new Float32Array(globalSnapshot[i].length);
      for (let j = 0; j < updated.length; j++) {
        updated[j] = globalSnapshot[i][j] + avgPseudoGrads[i][j];
      }
      const t = api.tensorFromArray(Array.from(updated), params[i].shape, {
        device: "webgpu",
      });
      api.copy_(params[i], t);
    }
    api.endStep();
    await api.markStep();

    console.error(`[${agentId}] round ${round}: loss=${avgLoss.toFixed(4)}`);
  }

  console.error(`[${agentId}] Done`);
  outerOpt.dispose();
  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(`[${agentId}] FATAL:`, e);
  process.exit(1);
});
