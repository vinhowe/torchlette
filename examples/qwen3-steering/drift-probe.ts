/**
 * #79 drift probe: is the cross-generation slowdown a ONE-TIME step or
 * MONOTONIC accumulation? Runs 6 generations in one process via generateChat
 * (the capture-based path), reporting per-generation steady tok/s + phase
 * breakdown + storage/tape counters between generations.
 *
 * Run SOLO: TORCHLETTE_STEP_TAPE=1 npx tsx examples/qwen3/drift-probe.ts
 */
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { storageTracker } from "../../src/graph/storage-tracker";
import { loadPretrainedQwen3 } from "../qwen3/loader";
import { AutoTokenizer } from "@huggingface/transformers";
import { generateChat } from "../../packages/qwen3-browser/src/generate";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_DIR);
  const tok = {
    encode: (t: string) => tokenizer.encode(t) as number[],
    decode: (ids: number[], o?: { skip_special_tokens?: boolean }) =>
      tokenizer.decode(ids, o) as string,
  };
  const model = await loadPretrainedQwen3(api, MODEL_DIR, { maxSeqLen: 512 });

  for (let g = 1; g <= 6; g++) {
    const stats = await generateChat(
      api, model, tok,
      [{ role: "user", content: "Tell me about your day." }],
      {},
      { maxNewTokens: 32, temperature: 0.7, topK: 1, topP: 1 }, // greedy: deterministic work
    );
    const s = storageTracker.stats();
    const d = stats.decodeBreakdown;
    console.log(
      `gen ${g}: ${stats.tokPerSec} tok/s | prefill ${stats.prefillMs}ms | ` +
      (d ? `b·l·f·s·m ${d.buildMs}·${d.lowerMs}·${d.fenceMs}·${d.sampleMs}·${d.stepMs} | ` : "") +
      `storages total=${s.totalStorages} reachable=${(s as Record<string, unknown>).reachableStorages ?? "?"} | heapMB ${(process.memoryUsage().heapUsed / 1e6).toFixed(0)}`,
    );
  }
  process.exit(0);
}
main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
