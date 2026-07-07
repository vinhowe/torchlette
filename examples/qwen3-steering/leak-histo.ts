/**
 * #79 leak-histogram probe: run N generations via generateChat and diff the
 * REACHABLE storage set at each generation boundary, histogramming the
 * per-generation delta by shape:dtype:view. Identifies WHAT leaks.
 *
 * Run SOLO: npx tsx examples/qwen3-steering/leak-histo.ts
 *           TORCHLETTE_STEP_TAPE=1 npx tsx examples/qwen3-steering/leak-histo.ts
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

type Live = ReturnType<typeof storageTracker.debugLiveSet>;

function keyOf(e: Live[number]): string {
  return `${e.view ? "V" : "O"} [${e.shape.join(",")}] ${e.dtype}`;
}

function diff(prev: Live, cur: Live) {
  const prevIds = new Set(prev.map((e) => e.id));
  const added = cur.filter((e) => !prevIds.has(e.id));
  const histo = new Map<string, number>();
  for (const e of added) histo.set(keyOf(e), (histo.get(keyOf(e)) ?? 0) + 1);
  return { addedCount: added.length, histo };
}

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

  let prev: Live | null = null;
  for (let g = 1; g <= 6; g++) {
    const stats = await generateChat(
      api, model, tok,
      [{ role: "user", content: "Tell me about your day." }],
      {},
      { maxNewTokens: 32, temperature: 0.7, topK: 1, topP: 1 },
    );
    const cur = storageTracker.debugLiveSet();
    const s = storageTracker.stats();
    console.log(
      `\ngen ${g}: ${stats.tokPerSec} tok/s | reachable=${s.reachableStorages} total=${s.totalStorages} heapMB ${(process.memoryUsage().heapUsed / 1e6).toFixed(0)}`,
    );
    if (prev) {
      const { addedCount, histo } = diff(prev, cur);
      console.log(`  Δreachable added-since-prev=${addedCount}`);
      const sorted = [...histo.entries()].sort((a, b) => b[1] - a[1]);
      for (const [k, n] of sorted.slice(0, 25)) console.log(`    ${n.toString().padStart(5)}  ${k}`);
    }
    prev = cur;
  }
  process.exit(0);
}
main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
