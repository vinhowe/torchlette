/**
 * THE gate the browser bug proved was missing: generateChat itself — the
 * shipped reference consumer — exercised under the step-tape on Node, with
 * tape HITS asserted. (kv-differential and taped-decode drive capture
 * directly; generateChat's loop was never tape-gated until the browser hit
 * the endStep-remnant comparator-reset bug.)
 *
 * Run SOLO: TORCHLETTE_STEP_TAPE=1 npx tsx examples/qwen3-steering/gen-tape-gate.ts
 */
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
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

  let last: NonNullable<Awaited<ReturnType<typeof generateChat>>["tape"]>;
  for (let g = 1; g <= 2; g++) {
    const stats = await generateChat(
      api, model, tok,
      [{ role: "user", content: "Tell me about your day." }],
      {},
      { maxNewTokens: 32, temperature: 0.7, topK: 1, topP: 1 },
    );
    last = stats.tape!;
    console.log(`gen ${g}: hits ${last.hits}/${last.calls} ready=${last.ready}`,
      JSON.stringify(last.recorder));
  }
  const r = last!.recorder!;
  const reasons = (r as unknown as { boundaryReasons?: Record<string, number> }).boundaryReasons ?? {};
  const ok2 =
    last!.hits >= Math.floor(last!.calls * 0.7) &&
    last!.ready &&
    !("endStep" in reasons) &&
    !("beginStep" in reasons);
  console.log(ok2 ? "GEN-TAPE GATE PASS" : `GEN-TAPE GATE FAIL: hits=${last!.hits}/${last!.calls} ready=${last!.ready} reasons=${JSON.stringify(reasons)}`);
  process.exit(ok2 ? 0 : 1);
}
main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
