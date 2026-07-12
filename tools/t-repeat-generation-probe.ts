/**
 * Task #89 gate: generating the SAME prompt TWICE (identical decode program)
 * must REPLAY the decode tape from the first generation — full hits, IDENTICAL
 * tokens — not re-record or silently mistrack.
 *
 * ROOT CAUSE (this probe's #89 diagnosis). The decode fn closes over `staticKV`
 * (the KV cache buffers). Per the documented arg-boundary contract, a
 * closure-captured value is FROZEN at record time (jax.jit semantics; see
 * capture.ts "closure values are FROZEN"). So the tape binds gen 1's KV buffers
 * as EXTERNAL inputs. The thing that differs between two generations is the KV
 * BUFFER IDENTITY — and it is GENUINELY IDENTITY (a large external the replay
 * cannot re-dress as volatile data), NOT a per-generation id leaking into the
 * key, NOT a stale 0-d position (positions ride tensorFromArray upload slots and
 * re-dress correctly; #71). The #71-precedent decision: the KV is identity, so
 * the two generations must SHARE the KV buffer (reset in place) to replay — a
 * FRESH KV per generation is a different external and correctly re-records /
 * (with a shared CapturedFn) silently replays the wrong buffer.
 *
 * The two arms prove the decision:
 *   default   — ONE staticKV reused (full zero + len=0) across both generations:
 *               gen 2 REPLAYS gen 1's tape at the warm hit rate with IDENTICAL
 *               tokens. PASS.
 *   FRESH_KV=1 — a fresh KV buffer per generation (the closure-contract
 *               VIOLATION): gen 2's replay reads the frozen (gen 1) buffer and
 *               diverges. The negative control — proves the gate discriminates.
 *
 * Consumer note: packages/{gemma2,qwen3}-browser generateChat allocates a fresh
 * staticKV + CapturedFn (+ residualHook) PER call, so a browser repeat re-records
 * from cold — the filing's "1 hit / 7". That is the frozen-closure contract, not
 * a tape guard bug; the remedy is to REUSE the KV + CapturedFn across identical
 * repeats (only sound when the residualHook is also unchanged).
 *
 * Run (Dawn substitute for the browser decode path — the identical model + tape):
 *   VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim TORCHLETTE_STEP_TAPE=1 \
 *     npx tsx tools/t-repeat-generation-probe.ts
 */
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { getWebGPUInitError, initWebGPU, setGPUMemoryLimit } from "../src/backend/webgpu";
import { Torchlette, type Tensor } from "../src/frontend/torchlette";
import { kvBucketLen } from "../packages/gemma2-browser/src/model";
import { loadPretrainedGemma2 } from "../examples/gemma2/loader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../ckpts/gemma-2-2b");
const PROMPT = JSON.parse(process.env.PROMPT ?? "[2,651,6037,576,6081,603]") as number[];
const NUM_NEW = Number(process.env.NUM_NEW ?? 14);
const log = (m: string) => console.error(`[repeat-gen] ${m}`);

async function main() {
  const flag = process.env.TORCHLETTE_STEP_TAPE;
  if (flag !== "1" && process.env.ALLOW_NOTAPE !== "1") { log("set TORCHLETTE_STEP_TAPE=1"); process.exit(1); }
  const { existsSync } = await import("node:fs");
  if (!existsSync(path.join(MODEL_DIR, "model.safetensors"))) {
    log(`SKIP: gemma-2-2b checkpoint not present at ${MODEL_DIR}`);
    process.exit(0);
  }
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  setGPUMemoryLimit(30 * 1024 * 1024 * 1024);
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedGemma2(api, MODEL_DIR, { maxSeqLen: 256, weightDtype: "f16" });
  const vocab = model.config.vocabSize;

  // Two FRESH staticKVs (real "new chats" — no stale KV leaks), allocated BEFORE
  // the step-scoped region so they persist. The decode reads the CURRENT
  // staticKV via the closure; the tape reads the KV as an external input, so a
  // fresh-buffer-per-generation is exactly the #89 scenario: identical PROGRAM,
  // different external KV buffer.
  const kvs = [model.allocStaticKV(256), model.allocStaticKV(256)];
  const prevScope = api.setStepScopedCleanup(true);
  let staticKV = kvs[0];

  const argmaxFrom = async (logits: Tensor, pos: number): Promise<number> => {
    const top = await api.readTopK(logits, 1, { offset: pos * vocab, length: vocab });
    logits.dispose();
    return top.indices[0];
  };

  // ONE CapturedFn shared across both generations — the tape must carry over.
  const decode = api.capture(
    (idx: Tensor) => api.noGrad(() => model.forward(idx, { staticKV }).logits),
    { key: () => `kv:bkt${kvBucketLen(staticKV.len + 1, 256)}:mod${model.attnModKey}` },
  );

  // FRESH_KV=1 exercises the CONTRACT-VIOLATING pattern (a fresh KV buffer in
  // the closure each generation — a frozen closure value the replay can't track,
  // so gen2 replays gen1's buffer → wrong tokens). Default is the CORRECT
  // pattern: ONE staticKV reused across generations (reset in place), so the
  // closure external is stable and the tape replays. This is the #89 decision:
  // the KV buffer is genuinely IDENTITY (a closure value, frozen at record per
  // the documented arg-boundary contract), NOT volatile data — reuse it.
  const FRESH_KV = process.env.FRESH_KV === "1";
  async function generate(gen: number): Promise<{ ids: number[]; hitsPerStep: boolean[] }> {
    staticKV = FRESH_KV ? kvs[gen] : kvs[0];
    if (!FRESH_KV) for (const t of [...staticKV.k, ...staticKV.v]) t.zero_(); // full zero reset
    staticKV.len = 0;
    await api.markStep();
    let nextTok: number;
    {
      const idx = api.tensorFromArray(PROMPT, [1, PROMPT.length]);
      const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
      nextTok = await argmaxFrom(logits, PROMPT.length - 1);
      await api.markStep();
    }
    const ids: number[] = [];
    const hitsPerStep: boolean[] = [];
    let count = 0;
    while (count < NUM_NEW && nextTok !== 1 && nextTok !== 107) {
      ids.push(nextTok);
      count++;
      const before = decode.stats().hits;
      const logits = (await decode(api.tensorFromArray([nextTok], [1, 1]))) as Tensor;
      hitsPerStep.push(decode.stats().hits > before);
      nextTok = await argmaxFrom(logits, 0);
      await api.markStep();
    }
    return { ids, hitsPerStep };
  }

  try {
    const g1 = await generate(0);
    const g2 = await generate(1);
    api.setStepScopedCleanup(prevScope);

    const hr = (h: boolean[]) => h.filter(Boolean).length / Math.max(h.length, 1);
    // Gen 1 tail (drop the first few cold steps that warm each KV bucket).
    const tailStart = Math.min(4, g1.hitsPerStep.length);
    const g1Tail = g1.hitsPerStep.slice(tailStart);
    const g1TailRate = hr(g1Tail);
    const g2Rate = hr(g2.hitsPerStep);

    const sameTokens = JSON.stringify(g1.ids) === JSON.stringify(g2.ids);

    log(`gen1 ids=${JSON.stringify(g1.ids)}`);
    log(`gen2 ids=${JSON.stringify(g2.ids)}`);
    log(`gen1 hitsPerStep=${g1.hitsPerStep.map((b) => (b ? "H" : "-")).join("")} (${g1.hitsPerStep.filter(Boolean).length}/${g1.hitsPerStep.length})`);
    log(`gen2 hitsPerStep=${g2.hitsPerStep.map((b) => (b ? "H" : "-")).join("")} (${g2.hitsPerStep.filter(Boolean).length}/${g2.hitsPerStep.length})`);
    log(`gen1 warm-tail hit rate=${g1TailRate.toFixed(3)} | gen2 hit rate=${g2Rate.toFixed(3)} | sameTokens=${sameTokens}`);

    console.log("=== REPEAT-GEN-STATS ===");
    console.log(JSON.stringify({
      gen1Ids: g1.ids, gen2Ids: g2.ids, sameTokens,
      gen1WarmTailHitRate: g1TailRate, gen2HitRate: g2Rate,
      gen2Hits: g2.hitsPerStep.filter(Boolean).length, gen2Steps: g2.hitsPerStep.length,
    }, null, 2));

    const pass = sameTokens && g2Rate >= g1TailRate - 1e-9 && g2Rate >= 0.9;
    console.log(pass
      ? "PASS: 2nd identical generation replays at >= gen1 warm-tail hit rate"
      : `FAIL (sameTokens=${sameTokens} gen2Rate=${g2Rate} gen1TailRate=${g1TailRate})`);
    process.exit(pass ? 0 : 1);
  } catch (e) {
    log(`FATAL ${e}\n${(e as Error).stack}`);
    process.exit(1);
  }
}
main();
