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

  // Gate-duration hardening (2b inc-2, coordinator-chartered): the inc-1
  // reaper×replay starvation (idle-retire destroying the compiled plan under
  // the live skeleton after K_IDLE=3 steady boundaries) was INVISIBLE to this
  // gate because two 32-token generations never asserted post-window warmth.
  // Generation 3 runs LONGER (64 tokens ≫ K_IDLE boundaries, all under a warm
  // tape) and the gate asserts (a) it stays essentially all-hits — a
  // mid-generation plan invalidation forces re-traces and fails here — and
  // (b) zero replay-layer invalidations across the whole run.
  //
  // HONESTY NOTE (differential measured 2026-07-08): with the inc-1 fix
  // toggled OFF, this gate still PASSES for decode — the decode template
  // dodges the reaper by LUCK (its harvested KV storages stay alive, so
  // retireIdleTemplate returns "live" and rests the clock). The true
  // regression differential for the reaper×replay class is
  // test/capture.spec.ts flag-on (small plans whose harvests die at the
  // boundary — those DID starve). This gate's role is narrower: it makes the
  // warm-across-the-window invariant EXECUTABLE for the decode shape, so if
  // decode's luck-protection ever erodes (harvest pruning, KV lifetime
  // changes), the strike fails HERE instead of shipping as silent re-traces.
  const { stReplayStats } = await import("../../src/executor/step-tape-replay");
  let last: NonNullable<Awaited<ReturnType<typeof generateChat>>["tape"]>;
  let gen3!: NonNullable<Awaited<ReturnType<typeof generateChat>>["tape"]>;
  const GEN3_TOKENS = 64; // decode calls ≫ K_IDLE steady boundaries
  for (let g = 1; g <= 3; g++) {
    const stats = await generateChat(
      api, model, tok,
      [{ role: "user", content: "Tell me about your day." }],
      {},
      { maxNewTokens: g === 3 ? GEN3_TOKENS : 32, temperature: 0.7, topK: 1, topP: 1 },
    );
    last = stats.tape!;
    if (g === 3) gen3 = stats.tape!;
    console.log(`gen ${g}: hits ${last.hits}/${last.calls} ready=${last.ready}`,
      JSON.stringify(last.recorder));
  }
  const r = last!.recorder!;
  const reasons = (r as unknown as { boundaryReasons?: Record<string, number> }).boundaryReasons ?? {};
  const rs = stReplayStats();
  // gen3 warmth-across-the-reaper-window: allow the bucket-transition
  // re-traces (KV bucket crossings are legitimate cold misses) but a reaper
  // strike costs a whole re-record cycle and lands well below this bar.
  const gen3Warm = gen3.hits >= Math.floor(gen3.calls * 0.85);
  const ok2 =
    last!.hits >= Math.floor(last!.calls * 0.7) &&
    last!.ready &&
    !("endStep" in reasons) &&
    !("beginStep" in reasons) &&
    gen3Warm &&
    rs.invalidations === 0 &&
    rs.missValidity === 0;
  console.log(
    ok2
      ? `GEN-TAPE GATE PASS (gen3 ${gen3.hits}/${gen3.calls} hits across the idle-retire window; invalidations=${rs.invalidations})`
      : `GEN-TAPE GATE FAIL: hits=${last!.hits}/${last!.calls} ready=${last!.ready} reasons=${JSON.stringify(reasons)} gen3=${gen3.hits}/${gen3.calls} invalidations=${rs.invalidations} missValidity=${rs.missValidity}`,
  );
  process.exit(ok2 ? 0 : 1);
}
main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
