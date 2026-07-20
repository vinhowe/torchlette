/**
 * THE CRYSTAL PUSH — CAMPAIGN 1 (unrolled-K decode) — P4 CUTOVER ROUTING GATE.
 *
 * Proves the generateChat cutover at the PRODUCT surface (not decodeBlock
 * directly): with the default flag, GREEDY decode routes through the unrolled-K
 * block; the flag opt-out (TORCHLETTE_UNROLLED_K=0) restores the per-token host
 * loop; and the two are BYTE-IDENTICAL. The top-k/top-p sampler is the §4 host
 * residue on BOTH (unchanged). Discriminator = whether a per-token CapturedFn
 * was built (stats.tape !== undefined ⇔ host loop; undefined ⇔ block path).
 *
 * Run: eval "$(tools/pick-gpu.sh)"; VULKAN_DEVICE_INDEX=$VULKAN_DEVICE_INDEX \
 *        LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH npx tsx tools/t-uk-cutover.ts
 */
import {
  generateChat,
  type GenerateOptions,
} from "../packages/qwen3-browser/src/generate";
import type { Qwen3Config } from "../packages/qwen3-browser/src/model";
import { Qwen3 } from "../packages/qwen3-browser/src/model";
import { getGpuUncapturedErrorCount, getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

const CONFIG: Qwen3Config = {
  vocabSize: 256,
  hiddenSize: 64,
  numLayers: 2,
  numHeads: 4,
  numKVHeads: 2,
  headDim: 16,
  intermediateSize: 128,
  ropeTheta: 1e6,
  rmsNormEps: 1e-6,
  maxSeqLen: 256,
};

let FAIL = 0;
const ok = (cond: boolean, msg: string) => {
  console.log(`${cond ? "PASS" : "FAIL"} — ${msg}`);
  if (!cond) FAIL++;
};

// Fixed prompt ids; decode(ids) = ids.join(",") so the emitted text streams the
// id sequence (byte-comparable across arms).
const PROMPT_IDS = [3, 14, 15, 92, 65, 35];
const tokenizer = {
  encode: (_t: string) => [...PROMPT_IDS],
  decode: (ids: number[], _o?: { skip_special_tokens?: boolean }) => ids.join(","),
};

async function run(
  api: Torchlette,
  model: Qwen3,
  opts: GenerateOptions,
): Promise<{ ids: number[]; tapeDefined: boolean }> {
  let text = "";
  const stats = await generateChat(
    api,
    model,
    tokenizer,
    [{ role: "user", content: "hi" }],
    {
      onDelta: (d) => {
        text += d;
      },
      onReplace: (t) => {
        text = t;
      },
    },
    opts,
  );
  const ids = text.length ? text.split(",").map(Number) : [];
  return { ids, tapeDefined: stats.tape !== undefined };
}

const eq = (a: number[], b: number[]) =>
  a.length === b.length && a.every((t, i) => t === b[i]);

async function main() {
  if (!(await initWebGPU())) {
    console.error(getWebGPUInitError() || "WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(1234);
  const model = new Qwen3(api, { ...CONFIG });

  console.log("=== P4 CUTOVER ROUTING GATE (random-init Qwen3) ===\n");

  const setFlag = (v: string | undefined) => {
    if (v === undefined) delete process.env.TORCHLETTE_UNROLLED_K;
    else process.env.TORCHLETTE_UNROLLED_K = v;
  };
  const MAXNEW = 14;

  // (1) GREEDY default (flag unset) routes through the BLOCK (no CapturedFn).
  setFlag(undefined);
  const gDefault = await run(api, model, { temperature: 0, maxNewTokens: MAXNEW });
  await api.markStep();
  ok(!gDefault.tapeDefined, "greedy DEFAULT routes through the block (tape undefined)");
  ok(gDefault.ids.length > 0, `greedy DEFAULT produced ${gDefault.ids.length} tokens`);

  // (2) GREEDY opt-out (flag=0) restores the per-token HOST loop (CapturedFn).
  setFlag("0");
  const gHost = await run(api, model, { temperature: 0, maxNewTokens: MAXNEW });
  await api.markStep();
  ok(gHost.tapeDefined, "greedy OPT-OUT (flag=0) routes through the host loop (tape defined)");

  // (3) BYTE-IDENTICAL cutover: block-default greedy == host opt-out greedy.
  ok(
    eq(gDefault.ids, gHost.ids),
    "cutover byte-identical: block-default greedy == host opt-out greedy" +
      (eq(gDefault.ids, gHost.ids)
        ? ""
        : `\n    block: [${gDefault.ids.join(",")}]\n    host : [${gHost.ids.join(",")}]`),
  );

  // (4) An explicit K routes greedy through the block too (byte-identical).
  setFlag("8");
  const gK8 = await run(api, model, { temperature: 0, maxNewTokens: MAXNEW });
  await api.markStep();
  ok(!gK8.tapeDefined, "greedy flag=8 routes through the block (tape undefined)");
  ok(eq(gK8.ids, gDefault.ids), "greedy flag=8 == greedy default (block K is streaming granularity only)");

  // (5) The top-k+top-p sampler (the shipped demo config) is the §4 HOST residue
  //     on BOTH the default and opt-out — routing is unchanged for it.
  setFlag(undefined);
  const sDefault = await run(api, model, {
    temperature: 0.7,
    topK: 20,
    topP: 0.95,
    maxNewTokens: MAXNEW,
  });
  await api.markStep();
  ok(sDefault.tapeDefined, "top-k+top-p sampler is the host residue on DEFAULT (tape defined)");

  // (6) zero uncaptured GPU errors across the greedy cutover.
  const uncaptured = getGpuUncapturedErrorCount();
  ok(uncaptured === 0, `zero uncaptured GPU errors across the cutover run — got ${uncaptured}`);

  setFlag(undefined);
  console.log(
    `\n=== VERDICT: ${
      FAIL === 0
        ? "PASS — greedy cutover routes to the block by default, byte-identical, residue on host"
        : `FAIL (${FAIL} checks)`
    } ===`,
  );
  process.exit(FAIL === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
