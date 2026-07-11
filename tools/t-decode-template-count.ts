/**
 * Task #71 real-decode measurement: how many distinct lowered-plan templates
 * does a static-KV qwen3 decode allocate over N steps? Reports
 * debugTemplateCount() after prefill and after decode, plus ms/token, so the
 * offset-in-identity fix can be measured before vs after on a real run.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *        npx tsx tools/t-decode-template-count.ts [numSteps=16] [f16|f32]
 */
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { debugTemplateCount } from "../src/executor/executor";
import { Torchlette } from "../src/frontend/torchlette";
import { loadPretrainedQwen3 } from "../examples/qwen3/loader";
import type { StaticKV } from "../examples/qwen3/model";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../ckpts/qwen3-1.7b");
const PROMPT = [785, 6722, 315, 9625, 374]; // "The capital of France is"

async function main() {
  const numSteps = Number(process.argv[2] ?? 16);
  const dtype = (process.argv[3] === "f32" ? "f32" : "f16") as "f32" | "f16";
  if (!(await initWebGPU()))
    throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedQwen3(api, MODEL_DIR, {
    maxSeqLen: 512,
    weightDtype: dtype,
  });
  const vocab = model.config.vocabSize;

  const tokens = [...PROMPT];
  const staticKV: StaticKV = model.allocStaticKV(512);
  {
    const idx = api.tensorFromArray(tokens, [1, tokens.length]);
    const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
    const top = await api.readTopK(logits, 1, { length: vocab });
    logits.dispose();
    tokens.push(top.indices[0]);
    await api.markStep();
  }
  const afterPrefill = debugTemplateCount();

  api.setStepScopedCleanup(true);
  const walls: number[] = [];
  const perStepTemplates: number[] = [];
  for (let i = 0; i < numSteps; i++) {
    const t0 = performance.now();
    const last = tokens[tokens.length - 1];
    const idx = api.tensorFromArray([last], [1, 1]);
    const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
    const top = await api.readTopK(logits, 1, { length: vocab });
    logits.dispose();
    tokens.push(top.indices[0]);
    await api.markStep();
    walls.push(performance.now() - t0);
    perStepTemplates.push(debugTemplateCount());
  }
  const afterDecode = debugTemplateCount();
  const late = walls.slice(Math.floor(walls.length / 2));
  const avg = late.reduce((a, b) => a + b, 0) / late.length;

  console.log(`\n=== decode template-count (dtype=${dtype}, steps=${numSteps}) ===`);
  console.log(`templates after prefill: ${afterPrefill}`);
  console.log(`templates after decode:  ${afterDecode}`);
  console.log(`per-step template count: ${JSON.stringify(perStepTemplates)}`);
  console.log(`decode growth over ${numSteps} steps: ${afterDecode - afterPrefill}`);
  console.log(`late-step ms/token: ${avg.toFixed(1)} (${(1000 / avg).toFixed(1)} tok/s)`);
  console.log(`tokens: ${JSON.stringify(tokens.slice(PROMPT.length))}`);
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
