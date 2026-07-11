/**
 * Task #71 real-decode measurement (gpt2 substitute — the qwen3-1.7B loader
 * stalls on this box before it finishes building the model, so a smaller real
 * model exercises the SAME per-position-varying narrow-offset decode pattern:
 * forwardCached slices `posIndices.narrow(1, posOffset, 1)` with a per-step
 * posOffset, and every consumer of that slice (the wpe embedding gather, etc.)
 * historically baked the offset into template identity → one template/step.
 *
 * From-scratch distilgpt2 (no weight download needed — template identity is a
 * function of graph STRUCTURE + offset, not weight values). Reports template
 * count growth across N decode steps (each a distinct posOffset). After #71 the
 * offset is data, so the decode steps should share templates (near-flat growth)
 * instead of forking one per position.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *        npx tsx tools/t-gpt2-decode-template-count.ts [steps=16]
 */
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { debugTemplateCount } from "../src/executor/executor";
import { Torchlette } from "../src/frontend/torchlette";
import { DISTILGPT2_CONFIG, GPT2 } from "../examples/gpt2/model";
import type { KVCache } from "../examples/gpt2/model";

async function main() {
  const steps = Number(process.argv[2] ?? 16);
  if (!(await initWebGPU()))
    throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  // From-scratch model (random init) — structure is all that matters here.
  const model = new GPT2(api, { ...DISTILGPT2_CONFIG });

  const PROMPT = [40, 716, 257]; // arbitrary 3 tokens
  let pastKVs: KVCache[] | undefined;
  let posOffset = 0;

  // Prefill.
  const firstLogits = api.noGrad(() => {
    const idx = api.tensorFromArray(PROMPT, [1, PROMPT.length]);
    const { logits, presentKVs } = model.forwardCached(idx, undefined, 0);
    pastKVs = presentKVs;
    return logits;
  });
  await firstLogits.cpu();
  firstLogits.dispose();
  posOffset = PROMPT.length;
  await api.markStep();

  const afterPrefill = debugTemplateCount();
  const perStep: number[] = [];
  const genTokens: number[] = [];
  let tok = 1;

  for (let i = 0; i < steps; i++) {
    const before = debugTemplateCount();
    const logits = api.noGrad(() => {
      const idx = api.tensorFromArray([tok], [1, 1]);
      const { logits: lg, presentKVs } = model.forwardCached(
        idx,
        pastKVs,
        posOffset,
      );
      pastKVs = presentKVs;
      return lg;
    });
    const data = new Float32Array(await logits.cpu());
    logits.dispose();
    // pick a deterministic next token to keep the loop going
    let best = 0;
    for (let v = 1; v < model.config.vocabSize; v++)
      if (data[v] > data[best]) best = v;
    tok = best;
    genTokens.push(best);
    posOffset += 1;
    await api.markStep();
    perStep.push(debugTemplateCount() - before);
  }

  const afterDecode = debugTemplateCount();
  const decodeGrowth = afterDecode - afterPrefill;

  // Byte-identity signal: the greedy token stream is a deterministic function
  // of the logits, so compiled-vs-lowered runs (fixed init/prompt via the same
  // seed path) that match tokens have matching argmax logits. Print it so a
  // TORCHLETTE_COMPILED_PLAN=0 run can be diffed against the default.
  console.log(`TOKENS=${JSON.stringify(genTokens)}`);
  console.log(`steps=${steps}`);
  console.log(`templates afterPrefill=${afterPrefill} afterDecode=${afterDecode}`);
  console.log(`decode-phase template growth=${decodeGrowth}`);
  console.log(`per-step growth=[${perStep.join(",")}]`);
  // Steady-state: after the first couple of decode steps warm the KV-length
  // templates, later steps (distinct posOffset only) should add ~0 templates.
  const steadyGrowth = perStep.slice(3).reduce((a, b) => a + b, 0);
  console.log(`steady (steps 3+) template growth=${steadyGrowth}`);
  console.log(
    steadyGrowth === 0
      ? "RESULT: per-position offset is DATA (steady decode adds 0 templates)"
      : `RESULT: steady decode still forks ${steadyGrowth} templates`,
  );
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
