/**
 * Minimal generation smoke: greedy decode with full recompute per token
 * (no KV cache — that's M2). Reads token ids from argv, prints generated ids.
 *
 * Usage: npx tsx examples/qwen3/generate-smoke.ts '[785,6722,315,9625,374]' 20
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { loadPretrainedQwen3 } from "./loader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");

async function main() {
  const tokens: number[] = JSON.parse(process.argv[2]);
  const numNew = Number(process.argv[3] ?? 20);

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedQwen3(api, MODEL_DIR, { maxSeqLen: 256 });
  const vocab = model.config.vocabSize;

  const t0 = Date.now();
  for (let step = 0; step < numNew; step++) {
    const idx = api.tensorFromArray(tokens, [1, tokens.length]);
    const { logits } = api.noGrad(() => model.forward(idx));
    const flat = new Float32Array(await logits.cpu());
    const off = (tokens.length - 1) * vocab;
    let best = 0;
    for (let v = 1; v < vocab; v++) {
      if (flat[off + v] > flat[off + best]) best = v;
    }
    tokens.push(best);
    process.stderr.write(`.`);
    await api.markStep();
  }
  const dt = (Date.now() - t0) / 1000;
  process.stderr.write(`\n${numNew} tokens in ${dt.toFixed(1)}s (full recompute, no KV cache)\n`);
  // Write to file (4th arg) if given — Dawn teardown can SIGSEGV before stdout flushes.
  const outFile = process.argv[4];
  if (outFile) {
    const fs = await import("node:fs");
    fs.writeFileSync(outFile, JSON.stringify(tokens));
  } else {
    console.log(JSON.stringify(tokens));
  }
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
