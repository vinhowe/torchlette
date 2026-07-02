/**
 * Decode-path bisection probe: one cached decode step vs no-cache reference,
 * compared layer-by-layer via collectHidden. Also validates the RoPE table
 * slice at posOffset>0 directly against Math.cos/sin.
 *
 * Run: npx tsx examples/qwen3/probe-decode.ts
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { loadPretrainedQwen3 } from "./loader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const PROMPT = [785, 6722, 315, 9625, 374];
const NEXT = 12095; // "Paris" — the (correct) greedy token after the prompt

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedQwen3(api, MODEL_DIR, { maxSeqLen: 256 });
  const cfg = model.config;
  const H = cfg.hiddenSize;

  // --- RoPE slice check at offset 5
  {
    const slice = model.ropeCos.narrow(0, 5, 1).contiguous();
    const vals = new Float32Array(await slice.cpu());
    const half = cfg.headDim / 2;
    let worst = 0;
    for (let i = 0; i < half; i++) {
      const freq = 1 / cfg.ropeTheta ** ((2 * i) / cfg.headDim);
      worst = Math.max(worst, Math.abs(vals[i] - Math.cos(5 * freq)));
    }
    console.log(`ropeCos slice@5 maxErr vs Math.cos: ${worst.toExponential(2)}`);
  }

  // --- Reference: no-cache forward over 6 tokens
  const fullSeq = [...PROMPT, NEXT];
  const ref = api.noGrad(() =>
    model.forward(api.tensorFromArray(fullSeq, [1, fullSeq.length]), {
      collectHidden: true,
    }),
  );
  const refHidden: Float32Array[] = [];
  for (const h of ref.hidden!) refHidden.push(new Float32Array(await h.cpu()));
  const refLogits = new Float32Array(await ref.logits.cpu());

  // --- Cached: prefill 5, then decode NEXT at posOffset 5
  const pre = api.noGrad(() =>
    model.forward(api.tensorFromArray(PROMPT, [1, PROMPT.length]), {
      collectHidden: true,
    }),
  );
  const preHidden: Float32Array[] = [];
  for (const h of pre.hidden!) preHidden.push(new Float32Array(await h.cpu()));

  const dec = api.noGrad(() =>
    model.forward(api.tensorFromArray([NEXT], [1, 1]), {
      pastKVs: pre.presentKVs,
      posOffset: PROMPT.length,
      collectHidden: true,
    }),
  );
  const decHidden: Float32Array[] = [];
  for (const h of dec.hidden!) decHidden.push(new Float32Array(await h.cpu()));
  const decLogits = new Float32Array(await dec.logits.cpu());

  // --- Compare
  const S = fullSeq.length;
  console.log("\nlayer | prefill pos0-4 vs ref | decode pos5 vs ref");
  for (let li = 0; li < refHidden.length; li++) {
    // prefill: positions 0..4 of ref
    const refPre = refHidden[li].subarray(0, (S - 1) * H);
    const dPre = maxAbsDiff(preHidden[li], refPre as Float32Array);
    // decode: position 5 of ref
    const refLast = refHidden[li].subarray((S - 1) * H, S * H);
    const dDec = maxAbsDiff(decHidden[li], refLast as Float32Array);
    console.log(
      `h[${li.toString().padStart(2)}]   ${dPre.toExponential(2)}              ${dDec.toExponential(2)}`,
    );
  }
  const refLastLogits = refLogits.subarray((S - 1) * cfg.vocabSize, S * cfg.vocabSize);
  console.log(`logits decode vs ref: ${maxAbsDiff(decLogits, refLastLogits as Float32Array).toExponential(2)}`);
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
