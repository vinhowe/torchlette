/**
 * Qwen3 parity harness: torchlette forward vs HF-transformers fp32 reference.
 *
 * Prereq: python3 examples/qwen3/dump-reference.py (writes ckpts/qwen3-1.7b/reference/).
 * Run:    npx tsx examples/qwen3/parity.ts
 *
 * Compares per-position logits and (with PARITY_HIDDEN=1) per-layer hidden
 * states on the reference prompts. Reports max/mean abs diff and top-5
 * agreement at the last position.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { loadPretrainedQwen3 } from "./loader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const REF_DIR = path.join(MODEL_DIR, "reference");

type PromptRef = {
  text: string;
  token_ids: number[];
  seq_len: number;
  logits_file: string;
  logits_shape: number[];
  hidden_file: string;
  hidden_shape: number[];
  top5_last_ids: number[];
  top5_last_logits: number[];
};

function readBin(file: string): Float32Array {
  const buf = fs.readFileSync(path.join(REF_DIR, file));
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function diffStats(a: Float32Array, b: Float32Array) {
  let maxAbs = 0;
  let sumAbs = 0;
  let maxIdx = -1;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxAbs) {
      maxAbs = d;
      maxIdx = i;
    }
    sumAbs += d;
  }
  return { maxAbs, meanAbs: sumAbs / a.length, maxIdx };
}

function topK(arr: Float32Array, offset: number, len: number, k: number): number[] {
  const idx = Array.from({ length: len }, (_, i) => i);
  idx.sort((x, y) => arr[offset + y] - arr[offset + x]);
  return idx.slice(0, k);
}

async function main() {
  const manifest = JSON.parse(
    fs.readFileSync(path.join(REF_DIR, "manifest.json"), "utf-8"),
  );
  const compareHidden = process.env.PARITY_HIDDEN === "1";

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  const weightDtype = (process.env.QWEN3_DTYPE === "f16" ? "f16" : "f32") as "f32" | "f16";
  // f16 weights: logits move by weight-quantization noise; top-5 agreement is
  // the gate and abs-diff is reported. f32: near-exact.
  const maxAbsGate = weightDtype === "f16" ? Number.POSITIVE_INFINITY : 1e-2;
  console.log(`weightDtype=${weightDtype}`);
  const model = await loadPretrainedQwen3(api, MODEL_DIR, { maxSeqLen: 256, weightDtype });
  const vocab = manifest.vocab_size as number;

  let allPass = true;
  for (const [pi, ref] of (manifest.prompts as PromptRef[]).entries()) {
    console.log(`\n=== prompt ${pi}: ${JSON.stringify(ref.text.slice(0, 50))} (seq=${ref.seq_len})`);
    const idx = api.tensorFromArray(ref.token_ids, [1, ref.seq_len]);

    const result = api.noGrad(() =>
      model.forward(idx, { collectHidden: compareHidden }),
    );
    const logits = new Float32Array(await result.logits.cpu());

    const refLogits = readBin(ref.logits_file);
    if (logits.length !== refLogits.length) {
      throw new Error(`logits length mismatch: ${logits.length} vs ${refLogits.length}`);
    }

    if (compareHidden && result.hidden) {
      const refHidden = readBin(ref.hidden_file);
      const [numStates, seq, hiddenSize] = ref.hidden_shape;
      for (let li = 0; li < numStates; li++) {
        const ours = new Float32Array(await result.hidden[li].cpu());
        const slice = refHidden.subarray(li * seq * hiddenSize, (li + 1) * seq * hiddenSize);
        const s = diffStats(ours, slice);
        console.log(
          `  hidden[${li.toString().padStart(2)}]  maxAbs=${s.maxAbs.toExponential(2)}  meanAbs=${s.meanAbs.toExponential(2)}`,
        );
      }
    }

    const s = diffStats(logits, refLogits);
    const lastOff = (ref.seq_len - 1) * vocab;
    const ourTop5 = topK(logits, lastOff, vocab, 5);
    const top5Match = ourTop5.every((id, i) => id === ref.top5_last_ids[i]);
    const pass = s.maxAbs < maxAbsGate && top5Match;
    allPass &&= pass;
    console.log(
      `  logits  maxAbs=${s.maxAbs.toExponential(3)}  meanAbs=${s.meanAbs.toExponential(3)}  ${pass ? "PASS" : "FAIL"}`,
    );
    console.log(`  top5 ours=${JSON.stringify(ourTop5)} ref=${JSON.stringify(ref.top5_last_ids)} match=${top5Match}`);
    await api.markStep();
  }

  console.log(`\n${allPass ? "ALL PROMPTS PASS" : "PARITY FAILURES — see above"}`);
  process.exit(allPass ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
