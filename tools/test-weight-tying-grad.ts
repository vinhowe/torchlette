/**
 * Does torchlette accumulate the gradient of a weight-tied tensor from BOTH
 * use sites? W is used as (a) an embedding lookup and (b) a linear weight —
 * exactly the GPT-2 wte/lm_head tie. The combined gradient must equal the
 * sum of the two single-path gradients. If torchlette captures only one
 * path, the tied embedding (79% of params in our model) learns at the wrong
 * rate — the leading suspect for the ~0.5 nat parity gap.
 */

import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

const log = (m: string) => console.error(`[tie] ${m}`);

async function gradOfW(
  api: Torchlette,
  Wdata: number[],
  vocab: number,
  embed: number,
  idxData: number[],
  xData: number[],
  rows: number,
  mode: "embed" | "linear" | "both",
): Promise<Float32Array> {
  await api.beginStep();
  const W = api.tensorFromArray(Wdata, [vocab, embed], {
    device: "webgpu",
    requiresGrad: true,
  });
  const idx = api.tensorFromArray(idxData, [rows], { device: "webgpu" });
  const x = api.tensorFromArray(xData, [rows, embed], { device: "webgpu" });

  const loss = api.tidy(() => {
    let acc: ReturnType<Torchlette["sum"]> | null = null;
    if (mode === "embed" || mode === "both") {
      // Embedding lookup of rows of W by idx, then sum.
      const e = api.embedding(W, idx); // [rows, embed]
      acc = api.sum(e);
    }
    if (mode === "linear" || mode === "both") {
      const lin = api.linear(x, W, null); // x[rows,embed] @ W^T[embed,vocab] -> [rows, vocab]
      const s = api.sum(lin);
      acc = acc ? api.add(acc, s) : s;
    }
    api.keep(acc!);
    return acc!;
  });

  await loss.backward();
  // biome-ignore lint/suspicious/noExplicitAny: grad isn't typed
  const g = (W as any).grad as { cpu(): Promise<Float32Array> } | null;
  if (!g) throw new Error(`mode ${mode}: W.grad is null`);
  const arr = new Float32Array(await g.cpu());
  api.endStep();
  await api.markStep();
  return arr;
}

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(1);

  const vocab = 6;
  const embed = 4;
  const rows = 3;
  // Deterministic small data.
  const Wdata = Array.from({ length: vocab * embed }, (_, i) => (i % 7) * 0.1 - 0.3);
  const idxData = [0, 2, 5];
  const xData = Array.from({ length: rows * embed }, (_, i) => ((i * 3) % 5) * 0.2 - 0.4);

  const gE = await gradOfW(api, Wdata, vocab, embed, idxData, xData, rows, "embed");
  const gL = await gradOfW(api, Wdata, vocab, embed, idxData, xData, rows, "linear");
  const gB = await gradOfW(api, Wdata, vocab, embed, idxData, xData, rows, "both");

  let maxErr = 0;
  for (let i = 0; i < gB.length; i++) {
    const expected = gE[i]! + gL[i]!;
    maxErr = Math.max(maxErr, Math.abs(gB[i]! - expected));
  }
  log(`embed-only grad norm:  ${Math.hypot(...gE).toFixed(4)}`);
  log(`linear-only grad norm: ${Math.hypot(...gL).toFixed(4)}`);
  log(`combined grad norm:    ${Math.hypot(...gB).toFixed(4)}`);
  log(`max|combined - (embed+linear)| = ${maxErr.toExponential(3)}`);
  const ok = maxErr < 1e-4;
  log(ok ? "PASS — tied gradient accumulates both paths" : "FAIL — tied gradient is WRONG");

  await destroyWebGPU();
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
