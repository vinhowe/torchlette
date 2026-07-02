import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { Qwen3 } from "./model";

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const cfg = { vocabSize: 128, hiddenSize: 64, numLayers: 1, numHeads: 2, numKVHeads: 1, headDim: 32, intermediateSize: 96, ropeTheta: 1e6, rmsNormEps: 1e-6, maxSeqLen: 32, weightDtype: "f16" as const };
  const m = new Qwen3(api, cfg, { device: "webgpu" });
  const block = m.layers.get(0) as any;
  console.log("qProj.weight.dtype:", block.attn.qProj.weight.dtype);
  console.log("embed.dtype:", m.embedTokens.weight.dtype);
  console.log("norm.dtype:", m.norm.weight.dtype);
  const src = api.tensorFromArray(new Array(64 * 64).fill(0.5), [64, 64], { dtype: block.attn.qProj.weight.dtype });
  block.attn.qProj.weight.copy_(src);
  const { logits } = api.noGrad(() => m.forward(api.tensorFromArray([1, 2], [1, 2])));
  console.log("forward ok, logits[0]:", new Float32Array(await logits.cpu())[0]);
  process.exit(0);
}
main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
