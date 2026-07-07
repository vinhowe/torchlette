/**
 * Just create model, read weights. No training. No backward.
 */
import { Torchlette, initWebGPU, nn } from "../src/index";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";

async function main() {
  await initWebGPU();
  const V = VOCAB_SIZE_DATA * 2 + 1, S = 10;
  const api = new Torchlette("webgpu", { enableFusion: false });
  api.manualSeed(42);
  const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });
  // Force model weights
  const params = model.parameters();
  const p0 = await params[0].cpu();
  const pLast = await params[params.length - 1].cpu();
  console.log(`param[0][0:4]: ${p0.slice(0, 4).map((v: number) => v.toFixed(6)).join(" ")}`);
  console.log(`param[${params.length-1}][0:4]: ${pLast.slice(0, 4).map((v: number) => v.toFixed(6)).join(" ")}`);
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
