/**
 * Forward-only: same weights, same input, just forward + loss.item(). No backward.
 */
import { Torchlette, initWebGPU, nn } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA, type BioWorld } from "../examples/toy-compartmentalization/src/lib/data";
import { createBioWorld } from "../examples/toy-compartmentalization/src/lib/bio-data";

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA * 2 + 1, S = 10, B = 32;
  // Fixed deterministic batch (no Math.random dependency)
  const tokens = new Uint32Array(B * S);
  const targets = new Int32Array(B * (S - 1));
  for (let i = 0; i < tokens.length; i++) tokens[i] = i % V;
  for (let i = 0; i < targets.length; i++) targets[i] = (i + 1) % V;

  const api = new Torchlette("webgpu", { enableFusion: false });
  api.manualSeed(42);
  const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });

  const tok = api.tensorFromArray(tokens, [B, S], { dtype: "i32" });
  const tgt = api.tensorFromArray(targets, [B * (S - 1)], { dtype: "i32" });
  const fwd = model.forward(tok);
  const logits = fwd.logits.narrow(1, 0, S - 1).contiguous().reshape([B * (S - 1), V]);
  const loss = crossEntropy(api, logits, tgt);
  const v = await loss.item();
  console.log(`forward loss: ${v.toFixed(6)}`);
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
