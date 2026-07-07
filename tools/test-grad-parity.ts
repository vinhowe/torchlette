/**
 * Compare single-step gradients: OLD vs NEW backward on same weights/input.
 * Uses a SINGLE Torchlette instance (no multi-instance aliasing).
 */
import { Torchlette, initWebGPU, nn } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA * 2 + 1, S = 10, B = 32;

  const api = new Torchlette("webgpu", { enableFusion: false });
  api.manualSeed(42);
  const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });

  // Fixed batch
  const batch = generateBatchWithCompartments({ seqLen: S, batchSize: B }, 2);

  // Run forward + backward + read grads
  await api.beginStep();
  const tok = api.tensorFromArray(batch.tokens, [B, S], { dtype: "i32" });
  const tgt = api.tensorFromArray(batch.targets as any, [B * (S - 1)], { dtype: "i32" });
  const loss = api.tidy(() => {
    const fwd = model.forward(tok);
    const logits = fwd.logits.narrow(1, 0, S - 1).contiguous().reshape([B * (S - 1), V]);
    const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
  });
  tok.dispose(); tgt.dispose();
  const lossVal = await loss.item();
  await loss.backward(); loss.dispose();

  // Read all gradients
  const params = model.parameters();
  const grads: { name: string; grad: number[] }[] = [];
  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    if (p.grad) {
      const g = await p.grad.cpu();
      grads.push({ name: `param[${i}] shape=[${p.shape}]`, grad: g });
    }
  }
  api.endStep();

  console.log(`loss: ${lossVal.toFixed(6)}`);
  console.log(`params with grads: ${grads.length}`);

  // Print first 4 gradient values for each param
  for (const { name, grad } of grads) {
    const first4 = grad.slice(0, 4).map(v => v.toFixed(6)).join(" ");
    const norm = Math.sqrt(grad.reduce((s, v) => s + v * v, 0));
    const hasNaN = grad.some(v => !Number.isFinite(v));
    console.log(`  ${name}: [${first4} ...] norm=${norm.toFixed(4)} nan=${hasNaN}`);
  }

  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
