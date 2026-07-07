/**
 * Print 10-step loss trajectory for comparison between OLD and NEW backward.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA * 2 + 1, S = 10, B = 32;
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(42);
  const m = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });
  const o = new Adam(m.parameters(), { lr: 1e-3 });
  const losses: number[] = [];
  for (let i = 0; i < 10; i++) {
    await api.beginStep();
    const b = generateBatchWithCompartments({ seqLen: S, batchSize: B }, 2);
    const t = api.tensorFromArray(b.tokens, [B, S], { dtype: "i32" });
    const g = api.tensorFromArray(b.targets as any, [B * (S - 1)], { dtype: "i32" });
    const l = api.tidy(() => {
      const f = m.forward(t);
      const lg = f.logits.narrow(1, 0, S - 1).contiguous().reshape([B * (S - 1), V]);
      const x = crossEntropy(api, lg, g); api.keep(x); return x;
    });
    t.dispose(); g.dispose();
    const v = await l.item(); losses.push(v);
    await l.backward(); l.dispose();
    o.step(); o.zeroGrad();
    api.endStep();
  }
  console.log("LOSSES: " + losses.map(v => v.toFixed(6)).join(" "));
  // Also read final param to verify weights updated
  const finalW = await m.parameters()[0].cpu();
  console.log("PARAM[0:4]: " + finalW.slice(0, 4).map((v: number) => v.toFixed(6)).join(" "));
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
