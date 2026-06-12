/**
 * Checkpoint+autocast+fusion on the REAL GPT-2 model — the combination the
 * full-finetuning spec routes around ("autocast disabled due to known
 * reshape issue", "fusion disabled due to reshape issues"). Compares loss +
 * a sampled grad against the plain reference for every combination.
 */
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim";

const CFG: GPT2Config = {
  vocabSize: 500, blockSize: 64, numLayers: 4, numHeads: 4,
  embedDim: 128, dropoutRate: 0.0,
};

async function run(ckpt: boolean, ac: boolean, fusion: boolean) {
  try {
    const api = new Torchlette("webgpu", { enableFusion: fusion, enableMemoryPlanning: true });
    const model = new GPT2(api, CFG, { device: "webgpu" });
    model.train();
    const B = 2, S = 16;
    let x = 1234;
    const tok = () => { x = (x * 1103515245 + 12345) % 2147483648; return x % CFG.vocabSize; };
    const input = api.tensorFromArray(Array.from({ length: B * S }, tok), [B, S]);
    const target = api.tensorFromArray(Array.from({ length: B * S }, tok), [B, S]);
    const STEPS = parseInt(process.env.STEPS ?? "1", 10);
    const useOpt = process.env.OPT === "1";
    const useMark = process.env.MARK === "1";
    const opt = useOpt ? new Adam(model.parameters(), { lr: 1e-3 }, api) : null;
    let lv = 0;
    for (let step = 0; step < STEPS; step++) {
      if (useMark) await api.beginStep();
      const fwd = () =>
        model.forwardWithLoss(input, target, { useCheckpoint: ckpt });
      const { loss } = ac ? api.autocast(fwd) : fwd();
      if (!loss) return "NULL LOSS";
      lv = await loss.item();
      await loss.backward();
      const last = step === STEPS - 1;
      if (opt && !last) {
        opt.step();
        opt.zeroGrad();
      } else if (!last) {
        for (const p of model.parameters()) p.zeroGrad();
      }
      if (!last) loss.dispose();
      if (useMark) {
        api.endStep();
        await api.markStep();
      }
    }
    const params = model.parameters();
    let nulls = 0, nonfinite = 0;
    const sample: number[] = [];
    for (const p of params) {
      const g = (p as unknown as { grad: { cpu(): Promise<Float32Array> } | null }).grad;
      if (!g) { nulls++; continue; }
      const a = await g.cpu();
      for (let i = 0; i < Math.min(4, a.length); i++) sample.push(a[i]);
      for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) { nonfinite++; break; }
    }
    return { lv, nulls, nonfinite, sample };
  } catch (e) {
    return `THROW: ${String(e).slice(0, 160)}`;
  }
}

async function main() {
  if (!(await initWebGPU())) process.exit(1);
  const ref = await run(false, false, false);
  if (typeof ref === "string") { console.log("reference failed:", ref); process.exit(0); }
  console.log(`reference        : loss=${ref.lv.toFixed(5)} nulls=${ref.nulls} nonfinite=${ref.nonfinite}`);
  const cases: Array<[string, boolean, boolean, boolean]> = [
    ["ckpt             ", true, false, false],
    ["ckpt+fusion      ", true, false, true],
    ["autocast         ", false, true, false],
    ["ckpt+autocast    ", true, true, false],
    ["ckpt+ac+fusion   ", true, true, true],
  ];
  for (const [name, c, a, f] of cases) {
    const got = await run(c, a, f);
    if (typeof got === "string") { console.log(`${name}: ${got}`); continue; }
    let worst = 0;
    for (let i = 0; i < Math.min(ref.sample.length, got.sample.length); i++)
      worst = Math.max(worst, Math.abs(got.sample[i] - ref.sample[i]));
    console.log(
      `${name}: loss=${got.lv.toFixed(5)} dLoss=${Math.abs(got.lv - ref.lv).toExponential(2)} nulls=${got.nulls} nonfinite=${got.nonfinite} gradSampleDiff=${worst.toExponential(2)}`,
    );
  }
  process.exit(0);
}
main().catch((e) => { console.error(e); process.exit(1); });
