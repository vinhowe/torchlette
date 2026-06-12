/**
 * Stage-4 phase 0 determinism gate: recording the SAME template twice must
 * produce identical canonical command streams. Everything in the
 * compile-from-IR migration assumes stream determinism; this pins it.
 *
 * Method: train-step-shaped plan (forward+backward+optimizer), capture the
 * compiled stream, invalidate the plan (keeping the template), re-execute
 * to force a re-recording, capture again, diff.
 */
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { initWebGPU } from "../src/backend/webgpu";
import { diffStreams, canonicalizeStream } from "../src/executor/stream-diff";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim";

const CFG: GPT2Config = {
  vocabSize: 200, blockSize: 32, numLayers: 2, numHeads: 4,
  embedDim: 64, dropoutRate: 0.0,
};

async function main() {
  if (!(await initWebGPU())) process.exit(1);
  const api = new Torchlette("webgpu", { enableFusion: true, enableMemoryPlanning: true });
  const model = new GPT2(api, CFG, { device: "webgpu" });
  model.train();
  const opt = new Adam(model.parameters(), { lr: 1e-3 }, api);
  let x = 7;
  const tok = () => { x = (x * 1103515245 + 12345) % 2147483648; return x % CFG.vocabSize; };
  const input = api.tensorFromArray(Array.from({ length: 32 }, tok), [2, 16]);
  const target = api.tensorFromArray(Array.from({ length: 32 }, tok), [2, 16]);

  const { getCompiledStreams, invalidateCompiledKeepTemplates } = await import(
    "../src/executor/executor"
  );

  const step = async () => {
    await api.beginStep();
    const { loss } = model.forwardWithLoss(input, target, {});
    await loss!.item();
    await loss!.backward();
    opt.step();
    opt.zeroGrad();
    api.endStep();
    await api.markStep();
  };

  // Steps 1-2: populate arena, then record.
  await step();
  await step();
  await step(); // first replay — streams now stable
  const first = getCompiledStreams();
  if (first.length === 0) {
    console.log("NO COMPILED STREAMS — gate inconclusive");
    process.exit(1);
  }
  // Invalidate compiled plans (templates survive) → next step re-records.
  invalidateCompiledKeepTemplates();
  await step();
  await step();
  const second = getCompiledStreams();
  if (second.length !== first.length) {
    console.log(`STREAM COUNT differs: ${first.length} vs ${second.length}`);
    process.exit(1);
  }
  let allEqual = true;
  for (let i = 0; i < first.length; i++) {
    const d = diffStreams(first[i].commands, second[i].commands);
    if (!d.equal) {
      allEqual = false;
      console.log(
        `DIVERGENCE stream ${i} (${first[i].label}) at cmd ${d.firstDivergence}:\n  A: ${d.a}\n  B: ${d.b}\n  lens ${d.lengthA}/${d.lengthB}`,
      );
    } else {
      console.log(`stream ${i} (${first[i].label}): ${d.lengthA} cmds IDENTICAL`);
    }
  }
  console.log(allEqual ? "DETERMINISM: PASS" : "DETERMINISM: FAIL");
  void canonicalizeStream;
  process.exit(allEqual ? 0 : 1);
}
main().catch((e) => { console.error(e); process.exit(1); });
