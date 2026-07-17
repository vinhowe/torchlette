/**
 * Stage-4 phase 0 determinism gate: building the SAME template twice must
 * produce identical canonical command streams. Everything in the
 * compile-from-IR migration assumes stream determinism; this pins it.
 *
 * Method: train-step-shaped plan (forward+backward+optimizer), capture the
 * compiled stream, invalidate the plan (keeping the template), re-execute
 * to force a re-build, capture again, diff.
 *
 * RE-BASED (task #43 recorded-build sunset): this gate now measures the
 * GENERATED (build-from-IR) build source under the DEFAULT flag state — the
 * recorded build is gone. Because build-from-IR coverage is per-plan gated, a
 * plan may build-from-IR on one pass and fall through to lowered on the other,
 * so the SET of compiled streams can differ across passes; determinism =
 * every plan that compiled on BOTH passes rebuilt byte-identically (the
 * label-matched intersection).
 */
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { diffStreams, canonicalizeStream } from "../src/executor/stream-diff";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim";

const CFG: GPT2Config = {
  vocabSize: 200, blockSize: 32, numLayers: 2, numHeads: 4,
  embedDim: 64, dropoutRate: 0.0,
};

async function main() {
  // Measures the GENERATED build source under the DEFAULT flag state (the
  // recorded build is gone — task #43 sunset). No BUILD_FROM_IR=0: build-from-IR
  // IS the build source now.
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

  // Steps 1-2: populate arena, then build-from-IR.
  await step();
  await step();
  await step(); // first replay — streams now stable
  const first = getCompiledStreams();
  if (first.length === 0) {
    console.log("NO COMPILED STREAMS — gate inconclusive");
    process.exit(1);
  }
  // Invalidate compiled plans (templates survive) → next step re-builds.
  invalidateCompiledKeepTemplates();
  // Mirror the FIRST phase's 3-step settle. The recorded build re-populated
  // getCompiledStreams() one step sooner; on the build-from-IR-only path (the
  // recorded build sunset, #43/D4) the post-invalidation rebuild needs the same
  // 3-step settle as the initial build (step 1 stale, step 2 empty as the
  // invalidation clears, step 3 rebuilt byte-identical) before it is stable.
  await step();
  await step();
  await step();
  const second = getCompiledStreams();
  // Re-based onto the generated build: a plan may build-from-IR on one pass and
  // fall through to lowered on the other (build-from-IR is per-plan coverage-
  // gated), so the SET of compiled streams can differ. Determinism = every plan
  // that compiled on BOTH passes rebuilt byte-identically. Diff the label-matched
  // intersection.
  const secondByLabel = new Map(second.map((s) => [s.label, s]));
  let allEqual = true;
  let matched = 0;
  for (const a of first) {
    const b = secondByLabel.get(a.label);
    if (!b) continue;
    matched++;
    const d = diffStreams(a.commands, b.commands);
    if (!d.equal) {
      allEqual = false;
      console.log(
        `DIVERGENCE stream (${a.label}) at cmd ${d.firstDivergence}:\n  A: ${d.a}\n  B: ${d.b}\n  lens ${d.lengthA}/${d.lengthB}`,
      );
    } else {
      console.log(`stream (${a.label}): ${d.lengthA} cmds IDENTICAL`);
    }
  }
  if (matched === 0) {
    console.log("NO OVERLAPPING COMPILED STREAMS — gate inconclusive");
    process.exit(1);
  }
  console.log(allEqual ? "DETERMINISM: PASS" : "DETERMINISM: FAIL");
  void canonicalizeStream;
  // Clean Dawn teardown before hard exit — a bare process.exit() leaves the
  // Vulkan/Dawn device handle live and segfaults (code 139) when this tool
  // is spawned from a vitest worker that already holds a WebGPU device
  // (nested-device contention). The parity probe does the same.
  try {
    destroyWebGPU();
  } catch {
    /* ignore */
  }
  process.exit(allEqual ? 0 : 1);
}
main().catch((e) => { console.error(e); process.exit(1); });
