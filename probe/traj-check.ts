/**
 * Behavior-identity trajectory recorder for the P0 pass-scaling changes.
 *
 * Runs a deterministic 30-step training trajectory on the real distilgpt2
 * weights with checkpoint + GradScaler + Adam — exercising every pass I touched
 * (reorderPlanForFusion, segmentPlanForExecution, enforceWriteAfterReadOrder's
 * checkpoint edges + in-place WAR + affinity). Prints per-step losses so the
 * SAME run before/after the changes (git stash) can be diffed: the fixes are
 * NULL iff the trajectories match. TORCHLETTE_COMPILED_PLAN=0 gives the lowered
 * direction.
 */
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler } from "../src/optim";
import { clipGradNorm_ } from "../src/nn";

const STEPS = 30;
const tokens = [
  2514, 307, 393, 407, 284, 307, 11, 326, 318, 262, 1808, 13, 2514, 307, 393, 407,
];

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  const model = await loadPretrainedGPT2(
    api,
    "./models/distilgpt2",
    { dropoutRate: 0 },
    { device: "webgpu" },
  );
  model.train();
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const input = api.tensorFromArray(tokens.slice(0, -1), [1, tokens.length - 1]);
  const target = api.tensorFromArray(tokens.slice(1), [1, tokens.length - 1]);

  const losses: number[] = [];
  for (let step = 0; step < STEPS; step++) {
    await scaler.resolveDeferred();
    await api.beginStep();
    optimizer.zeroGrad();
    const { loss } = model.forwardWithLoss(input, target, { useCheckpoint: true });
    const lossVal = await loss!.item();
    losses.push(lossVal);
    const scaled = scaler.scale(loss!);
    await scaled.backward();
    scaler.unscale_(optimizer);
    clipGradNorm_(api, model.parameters(), 1.0);
    scaler.step(optimizer);
    scaler.update();
    api.endStep();
    await api.markStep();
  }
  console.log("TRAJ " + losses.map((l) => l.toFixed(6)).join(","));
  process.exit(0);
}
main().catch((e) => {
  console.error(e);
  process.exit(1);
});
