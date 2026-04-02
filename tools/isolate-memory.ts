/**
 * Isolate which feature causes the storage oscillation.
 * Test: base, +compile, +checkpoint, +AMP (no GradScaler - needs f16).
 */
import { Torchlette, Adam, nn, initWebGPU } from "../src/index";
import { storageTracker } from "../src/graph/storage-tracker";

async function runTest(label: string, opts: { compile?: boolean; checkpoint?: boolean }) {
  const api = new Torchlette();
  const W1 = api.randn([32, 64], { device: "webgpu", requiresGrad: true });
  const W2 = api.randn([64, 10], { device: "webgpu", requiresGrad: true });
  const params = [W1, W2];
  const optimizer = new Adam(params, { lr: 1e-3 }, api);

  const forwardFn = (x: any, target: any) => {
    const h1 = api.relu(api.matmul(x, W1));
    const logits = api.matmul(h1, W2);
    return nn.crossEntropy(api, logits, target);
  };

  const compiledFn = opts.compile ? api.compile(forwardFn) : null;

  const checkpointedFn = opts.checkpoint
    ? (x: any, target: any) => nn.checkpoint(api, (inp: any) => {
        const h1 = api.relu(api.matmul(inp, W1));
        const logits = api.matmul(h1, W2);
        return nn.crossEntropy(api, logits, target);
      }, [x])
    : null;

  const counts: number[] = [];
  for (let step = 0; step < 8; step++) {
    await api.beginStep();

    const x = api.randn([4, 32], { device: "webgpu" });
    const target = api.tensorFromArray(
      Array.from({ length: 4 }, () => Math.floor(Math.random() * 10)),
      [4], { device: "webgpu", dtype: "i32" },
    );

    const loss = api.tidy(() => {
      let l;
      if (compiledFn) l = compiledFn(x, target);
      else if (checkpointedFn) l = checkpointedFn(x, target);
      else l = forwardFn(x, target);
      api.keep(l);
      return l;
    });

    await loss.item();
    await loss.backward();
    optimizer.step();
    optimizer.zeroGrad();
    x.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
    counts.push(storageTracker.stats().reachableStorages);
  }

  const growth = counts[counts.length - 1] - counts[0];
  console.log(`  ${label}: ${counts.join(' → ')} (growth: ${growth >= 0 ? '+' : ''}${growth})`);
}

async function main() {
  await initWebGPU();
  
  console.log("Isolating storage growth by feature:\n");
  await runTest("base          ", {});
  await runTest("compile       ", { compile: true });
  await runTest("checkpoint    ", { checkpoint: true });
  await runTest("compile+ckpt  ", { compile: true, checkpoint: true });
  
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
