/**
 * t-multiengine-reclaim-probe.ts — characterize + gate GPU-memory reclaim across
 * sequential engine construction in ONE process (task #94, item 2).
 *
 * The problem: building many engines in one Node process VkOOM'd probe harnesses
 * after ~8 engines. Root cause: the implicit new-engine path
 * (clearTemplateCacheForNewEngine → storageTracker.disposeAllForNewEngine)
 * ORPHANS the previous engine's GPU buffers rather than destroying them — a
 * deliberate safety (a still-live dead-engine wrapper GC-releasing a destroyed
 * buffer into the shared pool mid-run is the #84 corruption class). That safety
 * leaks device residency for the process lifetime.
 *
 * The fix: an explicit api.destroy() the caller invokes when done with an engine.
 * On webgpu it tears down the device (destroyWebGPU → device.destroy frees every
 * buffer; teardown callbacks reset the module-global pool / arena / memory
 * tracker), so the next initWebGPU()+engine starts from a clean device.
 *
 * This probe builds N engines sequentially, each running a few training steps of
 * a small from-scratch GPT-2, and reports GPU currentBytes after each engine.
 *   DESTROY=1 (default): call api.destroy() + re-init between engines → BOUNDED.
 *   DESTROY=0            : the legacy orphan-only path → GROWS (characterization).
 *
 * KNOWN RESIDUAL (honest characterization): with DESTROY=1 the TRACKED residency
 * reclaims fully to ~0 MB and results stay correct, but the FULL training-loop
 * path still emits Dawn "[Buffer] is associated with [Device]" validation errors
 * on each re-init — a deep cache (compiled-plan / kernel / staging) references a
 * prior-device buffer that is rebuilt, not read (simple-op correctness is clean;
 * the in-suite gate's fresh-engine matmul verifies this). These are surfaced
 * LOUDLY by the item-3 submit-drop guard under TORCHLETTE_STRICT_GPU. The
 * gpuErrDelta below tracks them; currentBytes is the reclaim metric that matters.
 *
 * Env: N (engines, default 12), STEPS (default 3), DESTROY (default 1),
 *      BOUND_MB (assertion cap on post-first-engine growth, default 64).
 *
 * VULKAN_DEVICE_INDEX + tools/vk-shim to pin a free GPU. Dawn requires
 * process.exit at the end.
 */
import {
  destroyWebGPU,
  getGpuUncapturedErrorCount,
  getGPUMemoryStats,
  initWebGPU,
} from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { Adam, CosineAnnealingLR, GradScaler } from "../src/optim/index";
import { clipGradNorm_ } from "../src/nn/index";

const N = parseInt(process.env.N ?? "12", 10);
const STEPS = parseInt(process.env.STEPS ?? "3", 10);
const DESTROY = (process.env.DESTROY ?? "1") === "1";
const BOUND_MB = parseFloat(process.env.BOUND_MB ?? "64");
const log = (m: string) => console.error(`[reclaim] ${m}`);

const CONFIG: GPT2Config = {
  vocabSize: 256,
  blockSize: 32,
  numLayers: 3,
  numHeads: 4,
  embedDim: 64,
  dropoutRate: 0,
};

async function runOneEngine(): Promise<void> {
  const BATCH = 2;
  const SEQ = 24;
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  const model = new GPT2(api, CONFIG, { device: "webgpu" });
  model.train(true);
  const params = model.parameters();
  const opt = new Adam(params, { lr: 1e-3, weightDecay: 0.01, adamW: true }, api);
  const sched = new CosineAnnealingLR(opt, STEPS, 1e-5);
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = CONFIG.vocabSize;
  let seed = 42;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff);
  const inp = new Int32Array(BATCH * SEQ);
  const tgt = new Int32Array(BATCH * SEQ);
  for (let stp = 0; stp < STEPS; stp++) {
    await scaler.resolveDeferred();
    for (let i = 0; i < BATCH * SEQ; i++) {
      inp[i] = rnd() % V;
      tgt[i] = rnd() % V;
    }
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
      device: "webgpu",
    });
    const l = api.tidy(() => {
      const ll = api.autocast(
        () => model.forwardWithLoss(input, target, { useCheckpoint: true }).loss!,
      );
      api.keep(ll);
      return ll;
    });
    const lossOut = api.noGrad(() => api.mul(l, 1));
    const scaled = scaler.scale(l);
    await scaled.backward();
    scaler.unscale_(opt);
    clipGradNorm_(api, params, 1.0);
    scaler.step(opt);
    await scaler.update();
    opt.zeroGrad();
    scaled.dispose();
    await lossOut.item();
    lossOut.dispose();
    input.dispose();
    target.dispose();
    sched.step();
  }
  if (DESTROY) await api.destroy();
}

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU init failed — skipping (no GPU).");
    process.exit(0);
  }
  log(
    `config: N=${N} steps=${STEPS} destroy=${DESTROY} boundMB=${BOUND_MB} ` +
      `model=${CONFIG.numLayers}L/${CONFIG.embedDim}d`,
  );
  const bytesAfter: number[] = [];
  for (let e = 0; e < N; e++) {
    await runOneEngine();
    // With destroy() the device was torn down inside runOneEngine — re-init for
    // the memory read + the next engine. Without destroy(), the device persists.
    if (DESTROY && e < N - 1) await initWebGPU();
    let cur = 0;
    try {
      cur = getGPUMemoryStats().currentBytes;
    } catch {
      cur = 0; // device torn down (last destroy iter) — nothing resident
    }
    bytesAfter.push(cur);
    log(
      `engine ${e}: currentBytes=${(cur / 1e6).toFixed(1)}MB gpuErr=${getGpuUncapturedErrorCount()}`,
    );
  }

  // Growth = worst post-first-engine residency minus the first engine's — the
  // first engine legitimately allocates the working set; a bounded reclaim keeps
  // later engines from ACCUMULATING on top of it.
  const first = bytesAfter[0];
  const maxLater = Math.max(...bytesAfter.slice(1), first);
  const growthMB = (maxLater - first) / 1e6;
  log(
    `SUMMARY: first=${(first / 1e6).toFixed(1)}MB maxLater=${(maxLater / 1e6).toFixed(1)}MB ` +
      `growth=${growthMB.toFixed(1)}MB (bound=${BOUND_MB}MB) destroy=${DESTROY}`,
  );

  let exit = 0;
  if (DESTROY) {
    if (growthMB > BOUND_MB) {
      log(
        `FAIL: growth ${growthMB.toFixed(1)}MB exceeds bound ${BOUND_MB}MB — reclaim leaked.`,
      );
      exit = 1;
    } else {
      log(`PASS: growth ${growthMB.toFixed(1)}MB within bound ${BOUND_MB}MB.`);
    }
  } else {
    log(`(characterization only; no bound asserted with DESTROY=0)`);
  }
  console.log(
    JSON.stringify({
      destroy: DESTROY,
      n: N,
      bytesAfterMB: bytesAfter.map((b) => +(b / 1e6).toFixed(2)),
      growthMB: +growthMB.toFixed(2),
      boundMB: BOUND_MB,
      pass: exit === 0,
    }),
  );
  try {
    destroyWebGPU();
  } catch {
    /* already torn down */
  }
  process.exit(exit);
}
main();
