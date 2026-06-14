/**
 * Browser LoRA training-trajectory gate (task #46).
 *
 * The browser LoRA trainer had NO training validation — the only browser spec
 * was a matmul check, and its perf baselines predate the arena-default flip
 * (2026-06-11) and implied-step-boundaries (2026-06-13). This runs the REAL
 * `LoRATrainer.train()` loop (the example's class, beginStep/markStep + Adam +
 * clipGradNorm + the implied-step-boundary path) in a real browser under the
 * CURRENT default, and asserts the two things that actually matter:
 *   1. loss DESCENDS over the run (the training path produces useful gradients);
 *   2. GPU storage stays FLAT (no per-step leak — the arena/lifetime invariants
 *      hold in the browser, not just Node).
 * It also records steady-state tok/s + fwd/bwd ms → the re-baseline for CLAUDE.md
 * "Baseline B".
 *
 * Small random-init model (no pretrained download): exercises the trainer's real
 * control flow, fast enough for CI. Run: `npm run test:browser` (needs xvfb on a
 * headless box: `xvfb-run -a npm run test:browser`).
 *
 * SCOPE / caveats (both tracked as follow-ups, task #47):
 *  - The example model's `Embedding` random init is std=1.0 (the framework
 *    default) vs GPT-2's ~0.02; untamed, from-scratch training NaNs. We scale
 *    wte/wpe down so the path is exercisable — but the model then starts near
 *    the trivial-text optimum, so the descent is shallow (a smoke signal, not a
 *    deep convergence curve). A faithful re-baseline of CLAUDE.md "Baseline B"
 *    needs the real pretrained DistilGPT-2 + LoRA, which this test does not load.
 *  - Storage is NOT flat: the example trainer leaks ~24 storages/step (clip+Adam
 *    on the LoRA raw params); the assertion below guards against WORSENING.
 * This is still a strict upgrade over the prior browser coverage (a matmul-only
 * spec): it proves the browser training path runs end-to-end without NaN, the
 * loss descends, storage growth is bounded, and it records live perf.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  getGPUMemoryStats,
  initWebGPU,
  storageTracker,
  syncWebGPU,
  Torchlette,
} from "../../src/browser";
import { GPT2WithLoRA } from "../../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora";
import { LoRATrainer } from "../../examples/gpt2-lora-trainer/src/lib/torchlette/trainer";

/** Byte-level tokenizer stub: avoids the BPE vocab/merges download. Maps each
 *  UTF-8 byte to an id in [0,256) — within the small model's vocabSize. Only
 *  `encode` is used by LoRATrainer.train(). */
class ByteTokenizer {
  readonly vocabSize = 256;
  encode(text: string): number[] {
    return Array.from(new TextEncoder().encode(text));
  }
  decode(ids: number[]): string {
    return new TextDecoder().decode(new Uint8Array(ids));
  }
}

describe("browser LoRA trainer — training trajectory", () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = await initWebGPU();
  });
  afterAll(async () => {
    if (webgpu) await syncWebGPU();
  });

  it(
    "loss descends and GPU storage stays flat over a real LoRATrainer run",
    async () => {
      if (!webgpu) return;

      const api = new Torchlette("webgpu", { enableFusion: true });
      api.manualSeed(1234);

      // Small random-init GPT-2 + LoRA (no pretrained weights). vocab 256 to
      // match the byte tokenizer; the rest small enough for a fast browser run.
      const model = new GPT2WithLoRA(
        api,
        {
          vocabSize: 256,
          blockSize: 64,
          numLayers: 2,
          numHeads: 2,
          embedDim: 64,
          dropoutRate: 0,
        },
        { rank: 4, alpha: 4 },
        "webgpu",
      );
      // The example model is only ever used with PRETRAINED weights. Its random
      // init uses the framework Embedding's std=1.0 normal (vs GPT-2's ~0.02);
      // with the tied wte as the output projection that explodes the logits
      // (loss ~44 / NaN). Scale wte+wpe into GPT-2's init range so this is a
      // valid training-PATH test, not a from-scratch-numerics test. (Verified in
      // Node: loss starts ~5.6 = ln(256) and descends.)
      await api.beginStep();
      api.mul_(model.wte.weight, 0.04);
      api.mul_(model.wpe.weight, 0.04);
      api.endStep();
      await api.markStep();
      const tokenizer = new ByteTokenizer() as unknown as ConstructorParameters<
        typeof LoRATrainer
      >[2];
      const trainer = new LoRATrainer(api, model, tokenizer);

      // Repeating toy text → the LoRA layers can memorize it (loss must drop).
      const text =
        "the quick brown fox jumps over the lazy dog. ".repeat(64);

      const STEPS = 30;
      const losses: number[] = [];
      const reachable: number[] = [];
      const stepMs: number[] = [];
      let fwdSum = 0;
      let bwdSum = 0;
      let phaseCount = 0;
      const SEQ = 32;
      const BATCH = 2;

      await trainer.train(
        text,
        {
          maxSteps: STEPS,
          batchSize: BATCH,
          seqLength: SEQ,
          learningRate: 1e-3,
          useAMP: false,
          useCheckpointing: false,
          fullFinetune: false,
        },
        {
          onStepEnd: (
            step: number,
            loss: number,
            timeMs: number,
            _mem?: number,
            phases?: {
              forward: number;
              backward: number;
              optimizer: number;
              cleanup: number;
            },
          ) => {
            losses.push(loss);
            reachable.push(storageTracker.stats().reachableStorages);
            stepMs.push(timeMs);
            // Skip warmup (first 5 steps) for the phase baseline.
            if (phases && step >= 5) {
              fwdSum += phases.forward;
              bwdSum += phases.backward;
              phaseCount++;
            }
          },
        },
      );

      expect(losses.length).toBe(STEPS);

      // 1. LOSS DESCENDS. Compare the mean of the first 5 steps to the mean of
      //    the last 5 — robust to per-step noise. A non-descending curve means
      //    the browser training path produced no useful gradient.
      const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;
      const early = mean(losses.slice(0, 5));
      const late = mean(losses.slice(-5));
      expect(
        late,
        `loss did not descend: early=${early.toFixed(4)} late=${late.toFixed(4)} | ${losses.map((l) => l.toFixed(2)).join(",")}`,
      ).toBeLessThan(early);

      // 2. STORAGE FLAT. Reachable-storage count must not grow step-over-step.
      //    Previously this leaked ~24/step — root-caused (task #47) to the
      //    COMPILED replay harvest re-creating VIEW result handles every replay
      //    and rcRetaining their base each time WITHOUT releasing the prior
      //    replay's retain (the lowered path balances it via the wrapper's
      //    view.destroyed; the compiled harvest had no equivalent). It was a
      //    handle leak (bases alias pooled buffers → GPU bytes flat), which is
      //    why it slipped past the profiler (never clips) and the regression
      //    check (measures bytes). FIXED: the compiled plan now owns its
      //    harvest view-base retains and releases the prior replay's set.
      const lateReach = reachable.slice(-11);
      const reachGrowth = Math.max(...lateReach) - Math.min(...lateReach);
      expect(
        reachGrowth,
        `reachable storage grew over the last 10 steps (leak regressed): ${reachable.join(",")}`,
      ).toBeLessThanOrEqual(2);

      // Re-baseline metrics (printed; not asserted — hardware-dependent).
      const steady = stepMs.slice(5);
      const avgStepMs = mean(steady);
      const tokPerSec = (BATCH * SEQ) / (avgStepMs / 1000);
      const mem = getGPUMemoryStats();
      console.log(
        `[lora-baseline] steady ${avgStepMs.toFixed(1)}ms/step | ${tokPerSec.toFixed(0)} tok/s | fwd ${(fwdSum / phaseCount).toFixed(1)}ms bwd ${(bwdSum / phaseCount).toFixed(1)}ms | reachable=${reachable.at(-1)} | curMB=${(mem.currentBytes / 1e6).toFixed(1)} | loss ${early.toFixed(3)}→${late.toFixed(3)}`,
      );
    },
    60000,
  );
});
