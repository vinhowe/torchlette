<script lang="ts">
  import "../../app.css";
  import { onMount } from "svelte";
  import { createModel, MESS3_CONFIG } from "$lib/model";
  import {
    generateBatch,
    generateBatchWithCompartments,
    setTransitionMatrices,
    VOCAB_SIZE_DATA,
  } from "$lib/data";

  // ─────────────────────────────────────────────────────────────────────────
  // Minimal reproduction harness for "Input not ready: adamStep[0]" at ~step 252.
  //
  // This page deliberately has NO Svelte reactivity in the training loop, NO
  // chart components, NO probe, NO render loop, NO remote engine. It imports
  // the same model + data + Torchlette as the mess3 page and runs Adam on
  // the GPU until it crashes or hits the configured step count.
  //
  // Variants are gated on URL params so we can binary-search what triggers
  // the bug:
  //
  //   ?steps=500              — how many steps to run (default 500)
  //   ?lr=2.5e-4              — Adam learning rate (default matches mess3)
  //   ?bs=64                  — batch size
  //   ?seqLen=10              — context length
  //   ?nComp=1                — number of compartments (1 = simple, 2 = mess3 default)
  //   ?selfLoop=0.765         — HMM self-loop prob
  //   ?seed=1024              — torch seed
  //   ?probeEvery=0           — call api.noGrad(model.forward) + .cpu() every N steps (0 = off).
  //                              Re-creates the mess3 probe path without the math.
  //   ?yieldKind=micro        — micro = await Promise.resolve(); raf = requestAnimationFrame; setTimeout = setTimeout(0)
  //   ?stepLog=10             — log a step every N steps (also unconditionally on errors)
  //   ?startGcPressure=0      — set to 1 to allocate junk in JS land between steps to drive GC
  //
  // Open the browser console for full step logs. The on-page log shows
  // sparse milestones + the final crash.
  // ─────────────────────────────────────────────────────────────────────────

  // URL param helpers
  function param(name: string, fallback: string): string {
    if (typeof window === "undefined") return fallback;
    const v = new URLSearchParams(window.location.search).get(name);
    return v ?? fallback;
  }
  const STEPS       = parseInt(param("steps", "500"), 10);
  const LR          = parseFloat(param("lr", "1e-2"));
  const BS          = parseInt(param("bs", "64"), 10);
  const SL          = parseInt(param("seqLen", "10"), 10);
  const N_COMP      = parseInt(param("nComp", "1"), 10);
  const SELF_LOOP   = parseFloat(param("selfLoop", "0.765"));
  const SEED        = parseInt(param("seed", "1024"), 10);
  const PROBE_EVERY = parseInt(param("probeEvery", "0"), 10);
  // probeKind: full = noGrad + tidy + cpu readback (mess3-equivalent)
  //            noReadback = noGrad + tidy, no cpu read
  //            noTidy = noGrad + cpu read, no tidy wrapper
  //            noNoGrad = tidy + cpu read, no noGrad wrapper
  //            forceOnly = just call api.force(model.parameters()[0]) — pure executor force
  //            allocOnly = create a tensor and dispose it (control)
  const PROBE_KIND  = param("probeKind", "full");
  const YIELD_KIND  = param("yieldKind", "micro");
  const STEP_LOG    = parseInt(param("stepLog", "10"), 10);
  const GC_PRESSURE = param("startGcPressure", "0") === "1";
  // Set BEFORE importing torchlette so the executor reads it on first call.
  const NO_ARENA    = param("noArena", "0") === "1";
  if (typeof globalThis !== "undefined" && NO_ARENA) {
    (globalThis as { __torchletteNoArena?: boolean }).__torchletteNoArena = true;
  }

  let started = $state(false);
  let running = $state(false);
  let done = $state(false);
  let crashed = $state<string | null>(null);
  let finalStep = $state(0);
  let firstLoss = $state<number | null>(null);
  let lastLoss  = $state<number | null>(null);
  let lines: string[] = $state([]);

  function log(msg: string) {
    console.log(msg);
    lines = [...lines, msg];
  }

  // Yield primitive — keeps each step on its own task so the browser can
  // scroll/repaint while the test runs.
  function yieldStep(): Promise<void> {
    if (YIELD_KIND === "raf") return new Promise((r) => requestAnimationFrame(() => r()));
    if (YIELD_KIND === "setTimeout") return new Promise((r) => setTimeout(r, 0));
    return Promise.resolve();
  }

  // Imperative training. No reactivity in the body — the only $state reads
  // are out of band (button gating, log appending). Anything reactive that
  // could fan-out updates inside trainStep is on purpose avoided.
  async function runRepro() {
    if (running) return;
    started = true;
    running = true;
    crashed = null;
    finalStep = 0;
    firstLoss = lastLoss = null;
    lines = [];

    log(`[repro252] STEPS=${STEPS} LR=${LR} BS=${BS} SL=${SL} nComp=${N_COMP} probeEvery=${PROBE_EVERY} yield=${YIELD_KIND} gcPressure=${GC_PRESSURE}`);

    let api: any = null;
    let model: any = null;
    let optimizer: any = null;
    let crossEntropy: any = null;
    let api_noGrad: ((fn: () => any) => any) | null = null;

    try {
      const tl = await import("torchlette");
      log(`[repro252] torchlette imported`);
      await tl.initWebGPU();
      log(`[repro252] WebGPU initialized`);
      api = new tl.Torchlette("webgpu", { enableFusion: true, memoryLimitBytes: 8 * 1024 * 1024 * 1024 });
      crossEntropy = tl.nn.functional.crossEntropy;
      api_noGrad = api.noGrad?.bind(api);
      log(`[repro252] Torchlette instance created`);

      setTransitionMatrices(SELF_LOOP);
      api.manualSeed(SEED);
      const vocabSize = VOCAB_SIZE_DATA * N_COMP + 1;

      model = createModel(api, tl.nn, {
        ...MESS3_CONFIG,
        seqLen: SL,
        vocabSize,
        posEncoding: "rope",
      });
      optimizer = new tl.Adam(model.parameters(), { lr: LR });
      log(`[repro252] model + optimizer created (${model.parameters().length} params)`);

      // Junk allocator to drive GC pressure on demand. Held in a closure
      // so V8 can't optimize it away.
      const junkBuckets: number[][] = [];
      const allocateJunk = () => {
        // ~1MB per call, kept for ~32 steps then released (sliding window)
        junkBuckets.push(new Array(250000).fill(Math.random()));
        if (junkBuckets.length > 32) junkBuckets.shift();
      };

      const t0Total = performance.now();
      for (let step = 0; step < STEPS; step++) {
        finalStep = step;
        const t0 = performance.now();

        await api.beginStep();

        const batch =
          N_COMP > 1
            ? generateBatchWithCompartments({ seqLen: SL, batchSize: BS }, N_COMP)
            : generateBatch({ seqLen: SL, batchSize: BS });
        const tok = api.tensorFromArray(batch.tokens, [BS, SL], { dtype: "i32" });
        const tgt = api.tensorFromArray(batch.targets, [BS * (SL - 1)], { dtype: "i32" });

        const loss = api.tidy(() => {
          const fwd = model.forward(tok);
          const logits = fwd.logits.narrow(1, 0, SL - 1).contiguous().reshape([BS * (SL - 1), vocabSize]);
          const l = crossEntropy(api, logits, tgt);
          api.keep(l);
          return l;
        });
        tok.dispose();
        tgt.dispose();

        // Always read the loss, like mess3 does on shouldLog steps. We read
        // every step so the readback path is exercised constantly — both to
        // catch NaNs early and to mirror the worst-case timing.
        const lossVal = await loss.item();
        await loss.backward();
        loss.dispose();
        optimizer.step();
        optimizer.zeroGrad();

        // Optional: emulate the mess3 probe by running an extra forward pass
        // with no_grad and reading a residual. This mirrors what
        // updateBeliefSimplex does without computing R^2. Used to test the
        // hypothesis that the probe corrupts state.
        if (PROBE_EVERY > 0 && step > 0 && step % PROBE_EVERY === 0) {
          if (PROBE_KIND === "allocOnly") {
            // Control: just allocate and dispose a tensor of the same shape.
            const t = api.tensorFromArray(batch.tokens, [BS, SL], { dtype: "i32" });
            t.dispose();
          } else if (PROBE_KIND === "forceOnly") {
            // Force the first model param. Pure force, no extra ops.
            const p = model.parameters()[0];
            await p.cpu();
          } else {
            const probeTok = api.tensorFromArray(batch.tokens, [BS, SL], { dtype: "i32" });
            let residual: any;
            if (PROBE_KIND === "noTidy") {
              // No tidy wrapper — leak intermediates intentionally to see if
              // tidy's cleanup is the corruption source.
              const fwd = api_noGrad!(() => model.forward(probeTok));
              residual = fwd.residuals[fwd.residuals.length - 1];
            } else if (PROBE_KIND === "noNoGrad") {
              // No noGrad wrapper — full autograd pass.
              residual = api.tidy(() => {
                const fwd = model.forward(probeTok);
                const last = fwd.residuals[fwd.residuals.length - 1];
                api.keep(last);
                return last;
              });
            } else {
              // "full" or "noReadback": noGrad + tidy
              residual = api.tidy(() => {
                const fwd = api_noGrad!(() => model.forward(probeTok));
                const last = fwd.residuals[fwd.residuals.length - 1];
                api.keep(last);
                return last;
              });
            }
            probeTok.dispose();
            if (PROBE_KIND !== "noReadback") {
              await residual.cpu();
            }
            residual.dispose();
          }
        }

        await api.endStep();

        if (GC_PRESSURE) allocateJunk();

        if (firstLoss == null) firstLoss = lossVal;
        lastLoss = lossVal;

        if (!Number.isFinite(lossVal)) {
          throw new Error(`Loss became non-finite at step ${step}: ${lossVal}`);
        }

        const elapsed = performance.now() - t0;
        if (step % STEP_LOG === 0) {
          log(`step ${step}: loss=${lossVal.toFixed(4)} (${elapsed.toFixed(0)}ms)`);
        }

        // Yield to the browser so it stays responsive and (importantly) so
        // V8 has a chance to schedule GC between steps.
        await yieldStep();
      }
      const totalMs = performance.now() - t0Total;
      log(`[repro252] DONE: ${STEPS} steps in ${(totalMs / 1000).toFixed(2)}s, avg ${(totalMs / STEPS).toFixed(0)}ms/step`);
      done = true;
    } catch (e: any) {
      crashed = `${e.message}\n${e.stack ?? ""}`;
      log(`[repro252] CRASH at step ${finalStep}: ${e.message}`);
      console.error(e);
    } finally {
      running = false;
    }
  }
</script>

<svelte:head><title>repro252 — minimal browser harness</title></svelte:head>

<div class="mx-auto max-w-[860px] px-6 pt-12 pb-24 text-[rgba(0,0,0,0.84)]">
  <h1 class="mb-3 text-[28px] font-bold leading-[1.15] tracking-[-0.02em]">
    repro252 — minimal browser harness
  </h1>
  <p class="mb-6 max-w-[680px] text-[15px] leading-[1.65] text-[rgba(0,0,0,0.7)]">
    Strips the mess3 demo down to: import torchlette, init WebGPU, build model + Adam, run Adam steps in a loop, log loss.
    No charts, no probe, no render loop, no Svelte reactivity inside the loop, no remote engine.
    Tweak via URL params (see top of source) to bisect what triggers
    <code class="rounded bg-[rgba(0,0,0,0.04)] px-1 font-mono text-[13px]">Input not ready: adamStep[0]</code>.
  </p>

  <div class="mb-4 flex flex-wrap items-center gap-x-3 gap-y-2 border-y border-[rgba(0,0,0,0.08)] py-3 font-mono text-[12px] text-[rgba(0,0,0,0.54)]">
    <button
      class="rounded border border-[rgba(0,0,0,0.15)] bg-blue-100 px-3 py-1 text-[rgba(0,0,0,0.84)] hover:border-[rgba(0,0,0,0.4)] disabled:opacity-50"
      onclick={runRepro}
      disabled={running}
    >
      {running ? `running… step ${finalStep}` : started ? "rerun" : "start"}
    </button>
    <span>steps={STEPS}</span>
    <span>lr={LR}</span>
    <span>bs={BS}</span>
    <span>sl={SL}</span>
    <span>nComp={N_COMP}</span>
    <span>probeEvery={PROBE_EVERY}</span>
    <span>probeKind={PROBE_KIND}</span>
    <span>noArena={NO_ARENA ? "1" : "0"}</span>
    <span>yield={YIELD_KIND}</span>
    {#if firstLoss != null}<span>first={firstLoss.toFixed(4)}</span>{/if}
    {#if lastLoss != null}<span>last={lastLoss.toFixed(4)}</span>{/if}
  </div>

  {#if crashed}
    <div class="mb-4 border-l-[3px] border-l-[#d62728] bg-[rgba(214,39,40,0.05)] px-[14px] py-2 font-mono text-[12px]">
      <div class="font-semibold text-[#9c1f20]">CRASHED at step {finalStep}</div>
      <pre class="mt-1 whitespace-pre-wrap text-[11px] leading-[1.5] text-[rgba(0,0,0,0.7)]">{crashed}</pre>
    </div>
  {:else if done}
    <div class="mb-4 border-l-[3px] border-l-[#2ca02c] bg-[rgba(46,160,46,0.05)] px-[14px] py-2 font-mono text-[12px]">
      Completed {STEPS} steps without error.
    </div>
  {/if}

  <pre class="max-h-[60vh] overflow-y-auto rounded border border-[rgba(0,0,0,0.08)] bg-white p-3 font-mono text-[11px] leading-[1.5] text-[rgba(0,0,0,0.7)]">{lines.join("\n")}</pre>

  <details class="mt-6 text-[12px] text-[rgba(0,0,0,0.54)]">
    <summary class="cursor-pointer">URL params reference</summary>
    <ul class="ml-5 mt-2 list-disc space-y-1">
      <li><code class="font-mono">?steps=500</code> — total steps to run</li>
      <li><code class="font-mono">?lr=2.5e-4</code> — Adam lr (mess3 url default 2.51e-4)</li>
      <li><code class="font-mono">?bs=64</code> / <code class="font-mono">?seqLen=10</code> / <code class="font-mono">?nComp=1</code></li>
      <li><code class="font-mono">?probeEvery=50</code> — emulate the mess3 probe (extra noGrad fwd + cpu read every N steps)</li>
      <li><code class="font-mono">?yieldKind=micro|raf|setTimeout</code> — yield primitive between steps</li>
      <li><code class="font-mono">?startGcPressure=1</code> — allocate junk in JS land each step to drive GC</li>
      <li><code class="font-mono">?stepLog=10</code> — log every N steps</li>
    </ul>
  </details>
</div>
