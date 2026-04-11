<script lang="ts">
  import "../../app.css";
  import { onMount } from "svelte";
  import { createXorWorld, type XorWorld } from "$lib/xor-data";
  import { createModel } from "$lib/model";
  import { config, initConfigUrlSync, resetConfigToDefaults, describeConfigDelta } from "$lib/xor-config.svelte";

  import { THEME, SERIES_PALETTE } from "$lib/theme";
  import { baseChartOpt, chartAxes, legendBlock } from "$lib/chart-helpers";
  import { DemoPage, Figure, LineChart, StatsBar, Stat } from "$lib/components";
  import { BorderedGroup, NumberInput, SelectInput, Slider } from "piston-controls";

  initConfigUrlSync();

  let api: any = $state(null);
  let nn: any = null;
  let Adam: any = null;
  let crossEntropy: any = null;
  let clipGradNorm: any = null;
  let gpuReady = $state(false);
  let gpuError = $state("");

  let training = $state(false);
  let trainingActive = $state(false);
  let step = $state(0);
  let lossHistory: number[] = $state([]);
  let singleANllHistory: number[] = $state([]);
  let singleBNllHistory: number[] = $state([]);
  let mixedNllHistory: number[] = $state([]);

  type ArchivedRun = {
    label: string;
    lossHistory: number[];
    singleANllHistory: number[];
    singleBNllHistory: number[];
    mixedNllHistory: number[];
  };
  let archivedRuns: ArchivedRun[] = $state([]);

  let model: any = $state(null);
  let optimizer: any = $state(null);
  let world: XorWorld | null = $state(null);

  const LOG_INTERVAL = 5;
  const EVAL_INTERVAL = 50;
  const LOG2 = Math.log(2);

  // Push optimizer LR whenever the slider moves.
  $effect(() => { if (optimizer) optimizer.setLR(config.optim.lr); });

  onMount(() => {
    (async () => {
      try {
        const tl = await import("torchlette");
        await tl.initWebGPU();
        api = new tl.Torchlette("webgpu", { enableFusion: true, memoryLimitBytes: 8 * 1024 * 1024 * 1024 });
        nn = tl.nn;
        Adam = tl.Adam;
        crossEntropy = tl.nn.functional.crossEntropy;
        clipGradNorm = tl.nn.clipGradNorm_;
        gpuReady = true;
      } catch (e: any) {
        gpuError = e.message || String(e);
      }
    })();
  });

  // ── Reactive chart options ────────────────────────────────────────────────
  function lineSeriesXY(name: string, data: number[], xStep: number, color: string, isCurrent: boolean) {
    return {
      name,
      type: 'line' as const,
      showSymbol: false,
      data: data.map((y, i) => [i * xStep, y]),
      lineStyle: { width: isCurrent ? 1.8 : 1.2, color, opacity: isCurrent ? 1 : 0.55 },
      itemStyle: { color },
      z: isCurrent ? 10 : 1,
    };
  }
  const runColor = (i: number) => SERIES_PALETTE[(i + 1) % SERIES_PALETTE.length];

  let lossOption = $derived({
    ...baseChartOpt(),
    grid: { top: archivedRuns.length > 0 ? 48 : 28, right: 12, bottom: 24, left: 44 },
    ...(archivedRuns.length > 0
      ? { legend: { ...legendBlock(), top: 2, left: 44, right: 12, type: 'scroll' } }
      : {}),
    ...chartAxes({ yType: 'log' }),
    series: [
      ...archivedRuns.map((r, i) => lineSeriesXY(`#${i + 1} ${r.label}`, r.lossHistory, LOG_INTERVAL, runColor(i), false)),
      lineSeriesXY(`current ${describeConfigDelta(config)}`, lossHistory, LOG_INTERVAL, THEME.accent, true),
    ],
  });

  let nllOption = $derived.by(() => {
    const archived: any[] = [];
    for (let i = 0; i < archivedRuns.length; i++) {
      const r = archivedRuns[i];
      archived.push(lineSeriesXY(`#${i + 1} A`, r.singleANllHistory, EVAL_INTERVAL, runColor(i), false));
      archived.push(lineSeriesXY(`#${i + 1} B`, r.singleBNllHistory, EVAL_INTERVAL, runColor(i), false));
      archived.push(lineSeriesXY(`#${i + 1} mix`, r.mixedNllHistory, EVAL_INTERVAL, runColor(i), false));
    }
    return {
      ...baseChartOpt(),
      legend: {
        ...legendBlock(),
        data: ['single-A (xor only)', 'single-B (F1 only)', 'mixed (both obs)'],
      },
      ...chartAxes({ yMin: 0, yMax: 1.2 }),
      series: [
        ...archived,
        lineSeriesXY('single-A (xor only)', singleANllHistory, EVAL_INTERVAL, THEME.danger, true),
        lineSeriesXY('single-B (F1 only)', singleBNllHistory, EVAL_INTERVAL, THEME.accent2, true),
        lineSeriesXY('mixed (both obs)', mixedNllHistory, EVAL_INTERVAL, THEME.accent, true),
        // Reference line at log(2)
        {
          name: 'log 2',
          type: 'line', showSymbol: false, animation: false,
          data: [],
          markLine: {
            silent: true, symbol: 'none', animation: false,
            data: [{
              yAxis: LOG2,
              lineStyle: { color: THEME.muted, type: 'dashed', width: 1, opacity: 0.6 },
              label: {
                formatter: `log 2 = ${LOG2.toFixed(3)}`,
                position: 'insideEndTop', color: THEME.muted, fontSize: 9,
              },
            }],
          },
        },
      ],
    };
  });

  // ── Model & training ──────────────────────────────────────────────────────
  async function resetModel() {
    step = 0;
    lossHistory = [];
    singleANllHistory = []; singleBNllHistory = []; mixedNllHistory = [];

    await api.beginStep();
    api.manualSeed(config.init.seed);

    world = createXorWorld({ nCompartments: config.world.nCompartments }, config.init.seed);
    world.reseedBatches(config.init.seed);

    model = createModel(api, nn, {
      vocabSize: world.vocabSize,
      seqLen: config.model.seqLen,
      embedDim: config.model.embedDim,
      numHeads: config.model.numHeads,
      numLayers: config.model.numLayers,
      mlpDim: config.model.embedDim * 4,
      posEncoding: config.model.posEncoding,
    });
    if (config.init.weightScale !== 1.0) {
      for (const p of model.parameters()) api.mul_(p, config.init.weightScale);
    }
    optimizer = new Adam(model.parameters(), {
      lr: config.optim.lr, weightDecay: config.optim.weightDecay,
      adamW: config.optim.weightDecay > 0,
    });
    api.endStep();
  }

  async function trainStep() {
    if (!model || !optimizer || !world) return;
    await api.beginStep();

    const batchSize = config.optim.batchSize;
    const seqLen = config.model.seqLen;
    const batch = world.generateTrainingBatch(
      batchSize, seqLen,
      config.objective.singleAPct, config.objective.singleBPct, config.objective.mixedPct,
    );
    const tok = api.tensorFromArray(batch.tokens, [batchSize, seqLen]);
    const N = batchSize * (seqLen - 1);
    const tgtShifted = new Float32Array(N);
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen - 1; t++) {
        tgtShifted[b * (seqLen - 1) + t] = batch.targets[b * seqLen + t];
      }
    }
    const tgt = api.tensorFromArray(tgtShifted, [N]);

    const loss = api.tidy(() => {
      const fwd = model.forward(tok);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([N, world!.vocabSize]);
      const l = crossEntropy(api, logits, tgt, { ignoreIndex: -1 });
      api.keep(l);
      return l;
    });
    tok.dispose(); tgt.dispose();

    const shouldLog = step % LOG_INTERVAL === 0;
    let finishRead: any = null;
    if (shouldLog) {
      const rt = api._runtime();
      finishRead = await rt.startItemReadback(loss._unwrap());
    }

    await loss.backward();
    loss.dispose();

    if (shouldLog && finishRead) {
      const val = await finishRead();
      if (Number.isFinite(val)) {
        lossHistory = [...lossHistory, val];
      } else {
        console.warn(`[step ${step}] non-finite loss (${val})`);
      }
    }

    clipGradNorm(api, model.parameters(), 1.0);
    optimizer.step();
    optimizer.zeroGrad();
    step++;
    await api.endStep();

    if (step % EVAL_INTERVAL === 0 && step > 0) {
      try { await evalAllTypes(); } catch (e: any) {
        console.error('Eval error:', e.message, e.stack);
      }
    }
  }

  async function evalAllTypes() {
    if (!model || !world) return;
    const evals = world.generateEvalBatches(200);
    const V = world.vocabSize;
    const SL = config.model.seqLen;

    const measure = async (e: { tokens: Uint32Array; targets: Uint32Array; seqLen: number; promptLen: number }) => {
      const n = e.targets.length;
      const padded = new Uint32Array(n * SL);
      for (let i = 0; i < n; i++) {
        for (let t = 0; t < e.seqLen; t++) padded[i * SL + t] = e.tokens[i * e.seqLen + t];
      }
      const tok = api.tensorFromArray(padded, [n, SL]);
      const logits = api.tidy(() => {
        const fwd = api.noGrad(() => model.forward(tok));
        api.keep(fwd.logits);
        return fwd.logits;
      });
      const logitsArr = new Float32Array(await logits.cpu());
      logits.dispose(); tok.dispose();
      const pos = e.promptLen - 1;
      let totalNll = 0;
      for (let i = 0; i < n; i++) {
        const off = (i * SL + pos) * V;
        const l0 = logitsArr[off + world!.bitOut[0]];
        const l1 = logitsArr[off + world!.bitOut[1]];
        const mx = Math.max(l0, l1);
        const lse = mx + Math.log(Math.exp(l0 - mx) + Math.exp(l1 - mx));
        const tgt = e.targets[i];
        const tgtLogit = (tgt === 0 ? l0 : l1);
        totalNll += lse - tgtLogit;
      }
      return totalNll / n;
    };

    const aNll = await measure(evals.singleA);
    const bNll = await measure(evals.singleB);
    const mNll = await measure(evals.mixed);
    singleANllHistory = [...singleANllHistory, aNll];
    singleBNllHistory = [...singleBNllHistory, bNll];
    mixedNllHistory = [...mixedNllHistory, mNll];
  }

  async function runTraining() {
    if (!model) await resetModel();
    training = true; trainingActive = true;
    try {
      while (training) {
        await trainStep();
        await new Promise((r) => setTimeout(r, 0));
      }
    } finally {
      trainingActive = false;
    }
  }

  function stopTraining() { training = false; }
  async function stopAndWait() {
    training = false;
    while (trainingActive) await new Promise((r) => setTimeout(r, 10));
  }

  function archiveCurrentRun() {
    if (lossHistory.length === 0) return;
    archivedRuns = [
      ...archivedRuns,
      {
        label: describeConfigDelta(config),
        lossHistory: [...lossHistory],
        singleANllHistory: [...singleANllHistory],
        singleBNllHistory: [...singleBNllHistory],
        mixedNllHistory: [...mixedNllHistory],
      },
    ];
  }

  async function clearAll() {
    await stopAndWait();
    model = null; optimizer = null; world = null; step = 0;
    lossHistory = [];
    singleANllHistory = []; singleBNllHistory = []; mixedNllHistory = [];
    archivedRuns = [];
  }

  let lastLoss = $derived(lossHistory.at(-1) ?? 0);
  let lastA = $derived(singleANllHistory.at(-1) ?? 0);
  let lastB = $derived(singleBNllHistory.at(-1) ?? 0);
  let lastMix = $derived(mixedNllHistory.at(-1) ?? 0);

  // ── Concrete sample display (illustrative, decoupled from training) ───────
  let sampleF1 = $state(1);
  let sampleF2 = $state(0);
  let sampleXor = $derived(sampleF1 ^ sampleF2);
  function resampleExample() {
    sampleF1 = Math.random() < 0.5 ? 0 : 1;
    sampleF2 = Math.random() < 0.5 ? 0 : 1;
  }

  // Token visual styles for the example sequence (light-mode distill palette).
  const TOK_BASE = 'rounded px-1.5 py-0.5 text-[10px] font-mono border';
  const TOK_BOS = `${TOK_BASE} border-[rgba(0,0,0,0.15)] bg-[rgba(0,0,0,0.04)] text-[rgba(0,0,0,0.54)]`;
  const TOK_A   = `${TOK_BASE} border-[#d62728]/40 bg-[#d62728]/10 text-[#9c1f20]`;
  const TOK_B   = `${TOK_BASE} border-[#2ca02c]/40 bg-[#2ca02c]/10 text-[#1f6f1f]`;
  const TOK_ASK = `${TOK_BASE} border-[#ff7f0e]/40 bg-[#ff7f0e]/10 text-[#a85307]`;
  const TOK_OUT = `${TOK_BASE} border-[#1f77b4]/40 bg-[#1f77b4]/10 text-[#13487a]`;
</script>

<DemoPage
  title="Toy Models of Compartmentalization — XOR Disambiguation"
  currentRoute="xor"
  {gpuError}
  {gpuReady}
  {training}
  onTrain={runTraining}
  onStop={stopTraining}
  onReset={async () => { await stopAndWait(); archiveCurrentRun(); await resetModel(); }}
  onDefaults={async () => { await stopAndWait(); resetConfigToDefaults(); }}
  onClear={clearAll}
>
  {#snippet lead()}
    Two bits <em class="italic text-[rgba(0,0,0,0.84)]">F1, F2</em> are sampled fresh every example.
    The model must predict <em class="italic text-[rgba(0,0,0,0.84)]">F2</em>. Compartment A's vocab
    only reports <em class="italic text-[rgba(0,0,0,0.84)]">F1 ⊕ F2</em>. Compartment B's vocab
    only reports <em class="italic text-[rgba(0,0,0,0.84)]">F1</em>. Either alone is uninformative
    about F2 (floor: <em class="italic text-[rgba(0,0,0,0.84)]">log 2 ≈ 0.693 nats</em>). Both
    together determine F2 exactly (<em class="italic text-[rgba(0,0,0,0.84)]">F2 = (F1⊕F2) ⊕ F1</em>).
  {/snippet}

  {#snippet intro()}
    <section class="rounded border border-[rgba(0,0,0,0.08)] bg-white px-4 py-3.5">
      <div class="mb-3 flex items-center gap-3 font-mono text-[12px]">
        <span class="text-[10px] font-semibold uppercase tracking-[0.08em] text-[rgba(0,0,0,0.84)]">Example</span>
        <span class="text-[rgba(0,0,0,0.54)]">
          F1=<strong class="font-semibold text-[rgba(0,0,0,0.84)]">{sampleF1}</strong>,
          F2=<strong class="font-semibold text-[rgba(0,0,0,0.84)]">{sampleF2}</strong>
          <span class="text-[11px]">&nbsp;→&nbsp; xor=F1⊕F2=<strong class="font-semibold text-[rgba(0,0,0,0.84)]">{sampleXor}</strong></span>
        </span>
        <button
          class="ml-auto rounded border border-[rgba(0,0,0,0.15)] px-2 py-0.5 font-mono text-[10px] text-[rgba(0,0,0,0.54)] hover:border-[rgba(0,0,0,0.4)] hover:text-[rgba(0,0,0,0.84)]"
          onclick={resampleExample}
        >
          resample
        </button>
      </div>

      {#each [
        { label: 'single-A', toks: [['BOS', TOK_BOS], [`bit_A(${sampleXor})`, TOK_A], ['ASK_F2', TOK_ASK], ['bit_out(?)', TOK_OUT]],
          info: 'model sees XOR only → P(F2=0)=P(F2=1)=½ → CE = log 2' },
        { label: 'single-B', toks: [['BOS', TOK_BOS], [`bit_B(${sampleF1})`, TOK_B], ['ASK_F2', TOK_ASK], ['bit_out(?)', TOK_OUT]],
          info: 'model sees F1 only → still can\'t determine F2 → CE = log 2' },
        { label: 'mixed', toks: [['BOS', TOK_BOS], [`bit_A(${sampleXor})`, TOK_A], [`bit_B(${sampleF1})`, TOK_B], ['ASK_F2', TOK_ASK], [`bit_out(${sampleF2})`, TOK_OUT]],
          info: `both bits in-context → F2 = ${sampleXor}⊕${sampleF1} = ${sampleF2} → CE = 0` },
      ] as row}
        <div class="mb-1.5 flex flex-wrap items-center gap-1 font-mono text-[11px]">
          <span class="min-w-[68px] text-[10px] font-semibold uppercase tracking-[0.05em] text-[rgba(0,0,0,0.84)]">{row.label}</span>
          {#each row.toks as [text, cls]}<span class={cls}>{text}</span>{/each}
          <span class="ml-3 text-[10px] text-[rgba(0,0,0,0.54)]">{row.info}</span>
        </div>
      {/each}

      <p class="mt-2.5 max-w-[720px] border-t border-[rgba(0,0,0,0.08)] pt-2.5 text-[11px] leading-[1.5] text-[rgba(0,0,0,0.54)]">
        Each sequence is a full training example with standard next-token CE loss. The prediction
        that matters is the last token (bit_out). Training on
        <em class="italic text-[rgba(0,0,0,0.84)]">single-*</em> examples only can never break
        through the log 2 floor — the model has no way to see both observations together, so it
        can't learn the XOR circuit. <em class="italic text-[rgba(0,0,0,0.84)]">mixed</em>
        examples make the composition learnable.
      </p>
    </section>
  {/snippet}

  {#snippet controls()}
    <BorderedGroup title="Objective" id="grp-obj" contentClass="p-2 space-y-2">
      <Slider id="single-a" label="single-A %" min={0} max={100} step={5}
              bind:value={config.objective.singleAPct} unit="%" />
      <Slider id="single-b" label="single-B %" min={0} max={100} step={5}
              bind:value={config.objective.singleBPct} unit="%" />
      <Slider id="mix"      label="mixed %"    min={0} max={100} step={5}
              bind:value={config.objective.mixedPct} unit="%" />
    </BorderedGroup>

    <BorderedGroup title="Init" id="grp-init" contentClass="p-2 space-y-2">
      <NumberInput id="seed" label="Seed" min={0} step={1} bind:value={config.init.seed} />
      <Slider id="ws" label="Weight scale" min={0.01} max={10} step={0.01} useLog={true}
              bind:value={config.init.weightScale} />
    </BorderedGroup>

    <BorderedGroup title="Model" id="grp-model" contentClass="p-2 space-y-2">
      <SelectInput id="dim" label="Embed dim" bind:value={config.model.embedDim}
        options={[{ value: 32 }, { value: 64 }, { value: 128 }]} />
      <Slider id="layers" label="Layers" min={1} max={4} step={1} bind:value={config.model.numLayers} />
      <SelectInput id="heads" label="Heads" bind:value={config.model.numHeads}
        options={[{ value: 1 }, { value: 2 }, { value: 4 }]} />
      <SelectInput id="pe" label="Pos encoding" bind:value={config.model.posEncoding}
        options={[{ value: 'learned' }, { value: 'rope' }]} />
    </BorderedGroup>

    <BorderedGroup title="Optimizer" id="grp-optim" contentClass="p-2 space-y-2">
      <Slider id="lr" label="Learning rate" min={1e-5} max={1e-1} step={1e-5} useLog={true}
              bind:value={config.optim.lr} />
      <Slider id="wd" label="Weight decay" min={0} max={0.2} step={0.01}
              bind:value={config.optim.weightDecay} />
      <Slider id="bs" label="Batch size" min={8} max={64} step={8}
              bind:value={config.optim.batchSize} />
    </BorderedGroup>
  {/snippet}

  {#snippet stats()}
    <StatsBar>
      <Stat label="step" value={step} />
      <Stat label="loss" value={lastLoss > 0 ? lastLoss.toFixed(3) : '—'} />
      <Stat label="A"    value={lastA > 0 ? lastA.toFixed(3) : '—'} />
      <Stat label="B"    value={lastB > 0 ? lastB.toFixed(3) : '—'} />
      <Stat label="mix"  value={lastMix > 0 ? lastMix.toFixed(3) : '—'} />
      {#if archivedRuns.length > 0}
        <button
          class="rounded border border-[rgba(0,0,0,0.15)] px-1.5 py-0.5 text-[10px] text-[rgba(0,0,0,0.54)] hover:border-[rgba(0,0,0,0.4)] hover:text-[rgba(0,0,0,0.84)]"
          onclick={() => { archivedRuns = []; }}
        >
          clear {archivedRuns.length} run{archivedRuns.length > 1 ? 's' : ''}
        </button>
      {/if}
    </StatsBar>
  {/snippet}

  {#snippet figures()}
    <Figure
      title="Training loss"
      caption="Full next-token cross-entropy over all positions. Floors when the model has learned the circuit, limited by the irreducible entropy of predicting the first observation token."
    >
      <LineChart option={lossOption} />
    </Figure>

    <Figure title="F2 prediction NLL per test type">
      {#snippet caption()}
        Cross-entropy of the F2 bit prediction (restricted to the two output bit tokens), on
        held-out eval batches of each sequence type. Single-comp lines should floor near
        <em class="italic text-[rgba(0,0,0,0.84)]">log 2</em> regardless of training regime — that's
        the fundamental info limit of one partial observation. Mixed-test line (blue) should drop
        to 0 if the model learned to combine observations across compartments; stays at log 2
        otherwise.
      {/snippet}
      <LineChart option={nllOption} />
    </Figure>
  {/snippet}
</DemoPage>
