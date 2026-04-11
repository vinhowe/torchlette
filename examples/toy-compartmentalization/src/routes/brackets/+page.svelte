<script lang="ts">
  import "../../app.css";
  import { onMount, untrack } from "svelte";
  import { createBracketsWorld, type BracketsWorld } from "$lib/brackets-data";
  import { createModel } from "$lib/model";
  import { config, initConfigUrlSync, resetConfigToDefaults, describeConfigDelta } from "$lib/brackets-config.svelte";

  import { THEME, SERIES_PALETTE, COMP_COLORS } from "$lib/theme";
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
  /** perCompAccHistories[c][evalIdx] = accuracy of compartment c at eval step evalIdx */
  let perCompAccHistories: number[][] = $state([]);
  let accMixHistory: number[] = $state([]);
  let cosSimHistory: number[] = $state([]);

  type ArchivedRun = {
    label: string;
    lossHistory: number[];
    perCompAccHistories: number[][];
    accMixHistory: number[];
    cosSimHistory: number[];
  };
  let archivedRuns: ArchivedRun[] = $state([]);
  let model: any = $state(null);
  let optimizer: any = $state(null);
  let world: BracketsWorld | null = $state(null);

  const LOG_INTERVAL = 5;
  const EVAL_INTERVAL = 50;

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
      resampleExample();
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
  const compColor = (c: number) => COMP_COLORS[c % COMP_COLORS.length];

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

  let accOption = $derived.by(() => {
    const archived: any[] = [];
    for (let i = 0; i < archivedRuns.length; i++) {
      const r = archivedRuns[i];
      for (let c = 0; c < r.perCompAccHistories.length; c++) {
        archived.push(lineSeriesXY(`#${i + 1} c${c}`, r.perCompAccHistories[c], EVAL_INTERVAL, runColor(i), false));
      }
      archived.push(lineSeriesXY(`#${i + 1} mix`, r.accMixHistory, EVAL_INTERVAL, runColor(i), false));
    }
    const currentComp = perCompAccHistories.map((h, c) =>
      lineSeriesXY(`comp ${c}`, h, EVAL_INTERVAL, compColor(c), true),
    );
    const nClose = config.world.nCompartments * config.world.nPairTypes;
    const chance = 1 / nClose;
    return {
      ...baseChartOpt(),
      legend: {
        ...legendBlock(),
        data: [...perCompAccHistories.map((_, c) => `comp ${c}`), 'mixed'],
      },
      ...chartAxes({ yMin: 0, yMax: 1 }),
      series: [
        ...archived,
        ...currentComp,
        lineSeriesXY('mixed', accMixHistory, EVAL_INTERVAL, THEME.accent, true),
        {
          name: 'chance', type: 'line', showSymbol: false, data: [],
          markLine: {
            silent: true, symbol: 'none', animation: false,
            data: [{
              yAxis: chance,
              lineStyle: { color: THEME.muted, type: 'dashed', width: 1, opacity: 0.6 },
              label: {
                formatter: `chance = 1/${nClose} = ${(chance * 100).toFixed(0)}%`,
                position: 'insideEndTop', color: THEME.muted, fontSize: 9,
              },
            }],
          },
        },
      ],
    };
  });

  let cosSimOption = $derived({
    ...baseChartOpt(),
    grid: { top: archivedRuns.length > 0 ? 48 : 28, right: 12, bottom: 24, left: 44 },
    ...(archivedRuns.length > 0
      ? { legend: { ...legendBlock(), top: 2, left: 44, right: 12, type: 'scroll' } }
      : {}),
    ...chartAxes({ yMin: -1, yMax: 1 }),
    series: [
      ...archivedRuns.map((r, i) => lineSeriesXY(`#${i + 1} ${r.label}`, r.cosSimHistory, EVAL_INTERVAL, runColor(i), false)),
      lineSeriesXY(`current ${describeConfigDelta(config)}`, cosSimHistory, EVAL_INTERVAL, THEME.warning, true),
    ],
  });

  // ── Model & training ──────────────────────────────────────────────────────
  async function resetModel() {
    step = 0;
    lossHistory = [];
    perCompAccHistories = []; accMixHistory = []; cosSimHistory = [];

    await api.beginStep();
    api.manualSeed(config.init.seed);
    world = createBracketsWorld(
      {
        contentLen: config.world.contentLen,
        maxDepth: config.world.maxDepth,
        nPairTypes: config.world.nPairTypes,
        nCompartments: config.world.nCompartments,
      },
      config.init.seed,
    );
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
    resampleExample();
  }

  async function trainStep() {
    if (!model || !optimizer || !world) return;
    await api.beginStep();
    const batchSize = config.optim.batchSize;
    const seqLen = config.model.seqLen;
    const batch = world.generateTrainingBatch(
      batchSize, seqLen,
      config.objective.singlePct, config.objective.mixedPct,
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
        console.warn(`[step ${step}] non-finite loss (${val}) — skipping chart point`);
      }
    }
    clipGradNorm(api, model.parameters(), 1.0);
    optimizer.step();
    optimizer.zeroGrad();
    step++;
    await api.endStep();

    if (step % EVAL_INTERVAL === 0 && step > 0) {
      try { await evalCloseAcc(); } catch (e: any) {
        console.error('Eval error:', e.message, e.stack);
      }
    }
  }

  async function evalCloseAcc() {
    if (!model || !world) return;
    const evals = world.generateEvalBatches(80);
    const V = world.vocabSize;
    const SL = config.model.seqLen;
    const closeCandidates = world.allCloseTokens;
    const nC = Math.min(4, world.nCompartments);

    const measure = async (e: ReturnType<typeof world.generateEvalBatches>['mixed']) => {
      const n = e.closePositions.length;
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
      let correct = 0, total = 0;
      for (let i = 0; i < n; i++) {
        for (const c of e.closePositions[i]) {
          const off = (i * SL + c.pos) * V;
          let best = -Infinity, arg = -1;
          for (const v of closeCandidates) {
            if (logitsArr[off + v] > best) { best = logitsArr[off + v]; arg = v; }
          }
          if (arg === c.correct) correct++;
          total++;
        }
      }
      return total > 0 ? correct / total : 0;
    };

    if (perCompAccHistories.length !== nC) {
      perCompAccHistories = Array.from({ length: nC }, () => []);
    }
    const newHistories = perCompAccHistories.map(h => [...h]);
    for (let c = 0; c < nC; c++) {
      const acc = await measure(evals.perComp[c]);
      newHistories[c].push(acc);
    }
    perCompAccHistories = newHistories;

    const mAcc = await measure(evals.mixed);
    accMixHistory = [...accMixHistory, mAcc];

    if (world.nCompartments >= 2) {
      const paired = world.generatePairedEvalBatch(40);
      const N = 40;
      const SLp = paired.seqLen;
      const d = config.model.embedDim;
      const midLayer = (residuals: any[]) => residuals[Math.floor((residuals.length - 1) / 2)];

      const getResiduals = async (compTokens: Uint32Array) => {
        const padded = new Uint32Array(N * SL);
        for (let i = 0; i < N; i++) {
          for (let t = 0; t < SLp; t++) padded[i * SL + t] = compTokens[i * SLp + t];
        }
        const tok = api.tensorFromArray(padded, [N, SL]);
        const res = api.tidy(() => {
          const fwd = api.noGrad(() => model.forward(tok));
          const mid = midLayer(fwd.residuals);
          api.keep(mid);
          return mid;
        });
        const arr = new Float32Array(await res.cpu());
        res.dispose(); tok.dispose();
        return arr;
      };

      const resA = await getResiduals(paired.perComp[0]);
      const resB = await getResiduals(paired.perComp[Math.min(1, world.nCompartments - 1)]);

      let totalCos = 0, count = 0;
      for (let i = 0; i < N; i++) {
        for (let t = 0; t < SLp; t++) {
          const off = (i * SL + t) * d;
          let dot = 0, normA = 0, normB = 0;
          for (let j = 0; j < d; j++) {
            const a = resA[off + j], b = resB[off + j];
            dot += a * b; normA += a * a; normB += b * b;
          }
          const denom = Math.sqrt(normA) * Math.sqrt(normB);
          if (denom > 1e-8) { totalCos += dot / denom; count++; }
        }
      }
      cosSimHistory = [...cosSimHistory, count > 0 ? totalCos / count : 0];
    }
  }

  async function runTraining() {
    if (!model) await resetModel();
    training = true; trainingActive = true;
    try {
      while (training) {
        await trainStep();
        await new Promise((r) => setTimeout(r, 0));
      }
    } finally { trainingActive = false; }
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
        perCompAccHistories: perCompAccHistories.map(h => [...h]),
        accMixHistory: [...accMixHistory],
        cosSimHistory: [...cosSimHistory],
      },
    ];
  }
  async function clearAll() {
    await stopAndWait();
    model = null; optimizer = null; world = null; step = 0;
    lossHistory = [];
    perCompAccHistories = []; accMixHistory = []; cosSimHistory = [];
    archivedRuns = [];
  }

  let lastLoss = $derived(lossHistory.at(-1) ?? 0);
  let lastCompAccs = $derived(perCompAccHistories.map(h => h.at(-1) ?? 0));
  let lastMix = $derived(accMixHistory.at(-1) ?? 0);
  let lastCosSim = $derived(cosSimHistory.at(-1) ?? 0);

  // ── Live example display ──────────────────────────────────────────────────
  let exampleSeq: number[] = $state([]);
  let exampleWorld: BracketsWorld | null = $state(null);
  let exampleKind: 'A' | 'B' | 'mix' = $state('mix');
  const OPEN_CHARS = ['(', '[', '{', '<'];
  const CLOSE_CHARS = [')', ']', '}', '>'];
  function resampleExample() {
    exampleWorld = createBracketsWorld(
      {
        contentLen: config.world.contentLen,
        maxDepth: config.world.maxDepth,
        nPairTypes: config.world.nPairTypes,
        nCompartments: config.world.nCompartments,
      },
      Math.floor(Math.random() * 100000),
    );
    const s = exampleWorld.sampleSequence(exampleKind);
    exampleSeq = s.tokens;
  }
  // Re-sample when any structural config changes.
  $effect(() => {
    config.world.contentLen; config.world.maxDepth; config.world.nPairTypes; config.world.nCompartments;
    untrack(resampleExample);
  });
  function tokSymbol(tok: number): string {
    if (!exampleWorld) return '';
    const info = exampleWorld.tokenInfo(tok);
    if (!info) return '?';
    if (info.kind === 'bos') return 'BOS';
    return info.kind === 'open' ? OPEN_CHARS[info.type] : CLOSE_CHARS[info.type];
  }
  function tokColor(tok: number): string {
    if (!exampleWorld) return THEME.muted;
    const info = exampleWorld.tokenInfo(tok);
    if (!info || info.kind === 'bos') return THEME.muted;
    return COMP_COLORS[info.comp % COMP_COLORS.length];
  }

  const TOK_BASE = 'inline-block min-w-[20px] rounded border px-1.5 py-0.5 text-center font-mono text-[12px] font-semibold';
</script>

<DemoPage
  title="Toy Models of Compartmentalization — Bracket Matching"
  currentRoute="brackets"
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
    Each compartment has its own pair of brackets. Comp A uses
    <em class="italic text-[rgba(0,0,0,0.84)]">(</em>,
    <em class="italic text-[rgba(0,0,0,0.84)]">)</em>; comp B uses
    <em class="italic text-[rgba(0,0,0,0.84)]">[</em>,
    <em class="italic text-[rgba(0,0,0,0.84)]">]</em>. Balanced sequences have every opener
    closed by a matching closer <em class="italic text-[rgba(0,0,0,0.84)]">from the same
    compartment</em>, at correct nesting depth. The stack-matching circuit is identical for both
    vocabularies — does training on single-compartment sequences alone let the model handle
    mixed-compartment test sequences where the two vocabularies interleave?
  {/snippet}

  {#snippet intro()}
    <section class="rounded border border-[rgba(0,0,0,0.08)] bg-white px-4 py-3.5">
      <div class="mb-3 flex items-center gap-3 font-mono text-[12px]">
        <span class="text-[10px] font-semibold uppercase tracking-[0.08em] text-[rgba(0,0,0,0.84)]">Example</span>
        <div class="flex gap-1">
          {#each [['A', 'single-A'], ['B', 'single-B'], ['mix', 'mixed']] as [k, label]}
            <button
              class="rounded border px-2 py-0.5 text-[10px]
                {exampleKind === k
                  ? 'border-[rgba(0,0,0,0.84)] text-[rgba(0,0,0,0.84)]'
                  : 'border-[rgba(0,0,0,0.15)] text-[rgba(0,0,0,0.54)] hover:border-[rgba(0,0,0,0.4)] hover:text-[rgba(0,0,0,0.84)]'}"
              onclick={() => { exampleKind = k as 'A' | 'B' | 'mix'; resampleExample(); }}
            >
              {label}
            </button>
          {/each}
        </div>
        <button
          class="ml-auto rounded border border-[rgba(0,0,0,0.15)] px-2 py-0.5 font-mono text-[10px] text-[rgba(0,0,0,0.54)] hover:border-[rgba(0,0,0,0.4)] hover:text-[rgba(0,0,0,0.84)]"
          onclick={resampleExample}
        >
          resample
        </button>
      </div>
      <div class="flex flex-wrap items-center gap-1 font-mono text-[12px]">
        {#each exampleSeq as tok}
          {@const color = tokColor(tok)}
          <span
            class={TOK_BASE}
            style="border-color: {color}66; background-color: {color}1a; color: {color};"
          >
            {tokSymbol(tok)}
          </span>
        {/each}
      </div>
      <p class="mt-2.5 max-w-[720px] border-t border-[rgba(0,0,0,0.08)] pt-2.5 text-[11px] leading-[1.5] text-[rgba(0,0,0,0.54)]">
        Bracket type color-coded by compartment. A closer must match its corresponding opener's
        compartment. In a mixed sequence, stack state can interleave comps — e.g.
        <code class="rounded bg-[rgba(0,0,0,0.04)] px-1 font-mono text-[11px] text-[rgba(0,0,0,0.84)]">( [ ] )</code>
        is valid but
        <code class="rounded bg-[rgba(0,0,0,0.04)] px-1 font-mono text-[11px] text-[rgba(0,0,0,0.84)]">( [ ) ]</code>
        isn't (closers don't match their openers' comps).
      </p>
    </section>
  {/snippet}

  {#snippet controls()}
    <BorderedGroup title="World" id="grp-world" contentClass="p-2 space-y-2">
      <Slider id="len"   label="Content length" min={4} max={16} step={2}
              bind:value={config.world.contentLen} />
      <Slider id="depth" label="Max depth"      min={2} max={8}  step={1}
              bind:value={config.world.maxDepth} />
      <Slider id="types" label="Pair types"     min={1} max={4}  step={1}
              bind:value={config.world.nPairTypes} />
      <NumberInput id="comps" label="Compartments" min={2} max={8} step={1}
                   bind:value={config.world.nCompartments} />
    </BorderedGroup>

    <BorderedGroup title="Objective" id="grp-obj" contentClass="p-2 space-y-2">
      <Slider id="single" label="single %" min={0} max={100} step={1}
              bind:value={config.objective.singlePct} unit="%" />
      <Slider id="mix"    label="mixed %"  min={0} max={100} step={1}
              bind:value={config.objective.mixedPct} unit="%" />
    </BorderedGroup>

    <BorderedGroup title="Init" id="grp-init" contentClass="p-2 space-y-2">
      <NumberInput id="seed" label="Seed" min={0} step={1} bind:value={config.init.seed} />
      <Slider id="ws" label="Weight scale" min={0.01} max={10} step={0.01} useLog={true}
              bind:value={config.init.weightScale} />
    </BorderedGroup>

    <BorderedGroup title="Model" id="grp-model" contentClass="p-2 space-y-2">
      <SelectInput id="dim"   label="Embed dim" bind:value={config.model.embedDim}
                   options={[{ value: 32 }, { value: 64 }, { value: 128 }]} />
      <Slider      id="layers" label="Layers"   min={1} max={4} step={1}
                   bind:value={config.model.numLayers} />
      <SelectInput id="heads" label="Heads" bind:value={config.model.numHeads}
                   options={[{ value: 1 }, { value: 2 }, { value: 4 }]} />
      <SelectInput id="pe"    label="Pos encoding" bind:value={config.model.posEncoding}
                   options={[{ value: 'learned' }, { value: 'rope' }]} />
    </BorderedGroup>

    <BorderedGroup title="Optimizer" id="grp-optim" contentClass="p-2 space-y-2">
      <Slider id="lr" label="Learning rate" min={1e-5} max={1e-1} step={1e-5} useLog={true}
              bind:value={config.optim.lr} />
      <Slider id="wd" label="Weight decay" min={0} max={0.5} step={0.01}
              bind:value={config.optim.weightDecay} />
      <NumberInput id="bs" label="Batch size" min={1} max={128} step={1}
                   bind:value={config.optim.batchSize} />
    </BorderedGroup>
  {/snippet}

  {#snippet stats()}
    <StatsBar>
      <Stat label="step" value={step} />
      <Stat label="loss" value={lastLoss > 0 ? lastLoss.toFixed(3) : '—'} />
      {#if lastCompAccs.length > 0}
        <Stat label="single" value={`[${lastCompAccs.slice(0, 4).map(a => (a * 100).toFixed(0) + '%').join(', ')}]`} />
      {:else}
        <Stat label="single" value="—" />
      {/if}
      <Stat label="mix" value={lastMix > 0 ? (lastMix * 100).toFixed(0) + '%' : '—'} />
      <Stat label="cos" value={lastCosSim ? lastCosSim.toFixed(3) : '—'} />
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
      caption="Full next-token cross-entropy over all positions."
    >
      <LineChart option={lossOption} />
    </Figure>

    <Figure title="Close-bracket accuracy per test type">
      {#snippet caption()}
        At every position where the ground truth is a close token, measure whether the model's
        argmax over <code class="rounded bg-[rgba(0,0,0,0.04)] px-1 font-mono">(</code>,
        <code class="rounded bg-[rgba(0,0,0,0.04)] px-1 font-mono">)</code>,
        <code class="rounded bg-[rgba(0,0,0,0.04)] px-1 font-mono">[</code>,
        <code class="rounded bg-[rgba(0,0,0,0.04)] px-1 font-mono">]</code> restricted to close
        tokens matches the correct compartment. Pure single-comp training converges to 100% on its
        own test distribution but typically leaves a gap on mixed test (the model hasn't learned to
        track interleaved compartment stacks). Mixed training closes that gap.
      {/snippet}
      <LineChart option={accOption} />
    </Figure>

    <Figure
      title="Cross-compartment cosine similarity"
      caption="Cosine similarity of middle-layer residual stream activations for structurally-identical bracket sequences rendered in compartment 0 vs compartment 1. High = unified representation (same bracket structure maps to similar internals regardless of which comp's tokens are used). Low = compartmentalized (separate internal representations per vocab)."
    >
      <LineChart option={cosSimOption} />
    </Figure>
  {/snippet}
</DemoPage>
