<script lang="ts">
  import "../../app.css";
  import { onMount } from "svelte";
  import { createBioWorld, type BioWorld, type BioConfig } from "$lib/bio-data";
  import { createModel } from "$lib/model";
  import { config, initConfigUrlSync, resetConfigToDefaults, describeConfigDelta } from "$lib/bio-config.svelte";

  import { THEME, SERIES_PALETTE, COMP_COLORS } from "$lib/theme";
  import { baseChartOpt, chartAxes, legendBlock } from "$lib/chart-helpers";
  import { DemoPage, Figure, LineChart, StatsBar, Stat } from "$lib/components";
  import { BorderedGroup, CheckboxInput, NumberInput, SelectInput, Slider } from "piston-controls";

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
  let cosineSimilarityHistory: number[] = $state([]);
  // accuracyHistories[c][i] = accuracy of compartment c at eval step i
  let accuracyHistories: number[][] = $state([]);
  // Translation loss on the second half (partB) of mirror examples, averaged over pairs.
  let translationLossHistory: number[] = $state([]);

  type ArchivedRun = {
    label: string;
    lossHistory: number[];
    cosineHistory: number[];
    accHistories: number[][];
    translationLossHistory: number[];
  };
  let archivedRuns: ArchivedRun[] = $state([]);
  let model: any = $state(null);
  let optimizer: any = $state(null);
  let world: BioWorld | null = $state(null);

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

  let translLossOption = $derived({
    ...baseChartOpt(),
    grid: { top: archivedRuns.length > 0 ? 48 : 28, right: 12, bottom: 24, left: 44 },
    ...(archivedRuns.length > 0
      ? { legend: { ...legendBlock(), top: 2, left: 44, right: 12, type: 'scroll' } }
      : {}),
    ...chartAxes({ yType: 'log' }),
    series: [
      ...archivedRuns.map((r, i) => lineSeriesXY(`#${i + 1} ${r.label}`, r.translationLossHistory, EVAL_INTERVAL, runColor(i), false)),
      lineSeriesXY(`current ${describeConfigDelta(config)}`, translationLossHistory, EVAL_INTERVAL, THEME.danger, true),
    ],
  });

  let accuracyOption = $derived.by(() => {
    const series = [
      ...archivedRuns.flatMap((r, i) =>
        r.accHistories.map((h, c) =>
          lineSeriesXY(`#${i + 1} c${c}`, h, EVAL_INTERVAL, runColor(i), false),
        ),
      ),
      ...accuracyHistories.map((h, c) =>
        lineSeriesXY(`comp ${c}`, h, EVAL_INTERVAL, compColor(c), true),
      ),
    ];
    return {
      ...baseChartOpt(),
      legend: { ...legendBlock(), data: accuracyHistories.map((_, c) => `comp ${c}`) },
      ...chartAxes({ yMin: 0, yMax: 1 }),
      series,
    };
  });

  let cosSimOption = $derived({
    ...baseChartOpt(),
    grid: { top: archivedRuns.length > 0 ? 48 : 28, right: 12, bottom: 24, left: 44 },
    ...(archivedRuns.length > 0
      ? { legend: { ...legendBlock(), top: 2, left: 44, right: 12, type: 'scroll' } }
      : {}),
    ...chartAxes({ yMin: -0.2, yMax: 1 }),
    series: [
      ...archivedRuns.map((r, i) => lineSeriesXY(`#${i + 1} ${r.label}`, r.cosineHistory, EVAL_INTERVAL, runColor(i), false)),
      lineSeriesXY(`current ${describeConfigDelta(config)}`, cosineSimilarityHistory, EVAL_INTERVAL, THEME.warning, true),
    ],
  });

  // ── Model & training ──────────────────────────────────────────────────────
  async function resetModel() {
    step = 0;
    lossHistory = [];
    cosineSimilarityHistory = [];
    accuracyHistories = [];
    translationLossHistory = [];

    api.manualSeed(config.init.seed);

    const w = config.world;
    const bioConfig: BioConfig = {
      nEntities: w.nEntities,
      nAttributes: w.nAttributes,
      nValues: w.nValues,
      nCompartments: w.nCompartments,
      tokensPerEntity: w.tokensPerEntity,
      tokensPerValue: w.tokensPerValue,
      bankSize: Math.max(200, w.nEntities + w.nValues + 10),
      mixCompartments: w.mixCompartments,
    };
    world = createBioWorld(bioConfig);
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
    if (config.init.embedScale !== 1.0) api.mul_(model.wte.weight, config.init.embedScale);
    if (config.init.headScale !== 1.0) api.mul_(model.lmHead.weight, config.init.headScale);
    if (config.init.residualZeroInit) {
      for (const layer of model.layers) {
        api.mul_(layer.outProj.weight, 0);
        api.mul_(layer.fc2.weight, 0);
      }
    }
    if (config.init.tieInit) await tieCompartmentEmbeddings();
    optimizer = new Adam(model.parameters(), {
      lr: config.optim.lr,
      weightDecay: config.optim.weightDecay,
      adamW: config.optim.weightDecay > 0,
    });
  }

  async function tieCompartmentEmbeddings() {
    if (!world || !model) return;
    const W = world.vocabSize;
    const D = config.model.embedDim;
    const wteData = new Float32Array(await model.wte.weight.cpu());
    const lmData = new Float32Array(await model.lmHead.weight.cpu());
    const copyRow = (src: number, dst: number) => {
      for (let d = 0; d < D; d++) {
        wteData[dst * D + d] = wteData[src * D + d];
        lmData[dst * D + d] = lmData[src * D + d];
      }
    };
    for (let c = 1; c < world.config.nCompartments; c++) {
      for (let eid = 0; eid < world.config.nEntities; eid++) {
        for (let t = 0; t < world.config.tokensPerEntity; t++) {
          copyRow(world.entityTokens[0][eid][t], world.entityTokens[c][eid][t]);
        }
      }
      for (let a = 0; a < world.config.nAttributes; a++) {
        copyRow(world.attrTokens[0][a], world.attrTokens[c][a]);
        copyRow(world.qaTokens[0][a], world.qaTokens[c][a]);
      }
      for (let v = 0; v < world.config.nValues; v++) {
        for (let t = 0; t < world.config.tokensPerValue; t++) {
          copyRow(world.valueTokens[0][v][t], world.valueTokens[c][v][t]);
        }
      }
    }
    const wteNew = api.tensorFromArray(wteData, [W, D]);
    const lmNew = api.tensorFromArray(lmData, [W, D]);
    api.copy_(model.wte.weight, wteNew);
    api.copy_(model.lmHead.weight, lmNew);
    wteNew.dispose(); lmNew.dispose();
  }

  async function trainStep() {
    if (!model || !optimizer || !world) return;
    await api.beginStep();

    const batchSize = config.optim.batchSize;
    const seqLen = config.model.seqLen;
    const batch = world.generateTrainingBatch(batchSize, seqLen, config.objective.translationPct / 100, config.objective.translationMode);
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
      try {
        await evalCosineSimilarity();
      } catch (e: any) {
        console.error('Eval error:', e.message, e.stack);
      }
    }
  }

  async function evalCosineSimilarity() {
    if (!model || !world) return;

    const evalIds = Array.from({ length: world.config.nEntities }, (_, i) => i);
    const attrId = 0;
    const evalData = world.generateEvalBatch(evalIds, attrId);
    const N = evalIds.length;
    const SL = config.model.seqLen;
    const V = world.vocabSize;
    const nComp = Math.min(4, world.config.nCompartments);
    const midLayer = (residuals: any[]) => residuals[Math.floor((residuals.length - 1) / 2)];

    const perCompMiddleRes: Float32Array[] = [];
    const perCompLogits: Float32Array[] = [];
    for (let c = 0; c < nComp; c++) {
      const padded = new Uint32Array(N * SL);
      for (let i = 0; i < N; i++) {
        for (let t = 0; t < evalData.seqLen; t++) {
          padded[i * SL + t] = evalData.tokens[c][i * evalData.seqLen + t];
        }
      }
      const tok = api.tensorFromArray(padded, [N, SL]);
      const [mid, logits] = api.tidy(() => {
        const fwd = api.noGrad(() => model.forward(tok));
        api.keep(midLayer(fwd.residuals), fwd.logits);
        return [midLayer(fwd.residuals), fwd.logits];
      });
      perCompMiddleRes.push(new Float32Array(await mid.cpu()));
      perCompLogits.push(new Float32Array(await logits.cpu()));
      mid.dispose(); logits.dispose(); tok.dispose();
    }

    const d = config.model.embedDim;
    const cosPos = evalData.promptLen - 1;
    const resA = perCompMiddleRes[0];
    const resB = perCompMiddleRes[Math.min(1, nComp - 1)];
    let totalCos = 0;
    for (let i = 0; i < N; i++) {
      const off = (i * SL + cosPos) * d;
      let dot = 0, normA = 0, normB = 0;
      for (let j = 0; j < d; j++) {
        const a = resA[off + j], b = resB[off + j];
        dot += a * b; normA += a * a; normB += b * b;
      }
      const denom = Math.sqrt(normA) * Math.sqrt(normB);
      if (denom > 1e-8) totalCos += dot / denom;
    }
    cosineSimilarityHistory = [...cosineSimilarityHistory, totalCos / N];

    if (accuracyHistories.length !== nComp) {
      accuracyHistories = Array.from({ length: nComp }, () => []);
    }
    const newHistories = accuracyHistories.map((h) => [...h]);
    for (let c = 0; c < nComp; c++) {
      const logits = perCompLogits[c];
      const targets = evalData.targets[c];
      let correct = 0;
      for (let i = 0; i < N; i++) {
        let ok = true;
        for (let t = 0; t < evalData.targetLen; t++) {
          const pos = evalData.promptLen - 1 + t;
          const off = (i * SL + pos) * V;
          let mx = -Infinity, arg = 0;
          for (let v = 0; v < V; v++) if (logits[off + v] > mx) { mx = logits[off + v]; arg = v; }
          if (arg !== targets[i * evalData.targetLen + t]) ok = false;
        }
        if (ok) correct++;
      }
      newHistories[c].push(correct / N);
    }
    accuracyHistories = newHistories;

    if (world.config.nCompartments >= 2) {
      const transEval = world.generateTranslationEvalBatch(evalIds, attrId);
      const numPairs = transEval.tokens.length;
      let totalNll = 0;
      let totalPositions = 0;
      for (let p = 0; p < numPairs; p++) {
        const padded = new Uint32Array(N * SL);
        for (let i = 0; i < N; i++) {
          for (let t = 0; t < transEval.seqLen; t++) {
            padded[i * SL + t] = transEval.tokens[p][i * transEval.seqLen + t];
          }
        }
        const tok = api.tensorFromArray(padded, [N, SL]);
        const logits = api.tidy(() => {
          const fwd = api.noGrad(() => model.forward(tok));
          api.keep(fwd.logits);
          return fwd.logits;
        });
        const logitsArr = new Float32Array(await logits.cpu());
        logits.dispose(); tok.dispose();
        for (let i = 0; i < N; i++) {
          for (let t = 0; t < transEval.targetLen; t++) {
            const pos = transEval.promptLen - 1 + t;
            const off = (i * SL + pos) * V;
            let mx = -Infinity;
            for (let v = 0; v < V; v++) if (logitsArr[off + v] > mx) mx = logitsArr[off + v];
            let sumExp = 0;
            for (let v = 0; v < V; v++) sumExp += Math.exp(logitsArr[off + v] - mx);
            const lse = mx + Math.log(sumExp);
            const tgt = transEval.targets[p][i * transEval.targetLen + t];
            totalNll += lse - logitsArr[off + tgt];
            totalPositions++;
          }
        }
      }
      const avgNll = totalPositions > 0 ? totalNll / totalPositions : 0;
      translationLossHistory = [...translationLossHistory, avgNll];
    }
  }

  async function runTraining() {
    if (!model) await resetModel();
    training = true;
    trainingActive = true;
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
        cosineHistory: [...cosineSimilarityHistory],
        accHistories: accuracyHistories.map((h) => [...h]),
        translationLossHistory: [...translationLossHistory],
      },
    ];
  }

  async function clearAll() {
    await stopAndWait();
    model = null; optimizer = null; world = null; step = 0;
    lossHistory = []; cosineSimilarityHistory = [];
    accuracyHistories = []; translationLossHistory = [];
    archivedRuns = [];
  }

  let lastLoss = $derived(lossHistory.at(-1) ?? 0);
  let lastCosSim = $derived(cosineSimilarityHistory.at(-1) ?? 0);
  let lastAccs = $derived(accuracyHistories.map((h) => h.at(-1) ?? 0));
  let lastTranslLoss = $derived(translationLossHistory.at(-1) ?? 0);

  const numel = (t: any) => t.shape.reduce((a: number, b: number) => a * b, 1);
  let paramBreakdown = $derived.by(() => {
    if (!model || !world) return null;
    const wte = numel(model.wte.weight);
    const wpe = model.wpe ? numel(model.wpe.weight) : 0;
    let attn = 0, mlp = 0, ln = 0;
    for (const layer of model.layers) {
      ln += numel(layer.ln1.weight) + numel(layer.ln1.bias) + numel(layer.ln2.weight) + numel(layer.ln2.bias);
      attn += numel(layer.qkv.weight) + numel(layer.qkv.bias) + numel(layer.outProj.weight) + numel(layer.outProj.bias);
      mlp += numel(layer.fc1.weight) + numel(layer.fc1.bias) + numel(layer.fc2.weight) + numel(layer.fc2.bias);
    }
    const lnF = numel(model.lnF.weight) + numel(model.lnF.bias);
    const head = numel(model.lmHead.weight);
    const total = wte + wpe + ln + attn + mlp + lnF + head;
    return { total, wte, wpe, attn, mlp, ln: ln + lnF, head };
  });
  const fmtK = (n: number) => n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : `${n}`;
</script>

<DemoPage
  title="Bio3: Factual Compartmentalization"
  currentRoute="bio"
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
    Entities have attributes with values. Multiple
    <em class="italic text-[rgba(0,0,0,0.84)]">compartments</em> encode the same facts using
    disjoint token sets. Does the transformer learn a unified fact store or separate
    representations per compartment? Cosine similarity of internal representations for the
    same queries in different encodings measures unification.
  {/snippet}

  {#snippet controls()}
    <BorderedGroup title="World" id="grp-world" contentClass="p-2 space-y-2">
      <NumberInput id="ent"  label="Entities"   min={10} max={1000} step={10} bind:value={config.world.nEntities} />
      <NumberInput id="attr" label="Attributes" min={1}  max={20}   step={1}  bind:value={config.world.nAttributes} />
      <NumberInput id="vals" label="Values"     min={5}  max={100}  step={5}  bind:value={config.world.nValues} />
      <NumberInput id="cmp"  label="Compartments" min={1} max={8}   step={1}  bind:value={config.world.nCompartments} />
      <Slider id="tok-ent" label="Tokens / entity" min={1} max={3} step={1}
              bind:value={config.world.tokensPerEntity} />
      <Slider id="tok-val" label="Tokens / value"  min={1} max={2} step={1}
              bind:value={config.world.tokensPerValue} />
      <CheckboxInput id="mix" label="Mix compartments per tuple"
                     bind:checked={config.world.mixCompartments} />
    </BorderedGroup>

    <BorderedGroup title="Objective" id="grp-obj" contentClass="p-2 space-y-2">
      <Slider id="trpct" label="Translation %" min={0} max={100} step={5}
              bind:value={config.objective.translationPct} unit="%" />
      <SelectInput id="mode" label="Mode" bind:value={config.objective.translationMode}
                   options={[{ value: 'mirror' }, { value: 'continuation' }, { value: 'dictionary' }]} />
    </BorderedGroup>

    <BorderedGroup title="Init" id="grp-init" contentClass="p-2 space-y-2">
      <NumberInput id="seed" label="Seed" min={0} step={1} bind:value={config.init.seed} />
      <CheckboxInput id="tied" label="Tied init across compartments"
                     bind:checked={config.init.tieInit} />
      <Slider id="ws"     label="Weight scale" min={0.1} max={10} step={0.05} useLog={true}
              bind:value={config.init.weightScale} />
      <Slider id="es"     label="Embed scale"  min={0}   max={2}  step={0.05}
              bind:value={config.init.embedScale} />
      <Slider id="hs"     label="Head scale"   min={0}   max={2}  step={0.05}
              bind:value={config.init.headScale} />
      <CheckboxInput id="resz" label="Residual zero-init"
                     bind:checked={config.init.residualZeroInit} />
    </BorderedGroup>

    <BorderedGroup title="Model" id="grp-model" contentClass="p-2 space-y-2">
      <SelectInput id="dim" label="Embed dim" bind:value={config.model.embedDim}
                   options={[{ value: 32 }, { value: 64 }, { value: 128 }]} />
      <Slider      id="layers" label="Layers" min={1} max={6} step={1}
                   bind:value={config.model.numLayers} />
      <SelectInput id="heads" label="Heads" bind:value={config.model.numHeads}
                   options={[{ value: 1 }, { value: 2 }, { value: 4 }, { value: 8 }]} />
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
      <Stat label="cos"  value={lastCosSim ? lastCosSim.toFixed(3) : '—'} />
      {#if lastAccs.length === 0}
        <Stat label="acc" value="—" />
      {:else}
        <Stat label="acc" value={`[${lastAccs.map(a => a.toFixed(2)).join(', ')}]`} />
      {/if}
      <Stat label="tr-loss" value={lastTranslLoss > 0 ? lastTranslLoss.toFixed(3) : '—'} />
      {#if paramBreakdown}
        <Stat
          value={fmtK(paramBreakdown.total)}
          title={`wte=${fmtK(paramBreakdown.wte)} wpe=${fmtK(paramBreakdown.wpe)} attn=${fmtK(paramBreakdown.attn)} mlp=${fmtK(paramBreakdown.mlp)} ln=${fmtK(paramBreakdown.ln)} head=${fmtK(paramBreakdown.head)}`}
        />
      {/if}
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
      caption="Cross-entropy on mixed bio paragraphs and QA pairs from both compartments."
    >
      <LineChart option={lossOption} />
    </Figure>

    <Figure title="Translation loss">
      {#snippet caption()}
        Cross-entropy on only the second-half (partB) positions of mirror-mode translation
        examples
        <code class="rounded bg-[rgba(0,0,0,0.04)] px-1 font-mono">[TR, partA, TR, partB]</code>,
        averaged across all directed compartment pairs. Measures how well the model can reproduce
        a fact stated in compartment A using compartment B's vocabulary.
      {/snippet}
      <LineChart option={translLossOption} />
    </Figure>

    <Figure
      title="QA accuracy"
      caption="Fraction of entities where argmax prediction matches the ground-truth value token for attribute 0, per compartment. Measures whether the model has memorized facts."
    >
      <LineChart option={accuracyOption} />
    </Figure>

    <Figure title="Cross-compartment cosine similarity">
      {#snippet caption()}
        Cosine similarity of middle-layer residual at the final prompt position (just before the
        predicted answer) for the same query in compartment A vs B. High = unified representation.
        Low = compartmentalized.
        {#if config.objective.translationPct > 0}
          Translation pairs ({config.objective.translationPct}% of batches, mode={config.objective.translationMode})
          provide cross-compartment supervision.
        {/if}
      {/snippet}
      <LineChart option={cosSimOption} />
    </Figure>

    {#if world}
      <Figure
        title="World"
        caption="{world.config.nEntities} entities, {world.config.nAttributes} attributes, {world.config.nValues} values/attribute, {world.config.nCompartments} compartments. Vocab: {world.vocabSize} tokens.{config.world.tokensPerEntity > 1 ? ` Entity names: ${config.world.tokensPerEntity} tokens.` : ''}"
      >
        <span></span>
      </Figure>
    {/if}

    {#if paramBreakdown}
      <Figure title="Parameters">
        {#snippet caption()}
          total: <strong class="font-medium text-[rgba(0,0,0,0.84)]">{fmtK(paramBreakdown.total)}</strong>
          &nbsp;·&nbsp; wte: {fmtK(paramBreakdown.wte)}
          &nbsp;·&nbsp; wpe: {fmtK(paramBreakdown.wpe)}
          &nbsp;·&nbsp; attn: {fmtK(paramBreakdown.attn)}
          &nbsp;·&nbsp; mlp: {fmtK(paramBreakdown.mlp)}
          &nbsp;·&nbsp; ln: {fmtK(paramBreakdown.ln)}
          &nbsp;·&nbsp; lm_head: {fmtK(paramBreakdown.head)}
        {/snippet}
        <span></span>
      </Figure>
    {/if}
  {/snippet}
</DemoPage>
