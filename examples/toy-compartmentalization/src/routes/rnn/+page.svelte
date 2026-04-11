<script lang="ts">
  import "../../app.css";
  import { onMount } from "svelte";
  import {
    initParams, initAdam, trainBatch, evalForward,
    type CPURNNParams, type AdamState,
  } from "$lib/rnn-cpu";
  import {
    generateDualTaskTokens,
    theoreticalEntropy, exploreBeliefSimplex,
    setTransitionMatrices, TRANSITION_MATRICES, TRANSITION_MATRICES_2,
    VOCAB_SIZE, VOCAB_SIZE_DATA,
  } from "$lib/data";
  import { config, initConfigUrlSync, resetConfigToDefaults } from "$lib/rnn-config.svelte";

  import { THEME, TOKEN_COLORS_RGB } from "$lib/theme";
  import { baseChartOpt, chartAxes, lineSeries, refLine } from "$lib/chart-helpers";
  import { DemoPage, Figure, LineChart, StatsBar, Stat } from "$lib/components";
  import { BorderedGroup, NumberInput, Slider } from "piston-controls";

  initConfigUrlSync();

  let training = $state(false);
  let step = $state(0);
  let lossHistory: number[] = $state([]);
  let params: CPURNNParams | null = $state(null);
  let adamState: AdamState | null = $state(null);

  let entropy = $state(theoreticalEntropy());

  let gtBeliefs: { b0: number; b1: number; b2: number }[] = $state([]);
  let hiddenPoints: { x: number; y: number; z: number }[] = $state([]);
  let probeR2 = $state(0);
  let stepsPerSec = $state(0);
  let lastVizStep = 0;
  let lastVizTime = 0;

  let gtCanvas: HTMLCanvasElement;
  let hiddenCanvas: HTMLCanvasElement;

  // ── Reactive chart options ────────────────────────────────────────────────
  let lossOption = $derived({
    ...baseChartOpt(),
    ...chartAxes({ xType: 'category', xData: lossHistory.map((_, i) => i), yType: 'log' }),
    series: [
      lineSeries(lossHistory, {
        markLine: refLine(entropy, `H=${entropy.toFixed(3)}`),
      }),
    ],
  });

  // ── 3D scatter (simplex) ──────────────────────────────────────────────────
  const CW = 400, CH = 400;
  let azimuth = $state(-0.4);
  let elevation = $state(0.3);
  let dragging = false;
  let dragStartX = 0, dragStartY = 0, dragStartAz = 0, dragStartEl = 0;
  let renderLoopRunning = false;

  function beliefRGB(p0: number, p1: number, p2: number): [number, number, number] {
    const clamp = (v: number) => Math.max(0, Math.min(255, v | 0));
    return [
      clamp(p0 * TOKEN_COLORS_RGB[0][0] + p1 * TOKEN_COLORS_RGB[1][0] + p2 * TOKEN_COLORS_RGB[2][0]),
      clamp(p0 * TOKEN_COLORS_RGB[0][1] + p1 * TOKEN_COLORS_RGB[1][1] + p2 * TOKEN_COLORS_RGB[2][1]),
      clamp(p0 * TOKEN_COLORS_RGB[0][2] + p1 * TOKEN_COLORS_RGB[1][2] + p2 * TOKEN_COLORS_RGB[2][2]),
    ];
  }

  function setupCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CW * dpr; canvas.height = CH * dpr;
    canvas.style.width = CW + 'px'; canvas.style.height = CH + 'px';
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    return ctx;
  }

  function project3D(x: number, y: number, z: number, cx: number, cy: number, scale: number, cosA: number, sinA: number, cosE: number, sinE: number) {
    const rx = x * cosA + z * sinA;
    const ryT = y, rzT = -x * sinA + z * cosA;
    return { sx: cx + rx * scale, sy: cy - (ryT * cosE - rzT * sinE) * scale, rz: ryT * sinE + rzT * cosE };
  }

  function draw3DScatter(canvas: HTMLCanvasElement, points: { x: number; y: number; z: number }[], colors: [number, number, number][], label: string) {
    if (!canvas) return;
    const ctx = setupCanvas(canvas);
    ctx.fillStyle = THEME.bg; ctx.fillRect(0, 0, CW, CH);
    const cx = CW / 2, cy = CH / 2, scale = Math.min(CW, CH) * 0.35;
    const cosA = Math.cos(azimuth), sinA = Math.sin(azimuth), cosE = Math.cos(elevation), sinE = Math.sin(elevation);

    const corners = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    ctx.strokeStyle = 'rgba(0,0,0,0.08)'; ctx.lineWidth = 0.5; ctx.beginPath();
    for (const [a, b] of edges) {
      const pa = project3D(corners[a][0], corners[a][1], corners[a][2], cx, cy, scale, cosA, sinA, cosE, sinE);
      const pb = project3D(corners[b][0], corners[b][1], corners[b][2], cx, cy, scale, cosA, sinA, cosE, sinE);
      ctx.moveTo(pa.sx, pa.sy); ctx.lineTo(pb.sx, pb.sy);
    }
    ctx.stroke();

    const n = points.length;
    ctx.fillStyle = THEME.fg; ctx.font = '12px system-ui'; ctx.textAlign = 'left'; ctx.fillText(label, 8, 14);
    ctx.fillStyle = THEME.muted; ctx.font = '10px system-ui'; ctx.textAlign = 'right'; ctx.fillText(`${n} pts`, CW - 8, 14);
    if (n === 0) return;

    const proj = new Array(n);
    for (let i = 0; i < n; i++) {
      const p = project3D(points[i].x, points[i].y, points[i].z, cx, cy, scale, cosA, sinA, cosE, sinE);
      proj[i] = { sx: p.sx, sy: p.sy, rz: p.rz, i };
    }
    proj.sort((a: any, b: any) => a.rz - b.rz);
    const rzMin = proj[0].rz, rzMax = proj[n - 1].rz, rzRange = rzMax - rzMin || 1;

    for (let i = 0; i < n; i++) {
      const p = proj[i];
      const t = (p.rz - rzMin) / rzRange;
      const c = colors[p.i];
      ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${(0.25 + 0.55 * t).toFixed(2)})`;
      ctx.beginPath(); ctx.arc(p.sx, p.sy, 1.0 + 1.5 * t, 0, 2 * Math.PI); ctx.fill();
    }
  }

  let gtColors: [number, number, number][] = [];
  let hiddenColors: [number, number, number][] = [];
  let gtPoints3D: { x: number; y: number; z: number }[] = [];
  let hiddenPoints3D: { x: number; y: number; z: number }[] = [];
  let hiddenTasks: number[] = [];
  // Distill blue / orange (light-friendly) for the two task colors.
  const TASK_COLORS: [number, number, number][] = [[31, 119, 180], [255, 127, 14]];

  function toSimplex3D(p0: number, p1: number, p2: number) {
    return { x: -p0 + p1, y: 0, z: -0.577 * (p0 + p1) + 1.155 * p2 };
  }

  function redrawSimplexes() {
    gtPoints3D = gtBeliefs.map(b => toSimplex3D(b.b0, b.b1, b.b2));
    gtColors = gtBeliefs.map(b => beliefRGB(b.b0, b.b1, b.b2));
    hiddenPoints3D = hiddenPoints.map(h => toSimplex3D(h.x, h.y, h.z));
    hiddenColors = hiddenPoints.map((_, i) => TASK_COLORS[hiddenTasks[i] ?? 0]);
  }

  function renderLoop() {
    if (!renderLoopRunning) return;
    const r2Label = probeR2 > 0 ? `Linear probe  R²=${probeR2.toFixed(3)}` : 'Linear probe';
    draw3DScatter(gtCanvas, gtPoints3D, gtColors, 'Ground truth beliefs');
    draw3DScatter(hiddenCanvas, hiddenPoints3D, hiddenColors, r2Label);
    requestAnimationFrame(renderLoop);
  }

  function initDrag(canvas: HTMLCanvasElement) {
    canvas.addEventListener('mousedown', (e: MouseEvent) => {
      dragging = true;
      dragStartX = e.clientX; dragStartY = e.clientY;
      dragStartAz = azimuth; dragStartEl = elevation;
      e.preventDefault();
    });
  }

  function syncSelfLoop() {
    setTransitionMatrices(config.world.selfLoop, config.world.sharedFrac);
    entropy = theoreticalEntropy();
    gtBeliefs = exploreBeliefSimplex(9);
    redrawSimplexes();
  }

  // Re-sync data when the slider moves
  $effect(() => {
    config.world.selfLoop; config.world.sharedFrac;
    syncSelfLoop();
  });

  onMount(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging) return;
      azimuth = dragStartAz + (e.clientX - dragStartX) * 0.008;
      elevation = dragStartEl + (e.clientY - dragStartY) * 0.008;
      elevation = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, elevation));
    };
    const onUp = () => { dragging = false; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);

    if (gtCanvas) initDrag(gtCanvas);
    if (hiddenCanvas) initDrag(hiddenCanvas);

    syncSelfLoop();
    renderLoopRunning = true;
    renderLoop();

    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      renderLoopRunning = false;
    };
  });

  function resetModel() {
    step = 0;
    lossHistory = [];
    hiddenPoints = [];
    probeR2 = 0; stepsPerSec = 0; lastVizStep = 0; lastVizTime = 0;
    params = initParams(VOCAB_SIZE, VOCAB_SIZE_DATA, config.model.hiddenDim);
    adamState = initAdam(params);
  }

  function trainFrame() {
    if (!params || !adamState) return;
    let lastLossVal = 0;
    for (let i = 0; i < config.viz.stepsPerFrame; i++) {
      const { tokens } = generateDualTaskTokens(config.optim.batchSize, config.model.seqLen);
      const { loss } = trainBatch(params, adamState, tokens, config.optim.batchSize, config.model.seqLen, config.optim.lr);
      lastLossVal = loss;
      step++;
      if (config.viz.vizInterval > 0 && step % config.viz.vizInterval === 0) {
        updateHiddenSimplex();
      }
    }
    lossHistory = [...lossHistory, lastLossVal];
  }

  function updateHiddenSimplex() {
    if (!params) return;
    const now = performance.now();
    if (lastVizTime > 0) {
      const elapsed = (now - lastVizTime) / 1000;
      const deltaSteps = step - lastVizStep;
      if (elapsed > 0) stepsPerSec = Math.round(deltaSteps / elapsed);
    }
    lastVizStep = step;
    lastVizTime = now;

    const { tokens: evalTokens, tasks: evalTaskIds } = generateDualTaskTokens(config.eval.batchSize, config.eval.seqLen);
    const hiddens = evalForward(params, evalTokens, config.eval.batchSize, config.eval.seqLen);

    const H = config.model.hiddenDim;
    const nPos = config.eval.seqLen - 1 - config.eval.burnIn;
    const nSamples = config.eval.batchSize * nPos;

    const acts: number[][] = [];
    const beliefs: number[][] = [];
    const taskLabels: number[] = [];

    for (let b = 0; b < config.eval.batchSize; b++) {
      const task = evalTaskIds[b];
      const mats = task === 0 ? TRANSITION_MATRICES : TRANSITION_MATRICES_2;
      let belief = [1/3, 1/3, 1/3];

      for (let t = 1; t < config.eval.seqLen; t++) {
        const dataTok = evalTokens[b * config.eval.seqLen + t];
        if (dataTok < 3) {
          const T = mats[dataTok];
          const next = [0, 0, 0];
          let sum = 0;
          for (let j = 0; j < 3; j++) {
            for (let i = 0; i < 3; i++) next[j] += belief[i] * T[i][j];
            sum += next[j];
          }
          for (let j = 0; j < 3; j++) next[j] /= sum;
          belief = next;
        }
        if (t >= config.eval.burnIn + 1) {
          const off = (b * config.eval.seqLen + t) * H;
          const act: number[] = [];
          for (let d = 0; d < H; d++) act.push(hiddens[off + d]);
          acts.push(act);
          beliefs.push([belief[0], belief[1], belief[2]]);
          taskLabels.push(task);
        }
      }
    }
    hiddenTasks = taskLabels;

    const task1Acts: number[][] = [];
    const task1Beliefs: number[][] = [];
    for (let i = 0; i < acts.length; i++) {
      if (taskLabels[i] === 0) {
        task1Acts.push(acts[i]);
        task1Beliefs.push(beliefs[i]);
      }
    }
    const probe = fitAffineProbe(task1Acts, task1Beliefs);
    const points: { x: number; y: number; z: number }[] = [];
    let ssRes = 0, ssTot = 0;
    const mean = [0, 0, 0];
    for (const b of beliefs) { mean[0] += b[0]; mean[1] += b[1]; mean[2] += b[2]; }
    mean[0] /= nSamples; mean[1] /= nSamples; mean[2] /= nSamples;

    for (let i = 0; i < nSamples; i++) {
      const h = acts[i];
      const p = [0, 0, 0];
      for (let s = 0; s < 3; s++) {
        p[s] = probe.bias[s];
        for (let j = 0; j < H; j++) p[s] += probe.W[s * H + j] * h[j];
      }
      points.push({ x: p[0], y: p[1], z: p[2] });
      for (let s = 0; s < 3; s++) {
        ssRes += (p[s] - beliefs[i][s]) ** 2;
        ssTot += (beliefs[i][s] - mean[s]) ** 2;
      }
    }
    probeR2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
    hiddenPoints = points;
    redrawSimplexes();
  }

  function fitAffineProbe(acts: number[][], beliefs: number[][]): { W: number[]; bias: number[] } {
    const n = acts.length, H = acts[0]?.length ?? config.model.hiddenDim, S = 3, dp1 = H + 1;
    const XtX = new Float64Array(dp1 * dp1);
    const XtY = new Float64Array(dp1 * S);
    for (let i = 0; i < n; i++) {
      const a = acts[i];
      for (let j = 0; j < H; j++) {
        for (let k = j; k < H; k++) XtX[j * dp1 + k] += a[j] * a[k];
        XtX[j * dp1 + H] += a[j];
        for (let s = 0; s < S; s++) XtY[j * S + s] += a[j] * beliefs[i][s];
      }
      for (let k = 0; k < H; k++) XtX[H * dp1 + k] += a[k];
      XtX[H * dp1 + H] += 1;
      for (let s = 0; s < S; s++) XtY[H * S + s] += beliefs[i][s];
    }
    for (let j = 0; j < dp1; j++)
      for (let k = 0; k < j; k++) XtX[j * dp1 + k] = XtX[k * dp1 + j];
    for (let j = 0; j < dp1; j++) XtX[j * dp1 + j] += 1e-6;
    const nm = dp1 + S;
    const aug = new Float64Array(dp1 * nm);
    for (let i = 0; i < dp1; i++) {
      for (let j = 0; j < dp1; j++) aug[i * nm + j] = XtX[i * dp1 + j];
      for (let j = 0; j < S; j++) aug[i * nm + dp1 + j] = XtY[i * S + j];
    }
    for (let col = 0; col < dp1; col++) {
      let maxVal = Math.abs(aug[col * nm + col]), maxRow = col;
      for (let row = col + 1; row < dp1; row++) {
        const v = Math.abs(aug[row * nm + col]);
        if (v > maxVal) { maxVal = v; maxRow = row; }
      }
      if (maxVal < 1e-12) return { W: new Array(S * H).fill(0), bias: [1/3, 1/3, 1/3] };
      if (maxRow !== col) {
        for (let j = 0; j < nm; j++) { const tmp = aug[col * nm + j]; aug[col * nm + j] = aug[maxRow * nm + j]; aug[maxRow * nm + j] = tmp; }
      }
      const pivot = aug[col * nm + col];
      for (let j = col; j < nm; j++) aug[col * nm + j] /= pivot;
      for (let row = 0; row < dp1; row++) {
        if (row === col) continue;
        const f = aug[row * nm + col];
        for (let j = col; j < nm; j++) aug[row * nm + j] -= f * aug[col * nm + j];
      }
    }
    const W: number[] = [], bias: number[] = [];
    for (let s = 0; s < S; s++) {
      for (let j = 0; j < H; j++) W.push(aug[j * nm + dp1 + s]);
      bias.push(aug[H * nm + dp1 + s]);
    }
    return { W, bias };
  }

  async function runTraining() {
    if (!params) resetModel();
    training = true;
    while (training) {
      trainFrame();
      await new Promise((r) => requestAnimationFrame(r));
    }
  }

  function stopTraining() { training = false; }

  function clearAll() {
    training = false;
    params = null; adamState = null; step = 0;
    lossHistory = []; hiddenPoints = []; probeR2 = 0;
    redrawSimplexes();
  }

  let lastLoss = $derived(lossHistory.at(-1) ?? 0);
</script>

<DemoPage
  title="Toy Models of Compartmentalization — Tiny RNN"
  currentRoute="rnn"
  gpuReady={true}
  {training}
  onTrain={runTraining}
  onStop={stopTraining}
  onReset={resetModel}
  onDefaults={resetConfigToDefaults}
  onClear={clearAll}
>
  {#snippet lead()}
    A 3-hidden-unit RNN learns next-token prediction on
    <em class="italic text-[rgba(0,0,0,0.84)]">two MESS3 variants</em>. With 3 hidden dims serving
    two 3-state belief distributions, the model must share representations. Does it
    <em class="italic text-[rgba(0,0,0,0.84)]">compartmentalize</em> (separate circuits per task)
    or <em class="italic text-[rgba(0,0,0,0.84)]">unify</em> (shared belief tracking)?
  {/snippet}

  {#snippet controls()}
    <BorderedGroup title="World" id="grp-world" contentClass="p-2 space-y-2">
      <Slider id="self-loop" label="Self-loop probability" min={0.3} max={0.9} step={0.01}
              bind:value={config.world.selfLoop} />
      <Slider id="shared" label="Shared fraction" min={0} max={1} step={0.05}
              bind:value={config.world.sharedFrac} />
    </BorderedGroup>

    <BorderedGroup title="Model" id="grp-model" contentClass="p-2 space-y-2">
      <Slider id="hidden" label="Hidden dim" min={2} max={12} step={1}
              bind:value={config.model.hiddenDim} />
      <Slider id="seq" label="Sequence length" min={10} max={100} step={5}
              bind:value={config.model.seqLen} />
    </BorderedGroup>

    <BorderedGroup title="Optimizer" id="grp-optim" contentClass="p-2 space-y-2">
      <Slider id="lr" label="Learning rate" min={1e-5} max={1} step={1e-5} useLog={true}
              bind:value={config.optim.lr} />
      <Slider id="bs" label="Batch size" min={16} max={256} step={16}
              bind:value={config.optim.batchSize} />
      <Slider id="lambda" label="λ (penalty)" min={0} max={5} step={0.1}
              bind:value={config.optim.lambda} />
    </BorderedGroup>

    <BorderedGroup title="Eval" id="grp-eval" contentClass="p-2 space-y-2">
      <NumberInput id="eb"  label="Eval batch"   min={50}  max={2000} step={50}
                   bind:value={config.eval.batchSize} />
      <NumberInput id="esl" label="Eval seqLen"  min={20}  max={500}  step={10}
                   bind:value={config.eval.seqLen} />
      <Slider      id="burn" label="Burn-in"    min={0}  max={50}   step={1}
                   bind:value={config.eval.burnIn} />
    </BorderedGroup>

    <BorderedGroup title="Visualization" id="grp-viz" contentClass="p-2 space-y-2">
      <Slider      id="spf" label="Steps / frame" min={10}  max={500}   step={10}
                   bind:value={config.viz.stepsPerFrame} />
      <NumberInput id="viz" label="Viz interval"  min={100} max={10000} step={100}
                   bind:value={config.viz.vizInterval} />
    </BorderedGroup>
  {/snippet}

  {#snippet stats()}
    <StatsBar>
      <Stat label="step" value={step} />
      <Stat label="loss" value={lastLoss > 0 ? lastLoss.toFixed(4) : '—'} />
      {#if stepsPerSec > 0}<Stat value={`${stepsPerSec} steps/s`} />{/if}
      {#if probeR2 > 0}<Stat label="R²" value={probeR2.toFixed(3)} />{/if}
    </StatsBar>
  {/snippet}

  {#snippet figures()}
    <Figure
      title="Training loss"
      caption="Cross-entropy. Dotted line marks the entropy floor H = {entropy.toFixed(3)} nats."
    >
      <LineChart option={lossOption} />
    </Figure>

    <Figure
      title="Belief simplex"
      caption="Left: ground-truth beliefs (equilateral simplex in 3D). Right: raw RNN hidden states colored by probed beliefs (blue = task 1, orange = task 2). Drag to rotate."
    >
      <div class="flex flex-wrap gap-4">
        <canvas bind:this={gtCanvas}     class="h-[400px] w-[400px] border border-[rgba(0,0,0,0.08)] bg-[#fffaf3]"></canvas>
        <canvas bind:this={hiddenCanvas} class="h-[400px] w-[400px] border border-[rgba(0,0,0,0.08)] bg-[#fffaf3]"></canvas>
      </div>
    </Figure>
  {/snippet}
</DemoPage>
