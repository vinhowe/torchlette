<script lang="ts">
  import "../app.css";
  import { onMount, tick } from "svelte";
  import { createModel, MESS3_CONFIG, type MESS3Model } from "$lib/model";
  import {
    generateBatch, generateBatchWithCompartments, generateBatchForComp, generatePairedBatch,
    theoreticalEntropy, beliefTrajectory, exploreBeliefSimplex,
    setTransitionMatrices,
    VOCAB_SIZE_DATA, NUM_STATES, STATIONARY_DIST,
    type DataConfig,
  } from "$lib/data";
  import { config, initConfigUrlSync, resetConfigToDefaults, describeConfigDelta } from "$lib/mess3-config.svelte";
  import { RpcClient } from "$lib/remote-transport";
  import { createRemoteEngine, type RemoteEngine, type RemoteEngineStats } from "../../../../src/remote/client-engine";

  initConfigUrlSync();

  let api: any = $state(null);
  let nn: any = null;
  let Adam: any = null;
  let crossEntropy: any = null;
  let getGPUMemoryStats: any = null;
  let gpuReady = $state(false);
  let gpuError = $state("");
  let rpcClient: RpcClient | null = null;
  let remoteEngine: RemoteEngine | null = null;
  let remoteStats: RemoteEngineStats | null = $state(null);
  let serverGpuMB = $state('');
  let isRemote = $state(false);

  let lrLog = $state(Math.log10(config.optim.lr));
  $effect(() => { lrLog = Math.log10(config.optim.lr); });
  let wsLog = $state(Math.log10(config.init.weightScale));
  $effect(() => { wsLog = Math.log10(config.init.weightScale); });

  let training = $state(false);
  let trainingActive = $state(false);
  let step = $state(0);
  let stepMs = $state(0);       // wall-clock ms for the last training step
  let stepsPerSec = $state(0);  // smoothed steps/sec
  let phaseTimings = $state(''); // per-phase breakdown
  let lossHistory: number[] = $state([]);
  let memoryHistory: number[] = $state([]);
  let model: MESS3Model | null = $state(null);
  let optimizer: any = $state(null);

  // Simplex visualization data
  // Ground truth: deterministic exploration (computed once)
  let gtBeliefs: { b0: number; b1: number; b2: number }[] = $state([]);
  let predBeliefs: number[][] = $state([]);
  let probeW: Float64Array | null = null;
  let probeBias: Float64Array | null = null;
  let probeR2: number = $state(0);
  /** Per-comp R² using probe fitted on comp 0 */
  let probeR2PerComp: number[] = $state([]);
  let probeR2History: number[] = $state([]);
  /** probeR2HistoryPerComp[c][evalIdx] = R² for comp c at eval step */
  let probeR2HistoryPerComp: number[][] = $state([]);

  const LOG_INTERVAL = 10;
  const VIZ_INTERVAL = 50;
  const PROBE_INTERVAL = 200;

  // Token colors: red, green, blue (matching the demo)
  const TOKEN_COLORS = [[232, 82, 74], [46, 204, 113], [74, 154, 222]];

  let echarts: any = null;
  let lossChart: any = null;
  let r2Chart: any = null;
  let memoryChart: any = null;
  let lossEl: HTMLDivElement;
  let r2El: HTMLDivElement;
  let memoryEl: HTMLDivElement;
  let gtCanvas: HTMLCanvasElement;
  let predCanvas: HTMLCanvasElement;

  let entropy = $state(theoreticalEntropy());

  function syncSelfLoop() {
    setTransitionMatrices(config.world.selfLoop);
    entropy = theoreticalEntropy();
    gtBeliefs = exploreBeliefSimplex(9);
    redrawSimplexes();
  }

  onMount(async () => {
    try {
      const tl = await import("torchlette");
      if (config.remote.enabled) {
        // Remote training: full remote engine with handle lifecycle management.
        rpcClient = new RpcClient(config.remote.url, (msg) => console.log(msg));
        await rpcClient.connect();
        remoteEngine = createRemoteEngine(rpcClient);
        remoteStats = remoteEngine.stats;
        isRemote = true;
        api = remoteEngine.torch;
      } else {
        // Local WebGPU training.
        await tl.initWebGPU();
        api = new tl.Torchlette("webgpu", { enableFusion: true, memoryLimitBytes: 8 * 1024 * 1024 * 1024 });
      }
      nn = tl.nn;
      Adam = tl.Adam;
      crossEntropy = tl.nn.functional.crossEntropy;
      getGPUMemoryStats = tl.getGPUMemoryStats;
      gpuReady = true;
    } catch (e: any) {
      gpuError = e.message || String(e);
    }
    echarts = await import("echarts");
    await tick();
    initCharts();
    // 3D mouse drag
    const onMove = (e: MouseEvent) => {
      if (!dragging3d) return;
      azimuth = dragStartAz + (e.clientX - dragStartX) * 0.008;
      elevation = dragStartEl + (e.clientY - dragStartY) * 0.008;
      elevation = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, elevation));
    };
    const onUp = () => { dragging3d = false; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);

    // Start 3D render loop
    if (gtCanvas) initDrag(gtCanvas);
    if (predCanvas) initDrag(predCanvas);
    renderLoopRunning = true;
    render3DLoop();
    syncSelfLoop();

    return () => {
      lossChart?.dispose(); r2Chart?.dispose(); memoryChart?.dispose();
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      renderLoopRunning = false;
    };
  });

  const C = { fg: '#e2e8f0', muted: '#64748b', grid: '#1e293b', surface: '#1e293b' };

  function baseOpt() {
    return {
      backgroundColor: 'transparent',
      textStyle: { color: C.muted, fontFamily: 'system-ui', fontSize: 11 },
      grid: { top: 28, right: 12, bottom: 24, left: 44 },
      animation: false,
    };
  }

  function initCharts() {
    if (!lossEl) return;
    lossChart = echarts.init(lossEl);
    r2Chart = echarts.init(r2El);
    memoryChart = echarts.init(memoryEl);
    updateLossChart();
    updateR2Chart();
    updateMemoryChart();
    syncSelfLoop();
  }

  function updateLossChart() {
    if (!lossChart) return;
    const steps = lossHistory.map((_, i) => i * LOG_INTERVAL);
    lossChart.setOption({
      ...baseOpt(),
      xAxis: { type: 'category', data: steps, axisLine: { lineStyle: { color: C.grid } }, axisLabel: { color: C.muted }, splitLine: { show: false } },
      yAxis: { type: 'log', axisLine: { show: false }, splitLine: { lineStyle: { color: C.grid } }, axisLabel: { color: C.muted } },
      tooltip: { trigger: 'axis', backgroundColor: C.surface, borderColor: C.grid, textStyle: { color: C.fg, fontSize: 11 } },
      series: [{
        type: 'line', data: lossHistory, showSymbol: false,
        lineStyle: { width: 1.5, color: '#60a5fa' },
        markLine: {
          silent: true, symbol: 'none', animation: false,
          data: [{ yAxis: entropy, lineStyle: { color: '#60a5fa', type: 'dotted', width: 1, opacity: 0.5 },
            label: { formatter: `H=${entropy.toFixed(3)}`, position: 'insideEndTop', color: C.muted, fontSize: 9 } }],
        },
      }],
    }, true);
  }

  const COMP_COLORS_R2 = ['#fbbf24', '#f472b6', '#34d399', '#a78bfa'];

  function updateR2Chart() {
    if (!r2Chart) return;
    const series: any[] = [];
    if (probeR2HistoryPerComp.length > 0) {
      for (let c = 0; c < probeR2HistoryPerComp.length; c++) {
        series.push({
          name: c === 0 ? 'comp 0 (fitted)' : `comp ${c} (transfer)`,
          type: 'line', data: probeR2HistoryPerComp[c].map((y, i) => [i * VIZ_INTERVAL, y]),
          showSymbol: true, symbolSize: 3,
          lineStyle: { width: c === 0 ? 2 : 1.5, color: COMP_COLORS_R2[c % COMP_COLORS_R2.length] },
          itemStyle: { color: COMP_COLORS_R2[c % COMP_COLORS_R2.length] },
        });
      }
    } else {
      series.push({
        name: 'R²', type: 'line', data: probeR2History.map((y, i) => [i * VIZ_INTERVAL, y]),
        showSymbol: true, symbolSize: 4,
        lineStyle: { width: 1.5, color: '#fbbf24' },
        itemStyle: { color: '#fbbf24' },
      });
    }
    r2Chart.setOption({
      ...baseOpt(),
      legend: probeR2HistoryPerComp.length > 1 ? {
        top: 2, right: 12, textStyle: { color: C.muted, fontSize: 10 },
        icon: 'circle', itemWidth: 8, itemHeight: 8,
      } : undefined,
      xAxis: { type: 'value', min: 0, axisLine: { lineStyle: { color: C.grid } }, axisLabel: { color: C.muted }, splitLine: { show: false } },
      yAxis: { type: 'value', min: 0, max: 1, axisLine: { show: false }, splitLine: { lineStyle: { color: C.grid } }, axisLabel: { color: C.muted } },
      tooltip: { trigger: 'axis', backgroundColor: C.surface, borderColor: C.grid, textStyle: { color: C.fg, fontSize: 11 } },
      series,
    }, true);
  }

  function updateMemoryChart() {
    if (!memoryChart) return;
    const steps = memoryHistory.map((_, i) => i * LOG_INTERVAL);
    memoryChart.setOption({
      ...baseOpt(),
      xAxis: { type: 'category', data: steps, axisLine: { lineStyle: { color: C.grid } }, axisLabel: { color: C.muted }, splitLine: { show: false } },
      yAxis: { type: 'value', name: 'MB', nameTextStyle: { color: C.muted, fontSize: 10 },
        axisLine: { show: false }, splitLine: { lineStyle: { color: C.grid } }, axisLabel: { color: C.muted } },
      series: [{
        type: 'line', data: memoryHistory, showSymbol: false,
        lineStyle: { width: 1.5, color: '#6ee7b7' },
        areaStyle: { color: 'rgba(110, 231, 183, 0.08)' },
      }],
    }, true);
  }

  // --- 3D scatter visualization ---
  const CW = 400, CH = 400;
  let azimuth = $state(-0.4);
  let elevation = $state(0.3);
  let zoom3d = $state(1.0);
  let dragging3d = false;
  let dragStartX = 0, dragStartY = 0, dragStartAz = 0, dragStartEl = 0;
  let renderLoopRunning = false;

  function beliefRGB(p0: number, p1: number, p2: number): [number, number, number] {
    const clamp = (v: number) => Math.max(0, Math.min(255, v | 0));
    return [
      clamp(p0 * TOKEN_COLORS[0][0] + p1 * TOKEN_COLORS[1][0] + p2 * TOKEN_COLORS[2][0]),
      clamp(p0 * TOKEN_COLORS[0][1] + p1 * TOKEN_COLORS[1][1] + p2 * TOKEN_COLORS[2][1]),
      clamp(p0 * TOKEN_COLORS[0][2] + p1 * TOKEN_COLORS[1][2] + p2 * TOKEN_COLORS[2][2]),
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
    ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, CW, CH);
    const cx = CW / 2, cy = CH / 2, scale = Math.min(CW, CH) * 0.35 * zoom3d;
    const cosA = Math.cos(azimuth), sinA = Math.sin(azimuth), cosE = Math.cos(elevation), sinE = Math.sin(elevation);

    // Wireframe cube
    const corners = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    ctx.strokeStyle = 'rgba(148,163,184,0.08)'; ctx.lineWidth = 0.5; ctx.beginPath();
    for (const [a, b] of edges) {
      const pa = project3D(corners[a][0], corners[a][1], corners[a][2], cx, cy, scale, cosA, sinA, cosE, sinE);
      const pb = project3D(corners[b][0], corners[b][1], corners[b][2], cx, cy, scale, cosA, sinA, cosE, sinE);
      ctx.moveTo(pa.sx, pa.sy); ctx.lineTo(pb.sx, pb.sy);
    }
    ctx.stroke();

    const n = points.length;
    ctx.fillStyle = '#e2e8f0'; ctx.font = '12px system-ui'; ctx.textAlign = 'left'; ctx.fillText(label, 8, 14);
    ctx.fillStyle = '#64748b'; ctx.font = '10px system-ui'; ctx.textAlign = 'right'; ctx.fillText(`${n} pts`, CW - 8, 14);
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
      ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${(0.15 + 0.45 * t).toFixed(2)})`;
      ctx.beginPath(); ctx.arc(p.sx, p.sy, 1.0 + 1.5 * t, 0, 2 * Math.PI); ctx.fill();
    }
  }

  // Precomputed 3D data
  let gtPoints3D: { x: number; y: number; z: number }[] = [];
  let gtColors: [number, number, number][] = [];
  let predPoints3D: { x: number; y: number; z: number }[] = [];
  let predColors: [number, number, number][] = [];
  /** Per-comp predicted beliefs (only populated when c>1). */
  let perCompPredBeliefs: number[][][] = [];

  // Compartment colors for overlay visualization
  const COMP_RGB: [number, number, number][] = [
    [251, 191, 36],   // gold (comp 0 — fitted)
    [244, 114, 182],  // pink
    [52, 211, 153],   // green
    [167, 139, 250],  // purple
    [251, 146, 60],   // orange
    [34, 211, 238],   // cyan
    [232, 121, 249],  // magenta
    [148, 163, 184],  // gray
  ];

  function toSimplex3D(p0: number, p1: number, p2: number) {
    return { x: -p0 + p1, y: 0, z: -0.577 * (p0 + p1) + 1.155 * p2 };
  }

  function redrawSimplexes() {
    gtPoints3D = gtBeliefs.map(b => toSimplex3D(b.b0, b.b1, b.b2));
    gtColors = gtBeliefs.map(b => beliefRGB(b.b0, b.b1, b.b2));

    if (config.world.nCompartments > 1 && perCompPredBeliefs.length > 0) {
      // Multi-comp: overlay all comps' probe predictions, colored by compartment.
      const allPts: { x: number; y: number; z: number }[] = [];
      const allCols: [number, number, number][] = [];
      for (let c = 0; c < perCompPredBeliefs.length; c++) {
        const col = COMP_RGB[c % COMP_RGB.length];
        for (const b of perCompPredBeliefs[c]) {
          allPts.push(toSimplex3D(b[0], b[1], b[2]));
          allCols.push(col);
        }
      }
      predPoints3D = allPts;
      predColors = allCols;
    } else {
      // Single comp: color by belief state (original).
      predPoints3D = predBeliefs.map(b => toSimplex3D(b[0], b[1], b[2]));
      predColors = predBeliefs.map(b => beliefRGB(b[0], b[1], b[2]));
    }
  }

  function render3DLoop() {
    if (!renderLoopRunning) return;
    const nC = config.world.nCompartments;
    let r2Label: string;
    if (nC > 1 && probeR2PerComp.length > 0) {
      const r2Str = probeR2PerComp.map((r, c) => `c${c}=${r.toFixed(2)}`).join(' ');
      r2Label = `Comp-0 probe on all comps  ${r2Str}`;
    } else {
      r2Label = probeR2 > 0 ? `Residual stream probe  R\u00b2=${probeR2.toFixed(3)}` : 'Residual stream probe';
    }
    draw3DScatter(gtCanvas, gtPoints3D, gtColors, 'Ground truth beliefs');
    draw3DScatter(predCanvas, predPoints3D, predColors, r2Label);
    requestAnimationFrame(render3DLoop);
  }

  function initDrag(canvas: HTMLCanvasElement) {
    canvas.addEventListener('mousedown', (e: MouseEvent) => {
      dragging3d = true;
      dragStartX = e.clientX; dragStartY = e.clientY;
      dragStartAz = azimuth; dragStartEl = elevation;
      e.preventDefault();
    });
    canvas.addEventListener('wheel', (e: WheelEvent) => {
      e.preventDefault();
      zoom3d = Math.max(0.2, Math.min(10, zoom3d * (1 - e.deltaY * 0.001)));
    }, { passive: false });
  }

  function syncLr() {
    config.optim.lr = 10 ** lrLog;
    if (optimizer) optimizer.setLR(config.optim.lr);
  }
  function syncWs() { config.init.weightScale = 10 ** wsLog; }

  async function resetModel() {
    step = 0;
    lossHistory = []; memoryHistory = [];
    predBeliefs = [];
    probeW = null; probeBias = null; probeR2 = 0;
    probeR2History = []; probeR2PerComp = []; probeR2HistoryPerComp = [];

    await api.beginStep();
    api.manualSeed(config.init.seed);
    // +1 for TR token used in paired/translation sequences.
    const vocabSize = VOCAB_SIZE_DATA * config.world.nCompartments + 1;
    model = createModel(api, nn, {
      ...MESS3_CONFIG,
      seqLen: config.model.seqLen,
      vocabSize,
      posEncoding: 'rope',
    });
    if (config.init.weightScale !== 1.0) {
      for (const p of model.parameters()) api.mul_(p, config.init.weightScale);
    }
    optimizer = new Adam(model.parameters(), {
      lr: config.optim.lr,
      weightDecay: config.optim.weightDecay,
      adamW: config.optim.weightDecay > 0,
    });
    api.endStep();
    // Pre-upload model weights via binary transport to avoid massive JSON
    // plans on the first training step.
    if (remoteEngine) {
      const uploaded = await remoteEngine.preUpload(model.parameters());
      console.log(`[preUpload] ${uploaded} param tensors`);
    }
    updateLossChart();
    updateMemoryChart();
    updateR2Chart();
    redrawSimplexes();
  }

  async function stopAndWait() {
    training = false;
    while (trainingActive) await new Promise((r) => setTimeout(r, 10));
  }

  async function trainStep() {
    if (!model || !optimizer) return;
    const t0 = performance.now();
    await api.beginStep();

    const sl = config.model.seqLen;
    const bs = config.optim.batchSize;
    const nC = config.world.nCompartments;
    const vocabSize = VOCAB_SIZE_DATA * nC + 1;
    const trTokenId = vocabSize - 1;
    const trFrac = config.objective.translationPct / 100;
    const useTr = nC > 1 && trFrac > 0 && Math.random() < trFrac;
    const batch = useTr
      ? generatePairedBatch({ seqLen: sl, batchSize: bs }, nC, trTokenId)
      : nC > 1
        ? generateBatchWithCompartments({ seqLen: sl, batchSize: bs }, nC)
        : generateBatch({ seqLen: sl, batchSize: bs });
    const tok = api.tensorFromArray(batch.tokens, [bs, sl], { dtype: 'i32' });
    const tgt = api.tensorFromArray(batch.targets, [bs * (sl - 1)], { dtype: 'i32' });

    const tData = performance.now();

    const loss = api.tidy(() => {
      const fwd = model!.forward(tok);
      const logits = fwd.logits.narrow(1, 0, sl - 1).contiguous()
        .reshape([bs * (sl - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt);
      api.keep(l);
      return l;
    });
    tok.dispose(); tgt.dispose();
    const tGraph = performance.now();

    const shouldLog = step % LOG_INTERVAL === 0;

    // Loss readback before backward (backward disposes loss).
    if (shouldLog) {
      const val = await loss.item();
      if (Number.isFinite(val)) {
        lossHistory = [...lossHistory, val];
        updateLossChart();
      } else {
        console.warn(`step ${step}: loss is ${val}, stopping`);
        training = false;
      }
    }
    const tRead = performance.now();

    await loss.backward();
    loss.dispose();
    const tBwd = performance.now();

    optimizer.step();
    optimizer.zeroGrad();
    const tOpt = performance.now();

    if (step > 0 && step % VIZ_INTERVAL === 0) {
      await updateBeliefSimplex(true);
    }

    step++;
    await api.endStep();
    if (remoteEngine && optimizer && model) {
      const keep = [...optimizer.getAllKeepTensors(), ...model.persistentTensors()];
      await remoteEngine.markStep(keep);
    }
    const tEnd = performance.now();

    if (shouldLog && getGPUMemoryStats && !isRemote) {
      const mem = getGPUMemoryStats();
      memoryHistory = [...memoryHistory, mem.currentBytes / (1024 * 1024)];
      updateMemoryChart();
    }

    const elapsed = tEnd - t0;
    stepMs = elapsed;
    stepsPerSec = stepsPerSec * 0.9 + (1000 / elapsed) * 0.1;
    if (shouldLog) {
      phaseTimings = `data=${(tData - t0).toFixed(0)} graph=${(tGraph - tData).toFixed(0)} read=${(tRead - tGraph).toFixed(0)} bwd=${(tBwd - tRead).toFixed(0)} opt=${(tOpt - tBwd).toFixed(0)} end=${(tEnd - tOpt).toFixed(0)}`;
      if (isRemote) console.log(`[step ${step}] ${(tEnd - t0).toFixed(0)}ms total | ${phaseTimings} | rpcs=${remoteStats?.executes}`);
    }

    if (isRemote && rpcClient && step % 10 === 0) {
      try {
        const s = await rpcClient.stats();
        if (s.gpu) serverGpuMB = `${s.gpu.currentMB}/${s.gpu.peakMB}MB`;
      } catch { /* ignore */ }
    }
  }

  /** Extract activations + beliefs from a comp-specific batch. */
  async function extractActivationsAndBeliefs(comp: number): Promise<{ activations: number[][]; beliefs: number[][] }> {
    const evalBatch = config.probe.batchSize;
    const sl = config.model.seqLen;
    const batch = generateBatchForComp({ seqLen: sl, batchSize: evalBatch }, comp);
    const tok = api.tensorFromArray(batch.tokens, [evalBatch, sl], { dtype: 'i32' });
    const residual = api.tidy(() => {
      const fwd = api.noGrad(() => model!.forward(tok));
      const lastRes = fwd.residuals[fwd.residuals.length - 1];
      api.keep(lastRes);
      return lastRes;
    });
    tok.dispose();
    const resData = await residual.cpu();
    residual.dispose();
    const dim = model!.config.embedDim;
    const nPos = sl - 1;
    const activations: number[][] = [];
    const beliefs: number[][] = [];
    for (let b = 0; b < evalBatch; b++) {
      for (let t = 0; t < nPos; t++) {
        const act: number[] = [];
        for (let d = 0; d < dim; d++) act.push(resData[b * sl * dim + t * dim + d]);
        activations.push(act);
        beliefs.push(Array.from(batch.beliefs[b * nPos + t]));
      }
    }
    return { activations, beliefs };
  }

  /** Compute R² of probe (probeW, probeBias) on given activations/beliefs. */
  function computeR2(activations: number[][], beliefs: number[][]): number {
    if (!probeW || !probeBias) return 0;
    const n = activations.length;
    const dim = activations[0].length;
    const mean = [0, 0, 0];
    for (let i = 0; i < n; i++) for (let s = 0; s < NUM_STATES; s++) mean[s] += beliefs[i][s];
    for (let s = 0; s < NUM_STATES; s++) mean[s] /= n;
    let ssRes = 0, ssTot = 0;
    for (let i = 0; i < n; i++) {
      for (let s = 0; s < NUM_STATES; s++) {
        let v = probeBias[s];
        for (let d = 0; d < dim; d++) v += probeW[s * dim + d] * activations[i][d];
        ssRes += (v - beliefs[i][s]) ** 2;
        ssTot += (beliefs[i][s] - mean[s]) ** 2;
      }
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }

  async function updateBeliefSimplex(refitProbe: boolean) {
    if (!model) return;

    // Always extract comp 0 for probe fitting + simplex display.
    const comp0 = await extractActivationsAndBeliefs(0);

    if (refitProbe || !probeW) {
      fitLinearProbe(comp0.activations, comp0.beliefs);
    }

    if (probeW && probeBias) {
      // Simplex display: apply probe to comp 0 activations.
      const pred: number[][] = [];
      const dim = comp0.activations[0].length;
      for (let i = 0; i < comp0.activations.length; i++) {
        const p = [0, 0, 0];
        for (let s = 0; s < NUM_STATES; s++) {
          let v = probeBias[s];
          for (let d = 0; d < dim; d++) v += probeW[s * dim + d] * comp0.activations[i][d];
          p[s] = v;
        }
        pred.push(p);
      }
      predBeliefs = pred;

      // Per-comp R²: probe fitted on comp 0, applied to all comps.
      // Also collect per-comp predicted beliefs for visualization.
      const nC = Math.min(4, config.world.nCompartments);
      const r2s: number[] = [computeR2(comp0.activations, comp0.beliefs)];
      const allCompPreds: number[][][] = [pred]; // comp 0 already computed
      for (let c = 1; c < nC; c++) {
        const compC = await extractActivationsAndBeliefs(c);
        r2s.push(computeR2(compC.activations, compC.beliefs));
        // Apply comp-0 probe to comp c's activations for visualization.
        const cPred: number[][] = [];
        for (let i = 0; i < compC.activations.length; i++) {
          const p = [0, 0, 0];
          for (let s = 0; s < NUM_STATES; s++) {
            let v = probeBias[s];
            for (let d = 0; d < dim; d++) v += probeW[s * dim + d] * compC.activations[i][d];
            p[s] = v;
          }
          cPred.push(p);
        }
        allCompPreds.push(cPred);
      }
      perCompPredBeliefs = allCompPreds;

      probeR2 = r2s[0];
      probeR2PerComp = r2s;
      probeR2History = [...probeR2History, probeR2];
      if (probeR2HistoryPerComp.length !== nC) {
        probeR2HistoryPerComp = Array.from({ length: nC }, () => []);
      }
      const newHist = probeR2HistoryPerComp.map(h => [...h]);
      for (let c = 0; c < nC; c++) newHist[c].push(r2s[c]);
      probeR2HistoryPerComp = newHist;
      updateR2Chart();
    }

    redrawSimplexes();
  }

  function fitLinearProbe(activations: number[][], beliefs: number[][]) {
    const n = activations.length;
    const d = activations[0].length;
    const dp1 = d + 1;

    // Solve: beliefs (all 3 components) = W @ [activations; 1] via normal equation
    const XtX = new Float64Array(dp1 * dp1);
    const XtY = new Float64Array(dp1 * NUM_STATES);

    for (let i = 0; i < n; i++) {
      const a = activations[i];
      for (let j = 0; j < d; j++) {
        for (let k = j; k < d; k++) XtX[j * dp1 + k] += a[j] * a[k];
        XtX[j * dp1 + d] += a[j];
        for (let s = 0; s < NUM_STATES; s++) XtY[j * NUM_STATES + s] += a[j] * beliefs[i][s];
      }
      for (let k = 0; k < d; k++) XtX[d * dp1 + k] += a[k];
      XtX[d * dp1 + d] += 1;
      for (let s = 0; s < NUM_STATES; s++) XtY[d * NUM_STATES + s] += beliefs[i][s];
    }
    for (let j = 0; j < dp1; j++)
      for (let k = 0; k < j; k++) XtX[j * dp1 + k] = XtX[k * dp1 + j];
    for (let j = 0; j < dp1; j++) XtX[j * dp1 + j] += 1e-5;

    const W = solveLinearSystem(XtX, XtY, dp1, NUM_STATES);
    if (W) {
      probeW = new Float64Array(NUM_STATES * d);
      probeBias = new Float64Array(NUM_STATES);
      for (let s = 0; s < NUM_STATES; s++) {
        for (let j = 0; j < d; j++) probeW[s * d + j] = W[j * NUM_STATES + s];
        probeBias[s] = W[d * NUM_STATES + s];
      }
    }
  }

  function solveLinearSystem(A: Float64Array, B: Float64Array, n: number, m: number): Float64Array | null {
    const nm = n + m;
    const aug = new Float64Array(n * nm);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) aug[i * nm + j] = A[i * n + j];
      for (let j = 0; j < m; j++) aug[i * nm + n + j] = B[i * m + j];
    }
    for (let col = 0; col < n; col++) {
      let maxVal = Math.abs(aug[col * nm + col]), maxRow = col;
      for (let row = col + 1; row < n; row++) {
        const v = Math.abs(aug[row * nm + col]);
        if (v > maxVal) { maxVal = v; maxRow = row; }
      }
      if (maxVal < 1e-12) return null;
      if (maxRow !== col) {
        for (let j = 0; j < nm; j++) {
          const tmp = aug[col * nm + j]; aug[col * nm + j] = aug[maxRow * nm + j]; aug[maxRow * nm + j] = tmp;
        }
      }
      const pivot = aug[col * nm + col];
      for (let j = col; j < nm; j++) aug[col * nm + j] /= pivot;
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const f = aug[row * nm + col];
        for (let j = col; j < nm; j++) aug[row * nm + j] -= f * aug[col * nm + j];
      }
    }
    const X = new Float64Array(n * m);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++) X[i * m + j] = aug[i * nm + n + j];
    return X;
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
    } finally { trainingActive = false; }
  }

  function stopTraining() { training = false; }
  async function clearAll() {
    await stopAndWait();
    model = null; optimizer = null; step = 0;
    lossHistory = []; memoryHistory = [];
    gtBeliefs = []; predBeliefs = [];
    probeW = null; probeBias = null; probeR2 = 0;
    probeR2History = []; probeR2PerComp = []; probeR2HistoryPerComp = [];
    updateLossChart(); updateMemoryChart(); updateR2Chart();
  }

  let lastLoss = $derived(lossHistory.at(-1) ?? 0);
  let paramCount = $derived(model ? model.parameters().reduce((s: number, p: any) => s + p.shape.reduce((a: number, b: number) => a * b, 1), 0) : 0);
</script>

<div class="page">
  <nav class="subnav">
    <a href="/" class="current">mess3</a>
    <a href="/bio">bio</a>
    <a href="/xor">xor</a>
    <a href="/brackets">brackets</a>
  </nav>
  <header>
    <h1>Belief State Geometry in Transformers</h1>
    <p class="lead">
      A tiny transformer learns next-token prediction on sequences from the <em>MESS3</em> process
      — a 3-state HMM. The optimal predictor must track a posterior over hidden states.
      We probe the residual stream to recover the belief simplex geometry live during training.
    </p>
  </header>

  {#if gpuError}
    <div class="error">WebGPU: {gpuError}</div>
  {/if}

  <div class="controls-bar">
    <div class="control-group">
      <span class="group-label">world</span>
      <label>self-loop <input type="range" min="0.3" max="0.9" step="0.01" bind:value={config.world.selfLoop} oninput={syncSelfLoop} /><span class="val">{config.world.selfLoop.toFixed(2)}</span></label>
      <label>c <input type="number" min="1" max="8" step="1" bind:value={config.world.nCompartments} style="width:36px" /></label>
      <label>tr% <input type="range" min="0" max="100" step="1" bind:value={config.objective.translationPct} /><span class="val">{config.objective.translationPct}</span></label>
    </div>
    <div class="control-group">
      <span class="group-label">init</span>
      <label>seed <input type="number" min="0" step="1" bind:value={config.init.seed} style="width:56px" /></label>
      <label>w× <input type="range" min="-2" max="1" step="0.05" bind:value={wsLog} oninput={syncWs} /><span class="val">{config.init.weightScale.toFixed(2)}</span></label>
    </div>
    <div class="control-group">
      <span class="group-label">model</span>
      <label>ctx <input type="range" min="5" max="20" step="1" bind:value={config.model.seqLen} /><span class="val">{config.model.seqLen}</span></label>
      <label>probe <input type="number" min="32" max="2048" step="32" bind:value={config.probe.batchSize} style="width:52px" /></label>
    </div>
    <div class="control-group">
      <span class="group-label">optim</span>
      <label>lr <input type="range" min="-5" max="0" step="0.1" bind:value={lrLog} oninput={syncLr} /><span class="val">{config.optim.lr.toExponential(1)}</span></label>
      <label>wd <input type="range" min="0" max="0.5" step="0.01" bind:value={config.optim.weightDecay} /><span class="val">{config.optim.weightDecay.toFixed(2)}</span></label>
      <label>batch <input type="range" min="16" max="128" step="16" bind:value={config.optim.batchSize} /><span class="val">{config.optim.batchSize}</span></label>
    </div>
    <div class="control-group">
      <span class="group-label">remote</span>
      <label><input type="checkbox" bind:checked={config.remote.enabled} /> gpu</label>
      <label>ws <input type="text" bind:value={config.remote.url} style="width:140px; font-size:10px" /></label>
    </div>
    <div class="control-actions">
      <button onclick={runTraining} disabled={!gpuReady || training}>train</button>
      <button onclick={stopTraining} disabled={!training}>stop</button>
      <button onclick={async () => { await stopAndWait(); await resetModel(); }} disabled={!gpuReady}>reset</button>
      <button onclick={async () => { await stopAndWait(); resetConfigToDefaults(); }}>defaults</button>
      <button onclick={clearAll}>clear</button>
      <span class="stat">step {step}</span>
      <span class="stat">loss {lastLoss > 0 ? lastLoss.toFixed(4) : '--'}</span>
      {#if probeR2PerComp.length > 0}
        <span class="stat">R² [{probeR2PerComp.map(r => r.toFixed(2)).join(', ')}]</span>
      {:else if probeR2 > 0}
        <span class="stat">R²={probeR2.toFixed(3)}</span>
      {/if}
      {#if paramCount}<span class="stat">{(paramCount / 1000).toFixed(1)}k</span>{/if}
      {#if stepMs > 0}<span class="stat">{stepMs.toFixed(0)}ms/step</span>{/if}
      {#if stepsPerSec > 0}<span class="stat">{stepsPerSec.toFixed(1)} step/s</span>{/if}
      {#if phaseTimings}<span class="stat" title="Phase breakdown (ms)">{phaseTimings}</span>{/if}
      {#if remoteStats}
        <span class="stat">
          rpc {remoteStats.executes}x {(remoteStats.bytesUp / 1024).toFixed(0)}KB↑ {(remoteStats.bytesDown / 1024).toFixed(0)}KB↓
          h={remoteEngine?.handles.size() ?? 0} rel={remoteStats.handlesReleased}
        </span>
        {#if serverGpuMB}<span class="stat">gpu {serverGpuMB}</span>{/if}
      {/if}
    </div>
  </div>

  <div class="figures">
    <section class="figure">
      <h2>Training Loss</h2>
      <p class="caption">
        Cross-entropy on MESS3. Dotted line = entropy floor H = {entropy.toFixed(3)} nats.
      </p>
      <div class="chart" bind:this={lossEl}></div>
    </section>

    <section class="figure">
      <h2>Belief Simplex</h2>
      <p class="caption">
        Each point is a (position, sequence) pair. Left: ground-truth beliefs from the HMM
        (color = belief distribution: red/green/blue for S1/S2/S3).
        Right: affine linear probe of the last-layer residual stream (R² measures probe fit).
        {#if config.world.nCompartments > 1}
          With c&gt;1, the probe is fitted on comp 0 only. Dots are colored by compartment —
          if unified, all colors trace the same fractal; if compartmentalized, only comp 0 (gold)
          shows the fractal while others scatter.
        {:else}
          The fractal should emerge as training converges.
        {/if}
      </p>
      <div class="simplex-pair">
        <canvas bind:this={gtCanvas} class="simplex-canvas"></canvas>
        <canvas bind:this={predCanvas} class="simplex-canvas"></canvas>
      </div>
    </section>

    <section class="figure">
      <h2>Linear Probe R²</h2>
      <p class="caption">How much belief state variance the residual stream linearly encodes. 1.0 = perfect recovery.</p>
      <div class="chart small" bind:this={r2El}></div>
    </section>

    <section class="figure">
      <h2>GPU Memory</h2>
      <p class="caption">Allocated GPU buffer memory.</p>
      <div class="chart small" bind:this={memoryEl}></div>
    </section>

    <section class="figure">
      <h2>MESS3 Process</h2>
      <p class="caption">
        3 hidden states, 3 tokens. Each state preferentially emits one token (p=0.77) and self-loops.
        The belief state traces a fractal on the 2-simplex.
      </p>
    </section>
  </div>
</div>

<style>
  .page { max-width: 860px; margin: 0 auto; padding: 32px 20px 64px; }
  .subnav { display: flex; gap: 16px; font-size: 11px; font-family: var(--font-mono); margin-bottom: 20px; }
  .subnav a { color: var(--muted); text-decoration: none; text-transform: uppercase; letter-spacing: 0.05em; }
  .subnav a:hover { color: var(--fg); }
  .subnav a.current { color: var(--fg); font-weight: 600; }
  header { margin-bottom: 28px; }
  h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 8px; }
  .lead { color: var(--muted); font-size: 14px; line-height: 1.6; max-width: 640px; }
  .lead em { color: var(--fg); font-style: normal; font-weight: 500; }
  .error { background: #450a0a; border: 1px solid #7f1d1d; padding: 6px 12px; font-size: 12px; margin-bottom: 16px; }

  .controls-bar {
    display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center;
    padding: 12px 0; margin-bottom: 28px;
    border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
  }
  .control-group {
    display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center;
    padding-right: 12px; border-right: 1px solid var(--border);
  }
  .control-group:last-of-type { border-right: none; }
  .group-label {
    font-size: 9px; font-family: var(--font-mono); color: var(--fg);
    text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
    padding-right: 4px;
  }
  .control-group label {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: var(--muted); font-family: var(--font-mono); white-space: nowrap;
  }
  .control-group input[type="range"] { width: 72px; }
  .val { color: var(--fg); min-width: 28px; text-align: right; }
  .control-actions { display: flex; align-items: center; gap: 8px; margin-left: auto; flex-wrap: wrap; }
  .stat { font-size: 11px; font-family: var(--font-mono); color: var(--muted); }

  .figures { display: flex; flex-direction: column; gap: 36px; }
  .figure h2 { font-size: 14px; font-weight: 600; margin-bottom: 2px; }
  .caption { font-size: 12px; color: var(--muted); line-height: 1.5; margin-bottom: 10px; max-width: 640px; }
  .chart { width: 100%; height: 200px; }
  .chart.small { height: 140px; }

  .simplex-pair { display: flex; gap: 12px; flex-wrap: wrap; }
  .simplex-canvas { width: 400px; height: 400px; }
</style>
