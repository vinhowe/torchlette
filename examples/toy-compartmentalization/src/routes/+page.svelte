<script lang="ts">
  import "../app.css";
  import { onMount } from "svelte";
  import { createModel, MESS3_CONFIG, type MESS3Model } from "$lib/model";
  import {
    generateBatch, generateBatchWithCompartments, generateBatchForComp, generatePairedBatch,
    theoreticalEntropy, exploreBeliefSimplex,
    setTransitionMatrices,
    VOCAB_SIZE_DATA, NUM_STATES,
  } from "$lib/data";
  import { config, initConfigUrlSync, resetConfigToDefaults, describeConfigDelta } from "$lib/mess3-config.svelte";
  import { RpcClient } from "$lib/remote-transport";
  import { createRemoteEngine, type RemoteEngine, type RemoteEngineStats } from "../../../../src/remote/client-engine";

  import { THEME, COMP_COLORS, TOKEN_COLORS_RGB, SERIES_PALETTE } from "$lib/theme";
  import { baseChartOpt, chartAxes, lineSeries, refLine, legendBlock } from "$lib/chart-helpers";
  import { DemoPage, Figure, LineChart, StatsBar, Stat } from "$lib/components";

  // Piston controls
  import { BorderedGroup, CheckboxInput, NumberInput, Slider, TextInput } from "piston-controls";

  initConfigUrlSync();

  // Toggling the remote checkbox swaps the engine in place — archives the
  // current run so its trajectory survives, tears down the old transport if
  // any, sets up the new one, and rebuilds the model on the new api. No
  // page reload, so archivedRuns and config UI state stay intact and you
  // can compare local vs remote runs side-by-side on the same charts.
  $effect(() => {
    const wantRemote = config.remote.enabled;
    if (gpuReady && wantRemote !== isRemote) {
      swapEngine(wantRemote).catch((e: any) => {
        gpuError = e?.message ?? String(e);
      });
    }
  });

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
  let serverHandles = $state(0);
  let isRemote = $state(false);

  let training = $state(false);
  let trainingActive = $state(false);
  let step = $state(0);
  let stepMs = $state(0);
  let stepsPerSec = $state(0);
  let phaseTimings = $state('');
  let lossHistory: number[] = $state([]);
  let memoryHistory: number[] = $state([]);
  let model: MESS3Model | null = $state(null);
  let optimizer: any = $state(null);

  // Simplex visualization data
  let gtBeliefs: { b0: number; b1: number; b2: number }[] = $state([]);
  let predBeliefs: number[][] = $state([]);
  let probeW: Float64Array | null = null;
  let probeBias: Float64Array | null = null;
  let probeR2: number = $state(0);
  let probeR2PerComp: number[] = $state([]);
  let probeR2History: number[] = $state([]);
  let probeR2HistoryPerComp: number[][] = $state([]);

  const LOG_INTERVAL = 10;
  const VIZ_INTERVAL = 50;
  const PROBE_INTERVAL = 200;

  // Token colors used by the belief→RGB mapping (sourced from shared theme)
  const TOKEN_COLORS = TOKEN_COLORS_RGB;

  let gtCanvas: HTMLCanvasElement;
  let predCanvas: HTMLCanvasElement;

  let entropy = $state(theoreticalEntropy());

  function syncSelfLoop() {
    setTransitionMatrices(config.world.selfLoop);
    entropy = theoreticalEntropy();
    gtBeliefs = exploreBeliefSimplex(9);
    redrawSimplexes();
  }

  // Build (or rebuild) the compute engine. Called once at mount and again
  // by swapEngine() when the user toggles the remote checkbox.
  async function setupEngine(wantRemote: boolean): Promise<void> {
    const tl = await import("torchlette");
    // Both modes need WebGPU initialized client-side. Remote mode used to
    // skip this and let createRemoteEngine register a CPU stub backend,
    // but the cpu device path corrupted the lazy-graph dtype/op decisions
    // and made remote training silently diverge from local. The client
    // now always builds webgpu-flavored plans, even in remote mode.
    await tl.initWebGPU();
    if (wantRemote) {
      rpcClient = new RpcClient(config.remote.url, (msg) => console.log(msg));
      await rpcClient.connect();
      remoteEngine = createRemoteEngine(rpcClient);
      remoteStats = remoteEngine.stats;
      isRemote = true;
      api = remoteEngine.torch;
    } else {
      rpcClient = null;
      remoteEngine = null;
      remoteStats = null;
      isRemote = false;
      api = new tl.Torchlette("webgpu", { enableFusion: true, memoryLimitBytes: 8 * 1024 * 1024 * 1024 });
    }
    nn = tl.nn;
    Adam = tl.Adam;
    crossEntropy = tl.nn.functional.crossEntropy;
    getGPUMemoryStats = tl.getGPUMemoryStats;
    gpuReady = true;
  }

  // Hot-swap the engine in place. Stops training, archives the current run
  // so it stays on the chart, drops references to the old engine + model
  // (their GPU buffers will be reclaimed by GC + finalization), then sets
  // up the new engine and rebuilds the model on it.
  async function swapEngine(wantRemote: boolean): Promise<void> {
    await stopAndWait();
    archiveCurrentRun();

    // CRITICAL: force any in-flight pending tensors on the OLD api so they
    // materialize on the OLD backend. The pending-tensor registry is
    // process-global — without this, the new api's beginStep (called from
    // resetModel below) would invoke forceAllPending, pick up the old
    // model's pending tensors, and try to execute them on the wrong
    // backend (e.g., dispatching a CPU matmul against a tensor that has
    // no `.data` array because it was supposed to live on WebGPU).
    // Once materialized, collectPendingRoots in engine.ts skips them.
    if (api) {
      try {
        await api.runtime.forceAllPending();
      } catch (e: any) {
        console.warn('[swap] force on old api failed:', e?.message ?? e);
      }
    }

    // If we're leaving remote mode, close the WebSocket so the server can
    // recycle its caches. (The server's ws.on('close') handler runs the
    // executor recycle when the last session disconnects.)
    if (rpcClient && !wantRemote) {
      rpcClient.close();
    }

    model = null;
    optimizer = null;

    await setupEngine(wantRemote);
    await resetModel();
  }

  onMount(() => {
    let cleanup: (() => void) | undefined;
    (async () => {
      try {
        await setupEngine(config.remote.enabled);
      } catch (e: any) {
        gpuError = e.message || String(e);
      }
      const onMove = (e: MouseEvent) => {
        if (!dragging3d) return;
        azimuth = dragStartAz + (e.clientX - dragStartX) * 0.008;
        elevation = dragStartEl + (e.clientY - dragStartY) * 0.008;
        elevation = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, elevation));
      };
      const onUp = () => { dragging3d = false; };
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);

      if (gtCanvas) initDrag(gtCanvas);
      if (predCanvas) initDrag(predCanvas);
      renderLoopRunning = true;
      render3DLoop();
      syncSelfLoop();

      cleanup = () => {
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
        renderLoopRunning = false;
      };
    })();
    return () => cleanup?.();
  });

  // ── Previous-run overlay ──────────────────────────────────────────────────
  // Reset archives the current run so its trajectory is preserved as a faded
  // line behind the new run; clear wipes both archived and current. Mirrors
  // the pattern used by bio/brackets/xor.
  type ArchivedRun = {
    /** "local" if the run was on in-process WebGPU, "remote" if it went
     *  through the Node WebSocket server. Drives the legend label so the
     *  user can tell two overlaid lines apart at a glance. */
    mode: "local" | "remote";
    label: string;
    lossHistory: number[];
    probeR2History: number[];
    probeR2HistoryPerComp: number[][];
  };
  let archivedRuns: ArchivedRun[] = $state([]);

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

  // Format an archived run's legend label as "#N [mode] config-delta"
  // so two overlaid lines (e.g. local vs remote on the same model) are
  // immediately distinguishable in the legend.
  const archiveLabel = (r: ArchivedRun, i: number) =>
    `#${i + 1} [${r.mode}] ${r.label}`;
  const currentLabel = () =>
    `current [${isRemote ? "remote" : "local"}] ${describeConfigDelta(config)}`;

  // Reactive echarts options — re-derived whenever their inputs change.
  let lossOption = $derived({
    ...baseChartOpt(),
    grid: { top: archivedRuns.length > 0 ? 48 : 28, right: 12, bottom: 24, left: 44 },
    ...(archivedRuns.length > 0
      ? { legend: { ...legendBlock(), top: 2, left: 44, right: 12, type: 'scroll' } }
      : {}),
    ...chartAxes({ yType: 'log' }),
    series: [
      ...archivedRuns.map((r, i) =>
        lineSeriesXY(archiveLabel(r, i), r.lossHistory, LOG_INTERVAL, runColor(i), false),
      ),
      {
        ...lineSeriesXY(currentLabel(), lossHistory, LOG_INTERVAL, THEME.accent, true),
        markLine: refLine(entropy, `H=${entropy.toFixed(3)}`),
      },
    ],
  });

  let r2Option = $derived.by(() => {
    const series: any[] = [];
    // Faded archived runs first, current run on top.
    for (let i = 0; i < archivedRuns.length; i++) {
      const r = archivedRuns[i];
      const c = runColor(i);
      const tag = `#${i + 1} [${r.mode}]`;
      if (r.probeR2HistoryPerComp.length > 0) {
        for (let k = 0; k < r.probeR2HistoryPerComp.length; k++) {
          series.push(lineSeriesXY(`${tag} c${k}`, r.probeR2HistoryPerComp[k], VIZ_INTERVAL, c, false));
        }
      } else {
        series.push(lineSeriesXY(tag, r.probeR2History, VIZ_INTERVAL, c, false));
      }
    }
    if (probeR2HistoryPerComp.length > 0) {
      for (let c = 0; c < probeR2HistoryPerComp.length; c++) {
        series.push({
          name: c === 0 ? 'comp 0 (fitted)' : `comp ${c} (transfer)`,
          type: 'line',
          data: probeR2HistoryPerComp[c].map((y, i) => [i * VIZ_INTERVAL, y]),
          showSymbol: true, symbolSize: 3,
          lineStyle: { width: c === 0 ? 2 : 1.5, color: COMP_COLORS[c % COMP_COLORS.length] },
          itemStyle: { color: COMP_COLORS[c % COMP_COLORS.length] },
          z: 10,
        });
      }
    } else {
      series.push({
        name: 'R²', type: 'line', data: probeR2History.map((y, i) => [i * VIZ_INTERVAL, y]),
        showSymbol: true, symbolSize: 4,
        lineStyle: { width: 1.5, color: COMP_COLORS[0] },
        itemStyle: { color: COMP_COLORS[0] },
        z: 10,
      });
    }
    return {
      ...baseChartOpt(),
      grid: { top: archivedRuns.length > 0 ? 48 : 28, right: 12, bottom: 24, left: 44 },
      ...(archivedRuns.length > 0 || probeR2HistoryPerComp.length > 1
        ? { legend: { ...legendBlock(), top: 2, left: 44, right: 12, type: 'scroll' } }
        : {}),
      ...chartAxes({ yMin: 0, yMax: 1 }),
      series,
    };
  });

  let memoryOption = $derived({
    ...baseChartOpt(),
    ...chartAxes({ xType: 'category', xData: memoryHistory.map((_, i) => i * LOG_INTERVAL) }),
    yAxis: {
      type: 'value' as const, name: 'MB',
      nameTextStyle: { color: THEME.muted, fontSize: 10 },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: THEME.grid } },
      axisLabel: { color: THEME.muted },
    },
    series: [lineSeries(memoryHistory, { color: THEME.accent2, area: true })],
  });

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
    ctx.fillStyle = THEME.bg; ctx.fillRect(0, 0, CW, CH);
    const cx = CW / 2, cy = CH / 2, scale = Math.min(CW, CH) * 0.35 * zoom3d;
    const cosA = Math.cos(azimuth), sinA = Math.sin(azimuth), cosE = Math.cos(elevation), sinE = Math.sin(elevation);

    // Wireframe cube
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

  // Precomputed 3D data
  let gtPoints3D: { x: number; y: number; z: number }[] = [];
  let gtColors: [number, number, number][] = [];
  let predPoints3D: { x: number; y: number; z: number }[] = [];
  let predColors: [number, number, number][] = [];
  let perCompPredBeliefs: number[][][] = [];

  // Compartment colors for overlay visualization
  const COMP_RGB: [number, number, number][] = [
    [212, 160, 23],   // gold (comp 0 — fitted)
    [194, 24, 91],    // crimson
    [56, 142, 60],    // green
    [123, 31, 162],   // purple
    [230, 81, 0],     // orange
    [0, 151, 167],    // teal
    [156, 39, 176],   // magenta
    [97, 97, 97],     // gray
  ];

  function toSimplex3D(p0: number, p1: number, p2: number) {
    return { x: -p0 + p1, y: 0, z: -0.577 * (p0 + p1) + 1.155 * p2 };
  }

  function redrawSimplexes() {
    gtPoints3D = gtBeliefs.map(b => toSimplex3D(b.b0, b.b1, b.b2));
    gtColors = gtBeliefs.map(b => beliefRGB(b.b0, b.b1, b.b2));

    if (config.world.nCompartments > 1 && perCompPredBeliefs.length > 0) {
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

  // Effect: when lr changes via Slider, push to optimizer if it exists.
  $effect(() => {
    if (optimizer) optimizer.setLR(config.optim.lr);
  });

  async function resetModel() {
    step = 0;
    lossHistory = []; memoryHistory = [];
    predBeliefs = [];
    probeW = null; probeBias = null; probeR2 = 0;
    probeR2History = []; probeR2PerComp = []; probeR2HistoryPerComp = [];

    await api.beginStep();
    api.manualSeed(config.init.seed);
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
    if (remoteEngine) {
      const uploaded = await remoteEngine.preUpload(model.parameters());
      console.log(`[preUpload] ${uploaded} param tensors`);
    }
    api.endStep();
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

    if (shouldLog) {
      const val = await loss.item();
      if (Number.isFinite(val)) {
        lossHistory = [...lossHistory, val];
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
        if (typeof s.handles === 'number') serverHandles = s.handles;
      } catch { /* ignore */ }
    }
  }

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
    const comp0 = await extractActivationsAndBeliefs(0);

    if (refitProbe || !probeW) {
      fitLinearProbe(comp0.activations, comp0.beliefs);
    }

    if (probeW && probeBias) {
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

      const nC = Math.min(4, config.world.nCompartments);
      const r2s: number[] = [computeR2(comp0.activations, comp0.beliefs)];
      const allCompPreds: number[][][] = [pred];
      for (let c = 1; c < nC; c++) {
        const compC = await extractActivationsAndBeliefs(c);
        r2s.push(computeR2(compC.activations, compC.beliefs));
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
    }

    redrawSimplexes();
  }

  function fitLinearProbe(activations: number[][], beliefs: number[][]) {
    const n = activations.length;
    const d = activations[0].length;
    const dp1 = d + 1;
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

  function archiveCurrentRun() {
    if (lossHistory.length === 0) return;
    archivedRuns = [
      ...archivedRuns,
      {
        mode: isRemote ? "remote" : "local",
        label: describeConfigDelta(config),
        lossHistory: [...lossHistory],
        probeR2History: [...probeR2History],
        probeR2HistoryPerComp: probeR2HistoryPerComp.map((h) => [...h]),
      },
    ];
  }

  async function clearAll() {
    await stopAndWait();
    model = null; optimizer = null; step = 0;
    lossHistory = []; memoryHistory = [];
    gtBeliefs = []; predBeliefs = [];
    probeW = null; probeBias = null; probeR2 = 0;
    probeR2History = []; probeR2PerComp = []; probeR2HistoryPerComp = [];
    archivedRuns = [];
  }

  let lastLoss = $derived(lossHistory.at(-1) ?? 0);
  let paramCount = $derived(model ? model.parameters().reduce((s: number, p: any) => s + p.shape.reduce((a: number, b: number) => a * b, 1), 0) : 0);
</script>

<DemoPage
  title="Belief State Geometry in Transformers"
  currentRoute="mess3"
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
    A tiny transformer learns next-token prediction on sequences from the
    <em class="italic text-[rgba(0,0,0,0.84)]">MESS3</em> process —
    a 3-state HMM. The optimal predictor must track a posterior over hidden states. We probe the
    residual stream to recover the belief simplex geometry live during training.
  {/snippet}

  {#snippet controls()}
    <BorderedGroup title="World" id="grp-world" contentClass="p-2 space-y-2">
      <Slider id="self-loop" label="Self-loop probability" min={0.3} max={0.9} step={0.01}
              bind:value={config.world.selfLoop} />
      <NumberInput id="ncomp" label="Compartments" min={1} max={8} step={1}
                   bind:value={config.world.nCompartments} />
      <Slider id="trpct" label="Translation %" min={0} max={100} step={1}
              bind:value={config.objective.translationPct} unit="%" />
    </BorderedGroup>

    <BorderedGroup title="Init" id="grp-init" contentClass="p-2 space-y-2">
      <NumberInput id="seed" label="Seed" min={0} step={1}
                   bind:value={config.init.seed} />
      <Slider id="wscale" label="Weight scale" min={0.01} max={10} step={0.01} useLog={true}
              bind:value={config.init.weightScale} />
    </BorderedGroup>

    <BorderedGroup title="Model" id="grp-model" contentClass="p-2 space-y-2">
      <Slider id="ctx" label="Context length" min={5} max={20} step={1}
              bind:value={config.model.seqLen} />
      <NumberInput id="probebs" label="Probe batch" min={32} max={2048} step={32}
                   bind:value={config.probe.batchSize} />
    </BorderedGroup>

    <BorderedGroup title="Optimizer" id="grp-optim" contentClass="p-2 space-y-2">
      <Slider id="lr" label="Learning rate" min={1e-5} max={1} step={1e-5} useLog={true}
              bind:value={config.optim.lr} />
      <Slider id="wd" label="Weight decay" min={0} max={0.5} step={0.01}
              bind:value={config.optim.weightDecay} />
      <Slider id="bs" label="Batch size" min={16} max={2048} step={16}
              bind:value={config.optim.batchSize} />
    </BorderedGroup>

    <BorderedGroup title="Remote" id="grp-remote" contentClass="p-2 space-y-2">
      <CheckboxInput id="remote-enable" label="Use remote GPU server"
                     bind:checked={config.remote.enabled} />
      <TextInput id="remote-url" label="WebSocket URL" bind:value={config.remote.url} />
    </BorderedGroup>
  {/snippet}

  {#snippet stats()}
    <StatsBar>
      <Stat label="step" value={step} />
      <Stat label="loss" value={lastLoss > 0 ? lastLoss.toFixed(4) : '—'} />
      {#if probeR2PerComp.length > 0}
        <Stat label="R²" value={`[${probeR2PerComp.map(r => r.toFixed(2)).join(', ')}]`} />
      {:else if probeR2 > 0}
        <Stat label="R²" value={probeR2.toFixed(3)} />
      {/if}
      {#if paramCount}<span>{(paramCount / 1000).toFixed(1)}k params</span>{/if}
      {#if stepMs > 0}<span>{stepMs.toFixed(0)} ms/step</span>{/if}
      {#if stepsPerSec > 0}<span>{stepsPerSec.toFixed(1)} steps/s</span>{/if}
      {#if phaseTimings}<Stat value={phaseTimings} mono title="Phase breakdown (ms)" />{/if}
      {#if remoteStats}
        <Stat
          mono
          value={`rpc ${remoteStats.executes}× ${(remoteStats.bytesUp / 1024).toFixed(0)} KB↑ ${(remoteStats.bytesDown / 1024).toFixed(0)} KB↓ h=${remoteEngine?.handles.size() ?? 0} rel=${remoteStats.handlesReleased}`}
        />
        {#if serverHandles > 0}<Stat label="srv-h" value={serverHandles} title="Handles currently held by the server (the leak indicator — should stay flat in steady state)" />{/if}
        {#if serverGpuMB}<Stat label="srv-gpu" value={serverGpuMB} />{/if}
      {/if}
    </StatsBar>
  {/snippet}

  {#snippet figures()}
    <Figure
      title="Training loss"
      caption="Cross-entropy on MESS3. Dotted line marks the entropy floor H = {entropy.toFixed(3)} nats."
    >
      <LineChart option={lossOption} />
    </Figure>

    <Figure title="Belief simplex">
      {#snippet caption()}
        Each point is a (position, sequence) pair. Left: ground-truth beliefs from the HMM
        (color = belief distribution: red/green/blue for S1/S2/S3).
        Right: affine linear probe of the last-layer residual stream (R² measures probe fit).
        {#if config.world.nCompartments > 1}
          With more than one compartment, the probe is fitted on comp 0 only. Dots are colored by
          compartment — if the model is unified, all colors trace the same fractal; if
          compartmentalized, only comp 0 (gold) shows the fractal while the others scatter.
        {:else}
          The fractal should emerge as training converges.
        {/if}
      {/snippet}
      <div class="flex flex-wrap gap-4">
        <canvas bind:this={gtCanvas} class="h-[400px] w-[400px] border border-[rgba(0,0,0,0.08)] bg-[#fffaf3]"></canvas>
        <canvas bind:this={predCanvas} class="h-[400px] w-[400px] border border-[rgba(0,0,0,0.08)] bg-[#fffaf3]"></canvas>
      </div>
    </Figure>

    <Figure
      title="Linear probe R²"
      caption="How much belief-state variance the residual stream linearly encodes. 1.0 = perfect recovery."
    >
      <LineChart option={r2Option} height={160} />
    </Figure>

    <Figure
      title="GPU memory"
      caption="Allocated GPU buffer memory (local mode only)."
    >
      <LineChart option={memoryOption} height={160} />
    </Figure>
  {/snippet}
</DemoPage>
