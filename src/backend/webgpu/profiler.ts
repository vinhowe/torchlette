/**
 * WebGPU Profiler — CPU API call timing + GPU timestamp queries.
 *
 * Activated by setting TORCHLETTE_PROFILE=1 environment variable.
 * When disabled, all functions are no-ops with zero overhead.
 */

const PROFILING_ENABLED =
  typeof process !== "undefined" && !!process.env?.TORCHLETTE_PROFILE;

// ---------------------------------------------------------------------------
// CPU-side API call profiling
// ---------------------------------------------------------------------------

interface ApiStats {
  count: number;
  totalMs: number;
  maxMs: number;
}
const apiStats = new Map<string, ApiStats>();

// ---------------------------------------------------------------------------
// Per-op CPU profiling (set from lazy.ts via setCurrentOpLabel)
// ---------------------------------------------------------------------------

interface OpStats {
  count: number;
  totalMs: number;
  maxMs: number;
}
const opStats = new Map<string, OpStats>();

// ---------------------------------------------------------------------------
// Phase tracking (forward / backward / optimizer / cleanup)
// ---------------------------------------------------------------------------

let currentPhase = "unknown";
const phaseStats = new Map<string, { totalMs: number; opCount: number }>();

// ---------------------------------------------------------------------------
// Module tracking (embedding / attention / mlp / layernorm / etc.)
// ---------------------------------------------------------------------------

let currentModule = "unknown";

// Per-module GPU stats
const gpuModuleStats = new Map<string, { totalNs: bigint; opCount: number }>();

// Per-module per-op GPU stats
const gpuModuleOpStats = new Map<
  string,
  Map<string, { count: number; totalNs: bigint; maxNs: bigint }>
>();

// ---------------------------------------------------------------------------
// GPU timestamp profiling
// ---------------------------------------------------------------------------

interface GpuPassRecord {
  label: string;
  phase: string;
  module: string;
  startSlot: number;
  endSlot: number;
}

let gpuQuerySet: GPUQuerySet | null = null;
let gpuResolveBuffer: GPUBuffer | null = null;
let gpuReadbackBuffer: GPUBuffer | null = null;
let gpuPassRecords: GpuPassRecord[] = [];
let gpuNextSlot = 0;
const GPU_MAX_PASSES = 2048; // 4096 timestamp slots (Dawn limit is 4096 queries)
let gpuMaxSlots = GPU_MAX_PASSES * 2;
let gpuTimestampsSupported = false;
let gpuDevice: GPUDevice | null = null;

// Accumulated GPU times per label
const gpuLabelStats = new Map<
  string,
  { count: number; totalNs: bigint; maxNs: bigint }
>();
const gpuPhaseStats = new Map<string, { totalNs: bigint; opCount: number }>();

// Per-phase, per-op GPU stats
const gpuPhaseOpStats = new Map<
  string,
  Map<string, { count: number; totalNs: bigint; maxNs: bigint }>
>();

// ---------------------------------------------------------------------------
// Plan analysis records
// ---------------------------------------------------------------------------

export interface PlanAnalysis {
  planIndex: number;
  totalNodes: number;
  segments: { fused: number; sequential: number };
  fusedNodes: number;
  fusionGroups: number;
  epilogueFusions: number;
  reductionFusions: number;
  sequentialOps: Record<string, number>;
  unfusedByShape: Record<string, { count: number; ops: Record<string, number> }>;
}

const planAnalyses: PlanAnalysis[] = [];

// ---------------------------------------------------------------------------
// Fusion fallback tracking
// ---------------------------------------------------------------------------

interface FusionFallbackEntry {
  count: number;
  totalNodes: number;
  details: unknown[];
}
const fusionFallbackStats = new Map<string, FusionFallbackEntry>();

export function recordFusionFallback(reason: string, groupSize: number, detail?: unknown): void {
  if (!PROFILING_ENABLED) return;
  const entry = fusionFallbackStats.get(reason);
  if (entry) {
    entry.count++;
    entry.totalNodes += groupSize;
    if (detail && entry.details.length < 3) entry.details.push(detail);
  } else {
    fusionFallbackStats.set(reason, {
      count: 1,
      totalNodes: groupSize,
      details: detail ? [detail] : [],
    });
  }
}

// ============================================================================
// Public API
// ============================================================================

export function isProfilingEnabled(): boolean {
  return PROFILING_ENABLED;
}

// ---------------------------------------------------------------------------
// CPU API call wrapping
// ---------------------------------------------------------------------------

export function profileApiCall<T>(name: string, fn: () => T): T {
  if (!PROFILING_ENABLED) return fn();
  const t0 = performance.now();
  const result = fn();
  const elapsed = performance.now() - t0;
  const entry = apiStats.get(name);
  if (entry) {
    entry.count++;
    entry.totalMs += elapsed;
    if (elapsed > entry.maxMs) entry.maxMs = elapsed;
  } else {
    apiStats.set(name, { count: 1, totalMs: elapsed, maxMs: elapsed });
  }
  return result;
}

// ---------------------------------------------------------------------------
// Per-op CPU timing (called from lazy.ts executePlan)
// ---------------------------------------------------------------------------

export function profileOpBegin(opName: string): number {
  if (!PROFILING_ENABLED) return 0;
  return performance.now();
}

export function profileOpEnd(opName: string, t0: number): void {
  if (!PROFILING_ENABLED) return;
  const elapsed = performance.now() - t0;

  // Op stats
  const entry = opStats.get(opName);
  if (entry) {
    entry.count++;
    entry.totalMs += elapsed;
    if (elapsed > entry.maxMs) entry.maxMs = elapsed;
  } else {
    opStats.set(opName, { count: 1, totalMs: elapsed, maxMs: elapsed });
  }

  // Phase stats
  const phase = phaseStats.get(currentPhase);
  if (phase) {
    phase.totalMs += elapsed;
    phase.opCount++;
  } else {
    phaseStats.set(currentPhase, { totalMs: elapsed, opCount: 1 });
  }
}

// ---------------------------------------------------------------------------
// Sub-op profiling (fine-grained breakdown within a single op dispatch)
// ---------------------------------------------------------------------------

const subOpStats = new Map<string, OpStats>();

export function profileSubOpBegin(): number {
  if (!PROFILING_ENABLED) return 0;
  return performance.now();
}

export function profileSubOpEnd(label: string, t0: number): void {
  if (!PROFILING_ENABLED) return;
  const elapsed = performance.now() - t0;
  const entry = subOpStats.get(label);
  if (entry) {
    entry.count++;
    entry.totalMs += elapsed;
    if (elapsed > entry.maxMs) entry.maxMs = elapsed;
  } else {
    subOpStats.set(label, { count: 1, totalMs: elapsed, maxMs: elapsed });
  }
}

// ---------------------------------------------------------------------------
// Phase control
// ---------------------------------------------------------------------------

export function setProfilePhase(phase: string): void {
  currentPhase = phase;
}

export function setProfileModule(module: string): void {
  currentModule = module;
}

export function getProfileModule(): string {
  return currentModule;
}

// ---------------------------------------------------------------------------
// Plan analysis recording
// ---------------------------------------------------------------------------

export function recordPlanAnalysis(analysis: PlanAnalysis): void {
  if (!PROFILING_ENABLED) return;
  analysis.planIndex = planAnalyses.length;
  planAnalyses.push(analysis);
}

// ---------------------------------------------------------------------------
// GPU timestamp initialization
// ---------------------------------------------------------------------------

export function initGpuTimestamps(device: GPUDevice): void {
  if (!PROFILING_ENABLED) return;
  gpuDevice = device;
  gpuTimestampsSupported = true;

  // Check device limits for max query count
  const maxQueryCount = (device.limits as any)?.maxQueryCount ?? 4096;
  const count = Math.min(GPU_MAX_PASSES * 2, maxQueryCount);
  gpuMaxSlots = count;

  // Use numeric constants for buffer usage to avoid dependency on globals
  // that may not be available in all environments
  const QUERY_RESOLVE = 0x0200;
  const COPY_SRC = 0x0004;
  const COPY_DST = 0x0008;
  const MAP_READ = 0x0001;

  gpuQuerySet = device.createQuerySet({
    type: "timestamp" as GPUQueryType,
    count,
  });

  // Resolve buffer: 8 bytes per timestamp slot
  gpuResolveBuffer = device.createBuffer({
    size: count * 8,
    usage: QUERY_RESOLVE | COPY_SRC,
  });

  // Readback buffer for mapping
  gpuReadbackBuffer = device.createBuffer({
    size: count * 8,
    usage: MAP_READ | COPY_DST,
  });

  console.log(`[profiler] GPU timestamps initialized: ${count} slots`);

  gpuPassRecords = [];
  gpuNextSlot = 0;
}

// ---------------------------------------------------------------------------
// Per-pass timestamp writes descriptor
// ---------------------------------------------------------------------------

/**
 * Returns a timestampWrites descriptor for the next compute pass.
 * Call this before beginComputePass(). Returns undefined if profiling
 * is disabled or GPU timestamps aren't supported.
 */
export function getTimestampWrites(
  label: string,
): GPUComputePassTimestampWrites | undefined {
  if (!PROFILING_ENABLED || !gpuTimestampsSupported || !gpuQuerySet) return undefined;
  if (gpuNextSlot + 2 > gpuMaxSlots) return undefined; // out of slots

  const startSlot = gpuNextSlot;
  const endSlot = gpuNextSlot + 1;
  gpuNextSlot += 2;

  gpuPassRecords.push({ label, phase: currentPhase, module: currentModule, startSlot, endSlot });

  return {
    querySet: gpuQuerySet,
    beginningOfPassWriteIndex: startSlot,
    endOfPassWriteIndex: endSlot,
  };
}

// ---------------------------------------------------------------------------
// Resolve timestamps into the resolve buffer (call before encoder.finish())
// ---------------------------------------------------------------------------

export function resolveGpuTimestamps(encoder: GPUCommandEncoder): void {
  if (
    !PROFILING_ENABLED ||
    !gpuTimestampsSupported ||
    !gpuQuerySet ||
    !gpuResolveBuffer ||
    gpuNextSlot === 0
  )
    return;

  encoder.resolveQuerySet(gpuQuerySet, 0, gpuNextSlot, gpuResolveBuffer, 0);

  // Copy resolve buffer to readback buffer
  if (gpuReadbackBuffer) {
    encoder.copyBufferToBuffer(
      gpuResolveBuffer,
      0,
      gpuReadbackBuffer,
      0,
      gpuNextSlot * 8,
    );
  }
}

// ---------------------------------------------------------------------------
// Read back GPU timestamps (async — call after queue.submit + onSubmittedWorkDone)
// ---------------------------------------------------------------------------

export async function readGpuTimestamps(): Promise<void> {
  if (
    !PROFILING_ENABLED ||
    !gpuTimestampsSupported ||
    !gpuReadbackBuffer ||
    !gpuDevice ||
    gpuPassRecords.length === 0
  )
    return;

  // Wait for GPU work to complete
  await gpuDevice.queue.onSubmittedWorkDone();

  try {
    await gpuReadbackBuffer.mapAsync(0x0001 /* MAP_READ */);
  } catch (e) {
    console.warn("[profiler] Failed to map readback buffer:", e);
    return;
  }

  const timestamps = new BigInt64Array(gpuReadbackBuffer.getMappedRange());

  for (const record of gpuPassRecords) {
    const startNs = timestamps[record.startSlot];
    const endNs = timestamps[record.endSlot];
    if (startNs === 0n && endNs === 0n) continue; // no data

    const durationNs = endNs - startNs;
    if (durationNs < 0n) continue; // invalid

    // Per-label stats
    const entry = gpuLabelStats.get(record.label);
    if (entry) {
      entry.count++;
      entry.totalNs += durationNs;
      if (durationNs > entry.maxNs) entry.maxNs = durationNs;
    } else {
      gpuLabelStats.set(record.label, {
        count: 1,
        totalNs: durationNs,
        maxNs: durationNs,
      });
    }

    // Per-phase GPU stats
    const phase = gpuPhaseStats.get(record.phase);
    if (phase) {
      phase.totalNs += durationNs;
      phase.opCount++;
    } else {
      gpuPhaseStats.set(record.phase, { totalNs: durationNs, opCount: 1 });
    }

    // Per-phase per-op GPU stats
    let phaseOps = gpuPhaseOpStats.get(record.phase);
    if (!phaseOps) {
      phaseOps = new Map();
      gpuPhaseOpStats.set(record.phase, phaseOps);
    }
    const poEntry = phaseOps.get(record.label);
    if (poEntry) {
      poEntry.count++;
      poEntry.totalNs += durationNs;
      if (durationNs > poEntry.maxNs) poEntry.maxNs = durationNs;
    } else {
      phaseOps.set(record.label, { count: 1, totalNs: durationNs, maxNs: durationNs });
    }

    // Per-module GPU stats
    const mod = gpuModuleStats.get(record.module);
    if (mod) {
      mod.totalNs += durationNs;
      mod.opCount++;
    } else {
      gpuModuleStats.set(record.module, { totalNs: durationNs, opCount: 1 });
    }

    // Per-module per-op GPU stats
    let modOps = gpuModuleOpStats.get(record.module);
    if (!modOps) {
      modOps = new Map();
      gpuModuleOpStats.set(record.module, modOps);
    }
    const moEntry = modOps.get(record.label);
    if (moEntry) {
      moEntry.count++;
      moEntry.totalNs += durationNs;
      if (durationNs > moEntry.maxNs) moEntry.maxNs = durationNs;
    } else {
      modOps.set(record.label, { count: 1, totalNs: durationNs, maxNs: durationNs });
    }
  }

  gpuReadbackBuffer.unmap();
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

function padR(s: string, len: number): string {
  return s.length >= len ? s : s + " ".repeat(len - s.length);
}

function padL(s: string, len: number): string {
  return s.length >= len ? s : " ".repeat(len - s.length) + s;
}

function nsToMs(ns: bigint): number {
  return Number(ns) / 1_000_000;
}

function nsToUs(ns: bigint): number {
  return Number(ns) / 1_000;
}

export function printProfileSummary(label: string): void {
  if (!PROFILING_ENABLED) return;

  console.log(`\n=== Profiling (${label}) ===\n`);

  // CPU API calls
  if (apiStats.size > 0) {
    const sorted = [...apiStats.entries()].sort(
      (a, b) => b[1].totalMs - a[1].totalMs,
    );
    console.log(
      `${padR("CPU API Call", 28)} ${padL("Count", 8)} ${padL("Total(ms)", 11)} ${padL("Avg(µs)", 10)} ${padL("Max(µs)", 10)}`,
    );
    console.log("─".repeat(69));
    for (const [name, s] of sorted) {
      const avgUs = (s.totalMs / s.count) * 1000;
      console.log(
        `${padR(name, 28)} ${padL(String(s.count), 8)} ${padL(s.totalMs.toFixed(1), 11)} ${padL(avgUs.toFixed(1), 10)} ${padL((s.maxMs * 1000).toFixed(1), 10)}`,
      );
    }
    console.log();
  }

  // CPU op type
  if (opStats.size > 0) {
    const sorted = [...opStats.entries()].sort(
      (a, b) => b[1].totalMs - a[1].totalMs,
    );
    console.log(
      `${padR("CPU Op Type", 28)} ${padL("Count", 8)} ${padL("Total(ms)", 11)} ${padL("Avg(µs)", 10)} ${padL("Max(µs)", 10)}`,
    );
    console.log("─".repeat(69));
    for (const [name, s] of sorted) {
      const avgUs = (s.totalMs / s.count) * 1000;
      console.log(
        `${padR(name, 28)} ${padL(String(s.count), 8)} ${padL(s.totalMs.toFixed(1), 11)} ${padL(avgUs.toFixed(1), 10)} ${padL((s.maxMs * 1000).toFixed(1), 10)}`,
      );
    }
    console.log();
  }

  // Sub-op breakdown (fine-grained timing within dispatch functions)
  if (subOpStats.size > 0) {
    const sorted = [...subOpStats.entries()].sort(
      (a, b) => b[1].totalMs - a[1].totalMs,
    );
    console.log(
      `${padR("CPU Sub-Op", 28)} ${padL("Count", 8)} ${padL("Total(ms)", 11)} ${padL("Avg(µs)", 10)} ${padL("Max(µs)", 10)}`,
    );
    console.log("─".repeat(69));
    for (const [name, s] of sorted) {
      const avgUs = (s.totalMs / s.count) * 1000;
      console.log(
        `${padR(name, 28)} ${padL(String(s.count), 8)} ${padL(s.totalMs.toFixed(1), 11)} ${padL(avgUs.toFixed(1), 10)} ${padL((s.maxMs * 1000).toFixed(1), 10)}`,
      );
    }
    console.log();
  }

  // GPU kernel time
  if (gpuLabelStats.size > 0) {
    const sorted = [...gpuLabelStats.entries()].sort(
      (a, b) => nsToMs(b[1].totalNs) - nsToMs(a[1].totalNs),
    );
    console.log(
      `${padR("GPU Kernel Time", 28)} ${padL("Count", 8)} ${padL("Total(ms)", 11)} ${padL("Avg(µs)", 10)} ${padL("Max(µs)", 10)}`,
    );
    console.log("─".repeat(69));
    for (const [name, s] of sorted) {
      const totalMs = nsToMs(s.totalNs);
      const avgUs = nsToUs(s.totalNs) / s.count;
      const maxUs = nsToUs(s.maxNs);
      console.log(
        `${padR(name, 28)} ${padL(String(s.count), 8)} ${padL(totalMs.toFixed(1), 11)} ${padL(avgUs.toFixed(1), 10)} ${padL(maxUs.toFixed(1), 10)}`,
      );
    }
    console.log();
  }

  // Phase summary
  if (phaseStats.size > 0 || gpuPhaseStats.size > 0) {
    const allPhases = new Set([
      ...phaseStats.keys(),
      ...gpuPhaseStats.keys(),
    ]);
    console.log(
      `${padR("Phase", 16)} ${padL("Ops", 8)} ${padL("CPU(ms)", 10)} ${padL("GPU(ms)", 10)}`,
    );
    console.log("─".repeat(46));
    for (const phase of allPhases) {
      const cpu = phaseStats.get(phase);
      const gpu = gpuPhaseStats.get(phase);
      const ops = cpu?.opCount ?? gpu?.opCount ?? 0;
      const cpuMs = cpu?.totalMs ?? 0;
      const gpuMs = gpu ? nsToMs(gpu.totalNs) : 0;
      console.log(
        `${padR(phase, 16)} ${padL(String(ops), 8)} ${padL(cpuMs.toFixed(1), 10)} ${padL(gpuMs.toFixed(1), 10)}`,
      );
    }
    console.log();
  }

  // Per-phase GPU kernel breakdown
  if (gpuPhaseOpStats.size > 0) {
    console.log("=== Per-Phase GPU Breakdown ===\n");
    for (const [phase, opsMap] of gpuPhaseOpStats) {
      const gpuPhase = gpuPhaseStats.get(phase);
      const phaseGpuMs = gpuPhase ? nsToMs(gpuPhase.totalNs) : 0;
      const phaseDispatches = gpuPhase?.opCount ?? 0;
      console.log(`--- ${phase} (${phaseDispatches} dispatches, ${phaseGpuMs.toFixed(0)}ms GPU) ---`);
      const sorted = [...opsMap.entries()].sort(
        (a, b) => nsToMs(b[1].totalNs) - nsToMs(a[1].totalNs),
      );
      console.log(
        `${padR("  Kernel", 30)} ${padL("Count", 8)} ${padL("Total(ms)", 11)} ${padL("Avg(µs)", 10)}`,
      );
      for (const [op, s] of sorted) {
        const totalMs = nsToMs(s.totalNs);
        const avgUs = nsToUs(s.totalNs) / s.count;
        console.log(
          `${padR("  " + op, 30)} ${padL(String(s.count), 8)} ${padL(totalMs.toFixed(1), 11)} ${padL(avgUs.toFixed(0), 10)}`,
        );
      }
      console.log();
    }
  }

  // Per-module GPU summary
  if (gpuModuleStats.size > 0) {
    const totalGpuNs = [...gpuModuleStats.values()].reduce((s, v) => s + v.totalNs, 0n);
    const sortedModules = [...gpuModuleStats.entries()].sort(
      (a, b) => nsToMs(b[1].totalNs) - nsToMs(a[1].totalNs),
    );
    console.log("=== Per-Module GPU Breakdown ===\n");
    console.log(
      `${padR("Module", 28)} ${padL("Dispatches", 11)} ${padL("GPU(ms)", 10)} ${padL("% GPU", 8)}`,
    );
    console.log("─".repeat(59));
    for (const [mod, s] of sortedModules) {
      const ms = nsToMs(s.totalNs);
      const pct = totalGpuNs > 0n ? (Number(s.totalNs * 10000n / totalGpuNs) / 100).toFixed(1) : "0.0";
      console.log(
        `${padR(mod, 28)} ${padL(String(s.opCount), 11)} ${padL(ms.toFixed(1), 10)} ${padL(pct + "%", 8)}`,
      );
    }
    console.log();

    // Detailed per-module per-op breakdown
    console.log("=== Per-Module Kernel Detail ===\n");
    for (const [mod, s] of sortedModules) {
      const modMs = nsToMs(s.totalNs);
      const opsMap = gpuModuleOpStats.get(mod);
      if (!opsMap || opsMap.size === 0) continue;
      console.log(`--- ${mod} (${s.opCount} dispatches, ${modMs.toFixed(0)}ms GPU) ---`);
      const sorted = [...opsMap.entries()].sort(
        (a, b) => nsToMs(b[1].totalNs) - nsToMs(a[1].totalNs),
      );
      console.log(
        `${padR("  Kernel", 30)} ${padL("Count", 8)} ${padL("Total(ms)", 11)} ${padL("Avg(µs)", 10)}`,
      );
      for (const [op, os] of sorted) {
        const totalMs = nsToMs(os.totalNs);
        const avgUs = nsToUs(os.totalNs) / os.count;
        console.log(
          `${padR("  " + op, 30)} ${padL(String(os.count), 8)} ${padL(totalMs.toFixed(1), 11)} ${padL(avgUs.toFixed(0), 10)}`,
        );
      }
      console.log();
    }
  }

  // Fusion fallback stats
  if (fusionFallbackStats.size > 0) {
    const sorted = [...fusionFallbackStats.entries()].sort(
      (a, b) => b[1].count - a[1].count,
    );
    console.log("=== Fusion Fallback Reasons ===\n");
    console.log(
      `${padR("Reason", 24)} ${padL("Count", 8)} ${padL("Nodes Lost", 12)}`,
    );
    console.log("─".repeat(46));
    for (const [reason, s] of sorted) {
      console.log(
        `${padR(reason, 24)} ${padL(String(s.count), 8)} ${padL(String(s.totalNodes), 12)}`,
      );
      for (const d of s.details) {
        console.log(`    detail: ${JSON.stringify(d)}`);
      }
    }
    console.log();
  }

  // Plan analysis
  if (planAnalyses.length > 0) {
    console.log("=== Plan Analysis ===\n");
    for (const pa of planAnalyses) {
      const fusionRate = pa.totalNodes > 0 ? (pa.fusedNodes / pa.totalNodes * 100).toFixed(1) : "0.0";
      console.log(
        `Plan ${pa.planIndex}: ${pa.totalNodes} nodes (${pa.fusedNodes} fused/${pa.totalNodes - pa.fusedNodes} seq, ${pa.fusionGroups} groups, ${fusionRate}% fused)`,
      );
      if (pa.epilogueFusions > 0 || pa.reductionFusions > 0) {
        console.log(`  Epilogue fusions: ${pa.epilogueFusions}, Reduction fusions: ${pa.reductionFusions}`);
      }
      // Top unfused ops
      const seqEntries = Object.entries(pa.sequentialOps).sort((a, b) => b[1] - a[1]);
      if (seqEntries.length > 0) {
        const top = seqEntries.slice(0, 10).map(([op, n]) => `${op}:${n}`).join(", ");
        console.log(`  Top unfused ops: ${top}`);
      }
      // Unfused fusible by shape
      const shapeEntries = Object.entries(pa.unfusedByShape).sort((a, b) => b[1].count - a[1].count);
      if (shapeEntries.length > 0) {
        console.log("  Unfused fusible by shape:");
        for (const [shape, info] of shapeEntries.slice(0, 8)) {
          const opsStr = Object.entries(info.ops).sort((a, b) => b[1] - a[1]).map(([op, n]) => `${op}:${n}`).join(", ");
          console.log(`    [${shape}]: ${info.count} ops (${opsStr})`);
        }
      }
      console.log();
    }
  }

  // Auto-write JSON if env var is set
  const jsonPath = typeof process !== "undefined" ? process.env?.TORCHLETTE_PROFILE_JSON : undefined;
  if (jsonPath) {
    writeProfileJSON(jsonPath);
  }
}

// ---------------------------------------------------------------------------
// JSON export
// ---------------------------------------------------------------------------

export function getProfileJSON(): object {
  // Build phases object with per-op kernel breakdown
  const phases: Record<string, any> = {};
  const allPhases = new Set([
    ...phaseStats.keys(),
    ...gpuPhaseStats.keys(),
    ...gpuPhaseOpStats.keys(),
  ]);
  for (const phase of allPhases) {
    const cpu = phaseStats.get(phase);
    const gpu = gpuPhaseStats.get(phase);
    const kernels: Record<string, any> = {};
    const phaseOps = gpuPhaseOpStats.get(phase);
    if (phaseOps) {
      for (const [op, s] of phaseOps) {
        kernels[op] = {
          count: s.count,
          gpu_ms: nsToMs(s.totalNs),
          avg_us: nsToUs(s.totalNs) / s.count,
          max_us: nsToUs(s.maxNs),
        };
      }
    }
    // Add CPU op times for this phase from opStats (approximate: opStats is global, not per-phase)
    phases[phase] = {
      cpu_ms: cpu?.totalMs ?? 0,
      gpu_ms: gpu ? nsToMs(gpu.totalNs) : 0,
      dispatch_count: gpu?.opCount ?? 0,
      kernels,
    };
  }

  // GPU totals
  const gpu_totals: Record<string, any> = {};
  for (const [op, s] of gpuLabelStats) {
    gpu_totals[op] = {
      count: s.count,
      gpu_ms: nsToMs(s.totalNs),
      avg_us: nsToUs(s.totalNs) / s.count,
      max_us: nsToUs(s.maxNs),
    };
  }

  // Compute summary
  let totalGpuMs = 0;
  let totalDispatches = 0;
  for (const [, s] of gpuPhaseStats) {
    totalGpuMs += nsToMs(s.totalNs);
    totalDispatches += s.opCount;
  }

  // Fusion rate from plan analyses
  let totalNodes = 0;
  let totalFused = 0;
  for (const pa of planAnalyses) {
    totalNodes += pa.totalNodes;
    totalFused += pa.fusedNodes;
  }
  const fusionRate = totalNodes > 0 ? totalFused / totalNodes : 0;

  // Top bottlenecks: (phase, op) pairs sorted by GPU time
  const bottlenecks: { phase: string; op: string; gpu_ms: number }[] = [];
  for (const [phase, opsMap] of gpuPhaseOpStats) {
    for (const [op, s] of opsMap) {
      bottlenecks.push({ phase, op, gpu_ms: nsToMs(s.totalNs) });
    }
  }
  bottlenecks.sort((a, b) => b.gpu_ms - a.gpu_ms);
  const topBottlenecks = bottlenecks.slice(0, 5).map(
    (b) => `${b.phase}/${b.op} (${b.gpu_ms.toFixed(1)}ms)`,
  );

  // Per-module breakdown
  const modules: Record<string, any> = {};
  for (const [mod, s] of gpuModuleStats) {
    const kernels: Record<string, any> = {};
    const modOps = gpuModuleOpStats.get(mod);
    if (modOps) {
      for (const [op, os] of modOps) {
        kernels[op] = {
          count: os.count,
          gpu_ms: nsToMs(os.totalNs),
          avg_us: nsToUs(os.totalNs) / os.count,
        };
      }
    }
    modules[mod] = {
      gpu_ms: nsToMs(s.totalNs),
      dispatch_count: s.opCount,
      kernels,
    };
  }

  return {
    phases,
    modules,
    plans: planAnalyses,
    gpu_totals,
    summary: {
      total_gpu_ms: totalGpuMs,
      total_dispatches: totalDispatches,
      fusion_rate: fusionRate,
      top_bottlenecks: topBottlenecks,
    },
  };
}

export function writeProfileJSON(filePath: string): void {
  if (!PROFILING_ENABLED) return;
  try {
    const fs = require("node:fs");
    const json = getProfileJSON();
    fs.writeFileSync(filePath, JSON.stringify(json, null, 2));
    console.log(`[profiler] Wrote profile JSON to ${filePath}`);
  } catch (e) {
    console.warn(`[profiler] Failed to write profile JSON to ${filePath}:`, e);
  }
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

export function resetProfileStats(): void {
  if (!PROFILING_ENABLED) return;
  apiStats.clear();
  opStats.clear();
  phaseStats.clear();
  gpuLabelStats.clear();
  gpuPhaseStats.clear();
  gpuPhaseOpStats.clear();
  gpuModuleStats.clear();
  gpuModuleOpStats.clear();
  subOpStats.clear();
  planAnalyses.length = 0;
  fusionFallbackStats.clear();
  gpuPassRecords = [];
  gpuNextSlot = 0;
}
