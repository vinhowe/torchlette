/**
 * WebGPU Profiler — CPU API call timing + GPU timestamp queries.
 *
 * Activated by setting TORCHLETTE_PROFILE=1 environment variable.
 * When disabled, all functions are no-ops with zero overhead.
 */

const PROFILING_ENABLED =
  typeof process !== "undefined" && !!process.env?.TORCHLETTE_PROFILE;

// ---------------------------------------------------------------------------
// Stat entry types
// ---------------------------------------------------------------------------

interface ApiStats {
  count: number;
  totalMs: number;
  maxMs: number;
}

interface OpStats {
  count: number;
  totalMs: number;
  maxMs: number;
}

interface GpuPassRecord {
  label: string;
  phase: string;
  module: string;
  startSlot: number;
  endSlot: number;
}

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

interface FusionFallbackEntry {
  count: number;
  totalNodes: number;
  details: unknown[];
}

// ---------------------------------------------------------------------------
// CPU-side profiling state
// ---------------------------------------------------------------------------

interface CpuProfileState {
  apiStats: Map<string, ApiStats>;
  opStats: Map<string, OpStats>;
  subOpStats: Map<string, OpStats>;
  currentPhase: string;
  phaseStats: Map<string, { totalMs: number; opCount: number }>;
  currentModule: string;
  planAnalyses: PlanAnalysis[];
  fusionFallbackStats: Map<string, FusionFallbackEntry>;
}

const cpuProfile: CpuProfileState = {
  apiStats: new Map(),
  opStats: new Map(),
  subOpStats: new Map(),
  currentPhase: "unknown",
  phaseStats: new Map(),
  currentModule: "unknown",
  planAnalyses: [],
  fusionFallbackStats: new Map(),
};

/** Reset CPU-side profiling state. */
function resetCpuProfileState(): void {
  cpuProfile.apiStats.clear();
  cpuProfile.opStats.clear();
  cpuProfile.subOpStats.clear();
  cpuProfile.currentPhase = "unknown";
  cpuProfile.phaseStats.clear();
  cpuProfile.currentModule = "unknown";
  cpuProfile.planAnalyses.length = 0;
  cpuProfile.fusionFallbackStats.clear();
}

// ---------------------------------------------------------------------------
// GPU timestamp profiling state
// ---------------------------------------------------------------------------

const GPU_MAX_PASSES = 2048; // 4096 timestamp slots (Dawn limit is 4096 queries)

interface GpuTimestampState {
  querySet: GPUQuerySet | null;
  resolveBuffer: GPUBuffer | null;
  readbackBuffer: GPUBuffer | null;
  passRecords: GpuPassRecord[];
  nextSlot: number;
  stagingBuffer: GPUBuffer | null;
  stagingSlots: number;
  maxSlots: number;
  supported: boolean;
  enabled: boolean;
  device: GPUDevice | null;
  labelStats: Map<string, { count: number; totalNs: bigint; maxNs: bigint }>;
  phaseStats: Map<string, { totalNs: bigint; opCount: number }>;
  phaseOpStats: Map<string, Map<string, { count: number; totalNs: bigint; maxNs: bigint }>>;
  moduleStats: Map<string, { totalNs: bigint; opCount: number }>;
  moduleOpStats: Map<string, Map<string, { count: number; totalNs: bigint; maxNs: bigint }>>;
}

const gpuTs: GpuTimestampState = {
  querySet: null,
  resolveBuffer: null,
  readbackBuffer: null,
  passRecords: [],
  nextSlot: 0,
  stagingBuffer: null,
  stagingSlots: 0,
  maxSlots: GPU_MAX_PASSES * 2,
  supported: false,
  enabled: true, // Can be disabled per-step to avoid V100/Dawn deadlock
  device: null,
  labelStats: new Map(),
  phaseStats: new Map(),
  phaseOpStats: new Map(),
  moduleStats: new Map(),
  moduleOpStats: new Map(),
};

/** Reset GPU timestamp profiling state (not the device/querySet). */
function resetGpuTimestampState(): void {
  gpuTs.passRecords = [];
  gpuTs.nextSlot = 0;
  if (gpuTs.stagingBuffer) {
    gpuTs.stagingBuffer.destroy();
    gpuTs.stagingBuffer = null;
  }
  gpuTs.stagingSlots = 0;
  gpuTs.labelStats.clear();
  gpuTs.phaseStats.clear();
  gpuTs.phaseOpStats.clear();
  gpuTs.moduleStats.clear();
  gpuTs.moduleOpStats.clear();
}

export function recordFusionFallback(reason: string, groupSize: number, detail?: unknown): void {
  if (!PROFILING_ENABLED) return;
  const entry = cpuProfile.fusionFallbackStats.get(reason);
  if (entry) {
    entry.count++;
    entry.totalNodes += groupSize;
    if (detail && entry.details.length < 3) entry.details.push(detail);
  } else {
    cpuProfile.fusionFallbackStats.set(reason, {
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

/**
 * Enable/disable GPU timestamp writes for subsequent compute passes.
 * When disabled, getTimestampWrites() returns undefined and resolveGpuTimestamps()
 * is a no-op. Use this to limit timestamp queries to a single step, avoiding
 * a V100/Dawn bug where accumulated resolveQuerySet operations corrupt Vulkan
 * fence state and cause mapAsync deadlocks.
 */
export function setTimestampsEnabled(enabled: boolean): void {
  gpuTs.enabled = enabled;
}

// ---------------------------------------------------------------------------
// CPU API call wrapping
// ---------------------------------------------------------------------------

export function profileApiCall<T>(name: string, fn: () => T): T {
  if (!PROFILING_ENABLED) return fn();
  const t0 = performance.now();
  const result = fn();
  const elapsed = performance.now() - t0;
  const entry = cpuProfile.apiStats.get(name);
  if (entry) {
    entry.count++;
    entry.totalMs += elapsed;
    if (elapsed > entry.maxMs) entry.maxMs = elapsed;
  } else {
    cpuProfile.apiStats.set(name, { count: 1, totalMs: elapsed, maxMs: elapsed });
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
  const entry = cpuProfile.opStats.get(opName);
  if (entry) {
    entry.count++;
    entry.totalMs += elapsed;
    if (elapsed > entry.maxMs) entry.maxMs = elapsed;
  } else {
    cpuProfile.opStats.set(opName, { count: 1, totalMs: elapsed, maxMs: elapsed });
  }

  // Phase stats
  const phase = cpuProfile.phaseStats.get(cpuProfile.currentPhase);
  if (phase) {
    phase.totalMs += elapsed;
    phase.opCount++;
  } else {
    cpuProfile.phaseStats.set(cpuProfile.currentPhase, { totalMs: elapsed, opCount: 1 });
  }
}

// ---------------------------------------------------------------------------
// Sub-op profiling (fine-grained breakdown within a single op dispatch)
// ---------------------------------------------------------------------------

export function profileSubOpBegin(): number {
  if (!PROFILING_ENABLED) return 0;
  return performance.now();
}

export function profileSubOpEnd(label: string, t0: number): void {
  if (!PROFILING_ENABLED) return;
  const elapsed = performance.now() - t0;
  const entry = cpuProfile.subOpStats.get(label);
  if (entry) {
    entry.count++;
    entry.totalMs += elapsed;
    if (elapsed > entry.maxMs) entry.maxMs = elapsed;
  } else {
    cpuProfile.subOpStats.set(label, { count: 1, totalMs: elapsed, maxMs: elapsed });
  }
}

// ---------------------------------------------------------------------------
// Phase control
// ---------------------------------------------------------------------------

export function setProfilePhase(phase: string): void {
  cpuProfile.currentPhase = phase;
}

export function setProfileModule(module: string): void {
  cpuProfile.currentModule = module;
}

export function getProfileModule(): string {
  return cpuProfile.currentModule;
}

// ---------------------------------------------------------------------------
// Plan analysis recording
// ---------------------------------------------------------------------------

export function recordPlanAnalysis(analysis: PlanAnalysis): void {
  if (!PROFILING_ENABLED) return;
  analysis.planIndex = cpuProfile.planAnalyses.length;
  cpuProfile.planAnalyses.push(analysis);
}

// ---------------------------------------------------------------------------
// GPU timestamp initialization
// ---------------------------------------------------------------------------

export function initGpuTimestamps(device: GPUDevice): void {
  if (!PROFILING_ENABLED) return;
  gpuTs.device = device;
  gpuTs.supported = true;

  // Check device limits for max query count
  const maxQueryCount = (device.limits as any)?.maxQueryCount ?? 4096;
  const count = Math.min(GPU_MAX_PASSES * 2, maxQueryCount);
  gpuTs.maxSlots = count;

  // Use numeric constants for buffer usage to avoid dependency on globals
  // that may not be available in all environments
  const QUERY_RESOLVE = 0x0200;
  const COPY_SRC = 0x0004;

  gpuTs.querySet = device.createQuerySet({
    type: "timestamp" as GPUQueryType,
    count,
  });

  // Resolve buffer: 8 bytes per timestamp slot
  gpuTs.resolveBuffer = device.createBuffer({
    size: count * 8,
    usage: QUERY_RESOLVE | COPY_SRC,
  });

  gpuTs.readbackBuffer = null;

  console.log(`[profiler] GPU timestamps initialized: ${count} slots`);

  gpuTs.passRecords = [];
  gpuTs.nextSlot = 0;
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
  if (!PROFILING_ENABLED || !gpuTs.supported || !gpuTs.querySet || !gpuTs.enabled) return undefined;
  if (gpuTs.nextSlot + 2 > gpuTs.maxSlots) return undefined; // out of slots

  const startSlot = gpuTs.nextSlot;
  const endSlot = gpuTs.nextSlot + 1;
  gpuTs.nextSlot += 2;

  gpuTs.passRecords.push({ label, phase: cpuProfile.currentPhase, module: cpuProfile.currentModule, startSlot, endSlot });

  return {
    querySet: gpuTs.querySet,
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
    !gpuTs.supported ||
    !gpuTs.querySet ||
    !gpuTs.resolveBuffer ||
    gpuTs.nextSlot === 0
  )
    return;

  // Don't resolve on the training encoder — defer to readGpuTimestamps()
  // so that the resolveQuerySet + copyBufferToBuffer + mapAsync all happen
  // in a single isolated submission after all training GPU work is complete.
  gpuTs.stagingSlots = gpuTs.nextSlot;
}

// ---------------------------------------------------------------------------
// Read back GPU timestamps (async — call after queue.submit + onSubmittedWorkDone)
// ---------------------------------------------------------------------------

/**
 * Process timestamp records from a BigInt64Array of raw GPU timestamps.
 * Accumulates per-label, per-phase, per-module stats.
 */
function processTimestampRecords(timestamps: BigInt64Array): void {
  for (const record of gpuTs.passRecords) {
    if (record.startSlot >= timestamps.length || record.endSlot >= timestamps.length) continue;
    const startNs = timestamps[record.startSlot];
    const endNs = timestamps[record.endSlot];
    if (startNs === 0n && endNs === 0n) continue; // no data

    const durationNs = endNs - startNs;
    if (durationNs < 0n) continue; // invalid

    // Per-label stats
    const entry = gpuTs.labelStats.get(record.label);
    if (entry) {
      entry.count++;
      entry.totalNs += durationNs;
      if (durationNs > entry.maxNs) entry.maxNs = durationNs;
    } else {
      gpuTs.labelStats.set(record.label, {
        count: 1,
        totalNs: durationNs,
        maxNs: durationNs,
      });
    }

    // Per-phase GPU stats
    const phase = gpuTs.phaseStats.get(record.phase);
    if (phase) {
      phase.totalNs += durationNs;
      phase.opCount++;
    } else {
      gpuTs.phaseStats.set(record.phase, { totalNs: durationNs, opCount: 1 });
    }

    // Per-phase per-op GPU stats
    let phaseOps = gpuTs.phaseOpStats.get(record.phase);
    if (!phaseOps) {
      phaseOps = new Map();
      gpuTs.phaseOpStats.set(record.phase, phaseOps);
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
    const mod = gpuTs.moduleStats.get(record.module);
    if (mod) {
      mod.totalNs += durationNs;
      mod.opCount++;
    } else {
      gpuTs.moduleStats.set(record.module, { totalNs: durationNs, opCount: 1 });
    }

    // Per-module per-op GPU stats
    let modOps = gpuTs.moduleOpStats.get(record.module);
    if (!modOps) {
      modOps = new Map();
      gpuTs.moduleOpStats.set(record.module, modOps);
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
}

/**
 * Flush and read GPU timestamps mid-step. Call AFTER the forward pass but
 * BEFORE the backward pass.
 *
 * V100/Dawn workaround: the timestamp-query device feature corrupts Dawn's
 * Vulkan fence mechanism after backward pass dispatches. onSubmittedWorkDone
 * and copyBufferToBuffer+mapAsync both deadlock permanently after backward.
 * However, they work correctly after the forward pass. This function reads
 * all forward-phase GPU timestamps while the fence mechanism still works,
 * then disables timestamp writes for the remainder of the step.
 *
 * Returns true if timestamps were successfully read, false otherwise.
 */
export async function flushAndReadGpuTimestamps(): Promise<boolean> {
  if (
    !PROFILING_ENABLED ||
    !gpuTs.supported ||
    !gpuTs.device ||
    !gpuTs.querySet ||
    !gpuTs.resolveBuffer ||
    gpuTs.passRecords.length === 0 ||
    gpuTs.nextSlot === 0
  )
    return false;

  // 1. Flush the shared encoder to submit all forward work
  const { flushSharedEncoder } = await import("./index");
  flushSharedEncoder();

  const slotsToRead = gpuTs.nextSlot;
  const byteSize = slotsToRead * 8;
  const MAP_READ = 0x0001;
  const COPY_DST = 0x0008;

  // 2. Resolve + copy timestamps in a separate submission
  const staging = gpuTs.device.createBuffer({
    size: byteSize,
    usage: MAP_READ | COPY_DST,
  });

  const encoder = gpuTs.device.createCommandEncoder();
  encoder.resolveQuerySet(gpuTs.querySet, 0, slotsToRead, gpuTs.resolveBuffer, 0);
  encoder.copyBufferToBuffer(gpuTs.resolveBuffer, 0, staging, 0, byteSize);
  gpuTs.device.queue.submit([encoder.finish()]);

  // 3. Fence using onSubmittedWorkDone (works after forward pass)
  if (typeof gpuTs.device.queue.onSubmittedWorkDone === "function") {
    const FENCE_TIMEOUT_MS = 10_000;
    const fenceOk = await Promise.race([
      gpuTs.device.queue.onSubmittedWorkDone().then(() => true),
      new Promise<false>((resolve) => setTimeout(() => resolve(false), FENCE_TIMEOUT_MS)),
    ]);
    if (!fenceOk) {
      console.warn("[profiler] onSubmittedWorkDone timed out after forward — skipping GPU timestamps");
      staging.destroy();
      gpuTs.enabled = false;
      return false;
    }
  }

  // 4. Read timestamp data
  const MAPASYNC_TIMEOUT_MS = 5_000;
  let mapOk = false;
  try {
    const result = await Promise.race([
      staging.mapAsync(MAP_READ).then(() => true),
      new Promise<false>((resolve) => setTimeout(() => resolve(false), MAPASYNC_TIMEOUT_MS)),
    ]);
    mapOk = result;
  } catch (e) {
    console.warn("[profiler] Failed to map timestamp staging buffer:", e);
    staging.destroy();
    gpuTs.enabled = false;
    return false;
  }

  if (!mapOk) {
    console.warn("[profiler] mapAsync timed out after forward — skipping GPU timestamps");
    staging.destroy();
    gpuTs.enabled = false;
    return false;
  }

  // 5. Process timestamp records
  const timestamps = new BigInt64Array(staging.getMappedRange());
  processTimestampRecords(timestamps);
  staging.unmap();
  staging.destroy();

  // 6. Disable timestamp writes for backward/optimizer to avoid corrupting
  //    Dawn's fence state. Forward GPU timing is captured; backward uses CPU timing.
  gpuTs.enabled = false;
  return true;
}

/**
 * Read GPU timestamps at end of step. With the mid-step readback
 * (flushAndReadGpuTimestamps), this is typically a no-op since forward
 * timestamps are already read and backward timestamps are skipped.
 */
export async function readGpuTimestamps(): Promise<void> {
  if (
    !PROFILING_ENABLED ||
    !gpuTs.supported ||
    !gpuTs.device ||
    !gpuTs.querySet ||
    !gpuTs.resolveBuffer ||
    gpuTs.passRecords.length === 0 ||
    gpuTs.stagingSlots === 0
  )
    return;

  // If timestamps were already processed by flushAndReadGpuTimestamps,
  // labelStats will be non-empty. Skip redundant readback.
  if (gpuTs.labelStats.size > 0) return;

  const MAP_READ = 0x0001;
  const COPY_DST = 0x0008;
  const slotsToRead = gpuTs.stagingSlots;
  const byteSize = slotsToRead * 8;

  // Drain the deferred fence from markStep()
  const { awaitDeferredFence } = await import("./index");
  await awaitDeferredFence();

  // Resolve + copy timestamps
  const staging = gpuTs.device.createBuffer({
    size: byteSize,
    usage: MAP_READ | COPY_DST,
  });

  const encoder = gpuTs.device.createCommandEncoder();
  encoder.resolveQuerySet(gpuTs.querySet, 0, slotsToRead, gpuTs.resolveBuffer, 0);
  encoder.copyBufferToBuffer(gpuTs.resolveBuffer, 0, staging, 0, byteSize);
  gpuTs.device.queue.submit([encoder.finish()]);

  const MAPASYNC_TIMEOUT_MS = 5_000;
  let mapOk = false;
  try {
    const result = await Promise.race([
      staging.mapAsync(MAP_READ).then(() => true),
      new Promise<false>((resolve) => setTimeout(() => resolve(false), MAPASYNC_TIMEOUT_MS)),
    ]);
    mapOk = result;
  } catch (e) {
    console.warn("[profiler] Failed to map staging buffer:", e);
    staging.destroy();
    return;
  }

  if (!mapOk) {
    console.warn("[profiler] mapAsync timed out (5s) — GPU timestamps unavailable (V100/Dawn timestamp-query bug)");
    staging.destroy();
    return;
  }

  const timestamps = new BigInt64Array(staging.getMappedRange());
  processTimestampRecords(timestamps);
  staging.unmap();
  staging.destroy();
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
  if (cpuProfile.apiStats.size > 0) {
    const sorted = [...cpuProfile.apiStats.entries()].sort(
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
  if (cpuProfile.opStats.size > 0) {
    const sorted = [...cpuProfile.opStats.entries()].sort(
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
  if (cpuProfile.subOpStats.size > 0) {
    const sorted = [...cpuProfile.subOpStats.entries()].sort(
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
  if (gpuTs.labelStats.size > 0) {
    const sorted = [...gpuTs.labelStats.entries()].sort(
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
  if (cpuProfile.phaseStats.size > 0 || gpuTs.phaseStats.size > 0) {
    const allPhases = new Set([
      ...cpuProfile.phaseStats.keys(),
      ...gpuTs.phaseStats.keys(),
    ]);
    console.log(
      `${padR("Phase", 16)} ${padL("Ops", 8)} ${padL("CPU(ms)", 10)} ${padL("GPU(ms)", 10)}`,
    );
    console.log("─".repeat(46));
    for (const phase of allPhases) {
      const cpu = cpuProfile.phaseStats.get(phase);
      const gpu = gpuTs.phaseStats.get(phase);
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
  if (gpuTs.phaseOpStats.size > 0) {
    console.log("=== Per-Phase GPU Breakdown ===\n");
    for (const [phase, opsMap] of gpuTs.phaseOpStats) {
      const gpuPhase = gpuTs.phaseStats.get(phase);
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
  if (gpuTs.moduleStats.size > 0) {
    const totalGpuNs = [...gpuTs.moduleStats.values()].reduce((s, v) => s + v.totalNs, 0n);
    const sortedModules = [...gpuTs.moduleStats.entries()].sort(
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
      const opsMap = gpuTs.moduleOpStats.get(mod);
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
  if (cpuProfile.fusionFallbackStats.size > 0) {
    const sorted = [...cpuProfile.fusionFallbackStats.entries()].sort(
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
  if (cpuProfile.planAnalyses.length > 0) {
    console.log("=== Plan Analysis ===\n");
    for (const pa of cpuProfile.planAnalyses) {
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
    void writeProfileJSON(jsonPath);
  }
}

// ---------------------------------------------------------------------------
// JSON export
// ---------------------------------------------------------------------------

export function getProfileJSON(): object {
  // Build phases object with per-op kernel breakdown
  const phases: Record<string, any> = {};
  const allPhases = new Set([
    ...cpuProfile.phaseStats.keys(),
    ...gpuTs.phaseStats.keys(),
    ...gpuTs.phaseOpStats.keys(),
  ]);
  for (const phase of allPhases) {
    const cpu = cpuProfile.phaseStats.get(phase);
    const gpu = gpuTs.phaseStats.get(phase);
    const kernels: Record<string, any> = {};
    const phaseOps = gpuTs.phaseOpStats.get(phase);
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
  for (const [op, s] of gpuTs.labelStats) {
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
  for (const [, s] of gpuTs.phaseStats) {
    totalGpuMs += nsToMs(s.totalNs);
    totalDispatches += s.opCount;
  }

  // Fusion rate from plan analyses
  let totalNodes = 0;
  let totalFused = 0;
  for (const pa of cpuProfile.planAnalyses) {
    totalNodes += pa.totalNodes;
    totalFused += pa.fusedNodes;
  }
  const fusionRate = totalNodes > 0 ? totalFused / totalNodes : 0;

  // Top bottlenecks: (phase, op) pairs sorted by GPU time
  const bottlenecks: { phase: string; op: string; gpu_ms: number }[] = [];
  for (const [phase, opsMap] of gpuTs.phaseOpStats) {
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
  for (const [mod, s] of gpuTs.moduleStats) {
    const kernels: Record<string, any> = {};
    const modOps = gpuTs.moduleOpStats.get(mod);
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
    plans: cpuProfile.planAnalyses,
    gpu_totals,
    summary: {
      total_gpu_ms: totalGpuMs,
      total_dispatches: totalDispatches,
      fusion_rate: fusionRate,
      top_bottlenecks: topBottlenecks,
    },
  };
}

export async function writeProfileJSON(filePath: string): Promise<void> {
  if (!PROFILING_ENABLED) return;
  try {
    const fs = await import("node:fs");
    const json = getProfileJSON();
    fs.writeFileSync(filePath, JSON.stringify(json, null, 2));
    console.log(`Profile JSON written to ${filePath}`);
  } catch (e) {
    console.warn(`[profiler] Failed to write profile JSON to ${filePath}:`, e);
  }
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

export function resetProfileStats(): void {
  if (!PROFILING_ENABLED) return;
  resetCpuProfileState();
  resetGpuTimestampState();
}
