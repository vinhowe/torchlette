import type { DeviceModel, ScheduleState, StaticCost } from "./schedule-state";

function tile(state: ScheduleState, key: string): number {
  const value = state.decorations.tileSizes[key];
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Decoration tileSizes.${key} must be positive`);
  }
  return value;
}

export function sharedMemoryBytes(state: ScheduleState): number {
  const depth = state.decorations.pipelineDepth;
  if (state.algorithm.workload.kind === "matmul") {
    const tileM = tile(state, "m");
    const tileN = tile(state, "n");
    const tileK = tile(state, "k");
    // The real WGSL matmul widens cooperative shared tiles to f32.
    return (tileM * tileK + tileK * tileN) * 4 * depth;
  }
  const kvRows = tile(state, "kvRows");
  const headDimension = tile(state, "headDimension");
  // Forward attention reuses K's shared allocation for V (`reuseShared: K`).
  return kvRows * headDimension * 4 * depth;
}

export function workEstimate(state: ScheduleState): {
  bytesMoved: number;
  flops: number;
} {
  const workload = state.algorithm.workload;
  if (workload.kind === "matmul") {
    const tileM = tile(state, "m");
    const tileN = tile(state, "n");
    const aReads = Math.ceil(workload.n / tileN);
    const bReads = Math.ceil(workload.m / tileM);
    const bytesMoved =
      workload.batch *
      (aReads * workload.m * workload.k * workload.inputBytes +
        bReads * workload.k * workload.n * workload.inputBytes +
        workload.m * workload.n * workload.outputBytes);
    return {
      bytesMoved,
      flops: 2 * workload.batch * workload.m * workload.n * workload.k,
    };
  }

  const qBlocks = Math.ceil(workload.sequence / tile(state, "qRows"));
  const elements =
    workload.batch *
    workload.heads *
    workload.sequence *
    workload.headDimension;
  const matrixElements =
    workload.batch * workload.heads * workload.sequence * workload.sequence;
  const bytesMoved =
    elements * workload.inputBytes +
    2 * qBlocks * elements * workload.inputBytes +
    elements * workload.outputBytes +
    workload.batch * workload.heads * workload.sequence * 4;
  // QKᵀ + PV are 4*N²*D FLOPs; online softmax is approximated as 5*N².
  const flops =
    4 * matrixElements * workload.headDimension + 5 * matrixElements;
  return { bytesMoved, flops };
}

export function rooflineEstimate(
  flops: number,
  bytesMoved: number,
  device: Pick<DeviceModel, "peakBandwidthGBs" | "peakFlopsTFLOPs">,
): Pick<
  StaticCost,
  | "arithmeticIntensity"
  | "ridgePoint"
  | "rooflineBound"
  | "attainableTFLOPs"
  | "predictedMs"
> {
  const bandwidthBytesPerSecond = device.peakBandwidthGBs * 1e9;
  const peakFlopsPerSecond = device.peakFlopsTFLOPs * 1e12;
  const arithmeticIntensity = bytesMoved > 0 ? flops / bytesMoved : Infinity;
  const ridgePoint = peakFlopsPerSecond / bandwidthBytesPerSecond;
  const rooflineBound =
    arithmeticIntensity < ridgePoint ? "bandwidth" : "compute";
  const attainableFlopsPerSecond = Math.min(
    peakFlopsPerSecond,
    bandwidthBytesPerSecond * arithmeticIntensity,
  );
  return {
    arithmeticIntensity,
    ridgePoint,
    rooflineBound,
    attainableTFLOPs: attainableFlopsPerSecond / 1e12,
    predictedMs:
      Math.max(
        flops / peakFlopsPerSecond,
        bytesMoved / bandwidthBytesPerSecond,
      ) * 1e3,
  };
}

export function calculateStaticCost(
  state: ScheduleState,
  device: DeviceModel,
): StaticCost {
  const shared = sharedMemoryBytes(state);
  const threads =
    state.decorations.workgroup.x *
    state.decorations.workgroup.y *
    state.decorations.workgroup.z;
  const byThreads = Math.max(
    0,
    Math.floor(device.maxResidentInvocationsPerComputeUnit / threads),
  );
  const byShared =
    shared > 0
      ? Math.max(0, Math.floor(device.sharedMemoryPerComputeUnit / shared))
      : device.maxResidentWorkgroupsPerComputeUnit;
  const residentWorkgroupsProxy = Math.min(
    device.maxResidentWorkgroupsPerComputeUnit,
    byThreads,
    byShared,
  );
  const occupancyLimiter =
    byThreads <= byShared &&
    byThreads <= device.maxResidentWorkgroupsPerComputeUnit
      ? "threads"
      : byShared <= device.maxResidentWorkgroupsPerComputeUnit
        ? "shared-memory"
        : "workgroup-slots";
  const work = workEstimate(state);
  const roofline = rooflineEstimate(work.flops, work.bytesMoved, device);
  return {
    sharedMemoryBytes: shared,
    sharedMemoryUtilization: shared / device.maxComputeWorkgroupStorageSize,
    workgroupThreads: threads,
    residentWorkgroupsProxy,
    occupancyProxy: Math.min(
      1,
      (residentWorkgroupsProxy * threads) /
        device.maxResidentInvocationsPerComputeUnit,
    ),
    occupancyLimiter,
    bytesMoved: work.bytesMoved,
    flops: work.flops,
    ...roofline,
  };
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

export function scheduleStateHash(state: ScheduleState): string {
  let hash = 0x811c9dc5;
  // Mirrors §5's proposed coordinate: skeleton + decorations + realizer.
  // The canonical byte encoding itself remains a frontend proposal (F12).
  const coordinate = {
    skeleton: state.skeleton,
    decorations: state.decorations,
    realizerId: state.realizerId,
  };
  for (const byte of new TextEncoder().encode(canonical(coordinate))) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193);
  }
  return `fnv1a32:${(hash >>> 0).toString(16).padStart(8, "0")}`;
}
