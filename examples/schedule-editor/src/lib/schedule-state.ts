export type MemorySpace = "global" | "workgroup-shared" | "register";
export type SyncScope = "workgroup";
export type ThreadLevel = "invocation" | "subgroup" | "workgroup";
export type Residency = MemorySpace;

export interface MatmulWorkload {
  kind: "matmul";
  m: number;
  n: number;
  k: number;
  batch: number;
  inputBytes: number;
  outputBytes: number;
}

export interface AttentionWorkload {
  kind: "attention-forward";
  batch: number;
  heads: number;
  sequence: number;
  headDimension: number;
  inputBytes: number;
  outputBytes: number;
}

export type Workload = MatmulWorkload | AttentionWorkload;

export interface LoopNode {
  id: string;
  axis: string;
  extent: string;
  execution: "parallel" | "sequential";
  children?: LoopNode[];
}

export interface StagingEdge {
  operand: string;
  from: MemorySpace;
  to: MemorySpace;
  scope: string;
  synchronization: SyncScope | "none";
}

export interface RoleAssignment {
  role: string;
  participants: string;
  responsibility: string;
}

export type Skeleton =
  | {
      visibility: "derived";
      grammar:
        | "tiled-matmul"
        | "elementwise"
        | "row-reduction"
        | "attention-shaped";
      loopNest: LoopNode[];
      stagingEdges: StagingEdge[];
      rolePartition: RoleAssignment[];
    }
  | {
      visibility: "opaque";
      kernelRef: string;
      reason: string;
    };

export interface Decorations {
  tileSizes: Record<string, number>;
  workgroup: { x: number; y: number; z: number };
  vectorWidth: 1 | 2 | 4;
  unrollFactor: number;
  pipelineDepth: 1;
  memorySpaces: MemorySpace[];
  syncScopes: SyncScope[];
  threadHierarchy: ThreadLevel[];
  operandResidency: Record<string, Residency>;
}

export interface ScheduleAtom {
  id: string;
  kind: "atomicAddF32-CAS" | "subgroup-op";
  semanticRef: string;
  footprintBytes: number;
  syncBehavior: string;
}

export interface AdmittedLemma {
  id: string;
  version: number;
  role: string;
}

export interface ScheduleState {
  schemaVersion: "1-proposal";
  id: string;
  name: string;
  authored: boolean;
  algorithm: {
    id: string;
    semanticRef: string;
    workload: Workload;
  };
  skeleton: Skeleton;
  decorations: Decorations;
  atoms: ScheduleAtom[];
  admittedLemmas: AdmittedLemma[];
  realizerId: "tile-ir-wgsl";
}

export interface DeviceModel {
  source: "adapter+fallback" | "fallback";
  maxComputeWorkgroupStorageSize: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxResidentInvocationsPerComputeUnit: number;
  maxResidentWorkgroupsPerComputeUnit: number;
  sharedMemoryPerComputeUnit: number;
  subgroupsSupported: boolean;
  peakBandwidthGBs: number;
  peakFlopsTFLOPs: number;
}

export interface StaticCost {
  sharedMemoryBytes: number;
  sharedMemoryUtilization: number;
  workgroupThreads: number;
  residentWorkgroupsProxy: number;
  occupancyProxy: number;
  occupancyLimiter: "threads" | "shared-memory" | "workgroup-slots";
  bytesMoved: number;
  flops: number;
  arithmeticIntensity: number;
  ridgePoint: number;
  rooflineBound: "bandwidth" | "compute";
  attainableTFLOPs: number;
  predictedMs: number;
}

export interface ScheduleHistoryPoint {
  id: number;
  label: string;
  state: ScheduleState;
  stateHash: string;
  cost: StaticCost;
}

export const FALLBACK_DEVICE: DeviceModel = {
  source: "fallback",
  maxComputeWorkgroupStorageSize: 16_384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxResidentInvocationsPerComputeUnit: 2_048,
  maxResidentWorkgroupsPerComputeUnit: 8,
  sharedMemoryPerComputeUnit: 65_536,
  subgroupsSupported: false,
  peakBandwidthGBs: 500,
  peakFlopsTFLOPs: 10,
};

export function cloneScheduleState(state: ScheduleState): ScheduleState {
  // Svelte wraps loaded state in deep proxies; structuredClone rejects proxies.
  // ScheduleState is deliberately JSON-only, so a JSON round-trip is exact.
  return JSON.parse(JSON.stringify(state)) as ScheduleState;
}

export async function readDeviceModel(): Promise<DeviceModel> {
  const gpu = (
    navigator as Navigator & {
      gpu?: {
        requestAdapter(): Promise<{
          limits?: {
            maxComputeWorkgroupStorageSize?: number;
            maxComputeInvocationsPerWorkgroup?: number;
          };
          features?: { has(name: string): boolean };
        } | null>;
      };
    }
  ).gpu;
  if (!gpu) return { ...FALLBACK_DEVICE };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { ...FALLBACK_DEVICE };
    return {
      ...FALLBACK_DEVICE,
      source: "adapter+fallback",
      maxComputeWorkgroupStorageSize:
        adapter.limits?.maxComputeWorkgroupStorageSize ??
        FALLBACK_DEVICE.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup:
        adapter.limits?.maxComputeInvocationsPerWorkgroup ??
        FALLBACK_DEVICE.maxComputeInvocationsPerWorkgroup,
      subgroupsSupported: adapter.features?.has("subgroups") ?? false,
    };
  } catch {
    return { ...FALLBACK_DEVICE };
  }
}
