export type IslandKind = "sequential" | "fused" | "reduction";

export interface Island {
  kind: IslandKind;
  members: number[];
}

export interface Partition {
  boundaryHash: number;
  islands: Island[];
}

export interface PlanNode {
  pos: number;
  op: string;
  shape: number[];
  dtype: string;
  label?: string;
}

export interface ScheduleDump {
  meta: { model: string; step: string; planFingerprint: string };
  partition: Partition;
  nodes: PlanNode[];
}

export interface MergeMove {
  op: "merge";
  leftMembers: number[];
  rightMembers: number[];
  leftKind: IslandKind;
  rightKind: IslandKind;
  resultKind: IslandKind;
}

export interface SplitMove {
  op: "split";
  islandMembers: number[];
  afterMember: number;
  beforeMember: number;
  leftKind: IslandKind;
  rightKind: IslandKind;
}

export type PartitionMove = MergeMove | SplitMove;

export interface HistoryEntry {
  forward: PartitionMove;
  inverse: PartitionMove;
  staticCost?: {
    stateHash: string;
    predictedMs: number;
    arithmeticIntensity: number;
    rooflineBound: "bandwidth" | "compute";
  };
}
