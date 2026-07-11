export type NcdLevel = "l0" | "l1";
export type PartitionKind = "group" | "stream";

export interface NcdAxis {
  id: string;
  label: string;
  size: number;
}

export interface NcdWire {
  id: string;
  label: string;
  axisIds: string[];
  elementBytes: number;
  tupleGroup?: string;
}

export interface StreamDecomposition {
  axisId: string;
  head: string;
  body: string;
  accumulatorWireIds: string[];
}

export type Streamability =
  | { kind: "decomposed"; axes: StreamDecomposition[] }
  | { kind: "none"; reason: string };

export interface NcdBox {
  id: string;
  label: string;
  kind: string;
  column: number;
  inputWireIds: string[];
  outputWireIds: string[];
  streamability: Streamability;
}

export interface NcdColumn {
  id: string;
  index: number;
  label: string;
}

export interface TupleGroup {
  id: string;
  label: string;
  wireIds: string[];
}

export interface ResidencyDecoration {
  wireId: string;
  column: number;
  level: NcdLevel;
}

export interface PartitionDecoration {
  axisId: string;
  kind: PartitionKind;
  size: number;
  label: string;
}

export interface DivisibilityDecoration {
  axisId: string;
  multiple: number;
  reason: string;
}

export interface NcdTerm {
  schemaVersion: "ncd-term-1-proposal";
  id: string;
  name: string;
  semantic: {
    axes: NcdAxis[];
    wires: NcdWire[];
    boxes: NcdBox[];
    columns: NcdColumn[];
    tupleGroups: TupleGroup[];
  };
  decorations: {
    residency: ResidencyDecoration[];
    partitions: PartitionDecoration[];
    divisibility: DivisibilityDecoration[];
    admittedLemmas: string[];
  };
}

export interface NcdDiagramModel {
  schemaVersion: "ncd-diagram-1";
  meta: {
    id: string;
    name: string;
    termSchemaVersion: NcdTerm["schemaVersion"];
  };
  axes: NcdAxis[];
  bundles: NcdWire[];
  functions: NcdBox[];
  columns: NcdColumn[];
  tupleGroups: TupleGroup[];
  labels: NcdTerm["decorations"];
}

export interface NapkinCost {
  transferByLevel: Record<NcdLevel, number>;
  memoryByLevel: Record<NcdLevel, number>;
  transferBytesByLevel: Record<NcdLevel, number>;
  memoryBytesByLevel: Record<NcdLevel, number>;
}

export interface ProjectionResult {
  ok: boolean;
  lines: string[];
  reason?: string;
}

export type PartitionRelabeling = {
  op: "partition";
  axisId: string;
  before?: PartitionDecoration;
  after?: PartitionDecoration;
};

export type RecolorRelabeling = {
  op: "recolor";
  wireId: string;
  column: number;
  before: NcdLevel;
  after: NcdLevel;
};

export type LemmaRelabeling = {
  op: "lemma";
  boxId: string;
  lemmaId: string;
  before: Pick<NcdBox, "label" | "kind" | "streamability">;
  after: Pick<NcdBox, "label" | "kind" | "streamability">;
  add: boolean;
};

export type NcdMove = PartitionRelabeling | RecolorRelabeling | LemmaRelabeling;

export interface NcdHistoryEntry {
  label: string;
  forward: NcdMove;
  inverse: NcdMove;
  termHash: string;
  cost: NapkinCost;
}
