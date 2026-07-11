import type {
  HistoryEntry,
  Island,
  IslandKind,
  MergeMove,
  Partition,
  PartitionMove,
  PlanNode,
  SplitMove,
} from "./types";

const KIND_CODE: Record<IslandKind, number> = {
  sequential: 0,
  fused: 1,
  reduction: 2,
};

const FUSIBLE_OPS = new Set(
  "abs add cast ceil cos div eq exp floor ge gelu gelu_erf gelu_tanh gt isfinite le log lt maximum minimum mod mul ne neg pow relu round rsqrt sigmoid sign silu sin softplus sqrt sub tanh where".split(
    " ",
  ),
);

function sameMembers(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((member, index) => member === b[index]);
}

export function boundaryHash(islands: readonly Island[]): number {
  let h = 0x811c9dc5;
  const mix = (value: number) => {
    h ^= value & 0xff;
    h = Math.imul(h, 0x01000193);
    h ^= (value >>> 8) & 0xff;
    h = Math.imul(h, 0x01000193);
    h ^= (value >>> 16) & 0xff;
    h = Math.imul(h, 0x01000193);
    h ^= (value >>> 24) & 0xff;
    h = Math.imul(h, 0x01000193);
  };
  mix(islands.length);
  for (const island of islands) {
    mix(KIND_CODE[island.kind]);
    mix(island.members.length);
    for (const member of island.members) mix(member);
  }
  return h >>> 0;
}

export function withHash(islands: Island[]): Partition {
  return { islands, boundaryHash: boundaryHash(islands) };
}

export function clonePartition(partition: Partition): Partition {
  return withHash(
    partition.islands.map((island) => ({
      kind: island.kind,
      members: [...island.members],
    })),
  );
}

export function mergeLegality(
  partition: Partition,
  selected: readonly number[],
  nodes: ReadonlyMap<number, PlanNode>,
): { legal: boolean; reason: string; indices?: [number, number] } {
  if (selected.length !== 2) {
    return { legal: false, reason: "Select exactly two islands." };
  }
  const [left, right] = [...selected].sort((a, b) => a - b);
  if (right !== left + 1) {
    return { legal: false, reason: "Merge requires adjacent emission-order islands." };
  }
  const pair = [partition.islands[left], partition.islands[right]];
  if (pair.some((island) => island.kind === "reduction")) {
    return { legal: false, reason: "Reduction islands are treated as opaque atoms." };
  }
  const members = pair.flatMap((island) => island.members);
  const memberNodes = members.map((member) => nodes.get(member));
  if (memberNodes.some((node) => !node)) {
    return { legal: false, reason: "Node metadata is incomplete." };
  }
  if (
    pair.some(
      (island) =>
        island.kind === "sequential" &&
        island.members.some((member) => !FUSIBLE_OPS.has(nodes.get(member)?.op ?? "")),
    )
  ) {
    return {
      legal: false,
      reason: "A sequential island contains an op the production registry does not mark fusible.",
    };
  }
  const shape = JSON.stringify(memberNodes[0]?.shape);
  if (memberNodes.some((node) => JSON.stringify(node?.shape) !== shape)) {
    return { legal: false, reason: "Elementwise members must have identical output shapes." };
  }
  return {
    legal: true,
    reason: "Adjacent, elementwise-capable, and shape-compatible.",
    indices: [left, right],
  };
}

export function makeMerge(
  partition: Partition,
  leftIndex: number,
  rightIndex: number,
  resultKind: IslandKind = "fused",
): HistoryEntry {
  const left = partition.islands[leftIndex];
  const right = partition.islands[rightIndex];
  const mergedMembers = [...left.members, ...right.members];
  const forward: MergeMove = {
    op: "merge",
    leftMembers: [...left.members],
    rightMembers: [...right.members],
    leftKind: left.kind,
    rightKind: right.kind,
    resultKind,
  };
  const inverse: SplitMove = {
    op: "split",
    islandMembers: mergedMembers,
    afterMember: left.members.at(-1)!,
    beforeMember: right.members[0],
    leftKind: left.kind,
    rightKind: right.kind,
  };
  return { forward, inverse };
}

export function makeSplit(
  partition: Partition,
  islandIndex: number,
  cutIndex: number,
): HistoryEntry {
  const island = partition.islands[islandIndex];
  if (!island || cutIndex < 1 || cutIndex >= island.members.length) {
    throw new Error("Split cut must be an interior member boundary");
  }
  const leftMembers = island.members.slice(0, cutIndex);
  const rightMembers = island.members.slice(cutIndex);
  const forward: SplitMove = {
    op: "split",
    islandMembers: [...island.members],
    afterMember: leftMembers.at(-1)!,
    beforeMember: rightMembers[0],
    leftKind: island.kind,
    rightKind: island.kind,
  };
  const inverse: MergeMove = {
    op: "merge",
    leftMembers,
    rightMembers,
    leftKind: island.kind,
    rightKind: island.kind,
    resultKind: island.kind,
  };
  return { forward, inverse };
}

export function applyMove(partition: Partition, move: PartitionMove): Partition {
  const islands = partition.islands.map((island) => ({
    kind: island.kind,
    members: [...island.members],
  }));
  if (move.op === "merge") {
    const leftIndex = islands.findIndex((island) => sameMembers(island.members, move.leftMembers));
    const right = islands[leftIndex + 1];
    if (
      leftIndex < 0 ||
      !right ||
      !sameMembers(right.members, move.rightMembers) ||
      islands[leftIndex].kind !== move.leftKind ||
      right.kind !== move.rightKind
    ) {
      throw new Error("Merge precondition no longer matches the current partition");
    }
    islands.splice(leftIndex, 2, {
      kind: move.resultKind,
      members: [...move.leftMembers, ...move.rightMembers],
    });
  } else {
    const index = islands.findIndex((island) => sameMembers(island.members, move.islandMembers));
    const cut = move.islandMembers.findIndex(
      (member, memberIndex) =>
        member === move.afterMember && move.islandMembers[memberIndex + 1] === move.beforeMember,
    );
    if (index < 0 || cut < 0) {
      throw new Error("Split precondition no longer matches the current partition");
    }
    islands.splice(
      index,
      1,
      { kind: move.leftKind, members: move.islandMembers.slice(0, cut + 1) },
      { kind: move.rightKind, members: move.islandMembers.slice(cut + 1) },
    );
  }
  return withHash(islands);
}

export function boundaryKeys(partition: Partition): Set<string> {
  const keys = new Set<string>();
  for (let index = 0; index < partition.islands.length - 1; index += 1) {
    const left = partition.islands[index].members.at(-1);
    const right = partition.islands[index + 1].members[0];
    keys.add(`${left}|${right}`);
  }
  return keys;
}

export function hexHash(hash: number): string {
  return `0x${hash.toString(16).padStart(8, "0")}`;
}
