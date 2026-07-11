# Requested partition contract

The editor exports one self-contained, optimistic-concurrency request. The
engine receives both a canonical full partition and an auditable derivation
from the detector base. It must validate that both forms produce the same
`boundaryHash`; neither representation is merely decorative.

```ts
type IslandKind = "sequential" | "fused" | "reduction";

interface Island {
  kind: IslandKind;
  members: number[]; // final-plan positions, in island emission order
}

interface Partition {
  boundaryHash: number; // unsigned 32-bit production FNV-1a
  islands: Island[];    // island emission order
}

interface MergeMove {
  op: "merge";
  leftMembers: number[];
  rightMembers: number[];
  leftKind: IslandKind;
  rightKind: IslandKind;
  resultKind: IslandKind;
}

interface SplitMove {
  op: "split";
  islandMembers: number[];
  afterMember: number;
  beforeMember: number;
  leftKind: IslandKind;
  rightKind: IslandKind;
}

interface RequestedPartitionMessage {
  schemaVersion: 1;
  type: "schedule.partition.request";
  planFingerprint: string;      // graph identity, e.g. 0x1234:0xabcd
  baseBoundaryHash: number;     // detector partition the moves expect
  requestedPartition: Partition;
  moves: Array<MergeMove | SplitMove>;
}
```

Member arrays, not transient UI indices, identify move operands. Each move also
carries its expected input kinds, making replay fail closed if the base or a
prior move diverges. A split cut is a pair of adjacent member positions rather
than an array offset; the engine must confirm that `afterMember` is immediately
before `beforeMember` in `islandMembers`. Result kinds are explicit because the
current three-kind representation cannot infer how a structural inverse should
restore pre-merge kinds.

## Engine processing

For `schedule.partition.request`, the engine should:

1. Resolve `planFingerprint` to the exact static semantic graph. Reject unknown
   or stale graphs.
2. Derive the detector partition and compare its unsigned `boundaryHash` with
   `baseBoundaryHash`. On mismatch, return a conflict with the fresh dump; do
   not attempt positional rebasing.
3. Replay `moves` in order with server-authoritative §2 checks: complete cover,
   uniqueness, emission ordering, dataflow convexity, checkpoint/WAR barriers,
   kind/atom capability, shape/broadcast compatibility, buffer counts, storage
   binding limits, chunking, and device class.
4. Canonicalize and hash the replay result with `partitionBoundaryHash`; require
   byte-for-byte equality with `requestedPartition`, including its hash.
5. Mix the requested `boundaryHash` into the template fingerprint, lower and
   compile under a distinct cache key, and return acceptance plus the canonical
   partition. Rejection should identify the failing move index and a stable
   reason code with human-readable detail.

Suggested response shape:

```ts
type Response =
  | {
      type: "schedule.partition.accepted";
      planFingerprint: string;
      partition: Partition;
      templateFingerprint: string;
    }
  | {
      type: "schedule.partition.rejected";
      code: "STALE_BASE" | "ILLEGAL_MOVE" | "FORM_MISMATCH" | "UNKNOWN_PLAN";
      moveIndex?: number;
      reason: string;
      current?: { planFingerprint: string; partition: Partition };
    };
```

The first live RPC can be request/response over the existing engine channel.
For genuinely live editing, the engine should additionally emit plan-change
events carrying a new dump and invalidating the editor's undo stack whenever
final-plan positions or the plan fingerprint change.
