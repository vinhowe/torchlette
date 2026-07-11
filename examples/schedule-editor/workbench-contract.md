# Schedule workbench benchmark contract

This extends the partition request channel with an island-in-isolation benchmark
RPC. The frontend currently declares and renders the protocol but does not send
it; the measured tier remains visibly “awaiting engine.”

```ts
interface ScheduleBenchRequest {
  schemaVersion: 1;
  type: "schedule.bench.request";
  requestId: string;
  planFingerprint: string;
  partitionBoundaryHash: number;
  islandMembers: number[];
  scheduleStateHash: string;
  scheduleState: ScheduleState;
  realizerId: "tile-ir-wgsl";
  workload: {
    warmupRuns: 3;
    sampleRuns: 7;
    statistic: "median";
    timing: "timestamp-query";
    isolation: "single-island";
  };
}

type ScheduleBenchResponse =
  | {
      schemaVersion: 1;
      type: "schedule.bench.result";
      requestId: string;
      scheduleStateHash: string;
      realizerId: "tile-ir-wgsl";
      medianMs: number;
      samplesMs: number[];
      warmupRuns: number;
      measuredAt: string;
      device: {
        adapterInfo?: Record<string, string>;
        limitsFingerprint: string;
      };
    }
  | {
      schemaVersion: 1;
      type: "schedule.bench.rejected";
      requestId: string;
      scheduleStateHash: string;
      code:
        | "STALE_PLAN"
        | "STALE_PARTITION"
        | "STATE_HASH_MISMATCH"
        | "ILLEGAL_STATE"
        | "REALIZER_UNSUPPORTED"
        | "TIMING_UNAVAILABLE";
      reason: string;
      fieldPath?: string;
    };
```

## Processing requirements

1. Resolve the plan and exact island using all three identity coordinates:
   plan fingerprint, partition boundary hash, and member positions.
2. Recompute the versioned canonical ScheduleState hash. The frontend's current
   sorted-JSON FNV hash is explicitly provisional; production must publish the
   canonical byte encoding and hash vector.
3. Run core legality and the selected realizer capability profile before any
   compilation. Return a stable refusal rather than silently clamping a field.
4. Compile under `(planFingerprint, boundaryHash, scheduleStateHash,
   realizerId)` so candidates coexist in caches.
5. Warm the allocation/pipeline pool, run three untimed warmups, then seven
   timestamp-query samples. Read results only after the batch completes and
   return the raw samples plus median.
6. Echo the state hash and device-limit identity so late results cannot paint a
   newer editor state or a different device model.

Measured results belong on the provenance entry for the exact ScheduleState,
not in the ScheduleState itself. A measurement is device- and workload-specific
evidence, not schedule identity.
