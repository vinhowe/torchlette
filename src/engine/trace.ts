import type { DType } from "../backend/types";

export type TraceEvent =
  | {
      type: "after_all";
      inputs: number[];
      output: number;
      outputKey: string;
    }
  | {
      type: "effect";
      op: string;
      input: number;
      output: number;
      locId?: number;
    }
  | {
      type: "lazy_op";
      op: string;
      traceId: number;
      epoch: number;
      inputs?: number[];
      shape?: number[];
      dtype?: DType;
      scalarValues?: number[];  // §8.2.1: scalar constants for cache key differentiation
    }
  | {
      type: "checkpoint_pack";
      packId: number;
      reachableBases: number[];
    }
  | {
      type: "checkpoint_recompute_start";
      packId: number;
      reachableBases: number[];
    }
  | {
      type: "checkpoint_recompute_finish";
      packId: number | null;
    }
  | {
      type: "rng_draw";
      opNonce: number;
      drawNonce: number;
      value: number;
    }
  | {
      type: "rng_basis";
      algorithmId: number;
      seed: number;
    }
  | {
      type: "rng_checkpoint_record_start";
    }
  | {
      type: "rng_checkpoint_record_finish";
      count: number;
    }
  | {
      type: "rng_checkpoint_replay_start";
      count: number;
    }
  | {
      type: "rng_checkpoint_replay_finish";
      count: number;
    }
  | {
      type: "publish_save";
    }
  | {
      type: "compiled_call";
      graphInstanceId: number;
      callInstanceId: number;
    }
  | {
      type: "force_plan";
      baseId: number;
      tokenIds: number[];
    }
  | {
      type: "loc_schedule";
      locId: number;
      locLogicalVersion: number;
    }
  | {
      type: "loc_commit";
      locId: number;
      locVersion: number;
    }
  | {
      type: "base_commit";
      baseId: number;
      mutId: number;
      baseCommitVersion: number;
    }
  | {
      type: "finalize_enqueue";
      recordId: number;
    }
  | {
      type: "finalize_drain";
      count: number;
    }
  | {
      type: "mark_step_begin";
    }
  | {
      type: "mark_step_finalize_bindings";
      count: number;
    }
  | {
      type: "mark_step_retain";
    }
  | {
      type: "mark_step_gc";
    }
  | {
      type: "mark_step_end";
    }
  | {
      type: "set_token";
      target: "global" | "loc";
      token: number;
      locId?: number;
    };

export class TraceRecorder {
  private readonly events: TraceEvent[] = [];

  record(event: TraceEvent): void {
    this.events.push(event);
  }

  snapshot(): TraceEvent[] {
    return this.events.slice();
  }
}
