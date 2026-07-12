/**
 * The realizer REGISTRY (schedule-state-design.md §4). v2 registers the SECOND
 * realizer — Triton at REQUEST authority — alongside the coordinate the identity
 * layer (§5) keys emitted artifacts by.
 *
 * Ruling R13: there is NO generic `Realizer<...>` protocol in v1/v2 — each realizer
 * is its own concrete thing (the WGSL kernel realizer and the model-editor emit
 * consume different inputs and cannot share an abstraction until a THIRD genuine
 * instance earns the rule-of-three conversation). So this registry is a THIN stub:
 * it holds the Triton realizer's four §4 fields (capabilityProfile, emit, costModel-
 * reference, verificationHarness-reference) and the realizer coordinate for identity,
 * NOT a unified protocol. costModel and the differential harness live outside `src/`
 * (the harness is a python tool + a spec that drives it) — they are REFERENCED here,
 * not inlined, so the registry stays a registry.
 */

import type { RealizerCoordinate } from "../canonical";
import type { TiledMatmulDescriptor } from "../matmul-skeleton";
import type { ScheduleState } from "../types";
import { emitTritonTiledMatmul, type TritonEmission } from "./triton-emit";
import {
  TRITON_CAPABILITY_PROFILE,
  type TritonCapabilityProfile,
} from "./triton-profile";

/**
 * The Triton realizer registry entry (§4 `{ capabilityProfile, emit, costModel,
 * verificationHarness }`, concrete for Triton). `emit` is the tiled-matmul emitter;
 * `costModel` and `verificationHarness` are REFERENCES (the cost model is deferred
 * to §7 P3; the verification harness is the cross-backend differential — a python
 * tool + a spec, not a `src/` artifact — so it is named by path, not imported).
 */
export interface TritonRealizerEntry {
  readonly realizer: "triton";
  readonly capabilityProfile: TritonCapabilityProfile;
  /** emit: the tiled-matmul emitter (the only family v2's walking skeleton emits). */
  readonly emit: (
    state: ScheduleState,
    desc: TiledMatmulDescriptor,
  ) => TritonEmission;
  /** costModel: deferred to §7 P3 (single-sourced when it lands); referenced. */
  readonly costModelRef: string;
  /**
   * verificationHarness: THE cross-backend differential — the same ScheduleState
   * through both realizers, numerically diffed on a V100. It is a python harness
   * (`tools/triton-realizer/run_kernel.py`) driven by a TS differential
   * (`tools/triton-realizer/cross-backend-differential.ts`), NOT a `src/` module.
   */
  readonly verificationHarnessRef: string;
  /** The realizer coordinate the identity layer (§5) keys emitted artifacts by. */
  readonly coordinate: RealizerCoordinate;
}

/** The registered Triton realizer (v2). */
export const TRITON_REALIZER: TritonRealizerEntry = {
  realizer: "triton",
  capabilityProfile: TRITON_CAPABILITY_PROFILE,
  emit: emitTritonTiledMatmul,
  costModelRef:
    "docs/schedule-state-design.md §7 P3 (deferred; single-sourced)",
  verificationHarnessRef:
    "tools/triton-realizer/cross-backend-differential.ts + run_kernel.py",
  coordinate: {
    realizer: "triton",
    capabilityProfileVersion:
      TRITON_CAPABILITY_PROFILE.capabilityProfileVersion,
    targetArch: TRITON_CAPABILITY_PROFILE.pinnedSurface.targetArch,
  },
};

/**
 * The WGSL realizer coordinate (the FIRST realizer — v1, tile-IR→WGSL). Present so
 * the identity layer can key WGSL-emitted artifacts distinctly from Triton ones on
 * the SAME schedule digest (the artifact-cache split, §5). The WGSL realizer's full
 * entry lives with the live matmul path; here we only need its coordinate for the
 * cross-backend identity comparison.
 */
export const WGSL_REALIZER_COORDINATE: RealizerCoordinate = {
  realizer: "wgsl-dawn",
  capabilityProfileVersion: 1,
  targetArch: "webgpu",
};
