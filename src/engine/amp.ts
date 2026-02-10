/**
 * AMP (Automatic Mixed Precision) inside compile (§12)
 *
 * Implements "select-gated commits" for dtype conversion:
 * - AMP transforms run ONLY inside compiled regions
 * - Ops like matmul compute in f16 but accumulate in f32
 * - At region exit, outputs are conditionally cast based on policy
 * - The "select-gated" pattern allows the same compiled code to work
 *   with or without AMP by using runtime flags
 */

import type { DType } from "../backend/types";
import { F16_ELIGIBLE, F32_REQUIRED } from "./dtype-rules";

// ============================================================================
// AMP Policy and Configuration
// ============================================================================

/**
 * Ops that benefit from f16 compute (compute-bound, not memory-bound).
 * Derived from the centralized Op Dtype Registry in dtype-rules.ts.
 */
export const F16_ELIGIBLE_OPS: ReadonlySet<string> = F16_ELIGIBLE;

/**
 * Ops that must always use f32 for numerical stability.
 * Derived from the centralized Op Dtype Registry in dtype-rules.ts.
 */
export const F32_REQUIRED_OPS: ReadonlySet<string> = F32_REQUIRED;

/**
 * AMP policy configuration.
 * Defines which dtypes to use for different operation categories.
 */
export type AMPPolicy = {
  /** Enable AMP transforms */
  enabled: boolean;
  /** Dtype for compute-bound ops (matmul, etc.) - typically f16 */
  computeDtype: DType;
  /** Dtype for accumulation - always f32 for stability */
  accumulateDtype: "f32";
  /** Dtype for memory-bound ops - typically stays f32 */
  memoryDtype: DType;
  /** Dtype for loss/reduction ops - always f32 */
  reductionDtype: "f32";
};

/**
 * Default AMP policy: f16 compute, f32 accumulate.
 */
export const DEFAULT_AMP_POLICY: AMPPolicy = {
  enabled: true,
  computeDtype: "f16",
  accumulateDtype: "f32",
  memoryDtype: "f32",
  reductionDtype: "f32",
};

/**
 * Disabled AMP policy: everything in f32.
 */
export const DISABLED_AMP_POLICY: AMPPolicy = {
  enabled: false,
  computeDtype: "f32",
  accumulateDtype: "f32",
  memoryDtype: "f32",
  reductionDtype: "f32",
};

// ============================================================================
// Autocast Context (§12)
// ============================================================================

/**
 * Autocast configuration for a compiled region.
 */
export type AutocastConfig = {
  /** Whether autocast is active */
  enabled: boolean;
  /** The AMP policy to use */
  policy: AMPPolicy;
  /** Device type (only 'webgpu' supports f16 currently) */
  deviceType: "cpu" | "webgpu";
};

/**
 * Autocast context state.
 * Tracks the current autocast configuration during compile staging.
 */
export type AutocastContext = {
  /** Stack of autocast configs (for nested autocast blocks) */
  configStack: AutocastConfig[];
  /** Current active config (top of stack, or disabled if empty) */
  current: AutocastConfig;
};

/**
 * Create a new autocast context (disabled by default).
 */
export function createAutocastContext(): AutocastContext {
  return {
    configStack: [],
    current: {
      enabled: false,
      policy: DISABLED_AMP_POLICY,
      deviceType: "cpu",
    },
  };
}

/**
 * Push a new autocast config onto the stack.
 */
export function pushAutocast(
  ctx: AutocastContext,
  config: Partial<AutocastConfig>,
): void {
  const newConfig: AutocastConfig = {
    enabled: config.enabled ?? true,
    policy: config.policy ?? DEFAULT_AMP_POLICY,
    deviceType: config.deviceType ?? ctx.current.deviceType,
  };
  ctx.configStack.push(newConfig);
  ctx.current = newConfig;
}

/**
 * Pop the current autocast config from the stack.
 * Returns to the previous config, or disabled if stack is empty.
 */
export function popAutocast(ctx: AutocastContext): void {
  ctx.configStack.pop();
  ctx.current =
    ctx.configStack.length > 0
      ? ctx.configStack[ctx.configStack.length - 1]
      : {
          enabled: false,
          policy: DISABLED_AMP_POLICY,
          deviceType: "cpu",
        };
}

// ============================================================================
// Select-Gated Commit Logic (§12)
// ============================================================================

/**
 * Result of computing the output dtype for an op under AMP.
 */
export type SelectGatedResult = {
  /** The dtype to use for this op's output */
  outputDtype: DType;
  /** Whether a cast is needed from the compute dtype */
  needsCast: boolean;
  /** The source dtype if cast is needed */
  sourceDtype?: DType;
  /** Whether this decision was "gated" by AMP policy */
  isGated: boolean;
};

/**
 * Determine the output dtype for an op under AMP policy.
 * This implements the "select-gated" logic where the output dtype
 * is conditionally determined based on the AMP policy.
 *
 * @param op The operation name
 * @param inputDtypes The dtypes of the input tensors
 * @param ctx The current autocast context
 * @returns The select-gated result with output dtype info
 */
export function computeSelectGatedDtype(
  op: string,
  inputDtypes: DType[],
  ctx: AutocastContext,
): SelectGatedResult {
  // If autocast is disabled, preserve input dtypes
  if (!ctx.current.enabled) {
    // Use the first input's dtype, or f32 as fallback
    const outputDtype = inputDtypes[0] ?? "f32";
    return {
      outputDtype,
      needsCast: false,
      isGated: false,
    };
  }

  const policy = ctx.current.policy;

  // Check if this op requires f32 for stability
  if (F32_REQUIRED_OPS.has(op)) {
    const hasF16Input = inputDtypes.some((d) => d === "f16");
    return {
      outputDtype: "f32",
      needsCast: hasF16Input,
      sourceDtype: hasF16Input ? "f16" : undefined,
      isGated: true,
    };
  }

  // Check if this op benefits from f16 compute
  if (F16_ELIGIBLE_OPS.has(op)) {
    // For compute-bound ops, output in f16 (accumulate in f32 internally)
    const hasF32Input = inputDtypes.some((d) => d === "f32");
    return {
      outputDtype: policy.computeDtype,
      needsCast: hasF32Input && policy.computeDtype === "f16",
      sourceDtype: hasF32Input ? "f32" : undefined,
      isGated: true,
    };
  }

  // For other ops (elementwise, etc.), preserve the dominant input dtype
  // with a preference for the policy's memory dtype
  const hasF16 = inputDtypes.some((d) => d === "f16");
  const hasF32 = inputDtypes.some((d) => d === "f32");

  if (hasF16 && hasF32) {
    // Mixed inputs - use the memory dtype from policy
    return {
      outputDtype: policy.memoryDtype,
      needsCast: true,
      sourceDtype: policy.memoryDtype === "f32" ? "f16" : "f32",
      isGated: true,
    };
  }

  // Uniform inputs - preserve the dtype
  const outputDtype = inputDtypes[0] ?? "f32";
  return {
    outputDtype,
    needsCast: false,
    isGated: ctx.current.enabled,
  };
}

// ============================================================================
// IR Transform Helpers for AMP
// ============================================================================

/**
 * Represents a cast op to be inserted for AMP.
 */
export type AMPCastNode = {
  /** Node ID to cast */
  inputNodeId: number;
  /** Source dtype */
  fromDtype: DType;
  /** Target dtype */
  toDtype: DType;
  /** Reason for the cast */
  reason: "amp_input_cast" | "amp_output_cast" | "amp_accumulator_cast";
};

/**
 * Compute required casts for an op's inputs under AMP.
 *
 * @param op The operation name
 * @param inputNodeIds Node IDs of inputs
 * @param inputDtypes Current dtypes of inputs
 * @param ctx Autocast context
 * @returns List of casts to insert before this op
 */
export function computeInputCasts(
  op: string,
  inputNodeIds: number[],
  inputDtypes: DType[],
  ctx: AutocastContext,
): AMPCastNode[] {
  if (!ctx.current.enabled) {
    return [];
  }

  const policy = ctx.current.policy;
  const casts: AMPCastNode[] = [];

  // For f16-eligible ops, cast f32 inputs to f16
  if (F16_ELIGIBLE_OPS.has(op) && policy.computeDtype === "f16") {
    for (let i = 0; i < inputNodeIds.length; i++) {
      if (inputDtypes[i] === "f32") {
        casts.push({
          inputNodeId: inputNodeIds[i],
          fromDtype: "f32",
          toDtype: "f16",
          reason: "amp_input_cast",
        });
      }
    }
  }

  // For f32-required ops, cast f16 inputs to f32
  if (F32_REQUIRED_OPS.has(op)) {
    for (let i = 0; i < inputNodeIds.length; i++) {
      if (inputDtypes[i] === "f16") {
        casts.push({
          inputNodeId: inputNodeIds[i],
          fromDtype: "f16",
          toDtype: "f32",
          reason: "amp_input_cast",
        });
      }
    }
  }

  return casts;
}

/**
 * Compute required cast for an op's output under AMP.
 *
 * @param op The operation name
 * @param computeDtype The dtype used for computation
 * @param targetDtype The desired output dtype
 * @param ctx Autocast context
 * @returns Cast info if needed, null otherwise
 */
export function computeOutputCast(
  op: string,
  computeDtype: DType,
  targetDtype: DType,
  ctx: AutocastContext,
): AMPCastNode | null {
  if (!ctx.current.enabled) {
    return null;
  }

  if (computeDtype !== targetDtype) {
    return {
      inputNodeId: -1, // Will be filled in by caller
      fromDtype: computeDtype,
      toDtype: targetDtype,
      reason: "amp_output_cast",
    };
  }

  return null;
}

// ============================================================================
// Compiled Region AMP State
// ============================================================================

/**
 * AMP state captured at compiled region entry.
 * Used for cache keying and select-gated commit decisions.
 */
export type CompiledRegionAMPState = {
  /** Whether AMP was enabled at region entry */
  ampEnabled: boolean;
  /** The policy hash for cache keying */
  policyHash: string;
  /** Input dtypes at region entry */
  inputDtypes: DType[];
  /** Expected output dtypes (after select-gated logic) */
  outputDtypes: DType[];
};

/**
 * Hash an AMP policy for cache keying.
 */
export function hashAMPPolicy(policy: AMPPolicy): string {
  if (!policy.enabled) {
    return "disabled";
  }
  return `${policy.computeDtype}:${policy.accumulateDtype}:${policy.memoryDtype}`;
}

/**
 * Capture AMP state at compiled region entry.
 */
export function captureRegionAMPState(
  ctx: AutocastContext,
  inputDtypes: DType[],
): CompiledRegionAMPState {
  return {
    ampEnabled: ctx.current.enabled,
    policyHash: hashAMPPolicy(ctx.current.policy),
    inputDtypes,
    outputDtypes: [], // Will be filled in after IR transform
  };
}
