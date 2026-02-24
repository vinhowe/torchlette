import type { DType } from "./backend/types";
import {
  DEFAULT_AMP_POLICY,
  F16_ELIGIBLE_OPS,
  F32_REQUIRED_OPS,
  popAutocast,
  pushAutocast,
} from "./engine/amp";
import { OP_DTYPE_RULES, promoteDtype } from "./engine/dtype-rules";
import type { LazyOpCode } from "./engine/lazy";
import type { Torchlette } from "./frontend";
import type { Tensor } from "./frontend-tensor";
import type { AutocastOptions, PackHook, UnpackHook } from "./frontend-types";

/**
 * Execute a function with automatic mixed precision (AMP) enabled.
 */
export function autocastImpl<T>(
  torch: Torchlette,
  fn: () => T,
  options?: AutocastOptions,
): T {
  const deviceType =
    options?.deviceType ?? (torch.runtime.currentDefaultDevice === "webgpu" ? "webgpu" : "cpu");

  pushAutocast(torch._getAutocastContext(), {
    enabled: options?.enabled ?? true,
    policy: options?.policy ?? DEFAULT_AMP_POLICY,
    deviceType,
  });

  // Set the engine's autocast context for AMP transforms in compile (§12)
  torch.engine.setAutocastContext(torch._getAutocastContext());

  try {
    return fn();
  } finally {
    popAutocast(torch._getAutocastContext());
    // Update engine's context to reflect the popped state
    torch.engine.setAutocastContext(
      torch._getAutocastContext().configStack.length > 0 ? torch._getAutocastContext() : null,
    );
  }
}

/**
 * Async version of autocast for async functions.
 */
export async function autocastAsyncImpl<T>(
  torch: Torchlette,
  fn: () => Promise<T>,
  options?: AutocastOptions,
): Promise<T> {
  const deviceType =
    options?.deviceType ?? (torch.runtime.currentDefaultDevice === "webgpu" ? "webgpu" : "cpu");

  pushAutocast(torch._getAutocastContext(), {
    enabled: options?.enabled ?? true,
    policy: options?.policy ?? DEFAULT_AMP_POLICY,
    deviceType,
  });

  // Set the engine's autocast context for AMP transforms in compile (§12)
  torch.engine.setAutocastContext(torch._getAutocastContext());

  try {
    return await fn();
  } finally {
    popAutocast(torch._getAutocastContext());
    // Update engine's context to reflect the popped state
    torch.engine.setAutocastContext(
      torch._getAutocastContext().configStack.length > 0 ? torch._getAutocastContext() : null,
    );
  }
}

/**
 * Context manager for intercepting tensor save/restore operations.
 * This is the foundation for gradient checkpointing (§10).
 */
export function savedTensorHooksImpl<T>(
  torch: Torchlette,
  packHook: PackHook,
  unpackHook: UnpackHook,
  fn: () => T,
): T {
  torch._savedTensorHooksStack.push({ packHook, unpackHook });
  try {
    return fn();
  } finally {
    torch._savedTensorHooksStack.pop();
  }
}

/**
 * Cast a tensor for autocast, with differentiable backward through the cast.
 * Backward only upcasts gradients (e.g., f16→f32), never downcasts,
 * to preserve gradient precision during mixed-precision training.
 */
export function autocastCastImpl(torch: Torchlette, a: Tensor, targetDtype: DType): Tensor {
  if (a.dtype === targetDtype) return a;
  const originalDtype = a.dtype;
  const inner = torch.runtime.cast(a._unwrap(), targetDtype);
  return torch._wrapWithGrad(inner, [a], (grad, _getSaved) => {
    if (grad.dtype === originalDtype) return [grad];
    // Only upcast: promote to whichever dtype is higher precision
    const target = promoteDtype(grad.dtype, originalDtype);
    if (target === grad.dtype) return [grad]; // originalDtype is lower → keep grad as-is
    return [torch.runtime.cast(grad, target)];
  });
}

/**
 * Unified autocast dispatch: applies autocast policy AND dtype promotion
 * based on the centralized Op Dtype Registry.
 */
export function applyAutocastImpl(torch: Torchlette, op: string, inputs: Tensor[]): Tensor[] {
  const rule = OP_DTYPE_RULES[op as LazyOpCode];

  // Autocast policy (only when active)
  if (torch._getAutocastContext().current.enabled) {
    const policy = torch._getAutocastContext().current.policy;
    // Check both the registry rule and the supplementary sets
    const isF16Eligible = (rule && rule.category === "f16_eligible") || F16_ELIGIBLE_OPS.has(op);
    const isF32Required = (rule && rule.category === "f32_required") || F32_REQUIRED_OPS.has(op);

    if (isF16Eligible && policy.computeDtype === "f16") {
      inputs = inputs.map(t => t.dtype === "f32" ? autocastCastImpl(torch, t, "f16") : t);
    }
    if (isF32Required) {
      inputs = inputs.map(t => t.dtype === "f16" ? autocastCastImpl(torch, t, "f32") : t);
    }
  }

  // Binary dtype promotion (always active, with differentiable cast)
  if (rule && rule.category === "promote_inputs" && inputs.length >= 2) {
    const [a, b] = inputs;
    if (a.dtype !== b.dtype) {
      const target = promoteDtype(a.dtype, b.dtype);
      return [
        a.dtype === target ? a : autocastCastImpl(torch, a, target),
        b.dtype === target ? b : autocastCastImpl(torch, b, target),
        ...inputs.slice(2),
      ];
    }
  }

  return inputs;
}
