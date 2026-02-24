import type { DeviceKind } from "./backend/types";
import type { Torchlette } from "./frontend";
import type { Tensor } from "./frontend-tensor";
import type { TensorCreateOptions } from "./frontend-types";

export function tensorFromArrayImpl(
  torch: Torchlette,
  values: number[] | Float32Array,
  shape: number[],
  options?: TensorCreateOptions,
): Tensor {
  return torch._wrap(
    torch.runtime.tensorFromArray(values, shape, options?.device),
    options?.requiresGrad ?? false,
  );
}

/**
 * Create a tensor filled with random values uniformly distributed in [0, 1).
 */
export function randImpl(
  torch: Torchlette,
  shape: number[],
  options?: TensorCreateOptions,
): Tensor {
  return torch._wrap(
    torch.runtime.rand(shape, options?.device),
    options?.requiresGrad ?? false,
  );
}

/**
 * Create a tensor filled with random values from a standard normal distribution.
 */
export function randnImpl(
  torch: Torchlette,
  shape: number[],
  options?: TensorCreateOptions,
): Tensor {
  return torch._wrap(
    torch.runtime.randn(shape, options?.device),
    options?.requiresGrad ?? false,
  );
}

/**
 * Create a tensor with values sampled from a Bernoulli distribution.
 * Each element is 1 with probability p, and 0 with probability (1-p).
 */
export function bernoulliImpl(
  torch: Torchlette,
  shape: number[],
  p: number,
  options?: TensorCreateOptions,
): Tensor {
  if (p < 0 || p > 1) {
    throw new Error(`Bernoulli probability must be between 0 and 1, got ${p}`);
  }
  return torch._wrap(
    torch.runtime.bernoulli(shape, p, options?.device),
    options?.requiresGrad ?? false,
  );
}

/**
 * Create a tensor filled with zeros.
 */
export function zerosImpl(
  torch: Torchlette,
  shape: number[],
  options?: TensorCreateOptions,
): Tensor {
  return torch._wrap(
    torch.runtime.zeros(shape, options?.device),
    options?.requiresGrad ?? false,
  );
}

/**
 * Create a tensor filled with ones.
 */
export function onesImpl(
  torch: Torchlette,
  shape: number[],
  options?: TensorCreateOptions,
): Tensor {
  return fullImpl(torch, shape, 1, options);
}

/**
 * Create a tensor filled with a specific value.
 */
export function fullImpl(
  torch: Torchlette,
  shape: number[],
  fillValue: number,
  options?: TensorCreateOptions,
): Tensor {
  return torch._wrap(
    torch.runtime.full(shape, fillValue, options?.device),
    options?.requiresGrad ?? false,
  );
}

/**
 * Create a 1-D tensor of evenly spaced values.
 * Like PyTorch's torch.arange(start, end, step).
 * If only one argument is given, it is treated as `end` (start=0, step=1).
 */
export function arangeImpl(
  torch: Torchlette,
  end: number,
  options?: { start?: number; step?: number; device?: DeviceKind; requiresGrad?: boolean },
): Tensor {
  const start = options?.start ?? 0;
  const step = options?.step ?? 1;
  return torch._wrap(
    torch.runtime.arange(end, start, step, options?.device),
    options?.requiresGrad ?? false,
  );
}

/**
 * Return the lower-triangular part of a matrix, zeroing elements above the k-th diagonal.
 */
export function trilImpl(torch: Torchlette, a: Tensor, k = 0): Tensor {
  torch._assertUsable(a);
  return torch._wrap(torch.runtime.tril(a._unwrap(), k), false);
}

/**
 * Return the upper-triangular part of a matrix, zeroing elements below the k-th diagonal.
 */
export function triuImpl(torch: Torchlette, a: Tensor, k = 0): Tensor {
  torch._assertUsable(a);
  return torch._wrap(torch.runtime.triu(a._unwrap(), k), false);
}
