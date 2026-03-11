/**
 * Weight initialization functions.
 * Similar to PyTorch's torch.nn.init.
 *
 * All init functions modify the tensor in-place via api.copy_() and return void.
 */

import type { Tensor, Torchlette } from "../frontend";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Calculate fan_in and fan_out from a weight tensor's shape.
 *
 * For a 2-D Linear weight of shape [outFeatures, inFeatures]:
 *   fanIn  = inFeatures  (shape[1])
 *   fanOut = outFeatures  (shape[0])
 *
 * For conv layers (4-D+), the receptive field size is multiplied in.
 */
export function calculateFanInAndFanOut(tensor: Tensor): {
  fanIn: number;
  fanOut: number;
} {
  const shape = tensor.shape;
  if (shape.length < 2) {
    throw new Error(
      "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions",
    );
  }
  const fanOut = shape[0];
  const fanIn = shape[1];
  if (shape.length > 2) {
    const receptiveField = shape.slice(2).reduce((a, b) => a * b, 1);
    return {
      fanIn: fanIn * receptiveField,
      fanOut: fanOut * receptiveField,
    };
  }
  return { fanIn, fanOut };
}

/**
 * Return the recommended gain value for the given nonlinearity.
 *
 * Supported nonlinearities:
 *   linear / identity  → 1
 *   sigmoid            → 1
 *   tanh               → 5/3
 *   relu               → sqrt(2)
 *   leaky_relu         → sqrt(2 / (1 + negativeSlope^2))
 */
export function calculateGain(nonlinearity: string, param?: number): number {
  switch (nonlinearity) {
    case "linear":
    case "identity":
    case "sigmoid":
      return 1;
    case "tanh":
      return 5 / 3;
    case "relu":
      return Math.sqrt(2);
    case "leaky_relu": {
      const negativeSlope = param ?? 0.01;
      return Math.sqrt(2 / (1 + negativeSlope * negativeSlope));
    }
    default:
      throw new Error(`Unsupported nonlinearity: ${nonlinearity}`);
  }
}

// ---------------------------------------------------------------------------
// Basic fills
// ---------------------------------------------------------------------------

/** Fill tensor with values drawn from N(mean, std). */
export function normal_(
  api: Torchlette,
  tensor: Tensor,
  mean: number = 0,
  std: number = 1,
): void {
  const tmp = api.randn(tensor.shape, { device: tensor.device });
  if (mean === 0 && std === 1) {
    api.copy_(tensor, tmp);
  } else if (mean === 0) {
    api.copy_(tensor, api.mul(tmp, std));
  } else if (std === 1) {
    api.copy_(tensor, api.add(tmp, mean));
  } else {
    api.copy_(tensor, api.add(api.mul(tmp, std), mean));
  }
}

/** Fill tensor with values drawn from U(low, high). */
export function uniform_(
  api: Torchlette,
  tensor: Tensor,
  low: number = 0,
  high: number = 1,
): void {
  const tmp = api.rand(tensor.shape, { device: tensor.device });
  if (low === 0 && high === 1) {
    api.copy_(tensor, tmp);
  } else {
    const range = high - low;
    api.copy_(tensor, api.add(api.mul(tmp, range), low));
  }
}

/** Fill tensor with a constant value. */
export function constant_(api: Torchlette, tensor: Tensor, val: number): void {
  api.fill_(tensor, val);
}

/** Fill tensor with zeros. */
export function zeros_(api: Torchlette, tensor: Tensor): void {
  api.fill_(tensor, 0);
}

/** Fill tensor with ones. */
export function ones_(api: Torchlette, tensor: Tensor): void {
  api.fill_(tensor, 1);
}

// ---------------------------------------------------------------------------
// Kaiming (He) initialization
// ---------------------------------------------------------------------------

export type KaimingOptions = {
  /** Whether to use fan_in (default) or fan_out. */
  mode?: "fan_in" | "fan_out";
  /** Nonlinearity used after this layer (default: "leaky_relu"). */
  nonlinearity?: string;
  /** Negative slope for leaky_relu (default: 0). Ignored for other nonlinearities. */
  negativeSlope?: number;
};

/**
 * Kaiming He normal initialization.
 * Fills the tensor with values drawn from N(0, std) where
 *   std = gain / sqrt(fan)
 */
export function kaimingNormal_(
  api: Torchlette,
  tensor: Tensor,
  options?: KaimingOptions,
): void {
  const mode = options?.mode ?? "fan_in";
  const nonlinearity = options?.nonlinearity ?? "leaky_relu";
  const negativeSlope = options?.negativeSlope ?? 0;
  const { fanIn, fanOut } = calculateFanInAndFanOut(tensor);
  const fan = mode === "fan_in" ? fanIn : fanOut;
  const gain = calculateGain(
    nonlinearity,
    nonlinearity === "leaky_relu" ? negativeSlope : undefined,
  );
  const std = gain / Math.sqrt(fan);
  normal_(api, tensor, 0, std);
}

/**
 * Kaiming He uniform initialization.
 * Fills the tensor with values drawn from U(-bound, bound) where
 *   bound = gain * sqrt(3 / fan)
 */
export function kaimingUniform_(
  api: Torchlette,
  tensor: Tensor,
  options?: KaimingOptions,
): void {
  const mode = options?.mode ?? "fan_in";
  const nonlinearity = options?.nonlinearity ?? "leaky_relu";
  const negativeSlope = options?.negativeSlope ?? 0;
  const { fanIn, fanOut } = calculateFanInAndFanOut(tensor);
  const fan = mode === "fan_in" ? fanIn : fanOut;
  const gain = calculateGain(
    nonlinearity,
    nonlinearity === "leaky_relu" ? negativeSlope : undefined,
  );
  const bound = gain * Math.sqrt(3 / fan);
  uniform_(api, tensor, -bound, bound);
}

// ---------------------------------------------------------------------------
// Xavier (Glorot) initialization
// ---------------------------------------------------------------------------

export type XavierOptions = {
  /** Gain factor (default: 1.0). */
  gain?: number;
};

/**
 * Xavier (Glorot) uniform initialization.
 * Fills the tensor with values drawn from U(-bound, bound) where
 *   bound = gain * sqrt(6 / (fanIn + fanOut))
 */
export function xavierUniform_(
  api: Torchlette,
  tensor: Tensor,
  options?: XavierOptions,
): void {
  const gain = options?.gain ?? 1.0;
  const { fanIn, fanOut } = calculateFanInAndFanOut(tensor);
  const bound = gain * Math.sqrt(6 / (fanIn + fanOut));
  uniform_(api, tensor, -bound, bound);
}

/**
 * Xavier (Glorot) normal initialization.
 * Fills the tensor with values drawn from N(0, std) where
 *   std = gain * sqrt(2 / (fanIn + fanOut))
 */
export function xavierNormal_(
  api: Torchlette,
  tensor: Tensor,
  options?: XavierOptions,
): void {
  const gain = options?.gain ?? 1.0;
  const { fanIn, fanOut } = calculateFanInAndFanOut(tensor);
  const std = gain * Math.sqrt(2 / (fanIn + fanOut));
  normal_(api, tensor, 0, std);
}
