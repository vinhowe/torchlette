import type { DeviceKind } from "../backend/types";
import type { Tensor, Torchlette } from "../frontend";

/**
 * Validate common optimizer constructor parameters.
 * Returns the resolved Torchlette instance and shared device.
 */
export function validateOptimizerParams(
  name: string,
  params: Tensor[],
  api?: Torchlette,
): { api: Torchlette; device: DeviceKind } {
  if (params.length === 0) {
    throw new Error(`${name} requires at least one parameter`);
  }
  const engine = api ?? params[0]._engine();
  const device = params[0].device;
  for (const param of params) {
    if (param._engine() !== engine) {
      throw new Error(
        `${name} parameters must share the same Torchlette instance`,
      );
    }
    if (param.device !== device) {
      throw new Error(`${name} parameters must share the same device`);
    }
    if (!param.requiresGrad) {
      throw new Error(`${name} parameters must have requiresGrad=true`);
    }
  }
  return { api: engine, device };
}
