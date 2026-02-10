/**
 * Shared WebGPU test helper.
 *
 * Auto-detects WebGPU availability at runtime.
 *
 * Use `TORCHLETTE_CPU_ONLY=1` to force-skip all WebGPU tests.
 */

import { initWebGPU, getWebGPUInitError } from "../../src/backend/webgpu";

/** If true, WebGPU tests are forcibly skipped. */
export const cpuOnly = process.env.TORCHLETTE_CPU_ONLY === "1";

let _webgpuAvailable: boolean | null = null;
let _initPromise: Promise<boolean> | null = null;

/**
 * Try to initialize WebGPU. Returns true if available, false otherwise.
 * Caches the result so subsequent calls are instant.
 */
export async function canUseWebGPU(): Promise<boolean> {
  if (cpuOnly) return false;
  if (_webgpuAvailable !== null) return _webgpuAvailable;
  if (!_initPromise) {
    _initPromise = initWebGPU().then((ok) => {
      _webgpuAvailable = ok;
      return ok;
    });
  }
  return _initPromise;
}

/**
 * Get the WebGPU init error message, if any.
 */
export function webgpuInitError(): string | null {
  return getWebGPUInitError();
}
