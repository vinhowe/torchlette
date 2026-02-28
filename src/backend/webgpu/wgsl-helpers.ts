/**
 * Shared WGSL code-generation helpers.
 *
 * Provides multi-buffer write tracking for dispatch functions.
 */

import { trackSharedEncoderWrite } from "./index";
import type { GPUBuffer } from "./gpu-types";

/** Track multiple buffers in the shared encoder write set. */
export function trackBuffers(...buffers: GPUBuffer[]): void {
  for (const buf of buffers) trackSharedEncoderWrite(buf);
}
