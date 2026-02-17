/**
 * Adam Parameter Packing
 *
 * Groups same-size optimizer parameters into contiguous packed buffers
 * so a single Adam dispatch can update all parameters in a size class.
 * Reduces dispatch count from O(params) to O(size_classes).
 *
 * Pack flow: copy individual buffers → packed buffers (encoder copies)
 * Dispatch: one Adam kernel per packed class
 * Unpack flow: copy packed results → individual buffers (encoder copies)
 */

import {
  getWebGPUDevice,
  trackSharedEncoderWrite,
  getSharedEncoderInstance,
} from "./index";
import type { PackedAdamSizeClass } from "../../engine/lowered-plan";

type GPUBuffer = { size: number; usage: number; destroy(): void };

// ============================================================================
// Classification
// ============================================================================

/**
 * Group parameters by element count. Returns packed classes (count > 1)
 * and singleton indices (count == 1, dispatched individually).
 */
export function classifyAdamParams(
  nodeElementCounts: number[],
): { packedClasses: PackedAdamSizeClass[]; singletonLocalIndices: number[] } {
  const bySize = new Map<number, number[]>();
  for (let i = 0; i < nodeElementCounts.length; i++) {
    const elems = nodeElementCounts[i];
    let arr = bySize.get(elems);
    if (!arr) {
      arr = [];
      bySize.set(elems, arr);
    }
    arr.push(i);
  }

  const packedClasses: PackedAdamSizeClass[] = [];
  const singletonLocalIndices: number[] = [];

  for (const [elems, indices] of bySize) {
    if (indices.length === 1) {
      singletonLocalIndices.push(indices[0]);
    } else {
      packedClasses.push({
        elementsPerParam: elems,
        paramCount: indices.length,
        totalElements: elems * indices.length,
        localParamIndices: indices,
      });
    }
  }

  // Sort packed classes by total elements (descending) for consistent ordering
  packedClasses.sort((a, b) => b.totalElements - a.totalElements);
  // Sort singletons by index
  singletonLocalIndices.sort((a, b) => a - b);

  return { packedClasses, singletonLocalIndices };
}

// ============================================================================
// Buffer Allocation
// ============================================================================

const STORAGE_COPY_USAGE = 0x80 | 0x04 | 0x08; // STORAGE | COPY_SRC | COPY_DST

/**
 * Allocate persistent packed GPUBuffers for each size class.
 * These buffers are reused across steps (never returned to pool).
 */
export function allocatePackedBuffers(
  classes: PackedAdamSizeClass[],
  emitF16: boolean,
): Array<{
  grad: GPUBuffer;
  param: GPUBuffer;
  m: GPUBuffer;
  v: GPUBuffer;
  paramF16: GPUBuffer | null;
}> {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device as any;

  return classes.map(cls => {
    const f32Bytes = cls.totalElements * 4;
    const f16Bytes = cls.totalElements * 2;
    return {
      grad: device.createBuffer({ size: f32Bytes, usage: STORAGE_COPY_USAGE }),
      param: device.createBuffer({ size: f32Bytes, usage: STORAGE_COPY_USAGE }),
      m: device.createBuffer({ size: f32Bytes, usage: STORAGE_COPY_USAGE }),
      v: device.createBuffer({ size: f32Bytes, usage: STORAGE_COPY_USAGE }),
      paramF16: emitF16
        ? device.createBuffer({ size: f16Bytes, usage: STORAGE_COPY_USAGE })
        : null,
    };
  });
}

/**
 * Allocate persistent config uniform buffers (one per packed class).
 */
export function allocateConfigBuffers(
  count: number,
  hasUnscale: boolean,
): GPUBuffer[] {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device as any;
  const configSize = hasUnscale ? 48 : 32;
  const result: GPUBuffer[] = [];
  for (let i = 0; i < count; i++) {
    result.push(device.createBuffer({
      size: configSize,
      usage: 0x40 | 0x08, // UNIFORM | COPY_DST
    }));
  }
  return result;
}

/**
 * Allocate persistent individual f16 buffers for packed params.
 * These are the targets for unpacking the packed f16 buffer.
 * Returns a Map from local param index → GPUBuffer.
 */
export function allocateIndividualF16Buffers(
  classes: PackedAdamSizeClass[],
): Map<number, any> {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device as any;
  const result = new Map<number, any>();
  for (const cls of classes) {
    const f16Bytes = cls.elementsPerParam * 2;
    for (const li of cls.localParamIndices) {
      result.set(li, device.createBuffer({ size: f16Bytes, usage: STORAGE_COPY_USAGE }));
    }
  }
  return result;
}

// ============================================================================
// Pack / Unpack (Encoder Copies)
// ============================================================================

/**
 * Encode copies from individual param buffers → packed buffers.
 * getBuffer(localIdx, kind) returns the GPUBuffer for the given param and buffer type.
 */
export function encodePack(
  encoder: any,
  classes: PackedAdamSizeClass[],
  packedBuffers: Array<{ grad: GPUBuffer; param: GPUBuffer; m: GPUBuffer; v: GPUBuffer }>,
  getBuffer: (localIdx: number, kind: "grad" | "param" | "m" | "v") => GPUBuffer,
): void {
  for (let ci = 0; ci < classes.length; ci++) {
    const cls = classes[ci];
    const packed = packedBuffers[ci];
    const bytesPerParam = cls.elementsPerParam * 4;

    for (let pi = 0; pi < cls.localParamIndices.length; pi++) {
      const localIdx = cls.localParamIndices[pi];
      const dstOffset = pi * bytesPerParam;

      encoder.copyBufferToBuffer(getBuffer(localIdx, "grad"), 0, packed.grad, dstOffset, bytesPerParam);
      encoder.copyBufferToBuffer(getBuffer(localIdx, "param"), 0, packed.param, dstOffset, bytesPerParam);
      encoder.copyBufferToBuffer(getBuffer(localIdx, "m"), 0, packed.m, dstOffset, bytesPerParam);
      encoder.copyBufferToBuffer(getBuffer(localIdx, "v"), 0, packed.v, dstOffset, bytesPerParam);
    }
  }
}

/**
 * Encode copies from packed buffers → individual param buffers (param, m, v only).
 * Grad is read-only, no unpack needed.
 * If emitF16, also copies packed f16 → individual f16 buffers.
 */
export function encodeUnpack(
  encoder: any,
  classes: PackedAdamSizeClass[],
  packedBuffers: Array<{ param: GPUBuffer; m: GPUBuffer; v: GPUBuffer; paramF16: GPUBuffer | null }>,
  getBuffer: (localIdx: number, kind: "param" | "m" | "v") => GPUBuffer,
  getF16Buffer: ((localIdx: number) => GPUBuffer | null) | null,
): void {
  for (let ci = 0; ci < classes.length; ci++) {
    const cls = classes[ci];
    const packed = packedBuffers[ci];
    const bytesPerParam = cls.elementsPerParam * 4;
    const f16BytesPerParam = cls.elementsPerParam * 2;

    for (let pi = 0; pi < cls.localParamIndices.length; pi++) {
      const localIdx = cls.localParamIndices[pi];
      const srcOffset = pi * bytesPerParam;

      encoder.copyBufferToBuffer(packed.param, srcOffset, getBuffer(localIdx, "param"), 0, bytesPerParam);
      encoder.copyBufferToBuffer(packed.m, srcOffset, getBuffer(localIdx, "m"), 0, bytesPerParam);
      encoder.copyBufferToBuffer(packed.v, srcOffset, getBuffer(localIdx, "v"), 0, bytesPerParam);

      if (packed.paramF16 && getF16Buffer) {
        const f16Buf = getF16Buffer(localIdx);
        if (f16Buf) {
          const f16SrcOffset = pi * f16BytesPerParam;
          encoder.copyBufferToBuffer(packed.paramF16, f16SrcOffset, f16Buf, 0, f16BytesPerParam);
        }
      }
    }
  }
}

// ============================================================================
// Config Buffer Update
// ============================================================================

// Pre-allocated typed arrays for config buffer writes
const _packedConfigData32 = new ArrayBuffer(32);
const _packedConfigF32_32 = new Float32Array(_packedConfigData32);
const _packedConfigU32_32 = new Uint32Array(_packedConfigData32);
const _packedConfigData48 = new ArrayBuffer(48);
const _packedConfigF32_48 = new Float32Array(_packedConfigData48);
const _packedConfigU32_48 = new Uint32Array(_packedConfigData48);

/**
 * Write Adam config into a pre-allocated uniform buffer.
 */
export function updatePackedConfigBuffer(
  queue: any,
  configBuf: GPUBuffer,
  config: { beta1: number; beta2: number; stepSize: number; eps: number;
            weightDecay: number; lrTimesWd: number; decoupledWd: boolean;
            invScale?: number },
  totalElements: number,
  hasUnscale: boolean,
): void {
  if (hasUnscale) {
    _packedConfigF32_48[0] = config.beta1;
    _packedConfigF32_48[1] = config.beta2;
    _packedConfigF32_48[2] = config.stepSize;
    _packedConfigF32_48[3] = config.eps;
    _packedConfigF32_48[4] = config.weightDecay;
    _packedConfigF32_48[5] = config.lrTimesWd;
    _packedConfigU32_48[6] = config.decoupledWd ? 1 : 0;
    _packedConfigU32_48[7] = totalElements;
    _packedConfigF32_48[8] = config.invScale ?? 1.0;
    _packedConfigU32_48[9] = 0;
    _packedConfigU32_48[10] = 0;
    _packedConfigU32_48[11] = 0;
    queue.writeBuffer(configBuf, 0, _packedConfigData48);
  } else {
    _packedConfigF32_32[0] = config.beta1;
    _packedConfigF32_32[1] = config.beta2;
    _packedConfigF32_32[2] = config.stepSize;
    _packedConfigF32_32[3] = config.eps;
    _packedConfigF32_32[4] = config.weightDecay;
    _packedConfigF32_32[5] = config.lrTimesWd;
    _packedConfigU32_32[6] = config.decoupledWd ? 1 : 0;
    _packedConfigU32_32[7] = totalElements;
    queue.writeBuffer(configBuf, 0, _packedConfigData32);
  }
}
