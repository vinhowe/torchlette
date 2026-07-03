/**
 * Strided scatter ops: stridedScatterCopy (+ chunked), stridedScatterAdd (+ chunked).
 */
import { recordedCopyBufferToBuffer } from "../../../executor/compiled-plan";
import type { BackendTensor } from "../../types";
import {
  cachedCreateBindGroup,
  createParamsBuffer,
  params,
  releaseParamsBuffer,
} from "../bind-group-cache";
import { resolveOutputBuffer } from "../buffer-arena";
import { bufferPool, destroyCopy } from "../buffer-pool";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { requireContext } from "../gpu-context";
import type { WebGPUTensor } from "../gpu-types";
import { asGPUTensor } from "../gpu-types";
import {
  checkContiguousStrides,
  compute2DDispatch,
  sizeOf,
  WORKGROUP_SIZE,
} from "../shape-utils";
import { getSharedEncoderInstance, submitOrCollect } from "../shared-encoder";
import { createTensor } from "../tensor";
import { createTileKernelDispatcher } from "../tile-dispatch";
import { trackSharedEncoderWrite } from "../webgpu-state";
import {
  flatOpSpec,
  stridedScatterAddTileIR,
  stridedScatterCopyTileIR,
} from "./ops-tile-ir";
import { ensureContiguous } from "./views";

type ScatterOp = "copy" | "add";

/** Element byte size per dtype (f16 is the only sub-4-byte dtype we store). */
function dtypeBytes(dtype: string | undefined): number {
  return dtype === "f16" ? 2 : 4;
}

/** Direct strided scatter dispatch (small tensors that fit in a single binding). */
function stridedScatterDirect(
  op: ScatterOp,
  baseTensor: WebGPUTensor,
  srcTensor: WebGPUTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  const { offset, viewShape, viewStrides } = options;
  const baseSize = sizeOf(baseTensor.shape);
  const viewSize = sizeOf(viewShape);
  const ctx = requireContext();

  // The scatter tile-IR kernels read/write array<f32>; a non-f32 tensor here
  // would be silently reinterpreted (observed: f16 pairs read as one f32 —
  // see examples/qwen3/probe-f16b.ts). Same-dtype FULL contiguous copies are
  // handled by the DMA fast path in stridedScatterImpl before reaching this.
  if (baseTensor.dtype !== "f32" || srcTensor.dtype !== "f32") {
    throw new Error(
      `stridedScatter${op === "copy" ? "Copy" : "Add"}: kernel path only supports f32 (got base=${baseTensor.dtype}, src=${srcTensor.dtype}). ` +
        `Non-f32 in-place updates are only supported as full contiguous copies.`,
    );
  }

  const maxSize = Math.max(baseSize, viewSize);
  const totalWorkgroups = Math.ceil(maxSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // ensureContiguous also materializes offset>0 views with contiguous
  // strides — the kernel reads `base` flat from element 0 and treats the
  // viewMeta `offset` as base-relative (offset-view class, task #58).
  const contiguousBase = ensureContiguous(baseTensor);
  const srcStrides = srcTensor.strides;
  const srcOffset = srcTensor.offset;

  const tileIR =
    op === "copy" ? stridedScatterCopyTileIR : stridedScatterAddTileIR;
  const code = tileIR(
    baseSize,
    viewShape,
    viewStrides,
    offset,
    srcStrides,
    srcOffset,
  );
  const opName = op === "copy" ? "Copy" : "Add";
  const key = `stridedScatter${opName}:${baseSize}:${viewShape.join("x")}:${viewStrides.join(",")}:${offset}:${srcStrides.join(",")}:${srcOffset}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const outBuffer = resolveOutputBuffer(ctx.device, baseSize * 4, [
    contiguousBase.buffer,
    srcTensor.buffer,
  ]);
  const paramsBuffer = createParamsBuffer(
    ctx.device,
    params(baseSize, viewSize),
  );
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
    contiguousBase.buffer,
    srcTensor.buffer,
    outBuffer,
    paramsBuffer,
  ]);
  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
  releaseParamsBuffer(paramsBuffer);

  if (contiguousBase !== baseTensor) destroyCopy(contiguousBase);
  return createTensor(baseTensor.shape, outBuffer);
}

/** Unified strided scatter: handles chunking for large tensors, dispatches direct for small. */
function stridedScatterImpl(
  op: ScatterOp,
  base: BackendTensor,
  src: BackendTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  const baseTensor = asGPUTensor(base);
  const srcTensor = asGPUTensor(src);
  const { offset, viewShape, viewStrides } = options;

  const baseSize = sizeOf(baseTensor.shape);
  const viewSize = sizeOf(viewShape);

  if (baseSize === 0) {
    throw new Error(
      `stridedScatter${op === "copy" ? "Copy" : "Add"}: empty base tensor`,
    );
  }

  const ctx = requireContext();
  const maxBindingSize =
    ctx.device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  // Byte math must follow the tensor dtype — with hardcoded f32 sizes an f16
  // full copy_ failed the fast-path size check below and fell into the
  // f32-typed kernel, silently corrupting the destination.
  const bpe = dtypeBytes(baseTensor.dtype);
  const baseSizeBytes = baseSize * bpe;
  const srcSizeBytes = srcTensor.size * bpe;

  // TRUE IN-PLACE fast path for full contiguous overwrites (copy_ semantics):
  // one recorded buffer-to-buffer DMA into the base's EXISTING buffer, zero
  // allocation, and — critically — the destination tensor keeps its storage
  // identity. The copy-on-write path below migrates the destination into a
  // fresh arena buffer every call, which makes persistent tensors updated
  // via copy_ (params, packed optimizer state) ping-pong against their own
  // plan positions: a perpetual external-input conflict, one fresh
  // multi-hundred-MB allocation per state tensor per step. In-place writes
  // follow the framework's existing in-place discipline (readers of the old
  // value are scheduled earlier in the plan; in-queue pass ordering
  // serializes them — same contract as the fused adamStep).
  // Applies to FULL overwrites and to contiguous REGION writes (copyInto_):
  // a contiguous src into a contiguous slice [offset, offset+viewSize) of a
  // contiguous base is a pure DMA at a byte offset. Without this, every
  // region write took the copy-on-write path below — a full-base migration
  // per call (148 region refills/step at 124M = tens of GB of churn, OOM).
  if (
    op === "copy" &&
    baseTensor.dtype === srcTensor.dtype &&
    offset >= 0 &&
    offset + viewSize <= baseSize &&
    checkContiguousStrides(viewShape, viewStrides) &&
    baseTensor.isContiguous &&
    (baseTensor.offset ?? 0) === 0 &&
    (srcTensor.isContiguous ||
      checkContiguousStrides(srcTensor.shape, srcTensor.strides)) &&
    srcTensor.buffer !== baseTensor.buffer &&
    viewSize * bpe <= srcTensor.buffer.size - (srcTensor.offset ?? 0) * bpe
  ) {
    const regionBytes = viewSize * bpe;
    trackSharedEncoderWrite(baseTensor.buffer);
    const enc = getSharedEncoderInstance();
    if (enc) {
      recordedCopyBufferToBuffer(
        enc,
        srcTensor.buffer,
        (srcTensor.offset ?? 0) * bpe,
        baseTensor.buffer,
        offset * bpe,
        regionBytes,
      );
    } else {
      const cmdEnc = ctx.device.createCommandEncoder();
      cmdEnc.copyBufferToBuffer(
        srcTensor.buffer,
        (srcTensor.offset ?? 0) * bpe,
        baseTensor.buffer,
        offset * bpe,
        regionBytes,
      );
      submitOrCollect(cmdEnc.finish());
    }
    // Ownership transfer (same pattern as the fused adamStep): the result
    // tensor wraps the base's buffer; neutralize the input tensor's release
    // so the buffer isn't double-released when its handle drops.
    if (baseTensor.ownsBuffer) bufferPool.decRef(baseTensor.buffer);
    baseTensor.destroy = () => {};
    return createTensor(
      baseTensor.shape,
      baseTensor.buffer,
      undefined,
      0,
      baseTensor.dtype,
    );
  }

  if (baseSizeBytes > maxBindingSize || srcSizeBytes > maxBindingSize) {
    const isFull = viewSize === baseSize && offset === 0;
    const viewContiguous = checkContiguousStrides(viewShape, viewStrides);
    const srcContiguous =
      srcTensor.isContiguous ||
      checkContiguousStrides(srcTensor.shape, srcTensor.strides);

    if (isFull && baseTensor.isContiguous && srcContiguous && viewContiguous) {
      return op === "copy"
        ? stridedScatterCopyChunkedSimple(baseTensor, srcTensor)
        : stridedScatterAddChunkedSimple(baseTensor, srcTensor);
    }
    // General chunked case for complex stride patterns — not yet implemented.
    // Fall through to direct implementation (will fail with validation error for truly large tensors).
  }

  return stridedScatterDirect(op, baseTensor, srcTensor, options);
}

/**
 * Copy src values into base tensor at positions defined by view metadata.
 * Returns a new tensor (does not mutate base).
 */
export function stridedScatterCopy(
  base: BackendTensor,
  src: BackendTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  return stridedScatterImpl("copy", base, src, options);
}

/**
 * Add src values into base tensor at positions defined by view metadata.
 * Returns a new tensor (does not mutate base).
 */
export function stridedScatterAdd(
  base: BackendTensor,
  src: BackendTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  return stridedScatterImpl("add", base, src, options);
}

/**
 * Chunked implementation for simple full contiguous copy (src -> output).
 * Used when both src and dest exceed buffer binding limit but are contiguous
 * and the copy covers all elements.
 */
function stridedScatterCopyChunkedSimple(
  baseTensor: WebGPUTensor,
  srcTensor: WebGPUTensor,
): BackendTensor {
  const totalElements = baseTensor.size;
  const bytesPerElement = 4; // f32

  const outBuffer = resolveOutputBuffer(
    requireContext().device,
    totalElements * bytesPerElement,
    [baseTensor.buffer, srcTensor.buffer],
  );

  const dispatcher = createTileKernelDispatcher(flatOpSpec("copy"));
  dispatcher.dispatchChunked(
    { src: srcTensor.buffer, out: outBuffer },
    { size: totalElements },
    {
      modes: { src: "chunked", out: "chunked" },
      sizeUniform: "size",
      totalElements,
      maxBytesPerElement: bytesPerElement,
    },
  );

  return createTensor(baseTensor.shape, outBuffer);
}

/**
 * Chunked implementation for simple full contiguous add (base + src -> output).
 * Used when both base and src exceed buffer binding limit but are contiguous
 * and the add covers all elements.
 */
function stridedScatterAddChunkedSimple(
  baseTensor: WebGPUTensor,
  srcTensor: WebGPUTensor,
): BackendTensor {
  const totalElements = baseTensor.size;
  const bytesPerElement = 4; // f32

  const outBuffer = resolveOutputBuffer(
    requireContext().device,
    totalElements * bytesPerElement,
    [baseTensor.buffer, srcTensor.buffer],
  );

  const dispatcher = createTileKernelDispatcher(flatOpSpec("add"));
  dispatcher.dispatchChunked(
    { base: baseTensor.buffer, src: srcTensor.buffer, out: outBuffer },
    { size: totalElements },
    {
      modes: { base: "chunked", src: "chunked", out: "chunked" },
      sizeUniform: "size",
      totalElements,
      maxBytesPerElement: bytesPerElement,
    },
  );

  return createTensor(baseTensor.shape, outBuffer);
}
