/**
 * Binary WebSocket frame encoding for upload/download payloads.
 *
 * JSON-encoded `number[]` is ~7-15x the byte size of raw f32 data; for
 * tensors over a few KB it dominates wire time. Binary frames avoid both
 * the overhead and the JSON parse cost on the receiver.
 *
 * Frame layout (little-endian):
 *
 *   ┌──────────┬──────────┬──────────┬────────────┬────────────┐
 *   │ id (u32) │ dtype(u8)│ rank (u8)│ rank × u32 │ raw bytes  │
 *   │          │          │          │  (shape)   │ (payload)  │
 *   └──────────┴──────────┴──────────┴────────────┴────────────┘
 *     4         1          1           2 pad + 4*rank
 *
 * Upload (client → server): client sends one binary frame (no text). Server
 *   decodes id, creates the tensor, responds with a text frame
 *   `{id, result: {handle}}`.
 * Download (server → client): client sends text `{id, method:"download"}`.
 *   Server responds with one binary frame carrying the tensor data.
 *
 * Text frames still carry all control messages (execute, release, hellos,
 * error responses, and the response envelopes for upload/execute/release).
 */

import type { DType } from "../backend/types";

const HEADER_FIXED_SIZE = 8; // id(4) + dtype(1) + rank(1) + padding(2)

// ============================================================================
// DType encoding
// ============================================================================

const DTYPE_CODES: Record<DType, number> = {
  f32: 0,
  f16: 1,
  i32: 2,
  u32: 3,
  bool: 4,
};
const DTYPE_BY_CODE: DType[] = ["f32", "f16", "i32", "u32", "bool"];

function dtypeToCode(dt: DType): number {
  const code = DTYPE_CODES[dt];
  if (code === undefined) throw new Error(`unsupported dtype: ${dt}`);
  return code;
}

function codeToDtype(code: number): DType {
  const dt = DTYPE_BY_CODE[code];
  if (!dt) throw new Error(`unknown dtype code: ${code}`);
  return dt;
}

// ============================================================================
// Typed array views
// ============================================================================

export type TensorBytes =
  | Float32Array
  | Int32Array
  | Uint32Array
  | Uint8Array;

function typedArrayFor(
  dt: DType,
  buffer: ArrayBuffer,
  byteOffset: number,
  byteLength: number,
): TensorBytes {
  switch (dt) {
    case "f32":
      return new Float32Array(buffer, byteOffset, byteLength / 4);
    case "f16":
      // No native f16 typed array in JS; carry as Uint16Array externally.
      // We return Float32Array views of already-widened data for f16,
      // but the server would hand us raw 2-byte values. Not used by the
      // current demo — throw for now to catch misuse.
      throw new Error("f16 binary framing not yet implemented");
    case "i32":
      return new Int32Array(buffer, byteOffset, byteLength / 4);
    case "u32":
      return new Uint32Array(buffer, byteOffset, byteLength / 4);
    case "bool":
      return new Uint8Array(buffer, byteOffset, byteLength);
  }
}

export function bytesOf(values: TensorBytes): ArrayBuffer {
  // Return a fresh ArrayBuffer slice containing exactly the values' bytes.
  // TypedArrays can be views into larger buffers; slice() copies.
  return values.buffer.slice(
    values.byteOffset,
    values.byteOffset + values.byteLength,
  );
}

// ============================================================================
// Encode
// ============================================================================

export interface BinaryFrame {
  id: number;
  dtype: DType;
  shape: number[];
  values: TensorBytes;
}

export function encodeBinaryFrame(frame: BinaryFrame): ArrayBuffer {
  const rank = frame.shape.length;
  if (rank > 255) throw new Error("tensor rank exceeds 255");
  const shapeBytes = rank * 4;
  const valueBytes = frame.values.byteLength;
  const total = HEADER_FIXED_SIZE + shapeBytes + valueBytes;

  const buf = new ArrayBuffer(total);
  const view = new DataView(buf);
  view.setUint32(0, frame.id >>> 0, true);
  view.setUint8(4, dtypeToCode(frame.dtype));
  view.setUint8(5, rank);
  // bytes 6–7 reserved (0)
  for (let i = 0; i < rank; i++) {
    view.setUint32(HEADER_FIXED_SIZE + i * 4, frame.shape[i] >>> 0, true);
  }
  new Uint8Array(buf).set(
    new Uint8Array(
      frame.values.buffer,
      frame.values.byteOffset,
      frame.values.byteLength,
    ),
    HEADER_FIXED_SIZE + shapeBytes,
  );
  return buf;
}

// ============================================================================
// Decode
// ============================================================================

export function decodeBinaryFrame(buffer: ArrayBuffer): BinaryFrame {
  if (buffer.byteLength < HEADER_FIXED_SIZE) {
    throw new Error(
      `binary frame too small: ${buffer.byteLength} < ${HEADER_FIXED_SIZE}`,
    );
  }
  const view = new DataView(buffer);
  const id = view.getUint32(0, true);
  const dtype = codeToDtype(view.getUint8(4));
  const rank = view.getUint8(5);
  // 2 bytes reserved
  const shapeBytes = rank * 4;
  if (buffer.byteLength < HEADER_FIXED_SIZE + shapeBytes) {
    throw new Error("binary frame truncated at shape");
  }
  const shape: number[] = new Array(rank);
  for (let i = 0; i < rank; i++) {
    shape[i] = view.getUint32(HEADER_FIXED_SIZE + i * 4, true);
  }
  const payloadOffset = HEADER_FIXED_SIZE + shapeBytes;
  const payloadLen = buffer.byteLength - payloadOffset;
  const values = typedArrayFor(dtype, buffer, payloadOffset, payloadLen);
  return { id, dtype, shape, values };
}

// ============================================================================
// Conversion helpers
// ============================================================================

/** Wrap a plain number[] as a dtype-appropriate typed array. */
export function valuesToTypedArray(
  values: number[] | TensorBytes,
  dtype: DType,
): TensorBytes {
  if (typeof values !== "object") {
    throw new Error("values must be an array or typed array");
  }
  if (ArrayBuffer.isView(values)) return values as TensorBytes;
  switch (dtype) {
    case "f32":
      return new Float32Array(values);
    case "i32":
      return new Int32Array(values);
    case "u32":
      return new Uint32Array(values);
    case "bool":
      return new Uint8Array(values);
    case "f16":
      throw new Error("f16 binary framing not yet implemented");
  }
}
