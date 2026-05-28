/**
 * Binary checkpoint format for DiLoCo agents.
 *
 * Layout:
 *   [u32 jsonLen LE][JSON header utf-8][raw f32 data, contiguous]
 *
 * JSON header: { shapes: number[][], dtype: "f32" }
 * Tensors are written in order; each tensor's raw f32 data follows the
 * previous one. Reader uses shapes[] to slice the data back into
 * Float32Arrays.
 *
 * Same envelope shape as the relay's binary frame format — easy to reason
 * about and zero ambiguity about boundaries.
 */

import * as fs from "node:fs";

export interface CheckpointHeader {
  shapes: number[][];
  dtype: "f32";
}

export function saveCheckpoint(
  path: string,
  shapes: number[][],
  tensors: Float32Array[],
): void {
  if (tensors.length !== shapes.length) {
    throw new Error(
      `saveCheckpoint: tensor count ${tensors.length} != shape count ${shapes.length}`,
    );
  }
  const header: CheckpointHeader = { shapes, dtype: "f32" };
  const headerBuf = Buffer.from(JSON.stringify(header), "utf-8");
  const headerLen = Buffer.alloc(4);
  headerLen.writeUInt32LE(headerBuf.length, 0);

  let totalBytes = 0;
  for (const t of tensors) totalBytes += t.byteLength;
  const dataBuf = Buffer.alloc(totalBytes);
  let off = 0;
  for (const t of tensors) {
    Buffer.from(t.buffer, t.byteOffset, t.byteLength).copy(dataBuf, off);
    off += t.byteLength;
  }

  fs.writeFileSync(path, Buffer.concat([headerLen, headerBuf, dataBuf]));
}

export function loadCheckpoint(path: string): {
  shapes: number[][];
  tensors: Float32Array[];
} {
  const buf = fs.readFileSync(path);
  const headerLen = buf.readUInt32LE(0);
  const header = JSON.parse(
    buf.subarray(4, 4 + headerLen).toString("utf-8"),
  ) as CheckpointHeader;
  if (header.dtype !== "f32") {
    throw new Error(`loadCheckpoint: unexpected dtype ${header.dtype}`);
  }
  const dataStart = 4 + headerLen;
  const tensors: Float32Array[] = [];
  let off = dataStart;
  for (const shape of header.shapes) {
    const numel = shape.reduce((a, b) => a * b, 1);
    const bytes = numel * 4;
    const arr = new Float32Array(numel);
    Buffer.from(buf.buffer, buf.byteOffset + off, bytes).copy(
      Buffer.from(arr.buffer),
    );
    tensors.push(arr);
    off += bytes;
  }
  return { shapes: header.shapes, tensors };
}
