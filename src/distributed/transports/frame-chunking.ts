/**
 * Wire-frame chunking for large tensor payloads.
 *
 * A v2 transport serializes each message to a binary frame
 *   [u32 LE envLen][envelope JSON][payload]
 * and ships it in ONE WebSocket message. That's fine node↔node (the `ws`
 * library handles huge frames), but a browser's native WebSocket silently
 * drops a single ~250 MB message — which is exactly the size of a 124M-param
 * f16 weight-sync or per-round pseudo-grad. So oversized frames are split here
 * into bounded, individually-routable chunk frames and reassembled on receipt.
 *
 * Each chunk is itself a normal wire frame whose envelope carries the ORIGINAL
 * `target` (so the envelope-routing relay forwards every chunk to the same
 * recipients, with no server changes) and a `__chunk` control msg with a frame
 * id + index/count. The receiver buffers chunks by id and, once all arrive,
 * concatenates the body slices back into the original frame and re-parses it.
 */
import type { PeerId, SendTarget } from "../protocol/messages";

const FOUR = 4;
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

/** Max bytes per WebSocket message. Browsers handle a few MB reliably; the
 *  250 MB single-frame case is what fails. 8 MB → ~32 chunks for a 124M f16. */
export const MAX_WS_FRAME = 8 * 1024 * 1024;

const CHUNK_TYPE = "__chunk";
let frameSeq = 0;

interface ChunkMsg {
  type: typeof CHUNK_TYPE;
  id: string;
  i: number;
  n: number;
}

function buildFrame(envelope: unknown, body: Uint8Array): Uint8Array {
  const envBytes = textEncoder.encode(JSON.stringify(envelope));
  const out = new Uint8Array(FOUR + envBytes.byteLength + body.byteLength);
  new DataView(out.buffer).setUint32(0, envBytes.byteLength, true);
  out.set(envBytes, FOUR);
  out.set(body, FOUR + envBytes.byteLength);
  return out;
}

/**
 * If `fullFrame` exceeds MAX_WS_FRAME, split it into chunk frames (reusing
 * `from`/`target` so the relay routes each identically); otherwise return null
 * (the caller sends `fullFrame` unchanged — small frames pay zero overhead).
 */
export function splitIntoChunks(
  fullFrame: Uint8Array,
  from: PeerId,
  target: SendTarget,
): Uint8Array[] | null {
  if (fullFrame.byteLength <= MAX_WS_FRAME) return null;
  // Leave headroom for the chunk envelope JSON.
  const bodyMax = MAX_WS_FRAME - 2048;
  const n = Math.ceil(fullFrame.byteLength / bodyMax);
  const id = `${from}:${frameSeq++}`;
  const frames: Uint8Array[] = [];
  for (let i = 0; i < n; i++) {
    const body = fullFrame.subarray(i * bodyMax, Math.min((i + 1) * bodyMax, fullFrame.byteLength));
    const env = { from, target, msg: { type: CHUNK_TYPE, id, i, n } satisfies ChunkMsg };
    frames.push(buildFrame(env, body));
  }
  return frames;
}

/**
 * If `frame` is a chunk frame, return its id/index/count + body slice; else
 * null (a normal frame — parse it the usual way). `frame` is a view over the
 * received WS message bytes.
 */
export function tryPeekChunk(
  frame: Uint8Array,
): { id: string; i: number; n: number; body: Uint8Array } | null {
  if (frame.byteLength < FOUR) return null;
  const dv = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
  const envLen = dv.getUint32(0, true);
  if (envLen <= 0 || envLen > frame.byteLength - FOUR) return null;
  let env: { msg?: ChunkMsg } | null = null;
  try {
    env = JSON.parse(textDecoder.decode(frame.subarray(FOUR, FOUR + envLen)));
  } catch {
    return null;
  }
  if (!env || env.msg?.type !== CHUNK_TYPE) return null;
  const m = env.msg;
  return { id: m.id, i: m.i, n: m.n, body: frame.subarray(FOUR + envLen) };
}

/** Reassembles chunk bodies into the original frame, keyed by frame id. */
export class FrameReassembler {
  private parts = new Map<
    string,
    { n: number; got: (Uint8Array | undefined)[]; received: number; bytes: number }
  >();

  /** Feed one chunk; returns the reassembled full frame when the last arrives. */
  feed(id: string, i: number, n: number, body: Uint8Array): Uint8Array | null {
    let e = this.parts.get(id);
    if (!e) {
      e = { n, got: new Array(n), received: 0, bytes: 0 };
      this.parts.set(id, e);
    }
    if (i < 0 || i >= e.n || e.got[i]) return null; // bad/duplicate index
    // Copy: `body` is a view over the transient WS message buffer.
    e.got[i] = body.slice();
    e.received++;
    e.bytes += body.byteLength;
    if (e.received !== e.n) return null;
    this.parts.delete(id);
    const full = new Uint8Array(e.bytes);
    let off = 0;
    for (const part of e.got) {
      full.set(part!, off);
      off += part!.byteLength;
    }
    return full;
  }
}
