import { describe, expect, it } from "vitest";
import { e3m0Dequantize, e3m0Quantize } from "../../src/distributed/e3m0";

/**
 * We can't test the full PeerJS networking in vitest (needs a browser).
 * But we can test the serialization round-trip that the gossip layer uses.
 */

// Re-implement serialize/deserialize here since they're not exported
// (they're internal to gossip.ts). This tests the same logic.
interface CompressedPseudoGrad {
  paramIndex: number;
  codes: Uint32Array;
  scales: Uint8Array;
  numValues: number;
}

interface SyncMessage {
  senderId: string;
  outerStep: number;
  grads: CompressedPseudoGrad[];
}

function serialize(msg: SyncMessage): ArrayBuffer {
  const senderBytes = new TextEncoder().encode(msg.senderId);
  let totalSize = 4 + 4 + 2 + senderBytes.length;
  for (const g of msg.grads) {
    totalSize += 4 + 4 + 4 + g.codes.byteLength + 4 + g.scales.byteLength;
  }
  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);
  let offset = 0;
  view.setUint32(offset, msg.outerStep, true);
  offset += 4;
  view.setUint32(offset, msg.grads.length, true);
  offset += 4;
  view.setUint16(offset, senderBytes.length, true);
  offset += 2;
  u8.set(senderBytes, offset);
  offset += senderBytes.length;
  for (const g of msg.grads) {
    view.setUint32(offset, g.paramIndex, true);
    offset += 4;
    view.setUint32(offset, g.numValues, true);
    offset += 4;
    view.setUint32(offset, g.codes.byteLength, true);
    offset += 4;
    u8.set(
      new Uint8Array(g.codes.buffer, g.codes.byteOffset, g.codes.byteLength),
      offset,
    );
    offset += g.codes.byteLength;
    view.setUint32(offset, g.scales.byteLength, true);
    offset += 4;
    u8.set(g.scales, offset);
    offset += g.scales.byteLength;
  }
  return buf;
}

function deserialize(buf: ArrayBuffer): SyncMessage {
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);
  let offset = 0;
  const outerStep = view.getUint32(offset, true);
  offset += 4;
  const numParams = view.getUint32(offset, true);
  offset += 4;
  const senderIdLen = view.getUint16(offset, true);
  offset += 2;
  const senderId = new TextDecoder().decode(
    u8.slice(offset, offset + senderIdLen),
  );
  offset += senderIdLen;
  const grads: CompressedPseudoGrad[] = [];
  for (let i = 0; i < numParams; i++) {
    const paramIndex = view.getUint32(offset, true);
    offset += 4;
    const numValues = view.getUint32(offset, true);
    offset += 4;
    const codesLen = view.getUint32(offset, true);
    offset += 4;
    const codes = new Uint32Array(buf.slice(offset, offset + codesLen));
    offset += codesLen;
    const scalesLen = view.getUint32(offset, true);
    offset += 4;
    const scales = new Uint8Array(buf.slice(offset, offset + scalesLen));
    offset += scalesLen;
    grads.push({ paramIndex, codes, scales, numValues });
  }
  return { senderId, outerStep, grads };
}

describe("Gossip sync message serialization", () => {
  it("round-trips a sync message with multiple parameters", () => {
    // Create pseudo-gradients for 3 parameters
    const pseudoGrads = [
      new Float32Array([1, -2, 4, -8, 0.5, -0.25, 0.125, 0]),
      new Float32Array([16, 32, 64, 128, -16, -32, -64, -128]),
      new Float32Array(Array.from({ length: 24 }, (_, i) => (i - 12) * 0.1)),
    ];

    // Compress
    const grads: CompressedPseudoGrad[] = pseudoGrads.map((pg, i) => {
      const padded = new Float32Array(Math.ceil(pg.length / 8) * 8);
      padded.set(pg);
      const { codes, scales } = e3m0Quantize(padded);
      return { paramIndex: i, codes, scales, numValues: pg.length };
    });

    const msg: SyncMessage = {
      senderId: "torchlette-diloco-abc123",
      outerStep: 42,
      grads,
    };

    // Serialize → deserialize
    const buf = serialize(msg);
    const restored = deserialize(buf);

    expect(restored.senderId).toBe("torchlette-diloco-abc123");
    expect(restored.outerStep).toBe(42);
    expect(restored.grads.length).toBe(3);

    // Decompress and verify values match
    for (let i = 0; i < 3; i++) {
      expect(restored.grads[i].paramIndex).toBe(i);
      expect(restored.grads[i].numValues).toBe(pseudoGrads[i].length);

      const origCompressed = e3m0Dequantize(
        grads[i].codes,
        grads[i].scales,
        grads[i].numValues,
      );
      const roundTripped = e3m0Dequantize(
        restored.grads[i].codes,
        restored.grads[i].scales,
        restored.grads[i].numValues,
      );

      for (let j = 0; j < pseudoGrads[i].length; j++) {
        expect(roundTripped[j]).toBe(origCompressed[j]);
      }
    }
  });

  it("handles empty pseudo-gradients", () => {
    const msg: SyncMessage = {
      senderId: "peer-empty",
      outerStep: 0,
      grads: [],
    };
    const buf = serialize(msg);
    const restored = deserialize(buf);
    expect(restored.senderId).toBe("peer-empty");
    expect(restored.grads.length).toBe(0);
  });

  it("preserves message size within expected bounds", () => {
    // 124M params = ~124M floats. With E3M0: ~124M * 5/8 bytes = ~77MB.
    // For a small test: 1024 floats → 640 bytes compressed
    const pg = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) pg[i] = (Math.random() - 0.5) * 0.01;
    const padded = new Float32Array(1024);
    padded.set(pg);
    const { codes, scales } = e3m0Quantize(padded);

    const msg: SyncMessage = {
      senderId: "test",
      outerStep: 1,
      grads: [{ paramIndex: 0, codes, scales, numValues: 1024 }],
    };
    const buf = serialize(msg);

    // Header ~30 bytes + per-param overhead ~16 bytes + compressed data ~640 bytes
    expect(buf.byteLength).toBeLessThan(1024);
    expect(buf.byteLength).toBeGreaterThan(600);
  });
});
