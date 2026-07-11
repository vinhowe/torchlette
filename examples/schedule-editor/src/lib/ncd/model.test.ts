import { describe, expect, it } from "vitest";
import attention from "../../../public/data/ncd/attention-naive.term.json";
import matmul from "../../../public/data/ncd/tiled-matmul.term.json";
import { deriveFlashAttention } from "./fa-script";
import {
  applyMove,
  deriveProjection,
  fromDiagram,
  napkinCost,
  onlineSoftmaxLemma,
  streamability,
  toDiagram,
} from "./model";
import type { NcdTerm } from "./types";

const MATMUL = matmul as NcdTerm;
const ATTENTION = attention as NcdTerm;

describe("editable NCD term", () => {
  it("round-trips term → diagram → term without information loss", () => {
    expect(fromDiagram(toDiagram(MATMUL))).toEqual(MATMUL);
    expect(fromDiagram(toDiagram(ATTENTION))).toEqual(ATTENTION);
  });

  it("accepts a decomposed stream and refuses ordinary softmax", () => {
    expect(streamability(MATMUL, "k").legal).toBe(true);
    const refused = streamability(ATTENTION, "x");
    expect(refused.legal).toBe(false);
    expect(refused.reason).toContain("softmax has no head/body decomposition");

    const admitted = applyMove(ATTENTION, onlineSoftmaxLemma(ATTENTION));
    expect(streamability(admitted, "x").legal).toBe(true);
  });

  it("reads the known tiled-matmul H/M directly from wire labels", () => {
    const cost = napkinCost(MATMUL);
    expect(cost.transferByLevel.l1).toBe(2_048);
    expect(cost.memoryByLevel.l1).toBe(2_048);
    expect(cost.memoryByLevel.l0).toBe(2_097_152);
  });

  it("reads the known naive-attention H/M directly from wire labels", () => {
    const cost = napkinCost(ATTENTION);
    expect(cost.transferByLevel.l1).toBe(14_155_776);
    expect(cost.memoryByLevel.l1).toBe(6_291_456);
    expect(cost.memoryByLevel.l0).toBe(3_145_728);
  });

  it("derives FlashAttention by gestures with the asserted final H/M", () => {
    const derived = deriveFlashAttention(ATTENTION);
    const cost = napkinCost(derived);
    expect(derived.decorations.admittedLemmas).toEqual([
      "online-softmax-rescaling",
    ]);
    expect(derived.decorations.partitions).toEqual([
      { axisId: "q", kind: "group", size: 64, label: "g_q" },
      { axisId: "x", kind: "stream", size: 32, label: "s_x" },
    ]);
    expect(cost.transferByLevel.l1).toBe(147_456);
    expect(cost.memoryByLevel.l1).toBe(98_304);
    expect(cost.memoryByLevel.l0).toBe(1_179_648);
    expect(deriveProjection(MATMUL).ok).toBe(true);
    expect(deriveProjection(derived).ok).toBe(true);
  });
});
