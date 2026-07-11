import { describe, expect, it } from "vitest";
import { GAME_LEVELS, softmaxGameLemma, welfordLemma } from "./game-levels";
import { applyMove } from "./model";

describe("NCD intrinsic-ladder levels", () => {
  it("derives every target from its known solved term with ten-percent slack", () => {
    for (const level of GAME_LEVELS) {
      expect(level.target.h).toBe(
        Math.ceil(level.solvedCost.transferByLevel.l1 * 1.1),
      );
      if (level.target.m !== undefined) {
        expect(level.target.m).toBe(
          Math.ceil(level.solvedCost.memoryByLevel.l1 * 1.1),
        );
      }
      expect(level.baselineCost.transferByLevel.l1).toBeGreaterThan(
        level.target.h,
      );
    }
  });

  it("gates exercise 1 to residency paint only", () => {
    expect(GAME_LEVELS[0].vocabulary).toEqual({
      paint: true,
      group: false,
      stream: false,
      lemma: null,
    });
  });

  it("makes Welford carried moments inspectable", () => {
    const level = GAME_LEVELS[1];
    const next = applyMove(level.baseline, welfordLemma(level.baseline));
    const inspection = next.semantic.boxes.find(
      (box) => box.id === "variance",
    )?.inspection;
    expect(inspection?.states.map((state) => state.symbol)).toEqual([
      "μ",
      "M2",
    ]);
    expect(inspection?.correction?.expression).toContain("δ²");
  });

  it("makes online-softmax state and correction inspectable", () => {
    const level = GAME_LEVELS[2];
    const next = applyMove(level.baseline, softmaxGameLemma(level.baseline));
    const inspection = next.semantic.boxes.find(
      (box) => box.id === "softmax-sum",
    )?.inspection;
    expect(inspection?.states.map((state) => state.symbol)).toEqual(["m", "ℓ"]);
    expect(inspection?.correction?.expression).toBe("exp(m_old − m_new)");
  });
});
