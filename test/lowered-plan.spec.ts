/**
 * Tests for lowered plan infrastructure (src/engine/lowered-plan.ts)
 *
 * Covers: op classification helpers and ENCODER_COPY_OPS.
 * The buildLoweredPlanFromAnalysis function is tested end-to-end via
 * the full test suite (841 tests including GPU training).
 */

import { describe, expect, it } from "vitest";
import {
  ENCODER_COPY_OPS,
  isDataSourceOp,
  isViewOp,
} from "../src/engine/lowered-plan";

describe("LoweredPlan", () => {
  // ========================================================================
  // Op classification helpers
  // ========================================================================
  describe("isDataSourceOp", () => {
    it("returns true for data source ops", () => {
      expect(isDataSourceOp("tensorFromArray")).toBe(true);
      expect(isDataSourceOp("zeros")).toBe(true);
      expect(isDataSourceOp("full")).toBe(true);
      expect(isDataSourceOp("arange")).toBe(true);
      expect(isDataSourceOp("rand")).toBe(true);
      expect(isDataSourceOp("randn")).toBe(true);
      expect(isDataSourceOp("bernoulli")).toBe(true);
    });

    it("returns false for non-data-source ops", () => {
      expect(isDataSourceOp("add")).toBe(false);
      expect(isDataSourceOp("matmul")).toBe(false);
      expect(isDataSourceOp("relu")).toBe(false);
      expect(isDataSourceOp("reshape")).toBe(false);
    });
  });

  describe("isViewOp", () => {
    it("returns true for view ops", () => {
      expect(isViewOp("reshape")).toBe(true);
      expect(isViewOp("transpose")).toBe(true);
      expect(isViewOp("permute")).toBe(true);
      expect(isViewOp("expand")).toBe(true);
      expect(isViewOp("narrow")).toBe(true);
    });

    it("returns false for non-view ops", () => {
      expect(isViewOp("add")).toBe(false);
      expect(isViewOp("matmul")).toBe(false);
      expect(isViewOp("contiguous")).toBe(false);
      expect(isViewOp("cast")).toBe(false);
    });
  });

  describe("ENCODER_COPY_OPS", () => {
    it("contains scatterAdd", () => {
      expect(ENCODER_COPY_OPS.has("scatterAdd")).toBe(true);
    });

    it("does not contain regular compute ops", () => {
      expect(ENCODER_COPY_OPS.has("add")).toBe(false);
      expect(ENCODER_COPY_OPS.has("matmul")).toBe(false);
    });
  });
});
