/**
 * Tests for lowered plan infrastructure (src/engine/lowered-plan.ts)
 *
 * Covers: LoweredPlanBuilder action recording, op classification helpers,
 * and structural correctness of the built LoweredPlan.
 */

import { describe, expect, it } from "vitest";
import type { FusedKernelRecipe } from "../src/backend/webgpu/fusion-types";
import {
  ENCODER_COPY_OPS,
  isDataSourceOp,
  isViewOp,
  LoweredPlanBuilder,
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

  // ========================================================================
  // LoweredPlanBuilder
  // ========================================================================
  describe("LoweredPlanBuilder", () => {
    it("builds empty plan", () => {
      const builder = new LoweredPlanBuilder(0);
      const plan = builder.build();

      expect(plan.actions).toEqual([]);
      expect(plan.planNodeCount).toBe(0);
    });

    it("records sequential node action", () => {
      const builder = new LoweredPlanBuilder(5);
      builder.recordNode("sequential", 3);
      const plan = builder.build();

      expect(plan.actions.length).toBe(1);
      expect(plan.actions[0]).toEqual({ kind: "sequential", nodeIndex: 3 });
      expect(plan.planNodeCount).toBe(5);
    });

    it("records view node action", () => {
      const builder = new LoweredPlanBuilder(5);
      builder.recordNode("view", 2);
      const plan = builder.build();

      expect(plan.actions[0]).toEqual({ kind: "view", nodeIndex: 2 });
    });

    it("records data-source node action", () => {
      const builder = new LoweredPlanBuilder(5);
      builder.recordNode("data-source", 0);
      const plan = builder.build();

      expect(plan.actions[0]).toEqual({ kind: "data-source", nodeIndex: 0 });
    });

    it("records prologue-skip node action", () => {
      const builder = new LoweredPlanBuilder(5);
      builder.recordNode("prologue-skip", 1);
      const plan = builder.build();

      expect(plan.actions[0]).toEqual({ kind: "prologue-skip", nodeIndex: 1 });
    });

    it("records fused kernel action", () => {
      const recipe: FusedKernelRecipe = {
        nodes: [],
        outputNodeId: 5,
        additionalOutputNodeIds: [],
        neededIntermediateNodeIds: [],
      };
      const builder = new LoweredPlanBuilder(10);
      builder.recordFused([2, 3, 4], 4, [5], [3], recipe, true);
      const plan = builder.build();

      const action = plan.actions[0] as any;
      expect(action.kind).toBe("fused");
      expect(action.coveredNodeIndices).toEqual([2, 3, 4]);
      expect(action.outputNodeIndex).toBe(4);
      expect(action.additionalOutputNodeIndices).toEqual([5]);
      expect(action.neededIntermediateNodeIndices).toEqual([3]);
      expect(action.recipe).toBe(recipe);
      expect(action.enableVectorization).toBe(true);
    });

    it("records matmul-epilogue action", () => {
      const builder = new LoweredPlanBuilder(10);
      builder.recordMatmulEpilogue(
        3,
        [3, 4, 5],
        5,
        [{ kind: "cast", toDtype: "f16" }],
        "f16",
        3,
      );
      const plan = builder.build();

      const action = plan.actions[0] as any;
      expect(action.kind).toBe("matmul-epilogue");
      expect(action.matmulNodeIndex).toBe(3);
      expect(action.coveredNodeIndices).toEqual([3, 4, 5]);
      expect(action.outputNodeIndex).toBe(5);
      expect(action.epilogueOps).toEqual([{ kind: "cast", toDtype: "f16" }]);
      expect(action.outputDtype).toBe("f16");
      expect(action.consumedCount).toBe(3);
    });

    it("records matmul-epilogue with prologues", () => {
      const builder = new LoweredPlanBuilder(10);
      builder.recordMatmulEpilogue(3, [3, 4], 4, [], "f32", 2, [
        { inputIndex: 0, castNodeIndex: 2, fromDtype: "f32", toDtype: "f16" },
      ]);
      const plan = builder.build();

      const action = plan.actions[0] as any;
      expect(action.prologues).toEqual([
        { inputIndex: 0, castNodeIndex: 2, fromDtype: "f32", toDtype: "f16" },
      ]);
    });

    it("records reduction-preamble action", () => {
      const builder = new LoweredPlanBuilder(10);
      builder.recordReductionPreamble(
        2,
        3,
        [2],
        [{ op: "mul", arity: 2 }],
        ["f32", "f32"],
        2,
      );
      const plan = builder.build();

      const action = plan.actions[0] as any;
      expect(action.kind).toBe("reduction-preamble");
      expect(action.preambleNodeIndex).toBe(2);
      expect(action.reductionNodeIndex).toBe(3);
      expect(action.chainOps).toEqual([{ op: "mul", arity: 2 }]);
      expect(action.consumedCount).toBe(2);
    });

    it("records reduction-epilogue action", () => {
      const builder = new LoweredPlanBuilder(10);
      builder.recordReductionEpilogue(
        3,
        [3, 4],
        4,
        [{ kind: "unary", op: "relu" }],
        "f32",
        2,
      );
      const plan = builder.build();

      const action = plan.actions[0] as any;
      expect(action.kind).toBe("reduction-epilogue");
      expect(action.reductionNodeIndex).toBe(3);
      expect(action.outputNodeIndex).toBe(4);
    });

    it("records reduction-fusion action", () => {
      const builder = new LoweredPlanBuilder(10);
      builder.recordReductionFusion(
        [2],
        3,
        [4],
        4,
        [{ op: "mul", arity: 2 }],
        ["f32", "f32"],
        [{ kind: "unary", op: "relu" }],
        "f32",
        3,
        false,
      );
      const plan = builder.build();

      const action = plan.actions[0] as any;
      expect(action.kind).toBe("reduction-fusion");
      expect(action.preambleNodeIndices).toEqual([2]);
      expect(action.reductionNodeIndex).toBe(3);
      expect(action.epilogueNodeIndices).toEqual([4]);
      expect(action.isMean).toBe(false);
    });

    it("records adam-batch action", () => {
      const builder = new LoweredPlanBuilder(100);
      builder.recordAdamBatch([10, 11, 12, 13]);
      const plan = builder.build();

      const action = plan.actions[0] as any;
      expect(action.kind).toBe("adam-batch");
      expect(action.nodeIndices).toEqual([10, 11, 12, 13]);
    });

    it("records compound pattern action", () => {
      const builder = new LoweredPlanBuilder(20);
      builder.recordCompound("softmax", [5, 6, 7, 8, 9], 9, 1);
      const plan = builder.build();

      const action = plan.actions[0] as any;
      expect(action.kind).toBe("compound");
      expect(action.name).toBe("softmax");
      expect(action.coveredNodeIndices).toEqual([5, 6, 7, 8, 9]);
      expect(action.dim).toBe(1);
    });

    it("records reclaim action", () => {
      const builder = new LoweredPlanBuilder(10);
      builder.recordReclaim();
      const plan = builder.build();

      expect(plan.actions[0]).toEqual({ kind: "reclaim" });
    });

    it("preserves action order in complex sequence", () => {
      const builder = new LoweredPlanBuilder(20);
      builder.recordNode("data-source", 0);
      builder.recordNode("data-source", 1);
      builder.recordNode("sequential", 2);
      builder.recordNode("view", 3);
      builder.recordMatmulEpilogue(4, [4, 5], 5, [], "f32", 2);
      builder.recordReclaim();
      builder.recordFused(
        [6, 7],
        7,
        [],
        [],
        {
          nodes: [],
          outputNodeId: 7,
          additionalOutputNodeIds: [],
          neededIntermediateNodeIds: [],
        },
        false,
      );
      builder.recordAdamBatch([10, 11]);

      const plan = builder.build();
      const kinds = plan.actions.map((a) => a.kind);
      expect(kinds).toEqual([
        "data-source",
        "data-source",
        "sequential",
        "view",
        "matmul-epilogue",
        "reclaim",
        "fused",
        "adam-batch",
      ]);
      expect(plan.planNodeCount).toBe(20);
    });
  });
});
