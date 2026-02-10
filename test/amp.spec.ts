import { beforeEach, describe, expect, it } from "vitest";

import {
  captureRegionAMPState,
  computeInputCasts,
  computeOutputCast,
  computeSelectGatedDtype,
  createAutocastContext,
  DEFAULT_AMP_POLICY,
  DISABLED_AMP_POLICY,
  F16_ELIGIBLE_OPS,
  F32_REQUIRED_OPS,
  hashAMPPolicy,
  popAutocast,
  pushAutocast,
  type AutocastContext,
} from "../src/engine/amp";

describe("AMP Policy", () => {
  it("DEFAULT_AMP_POLICY has f16 compute and f32 accumulate", () => {
    expect(DEFAULT_AMP_POLICY.enabled).toBe(true);
    expect(DEFAULT_AMP_POLICY.computeDtype).toBe("f16");
    expect(DEFAULT_AMP_POLICY.accumulateDtype).toBe("f32");
    expect(DEFAULT_AMP_POLICY.memoryDtype).toBe("f32");
  });

  it("DISABLED_AMP_POLICY has all f32", () => {
    expect(DISABLED_AMP_POLICY.enabled).toBe(false);
    expect(DISABLED_AMP_POLICY.computeDtype).toBe("f32");
    expect(DISABLED_AMP_POLICY.accumulateDtype).toBe("f32");
  });

  it("matmul is f16-eligible", () => {
    expect(F16_ELIGIBLE_OPS.has("matmul")).toBe(true);
    expect(F16_ELIGIBLE_OPS.has("linear")).toBe(true);
    expect(F16_ELIGIBLE_OPS.has("bmm")).toBe(true);
  });

  it("reduction ops require f32", () => {
    expect(F32_REQUIRED_OPS.has("sum")).toBe(true);
    expect(F32_REQUIRED_OPS.has("mean")).toBe(true);
    expect(F32_REQUIRED_OPS.has("softmax")).toBe(true);
    expect(F32_REQUIRED_OPS.has("log_softmax")).toBe(true);
  });
});

describe("Autocast Context", () => {
  let ctx: AutocastContext;

  beforeEach(() => {
    ctx = createAutocastContext();
  });

  it("creates disabled context by default", () => {
    expect(ctx.current.enabled).toBe(false);
    expect(ctx.configStack).toHaveLength(0);
  });

  it("pushAutocast enables autocast", () => {
    pushAutocast(ctx, { enabled: true });
    expect(ctx.current.enabled).toBe(true);
    expect(ctx.configStack).toHaveLength(1);
  });

  it("popAutocast restores previous state", () => {
    pushAutocast(ctx, { enabled: true });
    expect(ctx.current.enabled).toBe(true);

    popAutocast(ctx);
    expect(ctx.current.enabled).toBe(false);
    expect(ctx.configStack).toHaveLength(0);
  });

  it("supports nested autocast blocks", () => {
    // First level
    pushAutocast(ctx, { enabled: true, deviceType: "webgpu" });
    expect(ctx.current.enabled).toBe(true);
    expect(ctx.current.deviceType).toBe("webgpu");

    // Second level (disable temporarily)
    pushAutocast(ctx, { enabled: false });
    expect(ctx.current.enabled).toBe(false);

    // Pop back to first level
    popAutocast(ctx);
    expect(ctx.current.enabled).toBe(true);
    expect(ctx.current.deviceType).toBe("webgpu");

    // Pop to initial state
    popAutocast(ctx);
    expect(ctx.current.enabled).toBe(false);
  });

  it("uses default policy when not specified", () => {
    pushAutocast(ctx, {});
    expect(ctx.current.policy).toEqual(DEFAULT_AMP_POLICY);
  });

  it("uses custom policy when specified", () => {
    const customPolicy = {
      ...DEFAULT_AMP_POLICY,
      memoryDtype: "f16" as const,
    };
    pushAutocast(ctx, { policy: customPolicy });
    expect(ctx.current.policy.memoryDtype).toBe("f16");
  });
});

describe("Select-Gated Dtype (§12)", () => {
  let ctx: AutocastContext;

  beforeEach(() => {
    ctx = createAutocastContext();
  });

  describe("with autocast disabled", () => {
    it("preserves input dtype for matmul", () => {
      const result = computeSelectGatedDtype("matmul", ["f32", "f32"], ctx);
      expect(result.outputDtype).toBe("f32");
      expect(result.needsCast).toBe(false);
      expect(result.isGated).toBe(false);
    });

    it("preserves f16 input dtype", () => {
      const result = computeSelectGatedDtype("add", ["f16", "f16"], ctx);
      expect(result.outputDtype).toBe("f16");
      expect(result.needsCast).toBe(false);
    });
  });

  describe("with autocast enabled", () => {
    beforeEach(() => {
      pushAutocast(ctx, { enabled: true });
    });

    it("matmul outputs f16 (compute dtype)", () => {
      const result = computeSelectGatedDtype("matmul", ["f32", "f32"], ctx);
      expect(result.outputDtype).toBe("f16");
      expect(result.needsCast).toBe(true);
      expect(result.isGated).toBe(true);
    });

    it("sum requires f32 output", () => {
      const result = computeSelectGatedDtype("sum", ["f16"], ctx);
      expect(result.outputDtype).toBe("f32");
      expect(result.needsCast).toBe(true);
      expect(result.sourceDtype).toBe("f16");
    });

    it("mean requires f32 output", () => {
      const result = computeSelectGatedDtype("mean", ["f16"], ctx);
      expect(result.outputDtype).toBe("f32");
      expect(result.needsCast).toBe(true);
    });

    it("elementwise ops preserve dtype when uniform", () => {
      const result = computeSelectGatedDtype("add", ["f32", "f32"], ctx);
      expect(result.outputDtype).toBe("f32");
      expect(result.needsCast).toBe(false);
    });

    it("mixed dtype elementwise uses memory dtype", () => {
      const result = computeSelectGatedDtype("add", ["f16", "f32"], ctx);
      expect(result.outputDtype).toBe("f32"); // memoryDtype
      expect(result.needsCast).toBe(true);
    });
  });
});

describe("Input Casts for AMP", () => {
  let ctx: AutocastContext;

  beforeEach(() => {
    ctx = createAutocastContext();
  });

  it("no casts when autocast disabled", () => {
    const casts = computeInputCasts("matmul", [1, 2], ["f32", "f32"], ctx);
    expect(casts).toHaveLength(0);
  });

  describe("with autocast enabled", () => {
    beforeEach(() => {
      pushAutocast(ctx, { enabled: true });
    });

    it("casts f32 inputs to f16 for eligible ops", () => {
      const casts = computeInputCasts("matmul", [1, 2], ["f32", "f32"], ctx);
      expect(casts).toHaveLength(2);
      expect(casts[0].fromDtype).toBe("f32");
      expect(casts[0].toDtype).toBe("f16");
      expect(casts[0].reason).toBe("amp_input_cast");
    });

    it("does not cast already f16 inputs", () => {
      const casts = computeInputCasts("matmul", [1, 2], ["f16", "f16"], ctx);
      expect(casts).toHaveLength(0);
    });

    it("casts mixed inputs correctly", () => {
      const casts = computeInputCasts("matmul", [1, 2], ["f32", "f16"], ctx);
      expect(casts).toHaveLength(1);
      expect(casts[0].inputNodeId).toBe(1);
      expect(casts[0].fromDtype).toBe("f32");
    });

    it("casts f16 to f32 for required ops", () => {
      const casts = computeInputCasts("sum", [1], ["f16"], ctx);
      expect(casts).toHaveLength(1);
      expect(casts[0].fromDtype).toBe("f16");
      expect(casts[0].toDtype).toBe("f32");
    });

    it("no cast needed for f32 inputs to required ops", () => {
      const casts = computeInputCasts("sum", [1], ["f32"], ctx);
      expect(casts).toHaveLength(0);
    });
  });
});

describe("Output Cast for AMP", () => {
  let ctx: AutocastContext;

  beforeEach(() => {
    ctx = createAutocastContext();
    pushAutocast(ctx, { enabled: true });
  });

  it("returns null when dtypes match", () => {
    const cast = computeOutputCast("matmul", "f16", "f16", ctx);
    expect(cast).toBeNull();
  });

  it("returns cast info when conversion needed", () => {
    const cast = computeOutputCast("matmul", "f32", "f16", ctx);
    expect(cast).not.toBeNull();
    expect(cast?.fromDtype).toBe("f32");
    expect(cast?.toDtype).toBe("f16");
    expect(cast?.reason).toBe("amp_output_cast");
  });

  it("returns null when autocast disabled", () => {
    popAutocast(ctx);
    const cast = computeOutputCast("matmul", "f32", "f16", ctx);
    expect(cast).toBeNull();
  });
});

describe("AMP Policy Hashing", () => {
  it("disabled policy has 'disabled' hash", () => {
    expect(hashAMPPolicy(DISABLED_AMP_POLICY)).toBe("disabled");
  });

  it("enabled policy includes dtype info", () => {
    const hash = hashAMPPolicy(DEFAULT_AMP_POLICY);
    expect(hash).toContain("f16");
    expect(hash).toContain("f32");
  });

  it("different policies have different hashes", () => {
    const policy1 = { ...DEFAULT_AMP_POLICY };
    const policy2 = { ...DEFAULT_AMP_POLICY, memoryDtype: "f16" as const };
    expect(hashAMPPolicy(policy1)).not.toBe(hashAMPPolicy(policy2));
  });
});

describe("Compiled Region AMP State", () => {
  it("captures AMP state at region entry", () => {
    const ctx = createAutocastContext();
    pushAutocast(ctx, { enabled: true });

    const state = captureRegionAMPState(ctx, ["f32", "f32"]);

    expect(state.ampEnabled).toBe(true);
    expect(state.inputDtypes).toEqual(["f32", "f32"]);
    expect(state.policyHash).toContain("f16");
  });

  it("captures disabled state", () => {
    const ctx = createAutocastContext();
    const state = captureRegionAMPState(ctx, ["f32"]);

    expect(state.ampEnabled).toBe(false);
    expect(state.policyHash).toBe("disabled");
  });
});
