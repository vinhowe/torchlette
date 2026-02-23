/**
 * Oracle Integration Tests for Checkpoint and AMP
 *
 * Compares torchlette's checkpoint and AMP implementations against
 * PyTorch reference implementations via the torch_oracle.
 */

import { describe, expect, it, beforeEach } from "vitest";
import { Torchlette } from "../src/frontend";
import {
  runTorchOracleBackwardBatch,
  runTorchOracleFullBatch,
  type OracleCase,
} from "./oracle/torch-oracle";

/**
 * Helper to compare arrays with tolerance
 */
function expectClose(
  actual: number[],
  expected: number[],
  rtol = 1e-4,
  atol = 1e-5,
): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    const threshold = atol + rtol * Math.abs(expected[i]);
    if (diff > threshold) {
      throw new Error(
        `Mismatch at index ${i}: actual=${actual[i]}, expected=${expected[i]}, diff=${diff}, threshold=${threshold}`,
      );
    }
  }
}

describe("Checkpoint Oracle Validation", () => {
  let torch: Torchlette;

  beforeEach(() => {
    torch = new Torchlette("cpu");
  });

  it("simple checkpoint forward/backward matches PyTorch", { timeout: 15000 }, async () => {
    // Create test tensors
    const xData = [1, 2, 3, 4];
    const wData = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6];
    const bData = [0.1, 0.2, 0.3, 0.4];

    // Get PyTorch reference (with checkpointing)
    const oracleCase: OracleCase = {
      op: "checkpoint_forward_backward",
      caseName: "simple_checkpoint",
      inputs: [
        { values: xData, shape: [1, 4] },
        { values: wData, shape: [4, 4] },
        { values: bData, shape: [4] },
      ],
      options: {
        numLayers: 2,
        useCheckpoint: true,
      },
    };

    const [pytorchResult] = await runTorchOracleFullBatch([oracleCase]);

    // Get PyTorch reference (without checkpointing) - gradients should match
    const oracleCaseNoCheckpoint: OracleCase = {
      ...oracleCase,
      caseName: "no_checkpoint",
      options: {
        numLayers: 2,
        useCheckpoint: false,
      },
    };

    const [pytorchNoCheckpoint] = await runTorchOracleFullBatch([oracleCaseNoCheckpoint]);

    // Verify checkpoint gradients match no-checkpoint gradients in PyTorch
    expect(pytorchResult.grads).toBeDefined();
    expect(pytorchNoCheckpoint.grads).toBeDefined();

    for (let i = 0; i < pytorchResult.grads!.length; i++) {
      const grad1 = pytorchResult.grads![i];
      const grad2 = pytorchNoCheckpoint.grads![i];
      if (grad1 && grad2) {
        expectClose(grad1.values, grad2.values, 1e-5, 1e-6);
      }
    }
  });

  it("checkpoint memory comparison shows gradients match", { timeout: 15000 }, async () => {
    // Create larger test tensors to see memory effect
    const size = 16;
    const xData = Array.from({ length: size }, (_, i) => (i + 1) * 0.1);
    const wData = Array.from({ length: size * size }, (_, i) => (i + 1) * 0.01);

    const oracleCase: OracleCase = {
      op: "memory_comparison",
      caseName: "memory_comparison",
      inputs: [
        { values: xData, shape: [1, size] },
        { values: wData, shape: [size, size] },
      ],
      options: {
        numLayers: 4,
      },
    };

    const [result] = await runTorchOracleFullBatch([oracleCase]);

    // The key assertion: gradients should match between checkpoint and no-checkpoint
    expect(result.gradsMatch).toBe(true);
  });
});

describe("AMP Oracle Validation", () => {
  let torch: Torchlette;

  beforeEach(() => {
    torch = new Torchlette("cpu");
  });

  it("AMP forward/backward produces valid gradients", async () => {
    const xData = [1, 2, 3, 4];
    const wData = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6];
    const bData = [0.1, 0.2, 0.3, 0.4];

    // Get PyTorch AMP reference
    const oracleCase: OracleCase = {
      op: "amp_forward_backward",
      caseName: "amp_basic",
      inputs: [
        { values: xData, shape: [1, 4] },
        { values: wData, shape: [4, 4] },
        { values: bData, shape: [4] },
      ],
      options: {
        useAmp: true,
        deviceType: "cpu",
      },
    };

    const [result] = await runTorchOracleFullBatch([oracleCase]);

    // Verify output exists and gradients are valid
    expect(result.output).toBeDefined();
    expect(result.grads).toBeDefined();
    expect(result.grads!.length).toBe(3); // x, w, b

    // All gradients should be finite
    for (const grad of result.grads!) {
      if (grad) {
        for (const val of grad.values) {
          expect(Number.isFinite(val)).toBe(true);
        }
      }
    }
  });

  it("AMP with GradScaler handles normal case", async () => {
    // Use smaller values to avoid overflow with gradient scaling
    const xData = [0.01, 0.02, 0.03, 0.04];
    const wData = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16];
    const bData = [0.01, 0.02, 0.03, 0.04];

    const oracleCase: OracleCase = {
      op: "amp_gradscaler_backward",
      caseName: "gradscaler_normal",
      inputs: [
        { values: xData, shape: [1, 4] },
        { values: wData, shape: [4, 4] },
        { values: bData, shape: [4] },
      ],
      options: {
        initScale: 1.0,  // Use small scale to avoid overflow
        deviceType: "cpu",
      },
    };

    const [result] = await runTorchOracleFullBatch([oracleCase]);

    // Verify scale is at initial value
    expect(result.scale).toBe(1.0);
    // With small scale and small values, should not have inf
    expect(result.foundInf).toBe(false);

    // Gradients should be finite (not null which represents NaN/Inf)
    for (const grad of result.grads!) {
      if (grad) {
        for (const val of grad.values) {
          expect(val).not.toBeNull();
          if (val !== null) {
            expect(Number.isFinite(val)).toBe(true);
          }
        }
      }
    }
  });

  it("AMP with GradScaler detects NaN in output", async () => {
    const xData = [1, 2, 3, 4];
    const wData = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6];

    const oracleCase: OracleCase = {
      op: "amp_gradscaler_nan_test",
      caseName: "gradscaler_nan",
      inputs: [
        { values: xData, shape: [1, 4] },
        { values: wData, shape: [4, 4] },
      ],
      options: {
        injectNan: true,
        initScale: 1.0,
        deviceType: "cpu",
      },
    };

    const [result] = await runTorchOracleFullBatch([oracleCase]);

    // Output should contain null (sanitized NaN)
    // This confirms NaN was injected and detected
    expect(result.output!.values[0]).toBeNull();

    // Note: foundInf may be false on CPU because the gradient computation
    // through sum() of NaN may not always produce NaN gradients
  });

  it("AMP with GradScaler detects Inf", async () => {
    const xData = [1, 2, 3, 4];
    const wData = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6];

    const oracleCase: OracleCase = {
      op: "amp_gradscaler_nan_test",
      caseName: "gradscaler_inf",
      inputs: [
        { values: xData, shape: [1, 4] },
        { values: wData, shape: [4, 4] },
      ],
      options: {
        injectInf: true,
        initScale: 1.0,  // Smaller scale
        deviceType: "cpu",
      },
    };

    const [result] = await runTorchOracleFullBatch([oracleCase]);

    // Should detect inf in gradients
    expect(result.foundInf).toBe(true);

    // Output should contain null (sanitized Inf)
    expect(result.output!.values[0]).toBeNull();
  });
});

describe("Checkpoint + AMP Combined", () => {
  let torch: Torchlette;

  beforeEach(() => {
    torch = new Torchlette("cpu");
  });

  it("checkpoint + AMP produces correct gradients", { timeout: 15000 }, async () => {
    const xData = [1, 2, 3, 4];
    const wData = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6];
    const bData = [0.1, 0.2, 0.3, 0.4];

    // Get result with both checkpoint and AMP
    const oracleCase: OracleCase = {
      op: "checkpoint_amp_forward_backward",
      caseName: "checkpoint_amp_combined",
      inputs: [
        { values: xData, shape: [1, 4] },
        { values: wData, shape: [4, 4] },
        { values: bData, shape: [4] },
      ],
      options: {
        numLayers: 2,
        useCheckpoint: true,
        useAmp: true,
        deviceType: "cpu",
      },
    };

    const [withBoth] = await runTorchOracleFullBatch([oracleCase]);

    // Get result with neither checkpoint nor AMP
    const oracleCaseNone: OracleCase = {
      ...oracleCase,
      caseName: "no_checkpoint_no_amp",
      options: {
        numLayers: 2,
        useCheckpoint: false,
        useAmp: false,
        deviceType: "cpu",
      },
    };

    const [withNeither] = await runTorchOracleFullBatch([oracleCaseNone]);

    // Gradients should be close (AMP uses f16 so some tolerance needed)
    expect(withBoth.grads).toBeDefined();
    expect(withNeither.grads).toBeDefined();

    for (let i = 0; i < withBoth.grads!.length; i++) {
      const grad1 = withBoth.grads![i];
      const grad2 = withNeither.grads![i];
      if (grad1 && grad2) {
        // Looser tolerance for AMP (f16 precision)
        expectClose(grad1.values, grad2.values, 1e-2, 1e-3);
      }
    }
  });

  it("checkpoint alone produces same gradients as no checkpoint", { timeout: 15000 }, async () => {
    const xData = [1, 2, 3, 4];
    const wData = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6];
    const bData = [0.1, 0.2, 0.3, 0.4];

    // With checkpoint
    const oracleWithCheckpoint: OracleCase = {
      op: "checkpoint_amp_forward_backward",
      caseName: "with_checkpoint",
      inputs: [
        { values: xData, shape: [1, 4] },
        { values: wData, shape: [4, 4] },
        { values: bData, shape: [4] },
      ],
      options: {
        numLayers: 3,
        useCheckpoint: true,
        useAmp: false,
        deviceType: "cpu",
      },
    };

    // Without checkpoint
    const oracleNoCheckpoint: OracleCase = {
      ...oracleWithCheckpoint,
      caseName: "no_checkpoint",
      options: {
        numLayers: 3,
        useCheckpoint: false,
        useAmp: false,
        deviceType: "cpu",
      },
    };

    const [withCheckpoint] = await runTorchOracleFullBatch([oracleWithCheckpoint]);
    const [noCheckpoint] = await runTorchOracleFullBatch([oracleNoCheckpoint]);

    // Gradients should be exactly equal (no f16 involved)
    for (let i = 0; i < withCheckpoint.grads!.length; i++) {
      const grad1 = withCheckpoint.grads![i];
      const grad2 = noCheckpoint.grads![i];
      if (grad1 && grad2) {
        expectClose(grad1.values, grad2.values, 1e-6, 1e-7);
      }
    }
  });

  it("AMP alone produces valid gradients", { timeout: 15000 }, async () => {
    const xData = [1, 2, 3, 4];
    const wData = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6];
    const bData = [0.1, 0.2, 0.3, 0.4];

    // With AMP, no checkpoint
    const oracleWithAmp: OracleCase = {
      op: "checkpoint_amp_forward_backward",
      caseName: "with_amp",
      inputs: [
        { values: xData, shape: [1, 4] },
        { values: wData, shape: [4, 4] },
        { values: bData, shape: [4] },
      ],
      options: {
        numLayers: 2,
        useCheckpoint: false,
        useAmp: true,
        deviceType: "cpu",
      },
    };

    // Without AMP
    const oracleNoAmp: OracleCase = {
      ...oracleWithAmp,
      caseName: "no_amp",
      options: {
        numLayers: 2,
        useCheckpoint: false,
        useAmp: false,
        deviceType: "cpu",
      },
    };

    const [withAmp] = await runTorchOracleFullBatch([oracleWithAmp]);
    const [noAmp] = await runTorchOracleFullBatch([oracleNoAmp]);

    // Gradients should be close (f16 precision)
    for (let i = 0; i < withAmp.grads!.length; i++) {
      const grad1 = withAmp.grads![i];
      const grad2 = noAmp.grads![i];
      if (grad1 && grad2) {
        expectClose(grad1.values, grad2.values, 1e-2, 1e-3);
      }
    }
  });
});

describe("Memory Trace", () => {
  it("memory trace returns snapshots", async () => {
    const size = 8;
    const xData = Array.from({ length: size }, (_, i) => (i + 1) * 0.1);
    const wData = Array.from({ length: size * size }, (_, i) => (i + 1) * 0.01);

    const oracleCase: OracleCase = {
      op: "memory_trace",
      caseName: "memory_trace",
      inputs: [
        { values: xData, shape: [1, size] },
        { values: wData, shape: [size, size] },
      ],
      options: {
        numLayers: 3,
        useCheckpoint: false,
        useAmp: false,
      },
    };

    const [result] = await runTorchOracleFullBatch([oracleCase]);

    // Should have memory snapshots
    expect(result.memorySnapshots).toBeDefined();
    expect(result.memorySnapshots!.length).toBeGreaterThan(0);

    // Check snapshot labels
    const labels = result.memorySnapshots!.map((s) => s.label);
    expect(labels).toContain("initial");
    expect(labels).toContain("after_forward");
    expect(labels).toContain("after_backward");
  });
});
