import { describe, expect, it } from "vitest";
import matmulState from "../../public/data/schedule-states/tiled-matmul.json";
import {
  calculateStaticCost,
  rooflineEstimate,
  sharedMemoryBytes,
} from "./cost-model";
import {
  cloneScheduleState,
  FALLBACK_DEVICE,
  type ScheduleState,
} from "./schedule-state";

describe("static schedule cost model", () => {
  it("recognizes a matmul tile that exactly fills workgroup storage", () => {
    const state = cloneScheduleState(matmulState as ScheduleState);
    state.decorations.tileSizes = {
      ...state.decorations.tileSizes,
      m: 64,
      n: 64,
      k: 32,
    };
    expect(sharedMemoryBytes(state)).toBe(16_384);
    expect(
      calculateStaticCost(state, FALLBACK_DEVICE).sharedMemoryUtilization,
    ).toBe(1);
  });

  it("reports a thread-limited occupancy case", () => {
    const state = cloneScheduleState(matmulState as ScheduleState);
    state.decorations.workgroup = { x: 256, y: 1, z: 1 };
    const device = {
      ...FALLBACK_DEVICE,
      maxResidentInvocationsPerComputeUnit: 512,
      maxResidentWorkgroupsPerComputeUnit: 8,
      sharedMemoryPerComputeUnit: 65_536,
    };
    const cost = calculateStaticCost(state, device);
    expect(cost.residentWorkgroupsProxy).toBe(2);
    expect(cost.occupancyLimiter).toBe("threads");
    expect(cost.occupancyProxy).toBe(1);
  });

  it("separates bandwidth-bound and compute-bound roofline points", () => {
    const device = { peakBandwidthGBs: 0.0001, peakFlopsTFLOPs: 0.000001 };
    const bandwidth = rooflineEstimate(1_000, 1_000, device);
    const compute = rooflineEstimate(10_000, 100, device);
    expect(bandwidth.ridgePoint).toBe(10);
    expect(bandwidth.rooflineBound).toBe("bandwidth");
    expect(compute.rooflineBound).toBe("compute");
  });
});
