/**
 * OP CONFORMANCE — CPU project.
 *
 * Runs the generated op matrix on the CPU backend, comparing against an
 * INDEPENDENT JS f64 reference (see test/helpers/op-catalog.ts). No WebGPU:
 * this executes GPU-less (in CI too). The GPU-vs-CPU differential lives in
 * test/webgpu/conformance.spec.ts.
 */

import { describe } from "vitest";
import { Torchlette } from "../src/frontend/torchlette";
import { registerConformance } from "./helpers/op-catalog";

describe("op conformance (CPU)", () => {
  registerConformance({
    device: "cpu",
    makeApi: () => new Torchlette("cpu"),
  });
});
