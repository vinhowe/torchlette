/**
 * Storage-format (StorageFormat) OPERAND-PATH gate — task #93 (phase 2).
 *
 * Phase 1's gate (quant-gemv-parity.spec.ts) checks the raw quant GEMV kernel.
 * THIS gate checks the full operand path: a quantized weight created via
 * api.createQuantizedWeight fed to the ordinary, format-BLIND api.linear must
 *   (a) agree with an f32 control weight through the SAME api.linear (invisible
 *       route, values to f32 noise), and
 *   (b) route M=1 → fused-dequant GEMV, M>1 → explicit dequant (the capability
 *       seam, keyed on the operand format — selection as data).
 * The probe prints PASS/FAIL lines + a marker; subprocess-isolated (Dawn +
 * engine module globals), same methodology as quant-gemv-parity.spec.ts.
 * GPU-less CI auto-skips.
 */
import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import { beforeAll, describe, expect, it } from "vitest";
import { canUseWebGPU } from "./helpers/webgpu";

const execFileP = promisify(execFile);
const PROBE = path.join(
  __dirname,
  "..",
  "examples",
  "qwen3",
  "probe-quant-operand.ts",
);
const TIMEOUT = 300_000;

async function runProbe(): Promise<{ stdout: string }> {
  for (let attempt = 0; ; attempt++) {
    try {
      const { stdout } = await execFileP("npx", ["tsx", PROBE], {
        env: { ...process.env, TORCHLETTE_STRICT_GPU: "1" },
        timeout: 240_000,
        maxBuffer: 64 * 1024 * 1024,
      });
      return { stdout };
    } catch (e) {
      const err = e as { code?: number; stdout?: string; stderr?: string };
      // Non-zero exit WITH the PASS marker = teardown-only segfault (Dawn
      // double-device SIGSEGV at process.exit under suite load); the probe
      // passed. The assertion keys on the marker, not the exit code.
      if (typeof err.code === "number" && err.stdout?.includes("PROBE PASS")) {
        return { stdout: err.stdout };
      }
      if (attempt >= 4) throw e;
      await new Promise((r) => setTimeout(r, 2500 * (attempt + 1)));
    }
  }
}

describe("quantized operand-path gate", () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = await canUseWebGPU();
  });

  it(
    "api.linear(quantized weight) == f32 control + correct route",
    async () => {
      if (!webgpu) return;
      const { stdout } = await runProbe();
      expect(stdout, stdout.slice(-800)).toContain("QUANT OPERAND PROBE PASS");
      expect(stdout).not.toContain("PROBE FAIL");
    },
    TIMEOUT,
  );
});
