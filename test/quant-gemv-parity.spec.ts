/**
 * Weight-only int8-grouped GEMV kernel correctness gate (task #76, phase 1).
 *
 * The cross-path numerical guard CLAUDE.md requires for any new optimized
 * execution path: the quantized-operand GEMV kernel vs a f32 control that
 * matmuls the SAME dequantized values the kernel reconstructs on-the-fly. They
 * must agree to f32 rounding noise — isolating the unpack/dequant/accumulate
 * arithmetic from quantization error. Subprocess-isolated (Dawn + engine
 * module-globals), same methodology as compiled-plan-parity.spec.ts. The probe
 * exits 0 on pass / 1 on failure and prints per-shape lines.
 *
 * GPU-less CI auto-skips (canUseWebGPU false). The probe covers G∈{64,128} and
 * shapes through lm_head-scale N=151936 (the 2D row-grid path).
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
  "probe-quant-gemv.ts",
);
const TIMEOUT = 300_000;

async function runProbe(): Promise<{ stdout: string; code: number }> {
  for (let attempt = 0; ; attempt++) {
    try {
      const { stdout } = await execFileP("npx", ["tsx", PROBE], {
        env: { ...process.env, TORCHLETTE_STRICT_GPU: "1" },
        timeout: 240_000,
        maxBuffer: 64 * 1024 * 1024,
      });
      return { stdout, code: 0 };
    } catch (e) {
      const err = e as { code?: number; stdout?: string; stderr?: string };
      // A non-zero exit WITH the PASS marker in stdout is a teardown-only
      // segfault (Dawn double-device SIGSEGV at process.exit when the vitest
      // in-process device shares the GPU) — the probe DID pass. Return its
      // stdout; the assertion keys on the marker, not the exit code.
      if (typeof err.code === "number" && err.stdout) {
        return { stdout: err.stdout, code: err.code };
      }
      // Retry true spawn/transient failures (vkCreateDevice flake).
      if (attempt >= 4) throw e;
      await new Promise((r) => setTimeout(r, 2500 * (attempt + 1)));
    }
  }
}

describe("quantized int8 GEMV kernel gate", () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = await canUseWebGPU();
  });

  it(
    "quant GEMV == f32-dequant control (kernel exactness)",
    async () => {
      if (!webgpu) return;
      const { stdout } = await runProbe();
      // The PASS marker (printed after all cases + the GPU-error check) is the
      // correctness signal. The exit code is NOT asserted: under suite load the
      // probe shares the GPU with the vitest in-process Dawn device and can
      // SIGSEGV at teardown (process.exit) despite every case passing.
      expect(stdout, stdout.slice(-800)).toContain("QUANT GEMV PROBE PASS");
      expect(stdout).not.toContain("PROBE FAIL");
    },
    TIMEOUT,
  );
});
