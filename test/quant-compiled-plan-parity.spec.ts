/**
 * Storage-format (StorageFormat) COMPILED-PLAN gate — task #93 follow-up.
 *
 * The operand gate (quant-operand-parity.spec.ts) runs with the compiled +
 * generated plans DISABLED (TORCHLETTE_COMPILED_PLAN/GENERATED_PLAN=0), so it
 * never exercised the DEFAULT path — which is exactly where the "int8 decode
 * not engaging" bug lived (CLAUDE.md "Corollary 2": the differential must cross
 * the optimization's ACTIVATION threshold). The stage-4 build-from-IR capture
 * derived matmul-input metadata WITHOUT the packed format, so it planned a
 * plain f16 tiled matmul over the packed buffer, claimed "fully covered", and
 * ran a compiled plan that NEVER touched the seam → NaN, quant GEMV bypassed.
 *
 * THIS gate runs the M=1 quantized api.linear REPEATEDLY under the DEFAULT
 * plans (compiled + generated ON) and asserts the trajectory matches the f32
 * control AND the quant GEMV route engaged. Subprocess-isolated (Dawn + engine
 * module globals), same methodology as quant-operand-parity.spec.ts. GPU-less
 * CI auto-skips.
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
  "probe-quant-compiled.ts",
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
      if (
        typeof err.code === "number" &&
        err.stdout?.includes("QUANT COMPILED PROBE PASS")
      ) {
        return { stdout: err.stdout };
      }
      if (attempt >= 4) throw e;
      await new Promise((r) => setTimeout(r, 2500 * (attempt + 1)));
    }
  }
}

describe("quantized compiled-plan gate", () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = await canUseWebGPU();
  });

  it(
    "api.linear(quantized weight) under DEFAULT compiled+generated plans == f32 control + seam engaged",
    async () => {
      if (!webgpu) return;
      const { stdout } = await runProbe();
      expect(stdout, stdout.slice(-800)).toContain("QUANT COMPILED PROBE PASS");
      expect(stdout).not.toContain("PROBE FAIL");
    },
    TIMEOUT,
  );
});
