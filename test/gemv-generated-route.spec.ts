/**
 * GEMV (M=1 decode) route-engagement COMPILED-PLAN gate — task #95 follow-up.
 *
 * #95 added a routing fact (inputCast admits GEMV NT) to matmul variant
 * selection. The route was validated on the LOWERED path (kernel differential,
 * dispatch counter) — but that counter (`getGemvDispatchCount`, in
 * matmul/dispatch.ts) only ticks in dispatchTiledMatmul, which the
 * compiled/generated-plan replay BYPASSES. Once a decode template cuts over to
 * the generated stream, the lowered counter reads 0 even when the GEMV route
 * engages perfectly — so a #93/#95-class bypass at a compiled-plan DECISION
 * POINT (build-from-IR capture / directive detection / generator) would be
 * invisible to any lowered-counter test. This is the third Corollary-2 catch
 * (CLAUDE.md): the differential must cross the optimization's ACTIVATION
 * threshold.
 *
 * THIS gate runs an M=1 mixed f32×f16 api.linear (the browser decode projection
 * shape) REPEATEDLY under the DEFAULT compiled+generated plans and asserts the
 * GENERATED stream actually baked a GEMV dispatch (getGeneratedGemvDispatchCount
 * > 0) AND the trajectory matches the f32 control. It reproduces the bypass in
 * the negative direction (TORCHLETTE_GEMV=0 ⇒ generated GEMV count 0).
 * Subprocess-isolated (Dawn + engine module globals). GPU-less CI auto-skips.
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
  "probe-gemv-route-compiled.ts",
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
        err.stdout?.includes("GEMV COMPILED PROBE PASS")
      ) {
        return { stdout: err.stdout };
      }
      if (attempt >= 4) throw e;
      await new Promise((r) => setTimeout(r, 2500 * (attempt + 1)));
    }
  }
}

describe("GEMV generated-route gate", () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = await canUseWebGPU();
  });

  it(
    "M=1 decode api.linear under DEFAULT compiled+generated plans bakes a GEMV dispatch + matches f32 control",
    async () => {
      if (!webgpu) return;
      const { stdout } = await runProbe();
      expect(stdout, stdout.slice(-800)).toContain("GEMV COMPILED PROBE PASS");
      expect(stdout).not.toContain("PROBE FAIL");
      // The load-bearing signal: the generated stream baked ≥1 GEMV dispatch
      // (the lowered counter alone would read 0 post-cutover).
      const m = stdout.match(/generated=(\d+)/);
      expect(m, stdout.slice(-800)).not.toBeNull();
      expect(Number(m![1])).toBeGreaterThan(0);
    },
    TIMEOUT,
  );
});
