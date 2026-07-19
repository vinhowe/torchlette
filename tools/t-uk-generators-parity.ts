/**
 * THE CRYSTAL PUSH — CAMPAIGN 1 (unrolled-K decode) — P2 GENERATOR PARITY GATE.
 *
 * The P2 build-from-IR generators the K-block adds — arg-reduce (argmax/argmin),
 * max/min reduction routing, and fusedRMSNormForward — must produce a compiled
 * stream that is byte-identical to the lowered path. Because the FULL decode
 * block does not yet fullyCover (fusedRoPE remains uncovered — P2 census), the
 * block itself stays lowered and never exercises the GENERATED stream of these
 * ops. This gate isolates them in a graph that DOES fullyCover and DOES cut over
 * to compiled replay, so the generators are validated at the optimization's
 * ACTIVATION threshold (CLAUDE.md Corollary 2), not just op-level.
 *
 * The graph — a decode-softmax-argmax shape exercising all three:
 *   n  = rmsnorm(x, w)              // fusedRMSNormForward   (new generator)
 *   m  = max(n, dim=1, keepdim)     // max reduction         (new routing)
 *   e  = exp(n - m)                 // elementwise           (covered)
 *   id = argmax(e, dim=1)           // arg-reduce            (new generator)
 * run REPEATEDLY (recurring template → 2nd exec cuts over to compiled).
 *
 * Two isolated child arms (COMPILED_PLAN default vs =0); parent asserts the ids
 * are byte-identical AND the default arm reached compiled replay (a template
 * containing these ops compiled — getCompiledStreams grew vs the lowered arm).
 *
 * Run: eval "$(tools/pick-gpu.sh)"; VULKAN_DEVICE_INDEX=$VULKAN_DEVICE_INDEX \
 *        LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH npx tsx tools/t-uk-generators-parity.ts
 */
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { getCompiledStreams } from "../src/executor/executor";
import { Torchlette } from "../src/frontend/torchlette";

const R = 3;
const D = 64;

async function childArm(): Promise<void> {
  if (!(await initWebGPU())) {
    console.error(getWebGPUInitError() || "WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(7);
  api.setStepScopedCleanup(true);
  // Persistent inputs (outside any step → survive markStep).
  const xData = Array.from({ length: R * D }, (_, i) => Math.sin(i * 0.3) * 2);
  const wData = Array.from({ length: D }, (_, i) => 0.5 + 0.01 * (i % 7));
  const x = api.tensorFromArray(xData, [R, D]);
  const w = api.tensorFromArray(wData, [D]);
  await api.markStep();

  const compiledBefore = getCompiledStreams().length;
  let lastIds: number[] = [];
  for (let it = 0; it < 6; it++) {
    const ids = api.noGrad(() => {
      const n = api.rmsnorm(x, w, 1e-6); // fusedRMSNormForward
      const m = api.max(n, { dim: 1, keepdim: true }) as ReturnType<
        typeof api.rmsnorm
      >; // max reduction
      const e = api.exp(api.sub(n, m)); // elementwise
      return api.argmax(e, { dim: 1, keepdim: false }); // arg-reduce
    });
    lastIds = (await api.cpu(ids)).map((v) => Math.round(v));
    await api.markStep();
  }
  const compiledAfter = getCompiledStreams().length;
  console.log(
    `RESULT ${JSON.stringify({ ids: lastIds, compiledDelta: compiledAfter - compiledBefore })}`,
  );
  process.exit(0);
}

function runArm(compiled: boolean): { ids: number[]; compiledDelta: number } {
  const self = fileURLToPath(import.meta.url);
  const res = spawnSync(process.execPath, ["--import", "tsx", self], {
    env: {
      ...process.env,
      UK_GEN_CHILD: "1",
      TORCHLETTE_COMPILED_PLAN: compiled ? "1" : "0",
    },
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  const line = (res.stdout || "")
    .split("\n")
    .find((l) => l.startsWith("RESULT "));
  if (!line) {
    console.error(`arm compiled=${compiled} produced no RESULT:`);
    console.error((res.stdout || "").slice(-1500));
    console.error((res.stderr || "").slice(-1500));
    process.exit(1);
  }
  return JSON.parse(line.slice("RESULT ".length));
}

function main(): void {
  if (process.env.UK_GEN_CHILD === "1") {
    void childArm();
    return;
  }
  const comp = runArm(true);
  const low = runArm(false);
  let fail = 0;
  const ok = (c: boolean, m: string) => {
    console.log(`${c ? "PASS" : "FAIL"} — ${m}`);
    if (!c) fail++;
  };
  console.log(
    `compiled arm: ids=[${comp.ids.join(",")}] compiledDelta=${comp.compiledDelta}`,
  );
  console.log(
    `lowered  arm: ids=[${low.ids.join(",")}] compiledDelta=${low.compiledDelta}`,
  );
  ok(
    comp.ids.length === low.ids.length &&
      comp.ids.every((v, i) => v === low.ids[i]),
    "argmax+max+rmsnorm generated stream byte-identical to lowered",
  );
  ok(
    comp.compiledDelta > 0,
    `default arm reached compiled replay (template with argmax/max/rmsnorm compiled; delta=${comp.compiledDelta})`,
  );
  ok(
    low.compiledDelta === 0,
    "lowered arm (COMPILED_PLAN=0) compiled nothing (control)",
  );
  console.log(
    `\n=== VERDICT: ${fail === 0 ? "PASS — P2 generators compile-parity-clean" : `FAIL (${fail})`} ===`,
  );
  process.exit(fail === 0 ? 0 : 1);
}

main();
