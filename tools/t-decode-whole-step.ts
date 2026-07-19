/**
 * EVEREST P4a — THE DECODE DISCRIMINATOR.
 *
 * Question (docs/step-function-compiler-design.md P4a charter): does a DECODE
 * step run under TORCHLETTE_WHOLE_STEP as a whole-step-COMPILED function, the
 * way a training step does — such that the whole-step compiler could SUBSUME
 * the step tape's decode consumer (and P4b could delete step-tape*.ts)?
 *
 * The whole-step scope's ONLY mechanism is deferring backward's grad-write
 * force to the boundary (`_deferBackwardForce()` → autograd.ts / checkpoint.ts,
 * ALL backward-path consumers). A decode step is a `noGrad` forward — no
 * backward, no grad force, so the scope has nothing to defer. This harness
 * MEASURES that: it runs a real KV-cached decode loop under two arms in
 * isolated child processes (the flag is a module-load const) and compares:
 *   - tokens (byte-identical greedy stream)
 *   - lowered-plan template count (does whole-step change the plan structure?)
 *   - late-step ms/token (does whole-step change decode speed?)
 *
 * Arm `plain`      : decode, TORCHLETTE_WHOLE_STEP unset (the tape/per-plan path).
 * Arm `whole-step` : each decode forward wrapped in api.wholeStep(...), flag on.
 *
 * VERDICT A (subsumable): whole-step arm traces+compiles the decode step as a
 *   distinct whole-step function, tokens byte-identical, tok/s >= plain.
 * VERDICT B (blocks): whole-step is a structural no-op for decode (identical
 *   templates / tokens / tok/s) → decode is NOT subsumed → the tape survives
 *   decode-scoped in P4b.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *        npx tsx tools/t-decode-whole-step.ts [steps=16]
 */
import { getWebGPUInitError, initWebGPU, destroyWebGPU } from "../src/backend/webgpu";
import { debugTemplateCount } from "../src/executor/executor";
import { Torchlette } from "../src/frontend/torchlette";
import { DISTILGPT2_CONFIG, GPT2 } from "../examples/gpt2/model";
import type { KVCache } from "../examples/gpt2/model";

type Arm = "plain" | "whole-step";
interface ArmResult {
  tokens: number[];
  afterPrefill: number;
  afterDecode: number;
  steadyGrowth: number;
  lateMsPerTok: number;
  tokPerSec: number;
}

const STEPS = Number(process.argv[2] ?? 16);
const PROMPT = [40, 716, 257];

async function run(arm: Arm): Promise<ArmResult> {
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = new GPT2(api, { ...DISTILGPT2_CONFIG });

  let pastKVs: KVCache[] | undefined;
  let posOffset = 0;

  // Prefill (never whole-step-wrapped — the discriminator is about decode).
  const firstLogits = api.noGrad(() => {
    const idx = api.tensorFromArray(PROMPT, [1, PROMPT.length]);
    const { logits, presentKVs } = model.forwardCached(idx, undefined, 0);
    pastKVs = presentKVs;
    return logits;
  });
  await firstLogits.cpu();
  firstLogits.dispose();
  posOffset = PROMPT.length;
  await api.markStep();
  const afterPrefill = debugTemplateCount();

  api.setStepScopedCleanup(true);
  const perStepGrowth: number[] = [];
  const walls: number[] = [];
  const tokens: number[] = [];
  let tok = 1;

  for (let i = 0; i < STEPS; i++) {
    const before = debugTemplateCount();
    const t0 = performance.now();
    // The DECODE STEP. Under the whole-step arm the forward runs inside the
    // whole-step scope exactly as a training step's body does.
    const logits =
      arm === "whole-step"
        ? await api.wholeStep(() =>
            api.noGrad(() => {
              const idx = api.tensorFromArray([tok], [1, 1]);
              const { logits: lg, presentKVs } = model.forwardCached(idx, pastKVs, posOffset);
              pastKVs = presentKVs;
              return lg;
            }),
          )
        : api.noGrad(() => {
            const idx = api.tensorFromArray([tok], [1, 1]);
            const { logits: lg, presentKVs } = model.forwardCached(idx, pastKVs, posOffset);
            pastKVs = presentKVs;
            return lg;
          });
    const data = new Float32Array(await logits.cpu());
    logits.dispose();
    let best = 0;
    for (let v = 1; v < model.config.vocabSize; v++) if (data[v] > data[best]) best = v;
    tok = best;
    tokens.push(best);
    posOffset += 1;
    await api.markStep();
    walls.push(performance.now() - t0);
    perStepGrowth.push(debugTemplateCount() - before);
  }

  const afterDecode = debugTemplateCount();
  const late = walls.slice(Math.floor(walls.length / 2));
  const lateMsPerTok = late.reduce((a, b) => a + b, 0) / late.length;
  return {
    tokens,
    afterPrefill,
    afterDecode,
    steadyGrowth: perStepGrowth.slice(3).reduce((a, b) => a + b, 0),
    lateMsPerTok,
    tokPerSec: 1000 / lateMsPerTok,
  };
}

async function runArmInChild(arm: Arm): Promise<ArmResult> {
  const { execFileSync } = await import("node:child_process");
  // WHOLE_STEP is DEFAULT-ON since P4a Stage 2. BOTH arms inherit the default —
  // the ONLY variable is the `api.wholeStep(...)` WRAP around the decode
  // forward (arm `whole-step`) vs a bare decode forward (arm `plain`). The flag
  // value is irrelevant to `plain` (no scope entered ⇒ nothing to defer), so
  // isolating the wrap is the honest discriminator under the graduated default.
  const env: Record<string, string> = { ...process.env, DWS_ARM: arm };
  const out = execFileSync(process.execPath, ["--import", "tsx", import.meta.filename], {
    env,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
    maxBuffer: 64 * 1024 * 1024,
  });
  const line = out.split("\n").find((l) => l.startsWith("=== ARM-RESULT === "));
  if (!line) throw new Error(`arm ${arm}: no result line`);
  return JSON.parse(line.slice("=== ARM-RESULT === ".length)) as ArmResult;
}

async function main() {
  const armEnv = process.env.DWS_ARM as Arm | undefined;
  if (armEnv === "plain" || armEnv === "whole-step") {
    if (!(await initWebGPU())) {
      console.error(getWebGPUInitError() || "WebGPU init failed");
      process.exit(1);
    }
    const r = await run(armEnv);
    console.log(`=== ARM-RESULT === ${JSON.stringify(r)}`);
    destroyWebGPU();
    process.exit(0);
  }

  console.log(`=== DECODE DISCRIMINATOR (distilgpt2 KV-decode, ${STEPS} steps) ===`);
  const plain = await runArmInChild("plain");
  const whole = await runArmInChild("whole-step");

  const tokensMatch =
    plain.tokens.length === whole.tokens.length &&
    plain.tokens.every((t, i) => t === whole.tokens[i]);

  console.log(`\nplain      : templates ${plain.afterPrefill}->${plain.afterDecode} (steady growth ${plain.steadyGrowth}), ${plain.lateMsPerTok.toFixed(2)} ms/tok (${plain.tokPerSec.toFixed(1)} tok/s)`);
  console.log(`whole-step : templates ${whole.afterPrefill}->${whole.afterDecode} (steady growth ${whole.steadyGrowth}), ${whole.lateMsPerTok.toFixed(2)} ms/tok (${whole.tokPerSec.toFixed(1)} tok/s)`);
  console.log(`\ntokens byte-identical : ${tokensMatch}`);
  console.log(`template-count identical: ${plain.afterDecode === whole.afterDecode}`);
  const speedRatio = whole.tokPerSec / plain.tokPerSec;
  console.log(`tok/s ratio (whole/plain): ${speedRatio.toFixed(3)}`);

  const wholeStepIsNoop =
    tokensMatch &&
    plain.afterDecode === whole.afterDecode &&
    plain.steadyGrowth === whole.steadyGrowth;
  console.log(
    `\nVERDICT: ${
      wholeStepIsNoop
        ? "B — whole-step is a STRUCTURAL NO-OP for decode (identical templates+tokens+steady-growth). The decode step does NOT become a distinct whole-step-compiled function; the whole-step scope's sole mechanism (defer backward grad-force) never fires without a backward. The tape's decode consumer is NOT subsumable by whole-step → tape survives DECODE-SCOPED in P4b."
        : "A — whole-step changed decode structure; investigate subsumability."
    }`,
  );
  console.log(`=== DISCRIMINATOR-STATS === ${JSON.stringify({ tokensMatch, plainTemplates: plain.afterDecode, wholeTemplates: whole.afterDecode, speedRatio, verdict: wholeStepIsNoop ? "B" : "A" })}`);
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
