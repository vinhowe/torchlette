/**
 * Sliding-window DISCRIMINATION gate (#64 discrimination pattern).
 *
 * The 4096 window only binds on contexts >4096 tokens, which is impractical to
 * exercise here. Instead we gate the window modifier DIFFERENTIALLY on a short
 * sequence with a small synthetic window W:
 *   - window >= seqLen  MUST match plain causal (window inactive) to ~1e-5.
 *   - window <  seqLen  MUST DIFFER from causal (older keys masked out).
 * This proves the slidingWindow maskMod is wired, keyed, and actually alters
 * the attention output exactly when (and only when) it should.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim npx tsx examples/gemma2/window-discrimination.ts
 */

import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette, type Tensor } from "../../src/frontend/torchlette";
import type { AttnModifierSpec } from "../../src/backend/types";

function maxAbs(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  const B = 1;
  const H = 2;
  const S = 40;
  const D = 64;
  const scale = 1 / Math.sqrt(D);

  const randn = (n: number) => {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * 0.5;
    return a;
  };
  const q = api.tensorFromArray(randn(B * H * S * D), [B, H, S, D]);
  const k = api.tensorFromArray(randn(B * H * S * D), [B, H, S, D]);
  const v = api.tensorFromArray(randn(B * H * S * D), [B, H, S, D]);

  const run = (mod?: AttnModifierSpec): Promise<Float32Array> => {
    const out = api.noGrad(() =>
      api.scaledDotProductAttention(q, k, v, scale, true, mod),
    );
    return out.cpu().then((b) => new Float32Array(b));
  };

  const causal: AttnModifierSpec = { maskMods: [{ kind: "causal" }] };
  const win = (w: number): AttnModifierSpec => ({
    maskMods: [{ kind: "causal" }, { kind: "slidingWindow", window: w }],
  });

  const refCausal = await run(causal);
  await api.markStep();
  // window >= S: inactive → must match causal.
  const wideWin = await run(win(S + 8));
  await api.markStep();
  // window < S: active → must differ from causal.
  const narrowWin = await run(win(8));
  await api.markStep();

  const wideDiff = maxAbs(refCausal, wideWin);
  const narrowDiff = maxAbs(refCausal, narrowWin);

  console.log(`window>=S (inactive) vs causal: maxAbs=${wideDiff.toExponential(3)} (expect ~0)`);
  console.log(`window< S (active)  vs causal: maxAbs=${narrowDiff.toExponential(3)} (expect >0)`);

  const passWide = wideDiff < 1e-4;
  const passNarrow = narrowDiff > 1e-2;
  const pass = passWide && passNarrow;
  console.log(
    pass
      ? "WINDOW DISCRIMINATION PASS (inactive matches, active differs)"
      : `WINDOW DISCRIMINATION FAIL (wide=${passWide}, narrow=${passNarrow})`,
  );
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
