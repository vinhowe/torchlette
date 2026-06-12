/**
 * Minimal repro hunt: checkpoint + autocast + reshape.
 * A checkpointed region containing reshape + f16 casts (the transformer
 * block shape), gradients compared against the plain path. Reports first
 * configuration that diverges or crashes.
 */
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { checkpoint } from "../src/nn/checkpoint";
import type { Tensor } from "../src/frontend/torchlette";

async function run(useCkpt: boolean, useAC: boolean, fusion: boolean): Promise<number[] | string> {
  try {
    const api = new Torchlette("webgpu", { enableFusion: fusion });
    const D = 8;
    const w = api.tensorFromArray(
      Array.from({ length: D * D }, (_, i) => Math.sin(i + 1) * 0.3),
      [D, D],
      { device: "webgpu", requiresGrad: true },
    );
    const x = api.tensorFromArray(
      Array.from({ length: 2 * 4 * D }, (_, i) => Math.cos(i) * 0.5),
      [2, 4, D],
      { device: "webgpu" },
    );
    const block = (inp: typeof x) => {
      // reshape -> matmul (f16 under autocast) -> gelu -> reshape back
      const flat = api.reshape(inp, [8, D]);
      const h = api.matmul(flat, w);
      const g = api.gelu(h);
      return api.reshape(g, [2, 4, D]);
    };
    const body = () => {
      const mid = useCkpt
        ? checkpoint(api, (inp: Tensor) => block(inp), [x])
        : block(x);
      const out = useCkpt
        ? checkpoint(api, (inp: Tensor) => block(inp), [mid])
        : block(mid);
      return api.sum(api.mul(out, out));
    };
    const loss = useAC ? api.autocast(body) : body();
    await loss.backward();
    const grad = (w as unknown as { grad: { cpu(): Promise<Float32Array> } | null }).grad;
    if (!grad) return "NULL GRAD";
    return Array.from(await grad.cpu());
  } catch (e) {
    return `THROW: ${String(e).slice(0, 140)}`;
  }
}

async function main() {
  if (!(await initWebGPU())) process.exit(1);
  const ref = await run(false, false, false);
  if (typeof ref === "string") { console.log("reference failed:", ref); process.exit(0); }
  const cases: Array<[string, boolean, boolean, boolean]> = [
    ["ckpt          ", true, false, false],
    ["ac            ", false, true, false],
    ["ckpt+ac       ", true, true, false],
    ["ckpt+fusion   ", true, false, true],
    ["ckpt+ac+fusion", true, true, true],
  ];
  for (const [name, c, a, f] of cases) {
    const got = await run(c, a, f);
    if (typeof got === "string") { console.log(`${name}: ${got}`); continue; }
    let worst = 0;
    for (let i = 0; i < ref.length; i++) worst = Math.max(worst, Math.abs(got[i] - ref[i]));
    // f16 matmul tolerance ~1e-2 relative on these magnitudes
    console.log(`${name}: max|grad diff| = ${worst.toExponential(3)} ${worst > 0.05 ? "<<< DIVERGES" : "ok"}`);
  }
  process.exit(0);
}
main().catch((e) => { console.error(e); process.exit(1); });
