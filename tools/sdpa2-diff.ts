/**
 * SDPA-with-head-plumbing micro-parity, torchlette side.
 * Mirrors the attention module's chunk/reshape/permute around SDPA and
 * compares d(qkv) against the PyTorch reference.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";

const DIR =
  process.env.SDPA2_DIR ?? "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/sdpa2";
const B = 2,
  S = 16,
  H = 4,
  hd = 32;
const E = H * hd;
const scale = 1 / Math.sqrt(hd);
const log = (m: string) => console.error(`[sdpa2] ${m}`);

function rd(p: string): Float32Array {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
function relCos(a: Float32Array, ref: Float32Array): [number, number, number, number] {
  let d = 0,
    nr = 0,
    dot = 0,
    na = 0;
  for (let i = 0; i < ref.length; i++) {
    d += (a[i]! - ref[i]!) ** 2;
    nr += ref[i]! ** 2;
    dot += a[i]! * ref[i]!;
    na += a[i]! ** 2;
  }
  return [Math.sqrt(d) / (Math.sqrt(nr) + 1e-12), dot / (Math.sqrt(nr) * Math.sqrt(na) + 1e-12), Math.sqrt(na), Math.sqrt(nr)];
}

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  const qkvd = Array.from(rd(path.join(DIR, "qkv.f32")));
  const doutd = Array.from(rd(path.join(DIR, "dout.f32")));

  await api.beginStep();
  const x = api.tensorFromArray(qkvd, [B, S, 3 * E], { device: "webgpu", requiresGrad: true });
  const dOut = api.tensorFromArray(doutd, [B, S, E], { device: "webgpu" });

  const r = api.tidy(() => {
    const [qF, kF, vF] = x.chunk(3, -1);
    const toHeads = (t: Tensor) =>
      t.reshape([B, S, H, hd]).permute([0, 2, 1, 3]).contiguous();
    const q = toHeads(qF);
    const k = toHeads(kF);
    const v = toHeads(vF);
    const out = api.scaledDotProductAttention(q, k, v, scale, true);
    const attnFlat = out.permute([0, 2, 1, 3]).reshape([B, S, E]);
    api.keep(attnFlat);
    const l = api.sum(api.mul(attnFlat, dOut));
    api.keep(l);
    return { attnFlat, l };
  });
  const attnFlat = new Float32Array(await r.attnFlat.cpu());
  await r.l.backward();
  await api._runtime().forceAllPending();
  const dqkv = new Float32Array(await ((x as any).grad as Tensor).cpu());
  api.endStep();
  await api.markStep();

  const refAttn = rd(path.join(DIR, "attnflat.f32"));
  const refDqkv = rd(path.join(DIR, "dqkv.f32"));
  const [aE, aC] = relCos(attnFlat, refAttn);
  const [gE, gC, gtl, gpt] = relCos(dqkv, refDqkv);
  log(`forward attnFlat: rel_err=${aE.toExponential(3)} cos=${aC.toFixed(5)}`);
  log(`d(qkv)          : rel_err=${gE.toExponential(3)} cos=${gC.toFixed(5)}  (|tl|=${gtl.toFixed(4)} |pt|=${gpt.toFixed(4)})`);

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
