/**
 * SDPA backward micro-parity, torchlette side.
 *
 * Loads the q,k,v,dO written by sdpa_ref.py, runs the WebGPU fused
 * scaledDotProductAttention, and uses loss = sum(out * dO) so the backward
 * is seeded with exactly dO. Compares out (forward) and dq/dk/dv (backward)
 * against the PyTorch reference. Pinpoints the attention backward kernel.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";

const DIR =
  process.env.SDPA_DIR ?? "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/sdpa";
const B = 2,
  H = 4,
  S = 16,
  D = 32;
const scale = 1 / Math.sqrt(D);
const log = (m: string) => console.error(`[sdpa] ${m}`);

function rd(p: string): Float32Array {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
function norm(a: Float32Array): number {
  let s = 0;
  for (const x of a) s += x * x;
  return Math.sqrt(s);
}
function relErr(a: Float32Array, ref: Float32Array): [number, number] {
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
  return [Math.sqrt(d) / (Math.sqrt(nr) + 1e-12), dot / (Math.sqrt(nr) * Math.sqrt(na) + 1e-12)];
}

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });

  const qd = Array.from(rd(path.join(DIR, "q.f32")));
  const kd = Array.from(rd(path.join(DIR, "k.f32")));
  const vd = Array.from(rd(path.join(DIR, "v.f32")));
  const dOd = Array.from(rd(path.join(DIR, "dO.f32")));

  await api.beginStep();
  const q = api.tensorFromArray(qd, [B, H, S, D], { device: "webgpu", requiresGrad: true });
  const k = api.tensorFromArray(kd, [B, H, S, D], { device: "webgpu", requiresGrad: true });
  const v = api.tensorFromArray(vd, [B, H, S, D], { device: "webgpu", requiresGrad: true });
  const dO = api.tensorFromArray(dOd, [B, H, S, D], { device: "webgpu" });

  let outArr: Float32Array;
  const loss = api.tidy(() => {
    const out = api.scaledDotProductAttention(q, k, v, scale, true);
    api.keep(out);
    const l = api.sum(api.mul(out, dO));
    api.keep(l);
    return { out, l };
  });
  outArr = new Float32Array(await loss.out.cpu());
  await loss.l.backward();
  await api._runtime().forceAllPending();

  const dq = new Float32Array(await ((q as any).grad as Tensor).cpu());
  const dk = new Float32Array(await ((k as any).grad as Tensor).cpu());
  const dv = new Float32Array(await ((v as any).grad as Tensor).cpu());
  api.endStep();
  await api.markStep();

  const refOut = rd(path.join(DIR, "out.f32"));
  const refDq = rd(path.join(DIR, "dq.f32"));
  const refDk = rd(path.join(DIR, "dk.f32"));
  const refDv = rd(path.join(DIR, "dv.f32"));

  const [oE, oC] = relErr(outArr, refOut);
  const [qE, qC] = relErr(dq, refDq);
  const [kE, kC] = relErr(dk, refDk);
  const [vE, vC] = relErr(dv, refDv);
  log(`forward out : rel_err=${oE.toExponential(3)} cos=${oC.toFixed(5)}  (|tl|=${norm(outArr).toFixed(4)} |pt|=${norm(refOut).toFixed(4)})`);
  log(`dV          : rel_err=${vE.toExponential(3)} cos=${vC.toFixed(5)}  (|tl|=${norm(dv).toFixed(4)} |pt|=${norm(refDv).toFixed(4)})`);
  log(`dQ          : rel_err=${qE.toExponential(3)} cos=${qC.toFixed(5)}  (|tl|=${norm(dq).toFixed(4)} |pt|=${norm(refDq).toFixed(4)})`);
  log(`dK          : rel_err=${kE.toExponential(3)} cos=${kC.toFixed(5)}  (|tl|=${norm(dk).toFixed(4)} |pt|=${norm(refDk).toFixed(4)})`);

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
