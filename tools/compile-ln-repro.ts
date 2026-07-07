import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";

const R = 8, C = 1024, EPS = 1e-5;
const log = (m: string) => console.error(`[ln-repro] ${m}`);

function cpuLN(x: Float32Array): Float32Array {
  const out = new Float32Array(R * C);
  for (let r = 0; r < R; r++) {
    let mu = 0; for (let c = 0; c < C; c++) mu += x[r*C+c]!; mu /= C;
    let v = 0; for (let c = 0; c < C; c++){ const d=x[r*C+c]!-mu; v+=d*d; } v/=C;
    const inv = 1/Math.sqrt(v+EPS);
    for (let c = 0; c < C; c++) out[r*C+c] = (x[r*C+c]!-mu)*inv;
  }
  return out;
}
function maxRelRow(a: Float32Array, ref: Float32Array): {maxRel:number, badRows:number, sample:number[]} {
  let maxRel=0, badRows=0; const sample:number[]=[];
  for (let r=0;r<R;r++){
    let dn=0, rn=0;
    for(let c=0;c<C;c++){ const d=a[r*C+c]!-ref[r*C+c]!; dn+=d*d; rn+=ref[r*C+c]!**2; }
    const rel=Math.sqrt(dn)/(Math.sqrt(rn)+1e-12);
    if(rel>1e-3) badRows++;
    if(rel>maxRel) maxRel=rel;
    if(r<3){ // ratio compiled/ref for first elem of row
      sample.push(+(a[r*C]!/ (ref[r*C]!||1e-9)).toFixed(4));
    }
  }
  return {maxRel, badRows, sample};
}

async function main(){
  await initWebGPU();
  const tl = new Torchlette("webgpu", { enableFusion: true });
  const xVals = Array.from({length:R*C},(_,i)=> Math.sin(i*0.3)*2 + Math.cos(i*0.07));
  const ref = cpuLN(Float32Array.from(xVals));

  const chain = (x: Tensor): Tensor => {
    const mu = tl.mean(x, {dim:1, keepdim:true}) as Tensor;
    const xc = tl.sub(x, mu);
    const varr = tl.mean(tl.mul(xc, xc), {dim:1, keepdim:true}) as Tensor;
    const inv = tl.rsqrt(tl.add(varr, EPS));
    return tl.mul(xc, inv);
  };

  // direct (non-compiled) fused path
  const tx1 = tl.tensorFromArray(xVals, [R,C], {device:"webgpu"});
  const direct = new Float32Array(await chain(tx1).cpu());
  const d = maxRelRow(direct, ref);
  log(`DIRECT (enableFusion):   maxRelRow=${d.maxRel.toExponential(3)} badRows=${d.badRows}/${R} row0-2 ratio=${JSON.stringify(d.sample)}`);

  // compiled path
  const tx2 = tl.tensorFromArray(xVals, [R,C], {device:"webgpu"});
  const compiled = tl.compile(chain);
  const out = new Float32Array(await compiled(tx2).cpu());
  const cstats = maxRelRow(out, ref);
  log(`COMPILED (tl.compile):   maxRelRow=${cstats.maxRel.toExponential(3)} badRows=${cstats.badRows}/${R} row0-2 ratio=${JSON.stringify(cstats.sample)}`);
  // run compiled a 2nd time (cache path)
  const out2 = new Float32Array(await compiled(tl.tensorFromArray(xVals,[R,C],{device:"webgpu"})).cpu());
  const c2 = maxRelRow(out2, ref);
  log(`COMPILED 2nd call:        maxRelRow=${c2.maxRel.toExponential(3)} badRows=${c2.badRows}/${R}`);

  // Per-row diagnosis: derive the compiled `inv` and test which variance it used.
  // out_compiled[r,c] = xc[r,c] * inv_c[r]. Recover inv_c[r] robustly (max |xc| col).
  const xv = Float32Array.from(xVals);
  log(`per-row: inv_compiled vs rsqrt(var+eps) [correct] vs rsqrt(mean(x^2)+eps) [uncentered bug]`);
  for (let r = 0; r < R; r++) {
    let mu = 0; for (let c=0;c<C;c++) mu += xv[r*C+c]!; mu/=C;
    let varr=0, meanX2=0, bestc=0, bestAbs=0;
    for (let c=0;c<C;c++){ const x=xv[r*C+c]!; const d=x-mu; varr+=d*d; meanX2+=x*x; if(Math.abs(d)>bestAbs){bestAbs=Math.abs(d);bestc=c;} }
    varr/=C; meanX2/=C;
    const xc = xv[r*C+bestc]!-mu;
    const invComp = out[r*C+bestc]! / xc;
    const invCorrect = 1/Math.sqrt(varr+EPS);
    const invUncentered = 1/Math.sqrt(meanX2+EPS);
    if (r<5) log(`  row${r}: mu=${mu.toFixed(4)} invComp=${invComp.toFixed(5)} correct=${invCorrect.toFixed(5)} uncentered=${invUncentered.toFixed(5)} -> matches ${Math.abs(invComp-invUncentered)<Math.abs(invComp-invCorrect)?"UNCENTERED(bug)":"correct"}`);
  }
  await destroyWebGPU(); process.exit(0);
}
main().catch(e=>{log(`FATAL ${e?.stack||e}`); process.exit(1);});
