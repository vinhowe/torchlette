/**
 * Show scatterAdd race: repeated indices lose updates on WebGPU.
 */
import { Torchlette, initWebGPU } from "../src/index";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  // 100 sources, all pointing to index 0 — scatter-add should accumulate to 100.
  const out = api.zeros([3]);
  const indices = api.tensorFromArray(new Array(100).fill(0), [100]);
  const src = api.tensorFromArray(new Array(100).fill(1), [100]);

  const result = api.scatterAdd(out, indices, src, { dim: 0 });
  const got = Array.from(await result.cpu());
  console.log("100 sources → index 0:");
  console.log("  got:     ", got);
  console.log("  expected: [100, 0, 0]");
  console.log(got[0] === 100 ? "CORRECT" : `RACE: got ${got[0]} instead of 100 (lost ${100 - got[0]} updates)`);
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
