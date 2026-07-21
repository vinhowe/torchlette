/**
 * Isolated correctness check for the lazy `deviceTopK` op: packed [1,2,k]
 * output (row 0 = values desc, row 1 = token ids as f32) must byte-match a CPU
 * top-k reference AND the existing readTopK host kernel (same tie-break).
 *
 * Run: eval "$(tools/pick-gpu.sh)"; LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH \
 *        npx tsx tools/t-uk-topk-op.ts
 */
import {
  getGpuUncapturedErrorCount,
  getWebGPUInitError,
  initWebGPU,
} from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

let FAIL = 0;
const ok = (c: boolean, m: string) => {
  console.log(`${c ? "PASS" : "FAIL"} — ${m}`);
  if (!c) FAIL++;
};

/** CPU top-k: values desc, ties by smaller index. */
function cpuTopK(logits: number[], k: number): { v: number[]; i: number[] } {
  const idx = logits.map((_, i) => i);
  idx.sort((a, b) => (logits[b] - logits[a]) || (a - b));
  const top = idx.slice(0, k);
  return { v: top.map((i) => logits[i]), i: top };
}

async function main() {
  if (!(await initWebGPU())) {
    console.error(getWebGPUInitError() || "WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(7);

  const cases: { V: number; k: number; name: string }[] = [
    { V: 256, k: 1, name: "k=1" },
    { V: 256, k: 8, name: "k=8" },
    { V: 256, k: 64, name: "k=64" },
    { V: 1000, k: 40, name: "V=1000 k=40" },
    { V: 50, k: 50, name: "k==V" },
  ];

  for (const c of cases) {
    // Distinct-ish logits + a couple of deliberate ties to exercise tie-break.
    const arr = Array.from({ length: c.V }, (_, i) =>
      Math.sin(i * 12.9898 + 4.1) * 43758.5453,
    ).map((x) => x - Math.floor(x)); // (0,1)
    arr[3] = 0.99999;
    arr[7] = 0.99999; // tie at the very top → smaller index (3) wins first
    arr[10] = 0.9;
    arr[20] = 0.9; // a mid tie

    const t = api.tensorFromArray(arr, [1, 1, c.V]);
    const packed = api.deviceTopK(t, c.k); // [1,2,k]
    const out = new Float32Array(await api.cpu(packed));
    await api.markStep();
    const devV = Array.from(out.slice(0, c.k));
    const devI = Array.from(out.slice(c.k, 2 * c.k)).map((x) => Math.round(x));

    const ref = cpuTopK(arr, c.k);
    const idxMatch = devI.every((x, j) => x === ref.i[j]);
    const valMatch = devV.every((x, j) => Math.abs(x - ref.v[j]) < 1e-5);
    ok(
      idxMatch,
      `[${c.name}] indices byte-match CPU top-k` +
        (idxMatch ? "" : `\n  dev=[${devI}]\n  ref=[${ref.i}]`),
    );
    ok(valMatch, `[${c.name}] values match CPU top-k (1e-5)`);

    // Cross-check against readTopK (the host reference's own source) when k<=V.
    const top = await api.readTopK(t, Math.min(c.k, c.V), { length: c.V });
    await api.markStep();
    const rtI = Array.from(top.indices).slice(0, c.k);
    ok(
      devI.every((x, j) => x === rtI[j]),
      `[${c.name}] indices byte-match readTopK` +
        (devI.every((x, j) => x === rtI[j])
          ? ""
          : `\n  dev=[${devI}]\n  rtk=[${rtI}]`),
    );
  }

  ok(
    getGpuUncapturedErrorCount() === 0,
    `zero uncaptured GPU errors — got ${getGpuUncapturedErrorCount()}`,
  );
  console.log(`\n=== ${FAIL === 0 ? "PASS" : `FAIL (${FAIL})`} ===`);
  process.exit(FAIL === 0 ? 0 : 1);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
