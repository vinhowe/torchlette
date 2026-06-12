import { encodeTensors, decodeTensors } from "../src/distributed/wire-codec";
const a = Float32Array.from({ length: 4099 }, (_, i) => Math.sin(i) * 10 ** ((i % 9) - 4));
const b = Float32Array.from([0, -0, 1, -1, 65504, -65504, 1e-7, Infinity, -Infinity, NaN, 3.14159, -2.71828]);
for (const dtype of ["f32", "f16"] as const) {
  const enc = encodeTensors([a, b], dtype);
  const [da, db] = decodeTensors(enc, [[a.length], [b.length]], dtype);
  let worstRel = 0; // only over f16's NORMAL range (tiny values subnormal-flush by design)
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i]) < 1e-4) continue;
    const r = Math.abs((da[i] - a[i]) / a[i]);
    if (Number.isFinite(r)) worstRel = Math.max(worstRel, r);
  }
  const inf = db[7] === Infinity && db[8] === -Infinity && Number.isNaN(db[9]);
  console.log(`${dtype}: bytes=${enc.length} worstRel=${worstRel.toExponential(2)} specials=${inf ? "ok" : "BAD"} max=${db[4]}`);
}
process.exit(0);
