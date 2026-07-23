/**
 * COMPOSITE-CLOSURE F1 / C1 — the cost probe (design §3.2, the T1 fork).
 *
 * Exactly one op could put the derived CompNode-adjoint graph on a training-hot
 * path: standalone `softmax` has NO fused backward kernel, so if C2 replaces its
 * CPU closure the derived graph IS the GPU backward. (layernorm/CE/rmsnorm keep
 * their fused kernels, so their hot path is untouched.) This probe MEASURES the
 * derived softmax backward's node count + a GPU timing vs the 37-line hand
 * closure, so the C2 softmax deletion is gated on measured parity, not assumed.
 *
 *   node-count parity  → the derived graph is not heavier → C2 deletes the closure.
 *   materially heavier → C2 keeps the hand closure, derived form is reference-only.
 *
 * Also reports the measured derived-vs-hand max-abs reassociation delta (L-COMP).
 *
 * Run:  eval "$(tools/pick-gpu.sh)"; npx tsx tools/comp-adjoint-cost.ts
 */

import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import type { LazyRef } from "../src/graph/types";
import { SOFTMAX_DEF, vjpComposition } from "../src/ops/semantic";
import type { Tensor as RuntimeTensor } from "../src/runtime/tensor";

/** Count DISTINCT pending LazyIRNodes reachable from a tensor's lazy graph. */
function countNodes(t: RuntimeTensor): number {
  const seen = new Set<number>();
  const walk = (ref: LazyRef): void => {
    if (ref.kind !== "pending") return;
    if (seen.has(ref.node.id)) return;
    seen.add(ref.node.id);
    for (const inp of ref.node.inputs) walk(inp);
  };
  walk((t as unknown as { lazyRef: LazyRef }).lazyRef);
  return seen.size;
}

function rand(n: number, seed: number): number[] {
  const out: number[] = [];
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out.push((s / 0xffffffff) * 4 - 2);
  }
  return out;
}

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");
  const rt = api.runtime;

  const R = 1024;
  const C = 1024;
  const dim = 1;
  const xV = rand(R * C, 7);
  const gV = rand(R * C, 9);

  const mkX = () => api.tensorFromArray(xV, [R, C])._unwrap();
  const mkG = () => api.tensorFromArray(gV, [R, C])._unwrap();

  // --- Derived reverse-mode graph (the C1 pass). ---
  const derived = vjpComposition(SOFTMAX_DEF, rt, dim, { x: mkX() }, mkG());
  const derivedNodes = countNodes(derived.x);

  // --- Hand closure (softmaxImpl backward), transcribed. ---
  const handGrad = (x: RuntimeTensor, g: RuntimeTensor): RuntimeTensor => {
    const mx = rt.max(x, { dim, keepdim: true }) as RuntimeTensor;
    const sm = rt.div(
      rt.exp(rt.sub(x, mx)),
      rt.sum(rt.exp(rt.sub(x, mx)), {
        dim,
        keepdim: true,
      }) as RuntimeTensor,
    );
    const smg = rt.mul(sm, g);
    const sumsg = rt.sum(smg, { dim, keepdim: true }) as RuntimeTensor;
    return rt.mul(sm, rt.sub(g, sumsg));
  };
  const hand = handGrad(mkX(), mkG());
  const handNodes = countNodes(hand);

  // --- Numerical delta (L-COMP). ---
  const dA = await rt.cpu(
    vjpComposition(SOFTMAX_DEF, rt, dim, { x: mkX() }, mkG()).x,
  );
  const hB = await rt.cpu(handGrad(mkX(), mkG()));
  let maxAbs = 0;
  for (let i = 0; i < dA.length; i++)
    maxAbs = Math.max(maxAbs, Math.abs(dA[i] - hB[i]));

  // --- Timing (build + force + readback), steady-state. ---
  const time = async (
    build: () => RuntimeTensor,
    iters: number,
  ): Promise<number> => {
    for (let i = 0; i < 3; i++) await rt.cpu(build()); // warmup
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) await rt.cpu(build());
    return (performance.now() - t0) / iters;
  };
  const ITERS = 30;
  const derivedMs = await time(
    () => vjpComposition(SOFTMAX_DEF, rt, dim, { x: mkX() }, mkG()).x,
    ITERS,
  );
  const handMs = await time(() => handGrad(mkX(), mkG()), ITERS);

  const parity = derivedNodes <= handNodes + 1; // ±1 tolerance on node count
  console.log("=== COMPOSITE-CLOSURE C1 cost probe (softmax backward) ===");
  console.log(`shape                  : [${R}, ${C}]`);
  console.log(`derived node count     : ${derivedNodes}`);
  console.log(`hand    node count     : ${handNodes}`);
  console.log(
    `node-count parity      : ${parity ? "YES" : "NO"} (derived ${derivedNodes} vs hand ${handNodes})`,
  );
  console.log(`derived time (ms/iter) : ${derivedMs.toFixed(3)}`);
  console.log(`hand    time (ms/iter) : ${handMs.toFixed(3)}`);
  console.log(`derived/hand ratio     : ${(derivedMs / handMs).toFixed(3)}`);
  console.log(`L-COMP max-abs delta   : ${maxAbs.toExponential(3)}`);
  console.log(
    `T1 verdict             : ${parity ? "DELETE the hand softmax closure (parity)" : "KEEP hand closure, derived reference-only (heavier)"}`,
  );

  destroyWebGPU();
  process.exit(0);
}

main();
