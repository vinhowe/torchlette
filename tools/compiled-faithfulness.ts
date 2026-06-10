/**
 * Compiled-plan FAITHFULNESS probe.
 *
 * The compiled plan is an optimization: on the 2nd+ execution of a lowered-plan
 * template it records a flat command list and replays it. The naive "lowered"
 * plan is the reference. This probe answers the validation question raised by
 * the bounded-arena work: **does the compiled plan compute the SAME numbers as
 * the lowered plan?**
 *
 * Method: load fixed shared weights + a fixed batch, then run forward+backward
 * K times WITHOUT stepping the optimizer (weights & batch identical every
 * iteration). The math is therefore identical across iterations — but the
 * compiled plan only builds/activates on iteration >= ~2. So:
 *   - iteration 0  -> lowered plan (reference)
 *   - iteration K-1 -> compiled plan active
 * If the compiled plan is faithful, loss and grads are bit-identical across all
 * K iterations. Any drift run0 vs runK-1 is the compiled plan's infidelity.
 *
 * Env: NUM_LAYERS/NUM_HEADS/EMBED_DIM/SEQ_LEN/BATCH_SIZE (default 8M parity cfg),
 *      ITERS (default 5), TORCHLETTE_COMPILED_PLAN=0 to force lowered every run
 *      (control: should then be bit-identical trivially).
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";

const PARITY =
  process.env.PARITY_DIR ??
  "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/parity";
const L = parseInt(process.env.NUM_LAYERS ?? "8", 10);
const H = parseInt(process.env.NUM_HEADS ?? "4", 10);
const E = parseInt(process.env.EMBED_DIM ?? "128", 10);
const SEQ = parseInt(process.env.SEQ_LEN ?? "256", 10);
const BATCH = parseInt(process.env.BATCH_SIZE ?? "8", 10);
const ITERS = parseInt(process.env.ITERS ?? "5", 10);
const VOCAB = 50257;

const log = (m: string) => console.error(`[faith] ${m}`);

function readF32(p: string): Float32Array {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
function readI32(p: string): Int32Array {
  const b = fs.readFileSync(p);
  return new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
function tlName(key: string): string {
  if (key === "wte") return "wte.weight";
  if (key === "wpe") return "wpe.weight";
  if (key === "lnf.w") return "lnF.weight";
  if (key === "lnf.b") return "lnF.bias";
  const m = key.match(/^block\.(\d+)\.(.+)$/);
  if (!m) throw new Error(`bad key ${key}`);
  const [, i, rest] = m;
  const map: Record<string, string> = {
    "ln1.w": `h.${i}.ln1.weight`,
    "ln1.b": `h.${i}.ln1.bias`,
    "attn.qkv.w": `h.${i}.attn.cAttn.weight`,
    "attn.qkv.b": `h.${i}.attn.cAttn.bias`,
    "attn.proj.w": `h.${i}.attn.cProj.weight`,
    "attn.proj.b": `h.${i}.attn.cProj.bias`,
    "ln2.w": `h.${i}.ln2.weight`,
    "ln2.b": `h.${i}.ln2.bias`,
    "mlp.fc.w": `h.${i}.mlp.cFc.weight`,
    "mlp.fc.b": `h.${i}.mlp.cFc.bias`,
    "mlp.proj.w": `h.${i}.mlp.cProj.weight`,
    "mlp.proj.b": `h.${i}.mlp.cProj.bias`,
  };
  const out = map[rest!];
  if (!out) throw new Error(`bad key ${key}`);
  return out;
}

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "6000";

  const manifest: Record<string, number[]> = JSON.parse(
    fs.readFileSync(path.join(PARITY, "manifest.json"), "utf-8"),
  );

  const api = new Torchlette("webgpu", { enableFusion: true });
  const { GPT2 } = await import("../examples/gpt2/model.ts");
  const model = new GPT2(
    api,
    { vocabSize: VOCAB, blockSize: 1024, numLayers: L, numHeads: H, embedDim: E, dropoutRate: 0 },
    { device: "webgpu" },
  );
  const named = new Map<string, Tensor>(model.namedParameters());
  const paddedVocab = (model as { paddedVocabSize: number }).paddedVocabSize;

  await api.beginStep();
  for (const key of Object.keys(manifest)) {
    const data = readF32(path.join(PARITY, `w.${key}.f32`));
    const tlParam = named.get(tlName(key));
    if (!tlParam) throw new Error(`missing param for ${key}`);
    let loadData: Float32Array = data;
    let loadShape = manifest[key]!;
    if (key === "wte" && paddedVocab > VOCAB) {
      const padded = new Float32Array(paddedVocab * E);
      padded.set(data);
      loadData = padded;
      loadShape = [paddedVocab, E];
    }
    api.copy_(
      tlParam,
      api.tensorFromArray(Array.from(loadData), loadShape, { device: "webgpu" }),
    );
  }
  api.endStep();
  await api.markStep();
  model.train(true);

  const inp = Array.from(readI32(path.join(PARITY, "input.i32")));
  const tgt = Array.from(readI32(path.join(PARITY, "target.i32")));

  // CHANGING_INPUT: use a different batch each iter (from the TinyStories blob)
  // so the compiled plan, built on one batch, is REPLAYED on different inputs —
  // tests whether input tokens are correctly re-bound per step (vs baked in).
  let tokenBlob: Uint16Array | null = null;
  if (process.env.CHANGING_INPUT === "1") {
    const tb = fs.readFileSync(
      process.env.LOCAL_TOKENS ??
        "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin",
    );
    tokenBlob = new Uint16Array(tb.buffer, tb.byteOffset, tb.byteLength / 2);
  }

  const keys = Object.keys(manifest);
  const byKey = new Map<string, { first: Float32Array; last: Float32Array }>();
  const losses: number[] = [];
  // Flattened grads for iter 0 (lowered) and the final iter (compiled).
  let gradFirst: Float32Array | null = null;
  let gradLast: Float32Array | null = null;
  const compiledHits: number[] = [];

  for (let it = 0; it < ITERS; it++) {
    // Reset grads so each iteration computes the SAME fresh gradient (no accum).
    for (const key of keys) {
      named.get(tlName(key))!.zeroGrad();
    }

    await api.beginStep();
    let inpArr = inp;
    let tgtArr = tgt;
    if (tokenBlob) {
      const off = (it * BATCH * SEQ) % (tokenBlob.length - BATCH * SEQ - 1);
      inpArr = Array.from(tokenBlob.subarray(off, off + BATCH * SEQ));
      tgtArr = Array.from(tokenBlob.subarray(off + 1, off + 1 + BATCH * SEQ));
    }
    const input = api.tensorFromArray(inpArr, [BATCH, SEQ], { device: "webgpu" });
    const target = api.tensorFromArray(tgtArr, [BATCH, SEQ], { device: "webgpu" });
    const useCkpt = process.env.CHECKPOINT === "1";
    const useAutocast = process.env.AUTOCAST === "1";
    const loss = api.tidy(() => {
      const fwd = () =>
        model.forwardWithLoss(input, target, { useCheckpoint: useCkpt }).loss;
      const l = useAutocast ? api.autocast(fwd) : fwd();
      api.keep(l);
      return l;
    });
    const lossVal = await loss.item();
    await loss.backward();
    await api._runtime().forceAllPending();

    // CLIP: apply clipGradNorm_ after backward (small maxNorm forces clipping)
    // so we can test whether grad clipping replays faithfully under the compiled
    // plan. Compares clipped grads (iter0 lowered vs iterK compiled).
    if (process.env.CLIP === "1") {
      const { clipGradNorm_ } = await import("../src/nn/clip-grad.ts");
      const clipParams = keys.map((k) => named.get(tlName(k))!);
      clipGradNorm_(api, clipParams, 0.01);
      await api._runtime().forceAllPending();
    }

    // Snapshot grads into one flat buffer (only first & last iters to save RAM).
    if (it === 0 || it === ITERS - 1) {
      const parts: Float32Array[] = [];
      for (const key of keys) {
        const p = named.get(tlName(key))!;
        const g = (p as unknown as { grad: Tensor | null }).grad;
        if (!g) throw new Error(`no grad for ${key}`);
        let arr = new Float32Array(await g.cpu());
        if (key === "wte" && paddedVocab > VOCAB) arr = arr.subarray(0, VOCAB * E);
        parts.push(arr);
        if (it === 0) byKey.set(key, { first: arr, last: arr });
        else byKey.get(key)!.last = arr;
      }
      let total = 0;
      for (const a of parts) total += a.length;
      const flat = new Float32Array(total);
      let off = 0;
      for (const a of parts) {
        flat.set(a, off);
        off += a.length;
      }
      if (it === 0) gradFirst = flat;
      else gradLast = flat;
    }

    losses.push(lossVal);
    api.endStep();
    await api.markStep();
    log(`iter ${it}: loss = ${lossVal.toFixed(8)}`);
  }

  // Compare loss bit-exactness and grad rel_err between lowered (iter0) and
  // compiled-active (last iter).
  log("");
  log(`loss[0]      = ${losses[0]!.toFixed(8)}  (lowered)`);
  log(`loss[${ITERS - 1}]      = ${losses[ITERS - 1]!.toFixed(8)}  (compiled active)`);
  log(`loss abs diff = ${Math.abs(losses[0]! - losses[ITERS - 1]!).toExponential(3)}`);

  // CHANGING_INPUT mode: save the last iter's grads so an external diff can
  // compare a lowered run vs a compiled run on the SAME input sequence.
  if (process.env.SAVE_GRADS && gradLast) {
    fs.writeFileSync(
      process.env.SAVE_GRADS,
      Buffer.from(gradLast.buffer, gradLast.byteOffset, gradLast.byteLength),
    );
    log(`saved last-iter grads to ${process.env.SAVE_GRADS}`);
  }

  const relErr = (a: Float32Array, b: Float32Array): number => {
    let num = 0;
    let den = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i]! - b[i]!;
      num += d * d;
      den += a[i]! * a[i]!;
    }
    return Math.sqrt(num) / (Math.sqrt(den) + 1e-30);
  };

  if (gradFirst && gradLast) {
    log("");
    log(`GRAD lowered(iter0) vs compiled(iter${ITERS - 1}): rel_err = ${relErr(gradFirst, gradLast).toExponential(3)}`);

    // Per-param breakdown: which gradients diverge? Sort by rel_err desc.
    const perKey = keys
      .map((k) => {
        const { first, last } = byKey.get(k)!;
        return { k, rel: relErr(first, last) };
      })
      .sort((a, b) => b.rel - a.rel);
    log("  top divergent params (compiled vs lowered):");
    for (const { k, rel } of perKey.slice(0, 12)) {
      log(`    ${k.padEnd(22)} rel_err=${rel.toExponential(2)}`);
    }
    const clean = perKey.filter((p) => p.rel < 1e-4).length;
    log(`  ${clean}/${perKey.length} params match (<1e-4); ${perKey.length - clean} diverge`);

    // Ground-truth comparison: load the PyTorch reference grads (same key order)
    // and check which path matches. Lowered should be ~1e-6 vs PyTorch; if the
    // compiled path is far off, the compiled BACKWARD is the wrong one.
    const ptDir = path.join(PARITY, "pt");
    if (fs.existsSync(ptDir)) {
      const ptParts: Float32Array[] = [];
      for (const key of keys) {
        let arr = readF32(path.join(ptDir, `g.${key}.f32`));
        if (key === "wte" && paddedVocab > VOCAB && arr.length > VOCAB * E) {
          arr = arr.subarray(0, VOCAB * E);
        }
        ptParts.push(arr);
      }
      let tot = 0;
      for (const a of ptParts) tot += a.length;
      const pt = new Float32Array(tot);
      let o = 0;
      for (const a of ptParts) {
        pt.set(a, o);
        o += a.length;
      }
      if (pt.length === gradFirst.length) {
        log("");
        log(`GROUND TRUTH (PyTorch reference):`);
        log(`  lowered (iter0)  vs PyTorch: rel_err = ${relErr(pt, gradFirst).toExponential(3)}`);
        log(`  compiled(iter${ITERS - 1}) vs PyTorch: rel_err = ${relErr(pt, gradLast).toExponential(3)}`);
        log(`  => the path closer to PyTorch is the correct backward.`);
      } else {
        log(`(pt length ${pt.length} != tl length ${gradFirst.length}; skip ground-truth)`);
      }
    }
  }

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
