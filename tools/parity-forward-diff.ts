/**
 * Weight-transfer forward-loss diff — step 2 (torchlette side).
 *
 * Loads the shared weights + batch written by gen_parity_weights.py into a
 * plain torchlette GPT2, runs ONE fp32 forward+backward (no autocast, no
 * scaler, no checkpointing — isolate kernel numerics), and writes:
 *   parity/tl.loss.json     : forward loss
 *   parity/tl/g.<key>.f32   : grads per canonical key
 *
 * Then compare_parity.py diffs against the PyTorch reference.
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
const VOCAB = 50257;

const log = (m: string) => console.error(`[fwd-diff] ${m}`);

function readF32(p: string): Float32Array {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
function readI32(p: string): Int32Array {
  const b = fs.readFileSync(p);
  return new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}

// canonical key -> torchlette named-parameter path
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
    {
      vocabSize: VOCAB,
      blockSize: 1024,
      numLayers: L,
      numHeads: H,
      embedDim: E,
      dropoutRate: 0,
    },
    { device: "webgpu" },
  );
  // biome-ignore lint/suspicious/noExplicitAny: namedParameters typing
  const named = new Map<string, Tensor>(model.namedParameters());
  const paddedVocab = (model as { paddedVocabSize: number }).paddedVocabSize;

  // Load the shared weights. wte is generated at [VOCAB,E]; the torchlette
  // wte is [paddedVocab,E] — zero-pad the extra rows (they're narrowed out
  // of cross-entropy, so they don't affect loss or receive gradient).
  await api.beginStep();
  for (const key of Object.keys(manifest)) {
    const data = readF32(path.join(PARITY, `w.${key}.f32`));
    const tlParam = named.get(tlName(key));
    if (!tlParam) throw new Error(`torchlette missing param for ${key}`);
    let loadData: Float32Array = data;
    let loadShape = manifest[key]!;
    if (key === "wte" && paddedVocab > VOCAB) {
      const padded = new Float32Array(paddedVocab * E);
      padded.set(data); // first VOCAB*E filled, rest zero
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
  log("weights loaded");

  model.train(true);

  // Shared batch.
  const inp = Array.from(readI32(path.join(PARITY, "input.i32")));
  const tgt = Array.from(readI32(path.join(PARITY, "target.i32")));

  await api.beginStep();
  const input = api.tensorFromArray(inp, [BATCH, SEQ], { device: "webgpu" });
  const target = api.tensorFromArray(tgt, [BATCH, SEQ], { device: "webgpu" });
  // fp32, no autocast, no checkpointing — isolate kernel numerics.
  const loss = api.tidy(() => {
    const l = model.forwardWithLoss(input, target, {
      useCheckpoint: false,
    }).loss;
    api.keep(l);
    return l;
  });
  const lossVal = await loss.item();
  log(`torchlette fp32 forward loss = ${lossVal.toFixed(6)}`);
  await loss.backward();
  await api._runtime().forceAllPending();

  fs.mkdirSync(path.join(PARITY, "tl"), { recursive: true });
  fs.writeFileSync(
    path.join(PARITY, "tl.loss.json"),
    JSON.stringify({ loss: lossVal }),
  );
  let saved = 0;
  for (const key of Object.keys(manifest)) {
    const tlParam = named.get(tlName(key))!;
    // biome-ignore lint/suspicious/noExplicitAny: grad isn't typed
    const g = (tlParam as any).grad as Tensor | null;
    if (!g) throw new Error(`no grad for ${key} (${tlName(key)})`);
    let arr = new Float32Array(await g.cpu());
    if (key === "wte" && paddedVocab > VOCAB) {
      arr = arr.subarray(0, VOCAB * E); // drop padding rows for comparison
    }
    fs.writeFileSync(
      path.join(PARITY, "tl", `g.${key}.f32`),
      Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength),
    );
    saved++;
  }
  api.endStep();
  await api.markStep();
  log(`saved ${saved} torchlette grads to ${path.join(PARITY, "tl")}`);

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
